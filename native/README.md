# WeMM-Embedding Native (LibTorch / CUDA 13)

WeMM-Embedding の推論を **Python 不要** の C++ ネイティブ実装に移植したものです。
モデル本体は LibTorch と自前の CUDA カーネルで実装し、周辺処理はデファクトの
ネイティブライブラリで行います。

| 処理 | 実装 |
| --- | --- |
| モデル (Qwen3.5: Gated DeltaNet + Gated Attention, ViT) | LibTorch (ATen) + 自前 CUDA カーネル + cuBLASLt + FlashAttention / memory-efficient attention (LibTorch 同梱) |
| 重み | Hugging Face の `model.safetensors` を mmap で直接読み込み（変換不要） |
| トークナイザ | Hugging Face `tokenizers` (Rust) を C ABI で静的リンク |
| 画像デコード | OpenCV (`imdecode`) |
| 画像リサイズ | Pillow の bicubic 縮小をビット単位で移植 + LibTorch の uint8 AA カーネル |
| 動画デコード | FFmpeg (libavformat / libavcodec / libswscale) |
| チャットテンプレート / プレースホルダ展開 / M-RoPE 位置 | C++ で再実装 |

目的は「ネイティブ化による高速化」と「精度の維持」の両立です。前処理は
Transformers 5.2 + qwen-vl-utils 0.0.14（リポジトリ推奨構成）と**ビット単位で一致**し、
埋め込みはリファレンス (Transformers bf16) よりも**約 30 倍 fp32 モデルに近い**値を返します。
対応 GPU は Turing (sm_75) 〜 Blackwell (sm_120) です。

## 結果の概要 (RTX 3090, CUDA 13.0)

結果はすべて `tests/test_reference.cpp` / `apps/wemm_bench.cpp` / `tools/*.py` で再現できます
（下の「検証」節を参照）。

### 精度

`1 - cos(埋め込み, fp32 モデルの埋め込み)`（小さいほど良い、平均 / 最大）。fp32 参照は
Transformers の純 PyTorch 実装を TF32 無効の真の fp32 で実行したもの。

| モデル | fp32 参照のあるケース | Transformers bf16 | ネイティブ bf16 | **ネイティブ fp16（既定）** |
| --- | ---: | ---: | ---: | ---: |
| 2B | 22 | 1.23e-3 / 7.04e-3 | 3.29e-4 / 1.41e-3 | **3.66e-5 / 1.72e-4** |
| 4B | 19 | 1.31e-3 / 5.67e-3 | 3.32e-4 / 2.56e-3 | **4.35e-5 / 3.29e-4** |
| 9B | 6（fp32 は CPU で計算） | 8.24e-4 / 2.38e-3 | 2.88e-4 / 8.71e-4 | **2.90e-5 / 1.21e-4** |

ネイティブ fp16 は全ケースで Transformers bf16 より fp32 モデルに近く、平均誤差は約 1/30 です。
理由は 2 つあります。

- **fp32 残差ストリーム**: ViT と LM の残差を fp32 で保持し、GEMM のエピローグで直接 fp32 に加算する。
- **fp16 演算**: bf16 のチェックポイントは fp16 に誤差なく変換でき、fp16 は仮数部が 10bit（bf16 は 7bit）。
  RTX 3090 ではテンソルコア速度は bf16 と同じです。16bit で保持する中間値の最大は 1275 で、
  fp16 の上限 65504 に対して十分な余裕があります（全モデル・全ケースで計測）。
  オーバーフローが起きた場合は非有限値を検知してエラーにします（`compute_dtype=bf16` に切り替えられます）。

<details>
<summary>2B のケース別（1 − cos、fp32 モデル基準）</summary>

| ケース | 内容 | Transformers bf16 | ネイティブ bf16 | ネイティブ fp16 |
| --- | --- | ---: | ---: | ---: |
| text | 短文 (13 tok) | 5.2e-5 | 9.0e-6 | 0 |
| text_long | 長文・多言語・絵文字 (962 tok) | 6.3e-5 | 1.8e-5 | 0 |
| system | system + user | 5.8e-5 | 1.0e-5 | 0 |
| image | 298×224 JPEG | 7.6e-4 | 2.6e-4 | 1.1e-5 |
| image_small | 200×150 PNG（2 段リサイズ） | 5.1e-4 | 5.9e-5 | 2.3e-5 |
| image_chart | 850×600 WebP | 1.4e-3 | 5.8e-4 | 4.7e-5 |
| image_large | 1700×1200 JPEG (2020 tok) | 8.9e-4 | 3.3e-4 | 1.6e-5 |
| image_pixels | `min/max_pixels` 指定 | 7.0e-3 | 5.8e-4 | 2.9e-5 |
| image_text | 画像 + テキスト | 4.1e-4 | 1.3e-4 | 6.0e-6 |
| multi_image | テキスト・画像 2 枚の交互配置 | 1.3e-3 | 6.6e-4 | 1.0e-4 |
| rgba / palette / gray | RGBA PNG・パレット PNG・グレー JPEG | 8.3e-4 / 6.4e-4 / 7.2e-4 | 1.7e-4 / 1.1e-4 / 5.9e-5 | 1.2e-5 / 1.5e-5 / 1.7e-5 |
| exif / rgb16 / gray16 | EXIF 回転 JPEG・16bit RGB/グレー PNG | 2.3e-4 / 7.3e-4 / 3.9e-3 | 6.8e-5 / 2.2e-4 / 1.4e-3 | 3.5e-5 / 3.2e-5 / 1.7e-4 |
| video | 298×224, 41 frames (1485 tok) | 1.9e-3 | 4.5e-4 | 7.4e-5 |
| video_long | 640×360, 360 frames (2743 tok) | 9.6e-4 | 5.5e-4 | 2.3e-5 |
| video_nframes | `nframes=64` (7307 tok) | 7.9e-4 | 2.3e-4 | 2.0e-5 |
| video_segment | `video_start/end` + `fps=1` | 8.8e-4 | 4.0e-4 | 5.5e-5 |
| text_video / image_video | テキスト + 動画 / 画像 + 動画 | 1.4e-3 / 1.6e-3 | 4.0e-4 / 5.3e-4 | 5.7e-5 / 6.1e-5 |

</details>

前処理（チャットテンプレート、トークン列、`pixel_values` / `pixel_values_videos`、
グリッド、動画タイムスタンプ）は 2B/4B/9B の全ケースで参照と**完全一致**（差 0）しています。
対象は上表のすべてのケースで、RGBA 合成・パレット PNG・グレースケール・EXIF 付き JPEG・16bit PNG・
固定フレーム数/区間指定の動画・複数画像・system メッセージを含みます。

### 速度

環境: RTX 3090 (24 GB), Driver 580 / CUDA 13.0, 公式 LibTorch 2.10.0+cu130。
ベースラインは Transformers 5.2.0（推奨構成）+ flash-linear-attention 0.5.2（FLA 有りの方が速いので比較対象にしています）。

**forward レイテンシ**（バッチ 1、前処理除く、ms）

| 入力 | トークン | Transformers（PyTorch 実装） | Transformers + FLA | **ネイティブ** | FLA 比 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 2B text | 13 | 651 | 42.3 | **6.5** | 6.5× |
| 2B text_long | 962 | 637 | 69.6 | **55.2** | 1.26× |
| 2B image | 86 | 655 | 63.9 | **15.2** | 4.2× |
| 2B image_large | 2020 | 839 | 351.3 | **287.1** | 1.22× |
| 2B video | 1485 | 695 | 199.1 | **150.4** | 1.32× |
| 2B video_long | 2743 | 808 | 359.6 | **272.3** | 1.32× |
| 4B text | 13 | – | 63.3 | **13.0** | 4.9× |
| 4B text_long | 962 | – | 150.4 | **134.8** | 1.12× |
| 4B image | 86 | – | 76.7 | **25.4** | 3.0× |
| 4B image_large | 2020 | – | 517.3 | **449.9** | 1.15× |
| 4B video | 1485 | – | 332.5 | **272.4** | 1.22× |
| 4B video_long | 2743 | – | 578.3 | **498.3** | 1.16× |
| 9B text | 13 | – | 54.7 | **21.1** | 2.6× |
| 9B text_long | 962 | – | 262.8 | **234.4** | 1.12× |
| 9B image | 86 | – | 83.3 | **41.1** | 2.0× |
| 9B image_large | 2020 | – | 823.6 | **739.2** | 1.11× |
| 9B video | 1485 | – | 515.4 | **442.6** | 1.16× |
| 9B video_long | 2743 | – | 908.4 | **806.7** | 1.13× |

短い入力（検索クエリなど）は、起動オーバーヘッドの削減とカーネル融合の効果で数倍速くなります。
長い入力は GEMM（約 65 TFLOPS、16bit/fp32 累積のピーク付近）と ViT の注意が支配的です。
ネイティブ版は fp32 残差のぶん余計な計算をしたうえで 1.1〜1.3 倍です。

**前処理**（2B、1 入力、ms）

| 入力 | Transformers + qwen-vl-utils (decord) | **ネイティブ** |
| --- | ---: | ---: |
| image (298×224) | 163 | **6** |
| image_large (1700×1200) | 127 | **62** |
| video (41 frames) | 265 | **51** |
| video_long (360 frames, 640×360) | 629 | **132** |

**エンドツーエンドのスループット**（2B、前処理込み）

| ワークロード | Transformers + FLA | **ネイティブ** | 比 |
| --- | ---: | ---: | ---: |
| テキスト 512 件（8〜256 語、バッチ 32） | 64.2 件/s | **200.7 件/s** | 3.1× |
| 画像 64 件（4 種のサイズ、バッチ 8） | 3.7 件/s | **11.4 件/s** | 3.1× |
| 動画 16 件（バッチ 4） | 1.4 件/s | **4.1 件/s** | 2.9× |

ネイティブ版はパディング無しの可変長パッキングを使い、前処理を GPU 計算と重ねて実行します。
（Transformers の「PyTorch 実装」列は FLA 未導入時の既定経路で、NGC コンテナ既定の TF32 有効状態で計測しています。）

### GPU メモリ

**常駐量とピーク**（GiB、`torch.cuda.max_memory_allocated` 相当）

| モデル | Transformers bf16: ロード後 / video_long ピーク | **ネイティブ: ロード後 / video_long ピーク** | ピーク削減 |
| --- | ---: | ---: | ---: |
| 2B | 5.07 / 5.55 | **3.18 / 3.53** | −36% |
| 4B | 9.80 / 10.32 | **7.28 / 7.84** | −24% |
| 9B | 17.60 / 18.16 | **13.75 / 14.33** | −21% |

**大きなバッチ**（2B、1005 トークン × 32 本）: 改善前はピーク 7.67 GiB・予約 9.24 GiB でしたが、
**ピーク 5.12 GiB・予約 5.59 GiB** になりました（速度は ±1% 以内）。
約 2 万トークンの単一入力でも、ピークは 4.57 GiB です。

効いている施策は次のとおりです（すべて計算結果は不変です）。

- **埋め込み表を GPU に置かない**: 1〜2 GB の表は mmap したホストメモリに残し、参照する行だけをピン留めバッファ経由で転送します
  （`embeddings_on_gpu=true` で従来どおり VRAM に置けます。2B / 4B / 9B でそれぞれ 0.95 / 1.18 / 1.89 GiB 削減）。
- **`lm_head` と MTP の重みを読まない**: Transformers との差のうち約 1〜2 GB はこれによるものです。
- **GDN をチャンクの時間窓で処理**: 窓をまたぐ系列は状態を持ち越します。fp32 の作業領域は
  `gdn_window_bytes`（既定 512 MiB）で上限化されます。窓なしの場合と結果はビット単位で同一です。
- **MLP を行ブロック単位で処理**（`mlp_chunk_tokens`、既定 8192）。
- **残差の in-place 加算**: GEMM のエピローグで残差に直接加算するので、層ごとに残差を複製しません。
- **ロード時の一時領域を解放**: 融合する重みはホスト側で結合してから 1 回だけ転送し、ロード後にキャッシュを空にします。
- **expandable segments**: caching allocator の断片化を抑えます（`PYTORCH_CUDA_ALLOC_CONF` が未設定の場合のみ有効）。

## 構成

```
native/
├── CMakeLists.txt
├── include/wemm/           公開ヘッダ
│   ├── embedder.h          高レベル API（前処理 + バッチ化 + モデル）
│   ├── processor.h         チャットテンプレート / 画像・動画前処理 / トークナイズ / M-RoPE
│   ├── model.h             LibTorch モデル（ViT + Qwen3.5 LM）と ModelOptions
│   ├── c_api.h             C ABI（他言語バインディング用）
│   ├── media.h             OpenCV / FFmpeg / Pillow 互換リサイズ
│   ├── gemm.h              cuBLASLt ラッパ（エピローグ融合・オートチューニング）
│   └── kernels.h           自前 CUDA カーネル
├── src/
│   ├── model.cpp           重みロード、バッチ毎の前計算（RoPE 表・パッキング）、ディスパッチ
│   ├── model_fused.cpp     既定の実装（融合カーネル・cuBLASLt・fp32 残差・作業領域の上限化）
│   ├── model_reference.cpp 検証用の素の ATen 実装（Transformers の計算をそのまま追う）
│   ├── kernels/            CUDA カーネル（fused.cu: norm/RoPE/ゲート等、gdn.cu: Gated DeltaNet）
│   └── ...                 前処理・メディア・GEMM・C ABI など
├── tokenizers-ffi/         HF tokenizers (Rust) の C ABI ラッパ（ビルド時のみ cargo が必要）
├── apps/wemm_embed.cpp     CLI
├── apps/wemm_bench.cpp     レイテンシ計測
├── tests/                  参照比較テスト・C ABI テスト・カーネル単体ベンチ
└── tools/                  参照データ生成・ベースライン計測（開発用 Python。推論には不要）
```

## ビルド

### 必要なもの

- NVIDIA GPU: Turing (sm_75) 以降。sm_75 では fp16 + memory-efficient attention、
  単一バッファの GDN 状態カーネルを使います（いずれも 3090 上で同じ経路を強制して検証済み）。
- CUDA Toolkit 13.0（`nvcc`, cuBLASLt）
- LibTorch 2.10.0 + cu130（公式配布の `libtorch-shared-with-deps-2.10.0+cu130.zip`）
- OpenCV 4（core / imgproc / imgcodecs）
- FFmpeg 開発ライブラリ（libavformat / libavcodec / libavutil / libswscale）
- nlohmann/json 3.10+
- Rust ツールチェーン（`cargo`、**ビルド時のみ**。HF tokenizers を静的ライブラリにします）
- CMake 3.26+, Ninja 推奨

Ubuntu 24.04 の例:

```bash
sudo apt install cmake ninja-build libopencv-dev nlohmann-json3-dev \
  libavformat-dev libavcodec-dev libavutil-dev libswscale-dev pkg-config
curl -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal

wget https://download.pytorch.org/libtorch/cu130/libtorch-shared-with-deps-2.10.0%2Bcu130.zip
unzip libtorch-shared-with-deps-2.10.0+cu130.zip
```

### CMake

```bash
cmake -S native -B build -G Ninja \
  -DCMAKE_PREFIX_PATH=$PWD/libtorch \
  -DTORCH_CUDA_ARCH_LIST="8.6"          # 省略時は 7.5;8.0;8.6;8.9;9.0;10.0;12.0+PTX
cmake --build build
```

生成物:

- `libwemm.so` — ライブラリ本体
- `wemm_embed` — CLI
- `wemm_bench` — レイテンシ計測
- `wemm_example` — C++ API の最小サンプル
- `wemm_test_reference`, `wemm_test_c_api`, `wemm_bench_gemm`, `wemm_bench_gdn` — テスト/ベンチ

実行時に必要なのは LibTorch の共有ライブラリ、CUDA ドライバ、OpenCV / FFmpeg の共有ライブラリのみです。
Python インタプリタや Python パッケージは一切使いません（公式 LibTorch に対してビルドした
バイナリで `ldd` に Python 関連が現れないこと、およびテストが通ることを確認済み）。

> NGC コンテナ等で `LD_LIBRARY_PATH` が Python 版 torch を指している場合は、
> 意図しない libtorch が読まれないよう `LD_LIBRARY_PATH` を調整してください。

## 使い方

モデルは Hugging Face のディレクトリをそのまま使います（`config.json`,
`processor_config.json`, `tokenizer.json`, `model.safetensors`）。

### CLI

```bash
./build/wemm_embed --model /path/to/WeMM-Embedding-2B \
  --text "A dog is running on a beach." \
  --image /path/to/image.jpg \
  --video /path/to/video.mp4 \
  --dimension 256 --output emb.npy
```

大量の入力は JSONL で渡します（1 行 1 入力）。Hugging Face 形式のメッセージ列か省略形が使えます。

```jsonl
{"text": "A dog is running on a beach."}
{"image": "/data/cat.jpg"}
{"video": "/data/clip.mp4"}
[{"role": "user", "content": [{"type": "image", "image": "/data/a.jpg"}, {"type": "text", "text": "Find similar"}]}]
[{"role": "user", "content": [{"type": "video", "video": "/data/v.mp4", "nframes": 64}]}]
```

```bash
./build/wemm_embed --model /path/to/WeMM-Embedding-2B --input inputs.jsonl --output emb.npy
```

画像・動画要素は qwen-vl-utils と同じオプションを受け付けます:
`min_pixels`, `max_pixels`, `resized_height`, `resized_width`（画像/動画）、
`fps`, `nframes`, `min_frames`, `max_frames`, `video_start`, `video_end`, `total_pixels`（動画）。

### C++ API

```cpp
#include "wemm/embedder.h"

wemm::Embedder emb("/path/to/WeMM-Embedding-2B");
at::Tensor t = emb.embed_texts({"A dog is running on a beach."});        // [1, 2048], L2 正規化済み
at::Tensor i = emb.embed_images({"cat.jpg", "dog.jpg"}, /*dimension=*/256); // Matryoshka

wemm::VideoOptions vo;
vo.nframes = 64;
std::vector<wemm::Conversation> convs = {
    {{"user", {wemm::ContentItem::Image("a.jpg"), wemm::ContentItem::Text("Find similar")}}},
    {{"user", {wemm::ContentItem::Video("v.mp4", vo)}}},
};
at::Tensor e = emb.embed(convs);
```

（`apps/example.cpp` が同じ内容のビルド可能なサンプルです。）

`Embedder::embed` は入力を CPU スレッドで前処理しつつ（次グループの前処理を GPU 計算と並行）、
トークン長でソートしてパディング無しの可変長バッチに詰めて GPU に流します。

### 主なオプション（`wemm::EmbedderOptions` / `wemm::ModelOptions`）

| オプション | 既定 | 内容 |
| --- | --- | --- |
| `compute_dtype` | fp16 | 重み・活性の 16bit 型。bf16 は sm_80+ のみ |
| `attention` | Auto | FlashAttention (sm_80+) / memory-efficient attention (sm_75) |
| `fp32_residual` | true | 残差ストリームを fp32 で保持 |
| `embeddings_on_gpu` | false | 埋め込み表を VRAM に置く（既定は mmap したホストメモリ） |
| `gdn_window_bytes` | 512 MiB | GDN 1 層あたりの fp32 作業領域の上限 |
| `mlp_chunk_tokens` | 8192 | MLP を処理する行ブロック |
| `max_batch_tokens` | 32768 | 1 回の forward に詰めるトークン数 |
| `fused_kernels` | true | false で検証用の ATen 参照実装 |
| `intra_op_threads` | 16 | 画像/動画リサイズの ATen CPU スレッド数（プロセス全体の設定） |

C ABI では `wemm_create` の `options_json` に同名のキーを指定できます（`"compute_dtype": "bf16"` など）。

### C ABI

`include/wemm/c_api.h` を参照（`tests/test_c_api.c` が使用例）。

```c
wemm_embedder* e = wemm_create("/path/to/WeMM-Embedding-2B", 0, NULL);
const char* inputs[] = {"{\"text\": \"a dog\"}", "{\"image\": \"cat.jpg\"}"};
float* out = malloc(sizeof(float) * 2 * wemm_dim(e, 0));
wemm_embed_json(e, inputs, 2, 0, out);
wemm_destroy(e);
```

## 実装と最適化

### 前処理の厳密な再現

- **チャットテンプレート**: `chat_template.jinja`（`add_generation_prompt=False`）を C++ で再実装。
  Jinja の `trim` は Python の `str.strip()` と同じ空白集合で処理。
- **トークナイザ**: Python 版と同じ Rust 実装 (`tokenizers` 0.22, onig) を使用。
  `tokenizer.json` の normalizer（`\n$` 除去）と `<embedding>` を付加する post-processor がそのまま効きます。
- **画像**: qwen-vl-utils は PIL の `Image.resize(BICUBIC)`、続いて Transformers の
  高速プロセッサが torchvision の uint8 AA bicubic を使います。前者は Pillow の
  `libImaging/Resample.c`（22bit 固定小数点の 2 パス畳み込み）を C++ に移植してビット一致、
  後者は LibTorch の同じ uint8 カーネル (`_upsample_bicubic2d_aa`) を同じメモリ配置で呼び出します。
  RGBA は Pillow の `paste` と同じ `DIV255` 演算で白背景に合成。EXIF の回転は PIL と同様に無視。
- **動画**: FFmpeg でデコードし、decord と同じ swscale 変換 (`SWS_BICUBIC`) で RGB 化。
  フレーム選択（`smart_nframes` / `linspace().round()`、区間指定）、qwen-vl-utils の
  float AA bicubic + clamp + 偶数丸め、Qwen3-VL 動画プロセッサの再リサイズ、
  `<t seconds>` タイムスタンプ生成まで同一手順。
- **M-RoPE 位置**: `get_rope_index`（動画はフレーム毎に分割、タイムスタンプ方式）を再実装。

### モデル

- 重みは safetensors から直接ロードします。埋め込みには不要な `lm_head` と MTP の重みは読みません。
- 実装は 2 つあり、`model.cpp` がバッチ毎の前計算を共有したうえで振り分けます。
  - `model_fused.cpp`: 既定の融合カーネル実装。
  - `model_reference.cpp`: 素の ATen で Transformers の計算を追う検証用実装で、カーネルの検証に使います。
- 複数入力は**パディング無しの可変長パッキング**で 1 回の forward にまとめます。
  注意は varlen、GDN は系列境界で状態をリセット、conv1d は系列先頭でゼロ埋めします。
- **最終層は `<embedding>` トークンの行だけを計算**します（最終層の注意クエリ・MLP・norm を N 行に限定）。
- Gated DeltaNet（線形注意層）:
  - **準備カーネル** `gdn_chunk_prepare`（チャンク×ヘッド並列）。
    - チャンク (64) 単位で conv 出力から q/k/v を集め、L2 正規化します。
    - `g = -exp(A)·softplus(a+dt_bias)` と `β = sigmoid(b)`、チャンク内の累積減衰を計算します。
    - WY 表現 `(I + tril(βKKᵀ⊙decay))⁻¹` を 16×16 ブロック分割の三角逆行列で求めます。
    - 64×128 タイルを列方向に半分ずつ処理するので、共有メモリは約 54 KB で sm_75 にも収まります。
  - **状態伝播カーネル**: 状態 S をレジスタに常駐させます。
    sm_80+ では次チャンクの `w` / `kd` / `u` を `cp.async` で先読みし（96 KB）、
    それ以外は単一バッファ（64 KB）で処理します。両者の結果はビット単位で同一です。
  - **チャンク毎の出力**: 並列な fp32 バッチ GEMM で計算します（cuBLASLt, `COMPUTE_32F_PEDANTIC`）。
  - **時間窓処理**: 長い入力はチャンクの窓に分けて処理し、窓をまたぐ状態は持ち越します。
  - すべて **fp32** で計算します（Transformers の参照実装と同じ精度。fp64 の逐次再帰に対する相対誤差は約 1e-6）。
- 融合カーネル: RMSNorm / LayerNorm（fp32 入力 → 16bit 出力）、q/k の head RMSNorm + 部分 RoPE、
  シグモイドゲート、SiLU×up、ゲート付き RMSNorm、ViT の 2D RoPE、varlen causal conv1d + SiLU。
  いずれも bf16 / fp16 のテンプレートです。
- GEMM は cuBLASLt を直接呼び、**bias / GELU(tanh) / 残差加算をエピローグに融合**します。
  - 残差は in-place で加算します。
  - 形状バケット毎にアルゴリズムを実測でオートチューニングします（`WEMM_GEMM_AUTOTUNE=0` で無効化）。
  - 記述子は呼び出し毎に作るので、スレッドセーフです。
  - 線形注意の入力射影 (qkv|z|b|a) と MLP の gate|up はそれぞれ 1 つの GEMM に結合し、出力次元を 128 の倍数にパディングしています。
- 注意: sm_80+ では FlashAttention-2（GQA ネイティブ）を使います。
  sm_75 では memory-efficient attention を使い、KV ヘッドは複製します。
- ATen の fp32 matmul は、NGC コンテナの `TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=1` によって TF32 に落とされます。
  このため fp32 が必要な箇所は、計算型を明示して cuBLASLt を呼んでいます。
- cuBLASLt のハンドルは自前でデバイス毎に持ちます。公式 LibTorch は名前を変えた cuBLASLt を同梱しているので、
  ATen のハンドルとリンク先ライブラリを混在させないためです。

### CPU 側

- 前処理は入力単位でスレッド並列。ATen の CPU リサイズは 16 スレッド程度で頭打ちになるため
  `intra_op_threads=16`、FFmpeg のフレームスレッドは 8 を既定値にしています。
- `Embedder` は次グループの前処理を GPU 計算と重ねます。

## 検証

開発用の Python ツール（推論には不要）で参照データを作り、C++ テストで比較します。

```bash
# 参照環境: transformers==5.2.0, qwen-vl-utils[decord]==0.0.14（リポジトリ推奨構成）
FORCE_QWENVL_VIDEO_READER=decord python native/tools/make_reference.py \
  --model /path/to/WeMM-Embedding-2B --data testdata --out ref_2B --fp32
./build/wemm_test_reference /path/to/WeMM-Embedding-2B ref_2B
```

`make_reference.py` は bf16 と真の fp32（TF32 無効、flash-linear-attention 不使用）で
Transformers を実行し、プロンプト・トークン列・画素テンソル・埋め込みを保存します。
`wemm_test_reference` は前処理のビット一致と、埋め込みの cos 類似度（単体/バッチ）を検査します。

その他:

- `wemm_bench_gdn [len...]`: GDN カーネルを検証して時間を計測します。fp64 の逐次再帰と比較するほか、
  単一バッファ版（sm_75 経路）と時間窓分割版が通常版とビット一致することも確認します。
- `wemm_test_reference ... --fp16 | --bf16 | --mem-efficient | --unfused | --emb-gpu | --gdn-window N`:
  各経路を個別に検証します（sm_75 経路は `--fp16 --mem-efficient`）。
- `wemm_bench_gemm`: 主要 GEMM 形状で cuBLASLt 経路と ATen を比較。
- `tools/bench_hf.py`, `tools/bench_throughput_hf.py`: Transformers のベースライン計測。

## 制限事項

- 音声入力は元モデル同様に非対応です。
- GPU は Turing (sm_75) 以降です。sm_75 の経路は 3090 上で同じコードを強制して検証しましたが、Turing 実機では未検証です。
- 1 つの `Embedder` を複数スレッドから同時に呼ばないでください。
- 画像デコーダの差異: OpenCV は 4ch PNG をすべて RGBA として扱います。そのため透過付きパレット PNG と
  LA 画像は、PIL がアルファを捨てる場合でも白背景で合成されます。
- 動画のフレーム数はコンテナの `nb_frames`（無い場合は長さ×fps）を使います（decord と同じ）。
  可変フレームレートの動画では、decord とフレーム番号の付け方が異なる可能性があります。
- fp16 で万一オーバーフローした場合は、非有限値を検知してエラーにします（`compute_dtype=bf16` を推奨、sm_80+）。
