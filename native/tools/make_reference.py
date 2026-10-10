#!/usr/bin/env python3
"""Dump reference inputs/outputs from the Transformers implementation.

Development-only: the native runtime never imports Python. This script records,
for a fixed set of cases, the rendered prompt, token ids, pixel tensors and the
final embeddings (bf16 and fp32 runs) so the C++ port can be checked against them.

    FORCE_QWENVL_VIDEO_READER=decord python make_reference.py \
        --model /path/to/WeMM-Embedding-2B --data /path/to/testdata --out ref/
"""

import argparse
import json
import os

import torch
from qwen_vl_utils import process_vision_info
from safetensors.torch import save_file
from transformers import AutoModel, AutoProcessor


def cases(data):
    d = lambda name: os.path.join(data, name)  # noqa: E731
    return {
        "text": [{"type": "text", "text": "A dog is running on a beach."}],
        "text_long": [{"type": "text", "text": "Represent this document for retrieval:\n" + " ".join(
            f"Paragraph {i}: the quick brown fox jumps over the lazy dog, café naïve 東京 🚀." for i in range(40))}],
        "image": [{"type": "image", "image": d("hero.jpg")}],
        "image_small": [{"type": "image", "image": d("small.png")}],
        "image_chart": [{"type": "image", "image": d("chart.webp")}],
        "image_large": [{"type": "image", "image": d("large.jpg")}],
        "image_text": [{"type": "image", "image": d("hero.jpg")},
                       {"type": "text", "text": "Find a similar everyday scene."}],
        "video": [{"type": "video", "video": d("clip.mp4")}],
        "video_long": [{"type": "video", "video": d("long.mp4")}],
        "text_video": [{"type": "text", "text": "Retrieve the matching clip:"},
                       {"type": "video", "video": d("clip.mp4")}],
    }


def prepare(processor, content):
    messages = [{"role": "user", "content": content}]
    prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    images, videos, video_kwargs = process_vision_info(
        messages, image_patch_size=16, return_video_kwargs=True, return_video_metadata=True)
    video_metadata = None
    if videos is not None:
        videos, video_metadata = (list(x) for x in zip(*videos))
    inputs = processor(text=prompt, images=images, videos=videos, video_metadata=video_metadata,
                       return_tensors="pt", **video_kwargs)
    return prompt, inputs, video_metadata


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--fp32", action="store_true", help="also run an fp32 reference")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)
    prepared = {}
    for name, content in cases(args.data).items():
        prompt, inputs, meta = prepare(processor, content)
        prepared[name] = (content, prompt, inputs, meta)

    results = {}
    for dtype in ([torch.bfloat16, torch.float32] if args.fp32 else [torch.bfloat16]):
        model = AutoModel.from_pretrained(args.model, trust_remote_code=True, dtype=dtype).cuda().eval()
        for name, (_, _, inputs, _) in prepared.items():
            with torch.inference_mode():
                emb = model.embedding(**inputs.to("cuda")).float().cpu()
            results.setdefault(name, {})[str(dtype).split(".")[-1]] = emb
        del model
        torch.cuda.empty_cache()

    manifest = {}
    for name, (content, prompt, inputs, meta) in prepared.items():
        tensors = {k: v.contiguous() for k, v in inputs.items() if isinstance(v, torch.Tensor)}
        for k, v in results[name].items():
            tensors[f"embedding_{k}"] = v.contiguous()
        save_file(tensors, os.path.join(args.out, f"{name}.safetensors"))
        manifest[name] = {
            "content": content,
            "prompt": prompt,
            "video_metadata": [
                {"fps": float(m["fps"]), "frames_indices": [int(i) for i in m["frames_indices"]],
                 "total_num_frames": int(m["total_num_frames"])} for m in meta] if meta else None,
        }
    with open(os.path.join(args.out, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=1, ensure_ascii=False)

    names = list(prepared)
    if args.fp32:
        for n in names:
            a, b = results[n]["bfloat16"], results[n]["float32"]
            print(f"{n:12s} tokens={prepared[n][2]['input_ids'].shape[1]:6d} cos(bf16,fp32)={torch.cosine_similarity(a, b).item():.6f}")


if __name__ == "__main__":
    main()
