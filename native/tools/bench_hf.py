#!/usr/bin/env python3
"""Baseline latency of the Transformers implementation (model forward only).

    python bench_hf.py --model DIR --data testdata [--no-fla] [--iters 20]

Uses the same cases as make_reference.py. With --no-fla the pure-PyTorch
gated delta rule fallback is forced even if flash-linear-attention is installed.
"""

import argparse
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(__file__))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--ref", help="reference dir with embedding_float32 for accuracy")
    ap.add_argument("--no-fla", action="store_true")
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--cases", default="text,text_long,image,image_large,video,video_long")
    args = ap.parse_args()

    if args.no_fla:
        import transformers.utils.import_utils as iu
        iu.is_flash_linear_attention_available = lambda: False
        import transformers.utils as tu
        tu.is_flash_linear_attention_available = lambda: False

    from transformers import AutoModel, AutoProcessor
    from make_reference import cases, prepare

    processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModel.from_pretrained(args.model, trust_remote_code=True, dtype=torch.bfloat16).cuda().eval()
    mod = sys.modules[type(model.model.language_model.layers[0].linear_attn).__module__]
    print("chunk_gated_delta_rule:", model.model.language_model.layers[0].linear_attn.chunk_gated_delta_rule.__module__)

    all_cases = cases(args.data)
    for name in args.cases.split(","):
        content = all_cases[name]
        if args.batch > 1:
            # same conversation repeated (right-padded batch)
            prompt, _, _ = prepare(processor, content)
            from qwen_vl_utils import process_vision_info
            messages = [[{"role": "user", "content": content}] for _ in range(args.batch)]
            images, videos, vk = process_vision_info(messages, image_patch_size=16, return_video_kwargs=True,
                                                     return_video_metadata=True)
            meta = None
            if videos is not None:
                videos, meta = (list(x) for x in zip(*videos))
            inputs = processor(text=[prompt] * args.batch, images=images, videos=videos, video_metadata=meta,
                               padding=True, return_tensors="pt", **vk)
        else:
            _, inputs, _ = prepare(processor, content)
        inputs = inputs.to("cuda")
        with torch.inference_mode():
            for _ in range(3):
                emb = model.embedding(**inputs)
            torch.cuda.synchronize()
            t = time.perf_counter()
            for _ in range(args.iters):
                emb = model.embedding(**inputs)
            torch.cuda.synchronize()
            ms = (time.perf_counter() - t) * 1000 / args.iters
        extra = ""
        if args.ref:
            from safetensors.torch import load_file
            ref = load_file(os.path.join(args.ref, f"{name}.safetensors"))["embedding_float32"].cuda()
            extra = f"  cos(fp32)={torch.cosine_similarity(emb[:1].float(), ref).item():.6f}"
        print(f"{name:12s} tokens={inputs['input_ids'].shape[1]:6d} batch={args.batch} forward={ms:9.2f} ms{extra}")


if __name__ == "__main__":
    main()
