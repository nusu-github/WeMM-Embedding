#!/usr/bin/env python3
"""End-to-end throughput of the Transformers pipeline (preprocessing + forward).

    python bench_throughput_hf.py --model DIR --input inputs.jsonl --batch 32
Input lines use the wemm_embed shorthand: {"text": ...} | {"image": path} | {"video": path}.
"""
import argparse
import json
import time

import torch
from qwen_vl_utils import process_vision_info
from transformers import AutoModel, AutoProcessor


def to_messages(item):
    content = []
    if "image" in item:
        content.append({"type": "image", "image": item["image"]})
    if "video" in item:
        content.append({"type": "video", "video": item["video"]})
    if "text" in item:
        content.append({"type": "text", "text": item["text"]})
    return [{"role": "user", "content": content}]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--input", required=True)
    ap.add_argument("--batch", type=int, default=32)
    args = ap.parse_args()
    items = [json.loads(l) for l in open(args.input) if l.strip()]
    processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModel.from_pretrained(args.model, trust_remote_code=True, dtype=torch.bfloat16).cuda().eval()

    def run(batch_items):
        msgs = [to_messages(i) for i in batch_items]
        texts = [processor.apply_chat_template(m, tokenize=False, add_generation_prompt=False) for m in msgs]
        images, videos, vk = process_vision_info(msgs, image_patch_size=16, return_video_kwargs=True,
                                                 return_video_metadata=True)
        meta = None
        if videos is not None:
            videos, meta = (list(x) for x in zip(*videos))
        inputs = processor(text=texts, images=images, videos=videos, video_metadata=meta, padding=True,
                           return_tensors="pt", **vk).to("cuda")
        with torch.inference_mode():
            return model.embedding(**inputs).float().cpu()

    run(items[: args.batch])  # warm-up
    torch.cuda.synchronize()
    t = time.perf_counter()
    for i in range(0, len(items), args.batch):
        run(items[i:i + args.batch])
    torch.cuda.synchronize()
    dt = time.perf_counter() - t
    print(f"{len(items)} inputs in {dt:.2f}s  ->  {len(items) / dt:.1f} inputs/s (batch {args.batch})")


if __name__ == "__main__":
    main()
