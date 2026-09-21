#!/usr/bin/env python3
"""
Select RAFT training examples from dumped rollouts.
Writes a train.jsonl of selected prompt+completion pairs (token ids).

Usage:
  python -m scripts.raft_select --run=myrun --topk=1
  python -m scripts.raft_select --data-dir=/path/to/raft_data/myrun --topk=2 --out=train.jsonl
"""
import argparse
import os
import json
from collections import defaultdict
from nanochat.common import get_base_dir

parser = argparse.ArgumentParser()
parser.add_argument("--run", type=str, default=None, help="run name (used to find raft_data/<run>)")
parser.add_argument("--data-dir", type=str, default=None, help="explicit raft data dir")
parser.add_argument("--topk", type=int, default=1, help="top-k completions per prompt to keep (by reward)")
parser.add_argument("--threshold", type=float, default=None, help="minimum reward to keep")
parser.add_argument("--out", type=str, default="train.jsonl", help="output train jsonl file")
args = parser.parse_args()

base = args.data_dir or (os.path.join(get_base_dir(), "raft_data", args.run) if args.run else None)
if not base or not os.path.isdir(base):
    raise SystemExit(f"raft data dir not found: {base}")

# Read all rollouts under base/<step>/*.jsonl
entries_by_prompt = defaultdict(list)
for step_name in sorted(os.listdir(base), key=lambda x: int(x) if x.isdigit() else x):
    step_dir = os.path.join(base, step_name)
    if not os.path.isdir(step_dir):
        continue
    for fname in os.listdir(step_dir):
        if not fname.endswith('.jsonl'):
            continue
        path = os.path.join(step_dir, fname)
        with open(path, 'r') as fh:
            for line in fh:
                try:
                    item = json.loads(line)
                except Exception:
                    continue
                pid = item.get('prompt_id')
                if pid is None:
                    continue
                entries_by_prompt[pid].append(item)

# Select top-k by reward (and respect threshold)
selected = []
for pid, items in entries_by_prompt.items():
    items_sorted = sorted(items, key=lambda x: x.get('reward', 0.0), reverse=True)
    if args.threshold is not None:
        items_sorted = [x for x in items_sorted if x.get('reward', 0.0) >= args.threshold]
    keep = items_sorted[:args.topk]
    for it in keep:
        out = {
            'prompt_id': pid,
            'prompt_tokens': it.get('prompt_tokens'),
            'sequence_tokens': it.get('sequence_tokens'),
            'reward': it.get('reward'),
            'step': it.get('step'),
        }
        selected.append(out)

# Write train jsonl
with open(args.out, 'w') as fh:
    for s in selected:
        fh.write(json.dumps(s) + "\n")

print(f"Wrote {len(selected)} examples to {args.out}")
