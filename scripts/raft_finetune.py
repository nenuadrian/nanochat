#!/usr/bin/env python3
"""
Simple RAFT fine-tuning runner.
Loads a `train.jsonl` produced by `scripts/raft_select.py` (or any JSONL with
`prompt_tokens` and `sequence_tokens` arrays of token ids) and runs supervised
cross-entropy fine-tuning, saving checkpoints via `nanochat.checkpoint_manager.save_checkpoint`.

Usage:
  python -m scripts.raft_finetune --train-file=raft_train.jsonl --run=raft1 --epochs=1

This mirrors the optimizer setup used in `scripts/chat_rl.py`.
"""
import argparse
import json
import os
from pathlib import Path
from itertools import chain

import torch
from torch.utils.data import DataLoader, Dataset

from nanochat.checkpoint_manager import save_checkpoint, load_model
from nanochat.common import compute_init, get_base_dir, print0


class RaftDataset(Dataset):
    def __init__(self, path, max_len=1024):
        self.examples = []
        self.max_len = max_len
        with open(path, 'r') as fh:
            for line in fh:
                obj = json.loads(line)
                # sequence_tokens contains full sequence including prompt+completion
                toks = obj.get('sequence_tokens') or obj.get('completion_tokens')
                if toks is None:
                    # fallback: try to concatenate prompt + sequence
                    p = obj.get('prompt_tokens', [])
                    s = obj.get('sequence_tokens', [])
                    toks = p + s
                # truncate to max_len from the end
                toks = toks[-self.max_len:]
                self.examples.append(toks)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        toks = self.examples[idx]
        return torch.tensor(toks, dtype=torch.long)


def collate_fn(batch, pad_id=0):
    # pad to longest in batch
    maxl = max(x.size(0) for x in batch)
    out = torch.full((len(batch), maxl), pad_id, dtype=torch.long)
    for i, x in enumerate(batch):
        out[i, -x.size(0):] = x
    inputs = out[:, :-1]
    targets = out[:, 1:].clone()
    return inputs, targets


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-file', type=str, required=True)
    parser.add_argument('--run', type=str, default='raft-finetune')
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--max-len', type=int, default=1024)
    parser.add_argument('--device', type=str, default='')
    parser.add_argument('--output-tag', type=str, default=None)
    args = parser.parse_args()

    device_type = args.device or ''
    ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init(device_type)
    master = ddp_rank == 0

    # Load a base SFT model to fine-tune (source 'sft') and place in train mode
    model, tokenizer, meta = load_model('sft', device, phase='train')
    model.train()

    # Data
    ds = RaftDataset(args.train_file, max_len=args.max_len)
    if len(ds) == 0:
        raise SystemExit('No training examples found in ' + args.train_file)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True, collate_fn=lambda b: collate_fn(b, pad_id=tokenizer.encode_special('<|assistant_end|>')))

    # Optimizer like chat_rl uses
    optimizer = model.setup_optimizer(unembedding_lr=0.004, embedding_lr=0.2, matrix_lr=0.02, weight_decay=0.0)
    for g in optimizer.param_groups:
        g['lr'] = g.get('lr', 1e-3) * 0.01  # scale down default; user-managed via --lr
        g['initial_lr'] = g['lr']
    # override with explicit LR
    for g in optimizer.param_groups:
        g['lr'] = args.lr

    step = 0
    checkpoint_dir = os.path.join(get_base_dir(), 'chatrl_checkpoints', args.output_tag or args.run)
    for ep in range(args.epochs):
        for inputs, targets in dl:
            inputs = inputs.to(device)
            targets = targets.to(device)
            loss = model(inputs, targets, loss_reduction='mean')
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if master and step % 100 == 0:
                print0(f"RAFT step {step} ep {ep} loss={float(loss.item()):.4f}")
                # save checkpoint shard for safety
                model_data = {k: v.cpu() for k, v in model.state_dict().items()}
                try:
                    save_checkpoint(checkpoint_dir, step, model_data, optimizer.state_dict(), {'phase': 'raft', 'run': args.run}, rank=ddp_rank)
                except Exception as e:
                    print0(f"Warning: failed to save checkpoint: {e}")
            step += 1

    print0('RAFT finetune complete')

if __name__ == '__main__':
    main()
