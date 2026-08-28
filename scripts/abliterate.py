#!/usr/bin/env python3
"""
Abliteration tool for Hugging Face transformers models.

Permanent abliteration for Hugging Face models.
Modifies the model weights directly by adding a bias to the output projections
(o_proj and down_proj) that subtracts the refusal direction from the residual stream.

Collects activations on refusal and compliant examples, computes the
refusal direction, and modifies the model's forward pass to suppress
refusal behaviour.

Usage:
    python3 abliterate.py --model MODEL_PATH --refusal FILE --compliant FILE --output OUT_DIR
"""
import os
import argparse
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import json

class TextDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length=512):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.texts[idx],
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        return enc.input_ids.squeeze(0), enc.attention_mask.squeeze(0)

def collect_activations(model, dataloader, device, layers=None):
    """Collect hidden states from decoder layers."""
    activations = []
    hooks = []
    layer_outputs = {}

    def hook_fn(layer_idx):
        def fn(module, input, output):
            layer_outputs[layer_idx] = output[0].detach().cpu()
        return fn

    if layers is None:
        for i, layer in enumerate(model.model.layers):
            hook = layer.register_forward_hook(hook_fn(i))
            hooks.append(hook)
    else:
        for i in layers:
            hook = model.model.layers[i].register_forward_hook(hook_fn(i))
            hooks.append(hook)

    model.eval()
    with torch.no_grad():
        for input_ids, attn_mask in tqdm(dataloader, desc="Collecting activations"):
            input_ids = input_ids.to(device)
            attn_mask = attn_mask.to(device)
            _ = model(input_ids, attention_mask=attn_mask)
            batch_acts = {idx: layer_outputs.pop(idx) for idx in list(layer_outputs.keys())}
            activations.append(batch_acts)

    for hook in hooks:
        hook.remove()
    return activations

def compute_refusal_direction(refusal_acts, compliant_acts):
    """Compute the mean direction vector per layer."""
    layer_dirs = {}
    all_layers = set(refusal_acts[0].keys())
    for layer in all_layers:
        ref_stack = torch.cat(
            [act[layer].mean(dim=1, keepdim=False) for act in refusal_acts], dim=0
        )
        comp_stack = torch.cat(
            [act[layer].mean(dim=1, keepdim=False) for act in compliant_acts], dim=0
        )
        ref_mean = ref_stack.mean(dim=0, keepdim=True)
        comp_mean = comp_stack.mean(dim=0, keepdim=True)
        direction = ref_mean - comp_mean
        direction = direction / direction.norm()
        layer_dirs[layer] = direction
    return layer_dirs

def apply_abliteration_permanently(model, layer_dirs, alpha=1.0):
    """
    Permanently subtract the refusal direction from the residual stream
    by adding a bias to the o_proj and down_proj layers.
    """
    for layer_idx, direction in layer_dirs.items():
        layer = model.model.layers[layer_idx]
        d_vec = direction.to(model.device) * alpha  # shape (hidden_dim,)

        # --- Modify Self-Attention o_proj ---
        o_proj = layer.self_attn.o_proj
        if o_proj.bias is None:
            # Create a bias parameter if it doesn't exist
            o_proj.bias = nn.Parameter(torch.zeros(o_proj.out_features, device=o_proj.weight.device))
        # Add the negative direction (so we subtract it from the residual)
        o_proj.bias.data.add_(-d_vec)

        # --- Modify MLP down_proj ---
        down_proj = layer.mlp.down_proj
        if down_proj.bias is None:
            down_proj.bias = nn.Parameter(torch.zeros(down_proj.out_features, device=down_proj.weight.device))
        down_proj.bias.data.add_(-d_vec)

    return model

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True, help='Path to HF model')
    parser.add_argument('--refusal', required=True, help='Refusal examples file')
    parser.add_argument('--compliant', required=True, help='Compliant examples file')
    parser.add_argument('--output', required=True, help='Output directory')
    parser.add_argument('--alpha', type=float, default=1.0, help='Scaling factor')
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--max_length', type=int, default=512)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Loading model from {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    )
    model = model.to(device)

    with open(args.refusal) as f:
        refusal_texts = [l.strip() for l in f if l.strip()]
    with open(args.compliant) as f:
        compliant_texts = [l.strip() for l in f if l.strip()]

    print(f"Refusal: {len(refusal_texts)}, Compliant: {len(compliant_texts)}")

    ref_dataset = TextDataset(refusal_texts, tokenizer, args.max_length)
    comp_dataset = TextDataset(compliant_texts, tokenizer, args.max_length)
    ref_loader = DataLoader(ref_dataset, batch_size=args.batch_size, shuffle=False)
    comp_loader = DataLoader(comp_dataset, batch_size=args.batch_size, shuffle=False)

    print("Collecting refusal activations...")
    refusal_acts = collect_activations(model, ref_loader, device)
    print("Collecting compliant activations...")
    compliant_acts = collect_activations(model, comp_loader, device)

    print("Computing refusal direction...")
    layer_dirs = compute_refusal_direction(refusal_acts, compliant_acts)

    print("Applying permanent weight modification...")
    model = apply_abliteration_permanently(model, layer_dirs, args.alpha)

    print(f"Saving abliterated model to {args.output}")
    model.save_pretrained(args.output)
    tokenizer.save_pretrained(args.output)

    # Save directions for reference
    dirs_to_save = {str(k): v.cpu().tolist() for k, v in layer_dirs.items()}
    with open(os.path.join(args.output, 'refusal_directions.json'), 'w') as f:
        json.dump(dirs_to_save, f)

    print("Done. This model is now permanently abliterated.")

if __name__ == '__main__':
    main()
