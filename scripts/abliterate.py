#!/usr/bin/env python3
"""
Abliteration tool for Hugging Face transformers models.
Collects activations on refusal and compliant examples, computes the
refusal direction, and modifies the model's forward pass to suppress
refusal behaviour.

Usage:
    python3 abliterate.py --model MODEL_PATH --refusal FILE --compliant FILE --output OUT_DIR
"""
import os
import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import json
import numpy as np

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
    """
    Collect hidden states from specified layers (or all decoder layers) for each sample.
    Returns a list of dicts: {layer_index: tensor_of_hidden_states}
    """
    activations = []
    hooks = []
    layer_outputs = {}

    def hook_fn(layer_idx):
        def fn(module, input, output):
            # output is a tuple; first element is hidden states
            layer_outputs[layer_idx] = output[0].detach().cpu()
        return fn

    # Register hooks for all decoder layers by default
    if layers is None:
        # Assume model.model.layers exists for most transformers
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
            # copy the layer outputs
            batch_acts = {idx: layer_outputs.pop(idx) for idx in list(layer_outputs.keys())}
            activations.append(batch_acts)

    for hook in hooks:
        hook.remove()

    return activations

def compute_refusal_direction(refusal_acts, compliant_acts):
    """
    Compute the mean difference vector (refusal direction) per layer.
    Both inputs are lists of dicts {layer: tensor (batch, seq, hidden)}.
    We pool over the sequence dimension (mean) and then over batch.
    """
    layer_dirs = {}
    # assume all dicts have the same layers
    all_layers = set(refusal_acts[0].keys())

    for layer in all_layers:
        # Stack and pool over sequence (mean) for each sample
        ref_stack = torch.cat(
            [act[layer].mean(dim=1, keepdim=False) for act in refusal_acts], dim=0
        )  # (total_samples, hidden)
        comp_stack = torch.cat(
            [act[layer].mean(dim=1, keepdim=False) for act in compliant_acts], dim=0
        )
        ref_mean = ref_stack.mean(dim=0, keepdim=True)   # (1, hidden)
        comp_mean = comp_stack.mean(dim=0, keepdim=True)
        direction = ref_mean - comp_mean
        # Normalize to unit vector
        direction = direction / direction.norm()
        layer_dirs[layer] = direction

    return layer_dirs

def apply_abliteration(model, layer_dirs, alpha=1.0):
    """
    Modify the model's forward pass by subtracting the refusal direction
    from the hidden states after each layer.
    This is done via a permanent forward hook that subtracts the direction
    from the layer's output (residual stream).
    """
    for layer_idx, direction in layer_dirs.items():
        layer = model.model.layers[layer_idx]
        # Store direction in model's attribute for reference
        if not hasattr(model, '_abliteration_dirs'):
            model._abliteration_dirs = {}
        model._abliteration_dirs[layer_idx] = direction.to(model.device) * alpha

        def make_hook(idx, dir_vec):
            def hook(module, input, output):
                # output is a tuple (hidden_states, ...) for most layers
                if isinstance(output, tuple):
                    hidden = output[0]
                    batch_size, seq_len, hidden_dim = hidden.shape
                    # Expand direction to (1,1,hidden_dim) and subtract
                    shifted = hidden - dir_vec.unsqueeze(0).unsqueeze(0)
                    return (shifted,) + output[1:]
                else:
                    hidden = output
                    shifted = hidden - dir_vec.unsqueeze(0).unsqueeze(0)
                    return shifted
            return hook

        layer.register_forward_hook(make_hook(layer_idx, direction.to(model.device)))

    return model

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True, help='HF model path or name')
    parser.add_argument('--refusal', required=True, help='Text file with refusal examples (one per line)')
    parser.add_argument('--compliant', required=True, help='Text file with compliant examples (one per line)')
    parser.add_argument('--output', required=True, help='Output directory for modified model')
    parser.add_argument('--alpha', type=float, default=1.0, help='Scaling factor for direction')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size for activation collection')
    parser.add_argument('--max_length', type=int, default=512, help='Max sequence length')
    parser.add_argument('--device', default='cuda', help='Device to use')
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Loading model from {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    )
    model = model.to(device)

    # Load texts
    with open(args.refusal, 'r', encoding='utf-8') as f:
        refusal_texts = [line.strip() for line in f if line.strip()]
    with open(args.compliant, 'r', encoding='utf-8') as f:
        compliant_texts = [line.strip() for line in f if line.strip()]

    print(f"Refusal examples: {len(refusal_texts)}, Compliant: {len(compliant_texts)}")

    # Prepare dataloaders
    ref_dataset = TextDataset(refusal_texts, tokenizer, args.max_length)
    comp_dataset = TextDataset(compliant_texts, tokenizer, args.max_length)
    ref_dataloader = DataLoader(ref_dataset, batch_size=args.batch_size, shuffle=False)
    comp_dataloader = DataLoader(comp_dataset, batch_size=args.batch_size, shuffle=False)

    # Collect activations
    print("Collecting activations for refusal examples...")
    refusal_acts = collect_activations(model, ref_dataloader, device)
    print("Collecting activations for compliant examples...")
    compliant_acts = collect_activations(model, comp_dataloader, device)

    # Compute direction
    print("Computing refusal direction per layer...")
    layer_dirs = compute_refusal_direction(refusal_acts, compliant_acts)

    # Apply abliteration
    print("Applying abliteration...")
    model = apply_abliteration(model, layer_dirs, alpha=args.alpha)

    # Save the modified model
    print(f"Saving modified model to {args.output}")
    model.save_pretrained(args.output)
    tokenizer.save_pretrained(args.output)

    # Also save the directions for reference
    dirs_to_save = {str(k): v.cpu().tolist() for k, v in layer_dirs.items()}
    with open(os.path.join(args.output, 'refusal_directions.json'), 'w') as f:
        json.dump(dirs_to_save, f)

    print("Done.")

if __name__ == '__main__':
    main()
