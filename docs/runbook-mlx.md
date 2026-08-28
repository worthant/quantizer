# Runbook: an MLX release

`foundry-mlx3.sh` is self contained. It does not need `foundry.sh`, and it does
not build llama.cpp, which saves fifteen minutes on a box that only does MLX.

## What is different from GGUF

Worth reading once before touching anything.

**A quant is a directory, not a file.** A GGUF rung is one `.gguf`. An MLX rung
is `config.json` plus sharded safetensors plus the tokenizer. The loader takes
the directory or a repository id, never a file inside it. That is why every rung
of the ladder needs its own repository on the hub.

**The method leaves no trace in the output.** Plain rounding, clipping search,
distillation and per layer allocation all write the same affine quantized
checkpoint. Nothing downstream needs to know which produced it.

**`mlx_vlm.convert` is the primary tool, not `mlx_lm.convert`.** It is the one
that keeps the vision tower, and its output loads in the desktop app directly.
`mlx-lm` is only for a method `mlx-vlm` does not have, and then the result is
text only and the tower goes back on with `mlx3_reattach`.

**One bit is loadable but not producible.** `mlx-vlm` can run a one bit affine
checkpoint: packed uint32 weights, scales and biases, group size 32, 64 or 128,
`bits: 1` in the config. Nothing in the stack produces one. That gap is the most
interesting unclaimed thing in this area.

## Size arithmetic

Fitted on twenty five measured builds, good to about 0.05 GB for a 27B model:

```
GB = 0.92 + 3.365 * bits_per_weight
```

`0.92` is the vision tower, never quantized. `bits_per_weight` is the bit count
plus the cost of the scales, which is 32 bits of scale and offset per group.
Group 128 adds 0.25 to every weight, group 64 adds 0.5, group 32 adds 1.0.

The lever that matters, measured on one reference:

| build | bpw | GB | mean KLD |
| --- | --- | --- | --- |
| 3 bit, group 64 | 3.50 | 12.70 | 0.222903 |
| 3 bit, group 32 | 4.00 | 14.38 | 0.180793 |
| 4 bit, group 128 | 4.25 | 15.22 | never measured by anyone |
| 4 bit, group 64 | 4.50 | 16.05 | 0.055803 |
| 4 bit, group 32 | 5.00 | 17.75 | 0.044101 |
| 5 bit, group 64 | 5.50 | 19.43 | 0.015532 |

One extra level, 3 to 4 bits at the same group, divides divergence by four. Half
a bit spent on twice as many scales, group 64 to 32, divides it by 1.27. Levels
are worth about three times what scales are worth per unit of size. Every mixed
layout in the public survey sits on that plain line, because they all trade the
expensive thing for the cheap one.

## The box

```bash
export HF_TOKEN=hf_...
curl -sL https://raw.githubusercontent.com/<ORG>/<REPO>/main/foundry-mlx3.sh -o /mlx3.sh
source /mlx3.sh
mlx3_setup
mlx3_check
mlx3_persist
```

> [!IMPORTANT]
> `mlx3_check` prints the device. A cpu-only wheel says `cpu`, and then nothing
> below is worth starting. Then run `mlx3_bench`: it measures what the card
> actually does on the shape this work is made of. Tens of TFLOP/s instead of
> hundreds means the tensor cores are not being used and every estimate is off
> by roughly ten times.

The `mlx[cuda13]` wheel needs an Nvidia driver of 580 or newer. Filter the offer
by driver version before renting, especially for Blackwell parts.

`mlx3_disk` prints what each job needs on disk. Take at least 1 TB.

## Downloads

```bash
mlx3_get ref             # bf16 MLX checkpoint, the zero point. About 51 GB
mlx3_get eval            # eval_neutral
mlx3_get calib           # the calibration text, only for distillation
mlx3_get teacher         # the 8 bit build, only for distillation. About 30 GB
mlx3_get src             # original weights, only on a box that builds a new rung
mlx3_get base 4bit       # one of our published rungs
```

The bf16 reference is a pure format change from the original weights, so it
reproduces byte for byte anywhere. That is what makes it publishable as a shared
point of comparison. If it does not exist yet, `mlx_reference` in `foundry.sh`
builds it.

## The reference cache

```bash
mlx3_cache
```

One bf16 forward pass over 24 chunks, then 24 GB of logits on disk, about twelve
minutes. Every later measurement on this box reuses it. This is the MLX
equivalent of llama.cpp's `.kld` file, and skipping it means paying for the same
forward pass once per build measured.

## Building a rung

Three ways, in increasing order of what they buy.

**Plain.** Straight through `mlx_vlm.convert`, keeps the vision tower.

```bash
mlx3_quant 3 64
mlx3_quant 4 128
```

**Clipping search.** The built in quantizer lays the available levels evenly
between the smallest and the largest weight of each group, so one outlier costs
the other sixty three weights resolution. llama.cpp has searched for a better
range inside every k-quant since 2023; no tool in the MLX stack does. This tries
a grid of narrower ranges per group, keeps the one with the smallest squared
error, and hands the clipped weights back to `mx.quantize` so the packing is done
by the library and the output stays bit compatible.

```bash
mlx3_clip /mlx/<stem>-MLX-4bit
```

Output size equals input size to the byte, so the comparison needs no
interpolation. At `alpha = 1.000` it reproduces exactly what the library does
today, so the result can only tie or beat it. Read the printed squared error
drop before measuring: under two percent means clipping has nothing to give at
that bit count.

**Per tensor bit layout.** Every tensor gets its own bit count and group size.
This is the lever worth twenty percent where the other two are worth two to five.

```bash
mlx3_rules                        # writes the hand written default rule set
mlx3_gen 13.7                     # or generate one that lands on a target size
mlx3_price                        # what it costs, per role, before building
mlx3_build AD-13.7                # build it from bf16, with the clipping search
mlx3_try AD-13.7                  # build and rank on six chunks instead of 24
```

`mlx3_try` is the sweep tool: under two minutes of card time per candidate
against an hour for a distillation run. Sweep layouts first, distil the winner
last.

## Distillation

Gradient descent on the scales and offsets of an already quantized model,
minimizing divergence against a teacher. The integers stay where the base
quantization put them, which is why building a better base with `mlx3_clip`
first is worth doing: clipping picks better integers, this tunes the scales
around them, and neither replaces the other.

```bash
mlx3_mem 3 64 512 targets         # memory arithmetic before renting
mlx3_targets                      # teacher logits once, serves both lanes
mlx3_dwq 3 64 /mlx/<stem>-MLX-3bit-CLIP
```

It pays between 2 and 4 bits and does nearly nothing at 6 and 8, where there is
not enough damage left to repair. Measured on a 27B model at 3 bits, group 64,
batch 4, window 512, 150 steps, 8 bit teacher: validation loss 0.296 to 0.155,
so 47.6 percent, in one hour on an H200 NVL at 70.3 GB peak.

> [!WARNING]
> The base must be uniform. A mixed layout carries per tensor bits and group
> sizes, and passing `--bits` on top of that is undefined. Clip a mixed rung and
> publish it; distil only the uniform ones.

What to watch in the log: training and validation loss are the same quantity the
table measures. Falling steadily is right. Oscillating without falling means the
rate is too high, halve it. Falling a few percent over the whole run means too
low, triple it.

Three environment settings that took a day to find and are now defaults. They
are documented at the top of `foundry-mlx3.sh` and in
[troubleshooting.md](troubleshooting.md): `MLX_USE_CUDA_GRAPHS=0`,
`MLX_CUDA_GRAPH_CACHE_SIZE=2048`, and a 512 token window rather than 2048.

## Measuring

```bash
mlx3_kld /mlx/<stem>-MLX-4bit-CLIP
mlx3_all                          # everything on this box not yet measured
mlx3_table                        # the table, plus how it stands against the targets
mlx3_diff BUILD_A BUILD_B         # two builds across every metric, not just the mean
```

`mlx3_diff` exists because the mean hides what matters. At 3 bits the clipping
search moved the median 5.6 percent in the right direction and the 99th
percentile 3.3 percent in the wrong one, and the mean, which the tail dominates,
showed 0.8 percent and told you nothing.

Three checks decide what counts as a result at all:

```bash
mlx3_verify                       # runs the first, explains the other two
mlx3_fp_remote                    # are the four published 4 bit builds the same file
mlx3_repeat /mlx/<stem>-MLX-4bit  # is the measurement repeatable on this card
mlx3_tail                         # which way llama.cpp's tail cutoff moves it
```

`mlx3_repeat` measures the noise floor. Nothing below it is a result, and the
figure belongs next to every number in the published card.

## Vision and publishing

```bash
mlx3_reattach /mlx/X-4bit-CLIP-DWQ /mlx/X-4bit /mlx/X-4bit-CLIP-DWQ-VL
mlx3_push /mlx/X-4bit-CLIP-DWQ-VL 4bit-CLIP-DWQ
```

`mlx3_reattach` puts the text tensors from an mlx-lm build next to the untouched
tower from an mlx-vlm build. It refuses to write when it replaced nothing, which
is a lesson learned the hard way: an earlier version reported "replaced 0" and
wrote out the untouched base with a new quantization block, and that file was
published.

The result is a claim until it has been loaded and shown a picture.

## Two box plan

`mlx3_box hub` and `mlx3_box dwq` print the exact command list for each role,
with the checkpoints to verify at every stage. The hub box is cheap and bound by
disk and a single forward pass; the distillation box is expensive and should be
rented the moment the hub box starts building, not before.

A single 94 GB card runs the whole plan on its own, at roughly twice the time per
run. The second box is a speedup, not a requirement.
