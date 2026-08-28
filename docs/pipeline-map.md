# Pipeline map

Where everything lives, and why it is split that way.

## Repositories on Hugging Face

Every model release creates two repositories on the GGUF side, and one per rung
plus one for metrics on the MLX side. That split is deliberate: the weights
repository stays clean enough for a user to browse, and everything that proves
the numbers goes next door.

| repository | type | holds |
| --- | --- | --- |
| `AtomicChat/calib-corpora` | dataset | the calibration material, the recipes, the built corpora, the measurement corpora, and the pipeline that produces them |
| `AtomicChat/<Model>-GGUF` | model | the BF16 file, every quantized rung, the mmproj file for a vision model, the model card |
| `AtomicChat/<Model>-GGUF-metrics` | dataset | every log, the reference logits (`.kld`), the imatrix and its shards, `results.json` |
| `AtomicChat/<Model>-MLX-<label>` | model | one MLX checkpoint. An MLX quant is a directory, not a file, so each rung needs its own repository |
| `AtomicChat/<Model>-MLX-metrics` | dataset | the same idea as the GGUF metrics repository, for the MLX ladder |

The naming is generated, not typed. `use_model qwen3.8-27b Qwen/Qwen3.8-27B`
derives `AtomicChat/Qwen3.8-27B-GGUF` and `AtomicChat/Qwen3.8-27B-GGUF-metrics`
from the upstream repository id, and refuses an id that does not exist on the
hub. `find_repo <part of the name>` finds the real id when you are not sure.

### What goes in the metrics repository, and why

- `logs/` every log the run produced. `AUTOPUSH=1` sends each one the moment it
  is written, so a box that dies mid-run does not take its results with it.
- `kld/` the reference logits. This is a large file, fp16 logits over the whole
  vocabulary for every scored token. It gets published because people asked for
  it: with it, anyone can measure their own quant against exactly the reference
  we used. Over 45 GB it is split into numbered parts with a manifest.
- `imatrix/` the merged importance matrix and every shard it was merged from.
- `results.json` the parsed table. One file per box (`results-<hostname>.json`),
  stitched together by `merge_results`.

## calib-corpora

Read its own `README.md`, `RECIPE.md` and `METHOD.md` first, they are the real
documentation. The short version of the structure:

```
pool/       model agnostic raw material. Dialogue stored as structured
            messages, never as rendered markup
recipes/    one YAML per target model: shares, budgets, tokenizer, seed
builds/     the output of tools/build.py under one recipe, with a manifest
eval/       measurement corpora, disjoint from every build by construction
tools/      the pipeline that turns pool + recipe into a build
archive/    imported material, kept so its origin stays visible
```

Two things and only two are bound to a target model: the chat markup and the
vocabulary. So those are the only two things a build re-derives. Everything else
is shared.

> [!WARNING]
> A build made for one model is close to useless for another. Its special tokens
> do not exist in the other tokenizer, and its vocabulary sweep covers rows the
> other model does not have. `builds/<name>/calib_train.txt` is not a general
> corpus.

Three measurement corpora live under `eval/`, and `neutral` is the default:

| corpus | what it is |
| --- | --- |
| `eval/neutral/eval_neutral.txt` | 30 languages, no code. The default for every published number |
| `eval/code/eval_code_full.txt` | source code |
| `eval/agentic/eval_agentic.txt` | conversations in a model's own markup |

## Directories on a rented Linux box

Everything sits at the filesystem root. Simple absolute paths, no variable
substitution to get wrong, and no `/workspace` that a different provider names
differently.

| path | what |
| --- | --- |
| `/foundry.sh` | the toolbox itself |
| `/state.sh` | the current selection, written by `save_state`, read at the bottom of `foundry.sh` so every tmux pane agrees |
| `/src` | original weights from the hub |
| `/gguf` | GGUF files. `/gguf/external/<publisher>/` for other people's builds |
| `/kld` | the reference logits |
| `/imatrix` | shards and the merged file |
| `/eval` | the corpora |
| `/logs` | everything |
| `/recipes` | generated recipes before they go into calib-corpora |
| `/calib-corpora` | a clone of the dataset repository, when a corpus is being built |
| `/mlx` | MLX checkpoints, one directory per rung |
| `/hf` | `HF_HOME`, the hub cache. Can hold a second copy of anything pulled without `--local-dir` |

On macOS nothing can be created at `/`, so `FROOT` redirects the same layout
under `$HOME/foundry`. On Linux `FROOT` is empty and the paths above are used
verbatim.

## The flow, GGUF

```
upstream weights (hub)
        |
        |  get_upstream
        v
      /src
        |
        |  make_bf16          (convert_hf_to_gguf.py --no-nextn)
        v
  BF16 GGUF  ------------------------------.
        |                                   |
        |  im_shard on N boxes              |  base
        |  im_merge_all                     |
        v                                   v
   imatrix.gguf                     reference logits (.kld)
        |                                   |
        |  quantize / ladder                |
        v                                   |
   quantized rungs  ---------- kld ---------'
        |                                   |
        |  push_quants                      |  push_logs
        v                                   v
  <Model>-GGUF                     <Model>-GGUF-metrics
```

The corpus feeds in from the side: `get_calib <recipe-name>` pulls
`builds/<name>/calib_train.txt` for the imatrix, and `get_eval` pulls
`eval/neutral/eval_neutral.txt` for the measurements. Those two files are
different splits of one pipeline and never share a document.

## The flow, MLX

Different tools, same shape. `foundry-mlx3.sh` is self contained and does not
build llama.cpp at all, which saves fifteen minutes on a box that only does MLX.

```
upstream weights -> mlx3_quant       -> a plain rung
                 -> mlx3_build       -> a rung with a per tensor bit layout
                    (both feed through the same clipping search)
                                          |
                    mlx3_clip  ------------'   better rounding, identical size
                                          |
                    mlx3_dwq   ------------'   distillation against an 8 bit teacher
                                          |
                    mlx3_kld   ------------'   against the cached bf16 reference
```

One thing that has no GGUF equivalent: `mlx3_cache`. In llama.cpp the reference
logits go to disk because the two runs are separate processes. In MLX both
models are objects in one process, so the reference is computed once, kept as a
memory mapped file, and every later measurement on that box reuses it. Without
it, measuring nineteen builds means nineteen identical forward passes of a 51 GB
model thrown away.

## Other pieces that are not in this repository

- **`atomic-forge`** (GitHub org `AtomicChat`): the orchestration layer with
  self hosted runners on a DigitalOcean VPS and `workflow_dispatch` buttons.
  Older than these scripts and overlapping with them. Confirm its current status
  before relying on it.
- **The llama.cpp fork**: releases are sometimes built from an upstream pull
  request branch rather than master, when support for a new architecture has not
  landed yet.
- **The provider benchmark tracker**: measured quantize and imatrix throughput
  per chip, used to pick machines. See `renting-boxes.md`.
