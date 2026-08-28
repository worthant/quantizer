# Sharding the importance matrix

The trick that turns a six hour imatrix into a thirty minute one. It is the
single most valuable thing in this pipeline, because time to publication is what
the whole release strategy rests on.

## Why it is exact, not an approximation

The importance matrix is a sum of squared activations per tensor column. A sum
splits.

Every node reads the **same corpus file** and takes its own slice by chunk range
with `--from-chunk` and `--chunks`. Nobody cuts the text file. That matters: if
you split the text, each piece gets chunked independently and the chunk
boundaries land in different places, so the union is not what a single node
would have produced. Splitting by chunk range means the union is bit for bit the
chunking of one long run.

Merging with `llama-imatrix --in-file a,b,c` sums the squared activations, which
gives the same answer as one long run. This is not "close enough", it is the
same number.

## The sequence

**On every box**, once each:

```bash
export HF_TOKEN=hf_...
curl -sL https://raw.githubusercontent.com/<ORG>/<REPO>/main/foundry.sh -o /foundry.sh
source /foundry.sh
persist_shell
save_token
cuda_arch ; cuda_check ; setup ; build <arch> ; gpu_test
use_model <name> <upstream-repo>
get_calib <recipe-name>          # the same bytes everywhere
get_bf16                         # or make_bf16 on the box that has /src
pick_model                       # choose the same file on every box
```

> [!IMPORTANT]
> `get_calib` on every box, not a copy of the file passed around. The whole
> scheme rests on every node reading identical bytes. `IM_MODEL` has to be the
> same file too, and BF16 rather than Q8: Q8 is a memory compromise, not a better
> choice.

**On one box**, work out the split:

```bash
im_size                          # how many chunks the corpus is
im_plan 6                        # prints the exact command for each of six nodes
```

`im_plan` prints one line per node. Paste each into its own box:

```bash
im_shard 0 6
im_shard 1 6
...
im_shard 5 6
```

Each shard uploads itself to the metrics repository when it finishes, so the
merging box can pick it up without anyone copying files.

**Anywhere**, once every node is done:

```bash
im_status                        # what is here, what is in the repo, what ranges were covered
im_merge_all                     # or im_merge 6
```

`im_merge_all` pulls every shard from the metrics repository, merges, prints the
statistics, and uploads the merged file.

## The token count matters

`im_size` estimates tokens at four bytes each, which is usually low. `build.py`
printed the real number when the corpus was built. Set it, or the shards come
out uneven and the last node runs long:

```bash
IM_TOKENS_EXACT=<number>
save_state
im_size
```

## When boxes are not equal, and they never are

`im_shard` splits evenly, which assumes every machine runs at the same speed.
Rented machines differ by half again on the same cards, and the slowest one ends
up on the critical path with everyone else idle.

`im_range` takes an explicit range, so a long shard can be cut up and handed to
whoever is free:

```bash
im_range 7275 700 5a             # chunks 7275 to 7974, labelled 5a
im_range 7975 0   5b             # 7975 to the end of the corpus
```

Then the killed shard is excluded from the merge. `llama-imatrix` writes
snapshots as it goes, so a shard that was killed still left a partial file
behind, and merging that alongside the ranges that replaced it counts those
chunks twice:

```bash
IM_SKIP="shard-5-of-6" im_merge_all
```

If a shard finished in a pane that had an older copy of the file loaded and
never got to its own upload line:

```bash
push_shards                      # uploads everything in /imatrix not already in the repo
```

## Memory, before the run rather than after

`im_shard` computes this and refuses to start when it will not fit. The logits
tensor is `batch x vocab x 4` bytes and lives in VRAM alongside the weights, so a
batch tuned for a small vocabulary will OOM on a large one:

```
weights + (vocab * IM_BATCH * 4) + headroom  <=  VRAM the process can see
```

Two shards on one box each get their own pair of cards through
`CUDA_VISIBLE_DEVICES`, so `im_shard` counts only the visible ones. Counting all
of them would say everything fits when it does not.

If it does not fit, lower the batch:

```bash
IM_BATCH=2048 ; im_shard 0 6
```

## Settings and why they are what they are

| setting | value | why |
| --- | --- | --- |
| `--parse-special` | always on | without it every `<\|im_start\|>`, `<think>` and `<tool_call>` in the corpus is tokenized as literal punctuation, and the tokens the model actually emits appear zero times. For these recipes that is about 40 percent of the corpus calibrating nothing, and it fails silently |
| `IM_BATCH` | 8192 | measured: 1h22m instead of 2h12m for 9.3k chunks on 2x RTX 5090 against the default 512. It plateaus there, 16384 saves another minute |
| `IM_CTX` | 512 | the imatrix context, not the measurement context |
| `IM_NO_PPL` | 0 | `--no-ppl` is a few percent faster and removes the only sign of progress, leaving a forty minute run completely silent. Visibility is worth more |
| `--output-frequency` | 20 | how often it reports stored data. Also a heartbeat |

## The long context pass

`calib_longctx.txt` is a separate file and a separate run, at a large context,
with unbroken 16k to 32k token documents. It exercises the long range path that
a 512 or 4096 token window never reaches. Run it as a second pass and merge the
two files the same way, since merging is a sum either way.

## How much corpus is enough

There is a definite answer and it is cheap to get. Build at N and 2N chunks and
take the per tensor cosine between them; you are done when no tensor is still
moving. `calib-corpora/tools/converge.py` does this against `--save-frequency`
checkpoints, so one run produces the whole curve for free.

Measured on a 30B hybrid: about 8,500 chunks, roughly 4.3M tokens, past which
zero of 186 tensors change measurably. That is why 5M tokens is the standing
figure in the recipes.

Two things that look like quality metrics and are not:

- **`% Active`** from `--show-statistics` is the fraction of columns whose mean
  squared activation exceeds a hard threshold of `1e-5`. It measures activation
  magnitude, not calibration coverage. Early layers score low because their
  activations are genuinely small, and no corpus raises that.
- **The per layer sum of squared activations curve** is a model property, not a
  corpus property. Two corpora with cosine 0.855 between their imatrices produce
  per layer curves within 8 percent of each other.

What does mean something: the `counts` array, which says how many times each
expert was routed to. `counts[i] == 0` is a genuine hole.

## MTP blocks collect nothing

A multi token prediction head is never executed in a normal forward pass, so the
imatrix has no statistics for it at any corpus size. Assign it a low bit type and
`llama-quantize` refuses outright, halfway through, leaving a truncated file
behind. `quantize` finds the block with `find_mtp_block` and pins it to
`MTP_TYPE` (default `q5_k`) unless a rule names that block specifically. It is
one block out of many, so keeping it high costs almost nothing.
