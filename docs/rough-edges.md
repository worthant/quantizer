# Rough edges

Known bugs, things that are hardcoded to one model, and work that was started
and not finished. Written down so nobody has to rediscover them.

## Bugs worth fixing early

**`dryrun` references an undefined variable.** It builds its command line with
`$IMFLAG`, which is set nowhere in the file, so the dry run never sees the
importance matrix. It still reports a size, which is what it is for, but the
variable should either be defined or removed.

**Settings below the settings block override the settings block.** `foundry.sh`
defines its configuration near the top with `${VAR:-default}`, so the
environment wins. Further down, several of the same names are assigned plainly:

```
IM_CTX=512        IM_BATCH=8192     IM_OFREQ=20
IM_CORPUS=/eval/calib_train.txt     IM_MODEL=""
AUTOPUSH=1        GPUS=all
```

Those assignments run after the block at the top, so `export IM_BATCH=2048`
before sourcing has no effect. Prefixing the command still works
(`IM_BATCH=2048 im_shard 0 6`), and `/state.sh` is read at the very bottom so
saved state survives. Deleting the duplicates is the fix.

**Duplicate function definitions.** `set_ctx` and `wait_calib` are each defined
twice, and `save_state` is redefined in the addendum. The last definition wins,
which is intentional for `save_state` and accidental for the other two.

**`selfcheck` is out of date.** Its list predates a lot of the file. Missing
from it, among others: `make_mmproj`, `push_mmproj`, `check_blocks`,
`check_meta`, `check_template`, `newbox`, `test_vision`, `test_vision_all`,
`test_chat`, `test_tools`, `test_longctx`, `rename_quant`, `fix_tmux`,
`lastlog`, `dryrun`, `check_types`, `find_mtp_block`, `ladder_part`,
`merge_results`, `scan_secrets`, `corpus_check`, `push_demo_image`,
`make_bf16_mtp`, `im_wait_merge`, `push_quants`, `catch_up`.

**`mlx3_targets` can `mkdir -p ""`.** `MLX3_TARGETS` defaults to empty.
`mlx3_dwq` has a fallback that names a directory per window; `mlx3_targets` does
not and should get the same one.

**`MLX3_DATA` is a path that defaults to a word.** The default is the string
`default`, which means "use mlx-lm's own dataset". `mlx3_data` then runs
`mkdir -p default` and creates a directory of that name in the working
directory. Harmless, confusing.

**`mlx_quant_text` is a stub** that only prints a message and returns 1.

**`quant_files` sets `DEPTH` without `local`,** so it leaks into the shell.

## Hardcoded to one model

These need editing for every new model, and it is not obvious from `status` that
they do:

| place | hardcoded to |
| --- | --- |
| `write_ladder` | the Qwen3.8-27B ladder, including its block ranges and the measurement table in the comment |
| `del_old_ad` | a literal list of Qwen3.8-27B filenames |
| `mac_get` | `AtomicChat/Qwen3.8-27B-GGUF` and a fixed list of MLX rungs |
| `OUR_MLX`, `EXTERNAL_MLX`, `MLX3_SAME` | Qwen3.8-27B repositories |
| `mlx3_box`, `mlx3_plan`, `mlx3_help` | Qwen3.8-27B paths and its three target numbers |
| `mlx3_gen` | the model's dimensions (`H`, `I`, `V`, `L`, `F`, `N`) as literals |
| `ppl_compare` | the published GGUF reference perplexity, as a literal |

The block ranges in the ladder deserve a note of their own. They encode "the
first four and last twelve blocks of a 64 block model", written as a regular
expression over block numbers. On a model with a different depth those ranges
are wrong in a way that produces a working file with a worse layout, which is
the hardest kind of wrong to notice.

## Three generations of the MLX measurement

There are three KLD implementations across the two files, and only one of them
should be used:

| where | status |
| --- | --- |
| `mlx_kld` in `foundry.sh` | **obsolete.** Scores from position zero, which counts tokens the model predicted almost blind. Every number it produced is pessimistic |
| `mlx_kld2` in `foundry.sh` | follows llama.cpp conventions, recomputes the reference per build |
| `mlx3_kld` in `foundry-mlx3.sh` | **current.** Same conventions plus the cached reference |

Anything measured with the first one has to be redone before it goes in a table.
The MLX block in `foundry.sh` is superseded by `foundry-mlx3.sh` in general, and
keeping both invites someone to use the wrong one.

## Naming

The ladder labels rungs `AD-Q4_K`, `AD-IQ3_S` and so on, borrowed from
llama.cpp's ftype names. The naming standard we argued for publicly is the
opposite: name a quant by its actual measured bits per weight, because a label
that does not match the file's real bit count is misleading. The ladder was not
renamed to match. Decide one way and be consistent, because `rename_quant`
exists precisely because this has already been changed once and the metrics logs
had to be moved with the files.

## Unfinished directions

**Logs and metrics repository structure.** Logs parse fragilely, the structure
of the metrics repositories is not obvious to a reader, and some logs are lost
when a box dies before its upload. The stated next goal for these scripts was
simplicity, transparency and reliability, with orchestration afterwards.

**Automatic bit layout.** The per tensor layouts are hand written regular
expressions informed by imatrix statistics and by measuring alternatives against
each other. The direction worth pursuing is an algorithm that derives the layout
for any architecture instead. `mlx3_gen` is a first step on the MLX side: it
fixes the shape and moves only the level, so a sweep compares sizes rather than
a hundred unrelated ideas.

**Real benchmarks.** Every published number is KL divergence and top-1 agreement
on held out text. That measures how close a quant stays to the original, not
whether it is still useful. Running the ladder on actual agentic benchmarks was
planned and never done.

**One bit MLX.** `mlx-vlm` can load a one bit affine checkpoint and nothing in
the stack produces one. `mlx3_1bit_status` (in `foundry.sh` as
`mlx_1bit_status`) checks where that stands on the installed versions. The
contract is written down: packed uint32 weights, scales and biases, group size
32, 64 or 128, `bits: 1` in the config.

**`auto_fmt.py` exists in two places.** The canonical copy is
`calib-corpora/tools/auto_fmt.py`, selected by `chat.format: auto` in a recipe.
The copy here exists only so `install_auto_fmt` can curl it onto a box that has
cloned calib-corpora. Since that clone already contains the file, the function
could read from the clone and the duplicate could go.

**The `exclude` block in a recipe does nothing.** `build.py` decides what to
skip in its `usable()` function, which hardcodes `render == "dsv4"` and checks
`provenance.excluded_from_builds`, and never opens `recipe["exclude"]`. A recipe
listing three renderers excludes exactly one. Wire it up or delete the block, but
do not leave configuration that looks live and is not.
