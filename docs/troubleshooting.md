# Troubleshooting

Every entry here cost hours the first time. They are grouped by where they hit,
and most of them are already handled in the scripts; this document says what the
handling is for, so nobody removes it as noise.

## The environment

**`HF_TOKEN` looks absent even though the instance template set it.** An ssh
login shell does not inherit the container environment. PID 1 has it and you do
not. `load_env` (and `mlx3_env`) read it back out of `/proc/1/environ`, which is
the only way to see what the template actually set. If it is genuinely not
anywhere, export it and run `save_token` or `mlx3_persist`.

**A new tmux pane knows nothing.** Bash functions live in one shell process.
`save_state` writes the current selection to `/state.sh` and the bottom of
`foundry.sh` reads it back, so every pane agrees. Run `persist_shell` once per
box so a new pane sources the file by itself.

**`reload` says the version is the same after you just pushed.** Both http paths
to GitHub are cached: `raw.githubusercontent` behind a CDN and the contents API
behind its own layer. Either can serve a file that is minutes old. `reload` uses
`git` instead, which does not go through any of that, and prints the commit it
is actually on. If that commit is not what you pushed, the push did not land on
the default branch.

**CUDA sees no devices.** Forward compatibility libraries under
`/usr/local/cuda/compat` targeting an older driver than the host has. Run
`cuda_check`, which detects it and offers `fix_cuda_compat`.

**A function the help text mentions does not exist.** Editing a large shell file
by cutting between two function names quietly eats whatever sat between them.
`selfcheck` checks every advertised function is present.

## Conversion

**NaN in `token_embd` after conversion.** Add `--no-lazy` to
`convert_hf_to_gguf.py`. Lazy evaluation has broken the embedding tensor on at
least one architecture, and the symptom appears far away: the imatrix run
reports NaN in a layer 0 tensor, which is a consequence, not the cause. Confirm
with `--check-tensors` on the converted file and by checking the same tensor in
the original weights.

**The converter complains about RoPE.** Pin `transformers` below 5. The fifth
branch reworked RoPE and `standardize_rope_params` fails with
`AttributeError: 'PreTrainedConfig' object has no attribute
'max_position_embeddings'` on some configs.

**The file converts fine and then refuses to load.** The config promised an MTP
head the weights do not carry, so the converter wrote a block count it cannot
fill. `check_blocks` compares the declared block count against the tensors
actually present. Reconvert with `--no-nextn`.

**The config claims an MTP head that does not exist.** The config is not
evidence. `make_bf16_mtp` checks `model.safetensors.index.json` for actual
`nextn` or `mtp` tensors and refuses when there are none.

## The corpus

**A pool file fails to parse deep inside `build.py`.** It came down as a
git-lfs pointer rather than data. `check_pool` finds it by name, `fix_pool`
pulls the real file straight from the hub, which is simpler than getting
git-lfs working on a fresh container.

**`llama-tokenize` aborts with `invalid codepoint` on plain ASCII input.** A
UTF-8 conformance test literal such as `\xF4\x91\x92\x93` decodes past
`U+10FFFF`, and `unicode_cpt_to_utf8` throws instead of substituting `U+FFFD`.
It would kill an imatrix run partway through. `calib-corpora/tools/screen.py`
finds and quarantines such documents by divide and conquer.

**The vocabulary sweep overwrote another model's file.** `build_corpus`
compares `git status` before and after the sweep and stops if a touched file is
not named after this recipe. Restore it with `git checkout` and give
`vocab_sweep.py` an output name.

**The corpus renders with no tool markup at all.** Templates disagree about the
shape of a tool call. `corpus_check` tries the common shapes and reports which
one the model accepts, because the pool records have to match it or the agentic
slice renders with nothing.

**Two builds a week apart differ.** The chat template calls `strftime_now`, so
every rendered conversation carries the day it was built. Set `chat.pin_date` in
the recipe; `build_corpus` greps it into `FOUNDRY_PIN_DATE` and `auto_fmt.py`
refuses to render at all when a date is needed and none is pinned.

## Quantizing

**`llama-quantize` aborts partway through on the MTP tensors.** The MTP block
collects no imatrix data at any corpus size, so a low bit type is refused.
`quantize` pins it automatically. The check used to be "does any argument
mention `blk`", which every edge weighted rung does, so the pin was silently
skipped.

**The run aborts after reading the whole model, complaining about a type.** The
list under "allowed quantization types" is for the positional ftype argument.
`--tensor-type` takes a ggml type, and the two are different sets. `IQ2_M`,
`IQ3_M`, `IQ3_XS`, `Q2_K_S`, `Q3_K_*`, `Q4_K_*`, `Q5_K_*` are mixes, recipes for
which tensor gets which type, and cannot be assigned to a tensor. `check_types`
catches this before the model is read.

**A quant is bigger and worse than the size prediction said.** A k or i quant
needs rows divisible by 256. When they are not, llama.cpp silently swaps in a
block-32 type and keeps the name you asked for. `bits` prints divisibility per
tensor group and warns.

**The quant does not appear in its own row on the Hugging Face page.** That
table is built from `general.file_type`, not from the filename. Set it at
quantize time: `--override-kv general.file_type=int:15`. `check_meta` shows what
every file on the box actually declares.

**Moving experts from MXFP4 to Q4_K made the file larger and worse.** Q4_K is a
uniform grid, MXFP4 is logarithmic. For protecting a layer it is better to leave
MXFP4 alone than to bump it.

## Measuring

**`results` includes builds from a different model.** Stale logs in `/logs` from
whoever used this box before. `status` warns about it and `clean_run` removes
them. Uploading them would put another model's numbers in this model's metrics
repository.

**A row has no size and cannot be plotted.** The file lives on another box. Run
`results` there too, then `merge_results`.

**Numbers from another publisher do not match their published figures.** They
should not. Their figures were taken against their own reference and harness.
Download their files and re-measure against ours with `get_external` and
`kld_ext`. Their files are usable, their numbers are not.

**Nothing appears in the log for many minutes.** The pipe into `tee` switches
libc to 4 KB block buffering. Every long running command in both scripts is
wrapped in `stdbuf -oL -eL` for this reason.

## MLX specifics

**A distillation step dies with a CUDA graph error.** Four different messages,
one cause: `cudaGraphAddDependencies ... invalid argument`,
`cudaGraphInstantiate ... out of memory`, `cudaGraphAddKernelNode ... illegal
memory access`, and `Cache thrashing is happening`. A compiled graph remembers
the buffer addresses it was built with, the allocator reuses those buffers
between steps, and replaying the graph then reads memory belonging to something
else. Tuning `MLX_CUDA_GRAPH_CACHE_SIZE` only chooses which way it breaks.
`MLX_USE_CUDA_GRAPHS=0` removes the whole class at a cost of about two seconds
per step.

**Memory climbs by about a gigabyte per batch and a bigger card just dies
later.** With `--batch-size 1` every record goes at its own length,
`iterate_batches` pads a batch to its longest member, and one member means the
record itself. Six hundred records is up to six hundred distinct tensor shapes,
and the allocator cannot reuse a buffer sized for 1700 tokens to hold 1900.
Measured: a 94 GB card died at batch 50 and a 143 GB card at batch 442 of the
same 600. The fix is a 512 token window, which works on both causes at once:
smaller buffers, and almost every record truncating to exactly 512 so the long
tail of shapes collapses into one.

**`--data-path` at a local directory silently produces an empty validation
set.** In mlx-lm 0.31.3 a local path routes through `load_custom_hf_dataset`
rather than `load_local_dataset`, and there validation is sliced out of the
train split, so `valid.jsonl` is never read. The run dies with "Dataset must
have at least batch_size=1 examples but only has 0". Record length has nothing
to do with it. Until it is fixed upstream, `MLX3_DATA=default` uses mlx-lm's own
set, which also makes the result directly comparable to published runs.

**Records get dropped rather than truncated.** mlx-lm drops a sequence longer
than `max_seq_length`. If every record is longer, everything is dropped.
`mlx3_data` measures the real characters per token on this corpus and sizes
records at 90 percent of the window, then tokenizes twenty of them and prints
what actually came out.

**A finished checkpoint gets deleted by the cleanup branch.** The MLX CUDA
backend throws from a destructor at interpreter shutdown
(`Destroy(handle_) failed: driver shutting down`) after all the work is done,
and the shell then sees a non zero exit code. Every python file written by
`mlx3_write_py` ends with `os._exit(0)` to skip the destructors, and the shell
functions check for the output file rather than the exit code. That cost one
finished 13.68 GB checkpoint.

**`mlx3_reattach` reports "replaced 0" and writes a file anyway.** An earlier
version stripped the `language_model.` prefix before looking names up, which
matched nothing, and the output was the untouched base with a new quantization
block. It looked like a result and was not. The current version refuses to write
when it replaced nothing.

## Transit

**A large file transfer between boxes is slow and billed twice.** Traffic costs
money and the hour it takes is also rent. `send_base` does a direct rsync
between boxes rather than a round trip through the hub. On a shared provider,
throughput past about 5 Gbit/s has not been reachable in practice.

**Jumbo frames do not help.** On one bare metal pair with a 100G private VLAN,
`ping -M do` showed the ceiling was exactly 1500, and setting MTU 9000 made TCP
stall entirely (iperf3 dropped to kilobits). Check with `ping -M do` before
changing MTU.

**A monolithic GGUF is truncated with "data is not within the file bounds".**
The disk filled during conversion. The shards were intact and their checksums
matched across machines; only the merged file was bad.

**Quantizing got noticeably faster after killing other processes.** Parallel
work on the same node, including a KLD run, competes for the same cores and
disk. One heavy job per box.
