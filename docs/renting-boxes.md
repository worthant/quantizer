# Renting boxes

Everything in this pipeline runs on rented machines that exist for a few hours
and are then destroyed. Picking them badly is where the money goes.

> [!NOTE]
> Every number here comes from our own benchmarks, not from vendor material.
> Re-measure before trusting any of it: rented hardware drifts, and the same
> offer name covers very different machines.

## The three workloads want different machines

| workload | bound by | wants |
| --- | --- | --- |
| `llama-imatrix` | memory bandwidth, GPU | cards with enough VRAM to hold weights plus the logits tensor |
| `llama-quantize` | thread count | many physical cores. Measured: threads roughly 2x, AVX-512 roughly 1.2x |
| `llama-perplexity` (KLD) | GPU | enough VRAM for the file, plus all cards for the BF16 reference |
| MLX distillation | GPU memory and tensor cores | 94 GB minimum for a 27B, 140 GB comfortably |
| transit of very large files | network | this is the one that quietly costs the most |

The important consequence: **quantizing is a CPU job and calibrating is a GPU
job**, and putting them on the same machine means paying GPU prices for CPU work
or vice versa. On very large models they end up on different providers.

## Reading an offer

Things that are worth filtering on, in order:

1. **Physical cores, not `nproc`.** Set thread counts to the physical core
   count. `nproc` returns hyperthreads and gives 25 to 35 percent less
   throughput on quantize.
2. **Chip generation over core count.** More cores is not faster: a 128 core
   Bergamo measured slower than a 96 core Genoa on the same job. Zen4 and Zen5
   EPYC are the ones to look for.
3. **Driver version.** The `mlx[cuda13]` wheel needs 580 or newer. A Blackwell
   card on an older host will not start at all.
4. **Network.** Traffic is billed and a large transfer takes an hour that is
   also billed as rent. Filter by advertised inet speed and measure it.
5. **Disk.** `disk_plan` and `mlx3_disk` print what a job needs. Running out
   halfway through a 210 GB download is a wasted hour.

**Boxes on the same provider vary wildly.** Measure the spread with three runs
and take the median. One machine in our benchmark set was six times slower than
its identical twin on the same core count; that was a faulty box, not a property
of the chip family.

## Standard environment

CUDA 13.0, the `nvidia/cuda` devel image on Ubuntu 24.04. Older CUDA lines are
not used. Put the CUDA version in the caption of any published benchmark chart.

`cuda_check` handles one thing that bites on rented containers: forward
compatibility libraries under `/usr/local/cuda/compat`. When the host driver is
newer than the libraries target, those libraries can only break CUDA
initialization, so `fix_cuda_compat` removes them and the host driver wins.
Symptom without this: `llama-cli --list-devices` prints an empty or short list.

`cuda_arch` reads compute capability off the card and prints the exact `build`
line. It is not guessable from the card name: H100 and H200 are both 90, RTX
5090 is 120, B200 is 100.

## Speculative renting

For a fan out, rent several offers at once and use whichever comes up first.
Instances that fail to start or come up degraded are common enough that waiting
serially on one costs more than the extra minutes of the others.

The counterpart to that is a reaper. Orphaned instances that nobody destroyed
are a real and recurring cost. If the orchestration layer has a watchdog, keep
it running; if it does not, put a calendar reminder on it.

## The hardware for published numbers

Deliberately not the biggest machine available. Benchmarks are published on a
setup close to what readers have, on the theory that the more powerful the
stand, the less the community relates the numbers to itself. Two RTX 5090
rather than four.

The one exception is the BF16 reference, which uses every card in the box
because BF16 does not fit on fewer. That is stated in the model card.

`use_gpus 0,1` holds the quant measurements to the published setup while `base`
ignores it.

## Connecting

```bash
vastai ssh-url <instance-id>
TERM=xterm ssh root@<ip> -p <port>
```

`TERM=xterm` avoids a terminfo mismatch inside the container. The instance
template drops you straight into tmux, so tmux does not need installing.

One pane per process, everything in the foreground. `ps` and the tmux tab list
are the process monitor. Nothing in either script backgrounds anything, and that
is on purpose: a run you cannot see is a run you find out about an hour later.

First two commands on any box, before the real work:

```bash
apt-get install -y vim
fix_tmux
```

`fix_tmux` turns on mouse scrolling and 200k lines of history. Searching inside
the buffer is `ctrl-b [` then `/` and a pattern. Without it, reading back a
forty minute log is painful enough that people stop doing it.

## Cost, honestly

Spending is spiky and hard to predict, because it is driven by when models are
released rather than by a plan. A single large model can run into four figures
across conversion, calibration, the quant ladder and the measurements. The
levers that actually reduce it:

- publish the reference logits and the imatrix, so a box that dies does not take
  them with it and the next box pulls instead of recomputing
- `plan` and `audit` before starting, so nothing is recomputed that already
  exists on the hub
- shard the imatrix, so the total is the same GPU hours but on cheap parallel
  boxes for a short time rather than one expensive box for six hours
- `mlx3_try` at six chunks for sweeps, full 24 chunks only for anything that
  goes in a table
