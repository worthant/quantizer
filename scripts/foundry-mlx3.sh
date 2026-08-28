#!/bin/bash
# ================================================================== FOUNDRY MLX3
#
# Everything the MLX side needs, in one file. It does not need foundry.sh and
# it does not need foundry-mlx2.sh. On a bare rented box:
#
#   export HF_TOKEN=hf_...
#   curl -sL https://raw.githubusercontent.com/AtomicBot-ai/atomic-quantizer/main/foundry-mlx3.sh -o /mlx3.sh
#   source /mlx3.sh
#   mlx3_setup
#   mlx3_persist
#   mlx3_box b          the exact command list for this box, paste one at a time
#
# raw.githubusercontent sits behind a CDN and can hand back a file that is
# minutes old. If a fresh push is not showing up, clone instead:
#   git clone https://github.com/AtomicBot-ai/atomic-quantizer /quantizer
#   source /quantizer/foundry-mlx3.sh
#
# WHAT THIS FILE IS FOR
#
# Three numbers have to be beaten, all measured against one bf16 reference at
# context 4096 on eval_neutral:
#
#   13.70 GB   0.154161   maglun Mixed-3.80bpw
#   16.05 GB   0.053775   WaveCut 4bit-DWQ
#   17.57 GB   0.034350   maglun Mixed-4.95bpw
#
# Two methods are used and nothing else. Both are explained where they are
# defined. mlx3_plan prints the arithmetic behind every expected number.

MLX3_VERSION=2026-08-25.04

MLX3_UP=${MLX3_UP:-Qwen/Qwen3.8-27B}
MLX3_ORG=${MLX3_ORG:-AtomicChat}
MLX3_STEM=${MLX3_STEM:-Qwen3.8-27B}

MLX3_ROOT=${MLX3_ROOT:-/mlx}
MLX3_SRC=${MLX3_SRC:-/src}
MLX3_LOGS=${MLX3_LOGS:-/logs}
MLX3_EVALDIR=${MLX3_EVALDIR:-/eval}
MLX3_EVAL=${MLX3_EVAL:-/eval/neutral.txt}
MLX3_CALIB=${MLX3_CALIB:-/eval/calib_train.txt}
MLX3_METRICS=${MLX3_METRICS:-AtomicChat/Qwen3.8-27B-MLX-metrics}

MLX3_REF=${MLX3_REF:-/mlx/Qwen3.8-27B-MLX-bf16}
MLX3_TEACHER=${MLX3_TEACHER:-/mlx/Qwen3.8-27B-MLX-8bit}
MLX3_CACHE=${MLX3_CACHE:-/mlx/refcache}

# Measurement protocol. Same as the published GGUF table so the two can sit
# side by side: chunks of 4096, only the second half of each chunk scored,
# 24 chunks, KLD summed over the vocabulary with the reference first.
MLX3_CTX=${MLX3_CTX:-4096}
MLX3_FIRST=${MLX3_FIRST:-}
MLX3_STEP=${MLX3_STEP:-256}
MLX3_CHUNKS=${MLX3_CHUNKS:-24}
MLX3_TAIL=${MLX3_TAIL:-1}

# Clipping search grid. 1.000 is what every existing build did, so the search
# can only tie it or beat it. The fine steps near the top matter above 4 bits,
# where the best fraction sits around 0.98.
MLX3_ALPHAS=${MLX3_ALPHAS:-"1.000 0.995 0.985 0.970 0.955 0.940 0.920 0.900 0.880 0.860 0.840 0.810 0.780 0.750 0.700"}

# Distillation defaults, taken from published runs rather than from what fits
# in the least memory. See the comment above mlx3_dwq.
MLX3_DATA=${MLX3_DATA:-default}     # see the note above mlx3_dwq
MLX3_CHARS=${MLX3_CHARS:-0}   # 0 means measure it, see mlx3_data
MLX3_TRAIN=${MLX3_TRAIN:-600}
MLX3_VALID=${MLX3_VALID:-60}
# 512 rather than 2048, and the reason is not calibration quality.
#
# With --batch-size 1 every record goes at its own length and iterate_batches
# pads a batch to its longest member, which is that record. The logits are then
# [1, length, 248320], so a new length means a new buffer size for the MLX
# allocator to cache AND a new graph for the CUDA backend to compile, and
# nothing evicts either. Six hundred records is up to six hundred shapes.
# Measured: target computation grew about 0.8 GB per batch and a 94 GB card
# died at batch 50 while a 143 GB one reached 442.
#
# Most tulu records are longer than 512, so a 512 window truncates nearly all
# of them to exactly 512 and the shapes collapse to one. Memory goes flat.
# The eval scores positions 2048 to 4095, so this is a real mismatch, but DWQ
# tunes per-group scales and those do not depend on position to first order,
# and a run that finishes beats a run that does not.
MLX3_SEQ=${MLX3_SEQ:-512}
MLX3_BATCH=${MLX3_BATCH:-4}     # batch 1 makes one tensor shape per record
MLX3_LR=${MLX3_LR:-3e-7}
MLX3_SAMPLES=${MLX3_SAMPLES:-600}   # 0 drops the flag entirely

# How many compiled CUDA graphs the backend keeps. Both extremes have now been
# hit on this job and the mechanism is the same one.
#
# With --batch-size 1 every record goes at its own length, iterate_batches pads
# a batch to its longest member, and one member means the record itself. Six
# hundred records is up to six hundred distinct tensor shapes, and a distinct
# shape compiles a distinct graph.
#
#   1      the backend evicts on every step and refuses:
#          "Cache thrashing is happening, please set MLX_CUDA_GRAPH_CACHE_SIZE
#          to a larger value than 1", then aborts inside a destructor
#   6144   thousands of graphs are held, each pinning the allocations it
#          captured, and a 143 GB card runs out of memory
#
# 2048 sits between them. Watch nvidia-smi over the first fifty steps rather
# than trusting this number.
export MLX_CUDA_GRAPH_CACHE_SIZE=${MLX_CUDA_GRAPH_CACHE_SIZE:-2048}

# THE ONE THAT ACTUALLY FIXED IT.
#
# Every failure on a distillation step, across a whole day, came out of the
# same file and none came out of the arithmetic:
#
#   cudaGraphAddDependencies ... invalid argument
#   cudaGraphInstantiate     ... out of memory
#   cudaGraphAddKernelNode   ... an illegal memory access was encountered
#   Cache thrashing is happening, set MLX_CUDA_GRAPH_CACHE_SIZE larger than N
#
# A compiled graph remembers the buffer addresses it was built with, the
# allocator reuses those buffers between steps, and replaying the graph then
# reads memory that belongs to something else. Tuning the cache size only
# chose which way it broke: small gave thrashing and an abort, large gave out
# of memory or the illegal access. Turning graph capture off removes the whole
# class. Cost is per-kernel launch overhead, about 25 seconds a step instead
# of 27 at batch 4, which is nothing.
#
# Found with:
#   for f in $(find <site-packages>/mlx -name "*.so*"); do strings "$f"; done \
#     | grep -o "MLX_[A-Z_]*" | sort -u
# MLX_DISABLE_COMPILE is a different thing, operation fusion, and it does not
# help here.
export MLX_USE_CUDA_GRAPHS=${MLX_USE_CUDA_GRAPHS:-0}

# Where mlx_lm.dwq keeps the precomputed teacher logits. Passing --target-dir
# is what makes this job fit: the teacher is loaded, the targets are written as
# top-k values plus indices, and then
#     if has_targets and model is not None: del model
# the teacher leaves memory. After that loss_fn gathers the student's logits
# down to k with take_along_axis instead of running the KL over all 248320
# columns. Both of the two big terms go away: 37 GB instead of 91.
# One directory per window. Targets computed at 2048 are the wrong shape for a
# run at 512, and silently reusing them is worse than recomputing.
MLX3_TARGETS=${MLX3_TARGETS:-}

# Appended verbatim to the mlx_lm.dwq command line. For flags this file does
# not model yet, above all the precomputed targets path.
MLX3_EXTRA=${MLX3_EXTRA:-}
export HF_XET_HIGH_PERFORMANCE=1
export HF_HOME=${HF_HOME:-/hf}

# The four published 4 bit builds that are supposed to be the same file.
MLX3_SAME="
AtomicChat/Qwen3.8-27B-MLX-4bit
mlx-community/Qwen3.8-27B-4bit
lmstudio-community/Qwen3.8-27B-MLX-4bit
lukaskremla/Qwen3.8-27B-4bit-MLX
"


# ================================================================== setup

# An ssh login shell does not inherit the container environment, PID 1 has it.
mlx3_env() {
    local v val
    for v in HF_TOKEN HF_HOME HF_XET_HIGH_PERFORMANCE; do
        if [ -z "${!v}" ] && [ -r /proc/1/environ ]; then
            val=$(tr '\0' '\n' < /proc/1/environ | grep "^$v=" | head -1 | cut -d= -f2-)
            [ -n "$val" ] && export "$v=$val"
        fi
    done
    if [ -n "$HF_TOKEN" ]; then
        echo "HF_TOKEN present (${#HF_TOKEN} chars)"
    else
        echo "no HF_TOKEN anywhere. export it, then run mlx3_persist"
        return 1
    fi
}
mlx3_env > /dev/null 2>&1

mlx3_persist() {
    grep -q "source /mlx3.sh" ~/.bashrc 2>/dev/null || echo "source /mlx3.sh" >> ~/.bashrc
    if [ -n "$HF_TOKEN" ]; then
        grep -q "export HF_TOKEN=" ~/.bashrc 2>/dev/null \
            && sed -i "s|export HF_TOKEN=.*|export HF_TOKEN=$HF_TOKEN|" ~/.bashrc \
            || echo "export HF_TOKEN=$HF_TOKEN" >> ~/.bashrc
    fi
    echo "every new tmux pane now loads this file and has the token."
    echo "the box is disposable, the token is not. Revoke it when you are done."
}

mlx3_setup() {
    echo "=============== 1. card and disk ==============="
    nvidia-smi --query-gpu=name,memory.total,compute_cap,driver_version --format=csv
    df -h / | tail -1
    echo
    echo "llama.cpp is NOT built here. This box does MLX only and building it"
    echo "costs fifteen minutes for nothing."

    echo
    echo "=============== 2. packages ==============="
    mkdir -p $MLX3_ROOT $MLX3_LOGS $MLX3_EVALDIR $MLX3_SRC $HF_HOME
    apt-get update -qq 2>/dev/null
    apt-get install -y -qq python3-pip curl git 2>/dev/null

    # numpy 2 goes in FIRST. mlx-lm's dataset loader pulls scipy, scipy wants
    # numpy 2, and the box ships 1.26. Installed later, the import dies deep
    # inside with an unhelpful message.
    pip install --break-system-packages -q -U "numpy>=2.0,<2.8" || return 1
    pip install --break-system-packages -q -U "huggingface_hub[hf_xet]" || return 1
    pip install --break-system-packages -q -U "mlx[cuda13]" || return 1
    pip install --break-system-packages -q -U "mlx-lm[train]" || return 1
    pip install --break-system-packages -q -U mlx-vlm || return 1

    echo
    echo "=============== 3. does any of it work ==============="
    mlx3_check
    mlx3_write_py > /dev/null

    echo
    echo "=============== 4. next ==============="
    echo "  mlx3_persist        so a new tmux pane is not amnesiac"
    echo "  mlx3_box a|b|c      the exact command list for this box"
    echo "  mlx3_plan           the arithmetic behind the targets"
}

mlx3_check() {
    python3 - << 'CHKEOF' | tee $MLX3_LOGS/mlx3-env.txt
from importlib.metadata import version
import time
for p in ("mlx", "mlx-lm", "mlx-vlm", "numpy", "huggingface_hub"):
    try:
        print("  %-16s %s" % (p, version(p)))
    except Exception:
        print("  %-16s MISSING" % p)
import mlx.core as mx
import numpy as np
print("  device           %s" % mx.default_device())
print("  mlx/numpy bridge %s" % float(np.asarray(mx.ones((3,))).sum()))
print()
print("  the device line above is the real check: a cpu-only wheel says cpu.")
print("  For throughput run mlx3_bench, it is the number the plan depends on.")
try:
    import scipy
    print("  scipy            ok")
except Exception as e:
    print("  scipy            BROKEN: %s" % str(e).splitlines()[0][:60])
CHKEOF
    echo
    echo "written to $MLX3_LOGS/mlx3-env.txt. Every published number carries"
    echo "these versions: the flags on these tools move between releases."
}

# What the card actually does on the shape this work is made of. The plan's
# timing rests on this number, so it is measured rather than assumed.
#
# Two mistakes are easy to make here and both were in the old check. MLX is
# lazy: a loop that rebinds the result keeps only the last one, so nine of ten
# matmuls are discarded and never run, and dividing by ten inflates the answer
# tenfold. And the first matmul of a session includes kernel compilation, which
# has nothing to do with the card. So: warm up first, hold every result alive,
# evaluate once.
mlx3_bench() {
    python3 - "${1:-8192}" << 'BENCHEOF'
import sys, time
import mlx.core as mx

cap = int(sys.argv[1])
sizes = [n for n in (4096, 8192, 12288) if n <= cap]

print("device: %s" % mx.default_device(), flush=True)
print(flush=True)
print("%-10s %8s %14s" % ("dtype", "size", "TFLOP/s"), flush=True)
for dt, name in ((mx.bfloat16, "bfloat16"), (mx.float32, "float32")):
    for n in sizes:
        a = mx.random.normal((n, n)).astype(dt)
        b = mx.random.normal((n, n)).astype(dt)
        mx.eval(a, b)
        c = a @ b
        mx.eval(c)                       # warm up, compiles the kernel
        del c
        reps = 20 if n <= 8192 else 8
        t = time.perf_counter()
        for _ in range(reps):
            c = a @ b
            mx.eval(c)                   # one at a time: no giant graph to capture
            del c
        el = time.perf_counter() - t
        print("%-10s %8d %14.1f" % (name, n, reps * 2 * n ** 3 / el / 1e12), flush=True)
        del a, b
print(flush=True)
print("what the bfloat16 8192 row should look like, dense tensor core peak:")
print("  B200           2250 TFLOP/s peak, expect 1200 or more")
print("  H200 SXM        990 peak, expect 550 or more")
print("  H100 NVL        835 peak, expect 450 or more   (measured: 459.8)")
print("  RTX PRO 6000    200-250 peak, expect 120 or more")
print()
print("Tens instead of hundreds means the tensor cores are not being used and")
print("distillation would take roughly ten times the estimate.")
print()
print("If a row never prints, that size is where the backend stalls. Bisect it")
print("with mlx3_bench 4096 and mlx3_bench 8192, and check the driver version:")
print("the mlx[cuda13] wheel needs 580 or newer. A process at 100 percent of")
print("one core with nothing happening on the card is host side work, either")
print("kernel compilation or graph capture, not the card being slow. Try")
print("MLX_CUDA_GRAPH_CACHE_SIZE=1 mlx3_bench before spending time on it, and")
print("if that does not clear it, move the job to an sm_90 card and go on.")
BENCHEOF
}

mlx3_disk() {
cat << 'DISKEOF'

What each job needs on disk. Take at least 1 TB.

  bf16 reference checkpoint          51 GB   every box, it is the zero point
  reference logit cache              24 GB   every box, built once by mlx3_cache
  eval corpus                       0.8 MB
  8 bit teacher                      30 GB   only boxes that run mlx3_dwq
  original weights in /src           56 GB   only the box that builds a new rung
  one 4 bit rung                     16 GB
  one 3 bit rung                     13 GB

  box a   ref + cache + eval + two published rungs      about 105 GB
  box b   ref + cache + eval + src + teacher + rungs    about 190 GB
  box c   ref + cache + eval + teacher + 4bit rung      about 140 GB

The hub cache under HF_HOME can hold a second copy of anything pulled without
--local-dir. If space runs short mid job:  rm -rf $HF_HOME/*
DISKEOF
    df -h / | tail -1
}


# ================================================================== downloads

# mlx3_get src | ref | teacher | eval | calib | base LABEL | all
mlx3_get() {
    case "${1:-}" in
    src)
        if ls $MLX3_SRC/*.safetensors > /dev/null 2>&1; then
            echo "already here: $(du -sh $MLX3_SRC | cut -f1)"
            return 0
        fi
        echo "original weights of $MLX3_UP into $MLX3_SRC, about 56 GB."
        echo "Only needed on a box that BUILDS a new rung with mlx3_quant."
        hf download "$MLX3_UP" --local-dir $MLX3_SRC || return 1
        du -sh $MLX3_SRC ;;
    ref)
        if [ -f "$MLX3_REF/config.json" ]; then
            echo "already here: $(du -sh $MLX3_REF | cut -f1)"
            return 0
        fi
        echo "bf16 MLX reference, about 51 GB. This is the zero point every"
        echo "number is measured against. It is a pure format change from the"
        echo "original weights, so it reproduces byte for byte anywhere."
        hf download "$MLX3_ORG/$MLX3_STEM-MLX-bf16" --local-dir $MLX3_REF || return 1
        du -sh $MLX3_REF ;;
    teacher)
        if [ -f "$MLX3_TEACHER/config.json" ]; then
            echo "already here: $(du -sh $MLX3_TEACHER | cut -f1)"
            return 0
        fi
        echo "8 bit build as the distillation teacher, about 30 GB."
        echo "It sits 0.001273 from bf16, forty times below anything measured"
        echo "here, and it halves the memory against a 51 GB teacher. That is"
        echo "what makes mlx3_dwq fit on one card."
        hf download "$MLX3_ORG/$MLX3_STEM-MLX-8bit" --local-dir $MLX3_TEACHER || return 1
        du -sh $MLX3_TEACHER ;;
    base)
        if [ -z "$2" ]; then
            echo "mlx3_get base LABEL"
            echo "  published: mixed_3_4 4bit mixed_4_6 5bit 6bit 8bit"
            return 1
        fi
        local d=$MLX3_ROOT/$MLX3_STEM-MLX-$2
        if [ -f "$d/config.json" ]; then
            echo "already here: $(du -sh $d | cut -f1)"
            return 0
        fi
        hf download "$MLX3_ORG/$MLX3_STEM-MLX-$2" --local-dir "$d" || return 1
        du -sh "$d" ;;
    eval)
        if [ -f "$MLX3_EVAL" ]; then
            ls -lh $MLX3_EVAL
            return 0
        fi
        hf download AtomicChat/calib-corpora --repo-type dataset \
            --include "eval/neutral/eval_neutral.txt" --local-dir $MLX3_EVALDIR || return 1
        find $MLX3_EVALDIR -name "eval_neutral.txt" -exec cp {} $MLX3_EVAL \;
        ls -lh $MLX3_EVAL ;;
    calib)
        if [ -f "$MLX3_CALIB" ]; then
            ls -lh $MLX3_CALIB
            return 0
        fi
        echo "the calibration corpus, which mlx3_data turns into records."
        echo "Same text the importance matrix was collected on."
        hf download AtomicChat/calib-corpora --repo-type dataset \
            --include "builds/qwen3.8-27b/calib_train.txt" --local-dir $MLX3_EVALDIR || return 1
        find $MLX3_EVALDIR -name "calib_train.txt" -exec cp {} $MLX3_CALIB \;
        ls -lh $MLX3_CALIB ;;
    all)
        mlx3_get ref && mlx3_get eval && mlx3_get calib ;;
    *)
        echo "mlx3_get src | ref | teacher | eval | calib | base LABEL | all"
        return 1 ;;
    esac
}


# mlx3_wait LABEL
#
# The handoff between the two boxes. The hub builds a base, clips it and pushes
# it; this box polls for it and pulls it the moment it lands. Run it as the last
# command of the preparation block and walk away.
#
#   mlx3_wait 3bit-CLIP
mlx3_wait() {
    if [ -z "$1" ]; then
        echo "mlx3_wait LABEL        e.g. mlx3_wait 3bit-CLIP"
        echo "polls $MLX3_ORG/$MLX3_STEM-MLX-<LABEL> until it exists, then pulls it"
        return 1
    fi
    echo "waiting for $MLX3_ORG/$MLX3_STEM-MLX-$1 to appear. Ctrl-C to stop."
    while true; do
        if mlx3_get base "$1" 2>/dev/null; then
            echo
            echo "got it. Check the size above against what the hub reported: a"
            echo "smaller number means the upload was still in flight, in which"
            echo "case delete the directory and run this again."
            return 0
        fi
        echo "  not published yet, $(date +%H:%M:%S), retry in 30s"
        sleep 30
    done
}


# ================================================================== reference cache

# Recomputing the reference for every build measured against it is nineteen
# identical forward passes of a 51 GB model thrown away. This runs it once and
# keeps the logits, which is what llama.cpp does with its .kld file.
mlx3_cache() {
    if [ ! -f "$MLX3_REF/config.json" ]; then
        echo "no reference at $MLX3_REF. Run:  mlx3_get ref"
        return 1
    fi
    if [ ! -f "$MLX3_EVAL" ]; then
        echo "no corpus at $MLX3_EVAL. Run:  mlx3_get eval"
        return 1
    fi
    mlx3_write_py > /dev/null
    if [ -f "$MLX3_CACHE.json" ]; then
        echo "already here:"
        ls -lh "$MLX3_CACHE.f16"
        return 0
    fi
    echo "one bf16 forward pass over 24 chunks, then 24 GB of logits on disk."
    echo "About twelve minutes. Every later measurement on this box reuses it."
    date
    stdbuf -oL -eL python3 /mlx3_refcache.py "$MLX3_REF" "$MLX3_EVAL" \
        "$MLX3_CTX" "$MLX3_FIRST" "$MLX3_STEP" "$MLX3_CHUNKS" "$MLX3_CACHE" \
        2>&1 | tee $MLX3_LOGS/refcache.log
    date
    ls -lh "$MLX3_CACHE.f16"
}


# ================================================================== measurement

mlx3_kld() {
    local q="${1:-}"
    if [ -z "$q" ] || [ ! -f "${q%/}/config.json" ]; then
        echo "mlx3_kld /path/to/checkpoint"
        ls -d $MLX3_ROOT/*/ 2>/dev/null | sed "s/^/   /"
        return 1
    fi
    if [ ! -f "$MLX3_CACHE.json" ]; then
        echo "no reference cache. Run:  mlx3_cache"
        return 1
    fi
    mlx3_write_py > /dev/null
    local n
    n=$(basename "${q%/}")
    stdbuf -oL -eL python3 /mlx3_kld.py "$MLX3_CACHE" "${q%/}" \
        "$MLX3_LOGS/kld-$n.json" "$MLX3_TAIL" "${MLX3_QUICK:-0}" \
        2>&1 | tee "$MLX3_LOGS/kld-$n.log"
    mlx3_upload "$MLX3_LOGS/kld-$n.json" "logs/kld-$n.json"
    mlx3_upload "$MLX3_LOGS/kld-$n.log" "logs/kld-$n.log"
}

mlx3_all() {
    local d n
    for d in $MLX3_ROOT/*/; do
        [ -f "$d/config.json" ] || continue
        n=$(basename "${d%/}")
        case "$n" in *bf16*|*probe*|refcache*) continue ;; esac
        [ -f "$MLX3_LOGS/kld-$n.json" ] && [ "$MLX3_FORCE" != "1" ] && continue
        echo
        echo "########## $n ##########"
        mlx3_kld "${d%/}"
    done
    mlx3_table
}

mlx3_table() {
    python3 - "$MLX3_ROOT" "$MLX3_LOGS" << 'TBLEOF'
import glob, json, os, sys
root, log = sys.argv[1], sys.argv[2]
rows = []
for p in sorted(glob.glob(os.path.join(log, "kld-*.json"))):
    try:
        r = json.load(open(p))
    except Exception:
        continue
    n = os.path.basename(p)[4:-5]
    d = os.path.join(root, n)
    b = sum(os.path.getsize(f) for f in glob.glob(os.path.join(d, "*.safetensors")))
    r["name"] = n
    r["gb"] = round(b / 1e9, 2) if b else None
    rows.append(r)
rows.sort(key=lambda r: r.get("gb") or 0)

targets = [(13.70, 0.154161, "maglun 3.80bpw"),
           (16.05, 0.053775, "WaveCut 4bit-DWQ"),
           (17.57, 0.034350, "maglun 4.95bpw")]

if not rows:
    print("nothing measured on this box yet")
else:
    print()
    print("%-44s %7s %11s %9s %9s" % ("build", "GB", "mean KLD", "top-1 %", "ppl"))
    for r in rows:
        print("%-44s %7s %11.6f %9.2f %9.4f" % (
            r["name"][-44:], r.get("gb", ""), r["mean_kld"],
            r["top1_agree_pct"], r["quant_ppl"]))
print()
print("targets, and the best of ours at or below each size:")
for gb, kld, who in targets:
    ours = [r for r in rows if (r.get("gb") or 99) <= gb + 0.01]
    best = min(ours, key=lambda r: r["mean_kld"]) if ours else None
    if best is None:
        print("  %5.2f GB  %.6f  %-18s  nothing of ours at this size yet" % (gb, kld, who))
        continue
    win = 100.0 * (kld - best["mean_kld"]) / kld
    verdict = "BEATEN by %.1f %%" % win if win > 0 else "behind by %.1f %%" % -win
    print("  %5.2f GB  %.6f  %-18s  ours %.6f at %.2f GB -> %s"
          % (gb, kld, who, best["mean_kld"], best["gb"], verdict))
print()
try:
    a = json.load(open(os.path.join(log, "repeat-1.json")))
    b = json.load(open(os.path.join(log, "repeat-2.json")))
    d = 100.0 * abs(b["mean_kld"] - a["mean_kld"]) / a["mean_kld"]
    print("noise floor measured on this box: %.4f percent, from two passes of" % d)
    print("one checkpoint against the same cached reference. It is one")
    print("difference and not a distribution, so treat about three times it as")
    print("the smallest defensible gap and print it beside every number.")
except Exception:
    print("noise floor not measured on this box yet: mlx3_repeat DIR")
print()
TBLEOF
}


# ================================================================== the three checks

# WHAT THIS BLOCK IS FOR: it decides what counts as a result. Four publishers
# ship a 4 bit build that should be the same file, and their measured numbers
# spread by 0.17 percent. Either the files differ or the harness is noisy.
# Those two have opposite consequences and both are cheap to settle.
# mlx3_diff BUILD_A BUILD_B
#
# Two measured builds side by side across every metric, not just the mean.
# Written because the mean hides the thing that matters: at 3 bits the
# clipping search moved the median by 5.6 percent in the right direction and
# the 99th percentile by 3.3 percent in the wrong one, and the mean, which the
# tail dominates, showed 0.8 percent and told you nothing.
#
#   mlx3_diff Qwen3.8-27B-MLX-3bit Qwen3.8-27B-MLX-3bit-CLIP
mlx3_diff() {
    if [ -z "$2" ]; then
        echo "mlx3_diff BUILD_A BUILD_B      names as they appear in mlx3_table"
        ls $MLX3_LOGS/kld-*.json 2>/dev/null | sed "s|.*/kld-||; s|\.json$||; s|^|   |"
        return 1
    fi
    python3 - "$MLX3_LOGS" "$(basename ${1%/})" "$(basename ${2%/})" << 'DIFFEOF'
import json, os, sys
log, na, nb = sys.argv[1:4]

def get(n):
    p = os.path.join(log, "kld-%s.json" % n)
    if not os.path.exists(p):
        print("not measured on this box: %s" % n)
        sys.exit(1)
    return json.load(open(p))

a, b = get(na), get(nb)
print()
print("A: %s" % na)
print("B: %s" % nb)
print()
rows = [("mean KLD", "mean_kld", 1),
        ("median KLD", "median_kld", 1),
        ("90th KLD", "p90_kld", 1),
        ("95th KLD", "p95_kld", 1),
        ("99th KLD", "p99_kld", 1),
        ("max KLD", "max_kld", 1),
        ("quant ppl", "quant_ppl", 1),
        ("|mean delta p|", "mean_delta_p_points", 1),
        ("rms delta p", "rms_delta_p_points", 1),
        ("same top-1 %", "top1_agree_pct", -1)]
print("%-16s %13s %13s %11s   %s" % ("", "A", "B", "B vs A", ""))
for label, key, sign in rows:
    x, y = a.get(key), b.get(key)
    if x is None or y is None:
        continue
    # a signed metric whose best value is zero is compared by distance from it,
    # otherwise a bias of -3.1 improving to -2.4 reads as a 22 percent rise
    if key == "mean_delta_p_points":
        x, y = abs(x), abs(y)
    rel = 0.0 if x == 0 else 100.0 * (y - x) / abs(x)
    good = (rel * sign) < 0
    mark = "better" if abs(rel) > 0.25 and good else ("worse" if abs(rel) > 0.25 and not good else "noise")
    print("%-16s %13.6f %13.6f %+10.2f %%   %s" % (label, x, y, rel, mark))
print()
print("0.25 percent is three times the measured harness noise on one box, so")
print("anything smaller than that is marked noise and should not be argued")
print("about. A build can be better on the median and worse at the 99th: the")
print("mean is dominated by the tail, so read the whole column before deciding")
print("what a change did.")
DIFFEOF
}

mlx3_verify() {
    echo "=============== 1. are the four 4 bit builds one file ==============="
    echo "hashes straight off the hub, nothing downloaded, five seconds"
    mlx3_fp_remote
    echo
    echo "=============== 2. is the measurement repeatable ==============="
    echo "sixteen minutes of card time, once per box family, not per box:"
    echo "  mlx3_repeat $MLX3_ROOT/$MLX3_STEM-MLX-4bit"
    echo
    echo "=============== 3. which way llama.cpp's tail cutoff moves it ==============="
    mlx3_tail
}

mlx3_fp_remote() {
    python3 - $MLX3_SAME << 'FPREOF'
import sys
from huggingface_hub import HfApi
api = HfApi()
groups = {}
for repo in sys.argv[1:]:
    try:
        tree = list(api.list_repo_tree(repo, recursive=True))
    except Exception as e:
        print("%-46s cannot read: %s" % (repo, str(e).splitlines()[0][:36]))
        continue
    hits = []
    total = 0
    for f in tree:
        if not f.path.endswith(".safetensors"):
            continue
        lfs = getattr(f, "lfs", None)
        sha = getattr(lfs, "sha256", None) if lfs else None
        size = getattr(f, "size", 0) or 0
        total += size
        hits.append((f.path, sha, size))
    if not hits:
        print("%-46s no safetensors in the tree" % repo)
        continue
    print("%-46s %2d shards %8.2f GB  %s" % (repo, len(hits), total / 1e9,
          (sorted(hits)[0][1] or "no sha")[:12]))
    joint = "|".join(str(s) for _, s, _ in sorted(hits))
    groups.setdefault(joint, []).append(repo)
print()
vals = list(groups.values())
pairs = [g for g in vals if len(g) > 1]
if pairs and len(vals) > 1:
    print("PARTIAL MATCH. These hold byte for byte the same tensors:")
    for g in pairs:
        print("   ", ", ".join(g))
    print()
    print("That pair is the most useful line in this output. Two identical")
    print("files measured by our own harness can only differ by harness noise,")
    print("so the gap between their published numbers IS the noise floor, and")
    print("everything above it in the wider spread is real difference between")
    print("files that were all made by supposedly the same command.")
    print()
if len(vals) == 1 and len(vals[0]) > 1:
    print("IDENTICAL. Every repo above holds byte for byte the same tensors, so")
    print("the 0.17 percent spread between their measured numbers is harness")
    print("noise and nothing below it is a result. mlx3_repeat sizes it.")
else:
    print("%d distinct groups:" % len(vals))
    for g in vals:
        print("   ", ", ".join(g))
    print()
    print("Either the weights differ, or only the safetensors header does.")
    print("mlx3_fingerprint on two local copies tells them apart: it hashes the")
    print("tensor bytes and ignores the header. If the weights really differ,")
    print("the 0.17 percent belongs to the files and the harness noise is a")
    print("separate unknown that mlx3_repeat measures.")
FPREOF
}

mlx3_fingerprint() {
    local p="${1:-}"
    if [ -z "$p" ] || [ ! -f "${p%/}/config.json" ]; then
        echo "mlx3_fingerprint /path/to/checkpoint"
        return 1
    fi
    python3 - "${p%/}" << 'FPEOF'
import glob, hashlib, os, sys
import mlx.core as mx
import numpy as np
p = sys.argv[1]
h = hashlib.sha256()
n = 0
for f in sorted(glob.glob(os.path.join(p, "*.safetensors"))):
    d = mx.load(f)
    for k in sorted(d.keys()):
        a = d[k]
        h.update(k.encode())
        h.update(str(a.shape).encode())
        h.update(str(a.dtype).encode())
        h.update(np.ascontiguousarray(np.asarray(a.view(mx.uint8))).tobytes())
        n += 1
    del d
print("%s" % p)
print("  tensors    : %d" % n)
print("  fingerprint: %s" % h.hexdigest())
FPEOF
}

# Two full measurements of one checkpoint. Anything other than an exact match
# is nondeterminism in the reduction order on the card, and its size is the
# floor under every comparison in the table.
mlx3_repeat() {
    local q="${1:-}"
    if [ -z "$q" ] || [ ! -f "${q%/}/config.json" ]; then
        echo "mlx3_repeat /path/to/checkpoint"
        return 1
    fi
    local n
    n=$(basename "${q%/}")
    echo "pass 1 of 2, about eight minutes"
    MLX3_FORCE=1 mlx3_kld "${q%/}" || return 1
    cp "$MLX3_LOGS/kld-$n.json" $MLX3_LOGS/repeat-1.json
    echo
    echo "pass 2 of 2, about eight minutes"
    MLX3_FORCE=1 mlx3_kld "${q%/}" || return 1
    cp "$MLX3_LOGS/kld-$n.json" $MLX3_LOGS/repeat-2.json
    echo
    python3 - "$MLX3_LOGS" << 'REPEOF'
import json, sys
log = sys.argv[1]
a = json.load(open(log + "/repeat-1.json"))
b = json.load(open(log + "/repeat-2.json"))
print("%-22s %-17s %-17s %s" % ("", "pass 1", "pass 2", "relative"))
for k in ("mean_kld", "mean_kld_whole_vocab", "top1_agree_pct", "quant_ppl"):
    x, y = a[k], b[k]
    rel = 0.0 if x == 0 else 100.0 * (y - x) / x
    print("%-22s %-17.9f %-17.9f %+.6f %%" % (k, x, y, rel))
print()
d = abs(b["mean_kld"] - a["mean_kld"]) / a["mean_kld"] * 100.0
if d == 0.0:
    print("EXACTLY equal. The measurement is deterministic on this card, so the")
    print("0.17 percent spread between the four published 4 bit builds is a")
    print("property of those files. The noise floor is zero and a one percent")
    print("improvement is already a result.")
else:
    print("They differ by %.4f percent. That is the noise floor. Nothing below" % d)
    print("it is a result, and this figure belongs next to every number in the")
    print("published table.")
REPEOF
}

# The measurement computes both sums already. This reads them back out.
mlx3_tail() {
    python3 - "$MLX3_LOGS" << 'TAILEOF'
import glob, json, os, sys
log = sys.argv[1]
rows = []
for p in sorted(glob.glob(os.path.join(log, "kld-*.json"))):
    try:
        r = json.load(open(p))
    except Exception:
        continue
    if "mean_kld_whole_vocab" not in r:
        continue
    rows.append((os.path.basename(p)[4:-5], r["mean_kld"], r["mean_kld_whole_vocab"]))
if not rows:
    print("nothing measured on this box yet. Run one mlx3_kld, then this again.")
else:
    print("%-40s %12s %12s %10s" % ("build", "with cutoff", "whole vocab", "diff"))
    for n, a, b in rows:
        print("%-40s %12.6f %12.6f %+9.3f %%" % (n[-40:], a, b, 100.0 * (b - a) / a))
    print()
    print("llama.cpp drops vocabulary entries whose reference probability is")
    print("below e^-16, about one in ten million. Measured on this model, the")
    print("whole vocabulary sum is the SMALLER of the two by about 0.18 percent,")
    print("consistently. Those tail terms are p*(log p - log q) with q above p:")
    print("quantization smooths the distribution and lifts the floor under")
    print("improbable tokens, so the tail contributes negative terms and cutting")
    print("it off raises the number. The total is still non negative, only the")
    print("individual terms can be negative.")
    print()
    print("It only matters when one of our numbers sits next to a llama.cpp")
    print("number. Inside our own table every build used the same setting, so")
    print("the ranking does not move.")
TAILEOF
}


# ================================================================== build a rung

# mlx3_quant BITS [GROUP]
#
# A plain rung through mlx_vlm.convert, the only tool that keeps the vision
# tower, so the result loads in the app with nothing reassembled. Needs the
# original weights in /src.
#
# GROUP is how many neighbouring weights share one scale and one offset. The
# cost is two 16 bit numbers per group: group 128 adds 0.25 bits to every
# weight, group 64 adds 0.5, group 32 adds 1.0.
mlx3_quant() {
    if [ -z "$1" ]; then
        echo "mlx3_quant BITS [GROUP]        bits: 2 3 4 5 6 8"
        echo "  mlx3_quant 3 64      12.70 GB, the base for the low end lane"
        echo "  mlx3_quant 4 128     15.22 GB, the one size class nobody measured"
        return 1
    fi
    if ! ls $MLX3_SRC/*.safetensors > /dev/null 2>&1; then
        echo "no original weights at $MLX3_SRC. Run:  mlx3_get src"
        return 1
    fi
    local bits=$1 group=${2:-64} label out start rc
    case "$bits" in 2|3|4|5|6|8) ;; *) echo "bits: 2 3 4 5 6 8"; return 1 ;; esac
    label="${bits}bit"
    [ "$group" != "64" ] && label="${bits}bit-g${group}"
    out=$MLX3_ROOT/$MLX3_STEM-MLX-$label

    echo "target : $out"
    echo "bits   : $bits, group $group, mode affine, method rtn"
    echo "size   : about $(python3 -c "print('%.2f' % (0.92 + 3.365 * ($bits + 32.0/$group)))") GB"
    date
    start=$(date +%s)
    rm -rf "$out"
    stdbuf -oL -eL mlx_vlm.convert --hf-path $MLX3_SRC --mlx-path "$out" \
        -q --q-bits "$bits" --q-group-size "$group" --q-mode affine \
        --quant-method rtn 2>&1 | tee "$MLX3_LOGS/convert-$label.log"
    rc=${PIPESTATUS[0]}
    echo "took $(( $(date +%s) - start )) seconds"
    if [ ! -f "$out/config.json" ] || ! ls "$out"/*.safetensors > /dev/null 2>&1; then
        echo "nothing usable came out (exit code was $rc), removing the remains"
        rm -rf "$out"
        return 1
    fi
    du -sh "$out"
    echo
    echo "next:  mlx3_clip $out"
}


# ================================================================== method 1: clipping search

# mlx3_clip TEMPLATE [SUFFIX]
#
# WHAT IT DOES AND WHY IT IS WORTH TEN MINUTES
#
# The built in quantizer takes each group of G neighbouring weights, finds the
# smallest and the largest, and lays the available levels evenly between them:
#
#     w_hat = s*q + z,   s = (max - min) / (2^bits - 1),   z = min
#
# The step s is set by the single most extreme weight in the group, so one
# outlier costs the other sixty three weights resolution. llama.cpp has
# searched for a better range inside every k-quant since 2023. No tool in the
# MLX stack does, which is why all thirty one builds in the survey, ours and
# everyone else's, are rounded the same naive way.
#
# This tries a grid of narrower ranges per group, keeps the one with the
# smallest squared error, and hands the clipped weights back to mx.quantize so
# the packing is done by the library and the output is bit compatible with
# anything that loads MLX checkpoints. At alpha = 1.000 it reproduces exactly
# what the library does today, so the result can only tie or beat it.
#
# Bits and group size are read out of the TEMPLATE's own array shapes, not out
# of its config, so a mixed layout is reproduced exactly without describing it.
# Output size equals input size to the byte, so the comparison needs no
# interpolation and no argument.
#
# On synthetic weights the squared error falls about 21 percent at 3 bits,
# 10 at 4, 6 at 5. That is error on the WEIGHTS. How much survives sixty four
# layers is what mlx3_kld answers and nothing before it.
mlx3_clip() {
    if [ -z "$1" ]; then
        echo "mlx3_clip TEMPLATE_CHECKPOINT [SUFFIX]"
        echo "  mlx3_clip $MLX3_ROOT/$MLX3_STEM-MLX-4bit"
        echo
        ls -d $MLX3_ROOT/*/ 2>/dev/null | sed "s/^/   /"
        return 1
    fi
    local tmpl="${1%/}"
    local out="$tmpl-${2:-CLIP}"
    local start rc
    if [ ! -f "$tmpl/config.json" ]; then
        echo "no checkpoint at $tmpl"
        return 1
    fi
    if [ ! -f "$MLX3_REF/config.json" ]; then
        echo "no bf16 weights at $MLX3_REF. Run:  mlx3_get ref"
        return 1
    fi
    mlx3_write_py > /dev/null

    echo "template : $tmpl   $(du -sh $tmpl | cut -f1)"
    echo "weights  : $MLX3_REF"
    echo "target   : $out"
    echo "grid     : $MLX3_ALPHAS"
    echo
    echo "the grid is the fraction of the original range each group may keep."
    echo "1.000 is what every existing build did."
    date
    start=$(date +%s)
    rm -rf "$out"
    stdbuf -oL -eL python3 /mlx3_clip.py "$tmpl" "$MLX3_REF" "$out" "$MLX3_ALPHAS" \
        2>&1 | tee "$MLX3_LOGS/clip-$(basename $tmpl).log"
    rc=${PIPESTATUS[0]}
    echo "took $(( $(date +%s) - start )) seconds"
    if [ ! -f "$out/config.json" ] || ! ls "$out"/*.safetensors > /dev/null 2>&1; then
        echo "nothing usable came out (exit code was $rc), removing the remains"
        rm -rf "$out"
        return 1
    fi
    echo
    du -sh "$tmpl" "$out"
    echo
    echo "the two sizes above must match. Then:  mlx3_kld $out"
}


# ================================================================== layout

# The lever we spent two days not pulling. Clipping and distillation together
# move the divergence by two to five percent. A bit layout moves it by twenty.
#
# WHAT A LAYOUT IS. Every tensor gets its own bit count and group size instead
# of one setting for the whole model. Some roles are tiny and very sensitive,
# some are huge and tolerant, and the whole game is moving bits from the second
# kind to the first.
#
# THE ARITHMETIC, for this model, 26.893 B text parameters. A plain 3 bit
# group 64 build is 12.69 GB. Cost of lifting a role by one bit:
#
#   gate_proj, up_proj, down_proj   713 MB each      21.2 % of the model each
#   in_proj_qkv                     315 MB            9.4 %
#   in_proj_z, out_proj             189 MB            5.6 % each
#   embed_tokens, lm_head           159 MB            4.7 % each
#   q_proj                          126 MB            3.7 %
#   o_proj                           63 MB            1.9 %
#   k_proj, v_proj                   10 MB            0.3 % each
#   in_proj_a, in_proj_b              1.5 MB          0.04 % each
#
# And where the money comes from: embed_tokens from 3/64 to 2/128 frees
# 198 MB, gate_proj to group 128 frees 178 MB. A lookup error stays inside one
# token, which is why the vocabulary matrices are the right place to save.
#
# mlx3_price says what a rule set costs before anything is built, the way the
# GGUF side uses bits before quantize. mlx3_build then builds it out of the
# bf16 weights with the same clipping search mlx3_clip uses.

MLX3_RULES=${MLX3_RULES:-/mlx3-rules.txt}

# Writes the default rule set: AD-3.8, 13.67 GB, aimed just under maglun's
# 13.70 GB build. First match wins, so order matters.
mlx3_rules() {
    cat > "$MLX3_RULES" << 'RULESEOF'
# regex matched against the tensor name          bits  group
# first match wins, so the catch all goes last
# "skip" leaves the tensor in bf16 and records it as unquantized

# 0.04 percent of the model each, and the worst clipping error of any role.
# Full precision costs 37 MB for both.
\.in_proj_a\.weight$                              skip
\.in_proj_b\.weight$                              skip

# 0.3 percent each, ten megabytes per bit. Cheapest sensitive thing there is.
self_attn\.(k|v)_proj\.weight$                    8     64

# 9.4 percent of the model between them. A lookup error stays inside one
# token, so this is where the budget comes from rather than goes.
embed_tokens\.weight$                             2     128
lm_head\.weight$                                  3     128

self_attn\.(q|o)_proj\.weight$                    5     64

# The recurrent path. The GGUF ladder found ssm_out among the most valuable
# tensors in the model and maglun does not lift it at all.
linear_attn\.out_proj\.weight$                    5     64
linear_attn\.in_proj_(qkv|z)\.weight$             4     64

# 21.2 percent each. gate_proj pays for the rest by going to group 128.
mlp\.gate_proj\.weight$                           3     128
mlp\.up_proj\.weight$                             3     64
mlp\.down_proj\.weight$                           3     64

.*                                                3     64
RULESEOF
    echo "wrote $MLX3_RULES"
    echo
    echo "price it before building:  mlx3_price"
}

mlx3_price() {
    local rules="${1:-$MLX3_RULES}"
    if [ ! -f "$rules" ]; then
        echo "no rules at $rules. Run:  mlx3_rules"
        return 1
    fi
    if [ ! -f "$MLX3_REF/config.json" ]; then
        echo "no bf16 weights at $MLX3_REF. Run:  mlx3_get ref"
        return 1
    fi
    mlx3_write_py > /dev/null
    python3 /mlx3_layout.py price "$MLX3_REF" "$rules"
}

# mlx3_gen TARGET_GB [RULES_FILE]
#
# Writes a rule set that lands on a target size. The shape is fixed and only
# the level moves, so a sweep compares sizes rather than a hundred unrelated
# ideas. What the shape says, and every line of it is something we measured:
#
#   k_proj, v_proj at 8 bits      0.3 percent of the model, 10 MB per bit, and
#                                 the worst clipping error of any large role
#   in_proj_a, in_proj_b in bf16  0.04 percent each, worst error in the model
#   q, o                          two bits above the base
#   out_proj, in_proj_qkv, z      one bit above
#   down_proj, up_proj, gate_proj the base, and they are 21 percent each, so
#                                 the two lift counts below are the fine knob
#   embed_tokens, lm_head         group 128, a lookup error stays local
#
# Two lift counts give the granularity: how many of the 64 layers get
# down_proj two bits above the base, and up_proj one bit above. The search
# picks the combination closest to the target.
#
#   mlx3_gen 13.7 && mlx3_try AD-13.7
mlx3_gen() {
    if [ -z "$1" ]; then
        echo "mlx3_gen TARGET_GB [RULES_FILE]"
        echo "  mlx3_gen 10.5      the 16 GB Mac class"
        echo "  mlx3_gen 13.7      the 24 GB Mac class"
        echo "  mlx3_gen 17.4      the 32 GB Mac class"
        return 1
    fi
    python3 - "$1" "${2:-$MLX3_RULES}" << 'GENEOF'
import sys
H, I, V, L, F, N = 5120, 17408, 248320, 64, 16, 48
ROLE = {"embed_tokens": V*H, "lm_head": V*H, "gate": L*I*H, "up": L*I*H,
        "down": L*H*I, "q": F*12288*H, "k": F*1024*H, "v": F*1024*H,
        "o": F*H*6144, "ia": N*48*H, "ib": N*48*H, "qkv": N*10240*H,
        "z": N*6144*H, "out": N*H*6144}

def w(b, g):
    return 16.0 if b is None else b + 32.0 / g

def cl(b):
    return max(2, min(8, b))

def spec(base, emb, gg):
    return {"k": (8, 64), "v": (8, 64), "ia": (None, 0), "ib": (None, 0),
            "q": (cl(base+2), 64), "o": (cl(base+2), 64),
            "out": (cl(base+1), 64), "qkv": (cl(base+1), 64), "z": (cl(base+1), 64),
            "down": (cl(base), 64), "up": (cl(base), 64), "gate": (cl(base), gg),
            "embed_tokens": (cl(base+emb), 128), "lm_head": (cl(base+emb), 128)}

def size(s, de, ue):
    t = 0.0
    for k, n in ROLE.items():
        b, g = s[k]
        per = n / L
        if k == "down" and de:
            t += de*per*w(cl(b+2), g)/8 + (L-de)*per*w(b, g)/8
        elif k == "up" and ue:
            t += ue*per*w(cl(b+1), g)/8 + (L-ue)*per*w(b, g)/8
        else:
            t += n*w(b, g)/8
    return 0.92 + t/1e9

target, out = float(sys.argv[1]), sys.argv[2]
best = None
for base in range(2, 7):
    for emb in (-1, 0, 1):
        for gg in (64, 128):
            s = spec(base, emb, gg)
            for de in range(0, 65, 4):
                for ue in range(0, 65, 4):
                    z = size(s, de, ue)
                    d = abs(z - target)
                    if best is None or d < best[0]:
                        best = (d, z, base, emb, gg, de, ue, s)
_, z, base, emb, gg, de, ue, s = best

def edge(n):
    """The n layers at the two ends, as a regex alternation."""
    half = n // 2
    ids = list(range(half)) + list(range(L - (n - half), L))
    return "|".join(str(i) for i in ids)

lines = [
    "# generated by mlx3_gen for %.2f GB, lands on %.2f" % (target, z),
    "# base %d bits, embed offset %+d, gate group %d, down lift %d layers, up lift %d"
    % (base, emb, gg, de, ue),
    r"\.in_proj_a\.weight$                              skip",
    r"\.in_proj_b\.weight$                              skip",
    r"self_attn\.(k|v)_proj\.weight$                    8     64",
]
if de:
    lines.append(r"layers\.(%s)\.mlp\.down_proj\.weight$   %d  64" % (edge(de), cl(s["down"][0]+2)))
if ue:
    lines.append(r"layers\.(%s)\.mlp\.up_proj\.weight$     %d  64" % (edge(ue), cl(s["up"][0]+1)))
lines += [
    r"embed_tokens\.weight$                             %d    128" % s["embed_tokens"][0],
    r"lm_head\.weight$                                  %d    128" % s["lm_head"][0],
    r"self_attn\.(q|o)_proj\.weight$                    %d    64" % s["q"][0],
    r"linear_attn\.(out_proj|in_proj_qkv|in_proj_z)\.weight$   %d  64" % s["out"][0],
    r"mlp\.gate_proj\.weight$                           %d    %d" % (s["gate"][0], gg),
    r"mlp\.up_proj\.weight$                             %d    64" % s["up"][0],
    r"mlp\.down_proj\.weight$                           %d    64" % s["down"][0],
    r".*                                                %d    64" % s["down"][0],
]
open(out, "w").write("\n".join(lines) + "\n")
print("wrote %s" % out)
print("target %.2f GB, this layout is %.2f GB" % (target, z))
print("base %d bits, embed %+d, gate group %d, down lift %d, up lift %d"
      % (base, emb, gg, de, ue))
GENEOF
}

# mlx3_try LABEL [RULES]
# Build a layout and rank it on six chunks instead of twenty four. Under two
# minutes of card time per candidate, against an hour for a distillation run.
# Sweep layouts first, distil the winner last.
mlx3_try() {
    if [ -z "$1" ]; then
        echo "mlx3_try LABEL [RULES]"
        return 1
    fi
    mlx3_build "$1" "${2:-$MLX3_RULES}" || return 1
    MLX3_QUICK=${MLX3_QUICK:-6} mlx3_kld "$MLX3_ROOT/$MLX3_STEM-MLX-$1"
}

# mlx3_build LABEL [RULES]
mlx3_build() {
    if [ -z "$1" ]; then
        echo "mlx3_build LABEL [RULES]"
        echo "  mlx3_rules ; mlx3_price ; mlx3_build AD-3.8"
        return 1
    fi
    local label="$1"
    local rules="${2:-$MLX3_RULES}"
    local out="$MLX3_ROOT/$MLX3_STEM-MLX-$label"
    if [ ! -f "$rules" ]; then
        echo "no rules at $rules. Run:  mlx3_rules"
        return 1
    fi
    if [ ! -f "$MLX3_REF/config.json" ]; then
        echo "no bf16 weights at $MLX3_REF. Run:  mlx3_get ref"
        return 1
    fi
    mlx3_write_py > /dev/null
    echo "rules  : $rules"
    echo "weights: $MLX3_REF"
    echo "target : $out"
    python3 /mlx3_layout.py price "$MLX3_REF" "$rules"
    echo
    date
    local start rc
    start=$(date +%s)
    rm -rf "$out"
    stdbuf -oL -eL python3 /mlx3_layout.py build "$MLX3_REF" "$rules" "$out" \
        "$MLX3_ALPHAS" 2>&1 | tee "$MLX3_LOGS/build-$label.log"
    rc=${PIPESTATUS[0]}
    echo "took $(( $(date +%s) - start )) seconds"
    # Not the exit code: the CUDA backend can throw from a destructor after the
    # work is finished, and a finished checkpoint was once deleted for it.
    if [ ! -f "$out/config.json" ] || ! ls "$out"/*.safetensors > /dev/null 2>&1; then
        echo "nothing usable came out (exit code was $rc), removing the remains"
        rm -rf "$out"
        return 1
    fi
    du -sh "$out"
    echo
    echo "measure it:  mlx3_kld $out"
}

# ================================================================== method 2: distillation

# Memory arithmetic, printed before renting rather than discovered after.
mlx3_mem() {
    python3 - "${1:-4}" "${2:-64}" "${3:-2048}" "${4:-full}" << 'MEMEOF'
import sys
bits, group, seq = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
mode = sys.argv[4]                               # full | targets
params = 27.3e9          # the text half, the vision tower is not touched
vocab, hidden, layers = 248320, 5120, 64

# The term that was wrong. The loss is
#     kl_div_loss(scale * student_logits, scale * teacher_logits)
# over the FULL vocabulary, and every intermediate is a [seq, vocab] float32
# tensor: both scaled inputs, the teacher softmax, its log, the student
# log_softmax with its own max, exp and sum, the product, and then the
# backward pass through all of it. Twelve to fifteen of them are alive at
# once. This file used to count two and that is how a 27B model on a 143 GB
# card ran out of memory.
one = seq * vocab * 4 / 1e9
copies = 14 if mode == "full" else 2

teacher = 29.5 if mode == "full" else 0.0        # deleted once targets exist
student = 0.92 + 3.365 * (bits + 32.0 / group)
train = 2 * params / group
optim = train * 20 / 1e9        # fp32 param, grad, adam m, adam v, plus the
                                # astype copy loss_fn makes every step
acts = layers * seq * hidden * 2 / 1e9
logits = copies * one
peak = 2.0

print()
print("distillation at %d bits, group %d, window %d, targets %s"
      % (bits, group, seq, "precomputed" if mode == "targets" else "on the fly"))
print()
print("  teacher, 8 bit                        %6.1f GB" % teacher)
print("  student                               %6.1f GB" % student)
print("  %6.0f M trainable, fp32 with adam     %6.1f GB" % (train / 1e6, optim))
print("  activations with grad checkpoint      %6.1f GB" % acts)
print("  loss over the vocabulary, %2d x %.2f  %6.1f GB" % (copies, one, logits))
print("  allocator slack                       %6.1f GB" % peak)
total = teacher + student + optim + acts + logits + peak
print("  " + "-" * 46)
print("  total                                 %6.1f GB" % total)
print()
for name, cap in (("B200", 180), ("H200 NVL", 141), ("H100 NVL", 94),
                  ("RTX PRO 6000", 96)):
    head = cap - total
    v = "fits with room" if head > 30 else ("tight" if head > 10 else "NO")
    print("  %-14s %3d GB   headroom %+6.1f GB   %s" % (name, cap, head, v))
print()
if mode == "full":
    print("Two levers, in this order, before renting a bigger card:")
    print()
    print("  precomputed targets. mlx_lm.dwq can compute the teacher logits")
    print("  once, store them as top-k values plus indices, DELETE the teacher,")
    print("  and then gather the student's logits down to k so the loss stops")
    print("  touching 248320 columns. Both big terms vanish at once. Find the")
    print("  flag with:  mlx_lm.dwq --help 2>&1 | grep -i -B1 -A3 target")
    print("  then pass it through MLX3_EXTRA. Compare:  mlx3_mem %d %d %d targets"
          % (bits, group, seq))
    print()
    print("  a shorter window. The loss term is linear in it, so 2048 to 512")
    print("  divides %.1f GB by four." % logits)
else:
    print("This is the configuration to aim for. It fits on every card above,")
    print("and the window can stay at 2048 where the eval actually scores.")
print()
print("An estimate, not a measurement. Watch nvidia-smi over the first fifty")
print("steps: if it settles below the total, the rest of the run is flat.")
MEMEOF
}

# Calibration records, sized against the real tokenizer instead of a guess.
#
# This bit already cost a run. Records of 9000 characters are about 2250 tokens,
# and mlx-lm's loader does not truncate a sequence longer than max_seq_length,
# it drops it. Every record was longer than the 2048 window, every record was
# dropped, and the validation set came out empty:
#
#   Generating validation split: 60 examples
#   ValueError: Dataset must have at least batch_size=1 examples but only has 0
#
# So the length is now measured rather than assumed: tokenize a slice of the
# corpus, get the real characters per token, size the records to 90 percent of
# the window, then tokenize twenty of them and print what actually came out.
#
#   mlx3_data              sizes for MLX3_SEQ, currently 2048
#   mlx3_data 4096         sizes for a 4096 window
mlx3_data() {
    if [ ! -f "$MLX3_CALIB" ]; then
        echo "no calibration text at $MLX3_CALIB. Run:  mlx3_get calib"
        return 1
    fi
    local target="${1:-$MLX3_SEQ}"
    local tokdir="$MLX3_TEACHER"
    [ -f "$tokdir/tokenizer.json" ] || tokdir="$MLX3_REF"
    if [ ! -f "$tokdir/tokenizer.json" ]; then
        echo "no tokenizer anywhere. Run mlx3_get teacher or mlx3_get ref."
        return 1
    fi
    mkdir -p $MLX3_DATA
    echo "window   : $target tokens"
    echo "tokenizer: $tokdir"
    python3 - "$MLX3_CALIB" "$MLX3_DATA" "$target" "$MLX3_TRAIN" "$MLX3_VALID" \
              "$tokdir" "$MLX3_CHARS" << 'DATAEOF'
import json, sys
from transformers import AutoTokenizer

src, out, target, ntrain, nvalid, tokdir, forced = sys.argv[1:8]
target, ntrain, nvalid, forced = int(target), int(ntrain), int(nvalid), int(forced)

tok = AutoTokenizer.from_pretrained(tokdir)
text = open(src, encoding="utf-8", errors="replace").read()

probe = text[:200000]
n_tok = len(tok.encode(probe))
ratio = len(probe) / max(1, n_tok)
print("characters per token on this corpus: %.2f" % ratio)

if forced > 0:
    chars = forced
    print("record length forced to %d characters by MLX3_CHARS" % chars)
else:
    chars = int(target * 0.90 * ratio)
    print("record length %d characters, aimed at %d tokens, ninety percent of"
          % (chars, int(target * 0.90)))
    print("the window so nothing is long enough to be dropped")

need = chars * (ntrain + nvalid)
if len(text) < need:
    print("corpus is %d chars, %d wanted, so there will be fewer records"
          % (len(text), need))
recs = [text[i:i + chars] for i in range(0, min(len(text), need), chars)]
recs = [r for r in recs if len(r) > chars // 2]

for name, part in (("train", recs[:ntrain]), ("valid", recs[ntrain:ntrain + nvalid])):
    p = "%s/%s.jsonl" % (out, name)
    with open(p, "w", encoding="utf-8") as f:
        for r in part:
            f.write(json.dumps({"text": r}, ensure_ascii=False) + "\n")
    print("%-6s %4d records -> %s" % (name, len(part), p))

sample = recs[:20]
lens = sorted(len(tok.encode(r)) for r in sample)
print()
print("token length of the first %d records: min %d, median %d, max %d"
      % (len(sample), lens[0], lens[len(lens) // 2], lens[-1]))
if lens[-1] > target:
    print()
    print("STOP. %d records exceed the %d token window. mlx-lm DROPS those"
          % (sum(1 for l in lens if l > target), target))
    print("rather than truncating them, and if all of them are dropped the")
    print("validation set is empty and the run dies before the first step.")
    print("Either rerun with a larger window, mlx3_data 4096, or force a")
    print("shorter record: MLX3_CHARS=6000 mlx3_data")
    sys.exit(1)
print("all of them fit, the run will see every record")
print()
print("This is the corpus the importance matrix was collected on. The eval set")
print("is a different split of the same pipeline: a legitimate held out")
print("measurement, and also our own distribution. The card says so, and the")
print("winner gets measured on eval_agentic as well.")
DATAEOF
}

mlx3_dwq_help() {
    echo "read this before the first run: flags move between mlx-lm releases"
    echo
    if command -v mlx_lm.dwq > /dev/null 2>&1; then
        mlx_lm.dwq --help
    else
        python3 -m mlx_lm quant.dwq --help
    fi
}

# mlx3_targets [SEQ] [SAMPLES]
#
# The teacher logits, computed once, as top-k values plus indices on disk. Run
# it as its own pass so a failure costs only this phase rather than a loaded
# student and a built optimizer as well.
#
# The result serves BOTH lanes: same teacher, same data, same sample count,
# same seed, so 3 bit and 4 bit reuse one directory.
#
# Watch it from a second tmux pane:  nvidia-smi -l 5
# Flat memory means the shapes collapsed as intended. Still climbing means
# they did not, so halve MLX3_SEQ and start again.
mlx3_targets() {
    local seq="${1:-$MLX3_SEQ}"
    local samples="${2:-$MLX3_SAMPLES}"
    if [ ! -f "$MLX3_TEACHER/config.json" ]; then
        echo "no teacher at $MLX3_TEACHER. Run:  mlx3_get teacher"
        return 1
    fi
    local dataflag="--data-path $MLX3_DATA"
    [ "$MLX3_DATA" = "default" ] && dataflag=""
    local sampleflag="--num-samples $samples"
    [ "$samples" = "0" ] && sampleflag=""
    mkdir -p "$MLX3_TARGETS"
    echo "teacher : $MLX3_TEACHER"
    echo "targets : $MLX3_TARGETS"
    echo "window  : $seq, batch $MLX3_BATCH, graphs $MLX_USE_CUDA_GRAPHS"
    echo
    echo "in a second pane:  nvidia-smi -l 5"
    echo "memory has to stay flat. If it climbs by roughly a gigabyte a batch,"
    echo "the shapes are not collapsing: halve the window and start again."
    echo
    date
    if command -v mlx_lm.dwq > /dev/null 2>&1; then
        stdbuf -oL -eL mlx_lm.dwq --model "$MLX3_TEACHER" \
            --mlx-path /tmp/mlx3-unused $dataflag $sampleflag \
            --batch-size "$MLX3_BATCH" --max-seq-length "$seq" \
            --target-dir "$MLX3_TARGETS" --targets-only \
            2>&1 | tee "$MLX3_LOGS/targets-s$seq.log"
    else
        stdbuf -oL -eL python3 -m mlx_lm quant.dwq --model "$MLX3_TEACHER" \
            --mlx-path /tmp/mlx3-unused $dataflag $sampleflag \
            --batch-size "$MLX3_BATCH" --max-seq-length "$seq" \
            --target-dir "$MLX3_TARGETS" --targets-only \
            2>&1 | tee "$MLX3_LOGS/targets-s$seq.log"
    fi
    date
    du -sh "$MLX3_TARGETS"
    echo
    echo "both lanes reuse this. Next:  mlx3_dwq 3 64 /mlx/<stem>-MLX-3bit-CLIP"
}

# mlx3_dwq BITS GROUP BASE [TEACHER] [SEQ] [LR] [SAMPLES]
#
# WHAT IT DOES
#
# Gradient descent on the scales and the offsets of an already quantized model,
# minimizing the divergence between it and a teacher. The integers stay where
# the base quantization put them. That is why building a better base first with
# mlx3_clip is worth doing: clipping picks better integers, this tunes the
# scales around them, and neither replaces the other.
#
# WHAT TO EXPECT, AND WHERE THE NUMBERS COME FROM
#
# The quantity it minimizes is KL between this build and the teacher, which is
# the same quantity mlx3_kld measures, not a proxy for it. Two published 3 bit
# runs on other models at these settings, mlx-lm 0.28.4, rate 3e-7, batch 1,
# about 1.09 M tokens, report validation loss falling 0.146 -> 0.088 and
# 0.168 -> 0.089, so 40 and 47 percent.
#
# MEASURED ON THIS MODEL, 3 bits, group 64, batch 4, window 512, 150 steps,
# 8 bit teacher, tulu-3-sft-mixture, rate 3e-7, one hour on an H200 NVL:
#
#   validation loss 0.296 at step 0 -> 0.155 at step 149, so 47.6 percent
#   peak memory 70.3 GB, which also fits a 94 GB card
#
# That loss is against the TEACHER on tulu. The table measures against bf16 on
# eval_neutral, so the two are the same shape but not the same number. Only
# mlx3_kld says what actually happened.
#
#   3 bits, 12.70 GB : 0.221627 after clipping, 0.116 if 47.6 percent carries
#   4 bits, 16.05 GB : 0.054444 after clipping
#
# The 4 bit figure assumes 15 percent and there is no direct measurement for
# it. WaveCut reports 3.6 percent at 4 bits, which cannot be what the method
# does: damage at 3 bits is four times larger and the gain there is 40 percent,
# so a smaller gain at 3 bits is impossible and his run was undertrained.
# Range 10 to 25 percent, midpoint taken. If the measurement lands near 3, four
# bits has nothing left to repair and every remaining hour goes to the 3 bit
# lane.
#
# THE BASE MUST BE UNIFORM. A mixed layout carries per tensor bits and group
# sizes and passing --bits on top of it is undefined. Clip a mixed rung and
# publish it, distil only the uniform ones.
#
# DEFAULTS AND WHY
#
#   teacher   the 8 bit build. 0.001273 from bf16, forty times below anything
#             measured here, and half the memory of a bf16 teacher.
#   window    2048, the tool's own default, and also the range the eval scores:
#             chunks are 4096 and only the second half counts. A window of 512
#             tunes the scales for behaviour the measurement never looks at.
#   rate      3e-7, the value in both published 3 bit runs.
#   samples   600, about 1.2 M tokens at this window, the same order as those.
mlx3_dwq() {
    if [ -z "$3" ]; then
        echo "mlx3_dwq BITS GROUP BASE [TEACHER] [SEQ] [LR] [SAMPLES]"
        echo "  mlx3_dwq 3 64 $MLX3_ROOT/$MLX3_STEM-MLX-3bit-CLIP"
        echo "  mlx3_dwq 4 64 $MLX3_ROOT/$MLX3_STEM-MLX-4bit-CLIP"
        echo
        echo "check the memory first:  mlx3_mem 3 64 2048"
        return 1
    fi
    local bits=$1 group=$2
    local base="${3%/}"
    local teacher="${4:-$MLX3_TEACHER}"
    local seq="${5:-$MLX3_SEQ}"
    local lr="${6:-$MLX3_LR}"
    local samples="${7:-$MLX3_SAMPLES}"
    local out="$base-DWQ"
    local tag
    tag="$(basename $base)-${bits}b-g${group}-s${seq}"

    if [ ! -f "$base/config.json" ]; then
        echo "no base at $base"
        return 1
    fi
    if [ ! -f "$teacher/config.json" ]; then
        echo "no teacher at $teacher. Run:  mlx3_get teacher"
        return 1
    fi
    # WHY THE DEFAULT IS NOT OUR OWN CORPUS, mlx-lm 0.31.3
    #
    # --data-path pointing at a local directory of train.jsonl and valid.jsonl
    # does not work. It routes through load_custom_hf_dataset rather than
    # load_local_dataset, and there, in mlx_lm/tuner/datasets.py:
    #
    #     train_split = ds.get("train_split", "train[:80%]")
    #     valid_split = ds.get("valid_split", "train[-10%:]")
    #
    # validation is sliced out of the TRAIN split and valid.jsonl is never
    # read. The result is an empty validation set and the run dies before the
    # first step with "Dataset must have at least batch_size=1 examples but
    # only has 0". Record length has nothing to do with it: 1000 token records
    # against a 4096 window fail the same way, and max_seq_length does not
    # appear in that file at all. Isolated by two runs, one with --data-path
    # and no --num-samples (fails), one with neither (works).
    #
    #   MLX3_DATA=default     mlx-lm's own set, allenai/tulu-3-sft-mixture.
    #                         Foreign calibration, which goes in the model
    #                         card. It also makes the result directly
    #                         comparable to the published runs and removes any
    #                         argument that we calibrated on our own eval
    #   MLX3_DATA=/dwq-data   our corpus, for when the bug upstream is fixed
    #   MLX3_SAMPLES=0        drop --num-samples. Innocent in the above, but
    #                         with the default set it is what stops the run
    #                         from walking the whole 939343 record mixture
    local dataflag="--data-path $MLX3_DATA"
    local sampleflag="--num-samples $samples"
    if [ "$MLX3_DATA" = "default" ]; then
        dataflag=""
        echo "data    : mlx-lm's own default set, ours is not used"
    else
        if [ ! -f "$MLX3_DATA/train.jsonl" ]; then
            echo "no calibration records. Run:  mlx3_data $seq"
            return 1
        fi
        if [ ! -s "$MLX3_DATA/valid.jsonl" ]; then
            echo "$MLX3_DATA/valid.jsonl is empty. Run:  mlx3_data $seq"
            return 1
        fi
    fi
    [ "$samples" = "0" ] && sampleflag=""

    # --target-dir is not optional on this model, see the note at the top of
    # the file. If the directory is empty dwq fills it first and drops the
    # teacher afterwards, so no separate pass is needed. If the target
    # computation itself runs out of memory, split it by hand: the same
    # command with --targets-only added, then the same command without it.
    local targets="$MLX3_TARGETS"
    [ -z "$targets" ] && targets="/mlx/dwq-targets-$seq"
    mkdir -p "$targets"

    echo "teacher : $teacher   $(du -sh $teacher | cut -f1)"
    echo "base    : $base   $(du -sh $base | cut -f1)"
    echo "target  : $out"
    echo "bits $bits, group $group, window $seq, batch $MLX3_BATCH, rate $lr"
    echo "cuda graphs: $MLX_USE_CUDA_GRAPHS   (1 breaks on this model, see the top of the file)"
    echo "flags   : $dataflag $sampleflag $MLX3_EXTRA"
    echo "targets : $targets"
    echo "graph cache: $MLX_CUDA_GRAPH_CACHE_SIZE   (1 aborts, 6144 runs out of memory)"
    mlx3_mem "$bits" "$group" "$seq" targets
    echo
    echo "WHAT TO WATCH. The log prints a training and a validation loss, and"
    echo "both are the same quantity the table measures. Falling steadily is"
    echo "right. Oscillating without falling means the rate is too high, halve"
    echo "it. Falling a few percent over the whole run means too low, triple it."
    echo "Published 3 bit runs fall by 40 to 47 percent."
    echo
    echo "IF IT DIES ON MEMORY WHILE COMPUTING TARGETS, it is not the card."
    echo "Measured: a 94 GB card died at batch 50 of 600 and a 143 GB card at"
    echo "batch 442 of the same 600. Half again the memory bought nine times"
    echo "the batches, which means the memory ACCUMULATES rather than peaking."
    echo "With batch size 1 every record goes at its own length, every length"
    echo "is a new tensor shape, and the allocator cannot reuse a buffer sized"
    echo "for 1700 tokens to hold 1900. Five hundred lengths is five hundred"
    echo "pools that never merge. A bigger card dies later, not never."
    echo
    echo "The fix is a shorter window, and it works on both causes at once:"
    echo "buffers are smaller, AND almost every record is longer than 512 so"
    echo "almost every one truncates to exactly 512. The long tail of lengths"
    echo "collapses into one dominant shape, the buffer cache starts being"
    echo "reused and the graph cache stops growing."
    echo
    echo "  --max-seq-length 512 --num-samples 300"
    echo
    echo "Use a fresh target directory when the window changes: targets are"
    echo "shaped by it. This function names one per window automatically."
    echo
    date
    rm -rf "$out"
    if command -v mlx_lm.dwq > /dev/null 2>&1; then
        stdbuf -oL -eL mlx_lm.dwq \
            --model "$teacher" --quantized-model "$base" --mlx-path "$out" \
            --bits "$bits" --group-size "$group" \
            $dataflag $sampleflag $MLX3_EXTRA \
            --target-dir "$targets" \
            --batch-size "$MLX3_BATCH" --max-seq-length "$seq" \
            --learning-rate "$lr" --grad-checkpoint \
            2>&1 | tee "$MLX3_LOGS/dwq-$tag.log"
    else
        stdbuf -oL -eL python3 -m mlx_lm quant.dwq \
            --model "$teacher" --quantized-model "$base" --mlx-path "$out" \
            --bits "$bits" --group-size "$group" \
            $dataflag $sampleflag $MLX3_EXTRA \
            --target-dir "$targets" \
            --batch-size "$MLX3_BATCH" --max-seq-length "$seq" \
            --learning-rate "$lr" --grad-checkpoint \
            2>&1 | tee "$MLX3_LOGS/dwq-$tag.log"
    fi
    date
    mlx3_upload "$MLX3_LOGS/dwq-$tag.log" "logs/dwq-$tag.log"
    if [ ! -f "$out/config.json" ]; then
        echo "nothing came out, read the log above"
        return 1
    fi
    du -sh "$out"
    echo
    echo "measure it:  mlx3_kld $out"
    echo "the output is text only. The vision tower goes back on with"
    echo "mlx3_reattach, and only for the builds worth publishing."
}


# ================================================================== vision tower

# mlx3_reattach TEXT_CHECKPOINT VISION_CHECKPOINT OUT
#
# Distillation runs through mlx-lm, which handles the language half only. This
# puts the text tensors from that build next to the untouched tower from an
# mlx-vlm build, under the names the vlm loader expects. The result is a claim
# until it has been loaded and shown a picture.
mlx3_reattach() {
    if [ -z "$3" ]; then
        echo "mlx3_reattach TEXT VISION OUT"
        echo "  mlx3_reattach /mlx/X-4bit-CLIP-DWQ /mlx/X-4bit /mlx/X-4bit-CLIP-DWQ-VL"
        return 1
    fi
    python3 - "${1%/}" "${2%/}" "${3%/}" << 'REATTEOF'
import glob, json, os, shutil, sys
import mlx.core as mx

text_p, vis_p, out_p = sys.argv[1:4]

def read_all(path):
    out = {}
    for f in sorted(glob.glob(os.path.join(path, "*.safetensors"))):
        out.update(mx.load(f))
    return out

tw = read_all(text_p)
vw = read_all(vis_p)
print("text build   : %d tensors" % len(tw))
print("vision build : %d tensors" % len(vw))
if not tw or not vw:
    sys.exit(1)

# Exact names first. mlx_lm writes the same keys it read, including the
# language_model. prefix, so the two builds line up one to one. An earlier
# version detected that prefix and stripped it before looking the name up,
# which matched nothing: it reported "replaced 0, not found 1847" and wrote out
# the untouched base with a new quantization block. That file was published.
# Anything that replaces nothing now stops instead of writing.
merged = {}
rep = kept = 0
for k, v in vw.items():
    if k in tw:
        merged[k] = tw[k]; rep += 1
    else:
        merged[k] = v; kept += 1

if rep == 0:
    # Fallback for a text build whose names really are prefix free.
    sample = sorted(tw.keys())[len(tw) // 2]
    cands = [k for k in vw if k.endswith("." + sample) or k.endswith(sample)]
    if cands:
        prefix = cands[0][: len(cands[0]) - len(sample)]
        print("no exact matches, trying prefix %r" % prefix)
        merged = {}
        rep = kept = 0
        for k, v in vw.items():
            src = k[len(prefix):] if k.startswith(prefix) else None
            if src is not None and src in tw:
                merged[k] = tw[src]; rep += 1
            else:
                merged[k] = v; kept += 1

print("replaced from the text build : %d" % rep)
print("kept from the vlm build      : %d" % kept)
if rep == 0:
    print()
    print("NOTHING WAS REPLACED, refusing to write. The output would be the")
    print("vision build with a different quantization block, which looks like")
    print("a result and is not one.")
    print("  text sample: %s" % sorted(tw.keys())[len(tw) // 2])
    print("  vlm  sample: %s" % sorted(vw.keys())[len(vw) // 2])
    sys.exit(1)

os.makedirs(out_p, exist_ok=True)
for f in glob.glob(os.path.join(vis_p, "*")):
    if f.endswith(".safetensors") or f.endswith(".safetensors.index.json"):
        continue
    if os.path.isfile(f):
        shutil.copy2(f, out_p)
tcfg = json.load(open(os.path.join(text_p, "config.json")))
vcfg = json.load(open(os.path.join(out_p, "config.json")))
if "quantization" in tcfg:
    vcfg["quantization"] = tcfg["quantization"]
    json.dump(vcfg, open(os.path.join(out_p, "config.json"), "w"), indent=2)
    print("quantization block copied from the text build")
idx = os.path.join(out_p, "model.safetensors.index.json")
if os.path.exists(idx):
    os.remove(idx)
mx.save_safetensors(os.path.join(out_p, "model.safetensors"), merged,
                    metadata={"format": "mlx"})
size = os.path.getsize(os.path.join(out_p, "model.safetensors"))
print()
print("wrote %s, %.2f GB" % (out_p, size / 1e9))
print("now load it and show it a picture. Until then this is a claim.")
sys.stdout.flush()
os._exit(0)
REATTEOF
}


# ================================================================== publishing

mlx3_upload() {
    [ -f "$1" ] || return 0
    [ -z "$HF_TOKEN" ] && return 0
    python3 - "$1" "$2" "$MLX3_METRICS" << 'UPEOF'
import sys
from huggingface_hub import HfApi
local, remote, repo = sys.argv[1:4]
try:
    HfApi().upload_file(path_or_fileobj=local, path_in_repo=remote,
                        repo_id=repo, repo_type="dataset")
    print("  uploaded -> %s :: %s" % (repo, remote))
except Exception as e:
    print("  upload failed: %s" % str(e).splitlines()[0][:70])
UPEOF
}

mlx3_push() {
    if [ -z "$2" ]; then
        echo "mlx3_push CHECKPOINT LABEL"
        echo "  mlx3_push $MLX3_ROOT/$MLX3_STEM-MLX-4bit-CLIP-DWQ 4bit-CLIP-DWQ"
        return 1
    fi
    local d="${1%/}"
    local repo="$MLX3_ORG/$MLX3_STEM-MLX-$2"
    local a
    if [ ! -f "$d/config.json" ]; then
        echo "no checkpoint at $d"
        return 1
    fi
    if grep -rlE 'hf_[A-Za-z0-9]{30,}' "$d" 2>/dev/null | head -1 | grep -q .; then
        echo "credential shaped string inside $d, refusing to upload"
        return 1
    fi
    echo "$d  ->  $repo   ($(du -sh $d | cut -f1))"
    # This used to prompt by default. A sequence left running overnight then
    # stopped on the first upload and nothing after it ran. The prompt is now
    # opt in: MLX3_ASK=1 brings it back. Nothing else in this file blocks.
    if [ "$MLX3_ASK" = "1" ]; then
        read -p "upload? [y/N] " a
        [ "$a" = "y" ] || return 1
    fi
    python3 - "$d" "$repo" << 'PUSHEOF'
import sys
from huggingface_hub import HfApi
local, repo = sys.argv[1:3]
api = HfApi()
try:
    api.create_repo(repo, exist_ok=True)
except Exception as e:
    print(str(e).splitlines()[0])
api.upload_folder(folder_path=local, repo_id=repo)
print("done: https://huggingface.co/%s" % repo)
PUSHEOF
}


# ================================================================== python on disk

# Kept as files rather than heredocs inside a pipeline: a heredoc attached to
# the wrong end of a pipe feeds the script to tee instead of to python.
mlx3_write_py() {

cat > /mlx3_refcache.py << 'RCEOF'
"""Run the reference once and keep its logits, the way llama.cpp keeps a .kld.

    python3 mlx3_refcache.py REF CORPUS CTX FIRST STEP MAX_CHUNKS OUT_PREFIX
"""
import json, os, sys
import mlx.core as mx
import numpy as np
from mlx_lm import load

ref, corpus, ctx, first, step, cap, out = sys.argv[1:8]
ctx, step, cap = int(ctx), int(step), int(cap)
first = int(first) if first else ctx // 2

model, tok = load(ref)
ids = tok.encode(open(corpus, encoding="utf-8", errors="replace").read())
n_chunk = len(ids) // ctx
if cap and cap < n_chunk:
    n_chunk = cap
per = ctx - 1 - first

probe = model(mx.array([ids[:8]]))
probe = probe.logits if hasattr(probe, "logits") else probe
vocab = int(probe.shape[-1])
del probe

rows = n_chunk * per
print("chunks %d, %d scored each, vocab %d" % (n_chunk, per, vocab), flush=True)
print("cache %.1f GB at %s.f16" % (rows * vocab * 2 / 1e9, out), flush=True)

mm = np.memmap(out + ".f16", dtype=np.float16, mode="w+", shape=(rows, vocab))
w = 0
for c in range(n_chunk):
    chunk = ids[c * ctx:(c + 1) * ctx]
    o = model(mx.array([chunk]))
    lg = (o.logits if hasattr(o, "logits") else o)[0]
    mx.eval(lg)
    for s in range(first, ctx - 1, step):
        e = min(s + step, ctx - 1)
        mm[w:w + (e - s)] = np.asarray(lg[s:e].astype(mx.float16))
        w += (e - s)
    del lg, o
    print("  chunk %d/%d" % (c + 1, n_chunk), flush=True)
mm.flush()
json.dump({"reference": ref, "ids": ids[:n_chunk * ctx], "ctx": ctx,
           "first": first, "step": step, "chunks": n_chunk, "per_chunk": per,
           "vocab": vocab, "rows": rows}, open(out + ".json", "w"))
print("done, %d rows" % w)

# The MLX CUDA backend throws from a destructor at interpreter shutdown:
#   terminate called ... Destroy(handle_) failed: driver shutting down
# It happens after all the work is done, but it aborts the process and the
# shell then sees a non zero exit code. That cost one finished 13.68 GB
# checkpoint, deleted by a cleanup branch that trusted the code. Leaving here
# without unwinding skips the destructors entirely.
sys.stdout.flush()
sys.stderr.flush()
os._exit(0)

RCEOF

cat > /mlx3_kld.py << 'K3EOF'
"""Measure one quant against the cached reference. One forward pass per chunk.

    python3 mlx3_kld.py CACHE_PREFIX QUANT OUT.json [TAIL_CUTOFF]

Definitions follow llama.cpp's llama-perplexity --kl-divergence:

  independent chunks, no cache carried between them
  only positions FIRST and later scored, FIRST = ctx/2, so every scored token
    has at least half a context behind it
  the last position of a chunk is dropped, it has no target
  KLD summed over the vocabulary, p taken from the reference
  delta p is the change in the probability of the true next token, in points

TAIL_CUTOFF=1 also applies llama.cpp's `if (p_log_base > -16.f)`. Both numbers
are printed so the size of that difference is visible rather than argued about.
"""
import json, math, os, sys, time
import mlx.core as mx
import numpy as np
from mlx_lm import load

cache, qnt_path, out_json = sys.argv[1:4]
tail = int(sys.argv[4]) if len(sys.argv) > 4 else 1
# Screening mode. A layout sweep needs a ranking, not a publishable number,
# and six of the twenty four chunks give the same ordering in a quarter of the
# time. Anything that goes in a table or a card is measured on all of them.
quick = int(sys.argv[5]) if len(sys.argv) > 5 else 0

meta = json.load(open(cache + ".json"))
ctx, first, step = meta["ctx"], meta["first"], meta["step"]
vocab, ids, n_chunk = meta["vocab"], meta["ids"], meta["chunks"]
if quick and quick < n_chunk:
    n_chunk = quick
mm = np.memmap(cache + ".f16", dtype=np.float16, mode="r",
               shape=(meta["rows"], vocab))

print("reference : %s (cached)" % meta["reference"], flush=True)
print("quant     : %s" % qnt_path, flush=True)
print("context   : %d, from %d, %d chunks, tail cutoff %s"
      % (ctx, first, n_chunk, "on" if tail else "off"), flush=True)
if quick:
    print("SCREENING on %d chunks, not for publication" % n_chunk, flush=True)

model, _ = load(qnt_path)

klds, klds_full, dp = [], [], []
nll_ref = nll_qnt = 0.0
top1 = scored = row = 0
t0 = time.time()

for c in range(n_chunk):
    chunk = ids[c * ctx:(c + 1) * ctx]
    o = model(mx.array([chunk]))
    lq_full = (o.logits if hasattr(o, "logits") else o)[0]
    mx.eval(lq_full)

    for s in range(first, ctx - 1, step):
        e = min(s + step, ctx - 1)
        n = e - s
        a = mx.array(np.ascontiguousarray(mm[row:row + n])).astype(mx.float32)
        b = lq_full[s:e].astype(mx.float32)
        logp = a - mx.logsumexp(a, axis=-1, keepdims=True)
        logq = b - mx.logsumexp(b, axis=-1, keepdims=True)
        term = mx.exp(logp) * (logp - logq)
        k_full = mx.sum(term, axis=-1)
        k_cut = mx.sum(mx.where(logp > -16.0, term, 0.0), axis=-1)
        hits = mx.sum(mx.argmax(a, axis=-1) == mx.argmax(b, axis=-1))
        tgt = mx.array(chunk[s + 1:e + 1])
        idx = mx.arange(n)
        lp_true, lq_true = logp[idx, tgt], logq[idx, tgt]
        d = (mx.exp(lq_true) - mx.exp(lp_true)) * 100.0

        mx.eval(k_full, k_cut, hits, lp_true, lq_true, d)
        klds_full.extend(np.asarray(k_full).tolist())
        klds.extend(np.asarray(k_cut if tail else k_full).tolist())
        dp.extend(np.asarray(d).tolist())
        nll_ref += -float(np.asarray(lp_true).sum())
        nll_qnt += -float(np.asarray(lq_true).sum())
        top1 += int(hits)
        scored += n
        row += n

    del lq_full, o
    if (c + 1) % 4 == 0 or c == n_chunk - 1:
        el = time.time() - t0
        eta = (n_chunk - c - 1) * el / (c + 1)
        print("  chunk %d/%d  %.1f s each  eta %dm%02ds"
              % (c + 1, n_chunk, el / (c + 1), eta // 60, eta % 60), flush=True)

klds.sort()
q = lambda f: klds[min(len(klds) - 1, int(len(klds) * f))]
rms = lambda v: math.sqrt(sum(x * x for x in v) / len(v))
res = {"reference": meta["reference"], "quant": qnt_path, "context": ctx,
       "score_from": first, "chunks": n_chunk, "tokens_scored": scored,
       "screening": bool(quick),
       "tail_cutoff": bool(tail),
       "reference_ppl": math.exp(nll_ref / scored),
       "quant_ppl": math.exp(nll_qnt / scored),
       "mean_kld": sum(klds) / len(klds),
       "mean_kld_whole_vocab": sum(klds_full) / len(klds_full),
       "median_kld": q(.50), "p90_kld": q(.90), "p95_kld": q(.95),
       "p99_kld": q(.99), "max_kld": klds[-1],
       "top1_agree_pct": 100.0 * top1 / scored,
       "mean_delta_p_points": sum(dp) / len(dp),
       "rms_delta_p_points": rms(dp)}
json.dump(res, open(out_json, "w"), indent=2)
print()
print("mean KLD      : %.6f   (whole vocabulary: %.6f)"
      % (res["mean_kld"], res["mean_kld_whole_vocab"]))
print("median KLD    : %.6f" % res["median_kld"])
print("90/95/99      : %.6f  %.6f  %.6f"
      % (res["p90_kld"], res["p95_kld"], res["p99_kld"]))
print("same top-1    : %.3f %%" % res["top1_agree_pct"])
print("quant ppl     : %.4f  (reference %.4f)"
      % (res["quant_ppl"], res["reference_ppl"]))
print("delta p pts   : mean %.4f  rms %.4f"
      % (res["mean_delta_p_points"], res["rms_delta_p_points"]))
print("written to %s" % out_json)

# The MLX CUDA backend throws from a destructor at interpreter shutdown:
#   terminate called ... Destroy(handle_) failed: driver shutting down
# It happens after all the work is done, but it aborts the process and the
# shell then sees a non zero exit code. That cost one finished 13.68 GB
# checkpoint, deleted by a cleanup branch that trusted the code. Leaving here
# without unwinding skips the destructors entirely.
sys.stdout.flush()
sys.stderr.flush()
os._exit(0)

K3EOF

cat > /mlx3_clip.py << 'CLIPEOF'
"""Rebuild a quantized MLX checkpoint with a searched clipping range.

    python3 mlx3_clip.py TEMPLATE BF16 OUT "1.000 0.995 ..."

The built in quantizer stretches the available levels between the smallest and
the largest weight of each group, so one outlier costs the whole group
resolution. This tries narrower ranges, keeps the one with the smallest squared
error per group, and hands the clipped weights back to mx.quantize, so all the
packing is done by the library and the output is bit compatible.

Nothing else changes: same tensors, shapes, bit counts, group sizes, shards,
config, tokenizer, vision tower. Bits and group size come from the template's
own array shapes:

    group size = input features / scales per row
    bit count  = 32 * packed columns / input features
"""
import glob, json, os, shutil, struct, sys, time

import mlx.core as mx

tmpl, ref, out, alpha_str = sys.argv[1:5]
alphas = [float(a) for a in alpha_str.split()]
if 1.0 not in alphas:
    alphas.insert(0, 1.0)


def st_keys(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        head = json.loads(f.read(n))
    return [k for k in head if k != "__metadata__"]


ref_files = sorted(glob.glob(os.path.join(ref, "*.safetensors")))
ref_map = {}
for f in ref_files:
    for k in st_keys(f):
        ref_map[k] = f
print("reference: %d tensors in %d shards" % (len(ref_map), len(ref_files)), flush=True)

# Optional. Without it the search minimizes plain squared error on the weights,
# which treats every input channel as equally used. What reaches the next layer
# is the error weighted by how large that channel's activations are, and on
# synthetic weights searching against the weighted objective roughly doubles
# the reduction. Drop a numpy archive here, one array per tensor base name,
# each of length input_features. The importance matrix from the GGUF build
# holds exactly these numbers under llama.cpp's tensor names.
imp = {}
imp_path = "/mlx/importance.npz"
if os.path.exists(imp_path):
    import numpy as np
    z = np.load(imp_path)
    imp = {k: mx.array(z[k].astype("float32")) for k in z.files}
    print("importance vectors: %d, objective is activation weighted" % len(imp),
          flush=True)
else:
    print("no %s, objective is plain squared error on the weights" % imp_path,
          flush=True)

cache_path = None
cache = {}


def ref_tensor(name):
    global cache_path, cache
    f = ref_map.get(name)
    if f is None:
        return None
    if f != cache_path:
        cache = mx.load(f)
        cache_path = f
    return cache.get(name)


os.makedirs(out, exist_ok=True)
for f in sorted(glob.glob(os.path.join(tmpl, "*"))):
    if f.endswith(".safetensors"):
        continue
    if os.path.isfile(f):
        shutil.copy2(f, out)
print("copied config, tokenizer and index from the template", flush=True)

shards = sorted(glob.glob(os.path.join(tmpl, "*.safetensors")))
n_tensors = 0
for s in shards:
    for k in st_keys(s):
        if k.endswith(".scales") and (k[:-7] + ".weight") in ref_map:
            n_tensors += 1
print("shards: %d, tensors to rebuild: %d" % (len(shards), n_tensors), flush=True)
print(flush=True)

done = 0
elems_done = 0
err_base_sum = 0.0
err_best_sum = 0.0
skipped = []
t0 = time.time()

for s in shards:
    name = os.path.basename(s)
    try:
        data, meta = mx.load(s, return_metadata=True)
    except TypeError:
        data, meta = mx.load(s), {"format": "mlx"}
    if not meta:
        meta = {"format": "mlx"}

    for base in sorted(k[:-7] for k in data if k.endswith(".scales")):
        wq_t = data[base + ".weight"]
        sc_t = data.get(base + ".scales")
        bi_t = data.get(base + ".biases")
        if bi_t is None:
            skipped.append((base, "no biases, not an affine tensor"))
            continue
        w = ref_tensor(base + ".weight")
        if w is None:
            skipped.append((base, "no reference weight under this name"))
            continue
        if w.ndim != 2:
            skipped.append((base, "not a matrix"))
            continue

        rows, cols = int(w.shape[0]), int(w.shape[1])
        group = cols // int(sc_t.shape[1])
        bits = int(round(32.0 * int(wq_t.shape[1]) / cols))
        if group not in (32, 64, 128) or cols % group != 0:
            skipped.append((base, "derived group %d, refusing" % group))
            continue
        if bits not in (2, 3, 4, 5, 6, 8):
            skipped.append((base, "derived %d bits, refusing" % bits))
            continue

        w32 = w.astype(mx.float32)
        g = w32.reshape(-1, group)
        lo = mx.min(g, axis=1, keepdims=True)
        hi = mx.max(g, axis=1, keepdims=True)
        mid = (lo + hi) / 2.0
        half = (hi - lo) / 2.0

        iv = imp.get(base)
        if iv is not None and int(iv.shape[0]) == cols:
            wgt = mx.broadcast_to(iv.reshape(1, cols), (rows, cols)).reshape(-1, group)
        else:
            wgt = None

        best_err = None
        best_a = None
        err_one = None
        for a in alphas:
            av = mx.array(a, dtype=mx.float32)
            clipped = mx.clip(g, mid - half * av, mid + half * av)
            cw = clipped.reshape(rows, cols).astype(w.dtype)
            q, sc, bi = mx.quantize(cw, group_size=group, bits=bits)
            dq = mx.dequantize(q, sc, bi, group_size=group, bits=bits)
            diff = (dq.astype(mx.float32) - w32).reshape(-1, group)
            sq = diff * diff
            if wgt is not None:
                sq = sq * wgt
            err = mx.sum(sq, axis=1, keepdims=True)
            mx.eval(err)
            if a == 1.0:
                err_one = err
            if best_err is None:
                best_err = err
                best_a = mx.full(err.shape, a, dtype=mx.float32)
            else:
                take = err < best_err
                best_err = mx.where(take, err, best_err)
                best_a = mx.where(take, av, best_a)
            del clipped, cw, q, sc, bi, dq, diff, sq

        clipped = mx.clip(g, mid - half * best_a, mid + half * best_a)
        cw = clipped.reshape(rows, cols).astype(w.dtype)
        q, sc, bi = mx.quantize(cw, group_size=group, bits=bits)
        mx.eval(q, sc, bi)

        if q.shape != wq_t.shape or sc.shape != sc_t.shape or bi.shape != bi_t.shape:
            print("SHAPE MISMATCH on %s" % base)
            print("  packed %s vs template %s" % (q.shape, wq_t.shape))
            print("  scales %s vs template %s" % (sc.shape, sc_t.shape))
            print("  derived bits %d group %d" % (bits, group))
            sys.exit(1)

        data[base + ".weight"] = q.astype(wq_t.dtype)
        data[base + ".scales"] = sc.astype(sc_t.dtype)
        data[base + ".biases"] = bi.astype(bi_t.dtype)

        e_one = float(mx.sum(err_one))
        e_best = float(mx.sum(best_err))
        a_mean = float(mx.mean(best_a))
        err_base_sum += e_one
        err_best_sum += e_best
        done += 1
        elems_done += rows * cols

        el = time.time() - t0
        eta = 0 if done == 0 else (n_tensors - done) * el / done
        drop = 0.0 if e_one == 0 else 100.0 * (e_best - e_one) / e_one
        print("  [%3d/%3d] %-44s %5dx%-5d b%d g%-3d err %+6.2f %% a=%.3f eta %dm%02ds"
              % (done, n_tensors, base[-44:], rows, cols, bits, group,
                 drop, a_mean, eta // 60, eta % 60), flush=True)

        del w32, g, lo, hi, mid, half, best_err, best_a, err_one
        del clipped, cw, q, sc, bi

    mx.save_safetensors(os.path.join(out, name), data, metadata=meta)
    print("  wrote %s" % name, flush=True)
    del data

print(flush=True)
if skipped:
    print("left exactly as they were, %d:" % len(skipped))
    for b, why in skipped[:20]:
        print("   %-52s %s" % (b[-52:], why))
    print(flush=True)

total = 0.0 if err_base_sum == 0 else 100.0 * (err_best_sum - err_base_sum) / err_base_sum
print("rebuilt      : %d tensors, %.2f B weights" % (done, elems_done / 1e9))
print("squared error: %+.3f percent against plain min max rounding" % total)
print()
if total > -2.0:
    print("Under two percent. At this bit count the plain range is already close")
    print("to the best one and the divergence will barely move. Put the hours")
    print("into distillation instead.")
else:
    print("The weights are measurably closer to the originals. Whether that")
    print("survives sixty four layers of composition is what mlx3_kld answers,")
    print("and nothing before it does.")


# The MLX CUDA backend throws from a destructor at interpreter shutdown:
#   terminate called ... Destroy(handle_) failed: driver shutting down
# It happens after all the work is done, but it aborts the process and the
# shell then sees a non zero exit code. That cost one finished 13.68 GB
# checkpoint, deleted by a cleanup branch that trusted the code. Leaving here
# without unwinding skips the destructors entirely.
sys.stdout.flush()
sys.stderr.flush()
os._exit(0)

CLIPEOF


cat > /mlx3_layout.py << 'LAYEOF'
"""Build a quantized MLX checkpoint from a per tensor rule set.

    python3 mlx3_layout.py price BF16_DIR RULES
    python3 mlx3_layout.py build BF16_DIR RULES OUT "1.000 0.995 ..."

A rule line is a regex, a bit count and a group size, or the word skip. The
first rule whose regex matches the tensor name wins, so the catch all goes
last. Anything skipped, and anything outside the language model, stays in
bf16 and is recorded as unquantized in the config so the loader does not try
to dequantize it.

Numbers come out of the same clipping search mlx3_clip uses: for each group of
weights, try several narrower ranges, keep the one with the smallest squared
error, and let mx.quantize do the packing.
"""
import glob
import json
import os
import re
import shutil
import struct
import sys
import time

import mlx.core as mx

SHARD_BYTES = 4_500_000_000


def st_info(path):
    """Names, shapes and dtypes from the safetensors header, no data read."""
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        head = json.loads(f.read(n))
    out = {}
    for k, v in head.items():
        if k == "__metadata__":
            continue
        out[k] = (tuple(v["shape"]), v["dtype"])
    return out


def read_rules(path):
    rules = []
    for line in open(path, encoding="utf-8"):
        line = line.split("#")[0].strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) == 2 and parts[1].lower() == "skip":
            rules.append((re.compile(parts[0]), None, None))
        elif len(parts) == 3:
            rules.append((re.compile(parts[0]), int(parts[1]), int(parts[2])))
        else:
            sys.exit("cannot read this rule line: %r" % line)
    if not rules:
        sys.exit("no rules in %s" % path)
    return rules


def decide(name, shape, rules):
    """(bits, group) or None for a tensor that stays in bf16."""
    if len(shape) != 2 or not name.endswith(".weight"):
        return None
    if not name.startswith("language_model."):
        return None                      # the vision tower is never touched
    for rx, bits, group in rules:
        if rx.search(name):
            if bits is None:
                return None
            if shape[1] % group != 0:
                return None
            return (bits, group)
    return None


def role_of(name):
    r = re.sub(r"layers\.\d+\.", "", name)
    r = r.replace("language_model.model.", "").replace("language_model.", "")
    return r.replace(".weight", "")


def scan(ref, rules):
    """Every tensor with its shape and its decision, in file order."""
    files = sorted(glob.glob(os.path.join(ref, "*.safetensors")))
    items = []
    for f in files:
        for name, (shape, dtype) in st_info(f).items():
            items.append((name, shape, f, decide(name, shape, rules)))
    return items


def price(ref, rules):
    items = scan(ref, rules)
    roles = {}
    total_bits = 0
    quantized = 0
    for name, shape, _f, d in items:
        n = 1
        for x in shape:
            n *= x
        bpw = 16.0 if d is None else d[0] + 32.0 / d[1]
        total_bits += n * bpw
        if d is not None:
            quantized += 1
        key = role_of(name)
        e = roles.setdefault(key, {"n": 0, "bits": 0.0, "how": {}})
        e["n"] += n
        e["bits"] += n * bpw
        e["how"][("bf16" if d is None else "%d/%d" % d)] = \
            e["how"].get("bf16" if d is None else "%d/%d" % d, 0) + 1

    params = sum(e["n"] for e in roles.values())
    print()
    print("%-30s %14s %7s  %-18s %8s" %
          ("role", "params", "share", "bits/group", "GB"))
    for k, e in sorted(roles.items(), key=lambda kv: -kv[1]["bits"]):
        how = " ".join("%s x%d" % (a, b) for a, b in sorted(e["how"].items()))
        print("%-30s %14d %6.2f%%  %-18s %8.3f" %
              (k, e["n"], 100.0 * e["n"] / params, how[:18], e["bits"] / 8 / 1e9))
    print("-" * 82)
    gb = total_bits / 8 / 1e9
    print("%-30s %14d %6.2f%%  %-18s %8.3f" %
          ("TOTAL", params, 100.0, "%.3f bpw" % (total_bits / params), gb))
    print()
    print("quantized tensors: %d of %d" % (quantized, len(items)))
    print("what to compare against, measured on one reference:")
    print("  12.69 GB  0.222903  plain 3 bit group 64")
    print("  13.37 GB  0.165901  our mixed_3_4 with clipping and distillation")
    print("  13.70 GB  0.154161  maglun Mixed-3.80bpw")
    print("  16.05 GB  0.053267  our 4bit with clipping and distillation")
    print("  17.57 GB  0.034350  maglun Mixed-4.95bpw")
    return gb


def build(ref, rules, out, alphas):
    items = scan(ref, rules)
    n_q = sum(1 for i in items if i[3] is not None)
    print("tensors: %d, to quantize: %d" % (len(items), n_q), flush=True)

    os.makedirs(out, exist_ok=True)
    for f in sorted(glob.glob(os.path.join(ref, "*"))):
        if f.endswith(".safetensors") or f.endswith(".safetensors.index.json"):
            continue
        if os.path.isfile(f):
            shutil.copy2(f, out)

    cfg = json.load(open(os.path.join(ref, "config.json")))
    quant = {"group_size": 64, "bits": 3, "mode": "affine"}

    cache_path, cache = None, {}
    shard, shard_bytes, shard_no = {}, 0, 1
    weight_map, written = {}, []
    done, err_base, err_best = 0, 0.0, 0.0
    t0 = time.time()

    def flush():
        nonlocal shard, shard_bytes, shard_no
        if not shard:
            return
        name = "model-%05d-of-SHARDS.safetensors" % shard_no
        mx.save_safetensors(os.path.join(out, name), shard,
                            metadata={"format": "mlx"})
        for k in shard:
            weight_map[k] = name
        written.append(name)
        print("  wrote %s, %.2f GB" % (name, shard_bytes / 1e9), flush=True)
        shard, shard_bytes, shard_no = {}, 0, shard_no + 1

    for name, shape, f, d in items:
        if f != cache_path:
            cache = mx.load(f)
            cache_path = f
        w = cache[name]

        if d is None:
            shard[name] = w
            shard_bytes += w.nbytes
            if len(shape) == 2 and name.endswith(".weight"):
                quant[name[: -len(".weight")]] = False
        else:
            bits, group = d
            rows, cols = int(shape[0]), int(shape[1])
            w32 = w.astype(mx.float32)
            g = w32.reshape(-1, group)
            lo = mx.min(g, axis=1, keepdims=True)
            hi = mx.max(g, axis=1, keepdims=True)
            mid = (lo + hi) / 2.0
            half = (hi - lo) / 2.0

            best_err = best_a = err_one = None
            for a in alphas:
                av = mx.array(a, dtype=mx.float32)
                cw = mx.clip(g, mid - half * av, mid + half * av)
                cw = cw.reshape(rows, cols).astype(w.dtype)
                q, sc, bi = mx.quantize(cw, group_size=group, bits=bits)
                dq = mx.dequantize(q, sc, bi, group_size=group, bits=bits)
                diff = (dq.astype(mx.float32) - w32).reshape(-1, group)
                e = mx.sum(diff * diff, axis=1, keepdims=True)
                mx.eval(e)
                if a == 1.0:
                    err_one = e
                if best_err is None:
                    best_err = e
                    best_a = mx.full(e.shape, a, dtype=mx.float32)
                else:
                    take = e < best_err
                    best_err = mx.where(take, e, best_err)
                    best_a = mx.where(take, av, best_a)
                del cw, q, sc, bi, dq, diff, e

            cw = mx.clip(g, mid - half * best_a, mid + half * best_a)
            cw = cw.reshape(rows, cols).astype(w.dtype)
            q, sc, bi = mx.quantize(cw, group_size=group, bits=bits)
            mx.eval(q, sc, bi)

            base = name[: -len(".weight")]
            shard[base + ".weight"] = q
            shard[base + ".scales"] = sc
            shard[base + ".biases"] = bi
            shard_bytes += q.nbytes + sc.nbytes + bi.nbytes
            quant[base] = {"group_size": group, "bits": bits}

            e1, e2 = float(mx.sum(err_one)), float(mx.sum(best_err))
            err_base += e1
            err_best += e2
            done += 1
            el = time.time() - t0
            eta = (n_q - done) * el / done
            print("  [%3d/%3d] %-42s %5dx%-5d b%d g%-3d err %+6.2f %% a=%.3f eta %dm%02ds"
                  % (done, n_q, base[-42:], rows, cols, bits, group,
                     0.0 if e1 == 0 else 100.0 * (e2 - e1) / e1,
                     float(mx.mean(best_a)), eta // 60, eta % 60), flush=True)
            del w32, g, lo, hi, mid, half, best_err, best_a, err_one, cw, q, sc, bi

        if shard_bytes >= SHARD_BYTES:
            flush()
    flush()

    # the shard count is only known at the end, so the names are fixed up here
    total = len(written)
    for i, old in enumerate(written, 1):
        new = "model-%05d-of-%05d.safetensors" % (i, total)
        os.rename(os.path.join(out, old), os.path.join(out, new))
        for k, v in weight_map.items():
            if v == old:
                weight_map[k] = new

    size = sum(os.path.getsize(p) for p in glob.glob(os.path.join(out, "*.safetensors")))
    json.dump({"metadata": {"total_size": size}, "weight_map": weight_map},
              open(os.path.join(out, "model.safetensors.index.json"), "w"), indent=2)

    cfg["quantization"] = quant
    if "text_config" in cfg and isinstance(cfg["text_config"], dict):
        cfg["text_config"]["quantization"] = quant
    json.dump(cfg, open(os.path.join(out, "config.json"), "w"), indent=2)

    print()
    print("rebuilt      : %d tensors in %d shards" % (done, total))
    print("squared error: %+.3f percent against plain min max rounding"
          % (0.0 if err_base == 0 else 100.0 * (err_best - err_base) / err_base))
    print("on disk      : %.2f GB" % (size / 1e9))


def main():
    mode = sys.argv[1]
    if mode == "price":
        price(sys.argv[2], read_rules(sys.argv[3]))
    elif mode == "build":
        alphas = [float(a) for a in sys.argv[5].split()]
        if 1.0 not in alphas:
            alphas.insert(0, 1.0)
        rules = read_rules(sys.argv[3])
        price(sys.argv[2], rules)
        print()
        build(sys.argv[2], rules, sys.argv[4], alphas)
    else:
        sys.exit("price or build")


if __name__ == "__main__":
    main()


# The MLX CUDA backend throws from a destructor at interpreter shutdown:
#   terminate called ... Destroy(handle_) failed: driver shutting down
# It happens after all the work is done, but it aborts the process and the
# shell then sees a non zero exit code. That cost one finished 13.68 GB
# checkpoint, deleted by a cleanup branch that trusted the code. Leaving here
# without unwinding skips the destructors entirely.
sys.stdout.flush()
sys.stderr.flush()
os._exit(0)

LAYEOF

    echo "wrote /mlx3_refcache.py /mlx3_kld.py /mlx3_clip.py /mlx3_layout.py"
}


# ================================================================== orientation

mlx3_plan() {
cat << 'PLANEOF'

SIZE ARITHMETIC

  Fitted on the twenty five measured builds, good to about 0.05 GB:

      GB = 0.92 + 3.365 * bits_per_weight

  0.92 is the vision tower, never quantized. bits_per_weight is the bit count
  plus the cost of the scales: 32 bits of scale and offset over one group.
  Group 128 adds 0.25, group 64 adds 0.5, group 32 adds 1.0.

      3 bit g64   3.50 bpw   12.70 GB   plain 0.222903
      3 bit g32   4.00 bpw   14.38 GB   plain 0.180793
      4 bit g128  4.25 bpw   15.22 GB   never measured by anyone
      4 bit g64   4.50 bpw   16.05 GB   plain 0.055803
      4 bit g32   5.00 bpw   17.75 GB   plain 0.044101
      5 bit g64   5.50 bpw   19.43 GB   plain 0.015532

  One extra level (3 -> 4 bits at the same group) divides divergence by four.
  Half a bit spent on twice as many scales (group 64 -> 32) divides it by 1.27.
  Levels are worth about three times what scales are worth per unit of size.
  That is why every mixed layout in the survey sits on the plain line: they all
  trade the expensive thing for the cheap one. Seven measurements of layout did
  not fail to find an answer, they found this one.

TARGETS

  13.70 GB   0.154161   maglun Mixed-3.80bpw
  16.05 GB   0.053775   WaveCut 4bit-DWQ
  17.57 GB   0.034350   maglun Mixed-4.95bpw

THE TWO METHODS AND THE EXPECTED NUMBERS

  Neither touches the bit layout. One changes which integer each weight lands
  on, the other changes the scales around those integers, so they compose.

  ladder                now        after clip      after clip + dwq   target
  3 bit g64, 12.70 GB   0.222903   0.180 (-21 %)   0.108 (-40 %)      0.154161 @ 13.70
  4 bit g64, 16.05 GB   0.055803   0.051 (-10 %)   0.043 (-15 %)      0.053775 @ 16.05

  Where the percentages come from:

  clip, 21 and 10 percent
    a numerical search on synthetic weights at group 64: gaussian, gaussian
    with 0.25 percent outliers, and student-t. Squared error on the WEIGHTS
    falls 21 percent at 3 bits, 10 at 4, 6 at 5, with the best kept fraction
    around 0.86 and 0.96. KLD is quadratic in the weight perturbation to first
    order, so that is a ceiling for the divergence, not a promise.

  dwq, 40 percent at 3 bits
    two published runs on other models at these exact settings, mlx-lm 0.28.4,
    rate 3e-7, batch 1, about 1.09 M tokens: validation loss 0.146 -> 0.088 and
    0.168 -> 0.089. That loss IS the divergence against the teacher, the same
    quantity this table measures, not a proxy for it.

  dwq, 15 percent at 4 bits
    no direct measurement. WaveCut reports 3.6 percent, which cannot be what
    the method does: damage at 3 bits is four times larger and the gain there
    is 40 percent, so a smaller gain at 3 bits is impossible and his run was
    undertrained. Range 10 to 25, midpoint taken. If the measurement lands near
    3 percent, four bits has nothing left to repair and every remaining hour
    goes to the three bit lane.

  A win at a SMALLER size is a cleaner claim than a win at the same size, and
  both lanes above are aimed that way.

BEFORE PUBLISHING

  Measure the winner on eval_agentic as well. The calibration text and the eval
  text are different splits of one corpus pipeline. That is a legitimate held
  out measurement and it is also our own distribution, while the competitors
  calibrated on someone else's. If the relative gain holds within a few points
  on the second corpus it is real. If it does not, the card says the gain was
  measured on our own calibration.

PLANEOF
}

mlx3_box() {
    case "${1:-}" in
    hub)
cat << 'BOXHEOF'

BOX 1, the hub. RTX PRO 6000 if the Blackwell check passed, otherwise an H200.
Rent this one FIRST. Everything it does is bound by disk and by a single
forward pass, so the cheap card costs minutes, not results.

Job: build the 3 bit rung, re-round both rungs, publish the bases the second
box needs, settle what counts as a result, measure what clipping alone bought.

  export HF_TOKEN=hf_...
  curl -sL https://raw.githubusercontent.com/AtomicBot-ai/atomic-quantizer/main/foundry-mlx3.sh -o /mlx3.sh
  source /mlx3.sh
  mlx3_setup
  mlx3_check                      <- READ THIS BEFORE ANYTHING ELSE

  Tens of thousands of GFLOP/s means the CUDA backend has kernels for this
  card and the B200 is safe too. A few hundred means it fell back to the
  processor: kill the instance and run the whole plan on two H200 instead.
  H200 is sm_90 and always works. Both Blackwell parts are new silicon
  against the youngest part of the MLX stack.

  mlx3_persist
  mlx3_get src
  mlx3_get ref
  mlx3_get eval
  mlx3_get base 4bit
  mlx3_verify
  mlx3_quant 3 64
  mlx3_clip /mlx/Qwen3.8-27B-MLX-3bit
  mlx3_push /mlx/Qwen3.8-27B-MLX-3bit-CLIP 3bit-CLIP
  mlx3_clip /mlx/Qwen3.8-27B-MLX-4bit
  mlx3_push /mlx/Qwen3.8-27B-MLX-4bit-CLIP 4bit-CLIP
  mlx3_cache
  mlx3_repeat /mlx/Qwen3.8-27B-MLX-4bit
  mlx3_kld /mlx/Qwen3.8-27B-MLX-3bit-CLIP
  mlx3_kld /mlx/Qwen3.8-27B-MLX-4bit-CLIP
  mlx3_table

The order is deliberate. The 3 bit base is published first because it is the
main bet and box 2 is waiting on it. The reference cache and the measurements
come after, because nothing is waiting on them.

Checkpoints:

  mlx3_check    tens of thousands of GFLOP/s, or move to H200
  mlx3_clip     the printed squared error drop, before any measurement. Under
                two percent at 4 bits means clipping has nothing there and the
                4 bit lane on box 2 is not worth its minutes
  mlx3_repeat   the two passes agree to every digit, or you now have a noise
                floor and it belongs next to every number in the card
  mlx3_kld      4bit-CLIP under 0.053775 at exactly 16.05 GB
                3bit-CLIP under 0.154161 at 12.70 GB, a full gigabyte lighter
                than the build it beats

BOXHEOF
        ;;
    dwq)
cat << 'BOXDEOF'

BOX 2, distillation. B200 if the Blackwell check on box 1 passed, otherwise a
second H200. The trigger is a command, not a clock: rent this box the moment
box 1 starts mlx3_quant 3 64. From there it is twelve minutes of building, six
of clipping and four of pushing before the base exists, which is exactly how
long this box needs for its own setup, its 82 GB and its reference cache.
Renting it earlier is paying for an idle card.

Filter the offer by driver version first. The mlx[cuda13] wheel needs 580 or
newer, and a B200 on an older host will not start at all.

One box runs BOTH lanes one after the other. On a B200 a 600 step run is about
eight minutes, so two lanes plus their measurements fit in well under an hour,
which is cheaper than two expensive boxes in parallel.

  export HF_TOKEN=hf_...
  curl -sL https://raw.githubusercontent.com/AtomicBot-ai/atomic-quantizer/main/foundry-mlx3.sh -o /mlx3.sh
  source /mlx3.sh
  mlx3_setup
  mlx3_check
  mlx3_persist
  mlx3_get ref
  mlx3_get teacher
  mlx3_get eval
  mlx3_get calib
  mlx3_cache
  mlx3_data
  mlx3_dwq_help
  mlx3_wait 3bit-CLIP             <- polls the hub, pulls it when it lands

Then the low end lane, which is the main bet:

  mlx3_mem 3 64 2048
  mlx3_dwq 3 64 /mlx/Qwen3.8-27B-MLX-3bit-CLIP
  mlx3_kld /mlx/Qwen3.8-27B-MLX-3bit-CLIP-DWQ
  mlx3_table

Then the 4 bit lane, unless box 1 reported under two percent from clipping
there:

  mlx3_wait 4bit-CLIP
  mlx3_dwq 4 64 /mlx/Qwen3.8-27B-MLX-4bit-CLIP
  mlx3_kld /mlx/Qwen3.8-27B-MLX-4bit-CLIP-DWQ
  mlx3_table

If there are minutes left, the group 32 variant. Twice as many tunable scales,
14.38 GB, which is above the 13.70 target but lands in the size class where
the best foreign build is 0.180793. It needs 83 GB, so a B200 or an H200 only:

  mlx3_get src
  mlx3_quant 3 32
  mlx3_clip /mlx/Qwen3.8-27B-MLX-3bit-g32
  mlx3_dwq 3 32 /mlx/Qwen3.8-27B-MLX-3bit-g32-CLIP
  mlx3_kld /mlx/Qwen3.8-27B-MLX-3bit-g32-CLIP-DWQ

IF THIS BOX NEVER HAPPENS. A 94 GB card runs the whole plan on its own: the
3 bit lane needs 67.3 GB and the 4 bit lane 70.7. On the hub, add mlx3_get
teacher and run both lanes there one after the other, seventeen minutes each
instead of eight. The second box is a speedup, not a requirement.

Checkpoints:

  mlx3_dwq      validation loss falls by more than a third over the run. Under
                15 percent means the rate is wrong, not the method: halve it
                if the loss oscillates, triple it if it crawls
  mlx3_kld      under 0.140 at 12.70 GB beats the 13.70 GB target and is a
                gigabyte lighter as well

BOXDEOF
        ;;
    *)
        echo "mlx3_box hub | dwq"
        echo "  hub  build, clip, verify, measure     rent first, cheap card"
        echo "  dwq  both distillation lanes          rent 15 minutes later"
        echo
        echo "cards, in order of what this workload actually needs:"
        echo "  B200            2.25 PFLOPS bf16, 180 GB   both lanes, ~8 min a run"
        echo "  H200 SXM         990 TFLOPS,      141 GB   ~17 min a run"
        echo "  H200 NVL         835 TFLOPS,      141 GB   no reason to prefer it"
        echo "  RTX PRO 6000  200-250 TFLOPS,      96 GB   the hub, half the price"
        ;;
    esac
}

mlx3_help() {
cat << 'H3EOF'

SETUP     mlx3_setup | mlx3_check | mlx3_bench | mlx3_persist | mlx3_disk
DOWNLOAD  mlx3_get src | ref | teacher | eval | calib | base LABEL
MEASURE   mlx3_cache | mlx3_kld DIR | mlx3_all | mlx3_table
          mlx3_diff A B                 two builds across every metric
HANDOFF   mlx3_wait LABEL               poll the hub for a base, then pull it
CHECKS    mlx3_verify | mlx3_fp_remote | mlx3_fingerprint DIR
          mlx3_repeat DIR | mlx3_tail
LAYOUT    mlx3_gen TARGET_GB             a rule set that lands on a size
          mlx3_try LABEL                 build and rank on six chunks
          mlx3_rules                     the hand written default
          mlx3_price [RULES]             what it costs, before building
          mlx3_build LABEL [RULES]       build it from bf16 with clipping
BUILD     mlx3_quant BITS [GROUP]        plain rung, needs /src
          mlx3_clip TEMPLATE [SUFFIX]    better rounding, identical size
DISTILL   mlx3_mem BITS GROUP SEQ [full|targets]
          mlx3_targets [SEQ] [SAMPLES]  teacher logits once, serves both lanes
          mlx3_data | mlx3_dwq_help
          mlx3_dwq BITS GROUP BASE [TEACHER] [SEQ] [LR] [SAMPLES]
VISION    mlx3_reattach TEXT VISION OUT
PUBLISH   mlx3_push DIR LABEL
PLAN      mlx3_plan | mlx3_box hub | mlx3_box dwq

FROM NOTHING, ON A FRESH BOX

  export HF_TOKEN=hf_...
  curl -sL https://raw.githubusercontent.com/AtomicBot-ai/atomic-quantizer/main/foundry-mlx3.sh -o /mlx3.sh
  source /mlx3.sh
  mlx3_setup
  mlx3_persist
  mlx3_box hub          then paste its lines one at a time

H3EOF
    echo "  running $MLX3_VERSION"
}

echo "foundry-mlx3 $MLX3_VERSION loaded. Start with:  mlx3_help"
