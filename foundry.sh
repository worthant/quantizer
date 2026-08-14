#!/bin/bash
# Foundry toolbox. Source it, then call one function at a time, in the foreground.
#
#   source /foundry.sh
#   use_nemotron
#   status          <- always tells you what to do next
#   menu            <- pick a command by number instead of typing it
#
# Nothing runs in the background. Every function says what it is doing,
# checks its inputs, and stops loudly when something is missing.

export HF_XET_HIGH_PERFORMANCE=1
export HF_HOME=/hf

BIN=/llama.cpp/build/bin
EVAL=/eval/eval_neutral.txt
BASE=/kld/base.kld
CTX=4096


# ================================================================== presets

use_nemotron() {
    MAIN=AtomicChat/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-GGUF
    METRICS=AtomicChat/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-GGUF-metrics
    METRICS_KIND=dataset
    show_preset
}

use_qwen() {
    MAIN=AtomicChat/Qwen3.8-27B-GGUF
    METRICS=AtomicChat/Qwen3.8-27B-GGUF-metrics
    METRICS_KIND=dataset
    show_preset
}

show_preset() {
    echo "main repo    : $MAIN"
    echo "metrics repo : $METRICS  (type: $METRICS_KIND)"
    echo "eval corpus  : $EVAL"
    echo "context      : $CTX"
    echo "bf16 file is found on disk by find_bf16, nothing is hardcoded"
}

need_preset() {
    if [ -z "$MAIN" ]; then
        echo "No preset loaded. Run use_nemotron or use_qwen first."
        return 1
    fi
}

ask() {
    read -p "$1 [y/N] " answer
    if [ "$answer" = "y" ] || [ "$answer" = "Y" ]; then
        return 0
    fi
    echo "skipped"
    return 1
}


# ================================================================== status

mark() {
    if [ -e "$2" ]; then
        echo "  [x] $1"
    else
        echo "  [ ] $1        ->  run:  $3"
    fi
}

status() {
    echo
    if [ -z "$MAIN" ]; then
        echo "  [ ] preset loaded      ->  run:  use_nemotron"
    else
        echo "  [x] preset: $MAIN"
    fi
    mark "cuda checked"       /logs/env.txt              "cuda_check"
    mark "tools installed"    /usr/bin/ninja             "setup"
    mark "llama.cpp built"    $BIN/llama-perplexity      "build 120"
    mark "eval corpus"        $EVAL                      "get_eval"

    if find /gguf -name "*.gguf" 2>/dev/null | grep -qi bf16; then
        echo "  [x] bf16 on disk"
    else
        echo "  [ ] bf16 on disk       ->  run:  get_bf16      (box 1 only)"
    fi

    if ls /gguf/*.gguf >/dev/null 2>&1; then
        echo "  [x] quants on disk: $(ls /gguf/*.gguf | wc -l) files"
    else
        echo "  [ ] quants on disk     ->  run:  get_quants    (box 2 only)"
    fi

    mark "reference built"    $BASE                      "base   (box 1)  or  get_base / wait_base   (box 2)"

    KLDS=$(ls /logs/kld-*.log 2>/dev/null | wc -l)
    echo "  [$([ $KLDS -gt 0 ] && echo x || echo ' ')] kld logs: $KLDS       ->  run:  kld_all"
    BENCHES=$(ls /logs/bench-*.json 2>/dev/null | wc -l)
    echo "  [$([ $BENCHES -gt 0 ] && echo x || echo ' ')] bench logs: $BENCHES     ->  run:  bench_all"
    mark "results.json"       /logs/results.json         "results"
    echo
    echo "  full command list: help_me      pick by number: menu"
    echo
}

menu() {
    PS3="
number: "
    select picked in \
        "status" "cuda_check" "fix_cuda_compat" "gpu_test" "setup" "build 120" \
        "get_eval" "get_bf16" "get_quants" "find_bf16" \
        "base" "get_base" "wait_base" "kld_all" "bench_all" "results" \
        "push_base" "push_logs" "push_results" \
        "quit"
    do
        if [ "$picked" = "quit" ] || [ -z "$picked" ]; then
            break
        fi
        echo "--> $picked"
        eval "$picked"
        break
    done
}

help_me() {
cat << 'HELP_EOF'

PRESET
  use_nemotron / use_qwen    load repo names
  show_preset

ORIENTATION
  status                     what is done, what is next
  menu                       pick a command by number
  help_me                    this list

BOX SETUP
  check                      disk, GPUs, cores, RAM
  cuda_check                 toolkit vs host driver, writes /logs/env.txt
  fix_cuda_compat            drop compat libs when the host driver is newer
  setup                      apt + hugging face cli
  build 120                  compile. 120 blackwell, 90 hopper, 89 ada, 86 ampere
  gpu_test                   does ggml actually see the GPUs

DOWNLOAD, each one asks before pulling anything
  get_eval                   the held-out corpus
  get_bf16                   the reference weights, sharded or not
  get_quants                 every published quant, no bf16, no sidecars
  get_one "PATTERN"          one file by glob
  get_base                   pull an existing reference from the metrics repo
  find_bf16                  locate the bf16 file already on disk

MEASURE
  base                       write the reference into /kld/base.kld
  kld MODEL                  one quant against the reference
  kld_all                    every .gguf in /gguf
  bench MODEL                llama-bench, GPU 0 only, json out
  bench_all
  gen MODEL NGL "EXTRA"      llama-cli, 400 tokens, fixed seed

RESULTS
  results                    parse all logs into /logs/results.json

UPLOAD, one job each
  push_base                  reference + its log   -> metrics repo
  push_logs                  everything in /logs   -> metrics repo
  push_results               results.json          -> metrics repo
  push_model FILE            one gguf              -> main repo
  push_model_split FILE      split at 45 GB first  -> main repo
  push_card FILE             a README.md           -> main repo

FLOWS, these just call the steps above in order
  run_base_box               box 1: eval, bf16, reference, upload
  run_quant_box              box 2: eval, quants, wait for reference, measure

HELP_EOF
}


# ================================================================== box setup

check() {
    echo "=== free space on / ==="
    df -h /
    echo
    echo "=== GPUs ==="
    nvidia-smi --query-gpu=index,name,memory.total --format=csv
    echo
    echo "=== cores / RAM ==="
    nproc
    free -g | head -2
    echo
    echo "bf16 needs ~96 GB VRAM. Q8_0 needs ~40."
}

cuda_check() {
    mkdir -p /logs
    echo "=== toolkit in the container ==="
    nvcc --version | tail -2
    echo
    echo "=== host driver ==="
    nvidia-smi --query-gpu=name,driver_version --format=csv
    echo
    echo "=== forward compatibility libraries ==="
    if ls -d /usr/local/cuda*/compat 2>/dev/null; then
        echo
        echo "These exist for hosts whose driver is OLDER than the image target."
        echo "When the host driver is NEWER they break CUDA init with error 803."
        echo "Compare the two versions above. If the driver is newer, run fix_cuda_compat."
    else
        echo "none, nothing to worry about"
    fi

    {
        grep PRETTY_NAME /etc/os-release
        nvcc --version | tail -2
        nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
    } > /logs/env.txt
    echo
    echo "written to /logs/env.txt, it ships to the metrics repo with push_logs"
}

fix_cuda_compat() {
    rm -rf /usr/local/cuda*/compat
    ldconfig
    echo "Compat libraries removed. The host driver now wins."
}

setup() {
    apt-get update -qq
    apt-get install -y -qq build-essential cmake ninja-build git curl ccache \
        libcurl4-openssl-dev libssl-dev python3-pip
    pip install --break-system-packages -q -U "huggingface_hub[hf_xet]"
    mkdir -p /gguf /kld /eval /logs /hf
    echo
    hf version
    python3 --version
    echo "Missing something? Add it to this function, not to the server template."
}

build() {
    if [ -z "$1" ]; then
        echo "Pass the CUDA arch. build 120"
        return 1
    fi
    cd /
    git clone https://github.com/ggml-org/llama.cpp
    cd /llama.cpp
    git rev-parse --short HEAD > /logs/llama-commit.txt
    cat /logs/llama-commit.txt

    cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Release \
        -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=$1
    cmake --build build -j $(nproc) --target llama-perplexity llama-bench llama-cli llama-gguf-split

    ls -la $BIN/llama-perplexity $BIN/llama-bench $BIN/llama-cli $BIN/llama-gguf-split
    echo
    echo "Now run gpu_test."
}

gpu_test() {
    if [ ! -x $BIN/llama-cli ]; then
        echo "not built yet. Run build 120."
        return 1
    fi
    $BIN/llama-cli --list-devices
    echo
    echo "Every GPU should be listed above. If the list is empty or short,"
    echo "run fix_cuda_compat and try again."
}


# ================================================================== download

get_eval() {
    if [ -f $EVAL ]; then
        echo "already here:"
        wc -c $EVAL
        return 0
    fi
    ask "download the eval corpus?" || return 1
    hf download AtomicChat/calib-corpora --repo-type dataset \
        --include "eval_neutral.txt" --local-dir /eval
    if [ ! -f $EVAL ]; then
        find /eval -name "eval_neutral.txt"
        echo "Not at $EVAL. Move it there."
        return 1
    fi
    wc -c $EVAL
}

find_bf16() {
    FOUND=$(find /gguf -name "*.gguf" 2>/dev/null | grep -i bf16 | grep "00001-of-" | head -1)
    if [ -z "$FOUND" ]; then
        FOUND=$(find /gguf -name "*.gguf" 2>/dev/null | grep -i bf16 | head -1)
    fi
    if [ -z "$FOUND" ]; then
        echo "no bf16 in /gguf. Run get_bf16."
        return 1
    fi
    BF16_FIRST=$FOUND
    ls -lh $BF16_FIRST
    echo "bf16 entry point: $BF16_FIRST"
}

get_bf16() {
    need_preset || return 1
    if find /gguf -name "*.gguf" 2>/dev/null | grep -qi bf16; then
        echo "bf16 already on disk:"
        find_bf16
        return 0
    fi
    echo "This pulls the bf16 reference from $MAIN. Around 66 GB for Nemotron."
    ask "download?" || return 1
    hf download $MAIN --include "*bf16*" --include "*BF16*" --local-dir /gguf
    find_bf16
}

get_quants() {
    need_preset || return 1
    if ls /gguf/*.gguf >/dev/null 2>&1; then
        echo "already on disk:"
        ls -la /gguf/*.gguf
        ask "download anyway, to fill in what is missing?" || return 0
    else
        echo "This pulls every published quant from $MAIN. Around 185 GB for Nemotron."
        ask "download?" || return 1
    fi
    hf download $MAIN --include "*.gguf" \
        --exclude "*bf16*" --exclude "*BF16*" --exclude "*imatrix*" \
        --exclude "dflash-*" --exclude "dspark-*" --local-dir /gguf
    du -sh /gguf
    ls -la /gguf/*.gguf
}

get_one() {
    need_preset || return 1
    echo "pattern: $1"
    ask "download?" || return 1
    hf download $MAIN --include "$1" --local-dir /gguf
    ls -la /gguf
}

get_base() {
    need_preset || return 1
    hf download $METRICS --repo-type $METRICS_KIND --include "kld/*" --local-dir /kld
    if [ -f /kld/kld/base.kld ]; then
        mv /kld/kld/base.kld $BASE
    fi
    ls -lh $BASE
}


# ================================================================== measure

base() {
    need_preset || return 1
    find_bf16 || return 1
    if [ ! -f $EVAL ]; then
        echo "no corpus at $EVAL. Run get_eval."
        return 1
    fi

    echo "Writing the reference. All GPUs. Expect 4 to 8 minutes."
    date
    START=$(date +%s)

    $BIN/llama-perplexity -m $BF16_FIRST -f $EVAL \
        --kl-divergence-base $BASE -c $CTX -ngl 99 \
        2>&1 | tee /logs/base.log

    echo "took $(( $(date +%s) - START )) seconds"
    echo
    echo "=== SIZE OF THE REFERENCE ==="
    ls -lh $BASE
    echo "Under a few GB means shipping it through HF to the other box is fine."
}

kld() {
    if [ ! -f "$1" ]; then echo "no model at $1"; return 1; fi
    if [ ! -f $BASE ]; then echo "no reference at $BASE. Run base, or get_base."; return 1; fi

    NAME=$(basename $1 .gguf)
    echo "--------------------------------------------------"
    echo "KLD: $NAME"
    START=$(date +%s)

    $BIN/llama-perplexity -m $1 -f $EVAL \
        --kl-divergence-base $BASE --kl-divergence -c $CTX -ngl 99 \
        2>&1 | tee /logs/kld-$NAME.log

    echo "took $(( $(date +%s) - START )) seconds"
}

kld_all() {
    TOTAL=$(ls /gguf/*.gguf 2>/dev/null | wc -l)
    N=0
    for f in /gguf/*.gguf; do
        N=$(( N + 1 ))
        echo
        echo "########## $N of $TOTAL, roughly $(( (TOTAL - N) * 3 )) minutes left ##########"
        kld $f
    done
    results
}

bench() {
    NAME=$(basename $1 .gguf)
    echo "GPU 0 only, so the numbers stay comparable to the published card."
    CUDA_VISIBLE_DEVICES=0 $BIN/llama-bench -m $1 -p 512 -n 128 -ngl 99 -r 5 -o json \
        > /logs/bench-$NAME.json
    head -40 /logs/bench-$NAME.json
}

bench_all() {
    for f in /gguf/*.gguf; do
        bench $f
    done
    results
}

gen() {
    if [ -z "$2" ]; then
        echo 'gen MODEL NGL "EXTRA ARGS"'
        echo 'partial offload:   gen /gguf/x.gguf 20 ""'
        echo 'with MTP:          gen /gguf/x.gguf 20 "--spec-type draft-mtp"'
        return 1
    fi
    NAME=$(basename $1 .gguf)
    TAG=$(echo "ngl$2 $3" | tr -c 'a-zA-Z0-9' '-')

    CUDA_VISIBLE_DEVICES=0 $BIN/llama-cli -m $1 -ngl $2 $3 \
        -n 400 --seed 1234 -c 8192 -no-cnv \
        -p "Write a detailed technical explanation of how a mixture of experts layer routes tokens." \
        2>&1 | tee /logs/gen-$NAME-$TAG.log

    echo
    grep -E "eval time|tokens per second|accept" /logs/gen-$NAME-$TAG.log
}


# ================================================================== results

results() {
    python3 - << 'PYEOF'
import glob, json, os, re

def num(pattern, text):
    m = re.search(pattern, text)
    return float(m.group(1)) if m else None

out = {}

for path in glob.glob('/logs/kld-*.log'):
    name = os.path.basename(path)[4:-4]
    text = open(path, errors='ignore').read()
    row = out.setdefault(name, {'name': name})
    row['mean_kld']   = num(r'Mean\s+KLD:\s+([0-9.]+)', text)
    row['q99_kld']    = num(r'99\.0%\s+KLD:\s+([0-9.]+)', text)
    row['median_kld'] = num(r'Median\s+KLD:\s+([0-9.]+)', text)
    row['top1_pct']   = num(r'Same top p:\s+([0-9.]+)', text)
    row['rms_dp_pct'] = num(r'RMS\s+.p\s*:\s+([0-9.]+)', text)

for path in glob.glob('/logs/bench-*.json'):
    name = os.path.basename(path)[6:-5]
    row = out.setdefault(name, {'name': name})
    try:
        for entry in json.load(open(path)):
            key = 'pp512' if entry.get('n_prompt', 0) > 0 else 'tg128'
            row[key + '_tps'] = entry.get('avg_ts')
            row['gpu'] = entry.get('gpu_info')
            row['build'] = entry.get('build_commit')
    except Exception as e:
        row['bench_error'] = str(e)

for name, row in out.items():
    for folder in ('/gguf', '/gguf/experimental'):
        p = os.path.join(folder, name + '.gguf')
        if os.path.exists(p):
            row['size_bytes'] = os.path.getsize(p)
            row['size_gb'] = round(os.path.getsize(p) / 1e9, 2)

rows = sorted(out.values(), key=lambda r: r.get('size_gb') or 0)
with open('/logs/results.json', 'w') as f:
    json.dump(rows, f, indent=2)

print('%-50s %7s %10s %8s %8s' % ('file', 'GB', 'mean KLD', 'top-1', 'tg t/s'))
for r in rows:
    print('%-50s %7s %10s %8s %8s' % (
        r['name'][:50],
        r.get('size_gb', ''), r.get('mean_kld', ''),
        r.get('top1_pct', ''), r.get('tg128_tps', '')))
print()
print('written to /logs/results.json')
PYEOF
}


# ================================================================== upload

push_base() {
    need_preset || return 1
    ls -lh $BASE
    hf upload $METRICS --repo-type $METRICS_KIND $BASE kld/base.kld
    hf upload $METRICS --repo-type $METRICS_KIND /logs/base.log logs/base.log
    echo "Reference is up. The other box can pull it now."
}

push_logs() {
    need_preset || return 1
    du -sh /logs
    hf upload $METRICS --repo-type $METRICS_KIND /logs logs
}

push_results() {
    need_preset || return 1
    if [ ! -f /logs/results.json ]; then
        echo "no results.json. Run results first."
        return 1
    fi
    hf upload $METRICS --repo-type $METRICS_KIND /logs/results.json results.json
}

push_model() {
    need_preset || return 1
    if [ ! -f "$1" ]; then echo "no file at $1"; return 1; fi
    SIZE=$(stat -c %s $1)
    if [ $SIZE -gt 50000000000 ]; then
        echo "Over the 50 GB per-file limit on Hugging Face. Use push_model_split."
        return 1
    fi
    ls -lh $1
    hf upload $MAIN $1 $(basename $1)
}

push_model_split() {
    need_preset || return 1
    if [ ! -f "$1" ]; then echo "no file at $1"; return 1; fi
    PREFIX=/gguf/split/$(basename $1 .gguf)
    mkdir -p /gguf/split
    $BIN/llama-gguf-split --split --split-max-size 45G $1 $PREFIX
    ls -lh $PREFIX*
    hf upload $MAIN /gguf/split $(basename $1 .gguf)
    echo "Uploaded as a folder of shards. Readers point at shard 1."
}

push_card() {
    need_preset || return 1
    if [ ! -f "$1" ]; then echo "no file at $1"; return 1; fi
    hf upload $MAIN $1 README.md
}


# ================================================================== flows

run_base_box() {
    need_preset || return 1
    get_eval
    get_bf16
    base
    push_base
}

wait_base() {
    need_preset || return 1
    echo "Polling $METRICS for the reference. Ctrl-C to stop."
    while true; do
        get_base 2>/dev/null
        if [ -f $BASE ]; then
            echo "Got it."
            return 0
        fi
        echo "not there yet, $(date +%H:%M:%S), retry in 30s"
        sleep 30
    done
}

run_quant_box() {
    need_preset || return 1
    get_eval
    get_quants
    wait_base
    kld_all
    push_logs
    push_results
}
