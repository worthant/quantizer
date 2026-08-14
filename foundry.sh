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
CTX=4096

# Which held-out set we measure on. Everything downstream is named after it,
# so several sets can live side by side without overwriting each other.
EVALSET=neutral
EVAL=/eval/neutral.txt
BASE=/kld/base-neutral.kld

use_evalset() {
    if [ -z "$1" ]; then
        echo "use_evalset neutral | agentic | code       current: $EVALSET"
        return 1
    fi
    EVALSET=$1
    EVAL=/eval/$1.txt
    BASE=/kld/base-$1.kld
    echo "eval set  : $EVALSET"
    echo "corpus    : $EVAL"
    echo "reference : $BASE"
}


# ================================================================== presets

use_nemotron() {
    MAIN=AtomicChat/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-GGUF
    METRICS=AtomicChat/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-GGUF-metrics
    METRICS_KIND=dataset
    UPSTREAM=nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16
    show_preset
}

use_qwen() {
    MAIN=AtomicChat/Qwen3.8-27B-GGUF
    METRICS=AtomicChat/Qwen3.8-27B-GGUF-metrics
    METRICS_KIND=dataset
    UPSTREAM=Qwen/Qwen3.8-27B
    show_preset
}

show_preset() {
    echo "our gguf repo : $MAIN"
    echo "metrics repo  : $METRICS  (type: $METRICS_KIND)"
    echo "upstream      : $UPSTREAM   (original weights, only needed if we"
    echo "                have not published a bf16 gguf yet)"
    echo "eval corpus   : $EVAL"
    echo "context       : $CTX"
    echo
    echo "Run plan to see what exists where and what to do next."
}

reload() {
    curl -sL https://raw.githubusercontent.com/worthant/quantizer/main/foundry.sh -o /foundry.sh
    source /foundry.sh
    echo "reloaded from github"
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
  plan                       what exists on disk AND in both repos, and what
                             command to run next. Start here.
  status                     local checklist only
  ls_main / ls_metrics       list a remote repo with file sizes
  menu                       pick a command by number
  help_me                    this list
  reload                     re-pull this script from github

BOX SETUP
  check                      disk, GPUs, cores, RAM
  cuda_check                 toolkit vs host driver, writes /logs/env.txt
  fix_cuda_compat            drop compat libs when the host driver is newer
  setup                      apt + hugging face cli
  build 120                  compile. 120 blackwell, 90 hopper, 89 ada, 86 ampere
  gpu_test                   does ggml actually see the GPUs

CALIBRATION CORPUS
  ls_corpora                 what is in AtomicChat/calib-corpora
  get_recipe NAME            pull a recipe yaml and print it
  corpus_check DIR_OR_REPO   render the model chat template against system,
                             user, assistant, tool_call and tool_response, then
                             check every emitted marker is a real single token.
                             Run this BEFORE building a corpus for a new model.

DOWNLOAD, each one asks before pulling anything
  get_eval                   the neutral held-out corpus
  get_eval_set NAME          neutral | agentic | code. Switches every derived
                             path, so sets never overwrite each other
  use_evalset NAME           switch sets without downloading
  eval_size                  bytes, token estimate, chunk count at the current ctx
  set_ctx N                  change the evaluation context
  get_bf16                   the reference weights, sharded or not
  get_quants                 every published quant, no bf16, no sidecars
  get_one "PATTERN"          one file by glob
  get_base                   pull an existing reference from the metrics repo
  find_bf16                  locate the bf16 file already on disk

BUILD A BF16 THAT DOES NOT EXIST YET   (Qwen case, not Nemotron)
  get_upstream               original safetensors from UPSTREAM into /src
  setup_convert              torch and friends for the converter
  make_bf16                  convert /src into a bf16 gguf
  push_model_split FILE      publish it, split at 45 GB

MEASURE
  base                       write the reference into /kld/base-<set>.kld
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
    COMPAT_LIB=$(ls /usr/local/cuda/compat/libcuda.so.*.* 2>/dev/null | head -1)
    if [ -z "$COMPAT_LIB" ]; then
        echo "none present, nothing to do"
    else
        COMPAT_MAJOR=$(basename $COMPAT_LIB | sed "s/libcuda.so.//" | cut -d. -f1)
        HOST_MAJOR=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1 | cut -d. -f1)
        echo "compat libs target driver $COMPAT_MAJOR, host driver is $HOST_MAJOR"
        if [ "$HOST_MAJOR" -ge "$COMPAT_MAJOR" ]; then
            echo "Host driver is newer or equal, so these libs can only break CUDA init."
            fix_cuda_compat
        else
            echo "Host driver is older, so these libs are needed. Leaving them alone."
        fi
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

setup_convert() {
    echo "This installs torch and friends so convert_hf_to_gguf.py can run."
    echo "Around 3 GB of downloads. Only needed when we have to BUILD a bf16 gguf."
    ask "install?" || return 1
    pip install --break-system-packages -q -U -r /llama.cpp/requirements/requirements-convert_hf_to_gguf.txt
    echo "done"
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


# ================================================================== what exists where

ls_main() {
    need_preset || return 1
    list_repo "$MAIN" model
}

ls_metrics() {
    need_preset || return 1
    list_repo "$METRICS" "$METRICS_KIND"
}

list_repo() {
    python3 - "$1" "$2" << 'LSEOF'
import sys
from huggingface_hub import HfApi
repo, kind = sys.argv[1], sys.argv[2]

def human(n):
    for unit, div in (("GB", 1e9), ("MB", 1e6), ("KB", 1e3)):
        if n >= div:
            return "%.2f %s" % (n / div, unit)
    return "%d B" % n

try:
    tree = list(HfApi().list_repo_tree(repo, repo_type=kind, recursive=True))
except Exception as e:
    print("cannot read %s (%s): %s" % (repo, kind, e))
    sys.exit(1)
total = 0
for f in sorted(tree, key=lambda x: x.path):
    size = getattr(f, "size", None)
    if size is None:
        continue
    total += size
    print("%12s  %s" % (human(size), f.path))
print("---")
print("%12s  total, %d files" % (human(total), len([f for f in tree if getattr(f, "size", None)])))
LSEOF
}

plan() {
    need_preset || return 1
    python3 - "$MAIN" "$METRICS" "$METRICS_KIND" "$UPSTREAM" << 'PLANEOF'
import os, sys, glob
from huggingface_hub import HfApi

main, metrics, kind, upstream = sys.argv[1:5]
api = HfApi()

def files(repo, t):
    try:
        return api.list_repo_files(repo, repo_type=t)
    except Exception:
        return None

main_files = files(main, "model")
metric_files = files(metrics, kind)

def block(title, on_disk, remote_line, todo):
    print()
    print(title)
    print("  on disk         : %s" % on_disk)
    print("  remote          : %s" % remote_line)
    print("  -> %s" % todo)

# --- bf16
local_bf16 = [p for p in glob.glob("/gguf/**/*.gguf", recursive=True) if "bf16" in p.lower()]
if main_files is None:
    remote_bf16 = []
else:
    remote_bf16 = [f for f in main_files if "bf16" in f.lower() and f.endswith(".gguf")]

if local_bf16:
    todo = "nothing, it is here"
elif remote_bf16:
    todo = "run: get_bf16"
else:
    todo = "not published yet. run: get_upstream, then setup_convert, then make_bf16"
block("BF16 reference weights",
      ("%d files" % len(local_bf16)) if local_bf16 else "no",
      ("%d files in %s" % (len(remote_bf16), main)) if remote_bf16
        else ("repo unreadable" if main_files is None else "not in %s" % main),
      todo)

# --- kld base
local_base = bool(glob.glob("/kld/base-*.kld"))
remote_base = [f for f in (metric_files or []) if f.endswith(".kld")]
if local_base:
    todo = "nothing, it is here"
elif remote_base:
    todo = "run: get_base"
elif local_bf16:
    todo = "run: base      (this computes it, 4 to 8 minutes)"
else:
    todo = "get the bf16 first, then run: base"
block("KLD reference (.kld)",
      "yes: %s" % ", ".join(os.path.basename(p) for p in glob.glob("/kld/base-*.kld")) if local_base else "no",
      ("%d in %s" % (len(remote_base), metrics)) if remote_base
        else ("metrics repo unreadable, check METRICS_KIND" if metric_files is None
              else "not in %s" % metrics),
      todo)

# --- quants
local_q = [p for p in glob.glob("/gguf/*.gguf") if "bf16" not in p.lower()]
remote_q = [f for f in (main_files or [])
            if f.endswith(".gguf") and "bf16" not in f.lower()
            and "imatrix" not in f and not f.startswith(("dflash-", "dspark-"))]
block("Quants to measure",
      "%d files" % len(local_q),
      "%d files in %s" % (len(remote_q), main) if main_files is not None else "repo unreadable",
      "nothing, they are here" if len(local_q) >= len(remote_q) and local_q else "run: get_quants")
print()
PLANEOF
}

get_upstream() {
    need_preset || return 1
    echo "This pulls the ORIGINAL weights from $UPSTREAM into /src."
    echo "Only needed when we have not published a bf16 gguf ourselves."
    ask "download?" || return 1
    mkdir -p /src
    hf download $UPSTREAM --local-dir /src
    du -sh /src
    ls /src
}

make_bf16() {
    need_preset || return 1
    if [ ! -d /src ]; then
        echo "no original weights in /src. Run get_upstream."
        return 1
    fi
    if [ ! -f /llama.cpp/convert_hf_to_gguf.py ]; then
        echo "llama.cpp not cloned. Run build first."
        return 1
    fi
    OUT=/gguf/$(basename $MAIN | sed "s/-GGUF//")-bf16.gguf
    echo "converting /src -> $OUT"
    echo "If it fails with NaN in token_embd, add --no-lazy and rerun."
    date
    python3 /llama.cpp/convert_hf_to_gguf.py /src --outtype bf16 --outfile $OUT \
        2>&1 | tee /logs/convert.log
    date
    ls -lh $OUT
    echo
    echo "Publish it with push_model_split so it clears the 50 GB per-file limit."
}


# ================================================================== fetching one file

# fetch_one REPO KIND REMOTE_PATTERN DEST
# Matches against the FULL path inside the repo and against the bare filename,
# prints exactly what it found and where it put it, then places the file itself.
fetch_one() {
    python3 - "$1" "$2" "$3" "$4" << 'FETCHEOF'
import fnmatch, os, shutil, sys
from huggingface_hub import HfApi, hf_hub_download

repo, kind, pattern, dest = sys.argv[1:5]

try:
    files = HfApi().list_repo_files(repo, repo_type=kind)
except Exception as e:
    print("cannot read %s (%s): %s" % (repo, kind, e))
    sys.exit(1)

hits = [f for f in files
        if fnmatch.fnmatch(f, pattern) or fnmatch.fnmatch(os.path.basename(f), pattern)]

if not hits:
    print("nothing in %s matches %r" % (repo, pattern))
    print("the repo contains:")
    for f in sorted(files)[:60]:
        print("   ", f)
    sys.exit(1)

sizes = {}
try:
    for entry in HfApi().list_repo_tree(repo, repo_type=kind, recursive=True):
        s = getattr(entry, "size", None)
        if s is not None:
            sizes[entry.path] = s
except Exception:
    pass

hits.sort(key=lambda f: sizes.get(f, 0), reverse=True)

if len(hits) > 1:
    print("%d files match %r, taking the BIGGEST:" % (len(hits), pattern))
    for f in hits:
        mark = " <-- taking this" if f == hits[0] else ""
        print("   %8.2f MB  %s%s" % (sizes.get(f, 0) / 1e6, f, mark))

src_path = hits[0]
print()
print("  from : %s :: %s" % (repo, src_path))
print("  to   : %s" % dest)
print()

cached = hf_hub_download(repo, src_path, repo_type=kind)
os.makedirs(os.path.dirname(dest), exist_ok=True)
shutil.copyfile(cached, dest)
print("placed, %.2f MB" % (os.path.getsize(dest) / 1e6))
FETCHEOF
}

ls_corpora() {
    list_repo AtomicChat/calib-corpora dataset
}

get_recipe() {
    if [ -z "$1" ]; then
        echo "get_recipe NAME       e.g. get_recipe nemotron-3.5-lightning"
        echo "see what exists with: ls_corpora"
        return 1
    fi
    fetch_one AtomicChat/calib-corpora dataset "*$1*.yaml" /eval/recipe.yaml
    echo
    cat /eval/recipe.yaml
}


# ================================================================== download

# Exact paths inside AtomicChat/calib-corpora. Globs are not safe here:
# every set also has a *.manifest.jsonl next to it, which is not the corpus.
get_eval_set() {
    if [ -z "$1" ]; then
        echo "get_eval_set neutral | agentic | code"
        return 1
    fi
    case "$1" in
        neutral) REMOTE=eval/neutral/eval_neutral.txt ;;
        agentic) REMOTE=eval/agentic/eval_agentic.txt ;;
        code)    REMOTE=eval/code/eval_code_full.txt ;;
        *) echo "unknown set: $1. Run ls_corpora and pass an exact path instead."
           return 1 ;;
    esac

    use_evalset $1
    if [ -f $EVAL ]; then
        echo "already here:"
        eval_size
        return 0
    fi
    ask "download?" || return 1
    fetch_one AtomicChat/calib-corpora dataset "$REMOTE" $EVAL || return 1
    eval_size
}

get_eval() {
    get_eval_set neutral
}

eval_size() {
    if [ ! -f $EVAL ]; then
        echo "no corpus at $EVAL"
        return 1
    fi
    BYTES=$(stat -c %s $EVAL)
    TOKENS=$(( BYTES / 4 ))
    CHUNKS=$(( TOKENS / CTX ))
    echo
    echo "corpus : $EVAL"
    echo "size   : $BYTES bytes, roughly $TOKENS tokens"
    echo "at ctx $CTX that is about $CHUNKS chunks"
    if [ $CHUNKS -lt 2 ]; then
        echo
        echo "TOO SMALL. llama-perplexity needs at least two full contexts."
        echo "Either you grabbed the wrong file (run ls_corpora and look), or"
        echo "lower the context with set_ctx."
        return 1
    fi
    if [ $CHUNKS -lt 20 ]; then
        echo
        echo "That is a thin sample. Error bars on the mean KLD will be wide."
    fi
}

set_ctx() {
    if [ -z "$1" ]; then
        echo "set_ctx 4096      current: $CTX"
        return 1
    fi
    CTX=$1
    echo "context is now $CTX"
    eval_size
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
    if [ -f /kld/kld/base-$EVALSET.kld ]; then
        mv /kld/kld/base-$EVALSET.kld $BASE
    fi
    ls -lh $BASE
}


# ================================================================== corpus validation

# corpus_check DIR_OR_REPO
# Renders the model's own chat template against a conversation that uses every
# construct we calibrate on, then checks that every marker the template emits
# is a real single token in the vocabulary. If it is not, that part of the
# corpus calibrates nothing and the imatrix is quietly wrong.
corpus_check() {
    if [ -z "$1" ]; then
        echo "corpus_check /src                          local original weights"
        echo "corpus_check nvidia/Some-Model-BF16        straight from the hub"
        return 1
    fi
    pip install --break-system-packages -q -U jinja2 >/dev/null 2>&1
    python3 - "$1" << 'CHECKEOF'
import json, os, re, sys

target = sys.argv[1]

def load(name):
    if os.path.isdir(target):
        p = os.path.join(target, name)
        return open(p, encoding="utf-8").read() if os.path.exists(p) else None
    from huggingface_hub import hf_hub_download
    try:
        return open(hf_hub_download(target, name), encoding="utf-8").read()
    except Exception:
        return None

cfg_raw = load("tokenizer_config.json")
if cfg_raw is None:
    print("FAIL  no tokenizer_config.json at %s" % target)
    sys.exit(1)
cfg = json.loads(cfg_raw)

template = load("chat_template.jinja")
where = "chat_template.jinja"
if template is None:
    template = cfg.get("chat_template")
    where = "tokenizer_config.json"
if isinstance(template, list):
    template = template[0].get("template")

if not template:
    print("FAIL  no chat template anywhere. The corpus cannot be rendered")
    print("      the way the model actually sees text.")
    sys.exit(1)
print("ok    chat template found in %s, %d chars" % (where, len(template)))

# every construct we calibrate on
convo = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What is the weather in Paris?"},
    {"role": "assistant", "content": "", "tool_calls": [
        {"type": "function", "function": {"name": "get_weather",
         "arguments": '{"city": "Paris"}'}}]},
    {"role": "tool", "content": '{"temp_c": 19}'},
    {"role": "assistant", "content": "It is 19 degrees in Paris."},
]
tools = [{"type": "function", "function": {
    "name": "get_weather",
    "description": "Get the weather for a city",
    "parameters": {"type": "object", "properties": {"city": {"type": "string"}},
                   "required": ["city"]}}}]

from jinja2 import Environment
from jinja2.exceptions import TemplateError
env = Environment(trim_blocks=True, lstrip_blocks=True)
env.policies["json.dumps_kwargs"] = {"ensure_ascii": False}
try:
    tpl = env.from_string(template)
except Exception as e:
    print("FAIL  the template does not even parse as jinja: %s" % e)
    sys.exit(1)

rendered = None
for kwargs in (
    {"messages": convo, "tools": tools, "add_generation_prompt": True},
    {"messages": convo, "add_generation_prompt": True},
    {"messages": convo[:2] + convo[4:], "add_generation_prompt": True},
):
    try:
        rendered = tpl.render(bos_token="", eos_token="", **kwargs)
        keys = ", ".join(k for k in kwargs if k != "messages")
        print("ok    renders with: %s" % (keys or "messages only"))
        break
    except (TemplateError, Exception) as e:
        print("      render failed with %s: %s" % (list(kwargs), e))

if rendered is None:
    print("FAIL  the template will not render any of our conversations")
    sys.exit(1)

if "tool" not in rendered.lower():
    print("warn  the rendered text has no trace of the tool call. Either the")
    print("      template ignores tools, or it wants a different message shape.")
    print("      Agentic and tool data in the corpus will not exercise it.")

# which markers does it emit, and are they real tokens
markers = sorted(set(re.findall(r"<\|[^|>]{1,40}\|>|<[a-z_]{2,20}>|\[/?[A-Z_]{2,20}\]", rendered)))
if not markers:
    print("warn  the template emits no special markers at all, unusual")


known = set()
for v in (cfg.get("added_tokens_decoder") or {}).values():
    if isinstance(v, dict) and "content" in v:
        known.add(v["content"])
tok_raw = load("tokenizer.json")
if tok_raw:
    try:
        for a in json.loads(tok_raw).get("added_tokens", []):
            known.add(a["content"])
    except Exception:
        pass
add_raw = load("added_tokens.json")
if add_raw:
    try:
        known.update(json.loads(add_raw).keys())
    except Exception:
        pass

print()
print("markers the template emits, and whether they are single tokens:")
bad = []
for m in markers:
    if m in known:
        print("  ok    %s" % m)
    else:
        print("  BAD   %s   not in the vocabulary" % m)
        bad.append(m)

print()
if bad:
    print("RESULT  %d markers are not real tokens." % len(bad))
    print("        They will tokenize as literal punctuation, several tokens each,")
    print("        and that part of the corpus calibrates nothing useful.")
    print("        Fix the corpus rendering before running llama-imatrix.")
else:
    print("RESULT  every marker is a real single token.")
    print("        Run llama-imatrix with --parse-special or all of this is wasted.")
print()
print("first 600 chars of the rendered conversation, eyeball it:")
print("-" * 60)
print(rendered[:600])
CHECKEOF
}


# ================================================================== measure

base() {
    need_preset || return 1
    find_bf16 || return 1
    if [ ! -f $EVAL ]; then
        echo "no corpus at $EVAL. Run get_eval."
        return 1
    fi
    eval_size || return 1

    echo
    echo "Writing the reference. All GPUs. Expect 4 to 8 minutes."
    echo "Warnings about unused blk.52.nextn tensors are normal: that is the MTP"
    echo "block, which a plain forward pass never executes."
    date
    START=$(date +%s)

    $BIN/llama-perplexity -m $BF16_FIRST -f $EVAL \
        --kl-divergence-base $BASE -c $CTX -ngl 99 \
        2>&1 | tee /logs/base-$EVALSET.log

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
    echo "KLD on $EVALSET: $NAME"
    START=$(date +%s)

    $BIN/llama-perplexity -m $1 -f $EVAL \
        --kl-divergence-base $BASE --kl-divergence -c $CTX -ngl 99 \
        2>&1 | tee /logs/kld-$EVALSET--$NAME.log

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
    stem = os.path.basename(path)[4:-4]
    if '--' in stem:
        evalset, name = stem.split('--', 1)
    else:
        evalset, name = 'neutral', stem
    text = open(path, errors='ignore').read()
    row = out.setdefault((evalset, name), {'name': name, 'eval_set': evalset})
    row['mean_kld']   = num(r'Mean\s+KLD:\s+([0-9.]+)', text)
    row['q99_kld']    = num(r'99\.0%\s+KLD:\s+([0-9.]+)', text)
    row['median_kld'] = num(r'Median\s+KLD:\s+([0-9.]+)', text)
    row['top1_pct']   = num(r'Same top p:\s+([0-9.]+)', text)
    row['rms_dp_pct'] = num(r'RMS\s+.p\s*:\s+([0-9.]+)', text)

for path in glob.glob('/logs/bench-*.json'):
    name = os.path.basename(path)[6:-5]
    row = out.setdefault(('speed', name), {'name': name, 'eval_set': 'speed'})
    try:
        for entry in json.load(open(path)):
            key = 'pp512' if entry.get('n_prompt', 0) > 0 else 'tg128'
            row[key + '_tps'] = entry.get('avg_ts')
            row['gpu'] = entry.get('gpu_info')
            row['build'] = entry.get('build_commit')
    except Exception as e:
        row['bench_error'] = str(e)

for key, row in out.items():
    for folder in ('/gguf', '/gguf/experimental'):
        p = os.path.join(folder, row['name'] + '.gguf')
        if os.path.exists(p):
            row['size_bytes'] = os.path.getsize(p)
            row['size_gb'] = round(os.path.getsize(p) / 1e9, 2)

rows = sorted(out.values(), key=lambda r: (r.get('eval_set') or '', r.get('size_gb') or 0))
with open('/logs/results.json', 'w') as f:
    json.dump(rows, f, indent=2)

print('%-9s %-44s %7s %10s %8s %8s' % ('set', 'file', 'GB', 'mean KLD', 'top-1', 'tg t/s'))
for r in rows:
    print('%-9s %-44s %7s %10s %8s %8s' % (
        r.get('eval_set', ''), r['name'][:44],
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
    hf upload $METRICS --repo-type $METRICS_KIND $BASE kld/base-$EVALSET.kld
    hf upload $METRICS --repo-type $METRICS_KIND /logs/base-$EVALSET.log logs/base-$EVALSET.log
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
