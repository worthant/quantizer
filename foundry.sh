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

# Bump this on every change. reload compares it against what is on github.
FOUNDRY_VERSION=2026-08-14.29

export HF_XET_HIGH_PERFORMANCE=1
export HF_HOME=/hf

# Both http paths to github are cached: raw.githubusercontent behind a CDN, and
# the contents API behind its own layer. Either can serve a file that is minutes
# old, and reload then reports success on a copy that has not changed. git does
# not go through any of that, so a clone is the only version of this that can be
# trusted, and it can also show which commit it is actually on.
reload() {
    if [ -d /quantizer/.git ]; then
        git -C /quantizer fetch -q origin && git -C /quantizer reset -q --hard origin/HEAD || return 1
    else
        rm -rf /quantizer
        git clone -q https://github.com/worthant/quantizer /quantizer || return 1
    fi

    echo "commit: $(git -C /quantizer log -1 --format='%h %ad %s' --date=short)"
    local got
    got=$(grep -m1 "^FOUNDRY_VERSION=" /quantizer/foundry.sh | cut -d= -f2)
    echo "here: ${FOUNDRY_VERSION:-unknown}    at that commit: ${got:-none}"

    if [ "$got" = "$FOUNDRY_VERSION" ]; then
        echo "same version. That commit above is what is on the remote, so if it"
        echo "is not what you pushed, the push did not land on the default branch."
        return 0
    fi
    cp /quantizer/foundry.sh /foundry.sh
    source /foundry.sh
    echo "now running $FOUNDRY_VERSION"
}


# ------------------------------------------------------------------ basics
# Bash functions live in one shell process, so a new tmux pane knows nothing.
# save_state writes the current selection to /state.sh and the bottom of this
# file reads it back, so every pane agrees.
save_state() {
    cat > $STATE << STATEEOF
MAIN=$MAIN
RECIPE=$RECIPE
METRICS=$METRICS
METRICS_KIND=$METRICS_KIND
UPSTREAM=$UPSTREAM
EVALSET=$EVALSET
EVAL=$EVAL
BASE=$BASE
CTX=$CTX
IM_TOKENS_EXACT=$IM_TOKENS_EXACT
IM_MODEL=$IM_MODEL
IM_CORPUS=$IM_CORPUS
IM_CTX=$IM_CTX
GPUS=$GPUS
AUTOPUSH=$AUTOPUSH
INCLUDE_EXPERIMENTAL=$INCLUDE_EXPERIMENTAL
STATEEOF
}

persist_shell() {
    grep -q "source /foundry.sh" ~/.bashrc 2>/dev/null || echo "source /foundry.sh" >> ~/.bashrc
    echo "Added to ~/.bashrc. Every new tmux pane now loads this by itself."
}

save_token() {
    if [ -z "$HF_TOKEN" ]; then
        echo "export HF_TOKEN first, then run this"
        return 1
    fi
    grep -q "export HF_TOKEN=" ~/.bashrc 2>/dev/null \
        && sed -i "s|export HF_TOKEN=.*|export HF_TOKEN=$HF_TOKEN|" ~/.bashrc \
        || echo "export HF_TOKEN=$HF_TOKEN" >> ~/.bashrc
    echo "stored in ~/.bashrc, every new pane will have it"
    echo "this box is disposable, the token is not: revoke it when you are done"
}

# The cli's argument order changed between huggingface_hub versions and boxes
# rented at different times do not have the same one, so uploads go through the
# python API instead.
hf_put() {
    if [ -z "$HF_TOKEN" ] && [ ! -f ~/.cache/huggingface/token ]; then
        echo "no token in this shell. A new tmux pane does not inherit it."
        echo "  export HF_TOKEN=<yours>      and to stop repeating it:  save_token"
        return 1
    fi
    python3 - "$1" "$2" "$3" "$4" << 'PUTEOF'
import sys
from huggingface_hub import HfApi
local, remote, repo, kind = sys.argv[1:5]
try:
    HfApi().upload_file(path_or_fileobj=local, path_in_repo=remote,
                        repo_id=repo, repo_type=kind)
except Exception as e:
    msg = str(e).splitlines()[0]
    print(msg)
    if "401" in msg:
        print("that is an auth failure: the token is missing, wrong, or has no")
        print("write access to " + repo)
    sys.exit(1)
PUTEOF
}

hf_put_dir() {
    if [ -z "$HF_TOKEN" ] && [ ! -f ~/.cache/huggingface/token ]; then
        echo "no token in this shell. export HF_TOKEN, then save_token"
        return 1
    fi
    python3 - "$1" "$2" "$3" "$4" << 'PUTDEOF'
import sys
from huggingface_hub import HfApi
local, remote, repo, kind = sys.argv[1:5]
try:
    HfApi().upload_folder(folder_path=local, path_in_repo=remote,
                          repo_id=repo, repo_type=kind)
except Exception as e:
    print(str(e).splitlines()[0])
    sys.exit(1)
PUTDEOF
}

# Never guess a repo id.
find_repo() {
    if [ -z "$1" ]; then
        echo "find_repo QUERY        e.g. find_repo Qwen3.8"
        return 1
    fi
    python3 - "$1" << 'FINDEOF'
import sys
from huggingface_hub import HfApi
q = sys.argv[1]
try:
    hits = list(HfApi().list_models(search=q, sort="downloads", direction=-1, limit=40))
except Exception as e:
    print("search failed: %s" % e)
    sys.exit(1)
if not hits:
    print("nothing on the hub matches %r" % q)
    sys.exit(1)
print("%-52s %12s  %s" % ("repo id", "downloads", "updated"))
for m in hits:
    print("%-52s %12s  %s" % (m.id, m.downloads or 0,
                              str(getattr(m, "lastModified", ""))[:10]))
FINDEOF
}

# Boxes get reused across models and the previous one's weights are still on
# disk, so anything that picks a file filters by this stem.
model_stem() {
    [ -n "$MAIN" ] && basename "$MAIN" | sed "s/-GGUF$//"
}

INCLUDE_EXPERIMENTAL=${INCLUDE_EXPERIMENTAL:-0}

quant_files() {
    if [ "$INCLUDE_EXPERIMENTAL" = "1" ]; then
        DEPTH=2
    else
        DEPTH=1
    fi
    local stem
    stem=$(model_stem)
    find /gguf -maxdepth $DEPTH -name "*.gguf" 2>/dev/null \
        | grep -v -i "bf16" \
        | grep -v "/dflash-" \
        | grep -v "/dspark-" \
        | grep -v -i "imatrix" \
        | { [ -n "$stem" ] && grep -F "$stem" || cat; } \
        | sort
}

autopush() {
    local src_path="$1" dest_path="$2" err
    if [ "$AUTOPUSH" != "1" ] || [ -z "$METRICS" ] || [ ! -f "$src_path" ]; then
        return 0
    fi
    if err=$(hf_put "$src_path" "$dest_path" "$METRICS" "$METRICS_KIND" 2>&1); then
        echo "  uploaded -> $METRICS :: $dest_path"
    else
        echo "  UPLOAD FAILED for $dest_path:"
        echo "$err" | sed "s/^/    /"
    fi
}

setup_convert() {
    echo "This installs torch and friends so convert_hf_to_gguf.py can run."
    echo "Around 3 GB of downloads. Only needed when we have to BUILD a bf16 gguf."
    ask "install?" || return 1
    pip install --break-system-packages -q -U -r /llama.cpp/requirements/requirements-convert_hf_to_gguf.txt
    echo "done. If the converter complains about RoPE, pin transformers below 5."
}

# Every function this file advertises has to exist. Editing a big shell file by
# cutting between two function names quietly eats whatever sat between them.
selfcheck() {
    local f missing=""
    for f in save_state persist_shell save_token hf_put hf_put_dir reload \
             find_repo selfcheck model_stem quant_files autopush use_nemotron \
             use_qwen use_model show_preset need_preset ask mark status menu \
             help_me use_evalset set_ctx check cuda_check fix_cuda_compat \
             clean_run clean_gguf repo_ok token_check make_metrics make_repos \
             setup setup_convert build gpu_test get_tools check_pool fix_pool \
             install_auto_fmt setup_corpus build_corpus push_corpus \
             corpus_check make_recipe new_model pick_model get_eval \
             get_eval_set eval_size get_bf16 get_quants get_one find_bf16 \
             get_upstream make_bf16 base kld kld_all kld_ext bench bench_all \
             gen results bits quantize ladder write_ladder get_external \
             push_base push_logs push_results push_model push_model_split \
             push_card pull_logs get_imatrix get_calib wait_calib im_size \
             im_plan im_shard im_range im_merge im_merge_all im_stats \
             im_status push_shards plan ls_main ls_metrics ls_corpora \
             get_recipe use_gpus apply_gpus list_repo fetch_one \
             write_kld_readme send_base get_base; do
        type -t $f > /dev/null 2>&1 || missing="$missing $f"
    done
    if [ -n "$missing" ]; then
        echo "MISSING functions:$missing"
        echo "the file on disk is incomplete. Pull it again."
        return 1
    fi
    echo "all functions present, version $FOUNDRY_VERSION"
}

help_me() {
cat << 'HELP_EOF'

ORIENTATION   status | plan | menu | selfcheck | help_me | reload

PRESET        use_model NAME REPO | use_nemotron | use_qwen | find_repo QUERY

BOX SETUP     check | cuda_check | setup | build 120 | gpu_test
              persist_shell | save_token | clean_run | clean_gguf

REPOS         token_check | repo_ok | make_repos | ls_main | ls_metrics

NEW MODEL     new_model NAME REPO | make_recipe NAME REPO | corpus_check REPO
              get_upstream | setup_convert | make_bf16

CORPUS        get_tools | check_pool | fix_pool | install_auto_fmt
              setup_corpus | build_corpus NAME | push_corpus NAME
              get_calib NAME | wait_calib NAME | ls_corpora | get_recipe NAME

IMATRIX       pick_model | im_size | im_plan N | im_shard I N
              im_range FROM COUNT LABEL | im_merge_all | im_status
              push_shards | get_imatrix | im_stats FILE
              IM_TOKENS_EXACT=N  IM_BATCH=4096  IM_SKIP="name"  IM_NO_PPL=1

QUANTIZE      bits | bits 'RULE' 'RULE' | quantize LABEL TYPE ...
              write_ladder | ladder /ladder.txt --dry | ladder /ladder.txt
              get_external REPO PATTERN

MEASURE       get_eval | get_eval_set NAME | eval_size | set_ctx N | base
              use_gpus 0,1 | kld MODEL | kld_all | kld_ext
              bench MODEL | bench_all | gen MODEL NGL "EXTRA"
              results | pull_logs

UPLOAD        push_base | push_logs | push_results | push_model FILE
              push_model_split FILE | push_card FILE | send_base user@host PORT

HELP_EOF
}

# ------------------------------------------------------------------ presets

use_nemotron() {
    MAIN=AtomicChat/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-GGUF
    METRICS=AtomicChat/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-GGUF-metrics
    METRICS_KIND=dataset
    UPSTREAM=nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16
    RECIPE=nemotron-3.5-lightning
    save_state
    show_preset
}

# For anything without a preset. Names our repos by convention and refuses a
# repo id that does not exist, so a typo cannot propagate into every later step.
use_model() {
    if [ -z "$2" ]; then
        echo "use_model SHORTNAME UPSTREAM_REPO"
        echo "  use_model qwen3.8-27b Qwen/Qwen3.8-27B"
        return 1
    fi
    if ! python3 -c "
import sys
from huggingface_hub import HfApi
sys.exit(0 if HfApi().repo_exists(sys.argv[1]) else 1)
" "$2" 2>/dev/null; then
        echo "no such repo on the hub: $2"
        echo "find the real id with:   find_repo <part of the name>"
        return 1
    fi
    RECIPE=$1
    UPSTREAM=$2
    MAIN=AtomicChat/$(basename $2)-GGUF
    METRICS=AtomicChat/$(basename $2)-GGUF-metrics
    METRICS_KIND=dataset
    save_state
    show_preset
}

use_qwen() {
    use_model qwen3.8-27b Qwen/Qwen3.8-27B
}

show_preset() {
    echo "recipe name   : ${RECIPE:-not set}"
    echo "our gguf repo : $MAIN"
    echo "metrics repo  : $METRICS  (type: $METRICS_KIND)"
    echo "upstream      : $UPSTREAM"
    echo "eval corpus   : $EVAL"
    echo "context       : $CTX"
    echo
    echo "Run plan to see what exists where and what to do next."
}

need_preset() {
    if [ -z "$MAIN" ]; then
        echo "No preset loaded. Run use_model NAME REPO first."
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

use_evalset() {
    if [ -z "$1" ]; then
        echo "use_evalset neutral | agentic | code       current: $EVALSET"
        return 1
    fi
    EVALSET=$1
    EVAL=/eval/$1.txt
    BASE=/kld/base-$1.kld
    save_state
    echo "eval set  : $EVALSET"
    echo "corpus    : $EVAL"
    echo "reference : $BASE"
}

set_ctx() {
    if [ -z "$1" ]; then
        echo "set_ctx 4096      current: $CTX"
        return 1
    fi
    CTX=$1
    save_state
    echo "context is now $CTX"
    eval_size
}

# ------------------------------------------------------------------ orientation

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
        echo "  [ ] preset loaded      ->  run:  use_model NAME REPO"
    else
        echo "  [x] preset: $MAIN"
    fi
    mark "cuda checked"       /logs/env.txt              "cuda_check"
    mark "tools installed"    /usr/bin/ninja             "setup"
    mark "llama.cpp built"    $BIN/llama-imatrix         "build 120"
    mark "eval corpus"        $EVAL                      "get_eval"

    if find /gguf -name "*.gguf" 2>/dev/null | grep -qi bf16; then
        echo "  [x] bf16 on disk"
    else
        echo "  [ ] bf16 on disk       ->  run:  get_bf16, or make_bf16 for a new model"
    fi

    STALE=$(ls /logs/kld-*.log /logs/bench-*.json 2>/dev/null | grep -v -i "$(basename ${MAIN:-zzz} -GGUF)" | wc -l)
    if [ "$STALE" -gt 0 ]; then
        echo
        echo "  !! $STALE log(s) in /logs belong to a different model."
        echo "     results would mix them in and push_logs would upload them."
        echo "     run:  clean_run"
        echo
    fi

    QN=$(quant_files | wc -l)
    if [ "$QN" -gt 0 ]; then
        echo "  [x] quants on disk: $QN files"
    else
        echo "  [ ] quants on disk     ->  run:  get_quants, or quantize them here"
    fi

    mark "reference built"    $BASE                      "base"
    KLDS=$(ls /logs/kld-*.log 2>/dev/null | wc -l)
    echo "  [$([ $KLDS -gt 0 ] && echo x || echo ' ')] kld logs: $KLDS       ->  run:  kld_all"
    BENCHES=$(ls /logs/bench-*.json 2>/dev/null | wc -l)
    echo "  [$([ $BENCHES -gt 0 ] && echo x || echo ' ')] bench logs: $BENCHES     ->  run:  bench_all"
    mark "results.json"       /logs/results.json         "results"

    echo
    echo "  imatrix:"
    if [ -f /eval/calib_train.txt ]; then
        echo "    [x] calibration corpus"
    else
        echo "    [ ] calibration corpus  ->  run:  get_calib $RECIPE"
    fi
    if [ -n "$IM_MODEL" ] && [ -f "$IM_MODEL" ]; then
        echo "    [x] IM_MODEL: $(basename $IM_MODEL)"
    else
        echo "    [ ] IM_MODEL not set    ->  run:  pick_model"
    fi
    SHARDS=$(ls /imatrix/shard-*.gguf 2>/dev/null | wc -l)
    if [ -f /imatrix/imatrix.gguf ]; then
        echo "    [x] imatrix merged"
    elif [ "$SHARDS" -gt 0 ]; then
        echo "    [~] $SHARDS shard(s) here  ->  run:  im_merge_all once every node is done"
    else
        echo "    [ ] no imatrix          ->  run:  im_plan N, then im_shard I N on each box"
    fi

    echo
    echo "  full command list: help_me      pick by number: menu"
    echo "  remote view: plan               integrity: selfcheck"
    echo
}

menu() {
    echo
    echo "  model    : ${MAIN:-none, pick a preset first}"
    echo "  eval set : $EVALSET   context: $CTX"
    echo
    PS3="
number: "
    select picked in \
        "status" "plan" "help_me" "selfcheck" \
        "cuda_check" "setup" "build 120" "gpu_test" \
        "get_bf16" "get_quants" "find_bf16" "get_eval" "eval_size" \
        "pick_model" "im_size" "bits" \
        "base" "kld_all" "bench_all" "results" \
        "push_base" "push_logs" "push_results" "pull_logs" \
        "ls_main" "ls_metrics" "ls_corpora" \
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

# ------------------------------------------------------------------ box setup

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



# Measurements from a previous model on the same box would be parsed into this
# model's results and uploaded to its metrics repo. Wipe them when switching.
clean_run() {
    echo "about to remove measurements from any earlier model on this box:"
    echo "  /logs      $(ls /logs 2>/dev/null | wc -l) files"
    echo "  /kld       $(ls /kld 2>/dev/null | wc -l) files"
    echo "  /imatrix   $(ls /imatrix 2>/dev/null | wc -l) files"
    echo "the weights in /gguf and /src are left alone."
    ask "remove?" || return 1
    rm -rf /logs/* /kld/* /imatrix/*
    mkdir -p /logs /kld /imatrix
    echo "clean. Re-run cuda_check so env.txt exists again."
}

# Weights from an earlier model on this box. They are large and they confuse
# every file picker, so this offers to remove exactly those.
clean_gguf() {
    need_preset || return 1
    local stem others
    stem=$(model_stem)
    others=$(find /gguf /src -maxdepth 2 -name "*.gguf" 2>/dev/null | grep -v -F "$stem")
    if [ -z "$others" ]; then
        echo "nothing here belongs to another model"
        return 0
    fi
    echo "not part of $stem:"
    echo "$others" | xargs -r du -sh 2>/dev/null | sed "s/^/  /"
    echo
    du -sh /gguf
    ask "delete them?" || return 1
    echo "$others" | xargs -r rm -f
    find /gguf -type d -empty -delete
    du -sh /gguf
}

# Every log that says anything useful goes to the metrics repo the moment it
# is written. Set AUTOPUSH=0 to keep things local while experimenting.
AUTOPUSH=1

# Which GPUs the measurement runs may use. "all" means do not restrict.
# The reference needs all of them (bf16 does not fit on two cards), but the
# quant runs are deliberately held to a setup the community can relate to.
GPUS=all

use_gpus() {
    if [ -z "$1" ]; then
        echo "use_gpus 0,1     or     use_gpus all       current: $GPUS"
        return 1
    fi
    GPUS=$1
    save_state
    echo "quant measurements will run on GPUs: $GPUS"
}

apply_gpus() {
    if [ "$GPUS" = "all" ]; then
        unset CUDA_VISIBLE_DEVICES
    else
        export CUDA_VISIBLE_DEVICES=$GPUS
    fi
}

token_check() {
    if [ -z "$HF_TOKEN" ]; then
        echo "HF_TOKEN is not set. export the real one, then retry."
        return 1
    fi
    case "$HF_TOKEN" in
        PUT_*|hf_xxx*|CHANGE*)
            echo "HF_TOKEN is still a placeholder: $HF_TOKEN"
            echo "This file must never export a token. export it in your shell."
            return 1 ;;
    esac
    hf auth whoami
}

repo_ok() {
    if [ -z "$METRICS" ]; then
        echo "no preset loaded"
        return 1
    fi
    if python3 -c "
import sys
from huggingface_hub import HfApi
sys.exit(0 if HfApi().repo_exists(sys.argv[1], repo_type=sys.argv[2]) else 1)
" "$METRICS" "$METRICS_KIND" 2>/dev/null; then
        return 0
    fi
    echo
    echo "cannot see $METRICS as a $METRICS_KIND"
    echo
    echo "check the token first:   token_check"
    echo "if the token is fine, create the repo:   make_metrics"
    echo "if it exists as a model repo:   METRICS_KIND=model ; save_state"
    return 1
}

make_metrics() {
    if [ -z "$METRICS" ]; then
        echo "no preset loaded"
        return 1
    fi
    token_check || return 1
    echo "creating $METRICS as a $METRICS_KIND"
    ask "create?" || return 1
    hf repos create $METRICS --repo-type $METRICS_KIND
    repo_ok && echo "reachable now"
}

# Both repos for a release: the quants and the metrics beside them.
make_repos() {
    need_preset || return 1
    token_check || return 1
    echo "  $MAIN            (model)"
    echo "  $METRICS   ($METRICS_KIND)"
    ask "create both?" || return 1
    for pair in "$MAIN model" "$METRICS $METRICS_KIND"; do
        set -- $pair
        if python3 -c "
import sys
from huggingface_hub import HfApi
sys.exit(0 if HfApi().repo_exists(sys.argv[1], repo_type=sys.argv[2]) else 1)
" "$1" "$2" 2>/dev/null; then
            echo "  already there: $1"
        else
            hf repos create $1 --repo-type $2 > /dev/null 2>&1 \
                && echo "  created: $1" || echo "  FAILED: $1"
        fi
    done
    repo_ok && echo "metrics repo reachable"
}

# Clone the calibration dataset with its pipeline. Large pool files live in
# LFS, and a clone without git-lfs brings down pointer stubs instead of data,
# which then fail to parse as JSON somewhere deep inside the build.
get_tools() {
    token_check > /dev/null || return 1
    apt-get install -y -qq git-lfs > /dev/null 2>&1
    if git lfs version > /dev/null 2>&1; then
        git lfs install > /dev/null 2>&1
        HAVE_LFS=1
    else
        echo "git-lfs unavailable, large files will arrive as pointers and get"
        echo "fetched from the hub instead"
        HAVE_LFS=0
    fi

    if [ -d /calib-corpora ]; then
        echo "already at /calib-corpora"
        cd /calib-corpora
        git pull --ff-only 2>&1 || echo "local changes present, keeping them and skipping the pull"
        [ "$HAVE_LFS" = "1" ] && git lfs pull
        cd - > /dev/null
    else
        git clone https://huggingface.co/datasets/AtomicChat/calib-corpora /calib-corpora || return 1
        [ "$HAVE_LFS" = "1" ] && (cd /calib-corpora && git lfs pull)
    fi
    check_pool || true
    echo
    echo "if anything above is a pointer, run:  fix_pool"
}

# Pull the files that came down as pointers straight from the hub. Simpler and
# more reliable than getting git-lfs working on a fresh container.
fix_pool() {
    POINTERS=$(grep -l "^version https://git-lfs" /calib-corpora/pool/*/*.jsonl \
               /calib-corpora/pool/*/*/*.jsonl 2>/dev/null)
    if [ -z "$POINTERS" ]; then
        echo "no pointer files, nothing to fix"
        return 0
    fi
    for f in $POINTERS; do
        REL=${f#/calib-corpora/}
        echo "fetching $REL"
        hf download AtomicChat/calib-corpora --repo-type dataset \
            --include "$REL" --local-dir /calib-corpora > /dev/null || return 1
    done
    check_pool
}

# Every jsonl in the pool has to parse, and an LFS pointer does not. Find the
# bad file by name instead of watching build.py die on an anonymous line.
check_pool() {
    python3 - << 'POOLEOF'
import glob, json, os

bad, pointers, ok = [], [], 0
for path in sorted(glob.glob("/calib-corpora/pool/**/*.jsonl", recursive=True)):
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            first = f.readline()
            if first.startswith("version https://git-lfs"):
                pointers.append(path)
                continue
            json.loads(first)
        ok += 1
    except Exception as e:
        bad.append((path, str(e)[:60]))

print("%d pool files parse" % ok)
if pointers:
    print()
    print("%d files are LFS pointers, not data:" % len(pointers))
    for p in pointers[:10]:
        print("   ", p.replace("/calib-corpora/", ""))
    print()
    print("fix:  fix_pool")
if bad:
    print()
    print("%d files do not parse:" % len(bad))
    for p, e in bad[:10]:
        print("    %s   %s" % (p.replace("/calib-corpora/", ""), e))
if not pointers and not bad:
    print("pool is clean")
POOLEOF
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
    cmake --build build -j $(nproc) --target \
        llama-perplexity llama-bench llama-cli llama-gguf-split \
        llama-imatrix llama-quantize

    ls -la $BIN/
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
    BF16_OUT=/gguf/$(basename $MAIN | sed "s/-GGUF//")-bf16.gguf
    echo "converting /src -> $BF16_OUT"
    echo "If it fails with NaN in token_embd, add --no-lazy and rerun."
    date
    python3 /llama.cpp/convert_hf_to_gguf.py /src --outtype bf16 --outfile $BF16_OUT \
        2>&1 | tee /logs/convert.log
    date
    ls -lh $BF16_OUT
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

# The corpus is built on one box and pulled by the rest. This polls until it
# lands, so the other boxes can be started at the same time.
wait_calib() {
    if [ -z "$1" ]; then
        echo "wait_calib NAME"
        return 1
    fi
    echo "polling AtomicChat/calib-corpora for builds/$1. Ctrl-C to stop."
    while true; do
        if get_calib $1 2>/dev/null; then
            return 0
        fi
        echo "not built yet, $(date +%H:%M:%S), retry in 30s"
        sleep 30
    done
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
    save_state
    echo "context is now $CTX"
    eval_size
}

find_bf16() {
    local stem all
    stem=$(model_stem)
    all=$(find /gguf -name "*.gguf" 2>/dev/null | grep -i bf16 \
          | grep -v "/dflash-" | grep -v "/dspark-")

    if [ -n "$stem" ]; then
        local mine
        mine=$(echo "$all" | grep -F "$stem")
        if [ -z "$mine" ] && [ -n "$all" ]; then
            echo "there are bf16 files here, but none belongs to $stem:"
            echo "$all" | sed "s/^/   /"
            echo "this box was used for another model. Run get_bf16, or"
            echo "make_bf16, or free the disk with clean_gguf."
            return 1
        fi
        all=$mine
    fi

    FOUND=$(echo "$all" | grep "00001-of-" | head -1)
    if [ -z "$FOUND" ]; then
        FOUND=$(echo "$all" | head -1)
    fi
    if [ -z "$FOUND" ]; then
        echo "no bf16 for $stem in /gguf. Run get_bf16 or make_bf16."
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
    hf download $MAIN --include "*bf16*" --include "*BF16*" \
        --exclude "dflash-*" --exclude "dspark-*" --local-dir /gguf
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

# Every construct we calibrate on. Templates disagree about the shape of a tool
# call, so try the common ones and report which one this model accepts: the
# pool records have to match it or the agentic slice renders with no tool
# markup at all.
def convo(args_as_dict, tool_role="tool"):
    args = {"city": "Paris"} if args_as_dict else '{"city": "Paris"}'
    return [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is the weather in Paris?"},
        {"role": "assistant", "content": "", "tool_calls": [
            {"type": "function", "function": {"name": "get_weather", "arguments": args}}]},
        {"role": tool_role, "content": '{"temp_c": 19}', "name": "get_weather"},
        {"role": "assistant", "content": "It is 19 degrees in Paris."},
    ]

tools = [{"type": "function", "function": {
    "name": "get_weather",
    "description": "Get the weather for a city",
    "parameters": {"type": "object", "properties": {"city": {"type": "string"}},
                   "required": ["city"]}}}]
tools_bare = [t["function"] for t in tools]

from jinja2 import Environment
from jinja2.exceptions import TemplateError
env = Environment(trim_blocks=True, lstrip_blocks=True)
env.policies["json.dumps_kwargs"] = {"ensure_ascii": False}
try:
    tpl = env.from_string(template)
except Exception as e:
    print("FAIL  the template does not even parse as jinja: %s" % e)
    sys.exit(1)

attempts = [
    ("tool_calls arguments as a DICT, tools wrapped in type/function",
     dict(messages=convo(True), tools=tools, add_generation_prompt=True)),
    ("tool_calls arguments as a DICT, bare function schemas",
     dict(messages=convo(True), tools=tools_bare, add_generation_prompt=True)),
    ("tool_calls arguments as a JSON STRING",
     dict(messages=convo(False), tools=tools, add_generation_prompt=True)),
    ("no tools at all",
     dict(messages=convo(True)[:2] + convo(True)[4:], add_generation_prompt=True)),
]

rendered = None
winner = None
for label, kwargs in attempts:
    try:
        out = tpl.render(bos_token="", eos_token="", **kwargs)
    except Exception as e:
        print("  no    %-56s %s" % (label, str(e)[:60]))
        continue
    has_tool = "get_weather" in out
    print("  ok    %-56s tool markup: %s" % (label, "yes" if has_tool else "NO"))
    if rendered is None or (has_tool and "get_weather" not in rendered):
        rendered, winner = out, label
    if has_tool:
        break

if rendered is None:
    print("FAIL  the template will not render any of our conversations")
    sys.exit(1)

print()
print("using: %s" % winner)
if "get_weather" not in rendered:
    print()
    print("WARNING  no shape produced tool markup. The agentic slice of the")
    print("         corpus would render with none of it, and the imatrix would")
    print("         have no statistics for the tool token set at all.")
    print("         Check how the pool stores tool_calls before building.")

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
print("markers the template emits:")
special, plain = [], []
for m in markers:
    (special if m in known else plain).append(m)

for m in special:
    print("  special token   %s" % m)
for m in plain:
    print("  plain text      %s" % m)

print()
if not special:
    print("WARNING  not one marker is a real token. That usually means this is")
    print("         not the model's own template, or the tokenizer files do not")
    print("         match the weights. Do not build a corpus on this.")
else:
    print("RESULT  %d real special tokens, %d plain text delimiters." % (len(special), len(plain)))
    print("        Plain delimiters are fine: the model was trained on this same")
    print("        template, so it reads them as ordinary text at inference too,")
    print("        and the corpus matches. What matters is the special ones:")
    print("        run llama-imatrix with --parse-special or they get tokenized")
    print("        as punctuation and that part of the corpus calibrates nothing.")
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
    echo "Writing the reference. This one uses ALL GPUs regardless of use_gpus,"
    echo "because bf16 does not fit on fewer. Say so in the model card."
    echo "Warnings about unused blk.52.nextn tensors are normal: that is the MTP"
    echo "block, which a plain forward pass never executes."
    date
    START=$(date +%s)

    unset CUDA_VISIBLE_DEVICES
    stdbuf -oL -eL $BIN/llama-perplexity -m $BF16_FIRST -f $EVAL \
        --kl-divergence-base $BASE -c $CTX -ngl 99 \
        2>&1 | tee /logs/base-$EVALSET.log

    autopush /logs/base-$EVALSET.log logs/base-$EVALSET.log

    echo "took $(( $(date +%s) - START )) seconds"
    echo
    echo "=== SIZE OF THE REFERENCE ==="
    ls -lh $BASE
    echo
    echo "This file holds fp16 logits over the whole vocabulary for every scored"
    echo "token, so it is large by design and it is NOT worth shipping anywhere."
    echo "It reproduces from the bf16 weights in the time you just watched."
    echo "Measure the quants on this same box, and publish the log, not the blob."
}

kld() {
    if [ ! -f "$1" ]; then echo "no model at $1"; return 1; fi
    if [ ! -f $BASE ]; then echo "no reference at $BASE. Run base, or get_base."; return 1; fi

    NAME=$(basename $1 .gguf)
    apply_gpus
    echo "--------------------------------------------------"
    echo "KLD on $EVALSET: $NAME   (GPUs: $GPUS)"
    START=$(date +%s)

    stdbuf -oL -eL $BIN/llama-perplexity -m $1 -f $EVAL \
        --kl-divergence-base $BASE --kl-divergence -c $CTX -ngl 99 \
        2>&1 | tee /logs/kld-$EVALSET--$NAME.log

    echo "took $(( $(date +%s) - START )) seconds"
    autopush /logs/kld-$EVALSET--$NAME.log logs/kld-$EVALSET--$NAME.log
}

kld_all() {
    FILES=$(quant_files)
    if [ -z "$FILES" ]; then
        echo "no quants in /gguf. Run get_quants."
        echo "what is there now:"
        find /gguf -name "*.gguf" | sed "s/^/   /"
        return 1
    fi
    TOTAL=$(echo "$FILES" | wc -l)
    N=0
    for f in $FILES; do
        N=$(( N + 1 ))
        echo
        echo "########## $N of $TOTAL, roughly $(( (TOTAL - N) * 3 )) minutes left ##########"
        kld $f
    done
    results
}

# Measure other publishers' builds on OUR reference. Numbers taken against
# different references cannot be put in the same table, so their published
# figures are not usable directly, only their files are.
kld_ext() {
    local f n=0 total
    total=$(ls /gguf/external/*.gguf 2>/dev/null | wc -l)
    if [ "$total" = "0" ]; then
        echo "nothing in /gguf/external. Pull some with:"
        echo "  get_external unsloth/Qwen3.8-27B-GGUF '*UD-IQ3_XXS*'"
        return 1
    fi
    for f in /gguf/external/*.gguf; do
        n=$(( n + 1 ))
        echo
        echo "########## external $n of $total ##########"
        kld "$f"
    done
    results
}

bench() {
    NAME=$(basename $1 .gguf)
    echo "GPU 0 only, so the numbers stay comparable to the published card."
    CUDA_VISIBLE_DEVICES=0 $BIN/llama-bench -m $1 -p 512 -n 128 -ngl 99 -r 5 -o json \
        > /logs/bench-$NAME.json
    head -40 /logs/bench-$NAME.json
    autopush /logs/bench-$NAME.json logs/bench-$NAME.json
}

bench_all() {
    FILES=$(quant_files)
    if [ -z "$FILES" ]; then
        echo "no quants in /gguf. Run get_quants."
        return 1
    fi
    for f in $FILES; do
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


# ================================================================== new model scaffold

# make_recipe NAME UPSTREAM_REPO
# Reads config.json, tokenizer_config.json and the chat template straight off
# the hub and writes a complete recipe. Nothing is left for you to fill in by
# hand except the domain shares, which are a judgement call about the model's
# use, not a fact you can read out of a config file.
make_recipe() {
    if [ -z "$2" ]; then
        echo "make_recipe NAME UPSTREAM_REPO"
        echo "  make_recipe qwen3.8-27b Qwen/Qwen3.8-27B"
        return 1
    fi
    pip install --break-system-packages -q -U jinja2 pyyaml > /dev/null 2>&1
    mkdir -p /recipes
    python3 - "$1" "$2" << 'RECIPEEOF'
import json, os, re, sys
from huggingface_hub import hf_hub_download

name, repo = sys.argv[1], sys.argv[2]

# build.py resolves chat.format through FORMATTERS. auto_fmt drives the model's
# own template through transformers, so one renderer covers every model and no
# per-model module has to be written.
fmt_key = "auto"

def grab(fn):
    try:
        return open(hf_hub_download(repo, fn), encoding="utf-8").read()
    except Exception:
        return None

cfg_raw = grab("config.json")
if cfg_raw is None:
    print("cannot read config.json from %s" % repo)
    sys.exit(1)
cfg = json.loads(cfg_raw)
tok_cfg = json.loads(grab("tokenizer_config.json") or "{}")

# some repos nest the real config under text_config
inner = cfg.get("text_config") or cfg

def pick(*keys, default=None):
    for k in keys:
        for src_ in (inner, cfg):
            if src_.get(k) is not None:
                return src_[k]
    return default

vocab   = pick("vocab_size", default=0)
layers  = pick("num_hidden_layers", "n_layer", default=0)
ctxlen  = pick("max_position_embeddings", default=0)
window  = pick("sliding_window", default=None)
hidden  = pick("hidden_size", default=0)
inter   = pick("intermediate_size", default=0)
moe     = pick("moe_intermediate_size", default=None)
experts = pick("num_experts", "n_routed_experts", "num_local_experts", default=None)

print("read from %s:" % repo)
for k, v in (("vocab_size", vocab), ("layers", layers), ("context", ctxlen),
             ("sliding_window", window), ("hidden_size", hidden),
             ("intermediate_size", inter), ("moe_intermediate_size", moe),
             ("experts", experts)):
    print("  %-22s %s" % (k, v))

print()
print("superblock check, rows must divide by 256 or every k/i quant below")
print("4.5 bpw silently falls back to a block-32 type:")
shape_lines = []
for k, v in (("hidden_size", hidden), ("intermediate_size", inter),
             ("moe_intermediate_size", moe)):
    if not v:
        continue
    rem = v % 256
    verdict = "ok" if rem == 0 else "BREAKS k/i quants"
    print("  %-22s %6d   %%256 = %3d   %s" % (k, v, rem, verdict))
    shape_lines.append("%s = %d (%%256 = %d)" % (k, v, rem))

# chat template
template = grab("chat_template.jinja")
where = "chat_template.jinja"
if template is None:
    template = tok_cfg.get("chat_template")
    where = "tokenizer_config.json"
if isinstance(template, list):
    template = template[0].get("template")

add_bos = bool(tok_cfg.get("add_bos_token", False))
uses_date = bool(template and "strftime_now" in template)
thinking = bool(template and re.search(r"enable_thinking|<think>", template))

print()
if template:
    print("chat template found in %s, %d chars" % (where, len(template)))
else:
    print("NO CHAT TEMPLATE. The corpus cannot be rendered the way the model")
    print("actually reads text. Run corpus_check and decide before building.")
print("  add_bos_token       : %s" % add_bos)
print("  calls strftime_now  : %s%s" % (uses_date, "  <- pinned below" if uses_date else ""))
print("  mentions thinking   : %s" % thinking)

known_renders = ["dsv4", "nemotron", "muse-glimmer"]
excl = [r for r in known_renders if r not in name]

reasoning_share = 15 if thinking else 10
code_share = 18 if thinking else 23
longctx_share = 12 if window is None else 10

body = """# Calibration recipe for {repo}.
# Generated by foundry make_recipe. Everything under model: was read from the
# hub, not typed. The shares below are the one human judgement call: they say
# what this model is for, which no config file knows.
#
# Architecture facts that matter here:
{shapes}
#
# Shares:
#   agentic 25      tool markup is a dense token set natural text never emits,
#                   so without real weight the imatrix has no statistics for it
#   code {code}         the agentic evals in this class are code shaped
#   reasoning {reason}    {reason_why}
#   multilingual 14 non-Latin embedding rows are only exercised by real
#                   multilingual text
#   longctx {lctx}      {lctx_why}
#   vocab_sweep 10  regenerated against THIS tokenizer, {vocab} rows. A sweep
#                   built for another model is meaningless here
#   structured 5    JSON, YAML, TOML, SQL, as files and inside assistant turns
#   graphics 3      kept small for continuity with the other builds

name: {name}

model:
  repo: {repo}
  tokenizer: /src/tokenizer.json
  gguf: /gguf/{name}-bf16.gguf
  vocab_size: {vocab}
  layers: {layers}
  sliding_window: {window}
  context_length: {ctxlen}
  llama_cpp_commit: FILLED_AT_BUILD_TIME

seed: 20260814

chat:
  # auto_fmt drives this model's own chat_template through transformers, so
  # byte equality with inference holds by construction. Install it once with
  # install_auto_fmt; no per-model renderer is needed.
  format: {fmt}
  add_bos_per_document: {add_bos}
{date_pin}
calib_train:
  target_tokens: 5000000
  max_doc_fraction:
    default: 0.05
    longctx: 0.12
  shares:
    agentic: 25
    code: {code}
    reasoning: {reason}
    multilingual: 14
    longctx: {lctx}
    vocab_sweep: 10
    structured: 5
    graphics: 3

calib_longctx:
  target_tokens: 750000
  doc_tokens_min: 16384
  doc_tokens_max: 32768
  sources: [longctx]

exclude:
  # Anything pre-rendered with another model's markup. Those byte sequences are
  # not special tokens for this tokenizer, so they would calibrate on text this
  # model never sees.
  render: [{excl}]
  provenance_key: excluded_from_builds

notes:
  imatrix: >
    llama-imatrix defaults to parse_special = false. Without --parse-special the
    chat markup in calib_train.txt is tokenised as literal punctuation and the
    agentic and reasoning slices calibrate on text the model never sees.
""".format(
    repo=repo, name=name, fmt=fmt_key, vocab=vocab, layers=layers, ctxlen=ctxlen,
    window="null" if window is None else window,
    add_bos="true" if add_bos else "false",
    code=code_share, reason=reasoning_share, lctx=longctx_share,
    reason_why=("the thinking channel is on the default inference path here"
                if thinking else "no thinking channel found in the template"),
    lctx_why=("no sliding window, so long range behaviour is only exercised by "
              "long documents" if window is None else
              "sliding window present, short documents already exercise it"),
    shapes="\n".join("#   " + s for s in shape_lines) or "#   (no shape info)",
    excl=", ".join(excl),
    date_pin=('  # the template calls strftime_now, so the date is pinned or two\n'
              '  # builds on different days would differ\n'
              '  pin_date: "2026-08-14"\n' if uses_date else ""),
)

out = "/recipes/%s.yaml" % name
open(out, "w").write(body)
print()
if vocab:
    per_100k = vocab * 2 * 100000 / 1e9
    print()
    print("vocabulary is %d rows, so the kld reference costs about %.0f GB per" % (vocab, per_100k))
    print("100k scored tokens. At ctx 4096 with a 400k token corpus that is")
    print("roughly %.0f GB. Plan disk for it before running base." % (per_100k * 2))

print("wrote %s" % out)
print()
print("one thing this file cannot do for you: the domain shares are a claim")
print("about what this model is for. Read them once.")
RECIPEEOF
}

# One command for a brand new model: validate the template, then write a recipe
# with every architecture field already filled in.
new_model() {
    if [ -z "$2" ]; then
        echo "new_model NAME UPSTREAM_REPO"
        echo "  new_model qwen3.8-27b Qwen/Qwen3.8-27B"
        return 1
    fi
    echo "=============== 1. chat template and special tokens ==============="
    if ! corpus_check $2; then
        echo
        echo "Stopping. Nothing below can work without the model's own files."
        echo "If the repo id is wrong, find it with:   find_repo <part of the name>"
        return 1
    fi
    echo
    echo "=============== 2. recipe ==============="
    make_recipe $1 $2 || return 1
    echo
    echo "=============== 3. next ==============="
    echo "  cat /recipes/$1.yaml"
    echo "  get_tools                     # clone calib-corpora"
    echo "  cp /recipes/$1.yaml /calib-corpora/recipes/"
    echo "  then build the corpus with the pipeline in /calib-corpora/tools"
}

# Interactive picker for the model the imatrix is collected on. bf16 first:
# q8 is a memory compromise, not a better choice.
pick_model() {
    FILES=$(find /gguf -maxdepth 2 -name "*.gguf" 2>/dev/null \
            | grep -v "/dflash-" | grep -v "/dspark-" | grep -v -i "imatrix" \
            | grep -v -- "-0000[2-9]-of-" | sort)
    if [ -z "$FILES" ]; then
        echo "nothing in /gguf yet"
        return 1
    fi
    echo
    echo "bf16 is the right choice when it fits. q8 is a compromise for memory."
    echo
    PS3="
number: "
    select f in $FILES "cancel"; do
        if [ "$f" = "cancel" ] || [ -z "$f" ]; then
            break
        fi
        IM_MODEL=$f
        save_state
        echo "IM_MODEL = $IM_MODEL"
        ls -lh $IM_MODEL
        break
    done
}


# ================================================================== corpus build

# One generic renderer instead of one per model. Installs auto_fmt.py into the
# pipeline and registers it in build.py's FORMATTERS.
install_auto_fmt() {
    if [ ! -d /calib-corpora ]; then
        echo "no pipeline yet. Run get_tools."
        return 1
    fi
    curl -sL https://raw.githubusercontent.com/worthant/quantizer/main/auto_fmt.py \
        -o /calib-corpora/tools/auto_fmt.py || return 1

    if grep -q '"auto"' /calib-corpora/tools/build.py; then
        echo "already registered in FORMATTERS"
    else
        sed -i 's/^FORMATTERS = {/FORMATTERS = {\n    "auto":     "auto_fmt",/' \
            /calib-corpora/tools/build.py
        echo "registered in FORMATTERS"
    fi
    sed -n '/^FORMATTERS = {/,/^}/p' /calib-corpora/tools/build.py
}

setup_corpus() {
    pip install --break-system-packages -q -U transformers pyyaml jinja2 sentencepiece
    python3 -c "import transformers; print('transformers', transformers.__version__)"
    echo "If a converter later complains about RoPE, pin transformers below 5."
}

# build_corpus NAME
# Needs: the original weights in /src, the recipe in /recipes, get_tools done.
build_corpus() {
    if [ -z "$1" ]; then
        echo "build_corpus NAME        e.g. build_corpus qwen3.8-27b"
        return 1
    fi
    NAME=$1

    if [ ! -f /src/tokenizer.json ]; then
        echo "no tokenizer at /src/tokenizer.json. Run get_upstream."
        return 1
    fi
    if [ ! -d /calib-corpora ]; then
        echo "no pipeline. Run get_tools."
        return 1
    fi
    if [ ! -f /calib-corpora/tools/auto_fmt.py ]; then
        echo "generic renderer not installed. Run install_auto_fmt."
        return 1
    fi
    if [ ! -f /recipes/$NAME.yaml ]; then
        echo "no recipe at /recipes/$NAME.yaml. Run make_recipe or new_model."
        return 1
    fi

    export FOUNDRY_MODEL_DIR=/src
    PIN=$(grep -o 'pin_date: *"[^"]*"' /recipes/$NAME.yaml | cut -d'"' -f2)
    if [ -n "$PIN" ]; then
        export FOUNDRY_PIN_DATE=$PIN
        echo "date pinned to $PIN, so this build is reproducible"
    fi

    cd /calib-corpora

    check_pool || return 1

    echo
    echo "=============== vocabulary sweep for THIS tokenizer ==============="
    echo "A sweep built for another model covers a different vocabulary and is"
    echo "worthless here, so this is regenerated every time."

    # Compare before against after. A file can already be dirty for reasons
    # that have nothing to do with the sweep: repairing an LFS pointer replaces
    # the stub with real data, which git reports as a modification.
    git -C /calib-corpora status --porcelain pool/vocab_sweep/synthetic \
        | sort > /tmp/sweep-before.txt

    python3 tools/vocab_sweep.py --tokenizer /src/tokenizer.json --name $NAME \
        2>&1 | tee /logs/vocab-sweep-$NAME.log

    git -C /calib-corpora status --porcelain pool/vocab_sweep/synthetic \
        | sort > /tmp/sweep-after.txt

    echo
    echo "what the sweep itself changed:"
    comm -13 /tmp/sweep-before.txt /tmp/sweep-after.txt | sed "s/^/  /"
    echo "  (already dirty before the sweep, ignored:)"
    comm -12 /tmp/sweep-before.txt /tmp/sweep-after.txt | sed "s/^/    /"

    TOUCHED=$(comm -13 /tmp/sweep-before.txt /tmp/sweep-after.txt \
              | grep "^ M" | awk '{print $2}')
    for f in $TOUCHED; do
        case "$f" in
            *$NAME*) ;;
            *) echo
               echo "!! $f is not named after $NAME."
               echo "   The sweep wrote over another model's file. That file is"
               echo "   built for a different vocabulary and is now gone."
               echo "   Restore it and give vocab_sweep an output name:"
               echo "     git -C /calib-corpora checkout -- $f"
               echo "     python3 tools/vocab_sweep.py --help"
               return 1 ;;
        esac
    done

    echo
    echo "=============== building the corpus ==============="
    date
    python3 tools/build.py --recipe /recipes/$NAME.yaml 2>&1 | tee /logs/build-$NAME.log
    date

    echo
    echo "=============== what appeared ==============="
    ls -la /calib-corpora/builds/$NAME/ 2>/dev/null || find /calib-corpora/builds -newer /calib-corpora/recipes/$NAME.yaml

    if [ -f /calib-corpora/builds/$NAME/calib_train.txt ]; then
        cp /calib-corpora/builds/$NAME/calib_train.txt $IM_CORPUS
        im_size
        echo
        echo "publish it so the other boxes can pull the same bytes:"
        echo "  push_corpus $NAME"
    fi
    cd -
}

push_corpus() {
    if [ -z "$1" ]; then
        echo "push_corpus NAME"
        return 1
    fi
    token_check > /dev/null || return 1
    if [ ! -f /calib-corpora/builds/$1/calib_train.txt ]; then
        echo "no corpus at /calib-corpora/builds/$1/calib_train.txt"
        echo "the directory may exist and be empty from an earlier failed run."
        echo "run build_corpus $1 and read its output before pushing."
        return 1
    fi
    ls -la /calib-corpora/builds/$1
    cd /calib-corpora
    hf_put_dir builds/$1 builds/$1 AtomicChat/calib-corpora dataset
    hf_put /recipes/$1.yaml recipes/$1.yaml AtomicChat/calib-corpora dataset
    cd -
    echo "every box can now: get_calib $1"
}


# ================================================================== imatrix, fanned out
#
# Splitting by chunk range, not by cutting the text file. Every node reads the
# SAME corpus and takes its own slice with --from-chunk / --chunks, so the union
# is bit-for-bit the chunking a single node would have produced. Merging is a
# sum of squared activations, so --in-file gives the same answer as one long run.

IM_CTX=512
IM_BATCH=8192

# --no-ppl skips the perplexity pass, which is a few percent faster. It also
# removes the per-chunk line that was the only sign of progress, leaving a
# forty minute run completely silent. Visibility is worth more than the few
# percent, so it is off by default. Set IM_NO_PPL=1 to get it back.
IM_NO_PPL=${IM_NO_PPL:-0}

# How often llama-imatrix reports that it stored collected data. Also a
# heartbeat.
IM_OFREQ=20
IM_CORPUS=/eval/calib_train.txt
IM_MODEL=""

get_calib() {
    if [ -z "$1" ]; then
        echo "get_calib NAME      e.g. get_calib nemotron-3.5-lightning"
        echo "see what builds exist with: ls_corpora"
        return 1
    fi
    fetch_one AtomicChat/calib-corpora dataset "builds/$1/calib_train.txt" $IM_CORPUS || return 1
    im_size
}

im_size() {
    if [ ! -f $IM_CORPUS ]; then
        echo "no calibration corpus at $IM_CORPUS. Run get_calib."
        return 1
    fi
    BYTES=$(stat -c %s $IM_CORPUS)
    if [ -n "$IM_TOKENS_EXACT" ]; then
        IM_TOKENS=$IM_TOKENS_EXACT
        SOURCE="exact, from the build log"
    else
        IM_TOKENS=$(( BYTES / 4 ))
        SOURCE="estimated at 4 bytes per token, usually low"
    fi
    IM_CHUNKS=$(( IM_TOKENS / IM_CTX ))
    echo "corpus  : $IM_CORPUS"
    echo "size    : $BYTES bytes, $IM_TOKENS tokens ($SOURCE)"
    echo "at ctx $IM_CTX that is about $IM_CHUNKS chunks"
    if [ -z "$IM_TOKENS_EXACT" ]; then
        echo "build.py printed the real token count. Set it so the shards are even:"
        echo "  IM_TOKENS_EXACT=<number> ; save_state"
    fi
}

# im_plan N  ->  prints the exact command for each of N nodes. Paste one per box.
im_plan() {
    if [ -z "$1" ] || [ -z "$IM_MODEL" ]; then
        echo "im_plan N          after setting IM_MODEL=/gguf/<the q8 or bf16>.gguf"
        echo "current IM_MODEL: ${IM_MODEL:-not set}"
        return 1
    fi
    im_size || return 1
    N=$1
    PER=$(( IM_CHUNKS / N + 1 ))
    echo
    echo "$N nodes, about $PER chunks each"
    echo "run get_calib and set IM_MODEL to the same file on every box, then:"
    echo
    I=0
    while [ $I -lt $N ]; do
        echo "  node $I:   im_shard $I $N"
        I=$(( I + 1 ))
    done
    echo
    echo "then on any one box:   im_merge $N"
}

im_shard() {
    if [ -z "$2" ]; then
        echo "im_shard INDEX TOTAL        index is 0 based"
        return 1
    fi
    if [ ! -f "$IM_MODEL" ]; then
        echo "IM_MODEL is not set or missing: ${IM_MODEL:-unset}"
        return 1
    fi
    im_size || return 1

    # The logits tensor is batch x vocab x 4 bytes and lives in VRAM alongside
    # the weights. A batch tuned for a small vocabulary will OOM on a large one.
    VOCAB=$(python3 -c "
import sys
sys.path.insert(0, '/llama.cpp/gguf-py')
from gguf import GGUFReader
r = GGUFReader('$IM_MODEL')
for t in r.tensors:
    if 'token_embd' in t.name or t.name == 'output.weight':
        print(int(t.shape[1]) if len(t.shape) > 1 else 0); break
else:
    print(0)
" 2>/dev/null)
    VOCAB=$(echo "$VOCAB" | head -1 | tr -dc '0-9')
    if [ -n "$VOCAB" ] && [ "$VOCAB" -gt 0 ] 2>/dev/null; then
        LOGIT_GB=$(( VOCAB * IM_BATCH * 4 / 1000000000 ))
        MODEL_GB=$(( $(stat -c %s $IM_MODEL) / 1000000000 ))
        # Only the cards this process can see. Two shards on one box each get
        # their own pair through CUDA_VISIBLE_DEVICES, so counting all of them
        # would say everything fits when it does not.
        PER_GPU=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
        if [ -n "$CUDA_VISIBLE_DEVICES" ]; then
            NGPU=$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l)
        else
            NGPU=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | wc -l)
        fi
        VRAM_GB=$(( PER_GPU * NGPU / 1000 ))
        echo "visible GPUs: ${CUDA_VISIBLE_DEVICES:-all} ($NGPU cards)"
        echo "vocabulary $VOCAB, batch $IM_BATCH"
        echo "weights ~${MODEL_GB} GB + logits ~${LOGIT_GB} GB against ${VRAM_GB} GB of VRAM"
        if [ $(( MODEL_GB + LOGIT_GB + 4 )) -gt $VRAM_GB ]; then
            echo
            echo "That does not fit. Lower the batch and try again:"
            echo "  IM_BATCH=2048 ; im_shard $1 $2"
            return 1
        fi
    fi

    I=$1
    N=$2
    PER=$(( IM_CHUNKS / N + 1 ))
    FROM=$(( I * PER ))
    SHARD_OUT=/imatrix/shard-$I-of-$N.gguf
    mkdir -p /imatrix

    echo "node $I of $N: chunks $FROM onward"
    if [ $I -eq $(( N - 1 )) ]; then
        echo "last node, runs to the end of the corpus"
        LIMIT=""
    else
        echo "taking $PER chunks"
        LIMIT="--chunks $PER"
    fi
    date
    START=$(date +%s)

    PPLFLAG=""
    [ "$IM_NO_PPL" = "1" ] && PPLFLAG="--no-ppl"

    # stdbuf keeps the output line buffered. Without it the pipe into tee
    # switches libc to 4 KB blocks and nothing appears for many minutes.
    stdbuf -oL -eL $BIN/llama-imatrix -m $IM_MODEL -f $IM_CORPUS -o $SHARD_OUT \
        -ngl 99 -c $IM_CTX -b $IM_BATCH -ub $IM_BATCH \
        --parse-special $PPLFLAG --output-frequency $IM_OFREQ \
        --from-chunk $FROM $LIMIT \
        2>&1 | tee /logs/imatrix-shard-$I-of-$N.log

    echo "took $(( $(date +%s) - START )) seconds"
    ls -lh $SHARD_OUT
    autopush /logs/imatrix-shard-$I-of-$N.log logs/imatrix-shard-$I-of-$N.log
    if [ -f "$SHARD_OUT" ]; then
        hf_put "$SHARD_OUT" "imatrix/shard-$I-of-$N.gguf" "$METRICS" "$METRICS_KIND" \
            && echo "  shard uploaded, the merging box can pick it up"
    fi
}

# im_range FROM COUNT LABEL
# im_shard splits evenly, which assumes every box runs at the same speed. They
# do not: rented machines differ by half again on the same cards, and the
# slowest one ends up on the critical path. This takes an explicit range so a
# long shard can be cut up and handed to whoever is free.
#
#   im_range 7275 700 5a
#   im_range 7975 0   5b      count 0 means "to the end of the corpus"
im_range() {
    if [ -z "$2" ] || [ -z "$3" ]; then
        echo "im_range FROM COUNT LABEL      count 0 runs to the end"
        echo "  im_range 7275 700 5a"
        return 1
    fi
    if [ ! -f "$IM_MODEL" ]; then
        echo "IM_MODEL is not set or missing: ${IM_MODEL:-unset}"
        return 1
    fi
    FROM=$1
    COUNT=$2
    SHARD_OUT=/imatrix/shard-$3.gguf
    mkdir -p /imatrix

    if [ "$COUNT" = "0" ]; then
        LIMIT=""
        echo "chunks $FROM to the end of the corpus"
    else
        LIMIT="--chunks $COUNT"
        echo "chunks $FROM to $(( FROM + COUNT - 1 ))"
    fi
    date
    START=$(date +%s)

    PPLFLAG=""
    [ "$IM_NO_PPL" = "1" ] && PPLFLAG="--no-ppl"

    stdbuf -oL -eL $BIN/llama-imatrix -m $IM_MODEL -f $IM_CORPUS -o $SHARD_OUT \
        -ngl 99 -c $IM_CTX -b $IM_BATCH -ub $IM_BATCH \
        --parse-special $PPLFLAG --output-frequency $IM_OFREQ \
        --from-chunk $FROM $LIMIT \
        2>&1 | tee /logs/imatrix-shard-$3.log

    echo "took $(( $(date +%s) - START )) seconds"
    ls -lh $SHARD_OUT 2>/dev/null
    autopush /logs/imatrix-shard-$3.log logs/imatrix-shard-$3.log
    if [ -f "$SHARD_OUT" ]; then
        hf_put "$SHARD_OUT" "imatrix/shard-$3.gguf" "$METRICS" "$METRICS_KIND" \
            && echo "  shard uploaded"
    fi
}

# Merge whatever shards are present, whatever they are called. Use this after
# rebalancing with im_range, when the names no longer follow I-of-N.
# Upload every shard sitting in /imatrix that is not in the repo yet. Safe to
# run as often as you like. Use it whenever a shard finished in a pane that had
# an older copy of this file loaded and never got to its own upload line.
push_shards() {
    need_preset || return 1
    local f name
    for f in /imatrix/shard-*.gguf; do
        [ -f "$f" ] || continue
        name=$(basename "$f")
        echo "$name"
        hf_put "$f" "imatrix/$name" "$METRICS" "$METRICS_KIND" \
            && echo "  up" || echo "  failed"
    done
}

# What is on this box, what is in the repo, and whether the chunk ranges cover
# the corpus without a gap.
im_status() {
    need_preset || return 1
    echo "on this box:"
    ls -lh /imatrix/shard-*.gguf 2>/dev/null | awk '{print "  ", $9, $5}' || echo "   none"
    echo
    echo "in $METRICS:"
    python3 - "$METRICS" "$METRICS_KIND" << 'IMSTEOF'
import sys
from huggingface_hub import HfApi
repo, kind = sys.argv[1], sys.argv[2]
try:
    files = [f for f in HfApi().list_repo_files(repo, repo_type=kind)
             if f.startswith("imatrix/shard-")]
except Exception as e:
    print("   cannot read the repo: %s" % str(e).splitlines()[0])
    sys.exit(0)
if not files:
    print("   none yet")
for f in sorted(files):
    print("  ", f)
print()
print("%d shard(s) published" % len(files))
IMSTEOF
    echo
    echo "chunk ranges are in the logs:"
    grep -h "computing over\|removing initial" /logs/imatrix-shard-*.log 2>/dev/null \
        | sed "s/^/  /" | tail -20
}

# Merge whatever shards are present, whatever they are called.
#
# IM_SKIP holds substrings of shard names to leave out. A shard that was killed
# still left a partial file behind, because llama-imatrix writes snapshots as it
# goes, and merging that alongside the ranges that replaced it counts those
# chunks twice.
#
#   IM_SKIP="shard-5-of-6" im_merge_all
im_merge_all() {
    need_preset || return 1
    repo_ok || return 1
    if [ ! -f "$IM_MODEL" ]; then
        echo "IM_MODEL is not set. Run pick_model."
        return 1
    fi
    mkdir -p /imatrix
    hf download $METRICS --repo-type $METRICS_KIND --include "imatrix/shard-*" --local-dir /tmp/im > /dev/null
    cp -n /tmp/im/imatrix/shard-*.gguf /imatrix/ 2>/dev/null

    local f keep="" skipped="" pat
    for f in $(ls /imatrix/shard-*.gguf 2>/dev/null | sort); do
        for pat in $IM_SKIP; do
            case "$f" in *$pat*) skipped="$skipped $f"; continue 2 ;; esac
        done
        keep="$keep,$f"
    done
    keep=${keep#,}

    if [ -z "$keep" ]; then
        echo "no shards to merge"
        return 1
    fi
    echo "merging:"
    echo "$keep" | tr ',' '\n' | sed "s/^/  /"
    if [ -n "$skipped" ]; then
        echo "skipped (IM_SKIP):"
        for f in $skipped; do echo "  $f"; done
    fi
    echo

    # This build wants one comma separated list, not a repeated flag, and it
    # wants the model even for a pure merge.
    stdbuf -oL -eL $BIN/llama-imatrix -m "$IM_MODEL" --in-file "$keep" \
        -o /imatrix/imatrix.gguf 2>&1 | tee /logs/imatrix-merge.log

    if [ ! -f /imatrix/imatrix.gguf ]; then
        echo "merge produced nothing, read the log above"
        autopush /logs/imatrix-merge.log logs/imatrix-merge.log
        return 1
    fi
    ls -lh /imatrix/imatrix.gguf

    echo
    echo "=== merged statistics ==="
    stdbuf -oL -eL $BIN/llama-imatrix -m "$IM_MODEL" --in-file /imatrix/imatrix.gguf \
        --show-statistics 2>&1 | tee /logs/imatrix-stats.log | head -30

    autopush /logs/imatrix-merge.log logs/imatrix-merge.log
    autopush /logs/imatrix-stats.log logs/imatrix-stats.log
    hf_put /imatrix/imatrix.gguf imatrix/imatrix.gguf "$METRICS" "$METRICS_KIND"
}

im_merge() {
    if [ -z "$1" ]; then
        echo "im_merge N          N is the number of shards"
        return 1
    fi
    need_preset || return 1
    repo_ok || return 1
    N=$1
    mkdir -p /imatrix

    echo "pulling shards from $METRICS"
    hf download $METRICS --repo-type $METRICS_KIND --include "imatrix/*" --local-dir /imatrix
    find /imatrix -name "shard-*-of-$N.gguf" | sort

    GOT=$(find /imatrix -name "shard-*-of-$N.gguf" | wc -l)
    if [ $GOT -ne $N ]; then
        echo "only $GOT of $N shards are here. Wait for the rest, or pass the real count."
        return 1
    fi

    LIST=$(find /imatrix -name "shard-*-of-$N.gguf" | sort | paste -sd,)
    echo "merging: $LIST"

    stdbuf -oL -eL $BIN/llama-imatrix -m "$IM_MODEL" --in-file "$LIST" \
        -o /imatrix/imatrix.gguf 2>&1 | tee /logs/imatrix-merge.log
    ls -lh /imatrix/imatrix.gguf

    echo
    echo "=== merged statistics ==="
    stdbuf -oL -eL $BIN/llama-imatrix -m "$IM_MODEL" --in-file /imatrix/imatrix.gguf \
        --show-statistics 2>&1 | tee /logs/imatrix-stats.log | head -30
    echo
    echo "Sanity check: the sum of squared activations here should equal the sum"
    echo "across the shards. Compare against im_stats on each shard."

    autopush /logs/imatrix-merge.log logs/imatrix-merge.log
    autopush /logs/imatrix-stats.log logs/imatrix-stats.log
    hf_put /imatrix/imatrix.gguf imatrix/imatrix.gguf "$METRICS" "$METRICS_KIND"
}

# Block until the corpus this build needs shows up in the dataset.
wait_calib() {
    if [ -z "$1" ]; then
        echo "wait_calib NAME"
        return 1
    fi
    echo "waiting for builds/$1/calib_train.txt in calib-corpora. Ctrl-C to stop."
    while true; do
        if fetch_one AtomicChat/calib-corpora dataset "builds/$1/calib_train.txt" $IM_CORPUS 2>/dev/null; then
            im_size
            return 0
        fi
        echo "not published yet, $(date +%H:%M:%S), retry in 20s"
        sleep 20
    done
}

# Block until every node has uploaded its shard, then merge.
im_wait_merge() {
    if [ -z "$1" ]; then
        echo "im_wait_merge N"
        return 1
    fi
    need_preset || return 1
    N=$1
    echo "waiting for $N shards in $METRICS. Ctrl-C to stop."
    while true; do
        hf download $METRICS --repo-type $METRICS_KIND --include "imatrix/*" --local-dir /imatrix > /dev/null 2>&1
        GOT=$(find /imatrix -name "shard-*-of-$N.gguf" 2>/dev/null | wc -l)
        echo "  $GOT of $N here, $(date +%H:%M:%S)"
        if [ "$GOT" -ge "$N" ]; then
            break
        fi
        sleep 20
    done
    im_merge $N
}

im_stats() {
    if [ -z "$1" ]; then
        echo "im_stats FILE.gguf"
        return 1
    fi
    $BIN/llama-imatrix -m "$IM_MODEL" --in-file "$1" --show-statistics 2>&1 | head -30
}


# ================================================================== bit accounting

# bits [RULE ...]
# Reads the bf16 gguf, groups the tensors by role, and shows how much of the
# model each group actually is. With rules it also predicts the output size,
# so a target ("must fit 16 GB") can be solved before spending ten minutes
# quantizing to find out.
#
#   bits
#   bits 'ffn_down_exps=iq3_s' 'ffn_up_exps=iq2_m' '*=q8_0'
bits() {
    find_bf16 || return 1
    python3 - "$BF16_FIRST" "$@" << 'BITSEOF'
import fnmatch, sys, os
sys.path.insert(0, "/llama.cpp/gguf-py")
from gguf import GGUFReader

path, rules_raw = sys.argv[1], sys.argv[2:]

# Nominal bits per weight. Real files differ slightly because of per-tensor
# fallbacks, but this is close enough to aim a size at.
BPW = {
    "f32": 32.0, "f16": 16.0, "bf16": 16.0,
    "q8_0": 8.5, "q6_k": 6.5625, "q5_1": 6.0, "q5_k": 5.5, "q5_0": 5.5,
    "q4_k": 4.5, "q4_0": 4.5, "iq4_nl": 4.5, "iq4_xs": 4.25,
    "mxfp4": 4.25, "nvfp4": 4.5,
    "q3_k": 3.4375, "iq3_s": 3.4375, "iq3_xxs": 3.0625,
    "iq2_m": 2.6875, "q2_k": 2.625, "iq2_s": 2.5, "iq2_xs": 2.3125,
    "q2_0": 2.25, "iq2_xxs": 2.0625,
    "iq1_m": 1.75, "iq1_s": 1.5625, "q1_0": 1.125,
}

# Types that never read the importance matrix, whatever you pass.
BLIND = {"q2_0", "mxfp4", "nvfp4", "q8_0", "f16", "bf16", "f32", "q1_0"}

reader = GGUFReader(path)

def group_of(name):
    for key in ("attn_gate", "ssm_out", "ssm_in", "ssm_conv1d", "ssm_norm",
                "ssm_alpha", "ssm_beta", "ssm_dt", "ssm_a", "ssm_d",
                "nextn", "post_attention_norm",
                "ffn_down_exps", "ffn_up_exps", "ffn_gate_exps",
                "ffn_down_shexp", "ffn_up_shexp", "ffn_gate_shexp",
                "ffn_down", "ffn_up", "ffn_gate", "ffn_gate_inp",
                "attn_q", "attn_k", "attn_v", "attn_output", "attn_norm",
                "token_embd", "output"):
        if key in name:
            return key
    return "other"

groups = {}
for t in reader.tensors:
    n = 1
    for d in t.shape:
        n *= int(d)
    g = groups.setdefault(group_of(t.name), {"elems": 0, "count": 0, "ne0": set()})
    g["elems"] += n
    g["count"] += 1
    g["ne0"].add(int(t.shape[0]))

total = sum(g["elems"] for g in groups.values())

print("%-18s %6s %14s %7s   %s" % ("group", "count", "elements", "share", "row length / 256 / 64 / 32"))
for name, g in sorted(groups.items(), key=lambda kv: -kv[1]["elems"]):
    ne0s = sorted(g["ne0"])[:3]
    div = " ".join(
        "%d(%s%s%s)" % (v,
                        "K" if v % 256 == 0 else "-",
                        "6" if v % 64 == 0 else "-",
                        "3" if v % 32 == 0 else "-")
        for v in ne0s)
    print("%-18s %6d %14d %6.1f%%   %s" % (
        name, g["count"], g["elems"], 100.0 * g["elems"] / total, div))
print()
unmatched = [t.name for t in reader.tensors if group_of(t.name) == "other"]
if unmatched:
    big = sorted(((t.name, 1) for t in reader.tensors if group_of(t.name) == "other"))
    sizes = {}
    for t in reader.tensors:
        if group_of(t.name) != "other":
            continue
        n = 1
        for d in t.shape:
            n *= int(d)
        key = t.name.split(".", 2)[-1] if t.name.startswith("blk.") else t.name
        sizes[key] = sizes.get(key, 0) + n
    print()
    print("what fell into 'other', by suffix:")
    for k, v in sorted(sizes.items(), key=lambda kv: -kv[1])[:12]:
        print("  %-40s %14d  %5.1f%%" % (k, v, 100.0 * v / total))
    print()

print("K = row divides by 256, so k and i quants work on it")
print("6 = divides by 64,  3 = divides by 32")
print("A group without K cannot hold any k/i quant: llama.cpp silently swaps in")
print("a block-32 type and keeps the name you asked for.")

if not rules_raw:
    print()
    print("pass rules to predict a size, e.g.")
    print("  bits 'ffn_down_exps=iq3_s' 'ffn_up_exps=iq2_m' '*=q8_0'")
    sys.exit(0)

rules = []
for r in rules_raw:
    pat, _, ty = r.partition("=")
    rules.append((pat.strip(), ty.strip().lower()))

def type_for(name):
    for pat, ty in rules:
        if pat == "*" or fnmatch.fnmatch(name, pat) or pat in name:
            return ty
    return "q8_0"

print()
print("%-18s %-10s %8s %10s" % ("group", "type", "bpw", "GB"))
bytes_total = 0.0
warned = []
for name, g in sorted(groups.items(), key=lambda kv: -kv[1]["elems"]):
    ty = type_for(name)
    bpw = BPW.get(ty)
    if bpw is None:
        print("%-18s %-10s   unknown type" % (name, ty))
        continue
    gb = g["elems"] * bpw / 8 / 1e9
    bytes_total += gb
    print("%-18s %-10s %8.3f %10.2f" % (name, ty, bpw, gb))

    ne0 = sorted(g["ne0"])[0]
    if ty in ("q2_k", "q3_k", "q4_k", "q5_k", "q6_k") or ty.startswith("iq"):
        if ty != "iq4_nl" and ne0 % 256 != 0:
            warned.append("%s: %s needs rows divisible by 256, %d is not. "
                          "llama.cpp will fall back and the file will be bigger "
                          "and worse than this estimate." % (name, ty, ne0))
    if ty in BLIND and g["elems"] > total * 0.1:
        warned.append("%s: %s ignores the importance matrix, and this group is "
                      "%.0f%% of the model. Calibration buys nothing here."
                      % (name, ty, 100.0 * g["elems"] / total))

print("-" * 50)
print("%-18s %-10s %8s %10.2f GB" % ("TOTAL", "", "", bytes_total))
print("%-18s %-10s %8s %10.2f GiB" % ("", "", "", bytes_total * 1e9 / 2**30))
if warned:
    print()
    for w in warned:
        print("!! " + w)
BITSEOF
}


# ================================================================== quantize

# quantize LABEL FALLBACK_TYPE [--tensor-type RULE ...]
# The fallback type covers everything no rule matches. Rules do the real work:
# the expert tensors are most of the model, so they decide the file size, and
# everything else is cheap to keep high.
#
#   quantize AD-IQ4_NL Q8_0 \
#     --tensor-type 'ffn_down_exps=iq4_nl' \
#     --tensor-type 'ffn_up_exps=iq4_nl'
quantize() {
    if [ -z "$2" ]; then
        echo "quantize LABEL FALLBACK_TYPE [--tensor-type RULE ...]"
        echo "  quantize AD-IQ4_NL Q8_0 --tensor-type 'ffn_down_exps=iq4_nl' \\"
        echo "                          --tensor-type 'ffn_up_exps=iq4_nl'"
        return 1
    fi
    need_preset || return 1
    find_bf16 || return 1

    LABEL=$1
    FTYPE=$2
    shift 2

    STEM=$(basename $MAIN | sed "s/-GGUF//")
    QUANT_OUT=/gguf/$STEM-$LABEL.gguf

    IM=""
    if [ -f /imatrix/imatrix.gguf ]; then
        IM="--imatrix /imatrix/imatrix.gguf"
        echo "using /imatrix/imatrix.gguf"
    else
        echo "NO IMATRIX. This will be an uncalibrated build."
        ask "continue without calibration?" || return 1
    fi

    echo "$BF16_FIRST  ->  $QUANT_OUT   fallback $FTYPE"
    date
    START=$(date +%s)

    stdbuf -oL -eL $BIN/llama-quantize $IM "$@" $BF16_FIRST $QUANT_OUT $FTYPE $(nproc) \
        2>&1 | tee /logs/quantize-$LABEL.log

    echo "took $(( $(date +%s) - START )) seconds"
    ls -lh $QUANT_OUT
    autopush /logs/quantize-$LABEL.log logs/quantize-$LABEL.log
    echo
    echo "publish it with:  push_model $QUANT_OUT"
}

# The list llama-quantize prints under "allowed quantization types" is for the
# positional ftype argument. --tensor-type takes a ggml_type, and the two are
# different sets: IQ2_M, IQ3_M, IQ3_XS, Q2_K_S, Q3_K_*, Q4_K_*, Q5_K_* are
# mixes, recipes for which tensor gets which type, so they cannot be assigned
# to a tensor. Passing one aborts the run after the model has been read.
GGML_TYPES="f32 f16 bf16 q4_0 q4_1 q5_0 q5_1 q8_0 q2_k q3_k q4_k q5_k q6_k \
iq1_s iq1_m iq2_xxs iq2_xs iq2_s iq3_xxs iq3_s iq4_xs iq4_nl tq1_0 tq2_0 \
mxfp4 q1_0 q2_0"

check_types() {
    local r t bad=""
    for r in "$@"; do
        case "$r" in *=*) t=${r#*=} ;; *) continue ;; esac
        case " $GGML_TYPES " in
            *" $t "*) ;;
            *) bad="$bad $t" ;;
        esac
    done
    if [ -n "$bad" ]; then
        echo "not ggml types, cannot go in a --tensor-type rule:$bad"
        echo "those are ftype mixes. Assignable types are:"
        echo "  $GGML_TYPES" | fold -s -w 70 | sed "s/^/    /"
        return 1
    fi
}

# ladder FILE
# One rung per line:   LABEL | FALLBACK | rule rule rule
# Blank lines and lines starting with # are ignored.
#
# For each rung: predict the size, quantize, measure against the reference,
# publish the file, and move on. Split the file across boxes to run in
# parallel; nothing here depends on another rung.
ladder() {
    if [ -z "$1" ]; then
        echo "ladder FILE       run every rung in it"
        echo "ladder FILE --dry  only predict the sizes"
        return 1
    fi
    if [ ! -f "$1" ]; then
        echo "no such file: $1"
        return 1
    fi
    need_preset || return 1
    find_bf16 || return 1

    local line label fallback rules args r dry="" bad=0
    [ "$2" = "--dry" ] && dry=1

    while IFS= read -r line; do
        case "$line" in ""|\#*) continue ;; esac

        # A rung is LABEL | FALLBACK | rules. Anything else is not one: a
        # failed download leaves an html or json error page behind, and every
        # line of it would otherwise be treated as a quantization to run.
        if [ "$(echo "$line" | tr -cd '|' | wc -c)" -lt 2 ]; then
            echo "not a rung, skipping: $line"
            bad=$(( bad + 1 ))
            continue
        fi
        label=$(echo "$line" | cut -d'|' -f1 | tr -d ' ')
        fallback=$(echo "$line" | cut -d'|' -f2 | tr -d ' ')
        rules=$(echo "$line" | cut -d'|' -f3-)
        case "$label" in
            *[!A-Za-z0-9_.-]*|"")
                echo "label is not usable, skipping: $label"
                bad=$(( bad + 1 )); continue ;;
        esac

        echo
        echo "=================================================="
        echo "$label   fallback $fallback"
        echo "  $rules"
        echo

        if ! check_types $rules; then
            bad=$(( bad + 1 ))
            continue
        fi
        bits $rules "*=$fallback" | tail -6

        [ -n "$dry" ] && continue

        args=""
        for r in $rules; do
            args="$args --tensor-type $r"
        done
        quantize "$label" "$fallback" $args || continue

        local stem out
        stem=$(basename $MAIN | sed "s/-GGUF//")
        out=/gguf/$stem-$label.gguf
        [ -f "$out" ] || continue

        kld "$out"
        push_model "$out"
    done < "$1"

    if [ "$bad" -gt 0 ]; then
        echo
        echo "$bad line(s) were not rungs. If the whole file looked wrong, it is"
        echo "probably an error page from a failed download. Rewrite it with:"
        echo "  write_ladder"
    fi
    results
}

# The ladder lives here rather than in a separate file, so a failed download
# can never leave a json error page standing in for it.
write_ladder() {
    cat > /ladder.txt << 'LADDEREOF'
# Qwen3.8-27B. LABEL | FALLBACK | rules
#
# Only ggml types appear in rules. IQ2_M, IQ3_M, IQ3_XS and the Q*_K_S/M/L
# names are ftype mixes and cannot be assigned to a tensor, whatever the
# "allowed quantization types" list suggests. That leaves a real gap between
# iq2_s at 2.5 bpw and iq3_xxs at 3.06, which is why the rungs there move by
# changing which group steps rather than by finding a type in between.
#
# Naming is AD-<ffn_down>-<ffn_up>, collapsed when both match, so the name
# carries the layout and no file is named after a type it does not contain.
#
# ffn_down, ffn_gate and ffn_up are 21.2% each and set the size. down carries
# more of the quality and sits a step above the other two at most rungs; a
# preset gives all three the same type.
#
# attn_gate and ssm_out are 5.5% each, 11% together on this hybrid, so they
# move with the rung instead of sitting at q8_0.
#
# attn_k and attn_v are 0.3% each: q8_0 everywhere, it costs 0.05 bpw.
# output stays high, token_embd drops fast: same weight, different sensitivity.

# ---- above 24 GB
AD-Q8_0             | q8_0 |
AD-Q6_K             | q8_0 | ffn_down=q6_k ffn_gate=q6_k ffn_up=q6_k attn_q=q8_0 attn_gate=q6_k ssm_out=q8_0
AD-Q6_K-Q5_K        | q8_0 | ffn_down=q6_k ffn_gate=q5_k ffn_up=q5_k attn_q=q6_k attn_gate=q5_k ssm_out=q6_k

# ---- 32 GB card
AD-Q5_K             | q8_0 | ffn_down=q5_k ffn_gate=q5_k ffn_up=q5_k attn_q=q5_k attn_gate=q5_k ssm_out=q6_k
AD-Q5_K-Q4_K        | q8_0 | ffn_down=q5_k ffn_gate=q4_k ffn_up=q4_k attn_q=q5_k attn_gate=q4_k ssm_out=q5_k output=q6_k

# ---- 24 GB card
AD-Q4_K             | q8_0 | ffn_down=q4_k ffn_gate=q4_k ffn_up=q4_k attn_q=q4_k attn_gate=q4_k ssm_out=q5_k output=q6_k
AD-Q4_K-IQ4_XS      | q8_0 | ffn_down=q4_k ffn_gate=iq4_xs ffn_up=iq4_xs attn_q=iq4_xs attn_gate=iq4_xs ssm_out=q5_k output=q6_k token_embd=q5_k
AD-IQ4_XS           | q8_0 | ffn_down=iq4_xs ffn_gate=iq4_xs ffn_up=iq4_xs attn_q=iq4_xs attn_gate=iq4_xs ssm_out=iq4_xs output=q6_k token_embd=q5_k

# ---- 16 GB card
AD-IQ4_XS-IQ3_S     | q8_0 | ffn_down=iq4_xs ffn_gate=iq3_s ffn_up=iq3_s attn_q=iq4_xs attn_gate=iq4_xs ssm_out=iq4_xs output=q5_k token_embd=q4_k
AD-IQ3_S            | q8_0 | ffn_down=iq3_s ffn_gate=iq3_s ffn_up=iq3_s attn_q=iq4_xs attn_gate=iq3_s ssm_out=iq4_xs output=q5_k token_embd=q4_k
AD-IQ3_S-IQ3_XXS    | q8_0 | ffn_down=iq3_s ffn_gate=iq3_xxs ffn_up=iq3_xxs attn_q=iq4_xs attn_gate=iq3_s ssm_out=iq4_xs output=q5_k token_embd=iq4_xs
AD-IQ3_XXS          | q8_0 | ffn_down=iq3_xxs ffn_gate=iq3_xxs ffn_up=iq3_xxs attn_q=iq4_xs attn_gate=iq3_s ssm_out=iq3_s output=q5_k token_embd=iq4_xs

# ---- 12 GB card
AD-IQ3_XXS-IQ2_S    | q8_0 | ffn_down=iq3_xxs ffn_gate=iq2_s ffn_up=iq2_s attn_q=iq4_xs attn_gate=iq3_s ssm_out=iq3_s output=q5_k token_embd=iq3_xxs
AD-IQ2_S            | q8_0 | ffn_down=iq2_s ffn_gate=iq2_s ffn_up=iq2_s attn_q=iq3_s attn_gate=iq3_s ssm_out=iq3_xxs output=q5_k token_embd=iq3_xxs
AD-IQ2_S-IQ2_XS     | q8_0 | ffn_down=iq2_s ffn_gate=iq2_xs ffn_up=iq2_xs attn_q=iq3_s attn_gate=iq3_xxs ssm_out=iq3_xxs output=q4_k token_embd=iq3_xxs
AD-IQ2_XS           | q8_0 | ffn_down=iq2_xs ffn_gate=iq2_xs ffn_up=iq2_xs attn_q=iq3_xxs attn_gate=iq2_s ssm_out=iq3_xxs output=q4_k token_embd=iq2_xxs

# ---- 8 to 10 GB
AD-IQ2_XS-IQ2_XXS   | q8_0 | ffn_down=iq2_xs ffn_gate=iq2_xxs ffn_up=iq2_xxs attn_q=iq3_xxs attn_gate=iq2_s ssm_out=iq2_s output=q4_k token_embd=iq2_xxs
AD-IQ2_XXS          | q8_0 | ffn_down=iq2_xxs ffn_gate=iq2_xxs ffn_up=iq2_xxs attn_q=iq2_s attn_gate=iq2_s ssm_out=iq2_s output=iq4_xs token_embd=iq2_xxs
AD-IQ2_XXS-IQ1_M    | q8_0 | ffn_down=iq2_xxs ffn_gate=iq1_m ffn_up=iq1_m attn_q=iq2_s attn_gate=iq2_s ssm_out=iq2_s output=iq4_xs token_embd=iq2_xxs
AD-IQ2_XXS-IQ1_S    | q8_0 | ffn_down=iq2_xxs ffn_gate=iq1_s ffn_up=iq1_s attn_q=iq2_s attn_gate=iq2_s ssm_out=iq2_s output=iq4_xs token_embd=iq2_xxs

# ---- the standard mixes, same imatrix. People ask for these by name, and they
# are the baseline that decides whether the layouts above actually win.
Q6_K                | q6_k |
Q5_K_M              | q5_k_m |
Q4_K_M              | q4_k_m |
LADDEREOF
    echo "wrote /ladder.txt, $(grep -c '|' /ladder.txt) rungs"
    echo "check the sizes first:   ladder /ladder.txt --dry"
}

# Pull somebody else's build so it can be measured against the same reference.
# Comparisons across different references mean nothing.
get_external() {
    if [ -z "$2" ]; then
        echo "get_external REPO PATTERN"
        echo "  get_external ggml-org/Some-Model-GGUF '*NVFP4*'"
        return 1
    fi
    mkdir -p /gguf/external
    hf download $1 --include "$2" --local-dir /gguf/external
    ls -la /gguf/external
    echo
    echo "measure it with:  kld /gguf/external/<file>.gguf"
}


# ================================================================== results

results() {
    python3 - << 'PYEOF'
import glob, json, os, re, socket

def num(pattern, text):
    m = re.search(pattern, text)
    return float(m.group(1)) if m else None

# One row per file. Quality and speed used to land in separate rows because
# they were keyed differently; they are the same file measured two ways.
rows = {}

def row(name):
    return rows.setdefault(name, {'name': name})

failed = []

for path in glob.glob('/logs/kld-*.log'):
    stem = os.path.basename(path)[4:-4]
    evalset, name = stem.split('--', 1) if '--' in stem else ('neutral', stem)
    text = open(path, errors='ignore').read()
    mean = num(r'Mean\s+KLD:\s+([0-9.]+)', text)
    if mean is None:
        failed.append(name)
        continue
    r = row(name)
    r.setdefault('quality', {})[evalset] = {
        'mean_kld':   mean,
        'q99_kld':    num(r'99\.0%\s+KLD:\s+([0-9.]+)', text),
        'median_kld': num(r'Median\s+KLD:\s+([0-9.]+)', text),
        'top1_pct':   num(r'Same top p:\s+([0-9.]+)', text),
        'rms_dp_pct': num(r'RMS\s+.p\s*:\s+([0-9.]+)', text),
    }

for path in glob.glob('/logs/bench-*.json'):
    name = os.path.basename(path)[6:-5]
    r = row(name)
    try:
        for entry in json.load(open(path)):
            key = 'pp512_tps' if entry.get('n_prompt', 0) > 0 else 'tg128_tps'
            r[key] = entry.get('avg_ts')
            r['gpu'] = entry.get('gpu_info')
            r['build'] = entry.get('build_commit')
    except Exception as e:
        r['bench_error'] = str(e)

for name, r in rows.items():
    for folder in ('/gguf', '/gguf/experimental', '/gguf/external'):
        p = os.path.join(folder, name + '.gguf')
        if os.path.exists(p):
            r['size_bytes'] = os.path.getsize(p)
            r['size_gb'] = round(os.path.getsize(p) / 1e9, 2)

out = sorted(rows.values(), key=lambda r: r.get('size_gb') or 0)
for r in out:
    r['measured_on'] = socket.gethostname()

json.dump(out, open('/logs/results.json', 'w'), indent=2)

sets = sorted({s for r in out for s in r.get('quality', {})})
primary = 'neutral' if 'neutral' in sets else (sets[0] if sets else None)

print('%-46s %7s %10s %8s %9s %9s' % ('file', 'GB', 'mean KLD', 'top-1', 'pp512', 'tg128'))
for r in out:
    q = r.get('quality', {}).get(primary, {})
    print('%-46s %7s %10s %8s %9s %9s' % (
        r['name'][-46:],
        r.get('size_gb', ''),
        round(q['mean_kld'], 6) if q.get('mean_kld') is not None else '',
        round(q['top1_pct'], 2) if q.get('top1_pct') is not None else '',
        round(r['pp512_tps'], 1) if r.get('pp512_tps') is not None else '',
        round(r['tg128_tps'], 1) if r.get('tg128_tps') is not None else ''))

if len(sets) > 1:
    print()
    print('eval sets in the file: %s (table shows %s)' % (', '.join(sets), primary))
if failed:
    print()
    print('%d run(s) produced no KLD, left out:' % len(failed))
    for n in failed:
        print('   ', n)
print()
print('written to /logs/results.json')
PYEOF
    # Per box, so four boxes measuring different quants do not overwrite one
    # another. merge_results stitches them back together.
    autopush /logs/results.json results-$(hostname).json
}

# Pull every box's results file and stitch them into one.
merge_results() {
    need_preset || return 1
    repo_ok || return 1
    mkdir -p /merged
    hf download $METRICS --repo-type $METRICS_KIND --include "results-*.json" --local-dir /merged
    python3 - << 'MERGEEOF'
import glob, json

rows = {}
for path in sorted(glob.glob('/merged/**/results-*.json', recursive=True)):
    for r in json.load(open(path)):
        cur = rows.setdefault(r['name'], {})
        for k, v in r.items():
            if k == 'quality':
                cur.setdefault('quality', {}).update(v)
            elif v is not None:
                cur[k] = v

out = sorted(rows.values(), key=lambda r: r.get('size_gb') or 0)
json.dump(out, open('/logs/results.json', 'w'), indent=2)

print('%-46s %7s %10s %8s %9s   %s' % ('file', 'GB', 'mean KLD', 'top-1', 'tg128', 'from'))
for r in out:
    q = r.get('quality', {}).get('neutral', {})
    print('%-46s %7s %10s %8s %9s   %s' % (
        r['name'][-46:], r.get('size_gb', ''),
        round(q['mean_kld'], 6) if q.get('mean_kld') is not None else '',
        round(q['top1_pct'], 2) if q.get('top1_pct') is not None else '',
        round(r['tg128_tps'], 1) if r.get('tg128_tps') is not None else '',
        r.get('measured_on', '')))
print()
print('%d files merged into /logs/results.json' % len(out))
MERGEEOF
    hf_put /logs/results.json results.json "$METRICS" "$METRICS_KIND"
}


# ================================================================== upload

# The .kld holds fp16 logits over the whole vocabulary for every scored token.
# People asked for it so they can compare their own quants against the same
# reference, so it gets published. Over 45 GB it goes up as numbered parts.
push_base() {
    need_preset || return 1
    repo_ok || return 1
    if [ ! -f /logs/base-$EVALSET.log ]; then
        echo "no base log yet. Run base first."
        return 1
    fi

    hf_put /logs/base-$EVALSET.log logs/base-$EVALSET.log "$METRICS" "$METRICS_KIND"
    [ -f /logs/env.txt ] && hf_put /logs/env.txt logs/env.txt "$METRICS" "$METRICS_KIND"
    [ -f /logs/llama-commit.txt ] && hf_put /logs/llama-commit.txt logs/llama-commit.txt "$METRICS" "$METRICS_KIND"

    if [ ! -f $BASE ]; then
        echo "no .kld on disk, only the log went up"
        return 0
    fi

    SIZE=$(stat -c %s $BASE)
    echo
    echo "reference blob: $(( SIZE / 1000000000 )) GB"

    # A manifest, not a hash. The hub already verifies transport, and the file
    # is not bit reproducible across different GPU counts and drivers, so a
    # published checksum would mislead rather than help. The one thing worth
    # checking on the far side is that reassembly produced a whole file, and
    # the byte count does that instantly.
    echo "size_bytes $SIZE" > /kld/base-$EVALSET.manifest.txt
    echo "context $CTX" >> /kld/base-$EVALSET.manifest.txt
    echo "eval_set $EVALSET" >> /kld/base-$EVALSET.manifest.txt
    cat /kld/base-$EVALSET.manifest.txt

    if [ "$HASH_BLOB" = "1" ]; then
        echo "HASH_BLOB=1, hashing. Reads the whole file, takes a while."
        sha256sum $BASE | sed "s|/kld/||" > /kld/base-$EVALSET.kld.sha256
        cat /kld/base-$EVALSET.kld.sha256
    fi

    write_kld_readme

    if [ $SIZE -gt 45000000000 ]; then
        echo
        echo "over the 50 GB per-file limit, splitting into 45 GB parts"
        rm -rf /kld/parts
        mkdir -p /kld/parts
        split -b 45G -d --additional-suffix=.part $BASE /kld/parts/base-$EVALSET.kld.
        ls -lh /kld/parts
        cp /kld/base-$EVALSET.manifest.txt /kld/parts/
        [ -f /kld/base-$EVALSET.kld.sha256 ] && cp /kld/base-$EVALSET.kld.sha256 /kld/parts/
        cp /kld/README.md /kld/parts/
        echo
        ask "upload the parts now?" || return 0
        hf_put_dir /kld/parts kld "$METRICS" "$METRICS_KIND"
    else
        ask "upload the blob now?" || return 0
        hf_put $BASE kld/base-$EVALSET.kld "$METRICS" "$METRICS_KIND"
        hf_put /kld/base-$EVALSET.manifest.txt kld/base-$EVALSET.manifest.txt "$METRICS" "$METRICS_KIND"
        hf_put /kld/README.md kld/README.md "$METRICS" "$METRICS_KIND"
    fi
    echo "done"
}

write_kld_readme() {
    cat > /kld/README.md << READMEEOF
# KL divergence reference logits

Produced by \`llama-perplexity --kl-divergence-base\`. The file holds fp16
logits over the full vocabulary for every scored token, so any quantization of
this model can be measured against the same reference we used.

- model    : bf16 weights of $MAIN
- corpus   : $EVALSET held-out set from AtomicChat/calib-corpora
- context  : $CTX
- llama.cpp: see logs/llama-commit.txt
- hardware : see logs/env.txt

## If the file is split

Parts are named \`base-$EVALSET.kld.NN.part\`. Reassemble in order, then check
the hash:

\`\`\`bash
cat base-$EVALSET.kld.*.part > base-$EVALSET.kld
sha256sum -c base-$EVALSET.kld.sha256
\`\`\`

## Using it

\`\`\`bash
llama-perplexity -m your-quant.gguf -f eval_neutral.txt \\
  --kl-divergence-base base-$EVALSET.kld --kl-divergence -c $CTX -ngl 99
\`\`\`

Your corpus and context must match the ones above or the numbers mean nothing.
READMEEOF
    echo "wrote /kld/README.md"
}

# Pull the reference from the metrics repo, reassembling parts if needed.
get_base() {
    need_preset || return 1
    hf download $METRICS --repo-type $METRICS_KIND --include "kld/*" --local-dir /kld

    if [ -f /kld/kld/base-$EVALSET.kld ]; then
        mv /kld/kld/base-$EVALSET.kld $BASE
    elif ls /kld/kld/base-$EVALSET.kld.*.part > /dev/null 2>&1; then
        echo "reassembling parts"
        cat /kld/kld/base-$EVALSET.kld.*.part > $BASE
        rm -f /kld/kld/base-$EVALSET.kld.*.part
        if [ -f /kld/kld/base-$EVALSET.manifest.txt ]; then
            WANT=$(grep size_bytes /kld/kld/base-$EVALSET.manifest.txt | cut -d" " -f2)
            GOT=$(stat -c %s $BASE)
            if [ "$WANT" = "$GOT" ]; then
                echo "size matches the manifest: $GOT bytes"
            else
                echo "SIZE MISMATCH: expected $WANT, got $GOT. A part is missing or truncated."
                return 1
            fi
        fi
    fi
    ls -lh $BASE
}

# Straight box to box, no hub round trip. Much faster on a local-ish link.
#   send_base root@1.2.3.4 41234
send_base() {
    if [ -z "$1" ]; then
        echo "send_base user@host [ssh_port]"
        echo "on vast, get host and port from: vastai ssh-url <id>"
        return 1
    fi
    if [ ! -f $BASE ]; then
        echo "no reference at $BASE"
        return 1
    fi
    which rsync > /dev/null 2>&1 || apt-get install -y -qq rsync
    ls -lh $BASE
    rsync -avP -e "ssh -p ${2:-22} -o StrictHostKeyChecking=no" $BASE $1:$BASE
}

# Every box uploads its own logs as it goes, so merging results across boxes is
# just pulling them all down and re-running the parser.
pull_logs() {
    need_preset || return 1
    repo_ok || return 1
    hf download $METRICS --repo-type $METRICS_KIND --include "logs/*" --local-dir /tmp/pulled
    cp -n /tmp/pulled/logs/* /logs/ 2>/dev/null
    ls /logs | wc -l
    echo "logs in /logs now, running the parser over all of them"
    results
}

get_imatrix() {
    need_preset || return 1
    hf download $METRICS --repo-type $METRICS_KIND --include "imatrix/imatrix.gguf" --local-dir /tmp/im
    mkdir -p /imatrix
    cp /tmp/im/imatrix/imatrix.gguf /imatrix/imatrix.gguf
    ls -lh /imatrix/imatrix.gguf
}


push_logs() {
    need_preset || return 1
    repo_ok || return 1
    du -sh /logs
    hf_put_dir /logs logs "$METRICS" "$METRICS_KIND"
}

push_results() {
    need_preset || return 1
    repo_ok || return 1
    if [ ! -f /logs/results.json ]; then
        echo "no results.json. Run results first."
        return 1
    fi
    hf_put /logs/results.json results.json "$METRICS" "$METRICS_KIND"
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
    hf_put "$1" "$(basename $1)" "$MAIN" model
}

push_model_split() {
    need_preset || return 1
    if [ ! -f "$1" ]; then echo "no file at $1"; return 1; fi
    PREFIX=/gguf/split/$(basename $1 .gguf)
    mkdir -p /gguf/split
    $BIN/llama-gguf-split --split --split-max-size 45G $1 $PREFIX
    ls -lh $PREFIX*
    hf_put_dir /gguf/split "$(basename $1 .gguf)" "$MAIN" model
    echo "Uploaded as a folder of shards. Readers point at shard 1."
}

push_card() {
    need_preset || return 1
    if [ ! -f "$1" ]; then echo "no file at $1"; return 1; fi
    hf_put "$1" README.md "$MAIN" model
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


# ================================================================== state
# Read back whatever the last pane set, so a fresh tmux tab is not amnesiac.
if [ -f /state.sh ]; then
    . /state.sh
fi
