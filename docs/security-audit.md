# Security audit

What was checked, what was found, and what has to change before anyone else runs
these scripts.

Audited: `foundry.sh`, `foundry-mlx3.sh`, `auto_fmt.py`. 

## Credentials

**No credential is hardcoded anywhere in the three reviewed files.** The token is
read from the environment or from `/proc/1/environ`, and `token_check` explicitly
rejects placeholder values (`PUT_*`, `hf_xxx*`, `CHANGE*`). The comment next to it
states the rule: this file must never export a token.

Two functions write the token to disk on the rented machine:

| function | writes | where |
| --- | --- | --- |
| `save_token` | `export HF_TOKEN=...` | `~/.bashrc` on the box |
| `mlx3_persist` | the same | `~/.bashrc` on the box |

This is deliberate, so a new tmux pane has the token, and both print the warning
that the box is disposable and the token is not. The one real risk is snapshotting
an instance into an image with the token in it. Revoke the token when a job is
done.

## Where data can go

Every upload path is pinned to the `AtomicChat` organisation. `use_model` builds
`MAIN=AtomicChat/$(basename $2)-GGUF`, and `mlx_repo`, `mlx3_push` and
`mlx3_upload` do the same. The scripts cannot push to a personal account unless
someone edits `MAIN`, `METRICS` or `MLX3_ORG` by hand.

The single path out that is not Hugging Face is `send_base user@host PORT`, which
rsyncs the reference blob to a host typed by the operator, with
`StrictHostKeyChecking=no`. It cannot fire on its own.

## Gaps to close

> [!WARNING]
> These are the changes worth making before the scripts are used by someone who
> did not write them.

**1. Six URLs point at a personal GitHub account.** After the account is gone or
the repository is made private, `reload` and every bootstrap line break.

| file | function or place |
| --- | --- |
| `foundry.sh` | `reload` (the `git clone` line) |
| `foundry.sh` | `install_auto_fmt` (the `curl` line) |
| `foundry-mlx3.sh` | the header comment block |
| `foundry-mlx3.sh` | `mlx3_box hub` |
| `foundry-mlx3.sh` | `mlx3_box dwq` |
| `foundry-mlx3.sh` | `mlx3_help` |

Find them all with:

```bash
grep -rn "worthant" .
```

**2. `hf_put` does not scan for secrets.** `scan_secrets` runs inside
`hf_put_dir` but not inside `hf_put`, and `hf_put` is what `autopush`,
`push_model`, `push_card`, `push_mmproj` and `push_shards` use. Adding one line
at the top of `hf_put` closes it:

```bash
scan_secrets "$1" || return 1
```

**3. `mlx3_upload` does not scan at all**, and it uploads every log to the
metrics dataset.

**4. `mlx3_push` scans only for `hf_`.** The pattern in `foundry.sh` covers four
shapes and is the better one:

```
hf_[A-Za-z0-9]{30,}|sk-[A-Za-z0-9]{20,}|ghp_[A-Za-z0-9]{30,}|AKIA[0-9A-Z]{16}
```

**5. `results()` writes `socket.gethostname()`** into `results.json`, and
`autopush` names the file `results-<hostname>.json`. On a rented box that is
meaningless. Run it on a personal laptop and the laptop's hostname goes into a
public dataset repository.

**6. macOS paths contain the username.** `FROOT` is `$HOME/foundry`, so log
paths on a Mac carry the account name. `mac_info` also writes chip, core count,
memory, macOS version and library versions to `mac-env.txt`. None of the Mac
functions autopush, but do not run `push_logs` with `LOG_DIR` pointed at a Mac
without reading what is in there.

**7. `push_demo_image` uploads an arbitrary local file** to the public model
repository. A screenshot is whatever was on screen. Check it before pushing, and
pass the source note so nobody has to guess where the image came from.

## Checklist before publishing this repository

Run all of these from the repository root and read every hit.

```bash
# personal account references
grep -rn "worthant" .

# credential shapes, in the working tree
grep -rEn 'hf_[A-Za-z0-9]{20,}|sk-[A-Za-z0-9]{20,}|ghp_[A-Za-z0-9]{30,}|AKIA[0-9A-Z]{16}' .

# credential shapes, in the whole git history (the tree being clean is not enough)
git log -p --all | grep -nE 'hf_[A-Za-z0-9]{20,}|ghp_[A-Za-z0-9]{30,}|sk-[A-Za-z0-9]{20,}|AKIA[0-9A-Z]{16}'

# personal names, home directories, emails
grep -rniE 'boris|/home/[a-z]|/Users/|@gmail|@yandex|@mail\.ru' .

# keys of any kind
grep -rniE 'BEGIN [A-Z ]*PRIVATE KEY|ssh-rsa|ssh-ed25519' .

# who authored the commits, if the history is being carried over
git log --format='%an <%ae>' | sort -u
```

Then check the things grep cannot see:

- the private ssh key and `known_hosts` are not in the tree
- `.env`, `.envrc`, `*.token`, `*.pem`, `.netrc` are not in the tree

> [!IMPORTANT]
> A clean working tree says nothing about the history. If a token was ever
> committed and later removed, it is still in `git log -p`. If the checklist
> finds one, revoke the token first and rewrite or drop the history second.
> Revoking is the part that actually helps.

## Access to hand over

Not in this repository, and not something a script can transfer:

- the Hugging Face token with write access to the `AtomicChat` organisation
- the vast.ai API key, and any other provider account used for renting
- the payment method attached to those accounts
- the orchestrator VPS, if it is still in use
- ownership of the Hugging Face repositories, if any of them were created under
  a personal account rather than the organisation
