# DiffusionGemma-Playground

Run **DiffusionGemma 26B-A4B** — a block-diffusion *text* model on a Gemma MoE
backbone that generates by iterative denoising instead of token-by-token — in
Docker, with a browser chat UI where every reply denoises live inside its
message bubble. Developed on an RTX 5090 under WSL2; any CUDA GPU with enough
VRAM should work (see below).

Built from [llama.cpp PR #24423](https://github.com/ggml-org/llama.cpp/pull/24423)
(`llama-diffusion-cli`, by Daniel Han / Unsloth). The PR is an **open draft and
not merged** into llama.cpp master, so the official llama.cpp Docker images
don't include it — this Dockerfile builds it from the PR ref directly.

## Requirements

- NVIDIA GPU with enough VRAM for the quant you pick (Q4_K_M ≈ 16 GB; the
  RTX 5090's 32 GB fits it comfortably, Q8_0 at ~28 GB is tight but may fit)
- NVIDIA driver + Docker with the `nvidia` container runtime
  (`docker run --gpus all` must work)
- The Dockerfile targets CUDA arch **120 (Blackwell)** on a CUDA 12.8 base.
  For other GPUs, adjust `CMAKE_CUDA_ARCHITECTURES` (e.g. `89` for Ada,
  `86` for Ampere).

## Build

```bash
docker build -t diffusion-gemma .
```

The CUDA build takes ~10–20 minutes. Since the PR is a moving target, rebuild
with `--no-cache` to pick up upstream changes.

## Get the model

Download the GGUF from
[unsloth/diffusiongemma-26B-A4B-it-GGUF](https://huggingface.co/unsloth/diffusiongemma-26B-A4B-it-GGUF)
into `~/models` (kept on the host so it survives container restarts):

```bash
mkdir -p ~/models
curl -L --fail -C - -o ~/models/diffusiongemma-26B-A4B-it-Q4_K_M.gguf \
  https://huggingface.co/unsloth/diffusiongemma-26B-A4B-it-GGUF/resolve/main/diffusiongemma-26B-A4B-it-Q4_K_M.gguf
```

> **Gotcha:** if `~/models` doesn't exist when Docker first mounts it, Docker
> creates it owned by `root:root` and downloads into it fail with curl
> exit 23 (write error). Create it yourself first, or fix it with
> `sudo chown $USER:$USER ~/models`.

## Run

Interactive chat with the live denoising animation:

```bash
docker run -it --rm --gpus all -v ~/models:/root/.cache/llama.cpp \
  diffusion-gemma \
  -m /root/.cache/llama.cpp/diffusiongemma-26B-A4B-it-Q4_K_M.gguf \
  -ngl 99 -cnv -n 2048 --diffusion-visual
```

One-shot prompt:

```bash
docker run --rm --gpus all -v ~/models:/root/.cache/llama.cpp \
  diffusion-gemma \
  -m /root/.cache/llama.cpp/diffusiongemma-26B-A4B-it-Q4_K_M.gguf \
  -ngl 99 -p "Write a haiku about GPUs." -n 256
```

Useful flags:

| Flag | Effect |
|---|---|
| `-ngl 99` | offload all layers to the GPU |
| `-cnv` | interactive conversation mode |
| `--diffusion-visual` | live denoising animation in the terminal |
| `--diffusion-gpu-sampling on` | ~1.3–1.4× speedup on CUDA (may default to `auto`) |
| `--diffusion-gpu-sample-reduce on` | further CUDA sampling speedup (may default to `auto`) |

> **Canvas economics:** generation happens in fixed 256-token canvas blocks,
> and a block costs the same compute no matter how much of it you use —
> `-n 64` takes as long as `-n 256`, it just throws away ¾ of the canvas.
> Throughput comparisons against autoregressive decode are only meaningful
> for generations of ≥250 tokens.

There is no `llama-server` integration in the PR. Newer commits add a
model-specific `llama-diffusion-gemma-visual-server` that streams canvas
frames, but as of PR head `10a2613` it is reported broken in the thread
(never starts listening, ignores `-ngl`) and maintainers have pushed back on
a per-model server, so this project sticks with the CLI + bridge below.

## Experimental chat frontend (`chat/`)

A browser UI built around the diffusion character of the model. Since the PR
has no `llama-server` integration, a small zero-dependency Node bridge
(`chat/server.mjs`) keeps **one persistent container** alive in `-cnv`
conversation mode — the model loads into VRAM once (warm-started at server
boot) and stays there, prompts are piped over stdin, and multi-turn context is
handled natively by the CLI (KV-cached). Time-to-first-frame on a warm
session is ~90 ms; only changing the generation parameters forces a backend
restart (model reload).

The CLI sizes its visual canvas via TTY ioctl (and clips it to 80×24 when
piped), so the bridge runs it under `script` with an `stty`-sized pty —
64×150 by default, tunable via `CANVAS_ROWS`/`CANVAS_COLS`. The bridge parses
the ANSI canvas redraws (one full snapshot per denoising step, wrapped in
`ESC[?2026h…l` synchronized updates, then final text + stats + a `> ` readline
marker per turn) and streams them to the browser as NDJSON over a plain
streaming POST response — no WebSockets needed; the transport was never the
bottleneck.

The frontend is a plain chat — but each assistant bubble *is* the diffusion:

- replies **denoise live inside the bubble** (unresolved cells stay blank,
  freshly-resolved characters glow), then settle into clean text,
- every bubble keeps its frames and carries a **per-message timeline** —
  scrub through the denoising or hit ⟲ to replay it; the scrubber's last
  position is the settled text,
- Gemma's **thought channel** (`<|channel>thought … <channel|>`) is split out
  from the answer as a collapsible section,
- controls for `-n`, `--diffusion-eb-max-steps`, and
  `--diffusion-visual-interval`.

### Develop in the devcontainer

Open the repo in VS Code → "Reopen in Container". The devcontainer
(`.devcontainer/devcontainer.json`) is a Node 22 image with
docker-outside-of-docker, so the bridge talks to the **host** Docker daemon —
which is why `MODELS_HOST_DIR` must be the host path of your models dir. The
server starts automatically on port 8090 (forwarded).

Manual start inside the devcontainer:

```bash
node chat/server.mjs   # → http://localhost:8090
```

### Run without the devcontainer

Any environment with Node ≥ 18 and the docker CLI works:

```bash
MODELS_HOST_DIR=$HOME/models node chat/server.mjs
```

Or, if the host has no Node (mirrors the devcontainer setup):

```bash
docker run -d --name chat-bridge -p 8090:8090 \
  -v /var/run/docker.sock:/var/run/docker.sock \
  -v "$(command -v docker)":/usr/local/bin/docker:ro \
  -v "$PWD/chat":/app:ro -w /app \
  -e MODELS_HOST_DIR=$HOME/models \
  node:22 node server.mjs
```

Notes: one generation runs at a time (the bridge returns 409 while busy); the
"⟲ new chat" button sends the CLI's `/clear` to reset the conversation;
multiline prompts are collapsed to one line (cnv input is line-based); the
persistent container (`diffusion-session`) is removed on server shutdown.
If the backend crashes mid-turn (the PR thread reports CUDA aborts on long
pasted inputs), the next prompt restarts it automatically and the chat shows
a divider noting that the model lost the conversation context.

## Performance

Measured on an RTX 5090 (Q4_K_M, full offload): **~191 tok/s** effective
throughput, ~61 ms per denoising step at ~22 steps per 256-token block. The PR
thread cites roughly 22–40 tok/s on consumer GPUs, so expect less on smaller
cards.

Per-step time grows with conversation length in `-cnv`: every denoising step
attends over the full accumulated context, with no KV-cache amortization
across the canvas. On the 5090 this is mild (~66 → ~100 ms/step over five
1024-token turns, measured with `scripts/multiturn-bench.py`); with
`--n-cpu-moe` partial offload the PR thread reports throughput halving per
turn. Start a new chat to reset it.

## License

MIT. Not affiliated with Google DeepMind, Unsloth, or the llama.cpp project.
