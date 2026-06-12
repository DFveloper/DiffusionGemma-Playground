#!/usr/bin/env python3
"""Multi-turn throughput check for the persistent -cnv session.

Context: PR #24423 thread reports (Iipal, Jun 11) that throughput drops
roughly in half with each successive prompt in conversation mode. The chat
bridge keeps one persistent -cnv session, so per-turn degradation would
undermine its design. This sends a small conversation through one session
(no visual, single 256-token block per turn) and prints the stats per turn.

Usage: python3 scripts/multiturn-bench.py [n_tokens]
"""

import os
import re
import subprocess
import sys
import threading
import time

MODEL = os.environ.get("MODEL_FILE", "diffusiongemma-26B-A4B-it-Q4_K_M.gguf")
MODELS_DIR = os.environ.get("MODELS_HOST_DIR", os.path.expanduser("~/models"))
N_TOKENS = sys.argv[1] if len(sys.argv) > 1 else "256"
READY_TIMEOUT = 300
TURN_TIMEOUT = 180

PROMPTS = [
    "create a fibonacci script in python",
    "now do the same in javascript",
    "now in rust",
    "now in go",
    "now in c++",
]

# Like the bridge: run under `script` with an stty-sized pty — piped stdout
# is block-buffered and the readline "> " marker never flushes otherwise.
cli = (f"llama-diffusion-cli -m /root/.cache/llama.cpp/{MODEL} "
       f"-ngl 99 -cnv -n {N_TOKENS}")
inner = f"stty rows 24 cols 200 -echo; exec {cli} 2>/dev/null"
cmd = [
    "docker", "run", "-i", "--rm", "--name", "dg-multiturn-bench", "--gpus", "all",
    "-v", f"{MODELS_DIR}:/root/.cache/llama.cpp",
    "--entrypoint", "script", "diffusion-gemma", "-qfc", inner, "/dev/null",
]

subprocess.run(["docker", "rm", "-f", "dg-multiturn-bench"],
               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE)

buf = ""
buf_lock = threading.Lock()
stderr_tail = []

def pump(stream, sink):
    # os.read returns as soon as ANY bytes arrive; TextIOWrapper.read(n)
    # would block until n chars accumulate and miss the short "> " marker.
    fd = stream.fileno()
    while True:
        chunk = os.read(fd, 4096)
        if not chunk:
            return
        sink(chunk.decode("utf-8", errors="replace"))

def on_stdout(chunk):
    global buf
    with buf_lock:
        buf += chunk.replace("\r", "")

def on_stderr(chunk):
    stderr_tail.append(chunk)
    del stderr_tail[:-40]

threading.Thread(target=pump, args=(proc.stdout, on_stdout), daemon=True).start()
threading.Thread(target=pump, args=(proc.stderr, on_stderr), daemon=True).start()

ANSI = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]")

def wait_for_prompt(timeout):
    """Readline marker '> ' at end of output = CLI is waiting for input."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        if proc.poll() is not None:
            print("backend exited:\n" + "".join(stderr_tail)[-3000:])
            sys.exit(1)
        with buf_lock:
            plain = ANSI.sub("", buf)
        if re.search(r"(^|\n)> $", plain):
            return plain
        time.sleep(0.2)
    print("timed out; last stdout:\n" + plain[-2000:])
    print("last stderr:\n" + "".join(stderr_tail)[-2000:])
    proc.kill()
    sys.exit(1)

def take_buf():
    global buf
    with buf_lock:
        out, buf = buf, ""
    return ANSI.sub("", out)

print(f"loading model ({MODEL}, -n {N_TOKENS}) ...", flush=True)
wait_for_prompt(READY_TIMEOUT)
take_buf()
print("ready.\n", flush=True)

results = []
for i, prompt in enumerate(PROMPTS, 1):
    proc.stdin.write((prompt + "\n").encode())
    proc.stdin.flush()
    t0 = time.time()
    wait_for_prompt(TURN_TIMEOUT)
    wall = time.time() - t0
    out = take_buf()
    stats = [l for l in out.split("\n") if l.startswith(("total time:", "throughput:"))]
    results.append((prompt, wall, stats))
    print(f"turn {i}: {prompt!r}  (wall {wall:.1f}s)")
    for l in stats:
        print(f"   {l}")
    print(flush=True)

proc.stdin.close()
proc.terminate()

print("=" * 60)
print(f"{'turn':<5} {'wall s':>7} {'ms/step':>9} {'tok/s':>7}")
for i, (_, wall, stats) in enumerate(results, 1):
    joined = " ".join(stats)
    step = re.search(r"time per step: ([\d.]+)ms", joined)
    tps = re.search(r"throughput: ([\d.]+) tok/s", joined)
    print(f"{i:<5} {wall:>7.1f} {step.group(1) if step else '?':>9} {tps.group(1) if tps else '?':>7}")
