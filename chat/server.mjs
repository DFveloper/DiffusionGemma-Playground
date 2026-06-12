// Bridge between the browser and llama-diffusion-cli (llama.cpp PR #24423).
//
// The PR has no llama-server integration, so the bridge keeps ONE persistent
// container alive in -cnv conversation mode (model stays in VRAM, multi-turn
// context is native) and pipes prompts over stdin. The CLI sizes its visual
// canvas via TTY ioctl, so it runs under `script` with an `stty`-sized pty —
// otherwise the canvas is clipped to the 80x24 non-TTY default.
//
// Output protocol per turn: one full canvas snapshot per denoising step,
// wrapped in ESC[?2026h ... ESC[?2026l (synchronized update), then the final
// text + two stats lines, then a "\n> " readline marker. Frames are streamed
// to the client as NDJSON over a plain POST response. Zero npm dependencies.

import http from 'node:http';
import { spawn, spawnSync } from 'node:child_process';
import { readFile } from 'node:fs/promises';
import { join, extname } from 'node:path';
import { fileURLToPath } from 'node:url';

const PORT = Number(process.env.PORT || 8090);
const MODELS_HOST_DIR = process.env.MODELS_HOST_DIR || `${process.env.HOME}/models`;
const BACKEND_IMAGE = process.env.BACKEND_IMAGE || 'diffusion-gemma:latest';
const MODEL_FILE = process.env.MODEL_FILE || 'diffusiongemma-26B-A4B-it-Q4_K_M.gguf';
const SESSION_NAME = process.env.SESSION_NAME || 'diffusion-session';
const CANVAS_ROWS = Number(process.env.CANVAS_ROWS || 64);
const CANVAS_COLS = Number(process.env.CANVAS_COLS || 150);
const READY_TIMEOUT_MS = 4 * 60 * 1000;
const TURN_TIMEOUT_MS = 5 * 60 * 1000;
const DEFAULT_PARAMS = { tokens: 256, maxSteps: 48, interval: 1 };

const PUBLIC_DIR = join(fileURLToPath(new URL('.', import.meta.url)), 'public');
const MIME = { '.html': 'text/html', '.js': 'text/javascript', '.css': 'text/css', '.svg': 'image/svg+xml' };

const SYNC_START = '\x1b[?2026h';
const SYNC_END = '\x1b[?2026l';
const ANSI_RE = /\x1b\[[0-9;?]*[A-Za-z]/g;
const PROMPT_RE = /(^|\n)> $/; // cnv readline marker, only ever at buffer end

const stripAnsi = (s) => s.replace(ANSI_RE, '');

/* ---------------- persistent backend session ---------------- */

let session = null;
let turnsServed = 0; // turns completed since the last context reset (crash, restart, /clear)

function spawnSession(params) {
  spawnSync('docker', ['rm', '-f', SESSION_NAME], { stdio: 'ignore' });

  const cli = [
    'llama-diffusion-cli',
    '-m', `/root/.cache/llama.cpp/${MODEL_FILE}`,
    '-ngl', '99', '-cnv', '--diffusion-visual',
    '-n', String(params.tokens),
    '--diffusion-visual-interval', String(params.interval),
    ...(params.maxSteps ? ['--diffusion-eb-max-steps', String(params.maxSteps)] : []),
  ].join(' ');
  // pty sized via stty so the canvas isn't clipped; CLI stderr (load logs)
  // dropped so the pty stream stays parseable.
  const inner = `stty rows ${CANVAS_ROWS} cols ${CANVAS_COLS} -echo; exec ${cli} 2>/dev/null`;

  const child = spawn('docker', [
    'run', '--rm', '-i', '--name', SESSION_NAME, '--gpus', 'all',
    '-v', `${MODELS_HOST_DIR}:/root/.cache/llama.cpp`,
    '--entrypoint', 'script', BACKEND_IMAGE, '-qfc', inner, '/dev/null',
  ], { stdio: ['pipe', 'pipe', 'pipe'] });

  const s = {
    child,
    key: JSON.stringify(params),
    buf: '',
    ready: false,
    fresh: true,       // no turn served yet — context starts empty
    sink: null,        // active turn: { frame(), done(), fail() }
    step: 0,
    stderrTail: '',
  };
  s.readyPromise = new Promise((resolve, reject) => {
    const t = setTimeout(() => reject(new Error('backend did not become ready in time')), READY_TIMEOUT_MS);
    s.resolveReady = () => { clearTimeout(t); resolve(); };
    child.on('exit', () => { clearTimeout(t); reject(new Error(`backend exited during startup\n${s.stderrTail}`)); });
  });
  s.readyPromise.catch(() => {}); // avoid unhandled rejection when nobody awaits yet

  child.stdout.on('data', (chunk) => onSessionData(s, chunk.toString('utf8').replaceAll('\r', '')));
  child.stderr.on('data', (chunk) => { s.stderrTail = (s.stderrTail + chunk).slice(-4000); });
  child.on('exit', (code) => {
    s.sink?.fail(new Error(`backend exited (code ${code})\n${s.stderrTail}`));
    if (session === s) session = null;
  });

  console.log(`session: starting backend (${s.key})`);
  return s;
}

function onSessionData(s, text) {
  s.buf += text;

  // Extract every complete denoising frame currently in the buffer.
  for (;;) {
    const start = s.buf.indexOf(SYNC_START);
    if (start === -1) break;
    const end = s.buf.indexOf(SYNC_END, start);
    if (end === -1) break;
    const lines = stripAnsi(s.buf.slice(start + SYNC_START.length, end)).split('\n');
    s.buf = s.buf.slice(end + SYNC_END.length);
    s.step += 1;
    s.sink?.frame(s.step, lines);
  }

  // "\n> " at buffer end = readline waiting -> startup done or turn complete.
  if (!PROMPT_RE.test(s.buf)) return;
  if (!s.ready) {
    s.ready = true;
    s.resolveReady();
    console.log('session: backend ready (model loaded)');
  } else if (s.sink) {
    const lines = stripAnsi(s.buf).split('\n');
    const statsIdx = lines.findIndex((l) => l.startsWith('total time:'));
    const text_ = (statsIdx === -1 ? lines : lines.slice(0, statsIdx)).join('\n').trim();
    const stats = statsIdx === -1 ? '' : lines.slice(statsIdx, -1).join('\n').trim();
    const sink = s.sink;
    s.sink = null;
    sink.done({ steps: s.step, text: text_, stats });
  }
  s.buf = '';
  s.step = 0;
}

function stopSession() {
  session?.child.kill('SIGKILL');
  spawnSync('docker', ['rm', '-f', SESSION_NAME], { stdio: 'ignore' });
  session = null;
}

async function ensureSession(params, onStatus) {
  const key = JSON.stringify(params);
  if (session && session.key !== key) {
    onStatus('parameters changed — restarting backend (model reload)…');
    stopSession();
  }
  if (!session) {
    session = spawnSession(params);
    onStatus('loading model…');
  }
  if (!session.ready) onStatus('loading model…');
  await session.readyPromise;
  return session;
}

/* ---------------- http handlers ---------------- */

function handleGenerate(req, res) {
  let body = '';
  req.on('data', (c) => { body += c; });
  req.on('end', async () => {
    let payload;
    try {
      payload = JSON.parse(body);
    } catch {
      res.writeHead(400).end('bad json');
      return;
    }
    if (session?.sink) {
      res.writeHead(409, { 'Content-Type': 'application/json' });
      res.end(JSON.stringify({ error: 'a generation is already running' }));
      return;
    }

    res.writeHead(200, {
      'Content-Type': 'application/x-ndjson',
      'Cache-Control': 'no-cache',
      'X-Accel-Buffering': 'no',
    });
    const emit = (obj) => { if (!res.writableEnded) res.write(JSON.stringify(obj) + '\n'); };

    let s;
    try {
      s = await ensureSession({ ...DEFAULT_PARAMS, ...payload.params }, (m) => emit({ type: 'status', message: m }));
    } catch (err) {
      emit({ type: 'error', message: err.message });
      res.end();
      stopSession();
      return;
    }

    // A fresh session mid-conversation (crash or param restart) means the
    // model lost the chat history still shown in the UI — say so.
    if (s.fresh) {
      if (turnsServed > 0) emit({ type: 'notice', message: 'backend restarted — the model lost the previous conversation context' });
      turnsServed = 0;
      s.fresh = false;
    }

    const turnTimer = setTimeout(() => {
      emit({ type: 'error', message: 'turn timed out — backend restarted' });
      res.end();
      stopSession();
    }, TURN_TIMEOUT_MS);

    s.sink = {
      frame: (step, lines) => emit({ type: 'frame', step, lines }),
      done: (result) => { clearTimeout(turnTimer); turnsServed += 1; emit({ type: 'done', ...result }); res.end(); },
      fail: (err) => { clearTimeout(turnTimer); emit({ type: 'error', message: err.message }); res.end(); },
    };
    // Client gone mid-turn: stop streaming but let the turn finish so the
    // readline protocol stays in sync (the warm model is worth keeping).
    res.on('close', () => {
      if (res.writableEnded || s.sink === null) return;
      clearTimeout(turnTimer);
      s.sink = { frame: () => {}, done: () => { turnsServed += 1; }, fail: () => {} };
    });

    emit({ type: 'status', message: 'denoising…' });
    // cnv input is line-based: collapse whitespace so multiline prompts
    // don't submit early.
    s.child.stdin.write(payload.text.replace(/\s+/g, ' ').trim() + '\n');
  });
}

function handleReset(res) {
  if (session?.sink) {
    res.writeHead(409, { 'Content-Type': 'application/json' });
    res.end(JSON.stringify({ error: 'busy' }));
    return;
  }
  if (session?.ready) session.child.stdin.write('/clear\n');
  turnsServed = 0;
  res.writeHead(200, { 'Content-Type': 'application/json' });
  res.end(JSON.stringify({ ok: true }));
}

async function handleStatic(req, res) {
  const path = req.url === '/' ? '/index.html' : req.url.split('?')[0];
  try {
    const data = await readFile(join(PUBLIC_DIR, path));
    res.writeHead(200, { 'Content-Type': MIME[extname(path)] ?? 'application/octet-stream' });
    res.end(data);
  } catch {
    res.writeHead(404).end('not found');
  }
}

http
  .createServer((req, res) => {
    if (req.method === 'POST' && req.url === '/api/generate') return handleGenerate(req, res);
    if (req.method === 'POST' && req.url === '/api/reset') return handleReset(res);
    if (req.method === 'GET') return handleStatic(req, res);
    res.writeHead(405).end();
  })
  .listen(PORT, () => {
    console.log(`diffusion chat bridge on http://localhost:${PORT}`);
    console.log(`  image:  ${BACKEND_IMAGE}`);
    console.log(`  model:  ${MODELS_HOST_DIR}/${MODEL_FILE}`);
    console.log(`  canvas: ${CANVAS_ROWS}x${CANVAS_COLS} pty`);
    // Warm start: load the model while the user is still typing.
    session = spawnSession(DEFAULT_PARAMS);
  });

for (const sig of ['SIGINT', 'SIGTERM']) {
  process.on(sig, () => { stopSession(); process.exit(0); });
}
