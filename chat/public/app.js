/* Chat where the diffusion happens inside the message bubbles: every
 * assistant reply denoises live in place (unresolved cells stay blank,
 * freshly-resolved characters glow), then settles into clean text. Each
 * bubble keeps its frames and carries a timeline — scrub through the
 * denoising or hit play to replay it; the last scrubber position is the
 * settled text. */

const $ = (id) => document.getElementById(id);
const statusEl = $('status'), messagesEl = $('messages');
const inputEl = $('input'), sendBtn = $('send');

const escapeHtml = (s) =>
  s.replaceAll('&', '&amp;').replaceAll('<', '&lt;').replaceAll('>', '&gt;');

function setStatus(text, cls) {
  statusEl.textContent = text;
  statusEl.className = cls;
}

// Only follow the stream if the user hasn't scrolled away.
const nearBottom = () =>
  messagesEl.scrollHeight - messagesEl.scrollTop - messagesEl.clientHeight < 120;
const scrollDown = (force = false) => {
  if (force || nearBottom()) messagesEl.scrollTop = messagesEl.scrollHeight;
};

/* ---------- frame rendering ---------- */

/* Render one frame into a grid whose size only ever GROWS during a turn
 * (monotonic `geom`), so the bubble never shrinks mid-stream and the layout
 * stays still. Unresolved cells render as plain spaces — wrapping each in an
 * animated span made the page janky at thousands of cells per frame. */
function frameHtml(allLines, prevLines, geom) {
  let last = allLines.length - 1;
  while (last >= 0 && !allLines[last].trim()) last--;
  geom.h = Math.max(geom.h, Math.min(allLines.length, last + 2));
  geom.w = Math.max(geom.w, ...allLines.slice(0, last + 1).map((l) => l.trimEnd().length));
  const out = [];
  for (let y = 0; y < geom.h; y++) {
    const line = allLines[y] ?? '';
    const prev = prevLines ? (prevLines[y] ?? '') : null;
    let row = '';
    for (let x = 0; x < geom.w; x++) {
      const ch = line[x] ?? ' ';
      if (ch !== ' ' && prev !== null && (prev[x] ?? ' ') === ' ') {
        row += `<span class="f">${escapeHtml(ch)}</span>`;
      } else {
        row += escapeHtml(ch);
      }
    }
    out.push(row);
  }
  return out.join('\n');
}

/* ---------- thought channel ---------- */

// Gemma emits "<|channel>thought … <channel|>answer". Split them apart.
const THOUGHT_OPEN = /^\s*<\|channel>thought\s*/i;
function splitChannels(text) {
  const end = text.lastIndexOf('<channel|>');
  if (end !== -1) {
    const thought = text.slice(0, end).replace(THOUGHT_OPEN, '').trim();
    return { thought, answer: text.slice(end + '<channel|>'.length).trim() };
  }
  // Opened a thought channel but never closed it: the canvas ran out before
  // the model got to the answer — everything is thought.
  if (THOUGHT_OPEN.test(text)) {
    return { thought: text.replace(THOUGHT_OPEN, '').trim(), answer: '' };
  }
  return { thought: null, answer: text.trim() };
}

/* ---------- messages ---------- */

// System divider, e.g. "backend restarted — conversation context lost".
function addNotice(text, beforeEl = null) {
  const div = document.createElement('div');
  div.className = 'notice';
  div.textContent = text;
  messagesEl.insertBefore(div, beforeEl);
  scrollDown();
}

function addUserMessage(text) {
  const div = document.createElement('div');
  div.className = 'msg user';
  div.textContent = text;
  messagesEl.appendChild(div);
  scrollDown(true);
}

/* An assistant bubble that morphs: status line -> live canvas -> settled
 * text, with a per-message timeline (frames 0..n-1, position n = text). */
function assistantBubble() {
  const el = document.createElement('div');
  el.className = 'msg assistant';
  el.innerHTML = `
    <div class="afoot" hidden>
      <button class="play" title="replay denoising">⟲</button>
      <input type="range" min="0" max="0" value="0" />
      <span class="meta"></span>
    </div>
    <div class="abody"><span class="dstatus">…</span></div>`;
  messagesEl.appendChild(el);

  const body = el.querySelector('.abody');
  const foot = el.querySelector('.afoot');
  const playBtn = el.querySelector('.play');
  const scrub = el.querySelector('input');
  const meta = el.querySelector('.meta');

  const frames = [];
  const geom = { w: 60, h: 6 };  // monotonic canvas grid, shared by live + replay
  let settledHtml = null;
  let canvas = null;        // the <pre> while in canvas mode
  let playTimer = null;

  const stopPlay = () => { if (playTimer) { clearInterval(playTimer); playTimer = null; } };

  // Progress fill for the timeline track (--p is a percentage, see CSS).
  const syncScrub = () =>
    scrub.style.setProperty('--p', `${(100 * Number(scrub.value)) / Math.max(1, Number(scrub.max))}%`);

  function showFrame(i) {
    if (!canvas) {
      canvas = document.createElement('pre');
      canvas.className = 'dcanvas';
      body.replaceChildren(canvas);
    }
    canvas.innerHTML = frameHtml(frames[i].lines, i > 0 ? frames[i - 1].lines : null, geom);
  }

  function showSettled() {
    if (settledHtml === null) return;
    canvas = null;
    body.innerHTML = settledHtml;
  }

  function seek(i) {
    if (i >= frames.length && settledHtml !== null) showSettled();
    else if (frames[Math.min(i, frames.length - 1)]) showFrame(Math.min(i, frames.length - 1));
  }

  scrub.addEventListener('input', () => { stopPlay(); syncScrub(); seek(Number(scrub.value)); });

  playBtn.addEventListener('click', () => {
    stopPlay();
    if (!frames.length) return;
    let i = 0;
    playTimer = setInterval(() => {
      scrub.value = i;
      syncScrub();
      seek(i);
      if (i >= Number(scrub.max)) stopPlay();
      i += 1;
    }, 90);
  });

  return {
    el,
    status(text) {
      if (!canvas) body.innerHTML = `<span class="dstatus">${escapeHtml(text)}</span>`;
    },
    frame(ev) {
      frames.push(ev);
      foot.hidden = false;
      scrub.max = frames.length;      // last position reserved for settled text
      scrub.value = frames.length - 1;
      syncScrub();
      meta.textContent = `step ${frames.length}`;
      showFrame(frames.length - 1);
      scrollDown();
    },
    done(result) {
      const { thought, answer } = splitChannels(result.text);
      let html = '';
      if (thought) {
        // No answer = the thought ran out the canvas; open it so the bubble
        // isn't just a placeholder.
        html += `<details class="thought"${answer ? '' : ' open'}><summary>denoised thought channel</summary><pre>${escapeHtml(thought)}</pre></details>`;
      }
      const placeholder = thought
        ? '(no answer — the thought used the whole token budget; try raising tokens)'
        : '(empty)';
      html += `<div class="answer">${escapeHtml(answer || placeholder)}</div>`;
      settledHtml = html;
      const ms = result.stats.match(/total time:\s*([\d.]+)\s*ms/)?.[1];
      const tps = result.stats.match(/throughput:\s*([\d.]+)\s*tok\/s/)?.[1];
      meta.textContent = [`${result.steps} steps`, ms && `${Math.round(ms)} ms`, tps && `${Math.round(tps)} tok/s`]
        .filter(Boolean).join(' · ');
      meta.title = result.stats;
      foot.hidden = false;
      scrub.max = frames.length;
      scrub.value = frames.length;
      syncScrub();
      stopPlay();
      showSettled();
      scrollDown();
    },
    fail(message) {
      stopPlay();
      el.classList.add('error');
      body.textContent = `⚠ ${message}`;
      scrollDown();
    },
  };
}

/* ---------- generation ---------- */

async function generate(text, bubble) {
  const params = {
    tokens: Number($('p-tokens').value) || 256,
    maxSteps: Number($('p-steps').value) || undefined,
    interval: Number($('p-interval').value) || 1,
  };

  const res = await fetch('/api/generate', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ text, params }),
  });
  if (res.status === 409) throw new Error('a generation is already running');
  if (!res.ok) throw new Error(`bridge error ${res.status}`);

  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buf = '';
  let finished = false;

  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    buf += decoder.decode(value, { stream: true });
    const lines = buf.split('\n');
    buf = lines.pop();
    for (const line of lines) {
      if (!line.trim()) continue;
      const ev = JSON.parse(line);
      if (ev.type === 'status') { bubble.status(ev.message); setStatus(ev.message, 'busy'); }
      // The divider marks loss of everything before this turn, so it goes
      // above the user message that triggered it (bubble's previous sibling).
      else if (ev.type === 'notice') addNotice(ev.message, bubble.el.previousElementSibling ?? bubble.el);
      else if (ev.type === 'frame') { bubble.frame(ev); setStatus(`step ${ev.step}`, 'busy'); }
      else if (ev.type === 'done') { bubble.done(ev); finished = true; }
      else if (ev.type === 'error') throw new Error(ev.message);
    }
  }
  if (!finished) throw new Error('stream ended without a result');
}

$('composer').addEventListener('submit', async (e) => {
  e.preventDefault();
  const text = inputEl.value.trim();
  if (!text || sendBtn.disabled) return;
  inputEl.value = '';
  sendBtn.disabled = true;
  addUserMessage(text);
  const bubble = assistantBubble();
  bubble.status('…');
  setStatus('sending', 'busy');

  try {
    await generate(text, bubble);
    setStatus('idle', 'idle');
  } catch (err) {
    bubble.fail(err.message);
    setStatus('error', 'error');
  } finally {
    sendBtn.disabled = false;
    inputEl.focus();
  }
});

$('reset').addEventListener('click', async () => {
  const res = await fetch('/api/reset', { method: 'POST' });
  if (res.status === 409) { setStatus('busy — wait for the turn to finish', 'error'); return; }
  messagesEl.innerHTML = '';
  setStatus('idle', 'idle');
});

inputEl.addEventListener('keydown', (e) => {
  if (e.key === 'Enter' && !e.shiftKey) {
    e.preventDefault();
    $('composer').requestSubmit();
  }
});

inputEl.focus();
