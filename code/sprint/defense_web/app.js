'use strict';

const $ = (id) => document.getElementById(id);
const make = (tag, props = {}, children = []) => {
  const node = Object.assign(document.createElement(tag), props);
  for (const child of children) node.append(child);
  return node;
};

const DOMAINS = [
  ['Calendar & time', [0, 6, 29, 31, 37, 48, 49]],
  ['Smart home', [1, 8, 18, 23, 30, 33, 39, 40]],
  ['Music & media', [7, 20, 27, 35, 42, 44, 52, 53]],
  ['Audio', [14, 28, 34, 45]],
  ['Travel & local', [2, 11, 19, 41, 51]],
  ['Food & cooking', [3, 9, 16, 36]],
  ['Messages & social', [15, 17, 26, 32, 43, 46]],
  ['Lists', [21, 50, 54]],
  ['Knowledge & finance', [4, 10, 13, 22, 25, 38, 47]],
  ['Chat & general', [5, 12, 24]],
  ['Fallback', [55]],
];
const DOMAIN_OF = new Map(DOMAINS.flatMap(([name, ids]) => ids.map((id) => [id, name])));
const SETS = {
  scripts: ['Devanagari', 'Hinglish with fillers', 'Mixed script', 'English', 'Contextual digits'],
  noisy: ['ASR spelling', 'Hinglish with fillers', 'Contextual digits', 'Mixed script'],
  fallback: ['Trained fallback topic', 'Human-authored unsupported', 'Rephrase prompt'],
};
const BADGES = { accept: 'Routed', clarify: 'Ask to rephrase · Class 55', reject: 'Declined · Class 55' };
const SCRIPT_NAMES = { devanagari: 'Devanagari', roman: 'Roman', mixed: 'Mixed script' };
const CIRCUMFERENCE = 2 * Math.PI * 50;

const state = {
  ready: false, busy: false, names: [], cases: [], model: null, latencies: [], history: [], set: 'scripts',
};
const query = $('query');
const form = $('query-form');

/* ---------- theme ---------- */
const storage = {
  get(key) { try { return localStorage.getItem(key); } catch { return null; } },
  set(key, value) { try { localStorage.setItem(key, value); } catch { /* private mode */ } },
};
function applyTheme(theme) {
  if (theme) document.documentElement.dataset.theme = theme;
  else delete document.documentElement.dataset.theme;
}
applyTheme(new URLSearchParams(location.search).get('theme') || storage.get('anveshan-theme'));
$('theme').addEventListener('click', () => {
  const current = document.documentElement.dataset.theme
    || (matchMedia('(prefers-color-scheme: light)').matches ? 'light' : 'dark');
  const next = current === 'dark' ? 'light' : 'dark';
  applyTheme(next);
  storage.set('anveshan-theme', next);
});

/* ---------- live script meter ---------- */
function scriptMix(text) {
  const deva = (text.match(/[ऀ-ॿ]/g) || []).length;
  const roman = (text.match(/[A-Za-z]/g) || []).length;
  return { deva, roman, total: deva + roman };
}
function updateScriptMeter() {
  const { deva, roman, total } = scriptMix(query.value);
  const chip = $('script-chip');
  let kind = 'empty';
  if (total) kind = deva && roman ? 'mixed' : deva ? 'devanagari' : 'roman';
  chip.dataset.script = kind;
  chip.textContent = kind === 'empty' ? 'Waiting for text'
    : kind === 'mixed' ? `Mixed · ${Math.round((100 * deva) / total)}% Devanagari` : SCRIPT_NAMES[kind];
  $('bar-deva').style.width = total ? `${(100 * deva) / total}%` : '0';
  $('bar-roman').style.width = total ? `${(100 * roman) / total}%` : '0';
}

/* ---------- controls ---------- */
function updateControls() {
  $('run').disabled = !state.ready || state.busy || !query.value.trim();
  $('clear').disabled = state.busy || (!query.value && $('decision').dataset.state === 'idle');
  $('run-checks').disabled = !state.ready || state.busy || !state.cases.length;
  for (const chip of document.querySelectorAll('.chip')) chip.disabled = !state.ready || state.busy;
}
function onInput() {
  $('query-length').textContent = `${query.value.length} / ${query.maxLength}`;
  updateScriptMeter();
  updateControls();
}

/* ---------- examples ---------- */
function renderChips() {
  const holder = $('chips');
  holder.replaceChildren();
  const seen = new Set();
  for (const name of SETS[state.set]) {
    for (const item of state.cases.filter((c) => c.name === name)) {
      if (seen.has(item.text)) continue;
      seen.add(item.text);
      const chip = make('button', { type: 'button', className: 'chip', title: item.text }, [
        make('small', { textContent: item.name }), make('span', { textContent: item.text }),
      ]);
      chip.addEventListener('click', () => {
        query.value = item.text;
        onInput();
        form.requestSubmit();
      });
      holder.append(chip);
    }
  }
  updateControls();
}
for (const tab of document.querySelectorAll('[data-set]')) {
  tab.addEventListener('click', () => {
    state.set = tab.dataset.set;
    for (const other of document.querySelectorAll('[data-set]')) other.setAttribute('aria-selected', String(other === tab));
    renderChips();
  });
}

/* ---------- intent map ---------- */
function renderMap() {
  const holder = $('domains');
  holder.replaceChildren();
  for (const [name, ids] of DOMAINS) {
    const tiles = make('div', { className: 'tiles' });
    for (const id of ids) {
      const label = state.names[id] || `Intent ${id}`;
      const tile = make('span', { className: 'tile', textContent: id === 55 ? '55 · fallback' : String(id), title: `${id} · ${label}` });
      tile.dataset.intent = id;
      if (id === 55) tile.dataset.fallback = 'true';
      tile.setAttribute('role', 'img');
      tile.setAttribute('aria-label', `${id}, ${label}`);
      tiles.append(tile);
    }
    holder.append(make('div', { className: 'domain-group' }, [make('h4', { textContent: name }), tiles]));
  }
}
function paintMap(result) {
  const probabilities = result?.probabilities || [];
  const peak = Math.max(...probabilities, 1e-9);
  for (const tile of document.querySelectorAll('.tile')) {
    const id = Number(tile.dataset.intent);
    const share = result ? Math.sqrt((probabilities[id] || 0) / peak) : 0;
    tile.style.setProperty('--p', share.toFixed(3));
    tile.dataset.hot = String(share > 0.55);
    tile.dataset.chosen = String(Boolean(result) && id === result.intent_id);
    const p = result ? ` · ${(100 * (probabilities[id] || 0)).toFixed(1)}%` : '';
    tile.title = `${id} · ${state.names[id] || ''}${p}`;
  }
}

/* ---------- gauge ---------- */
function layoutGauge() {
  const policy = state.model?.policy;
  if (!policy) return;
  const declineBelow = 1 - policy.threshold;
  const routeAbove = 1 - policy.accept_below;
  $('zone-reject').style.width = `${100 * declineBelow}%`;
  $('zone-clarify').style.width = `${100 * (routeAbove - declineBelow)}%`;
  $('zone-accept').style.width = `${100 * (1 - routeAbove)}%`;
}
function setNeedle(confidence) {
  const needle = $('needle');
  if (confidence == null) {
    needle.hidden = true;
    $('confidence').textContent = '—';
    return;
  }
  needle.hidden = false;
  needle.style.left = `${Math.min(100, Math.max(0, 100 * confidence))}%`;
  $('confidence').textContent = `${(100 * confidence).toFixed(1)}%`;
}

/* ---------- latency ---------- */
function percentile(values, q) {
  if (!values.length) return null;
  const sorted = [...values].sort((a, b) => a - b);
  return sorted[Math.min(sorted.length - 1, Math.ceil(q * sorted.length) - 1)];
}
function showLatency(ms) {
  const limit = state.model?.latency_limit_ms || 50;
  const ring = $('ring');
  $('latency').textContent = ms == null ? '—' : ms.toFixed(1);
  ring.style.strokeDashoffset = ms == null ? CIRCUMFERENCE : CIRCUMFERENCE * (1 - Math.min(ms / limit, 1));
  ring.dataset.slow = String(ms != null && ms >= limit);
  const values = state.latencies;
  const fmt = (v) => (v == null ? '—' : `${v.toFixed(1)} ms`);
  $('p50').textContent = fmt(percentile(values, 0.5));
  $('p95').textContent = fmt(percentile(values, 0.95));
  $('count').textContent = String(values.length);
  drawSpark(limit);
}
function drawSpark(limit) {
  const svg = $('spark');
  svg.replaceChildren();
  const values = state.latencies.slice(-40);
  if (values.length < 2) return;
  const top = Math.max(limit * 1.2, ...values);
  const ns = 'http://www.w3.org/2000/svg';
  const y = (v) => 38 - (36 * v) / top;
  const budget = document.createElementNS(ns, 'line');
  for (const [k, v] of Object.entries({ x1: 0, x2: 200, y1: y(limit), y2: y(limit) })) budget.setAttribute(k, v);
  const line = document.createElementNS(ns, 'polyline');
  line.setAttribute('points', values.map((v, i) => `${(200 * i) / (values.length - 1)},${y(v).toFixed(2)}`).join(' '));
  svg.append(budget, line);
}

/* ---------- result rendering ---------- */
function setDecision(stateName, badge, title, note, id = '—', domain = '') {
  $('decision').dataset.state = stateName;
  $('badge').textContent = badge;
  $('decision-title').textContent = title;
  $('decision-note').textContent = note;
  $('intent-id').textContent = id;
  $('domain').textContent = domain;
}
function setStep(step, text, meta, { active = true, skipped = false } = {}) {
  const item = document.querySelector(`[data-step="${step}"]`);
  item.dataset.active = String(active);
  item.dataset.skipped = String(skipped);
  const ids = { input: 's-input', normalize: 's-normalized', dictionary: 's-dictionary', model: 's-model', policy: 's-policy' };
  $(ids[step]).textContent = text;
  const metaIds = { input: 's-input-meta', dictionary: 's-dictionary-meta', model: 's-model-meta' };
  if (meta != null && metaIds[step]) $(metaIds[step]).textContent = meta;
}
function resetSteps() {
  for (const step of ['input', 'normalize', 'dictionary', 'model', 'policy']) setStep(step, '—', null, { active: false });
}
function renderBars(result) {
  const list = $('bars');
  list.replaceChildren();
  for (const [index, alt] of result.alternatives.entries()) {
    const fill = make('i');
    fill.style.width = `${Math.max(1.5, 100 * alt.probability)}%`;
    list.append(make('li', {}, [
      make('div', { className: 'bar-label' }, [
        make('span', {}, [make('em', { textContent: String(alt.intent_id).padStart(2, '0') }), alt.intent_name]),
        make('b', { textContent: `${(100 * alt.probability).toFixed(1)}%` }),
      ]),
      make('div', { className: 'bar', title: index === 0 ? 'Model’s top choice' : '' }, [fill]),
    ]));
  }
}
function showResult(text, result) {
  const policy = state.model.policy;
  const name = result.action === 'accept' ? result.intent_name
    : result.action === 'clarify' ? 'Ask the user to rephrase' : 'Decline safely';
  const topName = state.names[result.top_model_intent] || `Intent ${result.top_model_intent}`;
  const note = result.action === 'accept' ? `Supported request · ${DOMAIN_OF.get(result.intent_id) || ''}`
    : result.top_model_intent === 55 ? 'The model itself predicted Class 55 (unsupported request).'
      : `Closest supported intent was “${topName}”, but confidence was too low to act on it.`;
  setDecision(result.action, BADGES[result.action], name, note, String(result.intent_id), DOMAIN_OF.get(result.intent_id) || '');
  setNeedle(result.top_confidence);
  state.latencies.push(result.inference_ms);
  showLatency(result.inference_ms);

  setStep('input', text, `${SCRIPT_NAMES[result.input_script] || result.input_script} · ${text.length} characters`);
  setStep('normalize', result.normalized_text === text ? 'Unchanged' : result.normalized_text);
  const second = result.dictionary_text;
  if (!second || result.views === 1) {
    setStep('dictionary', result.length_guard_applied ? 'Skipped by the length guard' : 'No second view needed', 'Single input view',
      { active: false, skipped: true });
  } else {
    setStep('dictionary', second === result.normalized_text ? 'Same as the normalized text' : second,
      result.view === 'Dictionary' ? 'Cross-script dictionary rewrite' : 'Normalized second view');
  }
  setStep('model', `${result.views} view${result.views > 1 ? 's' : ''} → intent ${result.top_model_intent}`,
    `${topName} · ${(100 * result.top_confidence).toFixed(1)}% confidence`);
  const rule = result.top_model_intent === 55 ? 'Model chose Class 55 → decline'
    : `Risk ${(100 * result.risk).toFixed(1)}% · route < ${(100 * policy.accept_below).toFixed(1)}%, decline ≥ ${(100 * policy.threshold).toFixed(1)}%`;
  setStep('policy', `${BADGES[result.action].split(' · ')[0]} → ${result.intent_id}`, null);
  document.querySelector('[data-step="policy"] small').textContent = rule;

  renderBars(result);
  paintMap(result);
  addHistory(text, result);
}
function addHistory(text, result) {
  state.history.unshift({ text, result });
  state.history = state.history.slice(0, 14);
  const list = $('history');
  list.replaceChildren();
  for (const item of state.history) {
    const button = make('button', { type: 'button', title: item.text }, [
      make('span', { className: 'dot' }),
      make('span', { className: 'q', textContent: item.text }),
      make('span', { className: 'meta', textContent: `${item.result.intent_id} · ${item.result.inference_ms.toFixed(1)} ms` }),
    ]);
    button.dataset.action = item.result.action;
    button.addEventListener('click', () => {
      query.value = item.text;
      onInput();
      form.requestSubmit();
    });
    list.append(make('li', {}, [button]));
  }
}

/* ---------- network ---------- */
async function predict(text) {
  for (let attempt = 0; attempt < 3; attempt += 1) {
    const response = await fetch('/api/predict', {
      method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ text }),
      signal: AbortSignal.timeout(8000),
    });
    const data = await response.json();
    if (response.status === 429) {
      await new Promise((resolve) => setTimeout(resolve, 60));
      continue;
    }
    if (!response.ok) throw new Error(data.error || 'Inference failed.');
    return data;
  }
  throw new Error('The model is busy. Try again.');
}

async function connect() {
  $('reconnect').hidden = true;
  $('runtime-status').dataset.status = 'loading';
  $('runtime-label').textContent = 'Connecting to local model';
  try {
    const get = async (path) => {
      const response = await fetch(path, { signal: AbortSignal.timeout(4000) });
      if (!response.ok) throw new Error(path);
      return response.json();
    };
    const [model, names, cases] = await Promise.all([get('/api/model'), get('/api/intents'), get('/api/cases')]);
    Object.assign(state, { model, names, cases, ready: true });
    query.maxLength = model.max_query_chars;
    $('runtime-status').dataset.status = 'ready';
    $('runtime-label').textContent = `Local CPU · ${model.threads} threads`;
    const millions = model.parameters / 1e6;
    $('stat-params').textContent = `${millions.toFixed(1)}M`;
    $('param-value').textContent = `${millions.toFixed(1)}M / ${(model.parameter_limit / 1e6).toFixed(0)}M`;
    $('param-bar').style.width = `${(100 * (model.parameters + model.dictionary_entries)) / model.parameter_limit}%`;
    $('param-note').textContent = `${model.dictionary_entries.toLocaleString('en-US')} dictionary entries counted toward the cap`;
    $('parity').textContent = `${model.matching_submission_rows.toLocaleString('en-US')} / ${model.submission_rows.toLocaleString('en-US')}`;
    $('release').textContent = `v${model.version}`;
    $('runtime-detail').textContent = `Bhasha · ONNX ${model.encoder_precision} · revision ${String(model.checkpoint_revision).slice(0, 8)}`;
    $('submission-hash').textContent = model.submission_sha256;
    $('model-hash').textContent = model.model_sha256;
    $('policy-hash').textContent = model.policy_sha256;
    layoutGauge();
    renderMap();
    renderChips();
    onInput();
  } catch {
    state.ready = false;
    $('runtime-status').dataset.status = 'offline';
    $('runtime-label').textContent = 'Model unavailable';
    $('reconnect').hidden = false;
  }
  updateControls();
}

async function infer() {
  const text = query.value.trim();
  if (!text || state.busy || !state.ready) return;
  state.busy = true;
  query.readOnly = true;
  updateControls();
  $('decision').setAttribute('aria-busy', 'true');
  setDecision('running', 'Running', 'Scoring your request…', 'Bhasha is running on this machine’s CPU.');
  try {
    showResult(text, await predict(text));
  } catch (error) {
    const offline = error.name === 'TimeoutError' || error instanceof TypeError;
    setDecision('error', 'Error', 'Could not run inference', offline ? 'Check that the local server is still running, then retry.' : error.message);
    if (offline) await connect();
  } finally {
    $('decision').removeAttribute('aria-busy');
    state.busy = false;
    query.readOnly = false;
    updateControls();
  }
}

async function runChecks() {
  if (state.busy || !state.ready) return;
  state.busy = true;
  updateControls();
  const body = $('check-rows');
  body.replaceChildren();
  const rows = state.cases.map((item) => {
    const result = make('span', { className: 'pill wait', textContent: 'Waiting' });
    const ms = make('td', { className: 'num', textContent: '—' });
    const expected = `${item.intent_id} · ${(item.actions || [item.action]).join(' / ')}`;
    body.append(make('tr', {}, [
      make('td', { textContent: item.name }), make('td', { className: 'req', textContent: item.text, title: item.text }),
      make('td', { className: 'muted', textContent: expected }), make('td', {}, [result]), ms,
    ]));
    return { item, result, ms };
  });
  let passed = 0;
  const times = [];
  for (const row of rows) {
    try {
      const output = await predict(row.item.text);
      const ok = output.intent_id === row.item.intent_id && (row.item.actions || [row.item.action]).includes(output.action);
      passed += ok;
      times.push(output.inference_ms);
      state.latencies.push(output.inference_ms);
      row.result.className = `pill ${ok ? 'pass' : 'fail'}`;
      row.result.textContent = ok ? `Pass · ${output.intent_id}` : `Got ${output.intent_id} · ${output.action}`;
      row.ms.textContent = output.inference_ms.toFixed(1);
    } catch (error) {
      row.result.className = 'pill fail';
      row.result.textContent = error.message;
    }
  }
  $('check-summary').hidden = false;
  $('check-score').textContent = `${passed} / ${rows.length}`;
  const p95 = percentile(times, 0.95);
  $('check-detail').textContent = `cases passed · p95 ${p95 == null ? '—' : p95.toFixed(1)} ms · max ${times.length ? Math.max(...times).toFixed(1) : '—'} ms`;
  showLatency(times.at(-1) ?? null);
  state.busy = false;
  updateControls();
}

/* ---------- wiring ---------- */
query.addEventListener('input', onInput);
query.addEventListener('keydown', (event) => {
  if ((event.ctrlKey || event.metaKey) && event.key === 'Enter' && !event.isComposing) {
    event.preventDefault();
    form.requestSubmit();
  }
});
form.addEventListener('submit', (event) => {
  event.preventDefault();
  infer();
});
$('clear').addEventListener('click', () => {
  query.value = '';
  setDecision('idle', 'Ready', 'Waiting for a request',
    'Type a request or pick an example. The decision, its confidence and the reasoning steps appear here.');
  setNeedle(null);
  resetSteps();
  paintMap(null);
  $('bars').replaceChildren(make('li', { className: 'empty', textContent: 'Run a request to see the ranking.' }));
  onInput();
  query.focus();
});
$('reconnect').addEventListener('click', connect);
$('run-checks').addEventListener('click', runChecks);

renderMap();
updateControls();
connect().then(() => {
  const params = new URLSearchParams(location.search);
  const initial = params.get('q');
  const checks = params.get('checks') === '1' ? runChecks : async () => {};
  if (initial && state.ready) {
    query.value = initial.slice(0, query.maxLength);
    onInput();
    infer().then(checks);
  } else if (state.ready) {
    checks();
  }
});
