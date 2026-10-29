import { RobotView } from './viewer.js';
import { showTrace } from './network.js';

const element = id => document.getElementById(id);
let config, viewer, joints = [], actual = [0, 0, 0], target = [0.4, 0, 0.4];
let version = 0, pending, busy = false, debounce;

function status(message, error = false) {
  element('status').textContent = message;
  element('status').classList.toggle('error', error);
}

async function request(url, data) {
  const response = await fetch(url, data === undefined ? {} : {
    method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(data),
  });
  const result = await response.json();
  if (!response.ok) throw new Error(result.error ?? 'Request failed');
  return result;
}

function updateError() {
  const error = Math.hypot(...target.map((value, i) => value - actual[i]));
  element('error').textContent = `${(error * 1000).toFixed(1)} mm`;
}

function setTarget(values, updateFields = true) {
  target = values;
  if (updateFields) ['x', 'y', 'z'].forEach((id, i) => { element(id).value = values[i].toFixed(3); });
  viewer?.setTarget(values);
  updateError();
}

function applyPose(result) {
  joints = result.joints;
  actual = result.actual;
  viewer?.setPose(config.names, joints, actual);
  config.names.forEach((name, i) => {
    const slider = element(`joint-${i}`);
    if (document.activeElement !== slider) slider.value = joints[i];
    element(`joint-value-${i}`).textContent = `${(joints[i] * 180 / Math.PI).toFixed(1)}°`;
  });
  element('actual').textContent = actual.map(value => value.toFixed(3)).join(' / ');
  element('latency').textContent = result.elapsed_ms === undefined ? '—' : `${result.elapsed_ms.toFixed(1)} ms`;
  updateError();
  showTrace(result.network_trace);
}

// Only one request runs at a time. Edits replace the pending request, and stale
// responses cannot overwrite a newer target or a manual joint change.
function queue(operation, data) {
  clearTimeout(debounce);
  pending = { operation, data, version: ++version };
  showTrace({ note: 'Waiting for the current prediction…' });
  drain();
}

async function drain() {
  if (busy || !pending) return;
  const job = pending;
  pending = undefined;
  busy = true;
  status(job.operation === 'predict' ? 'Predicting…' : 'Updating joint preview…');
  try {
    const result = await request(`/api/${job.operation}`, job.data);
    if (job.version === version) {
      applyPose(result);
      let message = job.operation === 'joints' ? 'Manual joint preview' :
        result.error < 0.01 ? 'Target reached' : 'Approximate pose · target not reached';
      if (result.clipped) message += ' · joint limits applied';
      status(message);
      element('connection').textContent = 'Local server connected';
    }
  } catch (error) {
    if (job.version === version) {
      status(error.message, true);
      showTrace({ note: `Prediction unavailable: ${error.message}` });
    }
  } finally {
    busy = false;
    drain();
  }
}

function readTarget() {
  const fields = ['x', 'y', 'z'].map(id => element(id).value.trim());
  const values = fields.map(Number);
  // Decimal precision is unrestricted; only empty, non-finite, or out-of-range values fail.
  if (fields.includes('') || values.some(value => !Number.isFinite(value) || value < -5 || value > 5)) return null;
  return values;
}

function solve() {
  const values = readTarget();
  if (!values) {
    status('Enter finite XYZ coordinates between −5 and +5 metres.', true);
    return;
  }
  setTarget(values, false);
  queue('predict', { target, seed: joints, model: element('model').value, path: element('model-path').value });
}

function targetEdited() {
  ++version;
  pending = undefined;
  clearTimeout(debounce);
  const values = readTarget();
  if (values) setTarget(values, false);
  showTrace({ note: 'Target changed. Run a prediction to inspect this input.' });
  if (element('live').checked) debounce = setTimeout(solve, 200);
}

function modelOption(id, label) {
  if (![...element('model').options].some(option => option.value === id)) {
    element('model').add(new Option(label, id));
  }
  element('model').value = id;
}

async function start() {
  config = await request('/api/config');
  joints = config.names.map(() => 0);
  element('robot-info').textContent = `Frame: ${config.frame} · Tip: ${config.tip}`;
  config.models.forEach(name => element('model').add(new Option(name, name)));
  config.names.forEach((name, i) => {
    const row = document.createElement('div'); row.className = 'joint';
    const label = document.createElement('label'); label.htmlFor = `joint-${i}`;
    const text = document.createElement('span'); text.textContent = name.replace(/_joint$/, '').replaceAll('_', ' ');
    const output = document.createElement('output'); output.id = `joint-value-${i}`; output.textContent = '0.0°';
    label.append(text, output);
    const slider = document.createElement('input'); slider.type = 'range'; slider.id = `joint-${i}`;
    slider.min = config.lower[i]; slider.max = config.upper[i]; slider.step = 0.005; slider.value = 0;
    slider.oninput = () => {
      joints[i] = Number(slider.value);
      output.textContent = `${(joints[i] * 180 / Math.PI).toFixed(1)}°`;
      queue('joints', { joints: [...joints] });
    };
    row.append(label, slider); element('joints').append(row);
  });
  try {
    viewer = new RobotView(element('viewer'), values => {
      // TransformControls already moved the target, so only update the form.
      target = values;
      ['x', 'y', 'z'].forEach((id, i) => { element(id).value = values[i].toFixed(3); });
      ++version; pending = undefined; clearTimeout(debounce); updateError();
      showTrace({ note: 'Target moved. Release the handle to predict.' });
    }, () => solve(), message => { element('viewer-status').textContent = message; });
    viewer.setTarget(target);
  } catch (error) {
    element('viewer-status').textContent = `3D view unavailable: ${error.message}`;
  }
  applyPose(await request('/api/joints', { joints }));
  element('connection').textContent = 'Local server connected';
  element('solve').onclick = solve;
  ['x', 'y', 'z'].forEach(id => { element(id).oninput = targetEdited; });
  element('live').onchange = () => { if (!element('live').checked) clearTimeout(debounce); };
  element('model').onchange = () => {
    element('model-description').textContent = element('model').value === 'numerical' ?
      'Numerical preview. No neural network is used.' : 'CPU prediction · recorded layer values appear below.';
    solve();
  };
  element('home').onclick = () => queue('joints', { joints: config.names.map(() => 0) });
  element('reset-camera').onclick = () => viewer?.resetCamera();
  element('collision').onchange = event => viewer?.showCollision(event.target.checked);
  element('obstacle').onchange = event => viewer?.setObstacles(event.target.checked ?
    [{ position: [0.4, 0.2, 0.25], size: [0.2, 0.2, 0.5] }] : []);
  element('load-path').onclick = () => {
    if (!element('model-path').value.trim()) { status('Enter a checkpoint path.', true); return; }
    modelOption('__custom__', 'Local checkpoint · default MLP'); solve();
  };
  element('checkpoint').onchange = async event => {
    const file = event.target.files[0];
    if (!file) return;
    try {
      status('Loading checkpoint…');
      const response = await fetch('/api/checkpoint', { method: 'POST',
        headers: { 'Content-Type': 'application/octet-stream', 'X-Filename': encodeURIComponent(file.name) }, body: file });
      const result = await response.json();
      if (!response.ok) throw new Error(result.error);
      modelOption(result.id, result.name); solve();
    } catch (error) { status(error.message, true); }
  };
}

start().catch(error => {
  status(`Could not initialize the studio: ${error.message}`, true);
  element('connection').textContent = 'Server unavailable · check the terminal';
});
