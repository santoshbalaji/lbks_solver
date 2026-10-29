// DOM-level control tests, without a browser or third-party test dependencies.
// Run with: node --experimental-vm-modules lbks_gui/test/test_frontend.mjs
import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import vm from 'node:vm';
import { fileURLToPath } from 'node:url';

export async function run() {
  const root = fileURLToPath(new URL('../web/', import.meta.url));
  const ids = new Map();
  class Element {
    constructor(tag = 'div') {
      this.tag = tag; this.children = []; this.value = ''; this.textContent = '';
      this.checked = false; this.style = {}; this.disabled = false;
      this.classList = { toggle() {} };
    }
    set id(value) { this._id = value; ids.set(value, this); }
    get id() { return this._id; }
    append(...items) { this.children.push(...items); }
    replaceChildren(...items) { this.children = items; }
    add(item) { this.children.push(item); }
    get options() { return this.children; }
    setAttribute(name, value) { this[name] = value; }
    scrollTo() {}
    get offsetLeft() { return 0; }
  }
  const html = await fs.readFile(root + 'index.html', 'utf8');
  for (const id of ['x', 'y', 'z']) {
    assert.match(html, new RegExp(`id="${id}"[^>]*step="any"`));
  }
  for (const match of html.matchAll(/id="([^"]+)"/g)) new Element().id = match[1];
  const get = id => ids.get(id);
  get('x').value = '0.4'; get('y').value = '0'; get('z').value = '0.4';
  get('live').checked = true; get('model').value = 'numerical';
  const document = { getElementById: get, createElement: tag => new Element(tag), activeElement: null };
  class Option extends Element { constructor(text, value) { super('option'); this.textContent = text; this.value = value; } }
  let view;
  class RobotView {
    constructor(container, move, release) { view = this; this.move = move; this.release = release; }
    setTarget(value) { this.target = value; }
    setPose(names, joints, actual) { this.joints = joints; this.actual = actual; }
    resetCamera() {} showCollision() {} setObstacles() {}
  }
  const trace = { parameters: 12, note: 'Test trace', stages: [
    { name: 'Input', kind: 'input', values: [.4, 0, .4], labels: ['X','Y','Z'], count: 3, min: 0, max: .4, mean: .267, zeros: 1 },
    { name: 'Linear', kind: 'Linear', values: [.2], labels: [], count: 1, min: .2, max: .2, mean: .2, zeros: 0,
      neurons: [{ bias: .1, other_sum: 0, terms: [{input: 0, value: .4, weight: .25, product: .1}] }] },
  ] };
  const predict = [], requests = [], intervals = new Map();
  let intervalId = 0;
  function response(data) { return { ok: true, json: async () => data }; }
  function pose(q = 0) { return { joints: [q, q], actual: [.4, 0, .4], error: 0, elapsed_ms: 10, network_trace: trace }; }
  const context = vm.createContext({ document, Option, console, setTimeout, clearTimeout,
    setInterval(fn) { intervals.set(++intervalId, fn); return intervalId; },
    clearInterval(id) { intervals.delete(id); },
    fetch: async (url, options) => {
      requests.push(url);
      if (url === '/api/config') return response({names:['joint_a','joint_b'], lower:[-3,-3], upper:[3,3], models:[], frame:'world', tip:'tool0'});
      if (url === '/api/joints') return response({...pose(), network_trace:{stages:[], note:'Manual preview'}});
      return new Promise(resolve => predict.push({data:JSON.parse(options.body), resolve}));
    },
  });
  const network = new vm.SourceTextModule(await fs.readFile(root + 'network.js','utf8'), {context});
  const viewer = new vm.SyntheticModule(['RobotView'], function() { this.setExport('RobotView', RobotView); }, {context});
  const app = new vm.SourceTextModule(await fs.readFile(root + 'app.js','utf8'), {context});
  await app.link(name => name === './viewer.js' ? viewer : network);
  await app.evaluate();
  const flush = () => new Promise(resolve => setTimeout(resolve, 0));
  await flush(); await flush();
  assert.equal(get('joints').children.length, 2);
  assert.equal(get('connection').textContent, 'Local server connected');

  get('solve').onclick(); await flush();
  assert.equal(predict.length, 1);
  get('x').value = '0.5'; get('x').oninput(); get('solve').onclick();
  predict[0].resolve(response(pose(1))); await flush(); await flush();
  assert.equal(view.joints[0], 0, 'stale prediction must not replace the pose');
  assert.equal(predict.length, 2);
  assert.equal(predict[1].data.target[0], .5);
  predict[1].resolve(response(pose(2))); await flush(); await flush();
  assert.equal(view.joints[0], 2);
  assert.equal(get('network-graph').children.length, 2);
  assert.equal(get('stage-count').textContent, '1 / 2');
  const linearValues = get('network-graph').children[1].children[2];
  linearValues.children[0].onclick();
  assert.match(get('neuron-detail').children[2].textContent, /Bias: 0.10000/);
  get('replay').onclick(); assert.equal(intervals.size, 1);
  [...intervals.values()][0](); assert.equal(get('stage-count').textContent, '2 / 2');
  [...intervals.values()][0](); assert.equal(intervals.size, 0);
  get('home').onclick(); await flush(); await flush();
  assert.equal(get('replay').disabled, true);
  assert.equal(view.joints[0], 0);
  assert.ok(requests.includes('/api/joints'));

  // Arbitrary decimal coordinates and the inclusive limits must be accepted.
  for (const value of ['0.401', '-0.12345', '5', '-5']) {
    const before = predict.length;
    get('x').value = value;
    get('solve').onclick(); await flush();
    assert.equal(predict.length, before + 1);
    assert.equal(predict.at(-1).data.target[0], Number(value));
    predict.at(-1).resolve(response(pose())); await flush(); await flush();
  }
  for (const value of ['', 'NaN', 'Infinity', '5.001', '-5.001']) {
    const before = predict.length;
    get('x').value = value;
    get('solve').onclick(); await flush();
    assert.equal(predict.length, before);
    assert.match(get('status').textContent, /finite XYZ/);
  }
  const beforeDrag = predict.length;
  view.move([.401, -.123, .456]); view.release(); await flush();
  assert.equal(predict.length, beforeDrag + 1);
  assert.equal(predict.at(-1).data.target[0], .401);
  predict.at(-1).resolve(response(pose())); await flush(); await flush();
  return 'Frontend controls, replay, coordinate validation, and dragged-target prediction passed.';
}

if (typeof process !== 'undefined' && process.argv[1] && fileURLToPath(import.meta.url) === process.argv[1]) console.log(await run());
