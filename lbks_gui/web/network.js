const element = id => document.getElementById(id);
const number = value => Number(value).toPrecision(5);
let stages = [], selected = 0, timer;

function node(tag, text, className) {
  const item = document.createElement(tag);
  if (text !== undefined) item.textContent = text;
  if (className) item.className = className;
  return item;
}

const explanations = {
  input: 'Cartesian XYZ input in metres.',
  Linear: 'Weighted sum: y[j] = Σ weight[j,i] × input[i] + bias[j].',
  ReLU: 'ReLU keeps positive values and replaces negative values with zero.',
  Tanh: 'Tanh maps each value into the range −1 to +1.',
  GELU: 'GELU smoothly gates each input using the normal distribution.',
  output: 'Raw predicted joint angles in radians, before joint limits.',
  limits: 'URDF limits clamp the predicted joints. This is not a learned layer.',
  fk: 'Forward kinematics maps applied joint angles to achieved XYZ. This is not a learned layer.',
};

export function stopReplay() {
  clearInterval(timer);
  timer = undefined;
  element('replay').textContent = '▶ Replay prediction';
}

function selectStage(index, scroll = true) {
  if (!stages.length) return;
  selected = Math.max(0, Math.min(index, stages.length - 1));
  const cards = element('network-graph').children;
  [...cards].forEach((card, i) => card.classList.toggle('selected', i === selected));
  if (scroll) element('network-graph').scrollTo({ left: Math.max(0, cards[selected].offsetLeft - cards[0].offsetLeft - 12), behavior: 'smooth' });
  element('stage').value = selected;
  element('stage-count').textContent = `${selected + 1} / ${stages.length}`;
  const stage = stages[selected];
  element('neuron-detail').replaceChildren(
    node('h3', stage.name), node('p', explanations[stage.kind] ?? stage.kind),
    node('pre', `Minimum  ${number(stage.min)}\nMaximum  ${number(stage.max)}\nMean     ${number(stage.mean)}\nZero     ${stage.zeros} / ${stage.count}\n\nClick a value to inspect it.`));
}

function inspect(stageIndex, index) {
  stopReplay();
  selectStage(stageIndex, false);
  const stage = stages[stageIndex];
  const label = stage.labels[index] ?? `Neuron ${index}`;
  let calculation = `${label} = ${number(stage.values[index])}\n`;
  const detail = stage.neurons?.[index];
  if (detail) {
    calculation += '\nStrongest weighted inputs:\n';
    for (const term of detail.terms) calculation +=
      `x[${term.input}] ${number(term.value)}\n× ${number(term.weight)} = ${number(term.product)}\n\n`;
    calculation += `Other inputs: ${number(detail.other_sum)}\nBias: ${number(detail.bias)}\n\nTheir sum gives this activation.`;
  } else if (stageIndex > 0 && stage.kind !== 'fk') {
    const previous = stages[stageIndex - 1].values[index];
    if (previous !== undefined) calculation += `Previous value = ${number(previous)}`;
  }
  element('neuron-detail').replaceChildren(node('h3', stage.name),
    node('p', explanations[stage.kind] ?? ''), node('pre', calculation));
}

export function showTrace(trace = {}) {
  stopReplay();
  stages = trace.stages ?? [];
  element('network-note').textContent = trace.note ?? 'Load a trained checkpoint to inspect real activations.';
  element('parameter-count').textContent = trace.parameters ? `${trace.parameters.toLocaleString()} parameters` : '';
  for (const id of ['replay', 'stage', 'previous-stage', 'next-stage']) element(id).disabled = !stages.length;
  element('stage').max = Math.max(0, stages.length - 1);
  element('stage-count').textContent = '0 / 0';
  const graph = element('network-graph');
  graph.replaceChildren();
  element('neuron-detail').replaceChildren(node('h3', 'Inspect a neuron'), node('p', 'Click an activation for its calculation.'));
  if (!stages.length) {
    const empty = node('div', undefined, 'empty-network');
    empty.append(node('span', 'XYZ → network → joints → robot'), node('p', trace.note ?? 'Load a trained checkpoint to begin.'));
    graph.append(empty);
    return;
  }
  stages.forEach((stage, stageIndex) => {
    const card = node('div', undefined, 'layer');
    card.append(node('h3', stage.name), node('p', `${stage.count} values · ${stage.zeros} zero`, 'stats'));
    const values = node('div', undefined, `values${stage.values.length <= 8 ? ' small' : ''}`);
    const maximum = Math.max(Math.abs(stage.min), Math.abs(stage.max));
    stage.values.forEach((value, index) => {
      const label = stage.labels[index] ?? `Neuron ${index}`;
      const button = node('button', stage.values.length <= 8 ? `${stage.count === 3 ? label : index}: ${number(value)}` : '', 'value');
      button.style.background = value === 0 ? '#bac6bf' :
        `hsl(${value < 0 ? 202 : 39} 55% ${25 + 35 * Math.abs(value) / maximum}%)`;
      button.title = `${label}: ${number(value)}`;
      button.setAttribute('aria-label', `${stage.name}, ${label}: ${number(value)}`);
      button.onclick = () => inspect(stageIndex, index);
      values.append(button);
    });
    card.append(values, node('p', stage.values.length < stage.count ?
      `First ${stage.values.length} of ${stage.count} values` : `min ${number(stage.min)} · max ${number(stage.max)}`, 'layer-footer'));
    graph.append(card);
  });
  selectStage(0);
}

element('replay').onclick = () => {
  if (timer) { stopReplay(); return; }
  selectStage(0);
  element('replay').textContent = 'Ⅱ Pause replay';
  timer = setInterval(() => {
    if (selected >= stages.length - 1) stopReplay();
    else selectStage(selected + 1);
  }, 650);
};
element('stage').oninput = event => { stopReplay(); selectStage(Number(event.target.value)); };
element('previous-stage').onclick = () => { stopReplay(); selectStage(selected - 1); };
element('next-stage').onclick = () => { stopReplay(); selectStage(selected + 1); };
