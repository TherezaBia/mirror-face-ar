import { CANONICAL_UVS } from '../src/data/canonical_uvs.js';
import { TRIANGLES } from '../src/data/triangles.js';
import { SYMMETRY_PAIRS } from '../src/data/pairs.js';
import { CANONICAL_CENTER_DISTANCES } from '../src/data/canonical_metrics.js';

const LANDMARK_COUNT = 468;

function assert(condition, message) {
  if (!condition) throw new Error(message);
}

assert(CANONICAL_UVS.length === LANDMARK_COUNT * 2, 'O atlas deve conter 468 pares UV.');
assert(CANONICAL_CENTER_DISTANCES.length === LANDMARK_COUNT, 'As distancias canonicas devem cobrir 468 vertices.');

for (let i = 0; i < CANONICAL_UVS.length; i++) {
  const coordinate = CANONICAL_UVS[i];
  assert(Number.isFinite(coordinate) && coordinate >= 0 && coordinate <= 1, `UV invalida no indice ${i}.`);
}

for (const distance of CANONICAL_CENTER_DISTANCES) {
  assert(Number.isFinite(distance) && distance >= 0, 'Distancia geodesica invalida.');
}

for (const triangle of TRIANGLES) {
  assert(triangle.length === 3, 'Cada face deve ser triangular.');
  assert(triangle.every((index) => index >= 0 && index < LANDMARK_COUNT), `Triangulo invalido: ${triangle}.`);
}

for (let i = 0; i < LANDMARK_COUNT; i++) {
  const pair = SYMMETRY_PAIRS[i];
  assert(Number.isInteger(pair) && pair >= 0 && pair < LANDMARK_COUNT, `Par ausente para ${i}.`);
  assert(SYMMETRY_PAIRS[pair] === i, `Par nao reciproco: ${i} -> ${pair}.`);

  const mirroredAxisError = Math.abs(CANONICAL_UVS[i * 2] + CANONICAL_UVS[pair * 2] - 1);
  assert(mirroredAxisError < 0.002, `Par ${i}/${pair} nao e simetrico no atlas.`);
}

console.log(`Atlas facial validado: ${LANDMARK_COUNT} UVs, ${TRIANGLES.length} triangulos e ${LANDMARK_COUNT} pares.`);
