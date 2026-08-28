import { CANONICAL_UVS } from './canonical_uvs.js';
import { CENTRAL_LANDMARKS } from './pairs.js';
import { TRIANGLES } from './triangles.js';

const LANDMARK_COUNT = 468;

// Distancia geodesica, em unidades do atlas, entre cada vertice e a linha central.
function computeCenterDistances() {
  const adjacency = Array.from({ length: LANDMARK_COUNT }, () => []);

  const addEdge = (a, b) => {
    const dx = CANONICAL_UVS[a * 2] - CANONICAL_UVS[b * 2];
    const dy = CANONICAL_UVS[a * 2 + 1] - CANONICAL_UVS[b * 2 + 1];
    adjacency[a].push([b, Math.hypot(dx, dy)]);
  };

  for (const [a, b, c] of TRIANGLES) {
    addEdge(a, b); addEdge(b, a);
    addEdge(b, c); addEdge(c, b);
    addEdge(c, a); addEdge(a, c);
  }

  const distances = new Float32Array(LANDMARK_COUNT).fill(Infinity);
  const visited = new Uint8Array(LANDMARK_COUNT);
  for (const index of CENTRAL_LANDMARKS) distances[index] = 0;

  for (let step = 0; step < LANDMARK_COUNT; step++) {
    let current = -1;
    let currentDistance = Infinity;

    for (let i = 0; i < LANDMARK_COUNT; i++) {
      if (!visited[i] && distances[i] < currentDistance) {
        current = i;
        currentDistance = distances[i];
      }
    }

    if (current === -1) break;
    visited[current] = 1;

    for (const [neighbor, edgeLength] of adjacency[current]) {
      const candidate = currentDistance + edgeLength;
      if (candidate < distances[neighbor]) distances[neighbor] = candidate;
    }
  }

  return distances;
}

export const CANONICAL_CENTER_DISTANCES = computeCenterDistances();
