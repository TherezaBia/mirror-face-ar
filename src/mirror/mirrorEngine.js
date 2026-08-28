import { CENTRAL_LANDMARKS, SYMMETRY_PAIRS } from '../data/pairs.js';
import { TRIANGLES } from '../data/triangles.js';
import { CANONICAL_UVS } from '../data/canonical_uvs.js';
import { CANONICAL_CENTER_DISTANCES } from '../data/canonical_metrics.js';
import { State } from '../config.js';

// Vertices do contorno externo do MediaPipe.
const CONTOUR_LANDMARKS = [
  10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377,
  152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109,
];

const ALPHA_WEIGHTS = new Float32Array(468);

(function precomputeAlphaWeights() {
  const adjacency = Array.from({ length: 468 }, () => []);
  for (const [a, b, c] of TRIANGLES) {
    if (a < 468 && b < 468 && c < 468) {
      adjacency[a].push(b, c);
      adjacency[b].push(a, c);
      adjacency[c].push(a, b);
    }
  }

  const distances = new Int32Array(468).fill(-1);
  const queue = [];
  for (const index of CONTOUR_LANDMARKS) {
    distances[index] = 0;
    queue.push(index);
  }

  for (let cursor = 0; cursor < queue.length; cursor++) {
    const current = queue[cursor];
    for (const neighbor of adjacency[current]) {
      if (distances[neighbor] === -1) {
        distances[neighbor] = distances[current] + 1;
        queue.push(neighbor);
      }
    }
  }

  for (let i = 0; i < 468; i++) {
    ALPHA_WEIGHTS[i] = Math.min(1, Math.max(0, distances[i] / 3));
  }
})();

export class MirrorEngine {
  /**
   * Atualiza somente a geometria em pose e as coordenadas de captura do unwrap.
   * As UVs finais permanecem canonicas e invariantes a pose.
   */
  updateMesh(landmarks, faceMesh) {
    const geometry = faceMesh.geometry;
    const positionAttribute = geometry.attributes.position;
    const alphaAttribute = geometry.attributes.aAlpha;
    const screenUvAttribute = geometry.attributes.aScreenUv;

    const aspect = window.innerWidth / window.innerHeight;
    const scaleXFactor = 2 * aspect;
    const scaleYFactor = 2;

    // MediaPipe expressa Z aproximadamente na mesma escala de X. X e Z sao
    // trazidos para a mesma metrica visual de Y antes de estimar o plano.
    const toMetric = (landmark) => ({
      x: landmark.x * aspect,
      y: landmark.y,
      z: (landmark.z || 0) * aspect,
    });

    const leftCheek = toMetric(landmarks[234]);
    const rightCheek = toMetric(landmarks[454]);
    const lateral = {
      x: rightCheek.x - leftCheek.x,
      y: rightCheek.y - leftCheek.y,
      z: rightCheek.z - leftCheek.z,
    };
    const lateralLength = Math.hypot(lateral.x, lateral.y, lateral.z) || 1e-6;
    lateral.x /= lateralLength;
    lateral.y /= lateralLength;
    lateral.z /= lateralLength;

    const sagittalCenter = { x: 0, y: 0, z: 0 };
    for (const index of CENTRAL_LANDMARKS) {
      const point = toMetric(landmarks[index]);
      sagittalCenter.x += point.x;
      sagittalCenter.y += point.y;
      sagittalCenter.z += point.z;
    }
    sagittalCenter.x /= CENTRAL_LANDMARKS.size;
    sagittalCenter.y /= CENTRAL_LANDMARKS.size;
    sagittalCenter.z /= CENTRAL_LANDMARKS.size;

    const isLeftHealthy = State.mirror.mode === 'left_healthy';
    const isRightHealthy = State.mirror.mode === 'right_healthy';
    const isMirroring = State.mirror.mode !== 'disabled' && State.mirror.strength > 0;

    for (let i = 0; i < 468; i++) {
      const landmark = landmarks[i];
      const baseAlpha = ALPHA_WEIGHTS[i];
      let targetX = landmark.x;
      let targetY = landmark.y;
      let targetZ = landmark.z || 0;

      if (isMirroring && !CENTRAL_LANDMARKS.has(i)) {
        // O lado anatomico vem do atlas fixo. Rotacao e escorco nao mudam esta decisao.
        const canonicalU = CANONICAL_UVS[i * 2];
        const shouldMirror = (isLeftHealthy && canonicalU < 0.5)
          || (isRightHealthy && canonicalU > 0.5);

        if (shouldMirror) {
          const pairIndex = SYMMETRY_PAIRS[i] ?? i;
          const healthyLandmark = landmarks[pairIndex];
          const healthyPoint = toMetric(healthyLandmark);
          const planeDistance = (
            (healthyPoint.x - sagittalCenter.x) * lateral.x
            + (healthyPoint.y - sagittalCenter.y) * lateral.y
            + (healthyPoint.z - sagittalCenter.z) * lateral.z
          );
          const reflectionScale = 1 + State.calibration.scaleX;
          const mirroredPoint = {
            x: healthyPoint.x - reflectionScale * planeDistance * lateral.x,
            y: healthyPoint.y - reflectionScale * planeDistance * lateral.y,
            z: healthyPoint.z - reflectionScale * planeDistance * lateral.z,
          };

          const mirroredX = mirroredPoint.x / aspect + State.calibration.offsetX;
          const mirroredY = mirroredPoint.y + State.calibration.offsetY;
          const mirroredZ = mirroredPoint.z / aspect;
          const featherWidth = Math.max(State.mirror.featherWidth, 0.0001);
          const normalizedDistance = Math.min(
            1,
            Math.max(0, CANONICAL_CENTER_DISTANCES[i] / featherWidth),
          );
          const centerBlend = normalizedDistance * normalizedDistance * (3 - 2 * normalizedDistance);
          const effectiveStrength = State.mirror.strength * baseAlpha * centerBlend;

          targetX = landmark.x * (1 - effectiveStrength) + mirroredX * effectiveStrength;
          targetY = landmark.y * (1 - effectiveStrength) + mirroredY * effectiveStrength;
          targetZ = (landmark.z || 0) * (1 - effectiveStrength) + mirroredZ * effectiveStrength;
        }
      }

      positionAttribute.setXYZ(
        i,
        (targetX - 0.5) * scaleXFactor,
        (0.5 - targetY) * scaleYFactor,
        -targetZ * State.view.zScale,
      );
      alphaAttribute.setX(i, baseAlpha);

      // O passe 1 sempre captura o rosto real, antes de qualquer espelhamento.
      screenUvAttribute.setXY(i, landmark.x, 1 - landmark.y);
    }

    positionAttribute.needsUpdate = true;
    alphaAttribute.needsUpdate = true;
    screenUvAttribute.needsUpdate = true;
    faceMesh.visible = true;
  }
}
