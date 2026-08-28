import { CENTRAL_LANDMARKS, SYMMETRY_PAIRS } from '../data/pairs.js';
import { TRIANGLES } from '../data/triangles.js';
import { State } from '../config.js';

// Vértices do contorno externo do MediaPipe (borda externa do rosto)
const CONTOUR_LANDMARKS = [
  10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377,
  152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109
];

// Vértices da borda imediata dos olhos e lábios para suavização
const APERTURE_RIM_LANDMARKS = new Set([
  33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246,
  263, 249, 390, 373, 374, 380, 381, 382, 362, 398, 384, 385, 386, 387, 388, 466,
  78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95
]);

// Pré-cálculo da distância geodésica na malha a partir do contorno para ancoragem sem degraus
const ALPHA_WEIGHTS = new Float32Array(468);

(function precomputeAlphaWeights() {
  const adj = Array.from({ length: 468 }, () => []);
  for (const [a, b, c] of TRIANGLES) {
    if (a < 468 && b < 468 && c < 468) {
      adj[a].push(b, c);
      adj[b].push(a, c);
      adj[c].push(a, b);
    }
  }

  const dist = new Int32Array(468).fill(-1);
  const queue = [];

  for (const idx of CONTOUR_LANDMARKS) {
    dist[idx] = 0;
    queue.push(idx);
  }

  while (queue.length > 0) {
    const u = queue.shift();
    for (const v of adj[u]) {
      if (dist[v] === -1) {
        dist[v] = dist[u] + 1;
        queue.push(v);
      }
    }
  }

  for (let i = 0; i < 468; i++) {
    let w = Math.min(1.0, Math.max(0.0, dist[i] / 3.0));
    if (APERTURE_RIM_LANDMARKS.has(i)) {
      w = Math.min(w, 0.88);
    }
    ALPHA_WEIGHTS[i] = w;
  }
})();

export class MirrorEngine {
  constructor() {}

  /**
   * Atualiza a malha facial com espelhamento simétrico sagital de alta fidelidade.
   */
  updateMesh(landmarks, faceMesh) {
    const geom = faceMesh.geometry;
    const posAttr = geom.attributes.position;
    const uvAttr = geom.attributes.uv;
    const alphaAttr = geom.attributes.aAlpha;

    const aspect = window.innerWidth / window.innerHeight;
    const scaleXFactor = 2.0 * aspect;
    const scaleYFactor = 2.0;

    // 1. Eixo Sagital Direto da Cabeça (Vetor 2D/3D Testa 10 -> Queixo 152)
    const pTesta = landmarks[10];
    const pQueixo = landmarks[152];

    const dx = pQueixo.x - pTesta.x;
    const dy = pQueixo.y - pTesta.y;
    const lenSq = dx * dx + dy * dy || 1e-6;
    const len = Math.sqrt(lenSq);

    const isLeftHealthy = State.mirror.mode === 'left_healthy';
    const isRightHealthy = State.mirror.mode === 'right_healthy';
    const isMirroring = State.mirror.mode !== 'disabled' && State.mirror.strength > 0.0;
    const strength = State.mirror.strength;

    for (let i = 0; i < 468; i++) {
      const lm = landmarks[i];
      const baseAlpha = ALPHA_WEIGHTS[i];

      let targetX = lm.x;
      let targetY = lm.y;
      let targetZ = lm.z || 0;
      let targetUvX = lm.x;
      let targetUvY = lm.y;

      if (isMirroring) {
        const isCentral = CENTRAL_LANDMARKS.has(i);

        if (!isCentral) {
          // Distância perpendicular com sinal até o eixo testa -> queixo
          const cross = dx * (lm.y - pTesta.y) - dy * (lm.x - pTesta.x);
          const signedDist = cross / len;

          // Modo espelho da câmera:
          //   signedDist < 0 -> Lado direito do paciente (lado esquerdo da tela)
          //   signedDist > 0 -> Lado esquerdo do paciente (lado direito da tela)
          let shouldMirror = false;
          if (isLeftHealthy && signedDist < -0.001) {
            shouldMirror = true;
          } else if (isRightHealthy && signedDist > 0.001) {
            shouldMirror = true;
          }

          if (shouldMirror) {
            const pairIdx = SYMMETRY_PAIRS[i] !== undefined ? SYMMETRY_PAIRS[i] : i;
            const healthyLm = landmarks[pairIdx];

            // 1. Projeção ortogonal do ponto saudável sobre o eixo sagital
            const tH = ((healthyLm.x - pTesta.x) * dx + (healthyLm.y - pTesta.y) * dy) / lenSq;
            const cHX = pTesta.x + tH * dx;
            const cHY = pTesta.y + tH * dy;

            // 2. Vetor perpendicular do ponto saudável ao eixo central
            const wX = healthyLm.x - cHX;
            const wY = healthyLm.y - cHY;

            // Reflexão pura perpendicular ao eixo da cabeça (sem torção vertical)
            const mirroredX = cHX - (wX * State.calibration.scaleX) + State.calibration.offsetX;
            const mirroredY = cHY - (wY * State.calibration.scaleX) + State.calibration.offsetY;
            const mirroredZ = healthyLm.z || 0;

            // Ancoragem perimétrica suave para fusão com a imagem real
            const effectiveStrength = strength * baseAlpha;

            targetX = lm.x * (1 - effectiveStrength) + mirroredX * effectiveStrength;
            targetY = lm.y * (1 - effectiveStrength) + mirroredY * effectiveStrength;
            targetZ = (lm.z || 0) * (1 - effectiveStrength) + mirroredZ * effectiveStrength;

            // Textura mapeada diretamente do tecido saudável
            targetUvX = healthyLm.x;
            targetUvY = healthyLm.y;
          }
        }
      }

      // Projeção 3D para GPU
      const x3d = (targetX - 0.5) * scaleXFactor;
      const y3d = (0.5 - targetY) * scaleYFactor;
      const z3d = -targetZ * State.view.zScale;

      posAttr.setXYZ(i, x3d, y3d, z3d);
      uvAttr.setXY(i, targetUvX, 1.0 - targetUvY);
      alphaAttr.setX(i, baseAlpha);
    }

    posAttr.needsUpdate = true;
    uvAttr.needsUpdate = true;
    alphaAttr.needsUpdate = true;
    geom.computeVertexNormals();
  }
}
