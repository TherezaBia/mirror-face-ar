import { CENTRAL_LANDMARKS, SYMMETRY_PAIRS } from '../data/pairs.js';
import { State } from '../config.js';

export class MirrorEngine {
  constructor() {}

  /**
   * Função utilitária de interpolação suave (Hermite Smoothstep)
   */
  smoothstep(min, max, value) {
    const x = Math.max(0, Math.min(1, (value - min) / (max - min)));
    return x * x * (3 - 2 * x);
  }

  /**
   * Atualiza a malha com espelhamento simétrico sincronizado (3D + UV + Fusão Central)
   */
  updateMesh(landmarks, faceMesh) {
    const geom = faceMesh.geometry;
    const posAttr = geom.attributes.position;
    const uvAttr = geom.attributes.uv;

    const aspect = window.innerWidth / window.innerHeight;
    const scaleXFactor = 2.0 * aspect;
    const scaleYFactor = 2.0;

    // 1. Eixo Sagital Adaptativo da Cabeça (Reta 3D da Testa ao Queixo)
    const pTesta = landmarks[10];
    const pQueixo = landmarks[152];
    const dyTotal = pQueixo.y - pTesta.y || 1e-5;

    const isLeftHealthy = State.mirror.mode === 'left_healthy';
    const isRightHealthy = State.mirror.mode === 'right_healthy';
    const isMirroring = State.mirror.mode !== 'disabled' && State.mirror.strength > 0.0;

    for (let i = 0; i < 468; i++) {
      const lm = landmarks[i];

      // Posição central sagital na altura Y do ponto atual
      const t = (lm.y - pTesta.y) / dyTotal;
      const centroX = pTesta.x + t * (pQueixo.x - pTesta.x);

      let targetX = lm.x;
      let targetY = lm.y;
      let targetZ = lm.z;

      let targetUvX = lm.x;
      let targetUvY = lm.y;

      if (isMirroring) {
        const isCentral = CENTRAL_LANDMARKS.has(i);
        const distFromCenter = lm.x - centroX;

        // distFromCenter < 0 -> Lado Esquerdo da face
        // distFromCenter > 0 -> Lado Direito da face
        const isPointOnLeft = distFromCenter < -0.005;
        const isPointOnRight = distFromCenter > 0.005;

        let shouldMirror = false;

        if (isLeftHealthy && isPointOnRight && !isCentral) {
          // Lado Esquerdo Saudável -> Espelha sobre o Lado Direito
          shouldMirror = true;
        } else if (isRightHealthy && isPointOnLeft && !isCentral) {
          // Lado Direito Saudável -> Espelha sobre o Lado Esquerdo
          shouldMirror = true;
        }

        if (shouldMirror) {
          const pairIdx = SYMMETRY_PAIRS[i] !== undefined ? SYMMETRY_PAIRS[i] : i;
          const healthyLm = landmarks[pairIdx];

          // Centro sagital na altura Y do ponto saudável
          const tHealthy = (healthyLm.y - pTesta.y) / dyTotal;
          const centroXHealthy = pTesta.x + tHealthy * (pQueixo.x - pTesta.x);

          // 1. ESPELHAMENTO GEOMÉTRICO 3D (Sincronia de Relevo e Movimento Muscular)
          if (State.mirror.synchronize3D) {
            const deltaX = centroXHealthy - healthyLm.x;
            const mirrored3DX = centroX + (deltaX * State.calibration.scaleX) + State.calibration.offsetX;
            const mirrored3DY = healthyLm.y + State.calibration.offsetY;
            const mirrored3DZ = healthyLm.z;

            // Fator de suavização (Feathering na linha média)
            const blendFactor = this.smoothstep(0.005, 0.035, Math.abs(distFromCenter));
            const s = State.mirror.strength * blendFactor;

            targetX = lm.x * (1 - s) + mirrored3DX * s;
            targetY = lm.y * (1 - s) + mirrored3DY * s;
            targetZ = lm.z * (1 - s) + mirrored3DZ * s;
          }

          // 2. ESPELHAMENTO DE TEXTURA UV (Projeção da Imagem da Pele)
          const mirroredUvX = healthyLm.x;
          const mirroredUvY = healthyLm.y;

          const blendFactorUV = this.smoothstep(0.005, 0.035, Math.abs(distFromCenter));
          const sUV = State.mirror.strength * blendFactorUV;

          targetUvX = lm.x * (1 - sUV) + mirroredUvX * sUV;
          targetUvY = lm.y * (1 - sUV) + mirroredUvY * sUV;
        }
      }

      // Projeção 3D para o Three.js / GPU
      const x3d = (targetX - 0.5) * scaleXFactor;
      const y3d = (0.5 - targetY) * scaleYFactor;
      const z3d = -targetZ * State.view.zScale;

      posAttr.setXYZ(i, x3d, y3d, z3d);
      uvAttr.setXY(i, targetUvX, 1.0 - targetUvY);
    }

    posAttr.needsUpdate = true;
    uvAttr.needsUpdate = true;
    geom.computeVertexNormals();
  }
}
