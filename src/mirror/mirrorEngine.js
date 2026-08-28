import { CENTRAL_LANDMARKS, SYMMETRY_PAIRS } from '../data/pairs.js';
import { State } from '../config.js';

export class MirrorEngine {
  constructor() {}

  /**
   * Atualiza a malha com espelhamento simétrico sincronizado (3D + UV)
   */
  updateMesh(landmarks, faceMesh) {
    const geom = faceMesh.geometry;
    const posAttr = geom.attributes.position;
    const uvAttr = geom.attributes.uv;

    const aspect = window.innerWidth / window.innerHeight;
    const scaleXFactor = 2.0 * aspect;
    const scaleYFactor = 2.0;

    // Eixo sagital central da face (nariz médio)
    const centroX = landmarks[168].x;
    const centroY = landmarks[168].y;

    const isLeftHealthy = State.mirror.mode === 'left_healthy';
    const isRightHealthy = State.mirror.mode === 'right_healthy';
    const isMirroring = State.mirror.mode !== 'disabled' && State.mirror.strength > 0.0;

    for (let i = 0; i < 468; i++) {
      const lm = landmarks[i];

      // Valores reais padrão
      let targetX = lm.x;
      let targetY = lm.y;
      let targetZ = lm.z;

      let targetUvX = lm.x;
      let targetUvY = lm.y;

      if (isMirroring) {
        const isCentral = CENTRAL_LANDMARKS.has(i);
        const isPointOnLeft = lm.x < centroX;
        const isPointOnRight = lm.x > centroX;

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

          // 1. ESPELHAMENTO GEOMÉTRICO 3D (Simetria de Movimento)
          if (State.mirror.synchronize3D) {
            const deltaX = centroX - healthyLm.x;
            const mirrored3DX = centroX + (deltaX * State.calibration.scaleX) + State.calibration.offsetX;
            const mirrored3DY = healthyLm.y + State.calibration.offsetY;
            const mirrored3DZ = healthyLm.z;

            const s = State.mirror.strength;
            targetX = lm.x * (1 - s) + mirrored3DX * s;
            targetY = lm.y * (1 - s) + mirrored3DY * s;
            targetZ = lm.z * (1 - s) + mirrored3DZ * s;
          }

          // 2. ESPELHAMENTO DE TEXTURA UV
          const mirroredUvX = healthyLm.x;
          const mirroredUvY = healthyLm.y;

          const s = State.mirror.strength;
          targetUvX = lm.x * (1 - s) + mirroredUvX * s;
          targetUvY = lm.y * (1 - s) + mirroredUvY * s;
        }
      }

      // Projeção final para a GPU
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
