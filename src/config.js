/**
 * Configurações e Estado Global da Aplicação
 */
export const AppConfig = {
  camera: {
    width: 1280,
    height: 720,
    facingMode: 'user',
  },
  mediapipe: {
    wasmPath: 'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm',
    modelPath: 'https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task',
  },
};

export const State = {
  // Configuração Clínica de Espelhamento
  mirror: {
    mode: 'left_healthy',        // 'left_healthy' | 'right_healthy' | 'disabled'
    strength: 1.0,               // 0.0 a 1.0
    featherWidth: 0.04,          // Largura da transição central
  },

  // Visualização e Renderização
  view: {
    mode: 'mirrored_face',       // 'mirrored_face' | 'wireframe'
    meshOpacity: 1.0,
    wireframeColor: '#38bdf8',
    zScale: 1.8,
  },

  // Calibração Anatômica Milimétrica (MEPP)
  calibration: {
    offsetX: 0.0,
    offsetY: 0.0,
    scaleX: 1.0,
  },

  // Status de Execução
  status: {
    modelReady: false,
    cameraReady: false,
    fps: 0,
  },
};
