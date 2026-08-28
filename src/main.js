import { State } from './config.js';
import { FaceTracker } from './tracker/faceTracker.js';
import { SceneManager } from './graphics/sceneManager.js';
import { MirrorEngine } from './mirror/mirrorEngine.js';
import { GuiController } from './ui/guiController.js';

// Elementos da Interface
const videoElement = document.getElementById('webcam');
const canvasElement = document.getElementById('webgl-canvas');
const fpsText = document.getElementById('fps-counter');
const statusText = document.getElementById('status-indicator');
const promptOverlay = document.getElementById('camera-prompt');
const btnStartCamera = document.getElementById('btn-start-camera');

// Instancia os Módulos Especializados
const sceneManager = new SceneManager(canvasElement, videoElement);
const faceTracker = new FaceTracker(videoElement);
const mirrorEngine = new MirrorEngine();
const guiController = new GuiController(sceneManager);

let frameCount = 0;
let lastFpsTime = performance.now();

// Loop Principal de Renderização (60 FPS)
function mainLoop() {
  requestAnimationFrame(mainLoop);

  // 1. Contador de FPS
  frameCount++;
  const now = performance.now();
  if (now - lastFpsTime >= 1000) {
    State.status.fps = Math.round((frameCount * 1000) / (now - lastFpsTime));
    fpsText.innerText = `FPS: ${State.status.fps}`;
    frameCount = 0;
    lastFpsTime = now;
  }

  // 2. Rastreamento e Atualização da Malha Facial
  if (State.status.modelReady && State.status.cameraReady) {
    const landmarks = faceTracker.detectFrame(now);

    if (landmarks) {
      mirrorEngine.updateMesh(landmarks, sceneManager.faceMesh);
    }
  }

  // 3. Renderização WebGL
  sceneManager.render();
}

// Inicialização da Câmera
async function onStartCameraClicked() {
  try {
    await faceTracker.startCamera();
    State.status.cameraReady = true;
    promptOverlay.style.display = 'none';
    statusText.innerText = 'Espelho Facial Ativo';
    statusText.className = 'status-ready';

    requestAnimationFrame(mainLoop);
  } catch (error) {
    console.error('Erro de câmera:', error);
    alert('Permissão de câmera necessária para a Terapia de Espelho.');
  }
}

// Inicialização Geral
async function bootstrap() {
  btnStartCamera.addEventListener('click', onStartCameraClicked);

  await faceTracker.init((msg) => {
    statusText.innerText = msg;
  });

  State.status.modelReady = true;
  statusText.innerText = 'IA Pronta! Clique para iniciar.';
  statusText.className = 'status-ready';
}

bootstrap();
