import { FilesetResolver, FaceLandmarker } from '@mediapipe/tasks-vision';
import { AppConfig } from '../config.js';

export class FaceTracker {
  constructor(videoElement) {
    this.video = videoElement;
    this.landmarker = null;
    this.isReady = false;
    this.lastVideoTime = -1;
  }

  async init(onStatusUpdate) {
    if (onStatusUpdate) onStatusUpdate('Baixando modelo MediaPipe...');

    const vision = await FilesetResolver.forVisionTasks(AppConfig.mediapipe.wasmPath);

    this.landmarker = await FaceLandmarker.createFromOptions(vision, {
      baseOptions: {
        modelAssetPath: AppConfig.mediapipe.modelPath,
        delegate: 'GPU',
      },
      runningMode: 'VIDEO',
      numFaces: 1,
      outputFaceBlendshapes: true,
    });

    this.isReady = true;
    if (onStatusUpdate) onStatusUpdate('IA Pronta! Conectando câmera...');
  }

  async startCamera() {
    const stream = await navigator.mediaDevices.getUserMedia({
      video: {
        width: { ideal: AppConfig.camera.width },
        height: { ideal: AppConfig.camera.height },
        facingMode: AppConfig.camera.facingMode,
      },
      audio: false,
    });

    this.video.srcObject = stream;
    await this.video.play();
    return stream;
  }

  detectFrame(now) {
    if (!this.isReady || !this.video || this.video.currentTime === this.lastVideoTime) {
      return null;
    }

    this.lastVideoTime = this.video.currentTime;
    const results = this.landmarker.detectForVideo(this.video, now);

    if (results.faceLandmarks && results.faceLandmarks.length > 0) {
      return results.faceLandmarks[0];
    }

    return null;
  }
}
