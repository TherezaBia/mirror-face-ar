import * as THREE from 'three';
import { TRIANGLES } from '../data/triangles.js';
import { State } from '../config.js';

export class SceneManager {
  constructor(canvas, video) {
    this.canvas = canvas;
    this.video = video;

    this.renderer = null;
    this.scene = null;
    this.camera = null;
    this.faceMesh = null;
    this.bgMesh = null;
    this.videoTexture = null;

    this.faceSmoothMaterial = null;
    this.wireframeMaterial = null;

    this.positions = new Float32Array(468 * 3);
    this.uvs = new Float32Array(468 * 2);
    this.alphas = new Float32Array(468);

    this.init();
  }

  init() {
    this.renderer = new THREE.WebGLRenderer({
      canvas: this.canvas,
      alpha: true,
      antialias: true,
      powerPreference: 'high-performance',
    });
    this.renderer.setSize(window.innerWidth, window.innerHeight);
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));

    this.scene = new THREE.Scene();

    const aspect = window.innerWidth / window.innerHeight;
    this.camera = new THREE.OrthographicCamera(-aspect, aspect, 1, -1, 0.01, 100);
    this.camera.position.z = 10;

    // 1. Textura do Vídeo da Câmera
    this.videoTexture = new THREE.VideoTexture(this.video);
    this.videoTexture.colorSpace = THREE.SRGBColorSpace;
    this.videoTexture.minFilter = THREE.LinearFilter;
    this.videoTexture.magFilter = THREE.LinearFilter;
    this.videoTexture.generateMipmaps = false;

    // 2. Fundo (Vídeo da Câmera em Fullscreen)
    const bgGeometry = new THREE.PlaneGeometry(2 * aspect, 2);
    const bgMaterial = new THREE.MeshBasicMaterial({
      map: this.videoTexture,
      depthWrite: false,
    });
    this.bgMesh = new THREE.Mesh(bgGeometry, bgMaterial);
    this.bgMesh.position.z = -5;
    this.scene.add(this.bgMesh);

    // 3. Geometria da Malha Facial 3D
    const faceGeometry = new THREE.BufferGeometry();
    const indices = [];
    for (const tri of TRIANGLES) {
      indices.push(tri[0], tri[1], tri[2]);
    }

    faceGeometry.setIndex(indices);
    faceGeometry.setAttribute('position', new THREE.BufferAttribute(this.positions, 3));
    faceGeometry.setAttribute('uv', new THREE.BufferAttribute(this.uvs, 2));
    faceGeometry.setAttribute('aAlpha', new THREE.BufferAttribute(this.alphas, 1));

    // 4. Materiais da Malha 3D (Shader com Suavização Geodésica de Bordas)
    this.faceSmoothMaterial = new THREE.ShaderMaterial({
      uniforms: {
        map: { value: this.videoTexture },
        uGlobalOpacity: { value: State.view.meshOpacity },
      },
      vertexShader: `
        attribute float aAlpha;
        varying vec2 vUv;
        varying float vAlpha;
        void main() {
          vUv = uv;
          vAlpha = aAlpha;
          gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
        }
      `,
      fragmentShader: `
        uniform sampler2D map;
        uniform float uGlobalOpacity;
        varying vec2 vUv;
        varying float vAlpha;
        void main() {
          vec4 texColor = texture2D(map, vUv);
          gl_FragColor = vec4(texColor.rgb, texColor.a * vAlpha * uGlobalOpacity);
        }
      `,
      transparent: true,
      side: THREE.DoubleSide,
      depthWrite: false,
    });

    this.wireframeMaterial = new THREE.MeshBasicMaterial({
      color: new THREE.Color(State.view.wireframeColor),
      wireframe: true,
      side: THREE.DoubleSide,
      transparent: true,
      opacity: 0.85,
    });

    this.faceMesh = new THREE.Mesh(faceGeometry, this.faceSmoothMaterial);
    this.faceMesh.position.z = 0;
    this.scene.add(this.faceMesh);

    // Modo espelho horizontal da câmera
    this.scene.scale.set(-1, 1, 1);

    window.addEventListener('resize', () => this.onResize());
  }

  setVisualMode(mode) {
    if (mode === 'mirrored_face') {
      this.faceMesh.material = this.faceSmoothMaterial;
    } else {
      this.faceMesh.material = this.wireframeMaterial;
    }
  }

  setOpacity(val) {
    this.faceSmoothMaterial.uniforms.uGlobalOpacity.value = val;
    this.wireframeMaterial.opacity = val;
  }

  onResize() {
    const aspect = window.innerWidth / window.innerHeight;
    this.camera.left = -aspect;
    this.camera.right = aspect;
    this.camera.top = 1;
    this.camera.bottom = -1;
    this.camera.updateProjectionMatrix();

    this.renderer.setSize(window.innerWidth, window.innerHeight);

    if (this.bgMesh) {
      this.bgMesh.geometry.dispose();
      this.bgMesh.geometry = new THREE.PlaneGeometry(2 * aspect, 2);
    }
  }

  render() {
    this.renderer.render(this.scene, this.camera);
  }
}
