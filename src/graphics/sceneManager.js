import * as THREE from 'three';
import { TRIANGLES } from '../data/triangles.js';
import { CANONICAL_UVS } from '../data/canonical_uvs.js';
import { SYMMETRY_PAIRS } from '../data/pairs.js';
import { CANONICAL_CENTER_DISTANCES } from '../data/canonical_metrics.js';
import { State } from '../config.js';

const LANDMARK_COUNT = 468;

function createIndices() {
  const indices = [];
  for (const [a, b, c] of TRIANGLES) indices.push(a, b, c);
  return indices;
}

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
    this.unwrapScene = null;
    this.unwrapTarget = null;
    this.mirrorScene = null;
    this.mirrorTarget = null;
    this.mirrorAtlasMaterial = null;
    this.atlasCamera = null;

    this.positions = new Float32Array(LANDMARK_COUNT * 3);
    this.screenUvs = new Float32Array(LANDMARK_COUNT * 2);
    this.alphas = new Float32Array(LANDMARK_COUNT);

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

    this.videoTexture = new THREE.VideoTexture(this.video);
    this.videoTexture.colorSpace = THREE.SRGBColorSpace;
    this.videoTexture.minFilter = THREE.LinearFilter;
    this.videoTexture.magFilter = THREE.LinearFilter;
    this.videoTexture.generateMipmaps = false;

    const bgGeometry = new THREE.PlaneGeometry(2 * aspect, 2);
    const bgMaterial = new THREE.MeshBasicMaterial({
      map: this.videoTexture,
      depthWrite: false,
    });
    this.bgMesh = new THREE.Mesh(bgGeometry, bgMaterial);
    this.bgMesh.position.z = -5;
    this.scene.add(this.bgMesh);

    this.createAtlasPipeline(createIndices());
    this.createFaceMesh(createIndices());

    // Exibe a camera como um espelho, sem alterar a orientacao interna do atlas.
    this.scene.scale.set(-1, 1, 1);

    window.addEventListener('resize', () => this.onResize());
  }

  createAtlasPipeline(indices) {
    const atlasSize = State.view.atlasSize;
    const targetOptions = {
      format: THREE.RGBAFormat,
      minFilter: THREE.LinearFilter,
      magFilter: THREE.LinearFilter,
      generateMipmaps: false,
      depthBuffer: false,
      stencilBuffer: false,
    };

    this.unwrapTarget = new THREE.WebGLRenderTarget(atlasSize, atlasSize, targetOptions);
    this.mirrorTarget = new THREE.WebGLRenderTarget(atlasSize, atlasSize, targetOptions);
    this.atlasCamera = new THREE.OrthographicCamera(-1, 1, 1, -1, 0, 1);

    const atlasPositions = new Float32Array(LANDMARK_COUNT * 3);
    const mirrorUvs = new Float32Array(LANDMARK_COUNT * 2);
    const signedCenterDistances = new Float32Array(LANDMARK_COUNT);

    for (let i = 0; i < LANDMARK_COUNT; i++) {
      const u = CANONICAL_UVS[i * 2];
      const v = CANONICAL_UVS[i * 2 + 1];
      const pairIndex = SYMMETRY_PAIRS[i] ?? i;

      atlasPositions[i * 3] = u * 2 - 1;
      atlasPositions[i * 3 + 1] = v * 2 - 1;
      mirrorUvs[i * 2] = CANONICAL_UVS[pairIndex * 2];
      mirrorUvs[i * 2 + 1] = CANONICAL_UVS[pairIndex * 2 + 1];
      const canonicalSide = u < 0.5 ? -1 : (u > 0.5 ? 1 : 0);
      signedCenterDistances[i] = canonicalSide * CANONICAL_CENTER_DISTANCES[i];
    }

    const screenUvAttribute = new THREE.BufferAttribute(this.screenUvs, 2);
    screenUvAttribute.setUsage(THREE.DynamicDrawUsage);

    const unwrapGeometry = new THREE.BufferGeometry();
    unwrapGeometry.setIndex(indices);
    unwrapGeometry.setAttribute('position', new THREE.BufferAttribute(atlasPositions, 3));
    unwrapGeometry.setAttribute('aScreenUv', screenUvAttribute);

    const unwrapMaterial = new THREE.ShaderMaterial({
      uniforms: { uVideo: { value: this.videoTexture } },
      vertexShader: `
        attribute vec2 aScreenUv;
        varying vec2 vScreenUv;
        void main() {
          vScreenUv = aScreenUv;
          gl_Position = vec4(position, 1.0);
        }
      `,
      fragmentShader: `
        uniform sampler2D uVideo;
        varying vec2 vScreenUv;
        void main() {
          gl_FragColor = texture2D(uVideo, vScreenUv);
        }
      `,
      side: THREE.DoubleSide,
      depthTest: false,
      depthWrite: false,
    });

    this.unwrapScene = new THREE.Scene();
    const unwrapMesh = new THREE.Mesh(unwrapGeometry, unwrapMaterial);
    unwrapMesh.frustumCulled = false;
    this.unwrapScene.add(unwrapMesh);

    const mirrorGeometry = new THREE.BufferGeometry();
    mirrorGeometry.setIndex(indices);
    mirrorGeometry.setAttribute('position', new THREE.BufferAttribute(atlasPositions, 3));
    mirrorGeometry.setAttribute('uv', new THREE.BufferAttribute(CANONICAL_UVS, 2));
    mirrorGeometry.setAttribute('aMirrorUv', new THREE.BufferAttribute(mirrorUvs, 2));
    mirrorGeometry.setAttribute('aSignedCenterDistance', new THREE.BufferAttribute(signedCenterDistances, 1));

    this.mirrorAtlasMaterial = new THREE.ShaderMaterial({
      uniforms: {
        uAtlas: { value: this.unwrapTarget.texture },
        uMirrorEnabled: { value: 1 },
        uTargetSide: { value: -1 },
        uStrength: { value: State.mirror.strength },
        uFeatherWidth: { value: State.mirror.featherWidth },
      },
      vertexShader: `
        attribute vec2 aMirrorUv;
        attribute float aSignedCenterDistance;
        varying vec2 vOriginalUv;
        varying vec2 vMirrorUv;
        varying float vSignedCenterDistance;
        void main() {
          vOriginalUv = uv;
          vMirrorUv = aMirrorUv;
          vSignedCenterDistance = aSignedCenterDistance;
          gl_Position = vec4(position, 1.0);
        }
      `,
      fragmentShader: `
        uniform sampler2D uAtlas;
        uniform float uMirrorEnabled;
        uniform float uTargetSide;
        uniform float uStrength;
        uniform float uFeatherWidth;
        varying vec2 vOriginalUv;
        varying vec2 vMirrorUv;
        varying float vSignedCenterDistance;
        void main() {
          vec4 originalColor = texture2D(uAtlas, vOriginalUv);
          vec4 mirroredColor = texture2D(uAtlas, vMirrorUv);
          float targetDistance = uTargetSide * vSignedCenterDistance;
          float centerBlend = smoothstep(0.0, max(uFeatherWidth, 0.0001), targetDistance);
          float mirrorAmount = uMirrorEnabled * centerBlend * uStrength;
          gl_FragColor = mix(originalColor, mirroredColor, mirrorAmount);
        }
      `,
      side: THREE.DoubleSide,
      depthTest: false,
      depthWrite: false,
    });

    this.mirrorScene = new THREE.Scene();
    const mirrorMesh = new THREE.Mesh(mirrorGeometry, this.mirrorAtlasMaterial);
    mirrorMesh.frustumCulled = false;
    this.mirrorScene.add(mirrorMesh);
    this.screenUvAttribute = screenUvAttribute;
  }

  createFaceMesh(indices) {
    const faceGeometry = new THREE.BufferGeometry();
    const positionAttribute = new THREE.BufferAttribute(this.positions, 3);
    positionAttribute.setUsage(THREE.DynamicDrawUsage);

    faceGeometry.setIndex(indices);
    faceGeometry.setAttribute('position', positionAttribute);
    faceGeometry.setAttribute('uv', new THREE.BufferAttribute(CANONICAL_UVS, 2));
    faceGeometry.setAttribute('aAlpha', new THREE.BufferAttribute(this.alphas, 1));
    // Compartilhado com o passe de unwrap; nao e lido pelo material final.
    faceGeometry.setAttribute('aScreenUv', this.screenUvAttribute);

    this.faceSmoothMaterial = new THREE.ShaderMaterial({
      uniforms: {
        map: { value: this.mirrorTarget.texture },
        uGlobalOpacity: { value: State.view.meshOpacity },
      },
      vertexShader: `
        attribute float aAlpha;
        varying vec2 vUv;
        varying float vAlpha;
        varying vec3 vViewPosition;
        void main() {
          vec4 viewPosition = modelViewMatrix * vec4(position, 1.0);
          vUv = uv;
          vAlpha = aAlpha;
          vViewPosition = viewPosition.xyz;
          gl_Position = projectionMatrix * viewPosition;
        }
      `,
      fragmentShader: `
        uniform sampler2D map;
        uniform float uGlobalOpacity;
        varying vec2 vUv;
        varying float vAlpha;
        varying vec3 vViewPosition;
        void main() {
          vec4 texColor = texture2D(map, vUv);
          vec3 surfaceNormal = cross(dFdx(vViewPosition), dFdy(vViewPosition));
          float facing = abs(surfaceNormal.z) / max(length(surfaceNormal), 0.00001);
          float visibility = smoothstep(0.12, 0.38, facing);
          float opacity = texColor.a * vAlpha * visibility * uGlobalOpacity;
          gl_FragColor = vec4(texColor.rgb, opacity);
        }
      `,
      extensions: { derivatives: true },
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
    this.faceMesh.visible = false;
    this.faceMesh.frustumCulled = false;
    this.scene.add(this.faceMesh);
  }

  setVisualMode(mode) {
    this.faceMesh.material = mode === 'mirrored_face'
      ? this.faceSmoothMaterial
      : this.wireframeMaterial;
  }

  setOpacity(value) {
    this.faceSmoothMaterial.uniforms.uGlobalOpacity.value = value;
    this.wireframeMaterial.opacity = value;
  }

  updateAtlasUniforms() {
    const material = this.mirrorAtlasMaterial;
    const enabled = State.mirror.mode !== 'disabled' && State.mirror.strength > 0;

    material.uniforms.uMirrorEnabled.value = enabled ? 1 : 0;
    material.uniforms.uTargetSide.value = State.mirror.mode === 'left_healthy' ? -1 : 1;
    material.uniforms.uStrength.value = State.mirror.strength;
    material.uniforms.uFeatherWidth.value = State.mirror.featherWidth;
  }

  renderAtlas() {
    const previousTarget = this.renderer.getRenderTarget();
    const previousColor = this.renderer.getClearColor(new THREE.Color());
    const previousAlpha = this.renderer.getClearAlpha();

    this.renderer.setClearColor(0x000000, 0);
    this.renderer.setRenderTarget(this.unwrapTarget);
    this.renderer.clear();
    this.renderer.render(this.unwrapScene, this.atlasCamera);

    this.updateAtlasUniforms();
    this.renderer.setRenderTarget(this.mirrorTarget);
    this.renderer.clear();
    this.renderer.render(this.mirrorScene, this.atlasCamera);

    this.renderer.setRenderTarget(previousTarget);
    this.renderer.setClearColor(previousColor, previousAlpha);
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
    if (this.faceMesh.visible && this.faceMesh.material === this.faceSmoothMaterial) {
      this.renderAtlas();
    }
    this.renderer.render(this.scene, this.camera);
  }
}
