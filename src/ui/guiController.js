import { GUI } from 'three/addons/libs/lil-gui.module.min.js';
import { State } from '../config.js';

export class GuiController {
  constructor(sceneManager) {
    this.sceneManager = sceneManager;
    this.gui = new GUI({ title: 'Terapia de Espelho (MEPP AR)' });
    this.setupControls();
  }

  setupControls() {
    // 1. Pasta de Terapia
    const therapyFolder = this.gui.addFolder('Modo de Terapia');
    
    therapyFolder.add(State.mirror, 'mode', {
      'Lado Esquerdo Saudável (Padrão)': 'left_healthy',
      'Lado Direito Saudável': 'right_healthy',
      'Desativado (Rosto Normal)': 'disabled',
    }).name('Lado Paralisado');

    therapyFolder.add(State.mirror, 'strength', 0.0, 1.0, 0.05).name('Força do Espelho');
    therapyFolder.add(State.mirror, 'synchronize3D').name('Sincronia 3D (Geometria + Pele)');

    // 2. Pasta de Visualização
    const viewFolder = this.gui.addFolder('Visualização');

    viewFolder.add(State.view, 'mode', {
      'Espelho Real (Textura)': 'mirrored_face',
      'Malha 3D (Wireframe)': 'wireframe',
    }).name('Estilo Visual').onChange((mode) => {
      this.sceneManager.setVisualMode(mode);
    });

    viewFolder.add(State.view, 'meshOpacity', 0.1, 1.0, 0.05).name('Opacidade').onChange((val) => {
      this.sceneManager.setOpacity(val);
    });

    viewFolder.add(State.view, 'zScale', 0.5, 4.0, 0.1).name('Relevo 3D');

    // 3. Pasta de Calibração Anatômica
    const calibFolder = this.gui.addFolder('Calibração Anatômica');
    calibFolder.add(State.calibration, 'offsetX', -0.05, 0.05, 0.002).name('Ajuste X');
    calibFolder.add(State.calibration, 'offsetY', -0.05, 0.05, 0.002).name('Ajuste Y');
    calibFolder.add(State.calibration, 'scaleX', 0.8, 1.2, 0.01).name('Largura');
    calibFolder.close();
  }
}
