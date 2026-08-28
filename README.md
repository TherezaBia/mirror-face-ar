# 🪞 Mirror Face AR - Terapia de Espelho Facial 3D em Tempo Real

Aplicação web open source para **Terapia de Espelho em Realidade Aumentada (ARMT - Augmented Reality Mirror Therapy)** voltada para a reabilitação de paralisia facial periférica (ex: Paralisia de Bell), utilizando **MediaPipe FaceLandmarker** e **Three.js / WebGL**.

---

## 🌟 Principais Funcionalidades

* **Rastreamento Facial 3D em Tempo Real:** Captura 468 pontos anatômicos da face com aceleração por GPU via WebAssembly.
* **Espelhamento Facial Sincronizado (3D + Textura):** Espelha a geometria 3D e a textura da pele do lado saudável sobre o lado paralisado, garantindo simetria perfeita em 60 FPS.
* **Modos Clínicos Selecionáveis:**
  * Lado Esquerdo Saudável $\rightarrow$ Espelha sobre o Lado Direito.
  * Lado Direito Saudável $\rightarrow$ Espelha sobre o Lado Esquerdo.
  * Força do Espelhamento ajustável progressivamente ($0\%$ a $100\%$).
* **Calibração Anatômica Milimétrica:** Microajustes sagitais de largura e alinhamento central (padrão MEPP).
* **Visualização Alternativa:** Modo Real com textura da pele ou Malha 3D Wireframe.
* **Zero Instalação:** Roda diretamente no navegador (Chrome, Edge, Safari, Firefox).

---

## 🏛️ Estrutura do Projeto

```
mirror-face-ar/
├── index.html                    # Ponto de entrada web
├── src/
│   ├── main.js                   # Orquestrador da aplicação
│   ├── config.js                 # Estado global e parâmetros clínicos
│   ├── tracker/
│   │   └── faceTracker.js        # Módulo de visão computacional (MediaPipe + Webcam)
│   ├── graphics/
│   │   └── sceneManager.js       # Gerenciador da cena Three.js e WebGL
│   ├── mirror/
│   │   └── mirrorEngine.js       # Motor de espelhamento facial 3D sincronizado
│   ├── ui/
│   │   └── guiController.js      # Painel de controle da interface (lil-gui)
│   ├── data/
│   │   ├── triangles.js          # 854 triângulos canônicos da malha
│   │   └── pairs.js              # 468 pares simétricos anatômicos
│   └── styles/
│       └── main.css              # Estilos e HUD médico
```

---

## 🚀 Como Executar Localmente

Como a aplicação utiliza módulos ES nativos e acesso à câmera, execute um servidor HTTP simples:

### Node.js (recomendado)

```bash
npm start
```

Abra `http://localhost:8080`. Durante o desenvolvimento, `npm run dev` reinicia o servidor automaticamente quando `server.js` é alterado.

### Alternativas

```bash
# Com Python:
python -m http.server 8080

# Ou com Node.js (npx):
npx serve .
```

Abra no navegador em `http://localhost:8080` e autorize o acesso à câmera.

---

## Pipeline de textura pose-invariante

O espelhamento de textura usa UVs fixas do modelo facial canônico do MediaPipe em três passes de GPU:

1. A malha real é desembrulhada do vídeo para um atlas facial off-screen.
2. O atlas é espelhado pelos pares anatômicos, com transição por distância geodésica até a linha central.
3. A malha na pose atual é renderizada usando somente as UVs canônicas.

Landmarks recebem suavização temporal EMA e superfícies em ângulo rasante perdem opacidade pela relação entre normal e direção de visão. Assim, rotação e inclinação da cabeça não alteram o endereço de textura de cada vértice.

Os dados UV ficam em `src/data/canonical_uvs.js` e podem ser verificados junto com topologia e pares por:

```bash
npm test
```

---

## 📜 Licença

MIT License - Software Livre e de Código Aberto.
