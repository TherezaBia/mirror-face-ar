import cv2
import mediapipe as mp
import numpy as np
import time


# ============================================================
# SCIPY OPCIONAL
# ============================================================

try:
    from scipy.optimize import linear_sum_assignment
    TEM_SCIPY = True
except ImportError:
    TEM_SCIPY = False


# ============================================================
# CONFIGURAÇÕES GERAIS
# ============================================================

MODEL_PATH = "face_landmarker.task"

CAMERA_ID = 0


# ============================================================
# FILTRO ADAPTATIVO
# ============================================================
#
# IMPORTANTE:
#
# Estes valores representam quanto do FRAME ANTERIOR
# será mantido.
#
# Valor alto  -> mais suavização
# Valor baixo -> resposta mais rápida
#
# Quando o rosto está parado:
#
#     ~0.70
#
# Quando há movimento rápido:
#
#     ~0.03
#
# ============================================================

SMOOTHING_PARADO = 0.72

SMOOTHING_RAPIDO = 0.03


# Movimento normalizado pela largura do rosto.
#
# Abaixo de VELOCIDADE_BAIXA:
# considera praticamente parado.
#
# Acima de VELOCIDADE_ALTA:
# praticamente remove a suavização.

VELOCIDADE_BAIXA = 0.0015

VELOCIDADE_ALTA = 0.028


# Quando a cabeça inteira se movimenta,
# todos os landmarks precisam reagir rapidamente.

GLOBAL_SPEED_GAIN = 1.20


# ============================================================
# COMPOSIÇÃO
# ============================================================

FEATHER_SIGMA = 2.5


# ============================================================
# OLHOS
# ============================================================
#
# Queremos preservar:
#
# - esclera
# - íris
# - pupila
#
# NÃO queremos preservar:
#
# - sobrancelha
# - pálpebra
# - região periocular
#
# ============================================================

EYE_OPENING_SCALE = 0.94


# ============================================================
# GEOMETRIA
# ============================================================

CENTER_TOL = 0.014

PAIR_MAX_DISTANCE = 0.075

EXPRESSION_GAIN = 1.0


# Limite de deformação facial.
#
# Proporcional à largura do rosto.
#
# Evita que um landmark com erro ocasional
# faça um triângulo "explodir".

MAX_EXPRESSION_DELTA = 0.15


MIN_TRIANGLE_AREA = 1.5

MIN_AREA_RATIO = 0.20

MAX_AREA_RATIO = 5.0


# ============================================================
# FPS
# ============================================================

FPS_SMOOTHING = 0.90


# ============================================================
# CONTORNO FACIAL
# ============================================================

FACE_OVAL = [
    10, 338, 297, 332, 284, 251,
    389, 356, 454, 323, 361, 288,
    397, 365, 379, 378, 400, 377,
    152, 148, 176, 149, 150, 136,
    172, 58, 132, 93, 234, 127,
    162, 21, 54, 103, 67, 109
]


# ============================================================
# ABERTURA VISÍVEL DOS OLHOS
# ============================================================

EYE_1 = [
    33,
    160,
    159,
    158,
    157,
    173,
    133,
    155,
    154,
    153,
    145,
    144,
    163,
    7
]


EYE_2 = [
    362,
    385,
    386,
    387,
    388,
    466,
    263,
    249,
    390,
    373,
    374,
    380,
    381,
    382
]


# ============================================================
# PONTOS PARA ESTIMAR A POSE DA CABEÇA
# ============================================================
#
# Evitamos boca e sobrancelha como principais referências,
# porque queremos separar:
#
# movimento da cabeça
#
# de
#
# movimento de expressão.
#
# ============================================================

POSE_ANCHORS = [
    10,
    151,
    9,
    8,
    168,
    6,
    197,
    195,
    5,
    4,
    1,

    234,
    454,
    127,
    356,
    162,
    389,

    21,
    251,

    54,
    284,

    103,
    332,

    67,
    297,

    109,
    338,

    93,
    323,

    132,
    361,

    45,
    275,

    220,
    440
]


# ============================================================
# PARES SIMÉTRICOS CONHECIDOS
# ============================================================

KNOWN_PAIRS = [

    # --------------------------------------------------------
    # CONTORNO FACIAL
    # --------------------------------------------------------

    (109, 338),
    (67, 297),
    (103, 332),
    (54, 284),
    (21, 251),
    (162, 389),
    (127, 356),
    (234, 454),
    (93, 323),
    (132, 361),
    (58, 288),
    (172, 397),
    (136, 365),
    (150, 379),
    (149, 378),
    (176, 400),
    (148, 377),

    # --------------------------------------------------------
    # SOBRANCELHAS
    # --------------------------------------------------------

    (107, 336),
    (66, 296),
    (105, 334),
    (63, 293),
    (70, 300),

    (55, 285),
    (65, 295),
    (52, 282),
    (53, 283),
    (46, 276),

    # --------------------------------------------------------
    # OLHOS / PÁLPEBRAS
    # --------------------------------------------------------

    (33, 263),
    (133, 362),

    (160, 387),
    (159, 386),
    (158, 385),
    (157, 384),
    (173, 398),

    (7, 249),
    (163, 390),
    (144, 373),
    (145, 374),
    (153, 380),
    (154, 381),
    (155, 382),

    (161, 388),
    (246, 466),

    # --------------------------------------------------------
    # NARIZ
    # --------------------------------------------------------

    (45, 275),
    (220, 440),
    (115, 344),
    (48, 278),
    (64, 294),
    (98, 327),
    (97, 326),

    # --------------------------------------------------------
    # BOCA EXTERNA
    # --------------------------------------------------------

    (61, 291),

    (37, 267),
    (39, 269),
    (40, 270),
    (185, 409),

    (84, 314),
    (181, 405),
    (91, 321),
    (146, 375),

    (82, 312),
    (81, 311),
    (80, 310),
    (191, 415),

    (78, 308),

    (87, 317),
    (178, 402),
    (88, 318),
    (95, 324),

    # --------------------------------------------------------
    # BOCA INTERNA
    # --------------------------------------------------------

    (72, 302),
    (73, 303),
    (74, 304),
    (184, 408),

    (76, 306),
    (77, 307),

    (90, 320),
    (180, 404),
    (85, 315),

    (38, 268),
    (41, 271),
    (42, 272),
    (183, 407),

    (62, 292),
    (96, 325),
    (89, 319),
    (179, 403),
    (86, 316),

    # --------------------------------------------------------
    # REGIÕES INTERNAS / BOCHECHA
    # --------------------------------------------------------

    (108, 337),
    (69, 299),
    (104, 333),
    (68, 298),
    (71, 301),

    (139, 368),
    (34, 264),
    (227, 447),
    (137, 366),

    (177, 401),
    (215, 435),
    (138, 367),
    (135, 364),

    (169, 394),
    (170, 395),
    (140, 369),
    (171, 396),

    (156, 383),
    (143, 372),

    (116, 345),
    (123, 352),
    (147, 376),

    (213, 433),
    (192, 416),
    (214, 434),

    (210, 430),
    (211, 431),

    (32, 262),
    (208, 428),

    (124, 353),
    (35, 265),

    (111, 340),
    (117, 346),

    (50, 280),
    (187, 411),

    (207, 427),
    (216, 436),

    (212, 432),
    (202, 422),

    (204, 424),
    (194, 418),

    (201, 421),

    (193, 417),
    (189, 413),

    (221, 441),
    (222, 442),
    (223, 443),
    (224, 444),
    (225, 445),
    (226, 446),

    (31, 261),
    (228, 448),

    (118, 347),
    (101, 330),

    (205, 425),
    (206, 426),

    (92, 322),
    (186, 410),

    (57, 287),
    (43, 273),

    (106, 335),
    (182, 406),
    (83, 313),

    (190, 414),
    (56, 286),

    (28, 258),
    (27, 257),
    (29, 259),

    (130, 359),
    (25, 255),

    (110, 339),
    (229, 449),

    (119, 348),
    (100, 329),

    (36, 266),
    (203, 423),

    (165, 391),
    (167, 393),

    (122, 351),
    (245, 465),
    (244, 464),
    (243, 463),

    (112, 341),

    (26, 256),
    (22, 252),
    (23, 253),
    (24, 254),

    (230, 450),

    (120, 349),
    (47, 277),

    (126, 355),
    (142, 371),
    (129, 358),

    (233, 453),
    (232, 452),
    (231, 451),

    (121, 350),
    (128, 357),

    (114, 343),
    (188, 412),

    (196, 419),
    (174, 399),

    (217, 437),
    (198, 420),

    (209, 429),

    (49, 279),

    (102, 331),
    (99, 328)
]


# ============================================================
# MEDIAPIPE
# ============================================================

BaseOptions = mp.tasks.BaseOptions

FaceLandmarker = mp.tasks.vision.FaceLandmarker

FaceLandmarkerOptions = (
    mp.tasks.vision.FaceLandmarkerOptions
)

VisionRunningMode = (
    mp.tasks.vision.RunningMode
)


options = FaceLandmarkerOptions(

    base_options=BaseOptions(
        model_asset_path=MODEL_PATH
    ),

    running_mode=VisionRunningMode.VIDEO,

    num_faces=1,

    min_face_detection_confidence=0.5,

    min_face_presence_confidence=0.5,

    min_tracking_confidence=0.5,

    output_facial_transformation_matrixes=True
)


landmarker = (
    FaceLandmarker.create_from_options(
        options
    )
)


# ============================================================
# FUNÇÕES AUXILIARES
# ============================================================

def normalizar(v):

    v = np.asarray(
        v,
        dtype=np.float32
    )

    norma = np.linalg.norm(v)

    if norma < 1e-8:
        return v

    return v / norma


# ============================================================
# CONVERTE LANDMARKS
# ============================================================

def landmarks_para_arrays(
    face_landmarks,
    largura,
    altura
):

    pontos_2d = []

    pontos_3d = []

    for lm in face_landmarks:

        px = lm.x * largura

        py = lm.y * altura

        pontos_2d.append(
            [
                px,
                py
            ]
        )

        # ----------------------------------------------------
        # Espaço pseudo-3D isotrópico
        # ----------------------------------------------------

        X = lm.x * largura

        Y = lm.y * largura

        Z = lm.z * largura

        pontos_3d.append(
            [
                X,
                Y,
                Z
            ]
        )

    return (
        np.asarray(
            pontos_2d,
            dtype=np.float32
        ),
        np.asarray(
            pontos_3d,
            dtype=np.float32
        )
    )


# ============================================================
# 3D -> PIXEL
# ============================================================

def ponto3d_para_pixel(
    p,
    largura,
    altura
):

    return np.array(
        [
            p[0],

            p[1]
            *
            altura
            /
            largura
        ],
        dtype=np.float32
    )


# ============================================================
# SMOOTHSTEP
# ============================================================

def smoothstep(t):

    t = np.clip(
        t,
        0.0,
        1.0
    )

    return (
        t
        *
        t
        *
        (
            3.0
            -
            2.0
            *
            t
        )
    )


# ============================================================
# FILTRO ADAPTATIVO
# ============================================================
#
# Esta é uma das principais mudanças desta versão.
#
# Cada landmark recebe sua própria suavização.
#
#
# Landmark parado:
#
#     filtro ~ 0.72
#
#
# Landmark se movendo rapidamente:
#
#     filtro ~ 0.03
#
#
# Além disso, quando a cabeça inteira se move,
# todos os landmarks reduzem a suavização.
#
# ============================================================

def aplicar_filtro_adaptativo(
    bruto_2d,
    bruto_3d,
    bruto_anterior_2d,
    filtrado_anterior_2d,
    filtrado_anterior_3d
):

    # ========================================================
    # PRIMEIRO FRAME
    # ========================================================

    if (
        bruto_anterior_2d is None
        or
        filtrado_anterior_2d is None
        or
        filtrado_anterior_3d is None
    ):

        return (
            bruto_2d.copy(),
            bruto_3d.copy(),
            0.0,
            SMOOTHING_PARADO,
            np.full(
                len(bruto_2d),
                SMOOTHING_PARADO,
                dtype=np.float32
            )
        )

    # ========================================================
    # ESCALA DO ROSTO
    # ========================================================

    largura_face = np.linalg.norm(
        bruto_2d[454]
        -
        bruto_2d[234]
    )

    largura_face = max(
        largura_face,
        1.0
    )

    # ========================================================
    # VELOCIDADE DE CADA LANDMARK
    # ========================================================

    deslocamentos = np.linalg.norm(
        bruto_2d
        -
        bruto_anterior_2d,
        axis=1
    )

    velocidades = (
        deslocamentos
        /
        largura_face
    )

    # ========================================================
    # MOVIMENTO GLOBAL DA CABEÇA
    # ========================================================

    ids_pose = [
        i
        for i in POSE_ANCHORS
        if i < len(
            velocidades
        )
    ]

    if len(ids_pose) > 0:

        velocidades_pose = velocidades[
            ids_pose
        ]

        # Percentil 75 responde melhor à rotação
        # do que somente média ou mediana.

        movimento_global = float(
            np.percentile(
                velocidades_pose,
                75
            )
        )

    else:

        movimento_global = float(
            np.median(
                velocidades
            )
        )

    # ========================================================
    # VELOCIDADE EFETIVA
    # ========================================================
    #
    # Se a cabeça inteira se move:
    #
    # todos os landmarks respondem mais rápido.
    #
    # Se apenas a boca se move:
    #
    # principalmente os landmarks da boca respondem rápido.
    #
    # ========================================================

    velocidade_global_aplicada = (
        movimento_global
        *
        GLOBAL_SPEED_GAIN
    )

    velocidades_efetivas = np.maximum(
        velocidades,
        velocidade_global_aplicada
    )

    # ========================================================
    # CONVERTE VELOCIDADE -> INTENSIDADE DO FILTRO
    # ========================================================

    denominador = (
        VELOCIDADE_ALTA
        -
        VELOCIDADE_BAIXA
    )

    denominador = max(
        denominador,
        1e-8
    )

    t = (
        velocidades_efetivas
        -
        VELOCIDADE_BAIXA
    ) / denominador

    t = smoothstep(
        t
    )

    # --------------------------------------------------------
    # PARADO:
    #
    # t = 0
    # filtro = SMOOTHING_PARADO
    #
    # RÁPIDO:
    #
    # t = 1
    # filtro = SMOOTHING_RAPIDO
    # --------------------------------------------------------

    pesos_anteriores = (
        SMOOTHING_PARADO
        +
        (
            SMOOTHING_RAPIDO
            -
            SMOOTHING_PARADO
        )
        *
        t
    )

    pesos_anteriores = pesos_anteriores.astype(
        np.float32
    )

    peso_2d = pesos_anteriores[
        :,
        None
    ]

    # ========================================================
    # FILTRA 2D
    # ========================================================

    filtrado_2d = (
        peso_2d
        *
        filtrado_anterior_2d
        +
        (
            1.0
            -
            peso_2d
        )
        *
        bruto_2d
    )

    # ========================================================
    # FILTRA 3D
    # ========================================================
    #
    # Usamos o MESMO peso do landmark 2D.
    #
    # Isso evita inconsistência:
    #
    # malha 2D em uma posição
    # malha 3D em outra.
    #
    # ========================================================

    filtrado_3d = (
        peso_2d
        *
        filtrado_anterior_3d
        +
        (
            1.0
            -
            peso_2d
        )
        *
        bruto_3d
    )

    filtro_medio = float(
        np.median(
            pesos_anteriores
        )
    )

    return (
        filtrado_2d.astype(
            np.float32
        ),

        filtrado_3d.astype(
            np.float32
        ),

        movimento_global,

        filtro_medio,

        pesos_anteriores
    )


# ============================================================
# BASE 3D NEUTRA
# ============================================================

def criar_base_neutra(
    pontos_3d
):

    centro = (
        pontos_3d[133]
        +
        pontos_3d[362]
    ) / 2.0

    # --------------------------------------------------------
    # EIXO HORIZONTAL
    # --------------------------------------------------------

    eixo_x = (
        pontos_3d[454]
        -
        pontos_3d[234]
    )

    eixo_x = normalizar(
        eixo_x
    )

    if eixo_x[0] < 0:

        eixo_x = -eixo_x

    # --------------------------------------------------------
    # EIXO VERTICAL
    # --------------------------------------------------------

    vertical = (
        pontos_3d[152]
        -
        centro
    )

    vertical = (
        vertical
        -
        np.dot(
            vertical,
            eixo_x
        )
        *
        eixo_x
    )

    eixo_y = normalizar(
        vertical
    )

    # --------------------------------------------------------
    # PROFUNDIDADE
    # --------------------------------------------------------

    eixo_z = np.cross(
        eixo_x,
        eixo_y
    )

    eixo_z = normalizar(
        eixo_z
    )

    largura_face = np.linalg.norm(
        pontos_3d[454]
        -
        pontos_3d[234]
    )

    largura_face = max(
        largura_face,
        1.0
    )

    return (
        centro,
        eixo_x,
        eixo_y,
        eixo_z,
        largura_face
    )


# ============================================================
# COORDENADA X LOCAL
# ============================================================

def coordenada_x_local(
    ponto,
    base
):

    centro, eixo_x, _, _, largura_face = base

    return (
        np.dot(
            ponto
            -
            centro,
            eixo_x
        )
        /
        largura_face
    )


# ============================================================
# MATRIZ DE REFLEXÃO 3D
# ============================================================

def criar_matriz_reflexao(
    eixo_x
):

    eixo_x = normalizar(
        eixo_x
    )

    return (
        np.eye(
            3,
            dtype=np.float32
        )
        -
        2.0
        *
        np.outer(
            eixo_x,
            eixo_x
        )
    )


# ============================================================
# ESTIMA POSE RÍGIDA
# ============================================================

def estimar_pose_rigida(
    neutro_3d,
    atual_3d
):

    ids = [
        i
        for i in POSE_ANCHORS
        if (
            i < len(
                neutro_3d
            )
            and
            i < len(
                atual_3d
            )
        )
    ]

    A = neutro_3d[
        ids
    ].astype(
        np.float64
    )

    B = atual_3d[
        ids
    ].astype(
        np.float64
    )

    centro_A = np.mean(
        A,
        axis=0
    )

    centro_B = np.mean(
        B,
        axis=0
    )

    A0 = (
        A
        -
        centro_A
    )

    B0 = (
        B
        -
        centro_B
    )

    H = (
        A0.T
        @
        B0
    )

    U, S, Vt = np.linalg.svd(
        H
    )

    R = (
        Vt.T
        @
        U.T
    )

    # Evita reflexão acidental
    if np.linalg.det(
        R
    ) < 0:

        Vt[
            -1,
            :
        ] *= -1

        R = (
            Vt.T
            @
            U.T
        )

    denominador = np.sum(
        A0 ** 2
    )

    if denominador < 1e-8:

        escala = 1.0

    else:

        escala = (
            np.sum(
                S
            )
            /
            denominador
        )

    escala = float(
        np.clip(
            escala,
            0.5,
            2.0
        )
    )

    t = (
        centro_B
        -
        escala
        *
        (
            R
            @
            centro_A
        )
    )

    return (
        escala,

        R.astype(
            np.float32
        ),

        t.astype(
            np.float32
        )
    )


# ============================================================
# APLICA POSE
# ============================================================

def aplicar_pose(
    ponto,
    escala,
    R,
    t
):

    return (
        escala
        *
        (
            R
            @
            ponto
        )
        +
        t
    )


# ============================================================
# REMOVE POSE
# ============================================================

def remover_pose(
    ponto,
    escala,
    R,
    t
):

    escala = max(
        escala,
        1e-8
    )

    return (
        R.T
        @
        (
            ponto
            -
            t
        )
        /
        escala
    )


# ============================================================
# PAREAMENTO GULOSO
# ============================================================

def pareamento_guloso(
    custo,
    max_distancia
):

    pares = []

    if custo.size == 0:

        return pares

    ordem = np.argsort(
        custo,
        axis=None
    )

    usados_esquerda = set()

    usados_direita = set()

    _, numero_colunas = custo.shape

    for flat_index in ordem:

        i = (
            flat_index
            //
            numero_colunas
        )

        j = (
            flat_index
            %
            numero_colunas
        )

        if i in usados_esquerda:
            continue

        if j in usados_direita:
            continue

        distancia = np.sqrt(
            custo[
                i,
                j
            ]
        )

        if distancia > max_distancia:
            continue

        pares.append(
            (
                i,
                j
            )
        )

        usados_esquerda.add(
            i
        )

        usados_direita.add(
            j
        )

    return pares


# ============================================================
# CRIA MAPA SIMÉTRICO
# ============================================================

def criar_mapa_simetrico(
    pontos_3d
):

    n = min(
        len(
            pontos_3d
        ),
        468
    )

    neutro = pontos_3d[
        :n
    ]

    base = criar_base_neutra(
        neutro
    )

    (
        centro,
        eixo_x,
        eixo_y,
        eixo_z,
        largura_face
    ) = base

    reflexao = criar_matriz_reflexao(
        eixo_x
    )

    centro_ids = []

    esquerda_ids = []

    direita_ids = []

    # ========================================================
    # CLASSIFICA PONTOS
    # ========================================================

    for i in range(
        n
    ):

        x = coordenada_x_local(
            neutro[
                i
            ],
            base
        )

        if abs(
            x
        ) <= CENTER_TOL:

            centro_ids.append(
                i
            )

        elif x < 0:

            esquerda_ids.append(
                i
            )

        else:

            direita_ids.append(
                i
            )

    mapa = {}

    usados_esquerda = set()

    usados_direita = set()

    # ========================================================
    # CENTRO
    # ========================================================

    for i in centro_ids:

        mapa[
            i
        ] = i

    # ========================================================
    # PARES CONHECIDOS
    # ========================================================

    for a, b in KNOWN_PAIRS:

        if (
            a >= n
            or
            b >= n
        ):

            continue

        xa = coordenada_x_local(
            neutro[
                a
            ],
            base
        )

        xb = coordenada_x_local(
            neutro[
                b
            ],
            base
        )

        if xa < xb:

            src = a

            dst = b

        else:

            src = b

            dst = a

        if coordenada_x_local(
            neutro[
                src
            ],
            base
        ) >= 0:

            continue

        if coordenada_x_local(
            neutro[
                dst
            ],
            base
        ) <= 0:

            continue

        mapa[
            src
        ] = dst

        usados_esquerda.add(
            src
        )

        usados_direita.add(
            dst
        )

    # ========================================================
    # PONTOS RESTANTES
    # ========================================================

    esquerda_restante = [
        i
        for i in esquerda_ids
        if i not in usados_esquerda
    ]

    direita_restante = [
        i
        for i in direita_ids
        if i not in usados_direita
    ]

    if (
        len(
            esquerda_restante
        ) > 0
        and
        len(
            direita_restante
        ) > 0
    ):

        refletidos = []

        for idx in esquerda_restante:

            delta = (
                neutro[
                    idx
                ]
                -
                centro
            )

            refletido = (
                centro
                +
                reflexao
                @
                delta
            )

            refletidos.append(
                refletido
            )

        refletidos = np.asarray(
            refletidos,
            dtype=np.float32
        )

        direita_pts = neutro[
            direita_restante
        ]

        diff = (
            refletidos[
                :,
                None,
                :
            ]
            -
            direita_pts[
                None,
                :,
                :
            ]
        )

        custo = np.sum(
            (
                diff
                /
                largura_face
            )
            ** 2,
            axis=2
        )

        # ====================================================
        # HUNGARIAN
        # ====================================================

        if TEM_SCIPY:

            linhas, colunas = (
                linear_sum_assignment(
                    custo
                )
            )

            pares_indices = list(
                zip(
                    linhas,
                    colunas
                )
            )

        else:

            pares_indices = pareamento_guloso(
                custo,
                PAIR_MAX_DISTANCE
            )

        # ====================================================
        # SALVA PARES
        # ====================================================

        for li, ri in pares_indices:

            distancia = (
                np.linalg.norm(
                    refletidos[
                        li
                    ]
                    -
                    direita_pts[
                        ri
                    ]
                )
                /
                largura_face
            )

            if distancia > PAIR_MAX_DISTANCE:

                continue

            src = esquerda_restante[
                li
            ]

            dst = direita_restante[
                ri
            ]

            mapa[
                src
            ] = dst

    return (
        mapa,
        centro_ids,
        base,
        reflexao
    )


# ============================================================
# TRIANGULAÇÃO DELAUNAY
# ============================================================

def criar_triangulos_delaunay_por_indices(
    pontos_2d,
    indices,
    largura,
    altura
):

    if len(
        indices
    ) < 3:

        return []

    subdiv = cv2.Subdiv2D(
        (
            0,
            0,
            largura,
            altura
        )
    )

    coordenadas = []

    ids_validos = []

    usados = set()

    # ========================================================
    # INSERE LANDMARKS
    # ========================================================

    for idx in indices:

        x = float(
            pontos_2d[
                idx,
                0
            ]
        )

        y = float(
            pontos_2d[
                idx,
                1
            ]
        )

        if not (
            0 <= x < largura
            and
            0 <= y < altura
        ):

            continue

        chave = (
            int(
                round(
                    x
                )
            ),
            int(
                round(
                    y
                )
            )
        )

        if chave in usados:

            continue

        usados.add(
            chave
        )

        try:

            subdiv.insert(
                (
                    x,
                    y
                )
            )

            coordenadas.append(
                [
                    x,
                    y
                ]
            )

            ids_validos.append(
                idx
            )

        except cv2.error:

            pass

    if len(
        coordenadas
    ) < 3:

        return []

    coordenadas = np.asarray(
        coordenadas,
        dtype=np.float32
    )

    lista = subdiv.getTriangleList()

    triangulos = set()

    # ========================================================
    # TRIÂNGULOS -> IDS DOS LANDMARKS
    # ========================================================

    for t in lista:

        vertices = np.asarray(
            [
                [
                    t[0],
                    t[1]
                ],

                [
                    t[2],
                    t[3]
                ],

                [
                    t[4],
                    t[5]
                ]
            ],
            dtype=np.float32
        )

        tri_ids = []

        valido = True

        for vertice in vertices:

            distancias = np.linalg.norm(
                coordenadas
                -
                vertice,
                axis=1
            )

            k = int(
                np.argmin(
                    distancias
                )
            )

            if distancias[
                k
            ] > 2.5:

                valido = False

                break

            tri_ids.append(
                ids_validos[
                    k
                ]
            )

        if not valido:

            continue

        if len(
            set(
                tri_ids
            )
        ) != 3:

            continue

        triangulos.add(
            tuple(
                sorted(
                    tri_ids
                )
            )
        )

    return list(
        triangulos
    )


# ============================================================
# CALIBRAÇÃO
# ============================================================

def calibrar(
    pontos_2d,
    pontos_3d,
    largura,
    altura
):

    (
        mapa,
        centro_ids,
        base,
        reflexao
    ) = criar_mapa_simetrico(
        pontos_3d
    )

    (
        centro,
        eixo_x,
        eixo_y,
        eixo_z,
        largura_face
    ) = base

    source_indices = []

    # ========================================================
    # SELECIONA METADE ESQUERDA
    # ========================================================

    for src in mapa.keys():

        if mapa[
            src
        ] == src:

            source_indices.append(
                src
            )

        else:

            x = coordenada_x_local(
                pontos_3d[
                    src
                ],
                base
            )

            if x < 0:

                source_indices.append(
                    src
                )

    source_indices = sorted(
        set(
            source_indices
        )
    )

    # ========================================================
    # TRIANGULAÇÃO FIXA
    # ========================================================

    triangulos = (
        criar_triangulos_delaunay_por_indices(
            pontos_2d,
            source_indices,
            largura,
            altura
        )
    )

    triangulos_validos = []

    for tri in triangulos:

        if all(
            i in mapa
            for i in tri
        ):

            triangulos_validos.append(
                tri
            )

    calibracao = {

        "neutro_3d": pontos_3d.copy(),

        "mapa": mapa,

        "centro_ids": set(
            centro_ids
        ),

        "source_indices": source_indices,

        "triangulos": triangulos_validos,

        "base": base,

        "reflexao": reflexao,

        "largura_face": largura_face
    }

    print()

    print(
        "========================================"
    )

    print(
        "CALIBRACAO POSE-AWARE CONCLUIDA"
    )

    print(
        "========================================"
    )

    print(
        "Pares esquerda -> direita:",
        sum(
            1
            for src, dst in mapa.items()
            if src != dst
        )
    )

    print(
        "Pontos centrais:",
        sum(
            1
            for src, dst in mapa.items()
            if src == dst
        )
    )

    print(
        "Triangulos:",
        len(
            triangulos_validos
        )
    )

    if TEM_SCIPY:

        print(
            "Pareamento: Hungarian / SciPy"
        )

    else:

        print(
            "Pareamento: guloso"
        )

    print(
        "Filtro temporal: ADAPTATIVO"
    )

    print(
        "Suavizacao virtual adicional: DESATIVADA"
    )

    print(
        "========================================"
    )

    print()

    return calibracao


# ============================================================
# CRIA GEOMETRIA VIRTUAL
# ============================================================
#
# IMPORTANTE:
#
# Nesta versão NÃO existe mais:
#
# VIRTUAL_SMOOTHING
#
# nem:
#
# virtual_anterior
#
# A malha virtual é calculada diretamente
# a partir dos landmarks adaptativamente filtrados.
#
# ============================================================

def criar_geometria_virtual(
    pontos_3d,
    calibracao
):

    neutro_3d = calibracao[
        "neutro_3d"
    ]

    mapa = calibracao[
        "mapa"
    ]

    reflexao = calibracao[
        "reflexao"
    ]

    largura_face = calibracao[
        "largura_face"
    ]

    # ========================================================
    # POSE RÍGIDA
    # ========================================================

    (
        escala,
        R,
        t
    ) = estimar_pose_rigida(
        neutro_3d,
        pontos_3d
    )

    virtual = {}

    # ========================================================
    # CADA LANDMARK ESQUERDO
    # ========================================================

    for src, dst in mapa.items():

        # ====================================================
        # PONTO CENTRAL
        # ====================================================

        if src == dst:

            virtual[
                src
            ] = pontos_3d[
                src
            ].copy()

            continue

        # ====================================================
        # REMOVE POSE DA CABEÇA
        # ====================================================

        src_sem_pose = remover_pose(
            pontos_3d[
                src
            ],
            escala,
            R,
            t
        )

        # ====================================================
        # MOVIMENTO DE EXPRESSÃO
        # ====================================================

        delta = (
            src_sem_pose
            -
            neutro_3d[
                src
            ]
        )

        # ====================================================
        # LIMITADOR DE OUTLIER
        # ====================================================

        tamanho_delta_normalizado = (
            np.linalg.norm(
                delta
            )
            /
            largura_face
        )

        if (
            tamanho_delta_normalizado
            >
            MAX_EXPRESSION_DELTA
        ):

            # ------------------------------------------------
            # CORREÇÃO IMPORTANTE
            #
            # A versão anterior dividia novamente por
            # largura_face.
            #
            # O correto é simplesmente:
            #
            # MAX / tamanho_atual
            # ------------------------------------------------

            fator = (
                MAX_EXPRESSION_DELTA
                /
                tamanho_delta_normalizado
            )

            delta = (
                delta
                *
                fator
            )

        # ====================================================
        # ESPELHA A DEFORMAÇÃO
        # ====================================================

        delta_espelhado = (
            reflexao
            @
            delta
        )

        # ====================================================
        # APLICA À ANATOMIA DIREITA
        # ====================================================

        destino_neutro = (
            neutro_3d[
                dst
            ]
            +
            EXPRESSION_GAIN
            *
            delta_espelhado
        )

        # ====================================================
        # REAPLICA POSE ATUAL
        # ====================================================

        destino_atual = aplicar_pose(
            destino_neutro,
            escala,
            R,
            t
        )

        virtual[
            src
        ] = destino_atual

    return (
        virtual,
        escala,
        R,
        t
    )


# ============================================================
# ÁREA TRIÂNGULO
# ============================================================

def area_triangulo(
    tri
):

    a = tri[
        0
    ]

    b = tri[
        1
    ]

    c = tri[
        2
    ]

    return abs(
        (
            b[0]
            -
            a[0]
        )
        *
        (
            c[1]
            -
            a[1]
        )
        -
        (
            b[1]
            -
            a[1]
        )
        *
        (
            c[0]
            -
            a[0]
        )
    ) * 0.5


# ============================================================
# WARP TRIANGULAR
# ============================================================

def acumular_triangulo(
    frame,
    acumulador,
    pesos,
    tri_src,
    tri_dst
):

    altura, largura = (
        frame.shape[
            :2
        ]
    )

    # ========================================================
    # LIMITES
    # ========================================================

    for p in tri_src:

        if not (
            0 <= p[0] < largura
            and
            0 <= p[1] < altura
        ):

            return

    for p in tri_dst:

        if not (
            0 <= p[0] < largura
            and
            0 <= p[1] < altura
        ):

            return

    # ========================================================
    # ÁREA
    # ========================================================

    area_src = area_triangulo(
        tri_src
    )

    area_dst = area_triangulo(
        tri_dst
    )

    if (
        area_src
        <
        MIN_TRIANGLE_AREA
    ):

        return

    if (
        area_dst
        <
        MIN_TRIANGLE_AREA
    ):

        return

    ratio = (
        area_dst
        /
        area_src
    )

    if (
        ratio
        <
        MIN_AREA_RATIO
        or
        ratio
        >
        MAX_AREA_RATIO
    ):

        return

    # ========================================================
    # RETÂNGULOS
    # ========================================================

    r1 = cv2.boundingRect(
        tri_src.astype(
            np.float32
        )
    )

    r2 = cv2.boundingRect(
        tri_dst.astype(
            np.float32
        )
    )

    x1, y1, w1, h1 = r1

    x2, y2, w2, h2 = r2

    if (
        w1 <= 0
        or
        h1 <= 0
        or
        w2 <= 0
        or
        h2 <= 0
    ):

        return

    if (
        x1 < 0
        or
        y1 < 0
        or
        x1 + w1 > largura
        or
        y1 + h1 > altura
    ):

        return

    if (
        x2 < 0
        or
        y2 < 0
        or
        x2 + w2 > largura
        or
        y2 + h2 > altura
    ):

        return

    # ========================================================
    # COORDENADAS RELATIVAS
    # ========================================================

    src_local = (
        tri_src
        -
        np.array(
            [
                x1,
                y1
            ],
            dtype=np.float32
        )
    )

    dst_local = (
        tri_dst
        -
        np.array(
            [
                x2,
                y2
            ],
            dtype=np.float32
        )
    )

    # ========================================================
    # RECORTE FONTE
    # ========================================================

    patch = frame[
        y1:y1 + h1,
        x1:x1 + w1
    ]

    if patch.size == 0:

        return

    # ========================================================
    # TRANSFORMAÇÃO AFIM
    # ========================================================

    matriz = cv2.getAffineTransform(
        src_local.astype(
            np.float32
        ),
        dst_local.astype(
            np.float32
        )
    )

    warped = cv2.warpAffine(
        patch,
        matriz,
        (
            w2,
            h2
        ),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101
    )

    # ========================================================
    # MÁSCARA TRIANGULAR
    # ========================================================

    mascara = np.zeros(
        (
            h2,
            w2
        ),
        dtype=np.uint8
    )

    cv2.fillConvexPoly(
        mascara,
        np.round(
            dst_local
        ).astype(
            np.int32
        ),
        255,
        lineType=cv2.LINE_AA
    )

    peso = (
        mascara.astype(
            np.float32
        )
        /
        255.0
    )

    # ========================================================
    # ACUMULA
    # ========================================================

    acumulador[
        y2:y2 + h2,
        x2:x2 + w2
    ] += (
        warped.astype(
            np.float32
        )
        *
        peso[
            ...,
            None
        ]
    )

    pesos[
        y2:y2 + h2,
        x2:x2 + w2
    ] += peso


# ============================================================
# ESCALA POLÍGONO
# ============================================================

def escalar_poligono(
    poly,
    escala
):

    centro = np.mean(
        poly,
        axis=0,
        keepdims=True
    )

    return (
        centro
        +
        (
            poly
            -
            centro
        )
        *
        escala
    )


# ============================================================
# MÁSCARA FACIAL
# ============================================================

def criar_mascara_rosto(
    pontos_2d,
    largura,
    altura
):

    mascara = np.zeros(
        (
            altura,
            largura
        ),
        dtype=np.uint8
    )

    poly = pontos_2d[
        FACE_OVAL
    ].astype(
        np.int32
    )

    cv2.fillPoly(
        mascara,
        [
            poly
        ],
        255
    )

    return mascara


# ============================================================
# MÁSCARA DOS OLHOS
# ============================================================

def criar_mascara_olhos(
    pontos_2d,
    largura,
    altura
):

    mascara = np.zeros(
        (
            altura,
            largura
        ),
        dtype=np.uint8
    )

    for indices in [
        EYE_1,
        EYE_2
    ]:

        poly = pontos_2d[
            indices
        ].astype(
            np.float32
        )

        poly = escalar_poligono(
            poly,
            EYE_OPENING_SCALE
        )

        poly = np.round(
            poly
        ).astype(
            np.int32
        )

        cv2.fillPoly(
            mascara,
            [
                poly
            ],
            255
        )

    return mascara


# ============================================================
# MÁSCARA DO LADO DIREITO
# ============================================================

def criar_mascara_lado_direito(
    pontos_2d,
    largura,
    altura
):

    centro = (
        pontos_2d[
            133
        ]
        +
        pontos_2d[
            362
        ]
    ) / 2.0

    eixo = (
        pontos_2d[
            454
        ]
        -
        pontos_2d[
            234
        ]
    )

    norma = np.linalg.norm(
        eixo
    )

    if norma < 1e-8:

        eixo = np.array(
            [
                1.0,
                0.0
            ],
            dtype=np.float32
        )

    else:

        eixo = eixo / norma

    if eixo[
        0
    ] < 0:

        eixo = -eixo

    yy, xx = np.mgrid[
        0:altura,
        0:largura
    ]

    lado = (
        (
            xx
            -
            centro[
                0
            ]
        )
        *
        eixo[
            0
        ]
        +
        (
            yy
            -
            centro[
                1
            ]
        )
        *
        eixo[
            1
        ]
    )

    mascara = np.zeros(
        (
            altura,
            largura
        ),
        dtype=np.uint8
    )

    mascara[
        lado
        >
        0
    ] = 255

    return mascara


# ============================================================
# APLICA O ESPELHAMENTO
# ============================================================

def aplicar_espelhamento(
    frame,
    pontos_2d,
    pontos_3d,
    calibracao,
    desenhar_debug=False
):

    altura, largura = (
        frame.shape[
            :2
        ]
    )

    # ========================================================
    # GEOMETRIA VIRTUAL
    # ========================================================

    (
        virtual_3d,
        escala,
        R,
        t
    ) = criar_geometria_virtual(
        pontos_3d,
        calibracao
    )

    acumulador = np.zeros(
        frame.shape,
        dtype=np.float32
    )

    pesos = np.zeros(
        (
            altura,
            largura
        ),
        dtype=np.float32
    )

    triangulos_debug = []

    # ========================================================
    # PROCESSA TRIÂNGULOS
    # ========================================================

    for tri_ids in calibracao[
        "triangulos"
    ]:

        # ----------------------------------------------------
        # TRIÂNGULO DA ESQUERDA REAL
        # ----------------------------------------------------

        tri_src = np.asarray(
            [
                pontos_2d[
                    tri_ids[
                        0
                    ]
                ],

                pontos_2d[
                    tri_ids[
                        1
                    ]
                ],

                pontos_2d[
                    tri_ids[
                        2
                    ]
                ]
            ],
            dtype=np.float32
        )

        try:

            p0 = ponto3d_para_pixel(
                virtual_3d[
                    tri_ids[
                        0
                    ]
                ],
                largura,
                altura
            )

            p1 = ponto3d_para_pixel(
                virtual_3d[
                    tri_ids[
                        1
                    ]
                ],
                largura,
                altura
            )

            p2 = ponto3d_para_pixel(
                virtual_3d[
                    tri_ids[
                        2
                    ]
                ],
                largura,
                altura
            )

        except KeyError:

            continue

        # ----------------------------------------------------
        # TRIÂNGULO VIRTUAL DA DIREITA
        # ----------------------------------------------------

        tri_dst = np.asarray(
            [
                p0,
                p1,
                p2
            ],
            dtype=np.float32
        )

        acumular_triangulo(
            frame,
            acumulador,
            pesos,
            tri_src,
            tri_dst
        )

        triangulos_debug.append(
            tri_dst
        )

    # ========================================================
    # CONSTRÓI TEXTURA PROJETADA
    # ========================================================

    projetada = frame.copy()

    valido = (
        pesos
        >
        0.001
    )

    if np.any(
        valido
    ):

        projetada[
            valido
        ] = (
            acumulador[
                valido
            ]
            /
            pesos[
                valido
            ][
                :,
                None
            ]
        ).astype(
            np.uint8
        )

    # ========================================================
    # COBERTURA DOS TRIÂNGULOS
    # ========================================================

    mascara_cobertura = np.zeros(
        (
            altura,
            largura
        ),
        dtype=np.uint8
    )

    mascara_cobertura[
        valido
    ] = 255

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (
            3,
            3
        )
    )

    mascara_cobertura = cv2.morphologyEx(
        mascara_cobertura,
        cv2.MORPH_CLOSE,
        kernel
    )

    # ========================================================
    # ROSTO
    # ========================================================

    mascara_rosto = criar_mascara_rosto(
        pontos_2d,
        largura,
        altura
    )

    mascara_destino = cv2.bitwise_and(
        mascara_cobertura,
        mascara_rosto
    )

    # ========================================================
    # SOMENTE DIREITA
    # ========================================================

    mascara_direita = (
        criar_mascara_lado_direito(
            pontos_2d,
            largura,
            altura
        )
    )

    mascara_destino = cv2.bitwise_and(
        mascara_destino,
        mascara_direita
    )

    # ========================================================
    # OLHOS ORIGINAIS
    # ========================================================

    mascara_olhos = criar_mascara_olhos(
        pontos_2d,
        largura,
        altura
    )

    mascara_destino[
        mascara_olhos
        >
        0
    ] = 0

    # ========================================================
    # ALPHA
    # ========================================================

    alpha = (
        mascara_destino.astype(
            np.float32
        )
        /
        255.0
    )

    alpha = cv2.GaussianBlur(
        alpha,
        (
            0,
            0
        ),
        FEATHER_SIGMA
    )

    # ========================================================
    # RESTRIÇÕES
    # ========================================================

    alpha[
        mascara_olhos
        >
        0
    ] = 0.0

    alpha[
        mascara_rosto
        ==
        0
    ] = 0.0

    alpha[
        mascara_direita
        ==
        0
    ] = 0.0

    alpha_3 = alpha[
        ...,
        None
    ]

    # ========================================================
    # COMPOSIÇÃO
    # ========================================================

    resultado = (
        frame.astype(
            np.float32
        )
        *
        (
            1.0
            -
            alpha_3
        )
        +
        projetada.astype(
            np.float32
        )
        *
        alpha_3
    )

    resultado = np.clip(
        resultado,
        0,
        255
    ).astype(
        np.uint8
    )

    # ========================================================
    # GARANTE OLHOS ORIGINAIS
    # ========================================================

    resultado[
        mascara_olhos
        >
        0
    ] = frame[
        mascara_olhos
        >
        0
    ]

    # ========================================================
    # DEBUG
    # ========================================================

    if desenhar_debug:

        # ----------------------------------------------------
        # TRIÂNGULOS
        # ----------------------------------------------------

        for tri in triangulos_debug:

            cv2.polylines(
                resultado,
                [
                    np.round(
                        tri
                    ).astype(
                        np.int32
                    )
                ],
                True,
                (
                    255,
                    0,
                    0
                ),
                1,
                cv2.LINE_AA
            )

        # ----------------------------------------------------
        # PONTOS DE POSE
        # ----------------------------------------------------

        for idx in POSE_ANCHORS:

            if idx >= len(
                pontos_2d
            ):

                continue

            x = int(
                pontos_2d[
                    idx,
                    0
                ]
            )

            y = int(
                pontos_2d[
                    idx,
                    1
                ]
            )

            cv2.circle(
                resultado,
                (
                    x,
                    y
                ),
                2,
                (
                    0,
                    255,
                    255
                ),
                -1
            )

    return (
        resultado,
        mascara_destino,
        mascara_olhos,
        mascara_cobertura
    )


# ============================================================
# CONFIGURAÇÃO OPENCV
# ============================================================

cv2.setUseOptimized(
    True
)


# ============================================================
# ABRE CÂMERA
# ============================================================

cap = cv2.VideoCapture(
    CAMERA_ID
)


if not cap.isOpened():

    print(
        "Erro ao abrir camera."
    )

    landmarker.close()

    raise SystemExit


# ============================================================
# TENTA REDUZIR BUFFER DA WEBCAM
# ============================================================
#
# Nem todo driver respeita esta configuração,
# mas quando funciona reduz frames antigos acumulados.
#
# ============================================================

cap.set(
    cv2.CAP_PROP_BUFFERSIZE,
    1
)


# ============================================================
# RESOLUÇÃO
# ============================================================

cap.set(
    cv2.CAP_PROP_FRAME_WIDTH,
    1280
)

cap.set(
    cv2.CAP_PROP_FRAME_HEIGHT,
    720
)


# Tenta solicitar 30 FPS
cap.set(
    cv2.CAP_PROP_FPS,
    30
)


# ============================================================
# ESTADO DO FILTRO
# ============================================================

bruto_anterior_2d = None

filtrado_anterior_2d = None

filtrado_anterior_3d = None


# ============================================================
# CALIBRAÇÃO
# ============================================================

calibracao = None


# ============================================================
# DEBUG
# ============================================================

debug = False


# ============================================================
# TEMPO MEDIAPIPE
# ============================================================

inicio = time.perf_counter()

ultimo_timestamp = 0


# ============================================================
# FPS
# ============================================================

tempo_frame_anterior = time.perf_counter()

fps_medio = None


# ============================================================
# DIAGNÓSTICO DO FILTRO
# ============================================================

movimento_global_atual = 0.0

filtro_medio_atual = SMOOTHING_PARADO


# ============================================================
# INSTRUÇÕES
# ============================================================

print()

print(
    "========================================"
)

print(
    "MASCARA FACIAL POSE-AWARE"
)

print(
    "COM FILTRO ADAPTATIVO"
)

print(
    "========================================"
)

print()

print(
    "C = calibrar / recalibrar"
)

print(
    "D = mostrar / esconder triangulos"
)

print(
    "R = remover calibracao"
)

print(
    "ESC = sair"
)

print()

print(
    "CALIBRACAO:"
)

print(
    "- fique aproximadamente de frente"
)

print(
    "- mantenha expressao neutra"
)

print(
    "- pressione C"
)

print()

print(
    "ESQUERDA DA TELA = FONTE"
)

print(
    "DIREITA DA TELA = DESTINO"
)

print()

print(
    "Filtro:"
)

print(
    "- parado = mais suavizacao"
)

print(
    "- movimento = resposta mais rapida"
)

print()


# ============================================================
# LOOP PRINCIPAL
# ============================================================

while True:

    # ========================================================
    # FPS
    # ========================================================

    agora_frame = time.perf_counter()

    dt_frame = (
        agora_frame
        -
        tempo_frame_anterior
    )

    tempo_frame_anterior = agora_frame

    if dt_frame > 0:

        fps_instantaneo = (
            1.0
            /
            dt_frame
        )

        if fps_medio is None:

            fps_medio = (
                fps_instantaneo
            )

        else:

            fps_medio = (
                FPS_SMOOTHING
                *
                fps_medio
                +
                (
                    1.0
                    -
                    FPS_SMOOTHING
                )
                *
                fps_instantaneo
            )

    else:

        fps_instantaneo = 0.0

    # ========================================================
    # TEMPO DE PROCESSAMENTO
    # ========================================================

    inicio_processamento = (
        time.perf_counter()
    )

    # ========================================================
    # CÂMERA
    # ========================================================

    ret, frame = cap.read()

    if not ret:

        break

    # ========================================================
    # SELFIE
    # ========================================================

    frame = cv2.flip(
        frame,
        1
    )

    altura, largura = (
        frame.shape[
            :2
        ]
    )

    # ========================================================
    # RGB
    # ========================================================

    rgb = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb
    )

    # ========================================================
    # TIMESTAMP
    # ========================================================

    timestamp = int(
        (
            time.perf_counter()
            -
            inicio
        )
        *
        1000
    )

    if timestamp <= ultimo_timestamp:

        timestamp = (
            ultimo_timestamp
            +
            1
        )

    ultimo_timestamp = timestamp

    # ========================================================
    # DETECÇÃO
    # ========================================================

    resultado_mp = (
        landmarker.detect_for_video(
            mp_image,
            timestamp
        )
    )

    saida = frame.copy()

    pontos_2d_atuais = None

    pontos_3d_atuais = None

    # ========================================================
    # ROSTO DETECTADO
    # ========================================================

    if resultado_mp.face_landmarks:

        face_landmarks = (
            resultado_mp.face_landmarks[
                0
            ]
        )

        # ====================================================
        # LANDMARKS BRUTOS
        # ====================================================

        (
            bruto_2d,
            bruto_3d
        ) = landmarks_para_arrays(
            face_landmarks,
            largura,
            altura
        )

        # ====================================================
        # FILTRO ADAPTATIVO
        # ====================================================

        (
            filtrado_2d,
            filtrado_3d,
            movimento_global_atual,
            filtro_medio_atual,
            pesos_filtro
        ) = aplicar_filtro_adaptativo(
            bruto_2d,
            bruto_3d,
            bruto_anterior_2d,
            filtrado_anterior_2d,
            filtrado_anterior_3d
        )

        # ====================================================
        # ATUALIZA ESTADO
        # ====================================================

        bruto_anterior_2d = (
            bruto_2d.copy()
        )

        filtrado_anterior_2d = (
            filtrado_2d.copy()
        )

        filtrado_anterior_3d = (
            filtrado_3d.copy()
        )

        pontos_2d_atuais = filtrado_2d

        pontos_3d_atuais = filtrado_3d

        # ====================================================
        # CALIBRADO
        # ====================================================

        if calibracao is not None:

            (
                saida,
                mascara_destino,
                mascara_olhos,
                mascara_cobertura
            ) = aplicar_espelhamento(
                frame,
                pontos_2d_atuais,
                pontos_3d_atuais,
                calibracao,
                desenhar_debug=debug
            )

            # =================================================
            # STATUS
            # =================================================

            cv2.putText(
                saida,
                "ESQUERDA -> DIREITA | POSE-AWARE",
                (
                    20,
                    32
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.62,
                (
                    0,
                    255,
                    255
                ),
                2,
                cv2.LINE_AA
            )

            cv2.putText(
                saida,
                "Filtro adaptativo ativo",
                (
                    20,
                    58
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.52,
                (
                    0,
                    255,
                    0
                ),
                1,
                cv2.LINE_AA
            )

            # =================================================
            # JANELAS DAS MÁSCARAS
            # =================================================

            cv2.imshow(
                "Mascara aplicada",
                mascara_destino
            )

            cv2.imshow(
                "Abertura ocular preservada",
                mascara_olhos
            )

        # ====================================================
        # NÃO CALIBRADO
        # ====================================================

        else:

            cv2.putText(
                saida,
                "Rosto neutro -> pressione C",
                (
                    20,
                    38
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.70,
                (
                    0,
                    255,
                    255
                ),
                2,
                cv2.LINE_AA
            )

    # ========================================================
    # ROSTO NÃO DETECTADO
    # ========================================================

    else:

        cv2.putText(
            saida,
            "Rosto nao detectado",
            (
                20,
                38
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.70,
            (
                0,
                0,
                255
            ),
            2,
            cv2.LINE_AA
        )

        # ----------------------------------------------------
        # Se perder o rosto, reinicia o estado temporal.
        #
        # Assim, quando o rosto reaparecer, o filtro não
        # tenta interpolar uma posição antiga.
        # ----------------------------------------------------

        bruto_anterior_2d = None

        filtrado_anterior_2d = None

        filtrado_anterior_3d = None

    # ========================================================
    # TEMPO DO FRAME
    # ========================================================

    fim_processamento = (
        time.perf_counter()
    )

    processamento_ms = (
        fim_processamento
        -
        inicio_processamento
    ) * 1000.0

    # ========================================================
    # DIAGNÓSTICO
    # ========================================================

    y_info = 90

    if fps_medio is not None:

        texto_fps = (
            f"FPS: {fps_medio:.1f}"
        )

    else:

        texto_fps = (
            "FPS: --"
        )

    cv2.putText(
        saida,
        texto_fps,
        (
            20,
            y_info
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        (
            255,
            255,
            255
        ),
        1,
        cv2.LINE_AA
    )

    y_info += 23

    cv2.putText(
        saida,
        (
            f"Movimento global: "
            f"{movimento_global_atual:.4f}"
        ),
        (
            20,
            y_info
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        (
            255,
            255,
            255
        ),
        1,
        cv2.LINE_AA
    )

    y_info += 23

    cv2.putText(
        saida,
        (
            f"Filtro atual: "
            f"{filtro_medio_atual:.2f}"
        ),
        (
            20,
            y_info
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        (
            255,
            255,
            255
        ),
        1,
        cv2.LINE_AA
    )

    y_info += 23

    cv2.putText(
        saida,
        (
            f"Processamento: "
            f"{processamento_ms:.1f} ms"
        ),
        (
            20,
            y_info
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        (
            255,
            255,
            255
        ),
        1,
        cv2.LINE_AA
    )

    # ========================================================
    # INDICADOR VISUAL DO FILTRO
    # ========================================================
    #
    # Quanto menor o filtro:
    #
    # mais rapidamente a máscara acompanha o rosto.
    #
    # ========================================================

    if filtro_medio_atual > 0.55:

        estado_filtro = (
            "ESTAVEL"
        )

    elif filtro_medio_atual > 0.25:

        estado_filtro = (
            "INTERMEDIARIO"
        )

    else:

        estado_filtro = (
            "RESPOSTA RAPIDA"
        )

    y_info += 23

    cv2.putText(
        saida,
        (
            f"Estado: "
            f"{estado_filtro}"
        ),
        (
            20,
            y_info
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        (
            0,
            255,
            255
        ),
        1,
        cv2.LINE_AA
    )

    # ========================================================
    # JANELA PRINCIPAL
    # ========================================================

    cv2.imshow(
        "Terapia Espelho Facial - Adaptativo",
        saida
    )

    # ========================================================
    # TECLADO
    # ========================================================

    tecla = (
        cv2.waitKey(
            1
        )
        &
        0xFF
    )

    # ========================================================
    # ESC
    # ========================================================

    if tecla == 27:

        break

    # ========================================================
    # C = CALIBRAR
    # ========================================================

    elif tecla in [
        ord(
            "c"
        ),
        ord(
            "C"
        )
    ]:

        if (
            pontos_2d_atuais
            is not None
            and
            pontos_3d_atuais
            is not None
        ):

            calibracao = calibrar(
                pontos_2d_atuais,
                pontos_3d_atuais,
                largura,
                altura
            )

    # ========================================================
    # D = DEBUG
    # ========================================================

    elif tecla in [
        ord(
            "d"
        ),
        ord(
            "D"
        )
    ]:

        debug = (
            not debug
        )

        print(
            "Debug:",
            "ATIVADO"
            if debug
            else
            "DESATIVADO"
        )

    # ========================================================
    # R = RESET DA CALIBRAÇÃO
    # ========================================================

    elif tecla in [
        ord(
            "r"
        ),
        ord(
            "R"
        )
    ]:

        calibracao = None

        print(
            "Calibracao removida."
        )


# ============================================================
# FINALIZA
# ============================================================

cap.release()

cv2.destroyAllWindows()

landmarker.close()