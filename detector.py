from ultralytics import YOLO
from PIL import Image, ImageFilter
import numpy as np
import os
import torch

# =====================================================
# DEVICE
# =====================================================
device = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print(f"[DEVICE]: {device}")

# =====================================================
# MODELO YOLO SEGMENTATION
# =====================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(
    BASE_DIR,
    "yolo",
    "model_oxford_seg.pt"
)

if not os.path.exists(MODEL_PATH):

    raise FileNotFoundError(
        f"Modelo não encontrado: {MODEL_PATH}"
    )

# =====================================================
# CARREGA YOLO
# =====================================================
model = YOLO(MODEL_PATH)

print("[YOLO]: Modelo carregado")

# =====================================================
# CLASSES VÁLIDAS
# =====================================================
CLASSES_VALIDAS = [
    "PRATO",
    "XICARA",
    "CANECA"
]

# =====================================================
# DETECTAR OBJETO
# =====================================================
def detectar_prato(img):

    # =================================================
    # PREDIÇÃO
    # =================================================
    # "cpu"
    # conf=0.40, confiabilidade, para nã oidentificar qualquer coisa
    results = model.predict(
        img,
        conf=0.30,
        verbose=False,
        device=device
    )

    # =================================================
    # SEM RESULTADO
    # =================================================
    if not results:

        print("[YOLO]: Nenhum resultado")

        return None

    r = results[0]

    # =================================================
    # SEM MÁSCARA
    # =================================================
    if r.masks is None:

        print("[YOLO]: Nenhuma máscara encontrada")
        print("================================")
        print("TOTAL BOXES:", len(r.boxes))

        for i, box in enumerate(r.boxes):

            cls = int(box.cls[0])

            nome = model.names[cls]

            conf = float(box.conf[0])

            print(
                f"[DETECÇÃO {i}] "
                f"{nome} "
                f"(conf={conf:.3f})"
            )

        print("MASKS:", r.masks)
        print("================================")
        
        return None

    masks = r.masks.data.cpu().numpy()

    boxes = r.boxes

    # =================================================
    # SOMENTE OBJETOS VÁLIDOS
    # =================================================
    deteccoes_validas = []

    for i, box in enumerate(boxes):

        cls = int(box.cls[0])

        nome = model.names[cls]

        conf = float(box.conf[0])

        print(
            f"[YOLO ENXERGOU]: "
            f"{nome} "
            f"(conf={conf:.2f})"
        )

        # =============================================
        # IGNORA QUALQUER OUTRA COISA
        # =============================================
        if nome not in CLASSES_VALIDAS:

            print(
                f"[IGNORADO]: {nome}"
            )

            continue

        # =============================================
        # ÁREA DA MÁSCARA
        # =============================================
        area = masks[i].sum()

        deteccoes_validas.append({

            "idx": i,

            "classe": nome,

            "conf": conf,

            "area": area
        })

    # =================================================
    # NENHUM OBJETO VÁLIDO
    # =================================================
    if len(deteccoes_validas) == 0:

        print(
            "[YOLO]: "
            "Nenhum plate/bowl/cup/mug/vase encontrado"
        )

        return None

    # =================================================
    # ESCOLHE MAIOR ÁREA
    # =================================================
    melhor = max(
        deteccoes_validas,
        key=lambda x: x["area"]
    )

    melhor_idx = melhor["idx"]

    print(
        f"[YOLO DETECTOU]: "
        f"{melhor['classe']} "
        f"(conf={melhor['conf']:.2f})"
    )

    # =================================================
    # MÁSCARA
    # =================================================
    mask = masks[melhor_idx]

    # =================================================
    # IMG -> NUMPY
    # =================================================
    img_np = np.array(
        img.convert("RGB")
    )

    h, w = img_np.shape[:2]

    # =================================================
    # RESIZE MÁSCARA
    # =================================================
    mask_img = Image.fromarray(
        (mask * 255).astype(np.uint8)
    )

    mask_img = mask_img.resize(
        (w, h),
        Image.Resampling.LANCZOS
    )

    # =================================================
    # BORDA SUAVE
    # =================================================
    mask_img = mask_img.filter(
        ImageFilter.GaussianBlur(radius=1)
    )

    # =================================================
    # FLOAT
    # =================================================
    mask = (
        np.array(mask_img)
        .astype(np.float32)
        / 255.0
    )

    # =================================================
    # 3 CANAIS
    # =================================================
    mask_3d = np.stack(
        [mask, mask, mask],
        axis=-1
    )

    # =================================================
    # SEGMENTAÇÃO
    # =================================================
    seg = (
        img_np.astype(np.float32)
        * mask_3d
    )

    # =================================================
    # FUNDO PRETO // ( OBS: **BRANCO RETIRADO**)
    # =================================================
    """background = np.ones_like(
        seg,
        dtype=np.float32
    ) * 255

    final = (
        seg +
        background * (1 - mask_3d)
    )"""
    final = seg.astype(np.uint8)

    final = np.clip(
        final,
        0,
        255
    ).astype(np.uint8)

    # =================================================
    # PIL FINAL
    # =================================================
    img_final = Image.fromarray(final)

    print(
        "[YOLO]: Segmentação concluída"
    )

    return img_final