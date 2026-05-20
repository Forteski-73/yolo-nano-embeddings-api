from ultralytics import YOLO
from PIL import Image
from rembg import remove
import os

# =====================================================
# MODELO YOLO-WORLD
# =====================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "yolo", "yolov8x-seg.pt")

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Modelo YOLO-WORLD não encontrado: {MODEL_PATH}")

model = YOLO(MODEL_PATH)

# classes alvo
CLASSES_VALIDAS = ["plate", "bowl", "cup", "mug", "vase"]
model.set_classes(CLASSES_VALIDAS)


# =====================================================
# DETECTAR OBJETO PRINCIPAL
# =====================================================
def detectar_prato(img):

    results = model.predict(
        img,
        conf=0.40,   # 🔥 mais preciso (menos ruído que 0.15)
        verbose=False
    )

    w_img, h_img = img.size
    centro_x, centro_y = w_img / 2, h_img / 2

    melhor_score = -1
    melhor_crop = None

    # =================================================
    # DETECÇÕES
    # =================================================
    for r in results:

        if r.boxes is None:
            continue

        for box in r.boxes:

            cls = int(box.cls[0])
            nome = model.names[cls]

            print(f"[YOLO ENXERGOU]: 🔍 {nome}")

            if nome not in CLASSES_VALIDAS:
                continue

            print(f"[YOLO DETECTOU]: ✅ {nome}")

            x1, y1, x2, y2 = map(int, box.xyxy[0])

            largura = x2 - x1
            altura = y2 - y1

            area = largura * altura

            box_cx = (x1 + x2) / 2
            box_cy = (y1 + y2) / 2

            distancia = ((box_cx - centro_x) ** 2 + (box_cy - centro_y) ** 2) ** 0.5

            # 🔥 score melhor equilibrado
            score = area - (distancia * 1.5)

            if score > melhor_score:

                melhor_score = score

                # =================================================
                # 🔥 PADDING INTELIGENTE (IMPORTANTE)
                # =================================================
                padding = int(0.07 * max(largura, altura))  # ótimo equilíbrio

                x1p = max(0, x1 - padding)
                y1p = max(0, y1 - padding)
                x2p = min(w_img, x2 + padding)
                y2p = min(h_img, y2 + padding)

                melhor_crop = img.crop((x1p, y1p, x2p, y2p))

    # =================================================
    # FALLBACK (CENTRO DA IMAGEM)
    # =================================================
    if melhor_crop is None:

        left = int(w_img * 0.15)
        top = int(h_img * 0.15)
        right = int(w_img * 0.85)
        bottom = int(h_img * 0.85)

        melhor_crop = img.crop((left, top, right, bottom))

    # =================================================
    # REMOVE FUNDO
    # =================================================
    img_sem_fundo = remove(melhor_crop)

    if img_sem_fundo.mode != "RGBA":
        img_sem_fundo = img_sem_fundo.convert("RGBA")

    # =================================================
    # FUNDO BRANCO FINAL
    # =================================================
    fundo = Image.new("RGB", img_sem_fundo.size, (255, 255, 255))
    fundo.paste(img_sem_fundo, mask=img_sem_fundo.split()[3])

    return fundo