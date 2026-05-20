from transformers import AutoImageProcessor, AutoModel
import torch
import numpy as np
from PIL import Image

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
# DINOv2
# =====================================================
processor = AutoImageProcessor.from_pretrained(
    "facebook/dinov2-base"
)

model = AutoModel.from_pretrained(
    "facebook/dinov2-base"
)

model.to(device)

model.eval()

print("[DINOv2]: Modelo carregado")

# =====================================================
# GERA EMBEDDING DINOv2
# =====================================================
def gerar_embedding(image: Image.Image):

    # ============================================
    # RGB
    # ============================================
    image = image.convert("RGB")

    # ============================================
    # PROCESSA IMAGEM
    # ============================================
    inputs = processor(
        images=image,
        return_tensors="pt"
    ).to(device)

    # ============================================
    # INFERÊNCIA
    # ============================================
    with torch.no_grad():

        outputs = model(**inputs)

        # ========================================
        # POOLING GLOBAL
        # ========================================
        embedding = outputs.last_hidden_state.mean(dim=1)

    # ============================================
    # CONVERTE PARA NUMPY
    # ============================================
    embedding = (
        embedding[0]
        .cpu()
        .numpy()
    )

    # ============================================
    # NORMALIZAÇÃO L2
    # ============================================
    embedding = (
        embedding /
        np.linalg.norm(embedding)
    )

    return embedding