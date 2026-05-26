# model.py
import torch
from transformers import AutoImageProcessor, AutoModel
from PIL import Image
import numpy as np

# Configuração de hardware dinâmica
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Carrega o processador e o modelo DINOv2 da Meta (versão Base)
# O DINOv2 é excelente para extrair texturas, formas e geometrias finas
PROCESSOR_NAME = "facebook/dinov2-base"
processor = AutoImageProcessor.from_pretrained(PROCESSOR_NAME)
model = AutoModel.from_pretrained(PROCESSOR_NAME).to(device)
model.eval()  # Modo de inferência (desativa dropout)

def gerar_embedding_dinov2(img: Image.Image) -> np.ndarray:
    """
    Gera um embedding normalizado de 768 dimensões focado em geometria e forma.
    """
    # Garante que a imagem está em RGB
    img_rgb = img.convert("RGB")
    
    # Pré-processamento oficial do modelo
    inputs = processor(images=img_rgb, return_tensors="pt").to(device)
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Extraímos o CLS token (índice 0), que representa a assinatura global da imagem
    embedding = outputs.last_hidden_state[:, 0, :]
    
    # Normalização L2: Garante que a busca por Distância de Cosseno seja ultra precisa
    embedding = torch.nn.functional.normalize(embedding, p=2, dim=1)
    
    return embedding[0].cpu().numpy()