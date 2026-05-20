from pydantic import BaseModel, Field
from typing import Optional

# =====================================================
# ANÁLISE BASE64
# =====================================================
class ImageRequest(BaseModel):

    image_base64: str = Field(
        ...,
        description="Imagem em Base64"
    )

# =====================================================
# ANÁLISE URL
# =====================================================
class UrlImageRequest(BaseModel):

    image_url: str = Field(
        ...,
        description="URL pública da imagem"
    )

# =====================================================
# TREINAMENTO
# =====================================================
class TrainRequest(BaseModel):

    # ================================================
    # URL OPCIONAL
    # ================================================
    image_url: Optional[str] = Field(
        default=None,
        description="URL pública da imagem"
    )

    # ================================================
    # BASE64 OPCIONAL
    # ================================================
    image_base64: Optional[str] = Field(
        default=None,
        description="Imagem em Base64"
    )

    # ================================================
    # CATEGORIA
    # ================================================
    categoria: str = Field(
        ...,
        min_length=1,
        max_length=100,
        description="Categoria da decoração/produto"
    )

    # ================================================
    # MODELO DINOv2
    # ================================================
    embedding_model: str = Field(
        default="dinov2-base",
        description="Modelo de embedding utilizado"
    )

    # ================================================
    # AUGMENTATIONS
    # ================================================
    augmentations: int = Field(
        default=20,
        ge=1,
        le=100,
        description="Quantidade de imagens augmentadas"
    )