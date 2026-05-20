import requests
import base64

from io import BytesIO

from PIL import (
    Image,
    ImageFile
)

# =====================================================
# EVITA ERRO COM IMAGEM PARCIAL
# =====================================================
ImageFile.LOAD_TRUNCATED_IMAGES = True

# =====================================================
# CONFIG
# =====================================================
MAX_IMAGE_SIZE_MB = 15

ALLOWED_FORMATS = [
    "jpeg",
    "jpg",
    "png",
    "webp",
    "bmp"
]

# =====================================================
# VALIDAR IMAGEM
# =====================================================
def validar_imagem(image_data: bytes):

    # ================================================
    # VAZIO
    # ================================================
    if not image_data:

        print("❌ Conteúdo vazio")

        return False

    # ================================================
    # TAMANHO
    # ================================================
    tamanho_mb = len(image_data) / (1024 * 1024)

    if tamanho_mb > MAX_IMAGE_SIZE_MB:

        print(
            f"❌ Imagem muito grande: "
            f"{tamanho_mb:.2f} MB"
        )

        return False

    # ================================================
    # FORMATO
    # ================================================
    try:

        img_test = Image.open(
            BytesIO(image_data)
        )

        formato = (
            img_test.format
            .lower()
        )

    except Exception:

        print("❌ Formato inválido")

        return False

    # ================================================
    # VALIDA FORMATO
    # ================================================
    if formato not in ALLOWED_FORMATS:

        print(
            f"❌ Formato inválido: {formato}"
        )

        return False

    return True

# =====================================================
# NORMALIZA IMAGEM
# =====================================================
def normalizar_imagem(img: Image.Image):

    try:

        # ============================================
        # RGB
        # ============================================
        img = img.convert("RGB")

        # ============================================
        # REMOVE EXIF ORIENTATION
        # ============================================
        try:

            exif = img.getexif()

            orientation = exif.get(274)

            if orientation == 3:

                img = img.rotate(
                    180,
                    expand=True
                )

            elif orientation == 6:

                img = img.rotate(
                    270,
                    expand=True
                )

            elif orientation == 8:

                img = img.rotate(
                    90,
                    expand=True
                )

        except Exception:
            pass

        return img

    except Exception as e:

        print(
            f"❌ Erro ao normalizar imagem: "
            f"{str(e)}"
        )

        return None

# =====================================================
# URL (ROBUSTO / PRODUÇÃO)
# =====================================================
def load_image_from_url(url: str):

    try:

        # ============================================
        # URL VAZIA
        # ============================================
        if not url:

            print("❌ URL vazia")

            return None

        headers = {

            "User-Agent": (
                "Mozilla/5.0 "
                "(Windows NT 10.0; Win64; x64)"
            ),

            "Accept": (
                "image/webp,image/apng,image/*,*/*;q=0.8"
            )
        }

        # ============================================
        # DOWNLOAD
        # ============================================
        response = requests.get(
            url,
            headers=headers,
            timeout=15,
            stream=True
        )

        # ============================================
        # STATUS
        # ============================================
        if response.status_code != 200:

            print(
                f"❌ HTTP inválido: "
                f"{response.status_code}"
            )

            return None

        # ============================================
        # CONTENT TYPE
        # ============================================
        content_type = response.headers.get(
            "Content-Type",
            ""
        )

        if "image" not in content_type.lower():

            print(
                f"❌ URL não retornou imagem: "
                f"{content_type}"
            )

            return None

        # ============================================
        # BYTES
        # ============================================
        image_data = response.content

        # ============================================
        # VALIDAÇÃO
        # ============================================
        if not validar_imagem(image_data):

            return None

        # ============================================
        # PIL
        # ============================================
        img = Image.open(
            BytesIO(image_data)
        )

        img = normalizar_imagem(img)

        if img is None:

            return None

        print(
            f"✅ Imagem carregada via URL "
            f"({img.width}x{img.height})"
        )

        return img

    except requests.exceptions.Timeout:

        print("❌ Timeout ao baixar imagem")

        return None

    except requests.exceptions.ConnectionError:

        print("❌ Erro de conexão")

        return None

    except requests.exceptions.RequestException as e:

        print(
            f"❌ Falha na requisição: "
            f"{str(e)}"
        )

        return None

    except Exception as e:

        print(
            f"❌ Erro ao processar imagem: "
            f"{str(e)}"
        )

        return None

# =====================================================
# BASE64 (ROBUSTO / PRODUÇÃO)
# =====================================================
def load_image_from_base64(base64_string: str):

    try:

        # ============================================
        # VAZIO
        # ============================================
        if not base64_string:

            print("❌ Base64 vazio")

            return None

        # ============================================
        # REMOVE DATA URI
        # ============================================
        if "," in base64_string:

            base64_string = (
                base64_string.split(",")[1]
            )

        # ============================================
        # DECODE
        # ============================================
        image_data = base64.b64decode(
            base64_string,
            validate=True
        )

        # ============================================
        # VALIDAÇÃO
        # ============================================
        if not validar_imagem(image_data):

            return None

        # ============================================
        # PIL
        # ============================================
        img = Image.open(
            BytesIO(image_data)
        )

        img = normalizar_imagem(img)

        if img is None:

            return None

        print(
            f"✅ Imagem Base64 carregada "
            f"({img.width}x{img.height})"
        )

        return img

    except base64.binascii.Error:

        print("❌ Base64 inválido")

        return None

    except Exception as e:

        print(
            f"❌ Erro ao processar Base64: "
            f"{str(e)}"
        )

        return None
        
# =====================================================
# PIL IMAGE -> BASE64
# =====================================================

import io
import base64

def image_to_base64(img):

    buffer = io.BytesIO()

    img.save(
        buffer,
        format="PNG"
    )

    img_bytes = buffer.getvalue()

    base64_str = base64.b64encode(
        img_bytes
    ).decode("utf-8")

    return f"data:image/png;base64,{base64_str}"