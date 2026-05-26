# =====================================================
# similarity.py
# =====================================================

import json
import cv2
import numpy as np

from sklearn.metrics.pairwise import cosine_similarity

from db import get_conn
from model import gerar_embedding

from features import (
    hsv_histogram,
    texture_lbp,
    edge_density,
    symmetry_score
)

# =====================================================
# CONFIGURAÇÕES IA
# =====================================================

LIMITE_CONFIANCA = 0.72
LIMITE_EMBEDDING = 0.70

LIMITE_HSV = 0.40
LIMITE_HSV_PENALIDADE = 0.55

DIFERENCA_MINIMA = 0.020

# =====================================================
# PESOS (HSV mais forte e dominante)
# =====================================================

PESO_EMBEDDING = 0.25
PESO_HSV       = 0.65
PESO_LBP       = 0.05
PESO_EDGE      = 0.03
PESO_SYMMETRY  = 0.02


# =====================================================
# AUXILIAR
# =====================================================

def gerar_percentual(score_porcentagem):
    return f"{score_porcentagem:.1f}%"


def normalize_vector(vec, size=36):
    """
    Garante compatibilidade entre versões antigas e novas do banco.
    Evita crash de shape mismatch (768 vs 36 etc).
    """
    vec = np.array(vec, dtype=np.float32).flatten()

    if len(vec) != size:
        fixed = np.zeros(size, dtype=np.float32)
        min_len = min(len(vec), size)
        fixed[:min_len] = vec[:min_len]
        vec = fixed

    vec = vec / (np.sum(vec) + 1e-6)
    return vec


# =====================================================
# CALCULAR SIMILARIDADE
# =====================================================

def calcular_similaridade(img):

    if img is None:
        return {
            "status": "invalid_object",
            "msg": (
                "NENHUM PRATO, TIGELA, XÍCARA, "
                "CANECA OU VASO FOI ENCONTRADO"
            )
        }

    print("\n================================================")
    print("🔍 INICIANDO ANÁLISE DE SIMILARIDADE")
    print("================================================")

    # =================================================
    # FEATURES TESTE
    # =================================================

    emb_teste = gerar_embedding(img).reshape(1, -1)

    hsv_teste = normalize_vector(
        hsv_histogram(img),
        size=36
    )

    lbp_teste = np.array(
        texture_lbp(img)
    ).reshape(1, -1)

    edge_teste = edge_density(img)
    sym_teste = symmetry_score(img)

    # =================================================
    # BANCO
    # =================================================

    conn = get_conn()
    cursor = conn.cursor()

    cursor.execute("""
        SELECT
            id,
            categoria,
            embedding_model,
            embedding,
            hsv_histogram,
            texture_lbp,
            edge_density,
            symmetry_score
        FROM image_embeddings
    """)

    rows = cursor.fetchall()

    categorias_scores = {}

    # =================================================
    # LOOP
    # =================================================

    for r in rows:

        (
            id_,
            categoria,
            embedding_model,
            embedding_json,
            hsv_json,
            lbp_json,
            edge_db,
            sym_db
        ) = r

        try:

            # =========================================
            # EMBEDDING
            # =========================================

            emb_db = np.array(
                json.loads(embedding_json)
            ).reshape(1, -1)

            sim_embedding = cosine_similarity(
                emb_teste,
                emb_db
            )[0][0]

            # =========================================
            # HSV (ROBUSTO)
            # =========================================

            hsv_db = normalize_vector(
                json.loads(hsv_json),
                size=36
            )

            sim_hsv = 1.0 - cv2.compareHist(
                hsv_teste.astype(np.float32),
                hsv_db.astype(np.float32),
                cv2.HISTCMP_BHATTACHARYYA
            )

            sim_hsv = float(max(0.0, sim_hsv))

            # =========================================
            # PENALIZAÇÃO FORTE DE COR
            # =========================================

            color_penalty = 1.0

            if sim_hsv < LIMITE_HSV:
                color_penalty *= 0.20  # quase elimina
            elif sim_hsv < LIMITE_HSV_PENALIDADE:
                color_penalty *= 0.60

            # =========================================
            # LBP
            # =========================================

            if lbp_json:
                lbp_db = np.array(
                    json.loads(lbp_json)
                ).reshape(1, -1)

                sim_lbp = cosine_similarity(
                    lbp_teste,
                    lbp_db
                )[0][0]
            else:
                sim_lbp = 0.5

            # =========================================
            # EDGE
            # =========================================

            edge_db = float(edge_db or 0)
            sim_edge = max(0, 1 - abs(edge_teste - edge_db))

            # =========================================
            # SYMMETRY
            # =========================================

            sym_db = float(sym_db or 0)
            sim_sym = max(0, 1 - abs(sym_teste - sym_db))

            # =========================================
            # SCORE FINAL
            # =========================================

            score_final = (
                sim_embedding * PESO_EMBEDDING +
                sim_hsv       * PESO_HSV +
                sim_lbp       * PESO_LBP +
                sim_edge      * PESO_EDGE +
                sim_sym       * PESO_SYMMETRY
            )

            # aplica penalização de cor
            score_final *= color_penalty

            # penalização de embedding fraco
            if sim_embedding < 0.60:
                score_final *= 0.70

            # =========================================
            # LOG
            # =========================================

            print("\n================================================")
            print(f"Categoria   : {categoria}")
            print(f"Embedding   : {sim_embedding:.4f}")
            print(f"HSV         : {sim_hsv:.4f}")
            print(f"LBP         : {sim_lbp:.4f}")
            print(f"Edge        : {sim_edge:.4f}")
            print(f"Symmetry    : {sim_sym:.4f}")
            print(f"Score final : {score_final:.4f}")

            # =========================================
            # MELHOR POR CATEGORIA
            # =========================================

            if (
                categoria not in categorias_scores
                or score_final > categorias_scores[categoria]["score"]
            ):

                categorias_scores[categoria] = {
                    "score": score_final,
                    "embedding": sim_embedding,
                    "hsv": sim_hsv,
                    "lbp": sim_lbp,
                    "edge": sim_edge,
                    "symmetry": sim_sym,
                    "id": id_,
                    "model": embedding_model
                }

        except Exception as e:
            print(f"[ERRO]: {categoria}")
            print(str(e))
            continue

    cursor.close()
    conn.close()

    # =================================================
    # SEM RESULTADO
    # =================================================

    if not categorias_scores:
        return {
            "status": "not_found",
            "msg": "NENHUMA DECORAÇÃO COMPATÍVEL FOI ENCONTRADA"
        }

    # =================================================
    # RANKING
    # =================================================

    ranking = sorted(
        categorias_scores.items(),
        key=lambda x: x[1]["score"],
        reverse=True
    )

    melhor_categoria = ranking[0][0]
    melhor_score = ranking[0][1]["score"]

    melhor_embedding = ranking[0][1]["embedding"]
    melhor_hsv = ranking[0][1]["hsv"]
    melhor_lbp = ranking[0][1]["lbp"]
    melhor_edge = ranking[0][1]["edge"]
    melhor_sym = ranking[0][1]["symmetry"]
    melhor_id = ranking[0][1]["id"]

    segundo_score = ranking[1][1]["score"] if len(ranking) > 1 else 0

    diferenca = melhor_score - segundo_score

    # =================================================
    # RESULTADO FINAL
    # =================================================

    print("\n================================================")
    print("🏆 RESULTADO FINAL")
    print("================================================")

    print(f"CATEGORIA    : {melhor_categoria}")
    print(f"SCORE        : {melhor_score:.4f}")
    print(f"EMBEDDING    : {melhor_embedding:.4f}")
    print(f"HSV          : {melhor_hsv:.4f}")
    print(f"LBP          : {melhor_lbp:.4f}")
    print(f"EDGE         : {melhor_edge:.4f}")
    print(f"SYMMETRY     : {melhor_sym:.4f}")
    print(f"DIFERENÇA    : {diferenca:.4f}")

    return {
        "status": "ok",
        "categoria": melhor_categoria,

        "similaridade": float(melhor_score * 100),
        "confianca": float(melhor_score),

        "similaridade_percentual": float(melhor_score * 100),

        "embedding": float(melhor_embedding),
        "hsv": float(melhor_hsv),
        "lbp": float(melhor_lbp),
        "edge": float(melhor_edge),
        "symmetry": float(melhor_sym),

        "embedding_id": melhor_id,
        "difference": float(diferenca)
    }