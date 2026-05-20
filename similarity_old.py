# =====================================================
# similarity.py
# =====================================================

import json
import numpy as np

from sklearn.metrics.pairwise import cosine_similarity

from db import get_conn

from model import gerar_embedding

from detector import detectar_prato

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

LIMITE_EMBEDDING = 0.72

DIFERENCA_MINIMA = 0.015

# =====================================================
# PESOS
# =====================================================
PESO_EMBEDDING = 0.45
PESO_HSV       = 0.35
PESO_LBP       = 0.15
PESO_EDGE      = 0.025
PESO_SYMMETRY  = 0.025


# =====================================================
# FUNÇÃO AUXILIAR: GERAR RESPOSTA AMIGÁVEL
# =====================================================
def gerar_percentual(categoria, score_porcentagem):
    return f"{score_porcentagem:.1f}%"


# =====================================================
# CALCULAR SIMILARIDADE
# =====================================================
def calcular_similaridade(img):

    # =================================================
    # SEGMENTAÇÃO
    # =================================================
    # img = detectar_prato(img)

    # =================================================
    # OBJETO NÃO ENCONTRADO
    # =================================================
    if img is None:

        return {
            "status": "invalid_object",
            "msg": (
                "NENHUM PRATO, TIGELA, XÍCARA, CANECA "
                "OU VASO FOI ENCONTRADO NA IMAGEM"
            )
        }

    print("\n======================")
    print("IMAGEM PROCESSADA PELO YOLO")

    # =================================================
    # EMBEDDING DINOv2
    # =================================================
    emb_teste = gerar_embedding(
        img
    ).reshape(1, -1)

    # =================================================
    # HSV
    # =================================================
    hsv_teste = np.array(
        hsv_histogram(img)
    ).reshape(1, -1)

    # =================================================
    # LBP
    # =================================================
    lbp_teste = np.array(
        texture_lbp(img)
    ).reshape(1, -1)

    # =================================================
    # EDGE
    # =================================================
    edge_teste = edge_density(img)

    # =================================================
    # SYMMETRY
    # =================================================
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

    # =================================================
    # MELHOR SCORE POR CATEGORIA
    # =================================================
    categorias_scores = {}

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
            # HSV
            # =========================================
            hsv_db = np.array(
                json.loads(hsv_json)
            ).reshape(1, -1)

            sim_hsv = cosine_similarity(
                hsv_teste,
                hsv_db
            )[0][0]

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

            edge_diff = abs(
                edge_teste - edge_db
            )

            sim_edge = max(
                0,
                1 - edge_diff
            )

            # =========================================
            # SYMMETRY
            # =========================================
            sym_db = float(sym_db or 0)

            sym_diff = abs(
                sym_teste - sym_db
            )

            sim_sym = max(
                0,
                1 - sym_diff
            )

            # =========================================
            # SCORE FINAL
            # =========================================
            score_final = (

                sim_embedding * PESO_EMBEDDING +

                sim_hsv * PESO_HSV +

                sim_lbp * PESO_LBP +

                sim_edge * PESO_EDGE +

                sim_sym * PESO_SYMMETRY
            )

            print("\n======================")

            print(f"Categoria      : {categoria}")

            print(f"Modelo         : {embedding_model}")

            print(f"EMBEDDING      : {sim_embedding:.4f}")

            print(f"HSV            : {sim_hsv:.4f}")

            print(f"LBP            : {sim_lbp:.4f}")

            print(f"EDGE           : {sim_edge:.4f}")

            print(f"SYMMETRY       : {sim_sym:.4f}")

            print(f"SCORE FINAL    : {score_final:.4f}")

            # =========================================
            # MELHOR SCORE
            # =========================================
            if categoria not in categorias_scores:

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

            else:

                if (
                    score_final >
                    categorias_scores[categoria]["score"]
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

    cursor.close()

    conn.close()

    # =================================================
    # SEM RESULTADO
    # =================================================
    if not categorias_scores:

        return {
            "status": "not_found",
            "msg": "Nenhum resultado encontrado"
        }

    # =================================================
    # RANKING
    # =================================================
    ranking = sorted(
        categorias_scores.items(),
        key=lambda x: x[1]["score"],
        reverse=True
    )

    # =================================================
    # MELHOR RESULTADO
    # =================================================
    melhor_categoria = ranking[0][0]

    melhor_score = ranking[0][1]["score"]

    melhor_embedding = ranking[0][1]["embedding"]

    melhor_hsv = ranking[0][1]["hsv"]

    melhor_lbp = ranking[0][1]["lbp"]

    melhor_edge = ranking[0][1]["edge"]

    melhor_sym = ranking[0][1]["symmetry"]

    melhor_id = ranking[0][1]["id"]

    # =================================================
    # SEGUNDO SCORE
    # =================================================
    if len(ranking) > 1:

        segundo_score = ranking[1][1]["score"]

    else:

        segundo_score = 0

    # =================================================
    # DIFERENÇA
    # =================================================
    diferenca = (
        melhor_score - segundo_score
    )

    print("\n======================")

    print(f"MELHOR SCORE : {melhor_score:.4f}")

    print(f"SEGUNDO SCORE: {segundo_score:.4f}")

    print(f"DIFERENCA    : {diferenca:.4f}")

    print(f"EMBEDDING    : {melhor_embedding:.4f}")

    print(f"HSV          : {melhor_hsv:.4f}")

    print(f"LBP          : {melhor_lbp:.4f}")

    print(f"EDGE         : {melhor_edge:.4f}")

    print(f"SYMMETRY     : {melhor_sym:.4f}")

    print(f"CATEGORIA    : {melhor_categoria}")


    # =================================================
    # SIMILARIDADE BAIXA
    # =================================================
    if (
        melhor_score < LIMITE_CONFIANCA
        and melhor_embedding < LIMITE_EMBEDDING
    ):
        score_pct = float(melhor_score * 100)
        return {
            "status": "similar",
            "msg": "DECORAÇÃO MAIS PARECIDA ENCONTRADA",
            "categoria": melhor_categoria,
            "percentual": gerar_percentual(melhor_categoria, score_pct),

            # =========================================
            # PERCENTUAL DE SIMILARIDADE
            # =========================================
            "similaridade": score_pct,

            "confianca": score_pct,

            "embedding": float(melhor_embedding * 100),

            "hsv": float(melhor_hsv * 100),

            "lbp": float(melhor_lbp * 100),

            "edge": float(melhor_edge * 100),

            "symmetry": float(melhor_sym * 100),

            "embedding_id": melhor_id
        }

    # =================================================
    # INCONCLUSIVO
    # =================================================
    if (
        diferenca < DIFERENCA_MINIMA
        and melhor_score < 0.80
    ):
        score_pct = float(melhor_score * 100)
        return {
            "status": "inconclusive",

            "msg": "DECORAÇÃO MAIS PARECIDA ENCONTRADA",

            "categoria": melhor_categoria,
            "percentual": gerar_percentual(melhor_categoria, score_pct),

            "similaridade": score_pct,

            "confianca": score_pct,

            "embedding": float(melhor_embedding * 100),

            "hsv": float(melhor_hsv * 100),

            "lbp": float(melhor_lbp * 100),

            "edge": float(melhor_edge * 100),

            "symmetry": float(melhor_sym * 100),

            "embedding_id": melhor_id
        }

    # =================================================
    # ENCONTRADO (Sucesso - Caso de Uso do seu JSON de exemplo)
    # =================================================
    score_pct = float(melhor_score * 100)
    return {
        "status": "ok",

        "msg": "DECORAÇÃO ENCONTRADA",

        "categoria": melhor_categoria,
        "percentual": gerar_percentual(melhor_categoria, score_pct),

        # =============================================
        # PERCENTUAL
        # =============================================
        "similaridade": score_pct,

        "confianca": score_pct,

        "embedding": float(melhor_embedding * 100),

        "hsv": float(melhor_hsv * 100),

        "lbp": float(melhor_lbp * 100),

        "edge": float(melhor_edge * 100),

        "symmetry": float(melhor_sym * 100),

        "embedding_id": melhor_id
    }
