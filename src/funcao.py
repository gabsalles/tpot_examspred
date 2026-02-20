def aplicar_feature_engineering_rh(df):
    df = df.copy()

    # --- 1. ABSENTEÍSMO E SOBRECARGA ---

    # Aceleração (Trimestre vs Semestre anterior)
    tmest_anterior_saude = (
        df["qgral_ultimo_smest_saude"] - df["qgral_ultimo_tmest_saude"]
    )
    df["tendencia_piora_saude"] = df["qgral_ultimo_tmest_saude"] - tmest_anterior_saude

    # Intensidade do absenteísmo
    df["dias_medios_por_afastamento_smest"] = df["qdias_afas_gral_ultimo_smest"] / (
        df["qgral_ultimo_smest_saude"] + 1
    )

    # Proporção Parental (O quanto das faltas são por causa da família?)
    df["prop_afast_parental"] = df["qdias_afas_paren_ultimo_smest"] / (
        df["qdias_afas_gral_ultimo_smest"] + 1
    )

    # Índice de Burnout (Dias de FDS trabalhados em relação aos dias normais no semestre)
    # Multiplicamos a média mensal por 6 para parear com a contagem do semestre
    dias_totais_smest = df["media_dias_trabalhados_mes_ult_smest"] * 6
    df["taxa_fds_trabalhado_smest"] = df["qdias_fds_trabalhados_ult_smest"] / (
        dias_totais_smest + 1
    )

    # --- 2. CROSS-FEATURES (UNIÃO DE STRINGS) ---

    # Forçamos a conversão para string para não dar erro se houver NaN
    # Cargo + Localidade
    df["cross_cargo_uf"] = (
        df["cargo_raiz"].astype(str) + "_" + df["uf_trabalha"].astype(str)
    )

    # Jornada + Demografia
    df["cross_jornada_sexo"] = (
        df["jornd_trabalha"].astype(str) + "_" + df["sexo"].astype(str)
    )

    # Conflito de Cargos (Liderado vs Chefe)
    if "cargo_supe_imeadiato" in df.columns:
        df["cross_hierarquia"] = (
            df["cargo_raiz"].astype(str)
            + "_responde_a_"
            + df["cargo_supe_imeadiato"].astype(str)
        )

    return df
