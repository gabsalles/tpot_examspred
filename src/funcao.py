def aplicar_feature_engineering_rh(df):
    """
    Aplica transformações matemáticas e lógicas para extrair sinal
    de dados esparsos de RH e Reintegração Judicial.
    """
    df = df.copy()

    # ==========================================================
    # 1. ABSENTEÍSMO E SAÚDE
    # ==========================================================

    # Aceleração (Trimestre atual vs Trimestre mais antigo dentro do semestre)
    tmest_anterior_saude = (
        df["qgral_ultimo_smest_saude"] - df["qgral_ultimo_tmest_saude"]
    )
    df["tendencia_piora_saude"] = df["qgral_ultimo_tmest_saude"] - tmest_anterior_saude

    # Intensidade do absenteísmo (Gravidade média por evento)
    df["dias_medios_por_afastamento_smest"] = df["qdias_afas_gral_ultimo_smest"] / (
        df["qgral_ultimo_smest_saude"] + 1
    )

    # Proporção Parental (O quanto das faltas são causadas por questões familiares?)
    df["prop_afast_parental"] = df["qdias_afas_paren_ultimo_smest"] / (
        df["qdias_afas_gral_ultimo_smest"] + 1
    )

    # Flag Binária de Zero-Inflation (Ajuda o modelo a dar o primeiro split)
    df["flag_teve_afastamento_recente"] = (df["qgral_ultimo_tmest_saude"] > 0).astype(
        int
    )

    # ==========================================================
    # 2. JORNADA, SOBRECARGA E BURNOUT
    # ==========================================================

    # Índice de Burnout de Final de Semana
    # (Dias de FDS trabalhados vs total de dias trabalhados esperados no semestre)
    dias_totais_smest = df["media_dias_trabalhados_mes_ult_smest"] * 6
    df["taxa_fds_trabalhado_smest"] = df["qdias_fds_trabalhados_ult_smest"] / (
        dias_totais_smest + 1
    )

    # Eficiência / Intensidade da Jornada (Horas por dia)
    df["intensidade_jornada"] = df["media_horas_realizadas_mes_ult_smest"] / (
        df["media_dias_trabalhados_mes_ult_smest"] + 1
    )

    # Aceleração de Sobrecarga (Trabalhou mais no último trimestre do que a média do semestre?)
    df["aceleracao_jornada_horas"] = df["media_horas_realizadas_mes_ult_tmest"] / (
        df["media_horas_realizadas_mes_ult_smest"] + 0.1
    )

    # ==========================================================
    # 3. CONTEXTO SOCIOECONÔMICO E HISTÓRICO JURÍDICO
    # ==========================================================

    # Pressão Financeira (Salário por boca para alimentar)
    df["renda_per_capita_estimada"] = df["salario"] / (df["qdepdt_func"] + 1)

    # Densidade de Reintegração (Litígios diluídos pela idade/tempo de vida)
    if "qtde_reint" in df.columns and "qidade_func" in df.columns:
        df["frequencia_reintegracao_por_idade"] = df["qtde_reint"] / (
            df["qidade_func"] + 1
        )

    # ==========================================================
    # 4. CROSS-FEATURES (UNIÃO DE STRINGS)
    # ==========================================================

    # Forçamos a conversão para string para evitar erros de tipos mistos ou NaNs
    df["cross_cargo_uf"] = (
        df["cargo_raiz"].astype(str) + "_UF_" + df["uf_trabalha"].astype(str)
    )
    df["cross_jornada_sexo"] = (
        df["jornd_trabalha"].astype(str) + "_Sexo_" + df["sexo"].astype(str)
    )

    if "cargo_supe_imeadiato" in df.columns:
        df["cross_hierarquia"] = (
            df["cargo_raiz"].astype(str)
            + "_RespA_"
            + df["cargo_supe_imeadiato"].astype(str)
        )

    if "rtpo_reint" in df.columns and "iadvog_parte" in df.columns:
        df["cross_perfil_juridico"] = (
            df["rtpo_reint"].astype(str) + "_Adv_" + df["iadvog_parte"].astype(str)
        )

    # ==========================================================
    # 5. ESCUDO ANTI-LEAKAGE (SEGURANÇA DO MODELO)
    # ==========================================================

    # Remove as proporções do target calculadas na base inteira para evitar que
    # o modelo decore o gabarito. O AutoGluon fará o Target Encoding por conta própria.
    cols_vazadas = [c for c in df.columns if str(c).startswith("prop_tempo_ativo_")]
    if cols_vazadas:
        # print(f"Removendo {len(cols_vazadas)} colunas com risco de Leakage...")
        df = df.drop(columns=cols_vazadas, errors="ignore")

    return df
