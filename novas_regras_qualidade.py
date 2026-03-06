"""
╔══════════════════════════════════════════════════════════════════╗
║         Configuração de Checks — treint_treino + absenteismo     ║
╚══════════════════════════════════════════════════════════════════╝
"""

import sys

sys.path.append("/Workspace/Repos/seu-projeto/src")  # ajuste o caminho

from pyspark.sql import functions as F
from dq_framework import run_dq_checks, COLUMN_TYPES


# ──────────────────────────────────────────────────────────────────
#  TIPOS CUSTOMIZADOS
# ──────────────────────────────────────────────────────────────────

COLUMN_TYPES["cpf"] = lambda p: [
    {"rule": "not_null"},
    {"rule": "regex", "params": r"^\d{3}\.\d{3}\.\d{3}-\d{2}$"},
]


# ──────────────────────────────────────────────────────────────────
#  CARREGA AS TABELAS
# ──────────────────────────────────────────────────────────────────

df_treino = spark.table("treint_treino_fnl_20260126")
df_absenteismo = spark.table("treint_absenteismo")

# Alternativa Delta:
# df_treino      = spark.read.format("delta").load("/mnt/datalake/silver/treint_treino")
# df_absenteismo = spark.read.format("delta").load("/mnt/datalake/silver/treint_absenteismo")


# ──────────────────────────────────────────────────────────────────
#  LISTAS DE COLUNAS POR TIPO
# ──────────────────────────────────────────────────────────────────

# Identificadores
colunas_id = ["cd_funcl"]

# Códigos (não são IDs únicos, mas são obrigatórios)
colunas_codigo = [
    "cd_cargo",
    "cd_depdc_trabalha",
    "cd_dir_rgnal_pertc",
    "cd_empr_pertc",
    "cd_funcl_supe_imediato",
    "cd_gerc_rgnal_pertc",
    "cdepdc_ag",
    "cdepdc_dpto",
    "cuf_reg",
    "cuf_trab_janei",
]

# Métricas numéricas — quantidade (inteiros, não negativos)
colunas_qtd = [
    "qgral_ult_smest",
    "qlicen_ult_smest",
    "qmes_licen_ult_smest",
    "qdia_afas_ult_smest",
    "qdia_afas_cid_probl_emocn_ult_smest",
    "qdia_afas_cid_probl_muscu_ult_smest",
    "qdia_afas_cid_sist_atnmo_ult_smest",
    "qcons_mdica_ult_smest",
    "qtrato_ult_tmes",
    "qlicen_ult_tmes",
    "qmes_licen_ult_tmes",
    "qdia_afas_ult_tmes",
    "qdia_afas_cid_probl_emocn_ult_tmes",
    "qdia_afas_cid_probl_muscu_ult_tmes",
    "qdia_afas_cid_sist_atnmo_ult_tmes",
    "qausen_sem_justf_ult_tmes",
    "qatrso_ult_tmes",
    "qtrato_ult_smest",
    "qmes_ativo",
    "qidade_func",
    "qidade_gestor",
    "qdepdt_func",
    "qdias_afas_gral_ultimo_smest",
    "qdias_afas_gral_ultimo_tmest",
    "qdias_afas_paren_ultimo_smest",
    "qdias_afas_paren_ultimo_tmest",
    "qdias_afas_saude_ultimo_smest",
    "qdias_afas_saude_ultimo_tmest",
    "qdias_fds_trabalhados_ult_smest",
    "qdias_fds_trabalhados_ult_tmest",
    "qdias_afas_acid_ultimo_smest",
    "qdias_afas_acid_ultimo_tmest",
    "qfunc_atv_depdc",
    "qlicen_acid_ultimo_smest",
    "qlicen_acid_ultimo_tmest",
    "qlicen_ultimo_smest",
    "qlicen_ultimo_tmest",
    "qmeses_casa",
    "qmeses_casa_gestor",
    "qmeses_gestao_gestor",
    "qmeses_gestao_lider_agrupado",
    "qacid_ultimo_smest",
    "qacid_ultimo_tmest",
    "qtde_media",
    "qtde_reint",
    "qtrato_ultimo_smest",
    "qtrato_ultimo_tmest",
    "meses_ativo",
    "pz_afast",  # absenteismo
]

# Médias / valores por dia (numérico, não negativo)
colunas_media = [
    "vmed_dia_afas_cid_probl_emocn_ult_smest",
    "vmed_dia_afas_cid_sist_atnmo_ult_smest",
    "vmed_gral_ult_tmes",
    "vmed_dia_afas_ult_tmes",
    "vmed_dia_afas_cid_probl_muscu_ult_tmes",
    "media_dias_trabalhados_mes_ult_smest",
    "media_dias_trabalhados_mes_ult_tmest",
    "media_horas_compensadas_mes_ult_smest",
    "media_horas_compensadas_mes_ult_tmest",
    "media_horas_realizadas_mes_ult_smest",
    "media_horas_realizadas_mes_ult_tmest",
    "media_saldo_horas_ult_smest",
    "media_saldo_horas_ult_tmest",
]

# Proporções (entre 0 e 1)
colunas_proporcao = [
    "prop_tempo_ativo_3y_cadvog_parte",
    "prop_tempo_ativo_3y_cargo_inicial",
    "prop_tempo_ativo_3y_cd_dir_rgnal_pertc",
    "prop_tempo_ativo_3y_cd_funcl_supe_imediato",
    "prop_tempo_ativo_3y_cuf_reg",
    "prop_tempo_ativo_3y_tpo_depdc_pertc",
    "prop_tempo_ativo_3y_uf_trabalha",
    "prop_tempo_ativo_4y_cadvog_parte",
    "prop_tempo_ativo_4y_cargo_inicial",
    "prop_tempo_ativo_4y_cd_dir_rgnal_pertc",
    "prop_tempo_ativo_4y_cd_funcl_supe_imediato",
    "prop_tempo_ativo_4y_cuf_reg",
    "prop_tempo_ativo_4y_tpo_depdc_pertc",
    "prop_tempo_ativo_4y_uf_trabalha",
    "prop_tempo_ativo_media",
]

# Flags (0 ou 1)
colunas_flag = [
    "cfunc_lider",
    "cfunc_ptdor_defic",
    "cfunc_sindz",
    "cpater_mater_smest",
    "cpater_mater_tmest",
    "cpedido_acidt_trab",
    "cpedido_dano",
    "cpedido_estab_acidt",
    "cpedido_peric",
    "cpdido_acidt_trab",  # treino
    "reint_desde_inicio",
    "sexo_FEMININO",
    "cempr_ligada",
    "cdmiss_aposn_inval",
    "cdmiss_justa_causa",
]

# Datas
colunas_data = [
    "dt_admis",
    "dt_nasc",
    "dt_reint1",
    "dt_first_dmiss",
]

# Texto opcional (podem ser nulos por design)
colunas_opcional = [
    "dult_dmiss_reint",
    "last_tpo_dmiss",
    "dsafra_mod",
    "cid_comar",
    "rid_comar",
    "ds_cid",  # absenteismo — cid pode ser nulo
    "cargo_supe_imeadiato",
    "supe_imediato",
]


# ──────────────────────────────────────────────────────────────────
#  CUSTOM TYPES: proporção
# ──────────────────────────────────────────────────────────────────

COLUMN_TYPES["proporcao"] = lambda p: [
    {"rule": "not_null"},
    {"rule": "min_value", "params": 0},
    {"rule": "max_value", "params": 1},
]


# ──────────────────────────────────────────────────────────────────
#  CHECKS
# ──────────────────────────────────────────────────────────────────

checks = [
    # ── TREINO ───────────────────────────────────────────────────
    {
        "table_name": "treint_treino",
        "df": df_treino,
        "column_groups": [
            {"columns": colunas_id, "type": "id"},
            {"columns": colunas_codigo, "type": "texto"},
            {"columns": colunas_qtd, "type": "numerico"},
            {"columns": colunas_media, "type": "numerico"},
            {"columns": colunas_proporcao, "type": "proporcao"},
            {"columns": colunas_flag, "type": "flag"},
            {"columns": colunas_data, "type": "data"},
            {"columns": colunas_opcional, "type": "texto_opcional"},
        ],
        "columns": {
            "cpf": {"type": "cpf"},
            "salario": {"type": "monetario"},
            "salario_last": {"type": "monetario"},
            "sexo": {"type": "categoria", "accepted_values": ["M", "F"]},
            "uf_trabalha": {
                "type": "categoria",
                "accepted_values": [
                    "AC",
                    "AL",
                    "AM",
                    "AP",
                    "BA",
                    "CE",
                    "DF",
                    "ES",
                    "GO",
                    "MA",
                    "MG",
                    "MS",
                    "MT",
                    "PA",
                    "PB",
                    "PE",
                    "PI",
                    "PR",
                    "RJ",
                    "RN",
                    "RO",
                    "RR",
                    "RS",
                    "SC",
                    "SE",
                    "SP",
                    "TO",
                ],
            },
            "sit_rh_last": {
                "type": "categoria",
                "accepted_values": [
                    "ATIVO",
                    "DESLIGADO",
                    "AFASTADO",
                    "INATIVO",  # ajuste conforme seus valores reais
                ],
            },
            "tempo_ativo_1y": {"type": "numerico"},
            "tempo_ativo_2y": {"type": "numerico"},
            "tempo_ativo_3y": {"type": "numerico"},
            "tempo_ativo_4y": {"type": "numerico"},
            "tempo_ativo_5y": {"type": "numerico"},
        },
        # ── Consistência interna (cross-column) ──────────────────
        "extra_rules": [
            # dt_admis deve ser anterior à dt_first_dmiss
            {
                "column": "dt_admis",
                "rule": "custom",
                "params": lambda df: df.filter(
                    F.col("dt_admis").isNotNull()
                    & F.col("dt_first_dmiss").isNotNull()
                    & (F.col("dt_admis") > F.col("dt_first_dmiss"))
                ),
            },
            # tempo_ativo deve ser cumulativo: 1y <= 2y <= 3y <= 4y <= 5y
            {
                "column": "tempo_ativo_1y",
                "rule": "custom",
                "params": lambda df: df.filter(
                    F.col("tempo_ativo_1y").isNotNull()
                    & F.col("tempo_ativo_2y").isNotNull()
                    & (F.col("tempo_ativo_1y") > F.col("tempo_ativo_2y"))
                ),
            },
            {
                "column": "tempo_ativo_2y",
                "rule": "custom",
                "params": lambda df: df.filter(
                    F.col("tempo_ativo_2y").isNotNull()
                    & F.col("tempo_ativo_3y").isNotNull()
                    & (F.col("tempo_ativo_2y") > F.col("tempo_ativo_3y"))
                ),
            },
            {
                "column": "tempo_ativo_3y",
                "rule": "custom",
                "params": lambda df: df.filter(
                    F.col("tempo_ativo_3y").isNotNull()
                    & F.col("tempo_ativo_4y").isNotNull()
                    & (F.col("tempo_ativo_3y") > F.col("tempo_ativo_4y"))
                ),
            },
            {
                "column": "tempo_ativo_4y",
                "rule": "custom",
                "params": lambda df: df.filter(
                    F.col("tempo_ativo_4y").isNotNull()
                    & F.col("tempo_ativo_5y").isNotNull()
                    & (F.col("tempo_ativo_4y") > F.col("tempo_ativo_5y"))
                ),
            },
            # Funcionário ATIVO não pode ter salário zerado ou nulo
            {
                "column": "salario",
                "rule": "custom",
                "params": lambda df: df.filter(
                    (F.col("sit_rh_last") == "ATIVO")
                    & (F.col("salario").isNull() | (F.col("salario") <= 0))
                ),
            },
            # Precisão: salario vs salario_last — variação > 50% é suspeita
            {
                "column": "salario",
                "rule": "accuracy",
                "params": {
                    "ref_df": df_treino.select("cd_funcl", "salario_last"),
                    "join_key": "cd_funcl",
                    "ref_column": "salario_last",
                    "tolerance": df_treino.agg((F.avg("salario_last") * 0.5)).collect()[
                        0
                    ][0]
                    or 9999,
                },
            },
            # semestral deve ser >= trimestral (afastamentos acumulam)
            {
                "column": "qdias_afas_gral_ultimo_smest",
                "rule": "custom",
                "params": lambda df: df.filter(
                    F.col("qdias_afas_gral_ultimo_smest").isNotNull()
                    & F.col("qdias_afas_gral_ultimo_tmest").isNotNull()
                    & (
                        F.col("qdias_afas_gral_ultimo_smest")
                        < F.col("qdias_afas_gral_ultimo_tmest")
                    )
                ),
            },
        ],
    },
    # ── ABSENTEISMO ───────────────────────────────────────────────
    {
        "table_name": "treint_absenteismo",
        "df": df_absenteismo,
        "column_groups": [
            {"columns": ["cd_ausen"], "type": "id"},
            {"columns": ["tpo_ausen"], "type": "texto"},
            {"columns": ["pz_afast"], "type": "numerico"},
            {"columns": ["dt_inic", "dt_fim"], "type": "data"},
        ],
        "columns": {
            "cpf": {"type": "cpf"},
            "cd_cid": {"type": "texto_opcional"},
            "ds_cid": {"type": "texto_opcional"},
            "iarq_igtao": {"type": "texto_opcional"},
            "higtao": {"type": "texto_opcional"},
            "digtao_ptcao": {"type": "texto_opcional"},
        },
        # ── Consistência interna ──────────────────────────────────
        "extra_rules": [
            # dt_inic deve ser <= dt_fim
            {
                "column": "dt_inic",
                "rule": "custom",
                "params": lambda df: df.filter(
                    F.col("dt_inic").isNotNull()
                    & F.col("dt_fim").isNotNull()
                    & (F.col("dt_inic") > F.col("dt_fim"))
                ),
            },
            # pz_afast deve ser consistente com a diferença dt_fim - dt_inic
            # tolerância de 1 dia por possíveis diferenças de contagem
            {
                "column": "pz_afast",
                "rule": "custom",
                "params": lambda df: df.filter(
                    F.col("dt_inic").isNotNull()
                    & F.col("dt_fim").isNotNull()
                    & F.col("pz_afast").isNotNull()
                    & (
                        F.abs(
                            F.col("pz_afast")
                            - F.datediff(F.col("dt_fim"), F.col("dt_inic"))
                        )
                        > 1
                    )
                ),
            },
            # Consistência entre tabelas: cpf em absenteismo deve existir em treino
            {
                "column": "cpf",
                "rule": "referential",
                "ref_df": df_treino,
                "ref_column": "cpf",
            },
        ],
    },
]


# ──────────────────────────────────────────────────────────────────
#  EXECUTA E SALVA
# ──────────────────────────────────────────────────────────────────

resultado_df = run_dq_checks(spark, checks)

# Inspeciona falhas
resultado_df.filter(F.col("status").isin("FAIL", "ERROR")).select(
    "table_name",
    "column_name",
    "quality_dimension",
    "rule",
    "failed_rows",
    "pct_passed",
    "status",
).orderBy("table_name", "quality_dimension").show(truncate=False)

# Salva pro Power BI (append para acumular histórico)
OUTPUT_PATH = "/mnt/datalake/gold/dq_resultados"
resultado_df.write.format("delta").mode("append").save(OUTPUT_PATH)

print(f"\n✅ Resultado salvo em {OUTPUT_PATH}")


# --------------
"""
╔══════════════════════════════════════════════════════════════════╗
║         Configuração de Checks — treint_treino + absenteismo     ║
╚══════════════════════════════════════════════════════════════════╝
"""

import sys

sys.path.append("/Workspace/Repos/seu-projeto/src")  # ajuste o caminho

from pyspark.sql import functions as F
from dq_framework import run_dq_checks, COLUMN_TYPES


# ──────────────────────────────────────────────────────────────────
#  TIPOS CUSTOMIZADOS
# ──────────────────────────────────────────────────────────────────

COLUMN_TYPES["cpf"] = lambda p: [
    {"rule": "not_null"},
    {"rule": "regex", "params": r"^\d{3}\.\d{3}\.\d{3}-\d{2}$"},
]


# ──────────────────────────────────────────────────────────────────
#  CARREGA AS TABELAS
# ──────────────────────────────────────────────────────────────────

df_treino = spark.table("treint_treino_fnl_20260126")
df_absenteismo = spark.table("treint_absenteismo")

# Alternativa Delta:
# df_treino      = spark.read.format("delta").load("/mnt/datalake/silver/treint_treino")
# df_absenteismo = spark.read.format("delta").load("/mnt/datalake/silver/treint_absenteismo")


# ──────────────────────────────────────────────────────────────────
#  LISTAS DE COLUNAS POR TIPO
# ──────────────────────────────────────────────────────────────────

# Identificadores
colunas_id = ["cd_funcl"]

# Códigos (não são IDs únicos, mas são obrigatórios)
colunas_codigo = [
    "cd_cargo",
    "cd_depdc_trabalha",
    "cd_dir_rgnal_pertc",
    "cd_empr_pertc",
    "cd_funcl_supe_imediato",
    "cd_gerc_rgnal_pertc",
    "cdepdc_ag",
    "cdepdc_dpto",
    "cuf_reg",
    "cuf_trab_janei",
]

# Métricas numéricas — quantidade (inteiros, não negativos)
colunas_qtd = [
    "qgral_ult_smest",
    "qlicen_ult_smest",
    "qmes_licen_ult_smest",
    "qdia_afas_ult_smest",
    "qdia_afas_cid_probl_emocn_ult_smest",
    "qdia_afas_cid_probl_muscu_ult_smest",
    "qdia_afas_cid_sist_atnmo_ult_smest",
    "qcons_mdica_ult_smest",
    "qtrato_ult_tmes",
    "qlicen_ult_tmes",
    "qmes_licen_ult_tmes",
    "qdia_afas_ult_tmes",
    "qdia_afas_cid_probl_emocn_ult_tmes",
    "qdia_afas_cid_probl_muscu_ult_tmes",
    "qdia_afas_cid_sist_atnmo_ult_tmes",
    "qausen_sem_justf_ult_tmes",
    "qatrso_ult_tmes",
    "qtrato_ult_smest",
    "qmes_ativo",
    "qidade_func",
    "qidade_gestor",
    "qdepdt_func",
    "qdias_afas_gral_ultimo_smest",
    "qdias_afas_gral_ultimo_tmest",
    "qdias_afas_paren_ultimo_smest",
    "qdias_afas_paren_ultimo_tmest",
    "qdias_afas_saude_ultimo_smest",
    "qdias_afas_saude_ultimo_tmest",
    "qdias_fds_trabalhados_ult_smest",
    "qdias_fds_trabalhados_ult_tmest",
    "qdias_afas_acid_ultimo_smest",
    "qdias_afas_acid_ultimo_tmest",
    "qfunc_atv_depdc",
    "qlicen_acid_ultimo_smest",
    "qlicen_acid_ultimo_tmest",
    "qlicen_ultimo_smest",
    "qlicen_ultimo_tmest",
    "qmeses_casa",
    "qmeses_casa_gestor",
    "qmeses_gestao_gestor",
    "qmeses_gestao_lider_agrupado",
    "qacid_ultimo_smest",
    "qacid_ultimo_tmest",
    "qtde_media",
    "qtde_reint",
    "qtrato_ultimo_smest",
    "qtrato_ultimo_tmest",
    "meses_ativo",
    "pz_afast",  # absenteismo
]

# Médias / valores por dia (numérico, não negativo)
colunas_media = [
    "vmed_dia_afas_cid_probl_emocn_ult_smest",
    "vmed_dia_afas_cid_sist_atnmo_ult_smest",
    "vmed_gral_ult_tmes",
    "vmed_dia_afas_ult_tmes",
    "vmed_dia_afas_cid_probl_muscu_ult_tmes",
    "media_dias_trabalhados_mes_ult_smest",
    "media_dias_trabalhados_mes_ult_tmest",
    "media_horas_compensadas_mes_ult_smest",
    "media_horas_compensadas_mes_ult_tmest",
    "media_horas_realizadas_mes_ult_smest",
    "media_horas_realizadas_mes_ult_tmest",
    "media_saldo_horas_ult_smest",
    "media_saldo_horas_ult_tmest",
]

# Proporções (entre 0 e 1)
colunas_proporcao = [
    "prop_tempo_ativo_3y_cadvog_parte",
    "prop_tempo_ativo_3y_cargo_inicial",
    "prop_tempo_ativo_3y_cd_dir_rgnal_pertc",
    "prop_tempo_ativo_3y_cd_funcl_supe_imediato",
    "prop_tempo_ativo_3y_cuf_reg",
    "prop_tempo_ativo_3y_tpo_depdc_pertc",
    "prop_tempo_ativo_3y_uf_trabalha",
    "prop_tempo_ativo_4y_cadvog_parte",
    "prop_tempo_ativo_4y_cargo_inicial",
    "prop_tempo_ativo_4y_cd_dir_rgnal_pertc",
    "prop_tempo_ativo_4y_cd_funcl_supe_imediato",
    "prop_tempo_ativo_4y_cuf_reg",
    "prop_tempo_ativo_4y_tpo_depdc_pertc",
    "prop_tempo_ativo_4y_uf_trabalha",
    "prop_tempo_ativo_media",
]

# Flags (0 ou 1)
colunas_flag = [
    "cfunc_lider",
    "cfunc_ptdor_defic",
    "cfunc_sindz",
    "cpater_mater_smest",
    "cpater_mater_tmest",
    "cpedido_acidt_trab",
    "cpedido_dano",
    "cpedido_estab_acidt",
    "cpedido_peric",
    "cpdido_acidt_trab",  # treino
    "reint_desde_inicio",
    "sexo_FEMININO",
    "cempr_ligada",
    "cdmiss_aposn_inval",
    "cdmiss_justa_causa",
]

# Datas
colunas_data = [
    "dt_admis",
    "dt_nasc",
    "dt_reint1",
    "dt_first_dmiss",
]

# Texto opcional (podem ser nulos por design)
colunas_opcional = [
    "dult_dmiss_reint",
    "last_tpo_dmiss",
    "dsafra_mod",
    "cid_comar",
    "rid_comar",
    "ds_cid",  # absenteismo — cid pode ser nulo
    "cargo_supe_imeadiato",
    "supe_imediato",
]


# ──────────────────────────────────────────────────────────────────
#  CUSTOM TYPES: proporção
# ──────────────────────────────────────────────────────────────────

COLUMN_TYPES["proporcao"] = lambda p: [
    {"rule": "not_null"},
    {"rule": "min_value", "params": 0},
    {"rule": "max_value", "params": 1},
]


# ──────────────────────────────────────────────────────────────────
#  HELPER: consistência semestral >= trimestral
#  Adicione ou remova pares aqui — o resto é automático
# ──────────────────────────────────────────────────────────────────

PARES_SMEST_TMEST = [
    # Afastamentos gerais
    ("qdias_afas_gral_ultimo_smest", "qdias_afas_gral_ultimo_tmest"),
    # Afastamentos parentais
    ("qdias_afas_paren_ultimo_smest", "qdias_afas_paren_ultimo_tmest"),
    # Afastamentos saúde
    ("qdias_afas_saude_ultimo_smest", "qdias_afas_saude_ultimo_tmest"),
    # Afastamentos acidente
    ("qdias_afas_acid_ultimo_smest", "qdias_afas_acid_ultimo_tmest"),
    # Finais de semana trabalhados
    ("qdias_fds_trabalhados_ult_smest", "qdias_fds_trabalhados_ult_tmest"),
    # Licenças acidente
    ("qlicen_acid_ultimo_smest", "qlicen_acid_ultimo_tmest"),
    # Licenças geral
    ("qlicen_ultimo_smest", "qlicen_ultimo_tmest"),
    # Tratamentos
    ("qtrato_ultimo_smest", "qtrato_ultimo_tmest"),
    # Tratamentos licença saúde
    ("qgral_trato_licen_smest_saude", "qgral_trato_licen_tmest_saude"),
    # Quantidade geral saúde
    ("qgral_ultimo_smest_saude", "qgral_ultimo_tmest_saude"),
    # Média dias trabalhados
    ("media_dias_trabalhados_mes_ult_smest", "media_dias_trabalhados_mes_ult_tmest"),
    # Média horas realizadas
    ("media_horas_realizadas_mes_ult_smest", "media_horas_realizadas_mes_ult_tmest"),
    # Média horas compensadas
    ("media_horas_compensadas_mes_ult_smest", "media_horas_compensadas_mes_ult_tmest"),
    # Média saldo horas
    ("media_saldo_horas_ult_smest", "media_saldo_horas_ult_tmest"),
    # Cacid trabl licen
    ("cacid_trabl_licen_acid_ultimo_smest", "cacid_trabl_licen_acid_ultimo_tmest"),
    # Cpater mater
    ("cpater_mater_smest", "cpater_mater_tmest"),
]


def regras_smest_tmest(pares: list) -> list:
    """Gera regras custom de consistência semestral >= trimestral para cada par."""
    return [
        {
            "column": smest,
            "rule": "custom",
            "params": lambda df, s=smest, t=tmest: df.filter(
                F.col(s).isNotNull() & F.col(t).isNotNull() & (F.col(s) < F.col(t))
            ),
        }
        for smest, tmest in pares
    ]


checks = [
    # ── TREINO ───────────────────────────────────────────────────
    {
        "table_name": "treint_treino",
        "df": df_treino,
        "column_groups": [
            {"columns": colunas_id, "type": "id"},
            {"columns": colunas_codigo, "type": "texto"},
            {"columns": colunas_qtd, "type": "numerico"},
            {"columns": colunas_media, "type": "numerico"},
            {"columns": colunas_proporcao, "type": "proporcao"},
            {"columns": colunas_flag, "type": "flag"},
            {"columns": colunas_data, "type": "data"},
            {"columns": colunas_opcional, "type": "texto_opcional"},
        ],
        "columns": {
            "cpf": {"type": "cpf"},
            "salario": {"type": "monetario"},
            "salario_last": {"type": "monetario"},
            "sexo": {"type": "categoria", "accepted_values": ["M", "F"]},
            "uf_trabalha": {
                "type": "categoria",
                "accepted_values": [
                    "AC",
                    "AL",
                    "AM",
                    "AP",
                    "BA",
                    "CE",
                    "DF",
                    "ES",
                    "GO",
                    "MA",
                    "MG",
                    "MS",
                    "MT",
                    "PA",
                    "PB",
                    "PE",
                    "PI",
                    "PR",
                    "RJ",
                    "RN",
                    "RO",
                    "RR",
                    "RS",
                    "SC",
                    "SE",
                    "SP",
                    "TO",
                ],
            },
            "sit_rh_last": {
                "type": "categoria",
                "accepted_values": [
                    "ATIVO",
                    "DESLIGADO",
                    "AFASTADO",
                    "INATIVO",  # ajuste conforme seus valores reais
                ],
            },
            "tempo_ativo_1y": {"type": "numerico"},
            "tempo_ativo_2y": {"type": "numerico"},
            "tempo_ativo_3y": {"type": "numerico"},
            "tempo_ativo_4y": {"type": "numerico"},
            "tempo_ativo_5y": {"type": "numerico"},
        },
        # ── Consistência interna (cross-column) ──────────────────
        "extra_rules": [
            # dt_admis deve ser anterior à dt_first_dmiss
            {
                "column": "dt_admis",
                "rule": "custom",
                "params": lambda df: df.filter(
                    F.col("dt_admis").isNotNull()
                    & F.col("dt_first_dmiss").isNotNull()
                    & (F.col("dt_admis") > F.col("dt_first_dmiss"))
                ),
            },
            # tempo_ativo deve ser cumulativo: 1y <= 2y <= 3y <= 4y <= 5y
            {
                "column": "tempo_ativo_1y",
                "rule": "custom",
                "params": lambda df: df.filter(
                    F.col("tempo_ativo_1y").isNotNull()
                    & F.col("tempo_ativo_2y").isNotNull()
                    & (F.col("tempo_ativo_1y") > F.col("tempo_ativo_2y"))
                ),
            },
            {
                "column": "tempo_ativo_2y",
                "rule": "custom",
                "params": lambda df: df.filter(
                    F.col("tempo_ativo_2y").isNotNull()
                    & F.col("tempo_ativo_3y").isNotNull()
                    & (F.col("tempo_ativo_2y") > F.col("tempo_ativo_3y"))
                ),
            },
            {
                "column": "tempo_ativo_3y",
                "rule": "custom",
                "params": lambda df: df.filter(
                    F.col("tempo_ativo_3y").isNotNull()
                    & F.col("tempo_ativo_4y").isNotNull()
                    & (F.col("tempo_ativo_3y") > F.col("tempo_ativo_4y"))
                ),
            },
            {
                "column": "tempo_ativo_4y",
                "rule": "custom",
                "params": lambda df: df.filter(
                    F.col("tempo_ativo_4y").isNotNull()
                    & F.col("tempo_ativo_5y").isNotNull()
                    & (F.col("tempo_ativo_4y") > F.col("tempo_ativo_5y"))
                ),
            },
            # Funcionário ATIVO não pode ter salário zerado ou nulo
            {
                "column": "salario",
                "rule": "custom",
                "params": lambda df: df.filter(
                    (F.col("sit_rh_last") == "ATIVO")
                    & (F.col("salario").isNull() | (F.col("salario") <= 0))
                ),
            },
            # Precisão: salario vs salario_last — variação > 50% é suspeita
            {
                "column": "salario",
                "rule": "accuracy",
                "params": {
                    "ref_df": df_treino.select("cd_funcl", "salario_last"),
                    "join_key": "cd_funcl",
                    "ref_column": "salario_last",
                    "tolerance": df_treino.agg((F.avg("salario_last") * 0.5)).collect()[
                        0
                    ][0]
                    or 9999,
                },
            },
            # Consistência: semestral >= trimestral para todos os pares
            *regras_smest_tmest(PARES_SMEST_TMEST),
        ],
    },
    # ── ABSENTEISMO ───────────────────────────────────────────────
    {
        "table_name": "treint_absenteismo",
        "df": df_absenteismo,
        "column_groups": [
            {"columns": ["cd_ausen"], "type": "id"},
            {"columns": ["tpo_ausen"], "type": "texto"},
            {"columns": ["pz_afast"], "type": "numerico"},
            {"columns": ["dt_inic", "dt_fim"], "type": "data"},
        ],
        "columns": {
            "cpf": {"type": "cpf"},
            "cd_cid": {"type": "texto_opcional"},
            "ds_cid": {"type": "texto_opcional"},
            "iarq_igtao": {"type": "texto_opcional"},
            "higtao": {"type": "texto_opcional"},
            "digtao_ptcao": {"type": "texto_opcional"},
        },
        # ── Consistência interna ──────────────────────────────────
        "extra_rules": [
            # dt_inic deve ser <= dt_fim
            {
                "column": "dt_inic",
                "rule": "custom",
                "params": lambda df: df.filter(
                    F.col("dt_inic").isNotNull()
                    & F.col("dt_fim").isNotNull()
                    & (F.col("dt_inic") > F.col("dt_fim"))
                ),
            },
            # pz_afast deve ser consistente com a diferença dt_fim - dt_inic
            # tolerância de 1 dia por possíveis diferenças de contagem
            {
                "column": "pz_afast",
                "rule": "custom",
                "params": lambda df: df.filter(
                    F.col("dt_inic").isNotNull()
                    & F.col("dt_fim").isNotNull()
                    & F.col("pz_afast").isNotNull()
                    & (
                        F.abs(
                            F.col("pz_afast")
                            - F.datediff(F.col("dt_fim"), F.col("dt_inic"))
                        )
                        > 1
                    )
                ),
            },
            # Consistência entre tabelas: cpf em absenteismo deve existir em treino
            {
                "column": "cpf",
                "rule": "referential",
                "ref_df": df_treino,
                "ref_column": "cpf",
            },
        ],
    },
]


# ──────────────────────────────────────────────────────────────────
#  EXECUTA E SALVA
# ──────────────────────────────────────────────────────────────────

resultado_df = run_dq_checks(spark, checks)

# Inspeciona falhas
resultado_df.filter(F.col("status").isin("FAIL", "ERROR")).select(
    "table_name",
    "column_name",
    "quality_dimension",
    "rule",
    "failed_rows",
    "pct_passed",
    "status",
).orderBy("table_name", "quality_dimension").show(truncate=False)

# Salva pro Power BI (append para acumular histórico)
OUTPUT_PATH = "/mnt/datalake/gold/dq_resultados"
resultado_df.write.format("delta").mode("append").save(OUTPUT_PATH)

print(f"\n✅ Resultado salvo em {OUTPUT_PATH}")
