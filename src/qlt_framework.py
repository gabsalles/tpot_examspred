"""
╔══════════════════════════════════════════════════════════════════╗
║              PySpark Data Quality Framework                      ║
║  Input  : DataFrames + tipos de coluna declarados               ║
║  Output : DataFrame consolidado pronto pro Power BI             ║
╚══════════════════════════════════════════════════════════════════╝

COMPORTAMENTO AUTOMÁTICO:
  Colunas NÃO declaradas recebem regras básicas inferidas pelo tipo Spark:
    StringType / ...     → not_null
    Numeric (Int, Long,
             Double...)  → not_null + not_negative
    DateType /
    TimestampType        → not_null + freshness_days (padrão: 30 dias)
    BooleanType          → not_null

  unique NÃO é aplicado automaticamente — só para colunas declaradas como "id",
  porque unicidade é semântica e o framework não pode inferir isso sozinho.

TIPOS DE COLUNA DISPONÍVEIS (para declarar explicitamente):
  - id                : not_null + unique
  - texto             : not_null
  - texto_opcional    : (nenhuma regra — opta a coluna pra fora)
  - numerico          : not_null + not_negative
  - numerico_signed   : not_null (aceita negativos)
  - monetario         : not_null + not_negative [+ max_value se declarado]
  - email             : not_null + regex de e-mail
  - data              : not_null
  - data_evento       : not_null + freshness_days (padrão: 30 dias)
  - flag              : not_null + accepted_values [0, 1]
  - categoria         : not_null [+ accepted_values se declarado]
  - chave_estrangeira : not_null + referential (ref_df e ref_column obrigatórios)

COMO DECLARAR NO NOTEBOOK:

  # Opção 1: sem declarar nada — tudo inferido automaticamente
  checks = [{"table_name": "pedidos", "df": df_pedidos}]

  # Opção 2: declara só as exceções e tipos específicos
  checks = [
    {
      "table_name": "pedidos",
      "df": df_pedidos,
      "columns": {
        "pedido_id"  : {"type": "id"},                         # garante unique
        "cliente_id" : {"type": "chave_estrangeira",
                        "ref_df": df_clientes, "ref_column": "id"},
        "valor"      : {"type": "monetario", "max_value": 10000},
        "status"     : {"type": "categoria",
                        "accepted_values": ["PAGO", "PENDENTE", "CANCELADO"]},
        "dt_pedido"  : {"type": "data_evento", "freshness_days": 7},
        "observacao" : {"type": "texto_opcional"},             # opta pra fora
      },
      # Regras extras além do tipo (qualquer lógica Spark):
      "extra_rules": [
        {"column": "valor", "rule": "custom",
         "params": lambda df: df.filter(
             (F.col("valor") > 500) & (F.col("status") != "PAGO")
         )},
      ],
      # Remove regras específicas de uma coluna:
      "skip_rules": {
        "cliente_id": ["not_null"],
      },
    }
  ]
"""

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import (
    StructType,
    StructField,
    StringType,
    LongType,
    DoubleType,
    TimestampType,
    IntegerType,
    FloatType,
    ShortType,
    ByteType,
    DecimalType,
    DateType,
    BooleanType,
    NumericType,
)
from datetime import datetime
from typing import Any


# ──────────────────────────────────────────────────────────────────
#  INFERÊNCIA AUTOMÁTICA por tipo Spark
#  Retorna rule_defs básicos para colunas não declaradas
# ──────────────────────────────────────────────────────────────────

_NUMERIC_TYPES = (
    IntegerType,
    LongType,
    FloatType,
    DoubleType,
    ShortType,
    ByteType,
    DecimalType,
)
_DATE_TYPES = (DateType, TimestampType)

_DEFAULT_FRESHNESS_DAYS = 30


def _infer_rules(spark_type) -> list[dict]:
    """Retorna as regras básicas para um tipo Spark."""
    if isinstance(spark_type, _NUMERIC_TYPES):
        return [{"rule": "not_null"}, {"rule": "not_negative"}]
    elif isinstance(spark_type, _DATE_TYPES):
        return [
            {"rule": "not_null"},
            {"rule": "freshness_days", "params": _DEFAULT_FRESHNESS_DAYS},
        ]
    elif isinstance(spark_type, BooleanType):
        return [{"rule": "not_null"}]
    else:  # StringType e qualquer outro
        return [{"rule": "not_null"}]


# ──────────────────────────────────────────────────────────────────
#  TIPOS DE COLUNA → regras explícitas
#
#  Para adicionar um tipo customizado no seu notebook:
#    from dq_framework import COLUMN_TYPES
#    COLUMN_TYPES["cpf"] = lambda p: [
#        {"rule": "not_null"},
#        {"rule": "regex", "params": r"^\d{3}\.\d{3}\.\d{3}-\d{2}$"},
#    ]
# ──────────────────────────────────────────────────────────────────

_EMAIL_REGEX = r"^[\w\.\+\-]+@[\w\-]+\.[a-zA-Z]{2,}$"

COLUMN_TYPES: dict[str, callable] = {
    "id": lambda p: [
        {"rule": "not_null"},
        {"rule": "unique"},
    ],
    "texto": lambda p: [
        {"rule": "not_null"},
    ],
    "texto_opcional": lambda p: [],  # opta a coluna pra fora — sem regras
    "numerico": lambda p: [
        {"rule": "not_null"},
        {"rule": "not_negative"},
    ],
    "numerico_signed": lambda p: [
        {"rule": "not_null"},
    ],
    "monetario": lambda p: [
        {"rule": "not_null"},
        {"rule": "not_negative"},
        *(
            [{"rule": "max_value", "params": p["max_value"]}]
            if "max_value" in p
            else []
        ),
    ],
    "email": lambda p: [
        {"rule": "not_null"},
        {"rule": "regex", "params": _EMAIL_REGEX},
    ],
    "data": lambda p: [
        {"rule": "not_null"},
    ],
    "data_evento": lambda p: [
        {"rule": "not_null"},
        {
            "rule": "freshness_days",
            "params": p.get("freshness_days", _DEFAULT_FRESHNESS_DAYS),
        },
    ],
    "flag": lambda p: [
        {"rule": "not_null"},
        {"rule": "accepted_values", "params": p.get("accepted_values", [0, 1])},
    ],
    "categoria": lambda p: [
        {"rule": "not_null"},
        *(
            [{"rule": "accepted_values", "params": p["accepted_values"]}]
            if "accepted_values" in p
            else []
        ),
    ],
    "chave_estrangeira": lambda p: [
        {"rule": "not_null"},
        {"rule": "referential", "ref_df": p["ref_df"], "ref_column": p["ref_column"]},
    ],
}


def _build_rule_defs(
    df: DataFrame,
    columns: dict,
    skip_rules: dict,
    column_groups: list,
) -> tuple[list[dict], dict]:
    """
    Constrói a lista flat de rule_defs combinando:
    - column_groups  → regras em lote para listas de colunas
    - columns        → declarações individuais (sobrescreve grupos e inferência)
    - não declaradas → inferência pelo tipo Spark
    - texto_opcional → opta a coluna pra fora
    - skip_rules     → remove regras específicas de uma coluna

    Retorna (rule_defs, col_type_map)
    """
    rule_defs = []
    col_type_map = {}
    skip = skip_rules or {}

    # Schema do DataFrame: col_name → spark_type
    schema_map = {field.name: field.dataType for field in df.schema.fields}

    # Expande column_groups em um dicionário coluna → config
    # (mesma estrutura do `columns` individual)
    groups_expanded = {}
    for group in column_groups or []:
        col_type = group.get("type")
        if col_type not in COLUMN_TYPES:
            raise ValueError(
                f"Tipo desconhecido no group: '{col_type}'. "
                f"Disponíveis: {list(COLUMN_TYPES.keys())}"
            )
        # Configurações extras do grupo (ex: max_value, freshness_days, accepted_values)
        group_cfg = {k: v for k, v in group.items() if k != "columns"}
        for col_name in group["columns"]:
            if col_name not in schema_map:
                print(
                    f"   ⚠️  column_group '{col_type}' — coluna '{col_name}' não existe no DataFrame — ignorada."
                )
                continue  # coluna não existe nessa tabela — ignora silenciosamente
            groups_expanded[col_name] = group_cfg

    # columns individual tem prioridade sobre groups_expanded
    declared = {**groups_expanded, **columns}

    # Avisa sobre colunas declaradas em `columns` que não existem no schema
    for col_name in columns:
        if col_name not in schema_map:
            print(
                f"   ⚠️  Coluna '{col_name}' declarada em 'columns' não existe no DataFrame — ignorada."
            )

    for col_name in schema_map.keys():
        skipped = skip.get(col_name, [])

        if col_name in declared:
            col_cfg = declared[col_name]
            col_type = col_cfg.get("type")

            if col_type not in COLUMN_TYPES:
                raise ValueError(
                    f"Tipo desconhecido: '{col_type}'. "
                    f"Disponíveis: {list(COLUMN_TYPES.keys())}"
                )

            col_type_map[col_name] = col_type
            rules = COLUMN_TYPES[col_type](col_cfg)

        else:
            # Coluna não declarada → inferência automática
            spark_type = schema_map[col_name]
            col_type = f"auto:{type(spark_type).__name__}"
            col_type_map[col_name] = col_type
            rules = _infer_rules(spark_type)

        for rule_def in rules:
            if rule_def["rule"] in skipped:
                continue
            rule_defs.append({"column": col_name, **rule_def})

    return rule_defs, col_type_map


# ──────────────────────────────────────────────────────────────────
#  CORE: executa cada regra
# ──────────────────────────────────────────────────────────────────


def _run_rule(
    df: DataFrame,
    table_name: str,
    column: str,
    rule: str,
    params: Any = None,
    ref_df: DataFrame = None,
    ref_column: str = None,
    _total_rows: int = None,
) -> dict:

    total_rows = _total_rows if _total_rows is not None else df.count()
    col_expr = F.col(column)
    now = datetime.now()

    base = {
        "table_name": table_name,
        "column_name": column,
        "rule": rule,
        "rule_name": None,  # preenchido no run_dq_checks via rule_def.get("name")
        "params": str(params) if params is not None and not callable(params) else None,
        "total_rows": total_rows,
        "run_timestamp": now,
    }

    try:
        if rule == "not_null":
            failed = df.filter(col_expr.isNull()).count()

        elif rule == "unique":
            total_non_null = df.filter(col_expr.isNotNull()).count()
            distinct = df.select(column).distinct().count()
            failed = total_non_null - distinct

        elif rule == "min_value":
            failed = df.filter(col_expr.isNotNull() & (col_expr < params)).count()

        elif rule == "max_value":
            failed = df.filter(col_expr.isNotNull() & (col_expr > params)).count()

        elif rule == "accepted_values":
            failed = df.filter(col_expr.isNotNull() & ~col_expr.isin(params)).count()

        elif rule == "regex":
            failed = df.filter(col_expr.isNotNull() & ~col_expr.rlike(params)).count()

        elif rule == "min_length":
            failed = df.filter(
                col_expr.isNotNull() & (F.length(col_expr) < params)
            ).count()

        elif rule == "max_length":
            failed = df.filter(
                col_expr.isNotNull() & (F.length(col_expr) > params)
            ).count()

        elif rule == "not_negative":
            failed = df.filter(col_expr.isNotNull() & (col_expr < 0)).count()

        elif rule == "freshness_days":
            max_date = df.agg(F.max(col_expr)).collect()[0][0]
            if max_date is None:
                failed = total_rows
            else:
                from datetime import date as date_type, datetime as datetime_type

                if isinstance(max_date, datetime_type):
                    delta = (now - max_date).days
                elif isinstance(max_date, date_type):
                    delta = (now.date() - max_date).days
                else:
                    # fallback para Timestamp do Spark (pyspark.sql.types)
                    delta = (now.date() - max_date.date()).days
                failed = total_rows if delta > params else 0

        elif rule == "referential":
            if ref_df is None or ref_column is None:
                raise ValueError("Regra 'referential' exige ref_df e ref_column.")
            ref_keys = ref_df.select(F.col(ref_column).alias("__ref__")).distinct()
            failed = (
                df.filter(col_expr.isNotNull())
                .join(ref_keys, col_expr == F.col("__ref__"), "left_anti")
                .count()
            )

        elif rule == "custom":
            if not callable(params):
                raise ValueError("Regra 'custom' exige params como callable.")
            failed = params(df).count()

        # ── cross_table ───────────────────────────────────────────
        # Consistência entre dois DataFrames: verifica se o valor de
        # uma coluna bate com a coluna correspondente em outro DF,
        # fazendo join por uma chave comum.
        #
        # params = {
        #   "ref_df"     : DataFrame,  # DF de referência
        #   "join_key"   : str,        # coluna chave para o join (mesmo nome nos dois DFs)
        #                              # ou dict {"left": "col_a", "right": "col_b"} se diferente
        #   "ref_column" : str,        # coluna a comparar no ref_df
        # }
        elif rule == "cross_table":
            ref_df_ct = params["ref_df"]
            join_key = params["join_key"]
            ref_col_ct = params["ref_column"]

            # Suporta chaves com nomes diferentes nos dois DFs
            if isinstance(join_key, dict):
                left_key = join_key["left"]
                right_key = join_key["right"]
            else:
                left_key = join_key
                right_key = join_key

            ref_renamed = ref_df_ct.select(
                F.col(right_key).alias("__join_key__"),
                F.col(ref_col_ct).alias("__ref_value__"),
            )

            failed = (
                df.filter(col_expr.isNotNull())
                .join(ref_renamed, df[left_key] == F.col("__join_key__"), "left")
                .filter(
                    F.col("__ref_value__").isNull()  # chave não encontrada
                    | (col_expr != F.col("__ref_value__"))  # valor divergente
                )
                .count()
            )

        # ── accuracy ──────────────────────────────────────────────
        # Precisão contra uma fonte de verdade: igual ao cross_table,
        # mas aceita uma tolerância numérica para diferenças de
        # arredondamento ou variações aceitáveis.
        #
        # params = {
        #   "ref_df"     : DataFrame,  # fonte de verdade
        #   "join_key"   : str | dict, # chave do join (mesmo formato do cross_table)
        #   "ref_column" : str,        # coluna na fonte de verdade
        #   "tolerance"  : float,      # diferença máxima aceitável (opcional, padrão 0)
        # }
        elif rule == "accuracy":
            ref_df_ac = params["ref_df"]
            join_key = params["join_key"]
            ref_col_ac = params["ref_column"]
            tolerance = params.get("tolerance", 0)

            if isinstance(join_key, dict):
                left_key = join_key["left"]
                right_key = join_key["right"]
            else:
                left_key = join_key
                right_key = join_key

            ref_renamed = ref_df_ac.select(
                F.col(right_key).alias("__join_key__"),
                F.col(ref_col_ac).alias("__ref_value__"),
            )

            joined = df.filter(col_expr.isNotNull()).join(
                ref_renamed, df[left_key] == F.col("__join_key__"), "left"
            )

            if tolerance == 0:
                failed = joined.filter(
                    F.col("__ref_value__").isNull()
                    | (col_expr != F.col("__ref_value__"))
                ).count()
            else:
                failed = joined.filter(
                    F.col("__ref_value__").isNull()
                    | (F.abs(col_expr - F.col("__ref_value__")) > tolerance)
                ).count()

        else:
            raise ValueError(f"Regra desconhecida: '{rule}'")

        passed = total_rows - failed
        pct_passed = round(passed / total_rows * 100, 2) if total_rows > 0 else 0.0
        status = "PASS" if failed == 0 else ("WARN" if pct_passed >= 95 else "FAIL")

        return {
            **base,
            "passed_rows": passed,
            "failed_rows": failed,
            "pct_passed": pct_passed,
            "status": status,
            "error_message": None,
        }

    except Exception as e:
        return {
            **base,
            "passed_rows": None,
            "failed_rows": None,
            "pct_passed": None,
            "status": "ERROR",
            "error_message": str(e),
        }


# ──────────────────────────────────────────────────────────────────
#  SCHEMA DO RESULTADO
# ──────────────────────────────────────────────────────────────────

RESULT_SCHEMA = StructType(
    [
        StructField("table_name", StringType(), True),
        StructField("column_name", StringType(), True),
        StructField("column_type", StringType(), True),
        StructField("rule", StringType(), True),
        StructField("rule_name", StringType(), True),
        StructField("params", StringType(), True),
        StructField("total_rows", LongType(), True),
        StructField("passed_rows", LongType(), True),
        StructField("failed_rows", LongType(), True),
        StructField("pct_passed", DoubleType(), True),
        StructField("status", StringType(), True),
        StructField("quality_dimension", StringType(), True),
        StructField("error_message", StringType(), True),
        StructField("run_timestamp", TimestampType(), True),
    ]
)

_DIMENSION_MAP = {
    "not_null": "Completude",
    "unique": "Unicidade",
    "referential": "Integridade",
    "freshness_days": "Atualidade",
    "cross_table": "Consistência",
    "accuracy": "Precisão",
}


def _dimension(rule: str) -> str:
    return _DIMENSION_MAP.get(rule, "Validade")


# ──────────────────────────────────────────────────────────────────
#  FUNÇÃO PRINCIPAL
# ──────────────────────────────────────────────────────────────────


def run_dq_checks(spark: SparkSession, checks: list[dict]) -> DataFrame:
    """
    Executa todas as verificações de qualidade e retorna um
    DataFrame consolidado com os resultados.

    Cada item de `checks` pode ter:
      table_name  : str        — nome lógico da tabela
      df          : DataFrame  — DataFrame Spark
      columns     : dict       — mapeamento coluna → tipo (opcional)
                                 colunas não declaradas são inferidas automaticamente
      extra_rules : list       — regras adicionais além dos tipos (opcional)
      skip_rules  : dict       — remove regras específicas de uma coluna (opcional)
    """
    results = []

    for check in checks:
        table_name = check["table_name"]
        df = check["df"]
        columns = check.get("columns", {})
        extra_rules = check.get("extra_rules", [])
        skip_rules = check.get("skip_rules", {})
        column_groups = check.get("column_groups", [])

        total_rows = df.count()
        n_declared = len(columns)
        n_auto = len(df.columns) - len(
            [c for c in columns if columns[c].get("type") != "texto_opcional"]
        )

        print(
            f"\n▶ [{table_name}]  {total_rows:,} linhas  |  "
            f"{n_declared} declaradas  |  "
            f"{len(df.columns) - n_declared} inferidas automaticamente"
        )

        rule_defs, col_type_map = _build_rule_defs(
            df, columns, skip_rules, column_groups
        )

        # Filtra extra_rules — ignora regras que referenciam colunas inexistentes
        df_columns = set(df.columns)
        extra_rules_validas = []
        for r in extra_rules:
            col = r.get("column", "")
            if col and col not in df_columns:
                print(
                    f"   ⚠️  extra_rule ignorada — coluna '{col}' não existe em [{table_name}]"
                )
            else:
                extra_rules_validas.append(r)

        rule_defs += extra_rules_validas

        for rule_def in rule_defs:
            column = rule_def["column"]
            rule = rule_def["rule"]
            params = rule_def.get("params")
            ref_df = rule_def.get("ref_df")
            ref_col = rule_def.get("ref_column")
            col_type = col_type_map.get(column, "extra")

            rule_name = rule_def.get("name", rule)
            print(f"   ├─ {column:<25} [{col_type:<22}]  {rule_name}", end=" ... ")

            result = _run_rule(
                df,
                table_name,
                column,
                rule,
                params,
                ref_df,
                ref_col,
                _total_rows=total_rows,
            )

            result["column_type"] = col_type
            result["rule_name"] = rule_def.get("name", rule)
            result["quality_dimension"] = rule_def.get("dimension", _dimension(rule))
            results.append(result)

            icon = {"PASS": "✅", "WARN": "⚠️", "FAIL": "❌", "ERROR": "💥"}.get(
                result["status"], "?"
            )
            print(f"{icon}  {result['pct_passed']}%")

    result_df = spark.createDataFrame(results, schema=RESULT_SCHEMA)
    print(f"\n✔ DQ concluído — {len(results)} verificações executadas.")
    return result_df


# ──────────────────────────────────────────────────────────────────
#  EXEMPLO DE USO
# ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from datetime import date

    spark = SparkSession.builder.appName("DataQuality").getOrCreate()

    df_clientes = spark.createDataFrame(
        [
            (1, "Ana Silva", "ana@email.com", "SP", date(2024, 1, 10)),
            (2, "João Lima", None, "RJ", date(2024, 3, 5)),
            (3, None, "x", "XX", date(2023, 1, 1)),
            (4, "Maria Costa", "maria@email.com", "MG", date(2024, 6, 20)),
            (4, "Pedro Alves", "pedro@email.com", "SP", date(2024, 7, 1)),
        ],
        ["id", "nome", "email", "uf", "dt_cadastro"],
    )

    df_pedidos = spark.createDataFrame(
        [
            (101, 1, 250.0, "PAGO", date(2024, 7, 1)),
            (102, 2, -10.0, "PENDENTE", date(2024, 7, 2)),
            (103, 9, 80.0, "CANCELADO", date(2024, 7, 3)),
            (104, 4, 1500.0, "PAGO", date(2024, 7, 4)),
            (105, 1, None, "RASCUNHO", date(2024, 7, 5)),
        ],
        ["pedido_id", "cliente_id", "valor", "status", "dt_pedido"],
    )

    checks = [
        # ── Sem declarar nada: tudo inferido automaticamente ──────
        {
            "table_name": "clientes",
            "df": df_clientes,
        },
        # ── Usando column_groups para aplicar regras em lote ──────
        {
            "table_name": "pedidos",
            "df": df_pedidos,
            "column_groups": [
                # Todas as colunas da lista recebem o mesmo tipo
                {"columns": ["pedido_id"], "type": "id"},
                {"columns": ["valor"], "type": "monetario", "max_value": 10000},
                {"columns": ["dt_pedido"], "type": "data_evento", "freshness_days": 30},
            ],
            # Declarações individuais sobrescrevem o grupo quando necessário
            "columns": {
                "cliente_id": {
                    "type": "chave_estrangeira",
                    "ref_df": df_clientes,
                    "ref_column": "id",
                },
                "status": {
                    "type": "categoria",
                    "accepted_values": ["PAGO", "PENDENTE", "CANCELADO"],
                },
            },
            # Regra de negócio extra
            "extra_rules": [
                {
                    "column": "valor",
                    "rule": "custom",
                    "params": lambda df: df.filter(
                        (F.col("valor") > 500) & (F.col("status") != "PAGO")
                    ),
                }
            ],
        },
    ]

    # ── Exemplos de cross_table e accuracy ────────────────────────
    #
    # Suponha que exista uma tabela de faturamento e um catálogo de
    # produtos com os preços oficiais:

    df_faturamento = spark.createDataFrame(
        [
            (101, 250.0),  # bate com pedidos
            (102, 999.0),  # diverge: pedido tem -10.0
            (104, 1500.0),  # bate
        ],
        ["pedido_id", "valor_faturado"],
    )

    df_catalogo = spark.createDataFrame(
        [
            (1, 80.0),  # produto 1: preço oficial 80.0
            (2, 120.0),  # produto 2: preço oficial 120.0
        ],
        ["produto_id", "preco_oficial"],
    )

    df_itens_pedido = spark.createDataFrame(
        [
            (101, 1, 80.0),  # ok
            (102, 2, 121.0),  # diverge 1.0 do catálogo
            (103, 1, 79.5),  # diverge 0.5 — dentro de tolerance=1.0
            (104, 2, 130.0),  # diverge 10.0 — fora de tolerance=1.0
        ],
        ["pedido_id", "produto_id", "preco_praticado"],
    )

    checks_extra = [
        {
            "table_name": "pedidos",
            "df": df_pedidos,
            "extra_rules": [
                # cross_table: valor do pedido deve bater com o faturamento
                # chave com o mesmo nome nos dois DFs → join_key como string
                {
                    "column": "valor",
                    "rule": "cross_table",
                    "params": {
                        "ref_df": df_faturamento,
                        "join_key": "pedido_id",
                        "ref_column": "valor_faturado",
                    },
                },
            ],
        },
        {
            "table_name": "itens_pedido",
            "df": df_itens_pedido,
            "extra_rules": [
                # accuracy: preço praticado deve bater com o catálogo
                # com tolerância de R$1,00 pra diferenças de arredondamento
                # chave com nomes diferentes nos dois DFs → join_key como dict
                {
                    "column": "preco_praticado",
                    "rule": "accuracy",
                    "params": {
                        "ref_df": df_catalogo,
                        "join_key": {"left": "produto_id", "right": "produto_id"},
                        "ref_column": "preco_oficial",
                        "tolerance": 1.0,
                    },
                },
            ],
        },
    ]

    resultado_extra = run_dq_checks(spark, checks_extra)
    resultado_extra.show(truncate=False)

    resultado_df = run_dq_checks(spark, checks)
    resultado_df.show(truncate=False)

    # Salva pro Power BI
    # resultado_df.write.format("delta").mode("overwrite").save("/mnt/datalake/dq/resultados")
    resultado_df.write.mode("overwrite").option("header", True).csv("/tmp/dq_results")
    print("\n✅ Resultado salvo em /tmp/dq_results")
