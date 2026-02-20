# 0.0.1v Trailblazer
# %pip install "autogluon>=1.0.0" "scikit-learn>=1.3.0" "pandas>=2.0.0" "scipy>=1.9.0" "matplotlib>=3.7.0" "seaborn>=0.12.0" "joblib>=1.3.0"


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
import re
import joblib
import os
from scipy.stats import entropy, skew
from scipy.stats.contingency import association
from autogluon.tabular import TabularPredictor
from sklearn.metrics import (
    confusion_matrix,
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score,
    accuracy_score,
    log_loss,
    f1_score,
    recall_score,
)
from sklearn.calibration import calibration_curve
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedKFold


# ---------------------------------------------------------------------------
# Utilitários de Associação
# ---------------------------------------------------------------------------


def _theils_u(x: pd.Series, y: pd.Series) -> float:
    """
    Theil's U (Incerteza Assimétrica) de x → y.
    Mede o quanto conhecer x reduz a incerteza sobre y.
    Retorna valor entre 0 (nenhuma associação) e 1 (associação perfeita).
    """
    x = x.astype(str).fillna("__NA__")
    y = y.astype(str).fillna("__NA__")

    s_xy = _conditional_entropy(x, y)
    x_counter = x.value_counts(normalize=True)
    s_x = entropy(x_counter)

    if s_x == 0:
        return 1.0  # x é constante → sem incerteza para reduzir
    return (s_x - s_xy) / s_x


def _conditional_entropy(x: pd.Series, y: pd.Series) -> float:
    """H(x|y): entropia condicional de x dado y."""
    y_vals = y.unique()
    h = 0.0
    for yv in y_vals:
        mask = y == yv
        p_y = mask.mean()
        x_given_y = x[mask].value_counts(normalize=True)
        h += p_y * entropy(x_given_y)
    return h


def _cramers_v(x: pd.Series, y: pd.Series) -> float:
    """Cramér's V entre duas variáveis categóricas."""
    ct = pd.crosstab(x.astype(str).fillna("__NA__"), y.astype(str).fillna("__NA__"))
    return association(ct.values, method="cramer")


def _eta_squared(cat: pd.Series, num: pd.Series) -> float:
    """Eta² — associação entre categórica e numérica (via ANOVA one-way)."""
    cat = cat.astype(str).fillna("__NA__")
    groups = [num[cat == c].dropna() for c in cat.unique()]
    grand_mean = num.dropna().mean()
    ss_between = sum(len(g) * (g.mean() - grand_mean) ** 2 for g in groups)
    ss_total = ((num.dropna() - grand_mean) ** 2).sum()
    return ss_between / ss_total if ss_total > 0 else 0.0


# ---------------------------------------------------------------------------
# Classe Principal
# ---------------------------------------------------------------------------


class AutoClassificationEngine:

    REQUIRED_PARAMS = ["target"]

    def __init__(self, key_params: dict):
        self._validate_params(key_params)

        self.params = key_params
        self.target = key_params["target"]
        self.pipeline = None
        self.predictor = None
        self.selected_features = None
        self.feature_importance = None

        # Estado aprendido no Treino
        self._log_cols: list = []
        self._outlier_bounds: dict = {}
        self._rare_categories: dict = {}

        # Rastreabilidade
        self.eliminated_features = {
            "leakage": [],
            "alta_cardinalidade": [],
            "colinearidade": [],
            "constantes_pos_rare": [],
            "importancia_nula": [],  # <-- NOVA LINHA AQUI
        }

        # Relatório de associações (preenchido no fit)
        self.association_report: dict = {}

    # -----------------------------------------------------------------------
    # 0. VALIDAÇÃO DE PARÂMETROS
    # -----------------------------------------------------------------------

    def _validate_params(self, params: dict):
        missing = [p for p in self.REQUIRED_PARAMS if p not in params]
        if missing:
            raise ValueError(f"Parâmetros obrigatórios ausentes: {missing}")

        valid_metrics = {
            "f1",
            "roc_auc",
            "accuracy",
            "log_loss",
            "average_precision",
            "recall",
            "precision",
        }
        metric = params.get("eval_metric", "f1")
        if metric not in valid_metrics:
            raise ValueError(
                f"eval_metric '{metric}' inválido. Opções: {valid_metrics}"
            )

        valid_presets = {
            "best_quality",
            "high_quality",
            "good_quality",
            "medium_quality",
            "optimize_for_deployment",
        }
        preset = params.get("presets", "best_quality")
        if preset not in valid_presets:
            raise ValueError(f"presets '{preset}' inválido. Opções: {valid_presets}")

    # -----------------------------------------------------------------------
    # 1. LIMPEZA E SANITY CHECKS
    # -----------------------------------------------------------------------

    def _standardize_and_clean(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        null_patterns = r"(?i)^(na|nan|null|none|n/a|-|\s*|unknown|\?)$"
        df = df.replace(null_patterns, np.nan, regex=True)

        exclude = self.params.get("features_to_exclude", [])
        df = df.drop(columns=[c for c in exclude if c in df.columns], errors="ignore")

        type_map = self.params.get("force_types", {})
        for dtype, patterns in type_map.items():
            for p in patterns:
                cols = (
                    [c for c in df.columns if re.search(p, c)]
                    if any(x in p for x in "^$*")
                    else [p]
                )
                for c in cols:
                    if c in df.columns:
                        if dtype in ["float", "int"]:
                            df[c] = pd.to_numeric(df[c], errors="coerce")
                        else:
                            df[c] = df[c].astype(str).replace("nan", np.nan)
        return df

    def _sanity_check(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Detecta e remove Leakage (numérico via Pearson + categórico via Theil's U)
        e variáveis de alta cardinalidade (provável IDs).
        """
        to_drop = []
        target_series = df[self.target]

        # A. Leakage numérico — Pearson > 98%
        num_cols = df.select_dtypes(include=[np.number]).columns
        if self.target in num_cols:
            corrs = df[num_cols].corr()[self.target].abs().sort_values(ascending=False)
            leaks = corrs[(corrs > 0.98) & (corrs.index != self.target)]
            if not leaks.empty:
                self.eliminated_features["leakage"].extend(leaks.index.tolist())
                to_drop.extend(leaks.index.tolist())

        # B. Leakage categórico — Theil's U > 98%
        cat_cols = df.select_dtypes(include=["object", "category"]).columns
        for col in cat_cols:
            if col == self.target or col in to_drop:
                continue
            u = _theils_u(df[col], target_series)
            if u > 0.98:
                self.eliminated_features["leakage"].append(col)
                to_drop.append(col)

        # C. Alta cardinalidade (> 50% valores únicos → provável ID)
        obs_count = len(df)
        for col in cat_cols:
            if col == self.target or col in to_drop:
                continue
            if df[col].nunique() / obs_count > 0.5:
                self.eliminated_features["alta_cardinalidade"].append(col)
                to_drop.append(col)

        return df.drop(columns=to_drop)

    def _handle_multicollinearity(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Remove variáveis redundantes usando a métrica de associação correta para
        cada par de tipos (num-num: Pearson, cat-cat: Theil's U, num-cat: Eta²).

        Critério de desempate: mantém a variável com MAIOR correlação com o target,
        não simplesmente a primeira do triângulo superior.
        """
        threshold = self.params.get("pipeline_settings", {}).get("corr_threshold", 0.90)
        target_series = df[self.target]

        num_cols = [
            c for c in df.select_dtypes(include=[np.number]).columns if c != self.target
        ]
        cat_cols = [
            c
            for c in df.select_dtypes(include=["object", "category"]).columns
            if c != self.target
        ]
        all_feat = num_cols + cat_cols

        # --- Calcula associação de cada feature com o target ---
        # Usa _is_categorical_target para não tratar int binário como contínuo
        target_assoc = {}
        target_is_cat = self._is_categorical_target(target_series)

        for col in all_feat:
            col_is_num = pd.api.types.is_numeric_dtype(df[col])
            try:
                if col_is_num and not target_is_cat:
                    target_assoc[col] = abs(df[col].corr(target_series.astype(float)))
                elif not col_is_num and target_is_cat:
                    target_assoc[col] = _theils_u(df[col], target_series)
                elif col_is_num and target_is_cat:
                    target_assoc[col] = _eta_squared(target_series, df[col])
                else:
                    target_assoc[col] = _eta_squared(
                        df[col], target_series.astype(float)
                    )
            except Exception:
                target_assoc[col] = 0.0

        # --- Calcula associação entre pares de features ---
        to_drop = set()

        for i, col_a in enumerate(all_feat):
            if col_a in to_drop:
                continue
            for col_b in all_feat[i + 1 :]:
                if col_b in to_drop:
                    continue

                a_is_num = pd.api.types.is_numeric_dtype(df[col_a])
                b_is_num = pd.api.types.is_numeric_dtype(df[col_b])

                try:
                    if a_is_num and b_is_num:
                        assoc = abs(df[col_a].corr(df[col_b]))
                    elif not a_is_num and not b_is_num:
                        # Theil's U é assimétrico → máximo das duas direções
                        assoc = max(
                            _theils_u(df[col_a], df[col_b]),
                            _theils_u(df[col_b], df[col_a]),
                        )
                    else:
                        # Par misto (cat ↔ num).
                        # Eta² mede "cat explica num" — só faz sentido nessa direção.
                        # Para capturar a associação simétrica, usamos também Pearson
                        # com a categórica codificada via label encoding ordinal simples.
                        cat_col, num_col = (
                            (col_a, col_b) if not a_is_num else (col_b, col_a)
                        )
                        eta = _eta_squared(df[cat_col], df[num_col])
                        # Pearson com codes da categórica como proxy da direção inversa
                        codes = df[cat_col].astype("category").cat.codes.astype(float)
                        pearson = abs(codes.corr(df[num_col]))
                        assoc = max(eta, pearson)
                except Exception:
                    assoc = 0.0

                if assoc > threshold:
                    assoc_a = target_assoc.get(col_a, 0.0)
                    assoc_b = target_assoc.get(col_b, 0.0)

                    # Desempate robusto: se a diferença for < 1%, considera empate
                    # e remove a feature com MAIS valores únicos (mais propensa a
                    # overfitting). Se ainda empatar, remove col_b por convenção.
                    if abs(assoc_a - assoc_b) < 0.01:
                        loser = (
                            col_a
                            if df[col_a].nunique() > df[col_b].nunique()
                            else col_b
                        )
                    else:
                        loser = col_b if assoc_a >= assoc_b else col_a

                    to_drop.add(loser)

        if to_drop:
            self.eliminated_features["colinearidade"].extend(list(to_drop))
            df = df.drop(columns=list(to_drop))

        return df

    # -----------------------------------------------------------------------
    # 2. ENGENHARIA DE FEATURES
    # -----------------------------------------------------------------------

    def _extract_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        date_cols = df.select_dtypes(include=["datetime64"]).columns
        for col in date_cols:
            df[f"{col}_month"] = df[col].dt.month
            df[f"{col}_weekday"] = df[col].dt.weekday
            df[f"{col}_hour_sin"] = np.sin(2 * np.pi * df[col].dt.hour / 24)
            df = df.drop(columns=[col])
        return df

    def _handle_rare_labels(
        self, df: pd.DataFrame, is_train=True, limit=0.01
    ) -> pd.DataFrame:
        cat_cols = df.select_dtypes(include=["object", "category"]).columns
        for col in cat_cols:
            if col == self.target:
                continue
            # CORREÇÃO: Converte a coluna para string para evitar erro de Category Index
            if df[col].dtype.name == "category":
                df[col] = df[col].astype(str)

            if is_train:
                freq = df[col].value_counts(normalize=True)
                self._rare_categories[col] = freq[freq >= limit].index.tolist()
            if col in self._rare_categories:
                df[col] = df[col].where(
                    df[col].isin(self._rare_categories[col]), other="OTHER"
                )

        # Remove colunas que viraram constantes após o agrupamento
        if is_train:
            const_cols = [
                c for c in cat_cols if c != self.target and df[c].nunique() <= 1
            ]
            if const_cols:
                self.eliminated_features["constantes_pos_rare"].extend(const_cols)
                df = df.drop(columns=const_cols)

        return df

    def _handle_outliers_and_log(self, df: pd.DataFrame, is_train=True) -> pd.DataFrame:
        df_num = df.select_dtypes(include=[np.number])
        for col in df_num.columns:
            if col == self.target:
                continue

            if is_train:
                Q1, Q3 = df[col].quantile([0.25, 0.75])
                IQR = Q3 - Q1
                if IQR > 0:
                    self._outlier_bounds[col] = (Q1 - 1.5 * IQR, Q3 + 1.5 * IQR)
                if abs(skew(df[col].dropna())) > 0.75 and df[col].min() >= 0:
                    self._log_cols.append(col)

            if col in self._outlier_bounds:
                lower, upper = self._outlier_bounds[col]
                df[col] = np.clip(df[col], lower, upper)
            if col in self._log_cols:
                df[col] = np.log1p(df[col])

        return df

    # -----------------------------------------------------------------------
    # 3. ANÁLISE DE ASSOCIAÇÃO (RELATÓRIO)
    # -----------------------------------------------------------------------

    @staticmethod
    def _is_categorical_target(series: pd.Series) -> bool:
        """
        Determina se o target deve ser tratado como categórico.
        Regra: dtype objeto/category OU numérico com <= 20 classes únicas
        (cobre targets binários e multiclasse codificados como int).
        """
        if not pd.api.types.is_numeric_dtype(series):
            return True
        return series.nunique() <= 20

    def _compute_association_report(self, df: pd.DataFrame) -> dict:
        """
        Calcula associação de cada feature com o target e entre as próprias features.
        Usa a métrica mais adequada para cada par de tipos:
          - num ↔ num (target contínuo)  : Pearson
          - cat → cat (target categórico): Theil's U
          - num → cat (target categórico): Eta²
          - cat → num (target contínuo)  : Eta²
        """
        feat_cols = [c for c in df.columns if c != self.target]
        target_series = df[self.target]

        # Semântica de "é categórico?" evita tratar targets int binários como contínuos
        target_is_cat = self._is_categorical_target(target_series)

        # Feature → Target
        feat_target = {}
        for col in feat_cols:
            col_is_num = pd.api.types.is_numeric_dtype(df[col])
            try:
                if col_is_num and not target_is_cat:
                    # Ambos contínuos → Pearson
                    v = abs(df[col].corr(target_series.astype(float)))
                    metric = "Pearson"
                elif not col_is_num and target_is_cat:
                    # Ambos categóricos → Theil's U (feature → target)
                    v = _theils_u(df[col], target_series)
                    metric = "Theil's U"
                elif col_is_num and target_is_cat:
                    # Feature numérica, target categórico → Eta²
                    v = _eta_squared(target_series, df[col])
                    metric = "Eta²"
                else:
                    # Feature categórica, target contínuo → Eta²
                    v = _eta_squared(df[col], target_series.astype(float))
                    metric = "Eta²"
            except Exception:
                v, metric = 0.0, "Erro"
            feat_target[col] = {"valor": round(v, 4), "metrica": metric}

        # Feature × Feature (matriz)
        n = len(feat_cols)
        matrix = pd.DataFrame(np.eye(n), index=feat_cols, columns=feat_cols)
        metric_matrix = pd.DataFrame("—", index=feat_cols, columns=feat_cols)

        for i, col_a in enumerate(feat_cols):
            for j, col_b in enumerate(feat_cols):
                if i >= j:
                    continue
                a_is_num = pd.api.types.is_numeric_dtype(df[col_a])
                b_is_num = pd.api.types.is_numeric_dtype(df[col_b])
                try:
                    if a_is_num and b_is_num:
                        v = abs(df[col_a].corr(df[col_b]))
                        m = "Pearson"
                    elif not a_is_num and not b_is_num:
                        v = max(
                            _theils_u(df[col_a], df[col_b]),
                            _theils_u(df[col_b], df[col_a]),
                        )
                        m = "Theil's U"
                    else:
                        cat_col = col_a if not a_is_num else col_b
                        num_col = col_b if not a_is_num else col_a
                        eta = _eta_squared(df[cat_col], df[num_col])
                        codes = df[cat_col].astype("category").cat.codes.astype(float)
                        pearson = abs(codes.corr(df[num_col]))
                        v = max(eta, pearson)
                        m = "Eta²/Pearson"
                except Exception:
                    v, m = 0.0, "Erro"

                matrix.loc[col_a, col_b] = v
                matrix.loc[col_b, col_a] = v
                metric_matrix.loc[col_a, col_b] = m
                metric_matrix.loc[col_b, col_a] = m

        return {
            "feat_target": feat_target,
            "feat_feat_matrix": matrix.astype(float),
            "metric_matrix": metric_matrix,
        }

    # -----------------------------------------------------------------------
    # 4. PIPELINE SKLEARN
    # -----------------------------------------------------------------------

    def _build_sklearn_pipeline(self, X: pd.DataFrame) -> Pipeline:
        numeric_features = X.select_dtypes(include=["int64", "float64"]).columns
        categorical_features = X.select_dtypes(include=["object", "category"]).columns

        num_steps = [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
        if self.params.get("pipeline_settings", {}).get("use_pca"):
            n_comps = self.params["pipeline_settings"].get("pca_components", 0.95)
            num_steps.append(("pca", PCA(n_components=n_comps)))

        num_transformer = Pipeline(steps=num_steps)
        cat_transformer = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="constant", fill_value="MISSING"))
            ]
        )

        preprocessor = ColumnTransformer(
            transformers=[
                ("num", num_transformer, numeric_features),
                ("cat", cat_transformer, categorical_features),
            ],
            verbose_feature_names_out=False,
        ).set_output(transform="pandas")

        return Pipeline(steps=[("preprocessor", preprocessor)])

    # -----------------------------------------------------------------------
    # 5. FIT
    # -----------------------------------------------------------------------

    def fit(self, train_data: pd.DataFrame, time_limit: int = None):
        """
        time_limit: segundos para o AutoGluon. Se None, usa o valor em key_params
                    ou 300s como fallback.
        """
        print("\n🚀 --- TREINAMENTO INICIADO ---")

        # Resolve time_limit com hierarquia: argumento > params > fallback
        if time_limit is None:
            time_limit = self.params.get("time_limit", 300)

        df = self._standardize_and_clean(train_data)
        df = self._sanity_check(df)
        df = self._handle_multicollinearity(df)
        df = self._extract_temporal_features(df)
        df = self._handle_rare_labels(df, is_train=True)
        if self.params.get("handle_outliers", True):
            df = self._handle_outliers_and_log(df, is_train=True)

        self.selected_features = [c for c in df.columns if c != self.target]
        X = df[self.selected_features]
        y = df[self.target]

        # Relatório de associações (antes do pipeline sklearn)
        print("\n🔗 Calculando associações (Pearson / Theil's U / Eta²)...")
        self.association_report = self._compute_association_report(df)
        print("   ✅ Relatório de associações pronto.")

        # -------------------------------------------------------------------
        # [NOVO] ETAPA 3: PRÉ-TREINO RÁPIDO PARA SELEÇÃO DE FEATURES
        # -------------------------------------------------------------------
        chosen_metric = self.params.get("eval_metric", "f1")

        if self.params.get("use_importance_filter", False):
            print("\n🔍 Executando Pré-treino Rápido para Seleção de Features...")
            train_pre = X.copy()
            train_pre[self.target] = y.values
            pre_time_limit = min(60, max(20, time_limit // 6))

            import warnings

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                pre_predictor = TabularPredictor(
                    label=self.target, eval_metric=chosen_metric, verbosity=0
                ).fit(
                    train_pre,
                    time_limit=pre_time_limit,
                    presets="optimize_for_deployment",
                    hyperparameters={"GBM": {}, "CAT": {}},
                    ag_args_ensemble={"fold_fitting_strategy": "sequential_local"},
                    num_cpus=self.params.get("num_cpus", 6),
                )

            fi = pre_predictor.feature_importance(train_pre)

            # --- ABORDAGEM 1: Importância positiva E significância estatística (p < 0.05) ---
            good_features = fi[
                (fi["importance"] > 0) & (fi["p_value"] < 0.05)
            ].index.tolist()
            # good_features = fi[fi["importance"] > 0].index.tolist()

            features_removidas = set(self.selected_features) - set(good_features)
            self.eliminated_features["importancia_nula"] = list(features_removidas)
            self.selected_features = good_features
            X = X[self.selected_features]  # Atualiza o X cortando as ruins
            print(
                f"   => Mantidas: {len(good_features)} | Removidas (Imp <= 0): {len(features_removidas)}"
            )

        print("\n📋 RESUMO DA FILTRAGEM DE VARIÁVEIS:")
        for reason, cols in self.eliminated_features.items():
            label = {
                "leakage": "Removidas por Leakage",
                "alta_cardinalidade": "Removidas por Alta Cardinalidade",
                "colinearidade": "Removidas por Colinearidade (>threshold)",
                "constantes_pos_rare": "Removidas por Constantes pós-RareLabel",
                "importancia_nula": "Removidas por Importância Nula (AutoGluon)",
            }[reason]
            print(f"   => {label}: {cols or 'Nenhuma'}")
        print(
            f"   => Features Finais ({len(self.selected_features)}): {self.selected_features}"
        )

        # -------------------------------------------------------------------
        # [NOVO] ETAPA 4: PIPELINE SKLEARN OPCIONAL
        # -------------------------------------------------------------------
        if self.params.get("use_sklearn_pipeline", True):
            self.pipeline = self._build_sklearn_pipeline(X)
            X_trans = self.pipeline.fit_transform(X, y)
            train_final = X_trans.copy()
        else:
            self.pipeline = None
            train_final = X.copy()

        train_final[self.target] = y.values

        chosen_preset = self.params.get("presets", "best_quality")
        print(
            f"\n🎯 Métrica: {chosen_metric} | Preset: {chosen_preset} | Time Limit: {time_limit}s"
        )

        # -------------------------------------------------------------------
        # [NOVO] ETAPA 4b: PODA DE MODELOS E TRAVA DO RAY
        # -------------------------------------------------------------------
        hyperparams = "default"
        if self.params.get("prune_models", False):
            hyperparams = {"GBM": {}, "CAT": {}, "XGB": {}, "FASTAI": {}}

        ag_args_ensemble = self.params.get(
            "ag_args_ensemble", {"fold_fitting_strategy": "sequential_local"}
        )

        self.predictor = TabularPredictor(
            label=self.target, eval_metric=chosen_metric
        ).fit(
            train_final,
            time_limit=time_limit,
            presets=chosen_preset,
            num_cpus=self.params.get("num_cpus", 6),
            ag_args_ensemble=ag_args_ensemble,
            hyperparameters=hyperparams,
        )

        try:
            self.feature_importance = self.predictor.feature_importance(train_final)
        except Exception:
            print("⚠️ Feature importance indisponível.")

        print("\n✅ Modelo Treinado com Sucesso.")
        return self.predictor.leaderboard(train_final, silent=True)

    # -----------------------------------------------------------------------
    # 5b. CROSS-VALIDATION EXTERNO
    # -----------------------------------------------------------------------

    def cross_validate(
        self,
        train_data: pd.DataFrame,
        n_folds: int = 5,
        time_limit_per_fold: int = None,
    ) -> pd.DataFrame:
        """
        Cross-validation estratificado externo com K folds.

        Para cada fold:
          - Aplica TODO o pipeline de pré-processamento de forma independente
            (evita data leakage entre folds)
          - Treina um AutoGluon completo no fold de treino
          - Avalia no fold de validação
          - Coleta todas as métricas + curvas

        Retorna um DataFrame com métricas por fold + média ± std.
        Os resultados ficam em self.cv_results para uso no plot_cv_report().

        Parâmetros
        ----------
        n_folds : int
            Número de folds (default 5). Recomendado entre 5 e 10.
        time_limit_per_fold : int
            Segundos de treino por fold. Se None, usa time_limit dos params
            dividido por n_folds (com mínimo de 60s).
        """
        if self.selected_features is None:
            raise RuntimeError("Execute fit() antes de cross_validate().")

        # Time limit por fold: divide o budget total ou usa fallback
        total_tl = self.params.get("time_limit", 300)
        if time_limit_per_fold is None:
            time_limit_per_fold = max(60, total_tl // n_folds)

        print(f"\n🔄 --- CROSS-VALIDATION INICIADO ({n_folds} folds) ---")
        print(f"   Time limit por fold: {time_limit_per_fold}s")

        # Pré-processamento base (sanity check, multicolinearidade, etc.)
        # já foi feito no fit() — aqui reutilizamos selected_features e pipeline
        # mas RECRIAMOS um engine limpo por fold para evitar leakage de estado.
        df_base = self._standardize_and_clean(train_data)
        available = [
            c
            for c in df_base.columns
            if c in self.selected_features or c == self.target
        ]
        df_base = df_base[available]
        df_base = self._extract_temporal_features(df_base)

        X_raw = df_base[self.selected_features]
        y_raw = df_base[self.target]

        pos_label = self.predictor.positive_class
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
        y_strat = (y_raw.astype(str) == str(pos_label)).astype(int)

        fold_metrics = []
        fold_curves = []  # fpr, tpr por fold para plotagem
        oof_probs = np.zeros(len(y_raw))  # out-of-fold probabilities
        oof_true = np.zeros(len(y_raw))

        chosen_metric = self.params.get("eval_metric", "f1")
        chosen_preset = self.params.get("presets", "best_quality")

        for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X_raw, y_strat)):
            print(
                f"\n   📂 Fold {fold_idx + 1}/{n_folds} "
                f"(treino={len(train_idx)}, val={len(val_idx)})"
            )

            X_tr_raw = X_raw.iloc[train_idx].copy()
            X_vl_raw = X_raw.iloc[val_idx].copy()
            y_tr = y_raw.iloc[train_idx]
            y_vl = y_raw.iloc[val_idx]

            # --- Pré-processamento independente por fold ---
            # rare labels aprendidos só no fold de treino
            rare_cats_fold = {}
            cat_cols = X_tr_raw.select_dtypes(include=["object", "category"]).columns
            for col in cat_cols:
                freq = X_tr_raw[col].value_counts(normalize=True)
                rare_cats_fold[col] = freq[freq >= 0.01].index.tolist()
                X_tr_raw[col] = X_tr_raw[col].where(
                    X_tr_raw[col].isin(rare_cats_fold[col]), other="OTHER"
                )
                X_vl_raw[col] = X_vl_raw[col].where(
                    X_vl_raw[col].isin(rare_cats_fold[col]), other="OTHER"
                )

            # outliers/log aprendidos só no fold de treino
            log_cols_fold, bounds_fold = [], {}
            for col in X_tr_raw.select_dtypes(include=[np.number]).columns:
                Q1, Q3 = X_tr_raw[col].quantile([0.25, 0.75])
                IQR = Q3 - Q1
                if IQR > 0:
                    bounds_fold[col] = (Q1 - 1.5 * IQR, Q3 + 1.5 * IQR)
                if (
                    abs(skew(X_tr_raw[col].dropna())) > 0.75
                    and X_tr_raw[col].min() >= 0
                ):
                    log_cols_fold.append(col)
            for col, (lo, hi) in bounds_fold.items():
                X_tr_raw[col] = np.clip(X_tr_raw[col], lo, hi)
                X_vl_raw[col] = np.clip(X_vl_raw[col], lo, hi)
            for col in log_cols_fold:
                X_tr_raw[col] = np.log1p(X_tr_raw[col])
                X_vl_raw[col] = np.log1p(X_vl_raw[col])

            # Pipeline sklearn aprendido só no fold de treino
            pipe_fold = self._build_sklearn_pipeline(X_tr_raw)
            X_tr_t = pipe_fold.fit_transform(X_tr_raw, y_tr)
            X_vl_t = pipe_fold.transform(X_vl_raw)

            train_ag = X_tr_t.copy()
            train_ag[self.target] = y_tr.values

            # AutoGluon por fold (silencioso)
            import warnings, logging

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                logging.disable(logging.CRITICAL)
                pred_fold = TabularPredictor(
                    label=self.target,
                    eval_metric=chosen_metric,
                    verbosity=0,
                ).fit(
                    train_ag,
                    time_limit=time_limit_per_fold,
                    presets=chosen_preset,
                    num_cpus=self.params.get("num_cpus", 6),
                    ag_args_ensemble=self.params.get(
                        "ag_args_ensemble",
                        {"fold_fitting_strategy": "sequential_local"},
                    ),
                )
                logging.disable(logging.NOTSET)

            y_prob_df = pred_fold.predict_proba(X_vl_t)
            y_prob_pos = y_prob_df[pos_label]
            y_pred = pred_fold.predict(X_vl_t)
            y_bin = (y_vl.astype(str) == str(pos_label)).astype(int)

            # Salva OOF
            oof_probs[val_idx] = y_prob_pos.values
            oof_true[val_idx] = y_bin.values

            fpr, tpr, thresh_roc = roc_curve(y_bin, y_prob_pos)
            roc_auc = auc(fpr, tpr)
            pr_auc = average_precision_score(y_bin, y_prob_pos)
            youden_t, _, _ = self._youden_threshold(fpr, tpr, thresh_roc)
            y_pred_y = (y_prob_pos >= youden_t).astype(int)

            fold_metrics.append(
                {
                    "Fold": fold_idx + 1,
                    "AUC-ROC": round(roc_auc, 4),
                    "Gini": round(2 * roc_auc - 1, 4),
                    "Avg Precision": round(pr_auc, 4),
                    "F1 (thr=0.5)": round(f1_score(y_bin, y_pred, zero_division=0), 4),
                    "F1 (Youden)": round(f1_score(y_bin, y_pred_y, zero_division=0), 4),
                    "Recall (0.5)": round(
                        recall_score(y_bin, y_pred, zero_division=0), 4
                    ),
                    "Log-Loss": round(log_loss(y_bin, y_prob_pos), 4),
                    "Accuracy": round(accuracy_score(y_bin, y_pred), 4),
                }
            )
            fold_curves.append(
                {"fpr": fpr, "tpr": tpr, "auc": roc_auc, "fold": fold_idx + 1}
            )

            print(
                f"      AUC={roc_auc:.4f} | AP={pr_auc:.4f} | "
                f"F1={fold_metrics[-1]['F1 (thr=0.5)']:.4f}"
            )

        # --- Resumo OOF global ---
        fpr_oof, tpr_oof, thresh_oof = roc_curve(oof_true, oof_probs)
        oof_auc = auc(fpr_oof, tpr_oof)
        oof_ap = average_precision_score(oof_true, oof_probs)

        # --- DataFrame de métricas ---
        metrics_df = pd.DataFrame(fold_metrics).set_index("Fold")
        mean_row = metrics_df.mean().rename("Média")
        std_row = metrics_df.std().rename("Std")
        summary_df = pd.concat(
            [metrics_df, mean_row.to_frame().T, std_row.to_frame().T]
        )

        print(f"\n📊 RESUMO CROSS-VALIDATION ({n_folds} folds):")
        print(f"   AUC-ROC  : {mean_row['AUC-ROC']:.4f} ± {std_row['AUC-ROC']:.4f}")
        print(
            f"   Avg Prec : {mean_row['Avg Precision']:.4f} ± {std_row['Avg Precision']:.4f}"
        )
        print(
            f"   F1(0.5)  : {mean_row['F1 (thr=0.5)']:.4f} ± {std_row['F1 (thr=0.5)']:.4f}"
        )
        print(f"   AUC OOF  : {oof_auc:.4f}  (out-of-fold global)")
        print(f"   AP  OOF  : {oof_ap:.4f}  (out-of-fold global)")

        # Armazena para uso no plot_cv_report
        self.cv_results = {
            "summary_df": summary_df,
            "fold_curves": fold_curves,
            "oof_probs": oof_probs,
            "oof_true": oof_true,
            "fpr_oof": fpr_oof,
            "tpr_oof": tpr_oof,
            "oof_auc": oof_auc,
            "oof_ap": oof_ap,
            "n_folds": n_folds,
            "pos_label": pos_label,
        }
        return summary_df

    def plot_cv_report(self):
        """
        Plota o relatório visual do cross-validation:
          1. Tabela de métricas por fold (com média ± std)
          2. Curvas ROC por fold + ROC OOF
          3. Distribuição das métricas por fold (boxplot)
          4. Distribuição OOF de probabilidades
          5. Calibração OOF
        """
        if not hasattr(self, "cv_results") or not self.cv_results:
            raise RuntimeError("Execute cross_validate() antes de plot_cv_report().")

        cv = self.cv_results
        n = cv["n_folds"]
        cmap = plt.cm.get_cmap("tab10", n)

        fig, axes = plt.subplots(2, 3, figsize=(20, 12), constrained_layout=True)

        # --- 1. Tabela de métricas por fold ---
        ax = axes[0, 0]
        ax.axis("off")
        # summary_df tem índice misto (ints dos folds + strings "Média"/"Std")
        # reset_index() gera coluna "index", não "Fold" — renomeamos explicitamente
        df_table = cv["summary_df"].reset_index().rename(columns={"index": "Fold"})
        df_table["Fold"] = df_table["Fold"].astype(str)
        table = ax.table(
            cellText=df_table.round(4).values,
            colLabels=df_table.columns.tolist(),
            loc="center",
            cellLoc="center",
            bbox=[0, 0, 1, 1],
        )
        table.auto_set_font_size(False)
        table.set_fontsize(8.5)
        table.scale(1, 1.6)

        # Colore as linhas de Média e Std
        n_rows = len(df_table)
        for j in range(len(df_table.columns)):
            table[n_rows - 1, j].set_facecolor("#d4edda")  # Média = verde
            table[n_rows, j].set_facecolor("#fff3cd")  # Std   = amarelo

        ax.set_title(f"Métricas por Fold  (n={n})", fontweight="bold", fontsize=12)

        # --- 2. Curvas ROC por fold + OOF ---
        ax = axes[0, 1]
        for fc in cv["fold_curves"]:
            ax.plot(
                fc["fpr"],
                fc["tpr"],
                lw=1.2,
                alpha=0.6,
                color=cmap(fc["fold"] - 1),
                label=f"Fold {fc['fold']} (AUC={fc['auc']:.3f})",
            )
        ax.plot(
            cv["fpr_oof"],
            cv["tpr_oof"],
            lw=2.5,
            color="black",
            linestyle="--",
            label=f"OOF Global (AUC={cv['oof_auc']:.3f})",
        )
        ax.plot([0, 1], [0, 1], "k:", alpha=0.3)
        ax.set_title("Curvas ROC por Fold", fontweight="bold")
        ax.set_xlabel("FPR")
        ax.set_ylabel("TPR")
        ax.legend(fontsize=7.5)

        # --- 3. Boxplot das métricas por fold ---
        ax = axes[0, 2]
        metric_cols = [
            "AUC-ROC",
            "Avg Precision",
            "F1 (thr=0.5)",
            "F1 (Youden)",
            "Accuracy",
        ]
        fold_only = cv["summary_df"].iloc[:n][metric_cols]
        ax.boxplot(
            [fold_only[c].values for c in metric_cols],
            labels=metric_cols,
            patch_artist=True,
            boxprops=dict(facecolor="#aec6e8", color="#1f77b4"),
            medianprops=dict(color="red", lw=2),
            whiskerprops=dict(color="#1f77b4"),
            capprops=dict(color="#1f77b4"),
        )
        # Adiciona pontos individuais
        for i, col in enumerate(metric_cols):
            y_vals = fold_only[col].values
            x_jitter = np.random.default_rng(42).uniform(-0.15, 0.15, len(y_vals))
            ax.scatter(
                np.ones(len(y_vals)) * (i + 1) + x_jitter,
                y_vals,
                color="#1f77b4",
                alpha=0.7,
                zorder=5,
                s=40,
            )
        ax.set_ylim(0, 1.05)
        ax.set_title("Distribuição das Métricas (por fold)", fontweight="bold")
        ax.set_ylabel("Score")
        ax.tick_params(axis="x", rotation=20, labelsize=8)

        # --- 4. Distribuição OOF de probabilidades ---
        ax = axes[1, 0]
        oof_df = pd.DataFrame(
            {
                "prob": cv["oof_probs"],
                "label": cv["oof_true"].astype(int),
            }
        )
        for lbl, color, name in [
            (0, "#ff7f0e", f"Neg (0)"),
            (1, "#1f77b4", f"Pos (1)"),
        ]:
            sns.kdeplot(
                oof_df[oof_df["label"] == lbl]["prob"],
                ax=ax,
                fill=True,
                alpha=0.4,
                color=color,
                label=name,
            )
        ax.set_title("Distribuição OOF por Classe", fontweight="bold")
        ax.set_xlabel("Probabilidade Predita")
        ax.set_xlim(0, 1)
        ax.legend(fontsize=9)

        # --- 5. Calibração OOF ---
        ax = axes[1, 1]
        prob_true_oof, prob_pred_oof = calibration_curve(
            cv["oof_true"], cv["oof_probs"], n_bins=10
        )
        ax.plot(
            prob_pred_oof,
            prob_true_oof,
            "o-",
            lw=2,
            color="#1f77b4",
            label="Calibração OOF",
        )
        ax.plot([0, 1], [0, 1], "k--", label="Perfeito")
        ax.set_title("Calibração OOF", fontweight="bold")
        ax.set_xlabel("Probabilidade Predita")
        ax.set_ylabel("Fração de Positivos")
        ax.legend(fontsize=9)

        # --- 6. Estabilidade — AUC por fold (linha do tempo) ---
        ax = axes[1, 2]
        aucs = [fc["auc"] for fc in cv["fold_curves"]]
        folds = [fc["fold"] for fc in cv["fold_curves"]]
        colors_bar = [cmap(i) for i in range(n)]
        bars = ax.bar(folds, aucs, color=colors_bar, edgecolor="white", width=0.6)
        ax.axhline(
            np.mean(aucs),
            color="red",
            linestyle="--",
            lw=1.5,
            label=f"Média ({np.mean(aucs):.3f})",
        )
        ax.axhline(
            cv["oof_auc"],
            color="black",
            linestyle=":",
            lw=1.5,
            label=f"OOF Global ({cv['oof_auc']:.3f})",
        )
        ax.set_ylim(max(0, min(aucs) - 0.05), 1.0)
        ax.set_title("AUC-ROC por Fold (Estabilidade)", fontweight="bold")
        ax.set_xlabel("Fold")
        ax.set_ylabel("AUC-ROC")
        ax.set_xticks(folds)
        ax.legend(fontsize=9)
        for bar, val in zip(bars, aucs):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.003,
                f"{val:.3f}",
                ha="center",
                fontsize=9,
                fontweight="bold",
            )

        plt.suptitle(
            f"Relatório de Cross-Validation  ({n} Folds Estratificados)  —  "
            f"AUC OOF: {cv['oof_auc']:.4f}  |  AP OOF: {cv['oof_ap']:.4f}",
            fontsize=15,
            fontweight="bold",
        )
        plt.show()

    # -----------------------------------------------------------------------
    # 6. PREDIÇÃO (PIPELINE DE INFERÊNCIA)
    # -----------------------------------------------------------------------

    def _preprocess_for_inference(self, data: pd.DataFrame):
        """Aplica o mesmo pré-processamento do treino (sem vazar estado)."""
        df_proc = self._standardize_and_clean(data)

        available_cols = [
            c
            for c in df_proc.columns
            if c in self.selected_features or c == self.target
        ]
        df_proc = df_proc[available_cols]
        df_proc = self._extract_temporal_features(df_proc)
        df_proc = self._handle_rare_labels(df_proc, is_train=False)
        df_proc = self._handle_outliers_and_log(df_proc, is_train=False)

        has_target = self.target in df_proc.columns
        X = df_proc[self.selected_features]
        y = df_proc[self.target] if has_target else None

        # Verifica se o pipeline do sklearn foi ativado no treino
        if self.pipeline is not None:
            X_trans = self.pipeline.transform(X)
        else:
            X_trans = X.copy()

        return X_trans, y

    def predict(self, data: pd.DataFrame) -> pd.Series:
        X_trans, _ = self._preprocess_for_inference(data)
        return self.predictor.predict(X_trans)

    def predict_proba(self, data: pd.DataFrame) -> pd.DataFrame:
        X_trans, _ = self._preprocess_for_inference(data)
        return self.predictor.predict_proba(X_trans)

    # -----------------------------------------------------------------------
    # 7. VISUALIZAÇÕES
    # -----------------------------------------------------------------------

    def _get_decile_stats(self, y_true, y_prob, bins=10):
        df = pd.DataFrame({"target": y_true, "prob": y_prob})
        df["decile"] = pd.qcut(
            df["prob"].rank(method="first", ascending=True),
            bins,
            labels=range(1, bins + 1),
        )
        stats = (
            df.groupby("decile", observed=False)
            .agg(count=("target", "count"), events=("target", "sum"))
            .reset_index()
        )
        stats["event_rate"] = stats["events"] / stats["count"]
        global_rate = df["target"].mean()
        stats["lift"] = stats["event_rate"] / global_rate

        # KS por decil (cumulativo)
        stats = stats.sort_values("decile", ascending=False)
        total_pos = stats["events"].sum()
        total_neg = (stats["count"] - stats["events"]).sum()
        stats["cum_pos_rate"] = stats["events"].cumsum() / total_pos
        stats["cum_neg_rate"] = (stats["count"] - stats["events"]).cumsum() / total_neg
        stats["ks"] = abs(stats["cum_pos_rate"] - stats["cum_neg_rate"])
        return stats.sort_values("decile")

    def _youden_threshold(self, fpr, tpr, thresholds):
        """Threshold de Youden: maximiza TPR - FPR."""
        idx = np.argmax(tpr - fpr)
        return thresholds[idx], fpr[idx], tpr[idx]

    def plot_association_report(self):
        """
        Plota:
        1. Associação de cada feature com o target (barras).
        2. Matriz de associação entre features (heatmap).
        """
        if not self.association_report:
            raise RuntimeError(
                "Execute fit() antes de plotar o relatório de associações."
            )

        feat_target = self.association_report["feat_target"]
        matrix = self.association_report["feat_feat_matrix"]

        fig, axes = plt.subplots(1, 2, figsize=(18, max(6, len(feat_target) * 0.5)))

        # --- Painel 1: Feature → Target ---
        ax = axes[0]
        sorted_feats = sorted(
            feat_target.items(), key=lambda x: x[1]["valor"], reverse=True
        )
        labels = [f"{k}\n({v['metrica']})" for k, v in sorted_feats]
        values = [v["valor"] for _, v in sorted_feats]

        colors = [
            "#d62728" if v > 0.8 else "#ff7f0e" if v > 0.5 else "#1f77b4"
            for v in values
        ]
        bars = ax.barh(labels[::-1], values[::-1], color=colors[::-1])
        ax.set_xlabel("Força de Associação")
        ax.set_title(f"Associação Feature → Target '{self.target}'", fontweight="bold")
        ax.axvline(
            0.8, color="red", linestyle="--", alpha=0.5, label="Risco Leakage (0.8)"
        )
        ax.axvline(
            0.5,
            color="orange",
            linestyle="--",
            alpha=0.5,
            label="Alta Associação (0.5)",
        )
        ax.legend(fontsize=8)
        for bar, val in zip(bars, values[::-1]):
            ax.text(
                bar.get_width() + 0.01,
                bar.get_y() + bar.get_height() / 2,
                f"{val:.3f}",
                va="center",
                fontsize=8,
            )

        # --- Painel 2: Matriz Feature × Feature ---
        ax2 = axes[1]
        mask = np.eye(len(matrix), dtype=bool)
        sns.heatmap(
            matrix,
            ax=ax2,
            mask=mask,
            cmap="RdYlGn_r",
            vmin=0,
            vmax=1,
            annot=len(matrix) <= 15,
            fmt=".2f",
            linewidths=0.5,
            square=True,
            cbar_kws={"label": "Associação"},
        )
        ax2.set_title(
            "Matriz de Associação entre Features\n(Pearson | Theil's U | Eta²)",
            fontweight="bold",
        )

        plt.suptitle("Relatório de Associações", fontsize=16, fontweight="bold")
        plt.tight_layout()
        plt.show()

    def plot_complete_report(
        self, test_data: pd.DataFrame, train_data: pd.DataFrame = None, bins: int = 10
    ):
        datasets = {"Teste": test_data}
        if train_data is not None:
            datasets["Treino"] = train_data

        results = {}
        pos_label = self.predictor.positive_class
        colors = {"Treino": "#ff7f0e", "Teste": "#1f77b4"}

        for name, data in datasets.items():
            X_trans, y = self._preprocess_for_inference(data)

            # 1. Pega as probabilidades primeiro
            y_prob = self.predictor.predict_proba(X_trans)
            y_prob_pos = y_prob[pos_label].values  # Força array numérico

            # 2. Converte a VERDADE (y) para 0 e 1 (Int)
            y_bin = (y.astype(str).values == str(pos_label)).astype(int)

            # 3. Converte a PREDIÇÃO (y_pred) para 0 e 1 (Int)
            y_pred_raw = self.predictor.predict(X_trans)
            y_pred = (y_pred_raw.astype(str).values == str(pos_label)).astype(int)

            # Agora tudo abaixo (fpr, tpr, f1_score) receberá apenas 0s e 1s puros.
            fpr, tpr, thresholds = roc_curve(y_bin, y_prob_pos)
            roc_auc = auc(fpr, tpr)
            prec_full, rec_full, thresh_pr = precision_recall_curve(y_bin, y_prob_pos)
            pr_auc = average_precision_score(y_bin, y_prob_pos)
            # precision_recall_curve retorna len(thresh_pr) + 1 pontos em prec/rec.
            # Truncamos para alinhar com thresh_pr em todos os usos posteriores.
            prec = prec_full[:-1]
            rec = rec_full[:-1]
            prob_true, prob_pred = calibration_curve(y_bin, y_prob_pos, n_bins=bins)

            youden_thresh, youden_fpr, youden_tpr = self._youden_threshold(
                fpr, tpr, thresholds
            )
            y_pred_youden = (y_prob_pos >= youden_thresh).astype(int)

            decile_stats = self._get_decile_stats(y_bin, y_prob_pos, bins=bins)
            ks_stat = decile_stats["ks"].max()

            metrics = {
                "AUC-ROC": roc_auc,
                "Gini": 2 * roc_auc - 1,
                "KS": ks_stat,
                "F1 (thr=0.5)": f1_score(
                    y_bin, y_pred, pos_label=1, average="binary", zero_division=0
                ),
                "F1 (Youden)": f1_score(
                    y_bin, y_pred_youden, pos_label=1, average="binary", zero_division=0
                ),
                "Recall (thr=0.5)": recall_score(
                    y_bin, y_pred, pos_label=1, average="binary", zero_division=0
                ),
                "Log-Loss": log_loss(y_bin, y_prob_pos),
                "Accuracy": accuracy_score(y_bin, y_pred),
                "Avg Precision": pr_auc,
            }

            results[name] = {
                "y_true": y_bin,
                "y_prob": y_prob_pos,
                "y_pred": y_pred,
                "y_pred_youden": y_pred_youden,
                "fpr": fpr,
                "tpr": tpr,
                "thresholds": thresholds,
                "youden_thresh": youden_thresh,
                "youden_fpr": youden_fpr,
                "youden_tpr": youden_tpr,
                "metrics": metrics,
                "decile_stats": decile_stats,
                "prec": prec,  # truncado (alinhado com thresh_pr)
                "rec": rec,  # truncado (alinhado com thresh_pr)
                "prec_full": prec_full,  # completo (para plot da curva PR)
                "rec_full": rec_full,  # completo (para plot da curva PR)
                "thresh_pr": thresh_pr,
                "calib_true": prob_true,
                "calib_pred": prob_pred,
            }

        # ----------------------------------------------------------------
        # LAYOUT DO REPORT
        # 4 colunas × 4 linhas
        # ----------------------------------------------------------------
        fig = plt.figure(figsize=(24, 22), constrained_layout=True)
        gs = fig.add_gridspec(4, 4)

        # --- Linha 0: Scorecard + Confusion Matrix (Youden) + Feature Importance ---
        ax_metrics = fig.add_subplot(gs[0, :2])
        self._plot_scorecard(ax_metrics, results)

        ax_cm_default = fig.add_subplot(gs[0, 2])
        self._plot_confusion(
            ax_cm_default, results["Teste"], pos_label, "Threshold 0.5"
        )

        ax_cm_youden = fig.add_subplot(gs[0, 3])
        self._plot_confusion(
            ax_cm_youden,
            results["Teste"],
            pos_label,
            f"Youden Teste ({results['Teste']['youden_thresh']:.2f})",
            use_youden=True,
        )

        # --- Linha 1: ROC + PR + Calibração + Densidade ---
        ax_roc = fig.add_subplot(gs[1, 0])
        for name, res in results.items():
            ls = "-" if name == "Teste" else "--"
            ax_roc.plot(
                res["fpr"],
                res["tpr"],
                ls,
                lw=2,
                color=colors[name],
                label=f"{name} AUC={res['metrics']['AUC-ROC']:.3f}",
            )
            if name == "Teste":
                ax_roc.scatter(
                    [res["youden_fpr"]],
                    [res["youden_tpr"]],
                    color="red",
                    zorder=5,
                    s=80,
                    label="Youden",
                )
        ax_roc.plot([0, 1], [0, 1], "k--", alpha=0.3)
        ax_roc.set_title("Curva ROC", fontweight="bold")
        ax_roc.set_xlabel("FPR")
        ax_roc.set_ylabel("TPR")
        ax_roc.legend(fontsize=8)

        ax_pr = fig.add_subplot(gs[1, 1])
        for name, res in results.items():
            ls = "-" if name == "Teste" else "--"
            ax_pr.plot(
                res["rec_full"],
                res["prec_full"],
                ls,
                lw=2,
                color=colors[name],
                label=f"{name} AP={res['metrics']['Avg Precision']:.3f}",
            )

        # Alerta visual de overfitting quando gap de AP entre Treino e Teste for > 0.10
        if "Treino" in results:
            ap_gap = (
                results["Treino"]["metrics"]["Avg Precision"]
                - results["Teste"]["metrics"]["Avg Precision"]
            )
            if ap_gap > 0.10:
                ax_pr.text(
                    0.5,
                    0.12,
                    f"⚠️ Possível Overfitting\n" f"ΔAP Treino−Teste = {ap_gap:.2f}",
                    transform=ax_pr.transAxes,
                    ha="center",
                    va="center",
                    fontsize=9,
                    fontweight="bold",
                    color="white",
                    bbox=dict(
                        boxstyle="round,pad=0.4", facecolor="#d62728", alpha=0.85
                    ),
                )

        ax_pr.set_title("Precision-Recall", fontweight="bold")
        ax_pr.set_xlabel("Recall")
        ax_pr.set_ylabel("Precision")
        ax_pr.legend(fontsize=8)

        ax_calib = fig.add_subplot(gs[1, 2])
        for name, res in results.items():
            ls = "-" if name == "Teste" else "--"
            ax_calib.plot(
                res["calib_pred"],
                res["calib_true"],
                "o",
                linestyle=ls,
                color=colors[name],
                label=name,
            )
        ax_calib.plot([0, 1], [0, 1], "k--", label="Perfeito")
        ax_calib.set_title("Curva de Calibração", fontweight="bold")
        ax_calib.set_xlabel("Prob. Predita")
        ax_calib.set_ylabel("Fração de Positivos")
        ax_calib.legend(fontsize=8)

        ax_hist = fig.add_subplot(gs[1, 3])
        for name, res in results.items():
            sns.kdeplot(
                res["y_prob"],
                label=name,
                ax=ax_hist,
                fill=True,
                alpha=0.3,
                color=colors[name],
            )
        ax_hist.axvline(
            results["Teste"]["youden_thresh"],
            color="red",
            linestyle="--",
            label=f"Youden ({results['Teste']['youden_thresh']:.2f})",
        )
        ax_hist.set_title("Densidade de Probabilidade", fontweight="bold")
        ax_hist.set_xlim(0, 1)
        ax_hist.legend(fontsize=8)

        # --- Linha 2: Threshold Analysis ---
        ax_thresh = fig.add_subplot(gs[2, :2])
        self._plot_threshold_analysis(ax_thresh, results["Teste"])

        ax_feat = fig.add_subplot(gs[2, 2:])
        if self.feature_importance is not None:
            top_feat = self.feature_importance.head(15)
            sns.barplot(
                x=top_feat["importance"],
                y=top_feat.index,
                ax=ax_feat,
                palette="viridis",
            )
            ax_feat.set_title("Top 15 Features Importantes", fontweight="bold")
        else:
            ax_feat.text(0.5, 0.5, "Indisponível", ha="center", va="center")

        # --- Linha 3: Decil + KS ---
        ax_decil = fig.add_subplot(gs[3, :3])
        self._plot_decil(ax_decil, results, colors, bins)

        ax_ks = fig.add_subplot(gs[3, 3])
        self._plot_ks_curve(ax_ks, results["Teste"])

        plt.suptitle(
            f"Relatório Completo de Performance — Classificação (target: {pos_label})",
            fontsize=18,
            weight="bold",
        )
        plt.show()

    # ---- Sub-plots auxiliares ----

    def _plot_scorecard(self, ax, results):
        ax.axis("off")
        metric_df = pd.DataFrame({k: v["metrics"] for k, v in results.items()})

        # Cor de fundo por métrica (verde = bom, vermelho = atenção)
        def _color(metric, val):
            good = metric not in ["Log-Loss"]
            if good:
                return (
                    "#d4edda" if val >= 0.7 else "#fff3cd" if val >= 0.5 else "#f8d7da"
                )
            else:
                return (
                    "#d4edda" if val <= 0.4 else "#fff3cd" if val <= 0.6 else "#f8d7da"
                )

        table_data = [
            [m] + [f"{metric_df.loc[m, c]:.4f}" for c in metric_df.columns]
            for m in metric_df.index
        ]
        table = ax.table(
            cellText=table_data,
            colLabels=["Métrica"] + list(results.keys()),
            loc="center",
            cellLoc="center",
            bbox=[0, 0, 1, 1],
        )
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 1.9)

        for i, m in enumerate(metric_df.index):
            for j, col in enumerate(metric_df.columns):
                val = metric_df.loc[m, col]
                table[i + 1, j + 1].set_facecolor(_color(m, val))

        ax.set_title("🏆 Scorecard do Modelo", fontweight="bold", fontsize=13)

    def _plot_confusion(self, ax, res, pos_label, title, use_youden=False):
        y_pred = res["y_pred_youden"] if use_youden else res["y_pred"]
        cm = confusion_matrix(res["y_true"], y_pred)

        # Rótulos semânticos: neg_label = classe 0, pos_label = classe 1
        neg_label = (
            f"Não {pos_label}" if isinstance(pos_label, str) else f"≠{pos_label}"
        )
        class_labels = [neg_label, str(pos_label)]

        # Monta anotações enriquecidas: valor absoluto + % da linha (taxa real)
        total = cm.sum()
        row_totals = cm.sum(axis=1, keepdims=True)
        pct = cm / row_totals * 100  # % por linha (precision/recall perspectiva)
        annot = np.array(
            [[f"{cm[i,j]}\n({pct[i,j]:.0f}%)" for j in range(2)] for i in range(2)]
        )

        sns.heatmap(
            cm,
            annot=annot,
            fmt="",
            cmap="Blues",
            ax=ax,
            cbar=False,
            annot_kws={"size": 11},
            xticklabels=class_labels,
            yticklabels=class_labels,
        )
        ax.set_title(f"Confusão — {title}", fontweight="bold", fontsize=10)
        ax.set_xlabel("Predito", fontsize=9)
        ax.set_ylabel("Real", fontsize=9)
        ax.tick_params(axis="x", labelsize=8, rotation=15)
        ax.tick_params(axis="y", labelsize=8, rotation=0)

    def _plot_threshold_analysis(self, ax, res):
        """Plota Precision, Recall e F1 em função do threshold de corte."""
        thresholds = res["thresh_pr"]
        prec = res["prec"]
        rec = res["rec"]
        f1 = 2 * prec * rec / (prec + rec + 1e-9)

        ax.plot(thresholds, prec, label="Precision", color="#2ca02c", lw=2)
        ax.plot(thresholds, rec, label="Recall", color="#d62728", lw=2)
        ax.plot(thresholds, f1, label="F1", color="#1f77b4", lw=2)

        # Youden: linha vertical roxa sólida com seta anotada
        ax.axvline(
            res["youden_thresh"],
            color="#9467bd",
            linestyle="-",
            lw=2,
            label=f"Youden ({res['youden_thresh']:.2f})",
        )
        ax.annotate(
            f"Youden\n{res['youden_thresh']:.2f}",
            xy=(res["youden_thresh"], 0.97),
            xytext=(res["youden_thresh"] + 0.04, 0.97),
            fontsize=7.5,
            color="#9467bd",
            fontweight="bold",
            arrowprops=dict(arrowstyle="->", color="#9467bd", lw=1.2),
            va="top",
        )

        # Default 0.5: linha vertical cinza tracejada
        ax.axvline(
            0.5,
            color="#7f7f7f",
            linestyle="--",
            lw=1.5,
            label="Default (0.5)",
        )

        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.05)
        ax.set_xlabel("Threshold de Decisão")
        ax.set_ylabel("Score")
        ax.set_title("Análise de Threshold (Teste)", fontweight="bold")
        ax.legend(fontsize=8)

    def _plot_decil(self, ax, results, colors, bins):
        all_decile_stats = []
        for name, res in results.items():
            d = res["decile_stats"].copy()
            d["Dataset"] = name
            all_decile_stats.append(d)
        combined_stats = pd.concat(all_decile_stats)

        sns.barplot(
            data=combined_stats,
            x="decile",
            y="event_rate",
            hue="Dataset",
            ax=ax,
            palette=colors,
        )

        for name, res in results.items():
            mean_val = res["y_true"].mean()
            ax.axhline(
                mean_val,
                color=colors[name],
                linestyle="--",
                linewidth=1.5,
                label=f"Média {name} ({mean_val:.1%})",
            )

        ks_stat = results["Teste"]["decile_stats"]["ks"].max()
        label_type = "Decil" if bins == 10 else "Quintil"
        ax.set_title(
            f"Taxa de Eventos por {label_type}  |  KS Máximo (Teste): {ks_stat:.4f}",
            fontweight="bold",
            fontsize=13,
        )
        ax.set_xlabel(
            f"{label_type} (1 = Menor Risco → {bins} = Maior Risco)", fontsize=11
        )
        ax.set_ylabel("Taxa de Eventos (Target=1)", fontsize=11)
        ax.legend(loc="upper left", fontsize=8)

        for container in ax.containers:
            ax.bar_label(container, fmt="%.1f%%", padding=3, fontsize=8)

    def _plot_ks_curve(self, ax, res):
        """Curva de Lorenz / KS (separação cumulativa de positivos e negativos)."""
        df_ks = pd.DataFrame({"target": res["y_true"], "prob": res["y_prob"]})
        df_ks = df_ks.sort_values("prob", ascending=False).reset_index(drop=True)

        total_pos = df_ks["target"].sum()
        total_neg = len(df_ks) - total_pos

        cum_pos = df_ks["target"].cumsum() / total_pos
        cum_neg = (1 - df_ks["target"]).cumsum() / total_neg
        pct_pop = np.linspace(0, 1, len(df_ks))
        ks_vals = abs(cum_pos - cum_neg)
        ks_max_idx = ks_vals.argmax()

        ax.plot(
            pct_pop, cum_pos.values, color="#1f77b4", lw=2, label="Cumulativo Positivos"
        )
        ax.plot(
            pct_pop, cum_neg.values, color="#d62728", lw=2, label="Cumulativo Negativos"
        )
        ax.plot([0, 1], [0, 1], "k--", alpha=0.3)

        x_ks = pct_pop[ks_max_idx]
        ax.annotate(
            f"KS={ks_vals.iloc[ks_max_idx]:.3f}",
            xy=(x_ks, (cum_pos.iloc[ks_max_idx] + cum_neg.iloc[ks_max_idx]) / 2),
            fontsize=9,
            color="green",
            fontweight="bold",
        )
        ax.vlines(
            x_ks,
            cum_neg.iloc[ks_max_idx],
            cum_pos.iloc[ks_max_idx],
            color="green",
            linestyle="--",
            lw=1.5,
        )

        ax.set_title("Curva KS (Teste)", fontweight="bold")
        ax.set_xlabel("% População (Ordenado por Score)")
        ax.set_ylabel("% Cumulativo")
        ax.legend(fontsize=8)

    # -----------------------------------------------------------------------
    # 8. SERIALIZAÇÃO (SAVE + LOAD)
    # -----------------------------------------------------------------------

    def save_bundle(self, path: str = "modelo_prod_v2"):
        os.makedirs(path, exist_ok=True)
        joblib.dump(
            {
                "pipeline": self.pipeline,
                "params": self.params,
                "log_cols": self._log_cols,
                "outlier_bounds": self._outlier_bounds,
                "rare_cats": self._rare_categories,
                "selected_features": self.selected_features,
                "eliminated_features": self.eliminated_features,
                "association_report": self.association_report,
            },
            f"{path}/assets.pkl",
        )
        self.predictor.save(f"{path}/autogluon")
        print(f"📦 Bundle salvo em '{path}/'")

    @staticmethod
    def load(path: str) -> "AutoClassificationEngine":
        """
        Reconstitui o engine completo a partir de um bundle salvo.
        Uso: engine = AutoClassificationEngine.load("modelo_prod_v2")
        """
        assets_path = f"{path}/assets.pkl"
        ag_path = f"{path}/autogluon"

        if not os.path.exists(assets_path):
            raise FileNotFoundError(f"Assets não encontrados em '{assets_path}'")
        if not os.path.exists(ag_path):
            raise FileNotFoundError(f"Modelo AutoGluon não encontrado em '{ag_path}'")

        assets = joblib.load(assets_path)

        engine = AutoClassificationEngine.__new__(AutoClassificationEngine)
        engine.params = assets["params"]
        engine.target = assets["params"]["target"]
        engine.pipeline = assets["pipeline"]
        engine._log_cols = assets["log_cols"]
        engine._outlier_bounds = assets["outlier_bounds"]
        engine._rare_categories = assets["rare_cats"]
        engine.selected_features = assets["selected_features"]
        engine.eliminated_features = assets.get("eliminated_features", {})
        engine.association_report = assets.get("association_report", {})
        engine.feature_importance = None

        engine.predictor = TabularPredictor.load(ag_path)

        print(f"✅ Engine carregado de '{path}/'")
        print(f"   Target: {engine.target} | Features: {engine.selected_features}")
        return engine
