# v0.1.9 Claude
#   [v0.1.9] NEW: feature_whitelist — features protegidas de todos os filtros de seleção
#   [v0.1.9] NEW: log de features resgatadas pela whitelist no resumo final
# v0.1.8 Claude
#   [v0.1.8] NEW: fit_selection() — seleção com dataset completo, sem AutoGluon
#   [v0.1.8] NEW: fit_model() — treina AutoGluon com features já selecionadas
#   [v0.1.8] REFACTOR: fit() vira wrapper backward-compatible (fit_selection + fit_model)
#   [v0.1.8] FIX: Boruta agora vê o dataset completo antes do split treino/holdout
# v0.1.7 Claude
#   [v0.1.7] NEW: null_importance_percentile exposto como parâmetro (default 50, era hardcoded 75)
#   [v0.1.7] TUNE: Null Importance agora filtra lixo, não sinal — Boruta faz a seleção fina
# v0.1.6 Claude
#   [v0.1.6] REMOVE: SMOTE removido completamente (import + _HAS_SMOTE + smote_tag no plot)
#   [v0.1.6] TUNE: boruta_iters default 15 → 20 (decisão mais estável)
#   [v0.1.6] TUNE: boruta_hit_pct default 0.40 → 0.55 (filtro mais seletivo)
# v0.1.5 Claude
#   [v0.1.5] FIX: infinity/crash no cross_validate — shift usa min(col_min, lower_bound)
#   [v0.1.5] FIX: _handle_outliers_and_log usa mesmo shift robusto em fit e inferência
#   [v0.1.5] FIX: TransformPipeline._step_outliers_and_log recebe mesmo fix
#   [v0.1.5] NEW: sanitização inf/-inf → NaN após log1p em todos os caminhos
#   [v0.1.4] NEW: TransformPipeline — pipeline de transformação portável, sem AutoGluon
#   [v0.1.4] NEW: engine.export_transform_pipeline() — exporta TransformPipeline para .pkl
#   [v0.1.4] NEW: TransformPipeline.describe() — resumo das etapas e features
#   [v0.1.4] NEW: warnings distintos para colunas extras e ausentes em transform()
# v0.1.3 Claude
# Changelog:
#   [v0.1.0] FIX: leakage no log-shift do cross_validate — shift calculado só no fold de treino
#   [v0.1.0] FIX: cross_validate respeita use_sklearn_pipeline=False
#   [v0.1.0] FIX: profile_all label correto para grupos float/pd.Interval
#   [v0.1.0] FIX: _tfidf_group_scores compatível com pandas 2.2+ (include_groups=False)
#   [v0.1.0] FIX: _plot_decil — alinhamento de labels, offset relativo, ylim expandido, clip_on=False
#   [v0.1.0] FIX: del pred_fold + gc.collect() em cada fold do CV (vazamento de memória)
#   [v0.1.0] FIX: guard multiclasse em fit() com mensagem clara
#   [v0.1.0] NEW: rare_label_min_freq exposto como parâmetro do engine
#   [v0.1.0] NEW: _handle_multicollinearity usa df.corr() para pares numéricos (O(n) → O(1))
#   [v0.1.0] NEW: save_bundle com versioning (framework/python/sklearn) e overwrite protection
#   [v0.1.0] NEW: load() emite aviso ao detectar versão diferente do framework
#   [v0.1.0] NEW: cores distintas treino (#F06000) vs teste (#0055A8) em todos os plots
#   [v0.1.0] NEW: Brier Score adicionado ao scorecard e métricas do CV
#   [v0.1.0] NEW: num_cpus padrão = os.cpu_count() ou 6
#   [v0.0.9] ProfileAnalyzer integrado — perfil de segmentos por decil/cluster/segmento
#   [v0.0.9] engine.profile_analyzer() — factory method pré-configurado com score do modelo
#   [v0.0.9] cross_validate: n_repeats (RepeatedStratifiedKFold) + OOF por média de repetições
#   [v0.0.9] cv_results persistido em save_bundle / restaurado em load
#   [v0.0.9] FIX: matplotlib.colormaps substitui plt.cm.get_cmap (deprecated 3.7+)
#   [v0.0.9] FIX: isinstance(dtype, pd.CategoricalDtype) substitui is_categorical_dtype (deprecated 2.1+)
#   [v0.0.9] FIX: log_cols em cross_validate alinhado com _handle_outliers_and_log (sem trava min>=0)

from __future__ import annotations

import gc
import hashlib
import numbers
import re
import builtins as _b
import os
import sys
import warnings
import logging
from math import ceil
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import sklearn

from scipy.stats import entropy, skew, spearmanr
from scipy.stats.contingency import association
from sklearn.feature_extraction.text import TfidfVectorizer
from autogluon.tabular import TabularPredictor
from sklearn.metrics import (
    confusion_matrix,
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score,
    precision_score,
    accuracy_score,
    log_loss,
    f1_score,
    recall_score,
    brier_score_loss,  # [NEW v0.1.0]
)
from sklearn.calibration import calibration_curve
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import (
    StandardScaler,
    OrdinalEncoder,
    MinMaxScaler,
    RobustScaler,
)
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.decomposition import PCA
from sklearn.model_selection import (
    StratifiedKFold,
    RepeatedStratifiedKFold,
    train_test_split,
)

try:
    from sklearn.preprocessing import TargetEncoder

    _HAS_TARGET_ENCODER = True
except ImportError:
    _HAS_TARGET_ENCODER = False

_FRAMEWORK_VERSION = "0.1.9"

# Cores canônicas para treino/teste — alta distinção visual e para daltônicos
_COLOR_TRAIN = "#F06000"  # laranja-forte
_COLOR_TEST = "#0055A8"  # azul-forte

# =============================================================================
# BLOCO 1 — Utilitários de Associação (AutoClassificationEngine)
# =============================================================================


def _theils_u(x: pd.Series, y: pd.Series) -> float:
    x = x.astype(str).fillna("__NA__")
    y = y.astype(str).fillna("__NA__")
    s_xy = _conditional_entropy(x, y)
    x_counter = x.value_counts(normalize=True)
    s_x = entropy(x_counter)
    if s_x == 0:
        return 1.0
    return (s_x - s_xy) / s_x


def _conditional_entropy(x: pd.Series, y: pd.Series) -> float:
    h = 0.0
    for yv in y.unique():
        mask = y == yv
        p_y = mask.mean()
        x_given_y = x[mask].value_counts(normalize=True)
        h += p_y * entropy(x_given_y)
    return h


def _cramers_v(x: pd.Series, y: pd.Series) -> float:
    ct = pd.crosstab(x.astype(str).fillna("__NA__"), y.astype(str).fillna("__NA__"))
    return association(ct.values, method="cramer")


def _eta_squared(cat: pd.Series, num: pd.Series) -> float:
    cat = cat.astype(str).fillna("__NA__")
    groups = [num[cat == c].dropna() for c in cat.unique()]
    grand_mean = num.dropna().mean()
    ss_between = sum(len(g) * (g.mean() - grand_mean) ** 2 for g in groups)
    ss_total = ((num.dropna() - grand_mean) ** 2).sum()
    return ss_between / ss_total if ss_total > 0 else 0.0


def _numeric_correlation(a: pd.Series, b: pd.Series, method: str = "pearson") -> float:
    pearson = abs(a.corr(b))
    if method == "pearson":
        return pearson
    try:
        spearman_val = abs(spearmanr(a.dropna(), b.dropna())[0])
    except Exception:
        spearman_val = 0.0
    if method == "spearman":
        return spearman_val
    return max(pearson, spearman_val)


def _make_profile_feat_name(col: str, val_token: str) -> str:
    """Gera nome seguro e legível para uma feature binária de perfil."""
    raw = f"pf__{col}__{val_token}"
    if len(raw) > 64:
        import hashlib as _hl

        suffix = _hl.md5(raw.encode()).hexdigest()[:6]
        raw = f"pf__{col[:24]}__{suffix}"
    return re.sub(r"[^A-Za-z0-9_]", "_", raw)


# =============================================================================
# BLOCO 2 — ProfileAnalyzer: utilitários de pré-processamento
# =============================================================================


def make_group_column(
    df: pd.DataFrame,
    mode: str = "decile",
    score_col: Optional[str] = None,
    group_col: Optional[str] = None,
    n_quantiles: int = 10,
    labels: Optional[List] = None,
) -> pd.Series:
    if mode in ("existing", "raw"):
        if group_col is None or group_col not in df.columns:
            raise ValueError(f"group_col='{group_col}' não encontrado no DataFrame.")
        return df[group_col].copy()

    if score_col is None or score_col not in df.columns:
        raise ValueError(f"score_col='{score_col}' não encontrado no DataFrame.")

    q = 10 if mode == "decile" else n_quantiles
    return pd.qcut(df[score_col], q, labels=labels, duplicates="drop")


def remove_high_cardinality(
    df: pd.DataFrame,
    cols: List[str],
    max_unique_ratio: float = 0.05,
    min_unique_abs: int = 100,
    suspicious_keywords: Optional[List[str]] = None,
    whitelist: Optional[List[str]] = None,
    blacklist: Optional[List[str]] = None,
) -> tuple:
    _kw = suspicious_keywords or [
        "id",
        "cpf",
        "cnpj",
        "doc",
        "matric",
        "registro",
        "hash",
        "uuid",
        "chave",
        "token",
        "telefone",
        "cel",
        "email",
        "rg",
        "nome",
        "sobrenome",
        "advog",
        "oab",
    ]
    _white = set(whitelist or [])
    _black = set(blacklist or [])
    n = len(df)

    cols_keep, cols_drop, rows = [], [], []
    for c in cols:
        u = df[c].astype("string").nunique(dropna=True)
        ratio = u / n if n > 0 else 0.0
        kw = any(k in c.lower() for k in _kw)

        if c in _black:
            drop = True
        elif c in _white:
            drop = False
        else:
            drop = (u >= min_unique_abs and ratio >= max_unique_ratio) or kw

        (cols_drop if drop else cols_keep).append(c)
        rows.append((c, u, round(ratio, 4), kw, drop))

    stats = pd.DataFrame(
        rows, columns=["coluna", "n_unique", "unique_ratio", "kw_match", "drop"]
    ).sort_values(["drop", "unique_ratio"], ascending=[False, False])
    return cols_keep, cols_drop, stats


def cap_top_k(
    df: pd.DataFrame,
    col: str,
    k: int = 50,
    other_label: str = "OUTROS",
) -> None:
    top = set(df[col].astype("string").value_counts(dropna=True).nlargest(k).index)
    df[col] = df[col].where(df[col].isin(top), other_label)


def binarize_numerics(
    df: pd.DataFrame,
    cols: Optional[List[str]] = None,
    quantiles: List[float] = None,
    exclude: Optional[List[str]] = None,
) -> pd.DataFrame:
    if quantiles is None:
        quantiles = [0, 0.25, 0.50, 0.75, 1.0]

    _excl = set(exclude or [])
    _cols = (
        cols
        if cols is not None
        else df.select_dtypes(include="number").columns.tolist()
    )
    _cols = [c for c in _cols if c not in _excl]

    new_bins: Dict[str, pd.Series] = {}
    to_drop: List[str] = []

    for col in _cols:
        try:
            limits = np.unique(df[col].quantile(quantiles).to_numpy())
            if len(limits) < 2:
                continue
            bins = pd.cut(df[col], bins=limits, include_lowest=True)
            new_bins[f"{col}_bin"] = bins.map(
                lambda x: f"{x.left}a{x.right}" if pd.notnull(x) else "nulo"
            )
            to_drop.append(col)
        except Exception as e:
            print(f"[binarize] Aviso: não foi possível binarizar '{col}': {e}")

    result = pd.concat([df, pd.DataFrame(new_bins, index=df.index)], axis=1)
    if to_drop:
        result = result.drop(columns=to_drop)
    return result


def _to_token(series: pd.Series) -> pd.Series:
    s = series.astype("string").fillna("nulo").str.strip()
    s = s.str.replace(r"^(nan|none|na|null|<na>)$", "nulo", flags=re.I, regex=True)
    s = s.str.replace(r"[\s\-]+", "_", regex=True)
    s = s.str.replace(".", "_DOT_", regex=False)
    return s


def build_token_df(df: pd.DataFrame, feature_cols: List[str]) -> pd.DataFrame:
    out = df[feature_cols].copy()
    for col in feature_cols:
        out[col] = col + "___" + _to_token(out[col])
    return out


# =============================================================================
# BLOCO 3 — ProfileAnalyzer: núcleo de métricas
# =============================================================================


def _compute_record_level_metrics(
    df_tokens: pd.DataFrame,
    group_col: pd.Series,
    feature_cols: List[str],
    group_value,
    add_k: float = 0.1,
) -> pd.DataFrame:
    df_work = df_tokens.copy()
    df_work["__group__"] = group_col.values

    mask_in = df_work["__group__"] == group_value
    df_in = df_work[mask_in]
    df_out = df_work[~mask_in]

    n_in = len(df_in)
    n_out = len(df_out)

    records: Dict[str, Dict] = {}
    for col in feature_cols:
        for grp_df, is_in in [(df_in, True), (df_out, False)]:
            vc = grp_df[col].value_counts(dropna=True)
            for token, cnt in vc.items():
                if token not in records:
                    records[token] = {"cnt_in": 0, "cnt_out": 0}
                if is_in:
                    records[token]["cnt_in"] += int(cnt)
                else:
                    records[token]["cnt_out"] += int(cnt)

    if not records:
        return pd.DataFrame()

    df_m = pd.DataFrame.from_dict(records, orient="index").reset_index()
    df_m.columns = ["Token_raw", "cnt_in", "cnt_out"]

    V = len(df_m)
    p_in = (df_m["cnt_in"] + add_k) / (n_in + add_k * V)
    p_out = (df_m["cnt_out"] + add_k) / (n_out + add_k * V)

    df_m["n_regs_in"] = n_in
    df_m["n_regs_out"] = n_out
    df_m["Pct_in"] = df_m["cnt_in"] / _b.max(n_in, 1)
    df_m["Pct_out"] = df_m["cnt_out"] / _b.max(n_out, 1)
    df_m["Lift"] = p_in / p_out

    return df_m


def _tfidf_group_scores(
    df_tokens: pd.DataFrame,
    group_col: pd.Series,
    feature_cols: List[str],
    group_value,
    min_df: int = 1,
    max_df: float = 0.95,
) -> Dict[str, float]:
    df_work = df_tokens.copy()
    df_work["__group__"] = group_col.values

    # [FIX v0.1.0] include_groups=False evita DeprecationWarning no pandas 2.2+
    docs = (
        df_work.groupby("__group__")[feature_cols]
        .apply(lambda x: list(x.values.flatten().astype(str)), include_groups=False)
        .sort_index()
    )

    groups = docs.index.tolist()
    if group_value not in groups:
        return {}

    # -------------------------------------------------------------------------
    # CORREÇÃO AQUI: Se for binário (<= 2 grupos), o max_df não pode ser 0.95,
    # senão tokens presentes em ambas as classes (100% de DF) zeram o vocabulário.
    # -------------------------------------------------------------------------
    if len(groups) <= 2 and isinstance(max_df, float) and max_df < 1.0:
        max_df = 1.0

    i = groups.index(group_value)

    try:
        tv = TfidfVectorizer(
            analyzer=lambda x: x,
            lowercase=False,
            norm="l2",
            sublinear_tf=True,
            min_df=min_df,
            max_df=max_df,
        )
        Xt = tv.fit_transform(docs)
        vocab = tv.get_feature_names_out()
        row = Xt[i].toarray()[0]
        return dict(zip(vocab, row))
    except ValueError:
        # Fallback seguro: se o TF-IDF podar tudo (ex: todos os termos são raros),
        # retorna dict vazio para não quebrar a execução.
        return {}


# =============================================================================
# BLOCO 4 — ProfileAnalyzer: classe principal
# =============================================================================


class ProfileAnalyzer:
    """
    Analisa o perfil predominante de segmentos em um DataFrame.

    Suporta qualquer tipo de agrupamento:
      - Decis de modelo  → group_mode="decile",   score_col="prob"
      - Clusters         → group_mode="existing", group_col="cluster"
      - Segmentos RFM    → group_mode="raw",       group_col="segmento_rfm"
      - Quantis custom   → group_mode="quantile",  score_col="prob", n_quantiles=5
      - Grupos de A/B    → group_mode="existing", group_col="variante"
    """

    def __init__(
        self,
        df: pd.DataFrame,
        group_mode: str = "decile",
        score_col: Optional[str] = None,
        group_col: Optional[str] = None,
        n_quantiles: int = 10,
        group_labels: Optional[List] = None,
        filter_col: Optional[str] = None,
        filter_value=None,
        label_col: Optional[str] = None,
        exclude_cols: Optional[List[str]] = None,
        cols_as_text: Optional[List[str]] = None,
        whitelist: Optional[List[str]] = None,
        blacklist: Optional[List[str]] = None,
        binarize: bool = True,
        quantile_bins: Optional[List[float]] = None,
        max_unique_ratio: float = 0.05,
        min_unique_abs: int = 100,
        cap_topk: Optional[Dict[str, int]] = None,
        add_k: float = 0.1,
        min_df: int = 1,
        max_df: float = 0.95,
    ):
        self.df_raw = df.copy()
        self.group_mode = group_mode
        self.score_col = score_col
        self.group_col = group_col
        self.n_quantiles = n_quantiles
        self.group_labels = group_labels
        self.filter_col = filter_col
        self.filter_value = filter_value
        self.label_col = label_col
        self.exclude_cols = list(exclude_cols or [])
        self.cols_as_text = list(cols_as_text or [])
        self.whitelist = whitelist
        self.blacklist = blacklist
        self.binarize = binarize
        self.quantile_bins = quantile_bins
        self.max_unique_ratio = max_unique_ratio
        self.min_unique_abs = min_unique_abs
        self.cap_topk = cap_topk or {}
        self.add_k = add_k
        self.min_df = min_df
        self.max_df = max_df

        self.df_ = None
        self.group_series_ = None
        self.feature_cols_ = None
        self.df_tokens_ = None
        self.groups_ = None

    def fit(self, verbose: bool = True) -> "ProfileAnalyzer":
        df = self.df_raw.copy()

        if self.filter_col and self.filter_value is not None:
            df = df[df[self.filter_col] == self.filter_value].copy()
            if verbose:
                print(
                    f"[fit] Filtro '{self.filter_col}=={self.filter_value}': {len(df)} registros."
                )

        group = make_group_column(
            df,
            mode=self.group_mode,
            score_col=self.score_col,
            group_col=self.group_col,
            n_quantiles=self.n_quantiles,
            labels=self.group_labels,
        )

        _excl = set(self.exclude_cols)
        if self.label_col:
            _excl.add(self.label_col)
        if self.score_col:
            _excl.add(self.score_col)
        if self.group_col:
            _excl.add(self.group_col)

        for col in self.cols_as_text:
            if col in df.columns:
                df[col] = df[col].astype("string")

        if self.binarize:
            numeric_to_bin = [
                c
                for c in df.select_dtypes(include="number").columns
                if c not in _excl and c not in self.cols_as_text
            ]
            df = binarize_numerics(
                df, cols=numeric_to_bin, quantiles=self.quantile_bins, exclude=_excl
            )

        for col, k in self.cap_topk.items():
            if col in df.columns:
                cap_top_k(df, col, k=k)

        candidates = [c for c in df.columns if c not in _excl]

        cols_keep, cols_drop, _ = remove_high_cardinality(
            df,
            candidates,
            max_unique_ratio=self.max_unique_ratio,
            min_unique_abs=self.min_unique_abs,
            whitelist=self.whitelist,
            blacklist=self.blacklist,
        )
        if verbose:
            print(
                f"[fit] Colunas descartadas (alta cardinalidade/PII): {len(cols_drop)}"
            )
            print(f"[fit] Colunas para análise: {len(cols_keep)}")

        df_tok = build_token_df(df, cols_keep)

        self.df_ = df
        self.group_series_ = group.reset_index(drop=True)
        self.feature_cols_ = cols_keep
        self.df_tokens_ = df_tok.reset_index(drop=True)
        self.groups_ = sorted(self.group_series_.dropna().unique())

        if verbose:
            print(f"[fit] Grupos encontrados: {self.groups_}")

        return self

    def profile_group(
        self,
        group_value,
        topn: int = 20,
        min_lift: float = 2.0,
        min_support_pct: float = 0.01,
        one_per_col: bool = False,
        include_raw: bool = False,
    ) -> pd.DataFrame:
        self._check_fitted()

        df_m = _compute_record_level_metrics(
            self.df_tokens_,
            self.group_series_,
            self.feature_cols_,
            group_value,
            add_k=self.add_k,
        )
        if df_m.empty:
            return df_m

        tfidf_map = _tfidf_group_scores(
            self.df_tokens_,
            self.group_series_,
            self.feature_cols_,
            group_value,
            min_df=self.min_df,
            max_df=self.max_df,
        )
        df_m["TFIDF"] = df_m["Token_raw"].map(tfidf_map).fillna(0.0)

        n_in = int(df_m["n_regs_in"].iloc[0]) if not df_m.empty else 1
        piso = 5 if n_in <= 200 else 10
        min_sup = _b.max(piso, ceil(min_support_pct * n_in))

        df_filt = df_m.query("cnt_in >= @min_sup and Lift >= @min_lift").copy()

        if len(df_filt) < topn:
            falta = topn - len(df_filt)
            df_fill = (
                df_m[~df_m["Token_raw"].isin(df_filt["Token_raw"])]
                .sort_values(["TFIDF", "cnt_in"], ascending=[False, False])
                .head(falta)
            )
            df_filt = pd.concat([df_filt, df_fill], ignore_index=True)

        if one_per_col and not df_filt.empty:
            df_filt[["__col__", "__val__"]] = df_filt["Token_raw"].str.split(
                "___", n=1, expand=True
            )
            df_filt = (
                df_filt.sort_values(
                    ["__col__", "Lift", "cnt_in", "TFIDF"],
                    ascending=[True, False, False, False],
                )
                .groupby("__col__", as_index=False)
                .head(1)
                .drop(columns=["__col__", "__val__"])
            )

        if not df_filt.empty:
            df_filt = df_filt.sort_values(
                ["Lift", "cnt_in", "TFIDF"], ascending=[False, False, False]
            ).head(topn)

        df_filt = df_filt.copy()
        df_filt["Atributo"] = (
            df_filt["Token_raw"]
            .str.replace("___", ": ", regex=False)
            .str.replace("_DOT_", ".", regex=False)
            .str.replace(r"(?<=\d)a(?=\d)", "–", regex=True)
            .str.replace(r"__+", "_", regex=True)
        )

        cols_out = ["Atributo", "cnt_in", "Pct_in", "Pct_out", "Lift", "TFIDF"]
        if include_raw:
            cols_out = ["Token_raw"] + cols_out
        return df_filt[[c for c in cols_out if c in df_filt.columns]].reset_index(
            drop=True
        )

    def profile_all(
        self,
        topn: int = 20,
        min_lift: float = 2.0,
        min_support_pct: float = 0.01,
        one_per_col: bool = False,
        display_fn=None,
    ) -> Dict:
        self._check_fitted()
        results = {}
        for g in self.groups_:
            # [FIX v0.1.0] Label correto para grupos inteiros, float e pd.Interval
            if isinstance(g, numbers.Integral):
                label = f"Faixa {int(g)+1} (grupo {g})"
            else:
                label = f"Grupo {g}"

            print(f"\n{'='*60}")
            print(f"  PERFIL PREDOMINANTE — {label}")
            print(f"{'='*60}")
            df_p = self.profile_group(
                g,
                topn=topn,
                min_lift=min_lift,
                min_support_pct=min_support_pct,
                one_per_col=one_per_col,
            )
            results[g] = df_p
            if display_fn:
                display_fn(df_p)
            else:
                print(df_p.to_string(index=False))
        return results

    def profile_all_display(
        self,
        topn: int = 20,
        min_lift: float = 2.0,
        min_support_pct: float = 0.01,
        one_per_col: bool = False,
    ) -> Dict:
        """
        Exibe os perfilamentos de forma visual e estruturada no notebook.
        Compatível com Databricks (display nativo) e Jupyter (IPython.display).
        Retorna o mesmo dict {grupo: DataFrame} que profile_all().
        """
        try:
            from IPython.display import display as _display, HTML as _HTML
        except ImportError:
            _display = print
            _HTML = lambda x: x

        self._check_fitted()
        results = {}

        header_colors = [
            "#0055A8",
            "#F06000",
            "#1a7a4a",
            "#8B0000",
            "#5B2D8E",
            "#B8860B",
            "#1C6E8C",
            "#A0522D",
            "#2E8B57",
            "#8B008B",
        ]

        for i, g in enumerate(self.groups_):
            if isinstance(g, numbers.Integral):
                label = f"Faixa {int(g)+1} — Grupo {g}"
            else:
                label = f"Grupo {g}"
            cor = header_colors[i % len(header_colors)]

            df_p = self.profile_group(
                g,
                topn=topn,
                min_lift=min_lift,
                min_support_pct=min_support_pct,
                one_per_col=one_per_col,
            )
            results[g] = df_p

            _display(
                _HTML(
                    f"<div style='background:{cor};color:white;padding:10px 16px;"
                    f"border-radius:6px;font-size:15px;font-weight:bold;"
                    f"margin-top:20px;letter-spacing:0.5px'>"
                    f"&#128269; PERFIL PREDOMINANTE &mdash; {label}"
                    f"</div>"
                )
            )

            if df_p.empty:
                _display(
                    _HTML(
                        "<p style='color:gray;font-style:italic;margin:6px 0'>Nenhum token acima do threshold.</p>"
                    )
                )
                continue

            fmt = {
                k: v
                for k, v in {
                    "Lift": "{:.2f}",
                    "TFIDF": "{:.3f}",
                    "Pct_in": "{:.1%}",
                    "Pct_out": "{:.1%}",
                }.items()
                if k in df_p.columns
            }

            lift_max = df_p["Lift"].max() if "Lift" in df_p.columns else 1.0

            styled = df_p.style
            if "Lift" in df_p.columns:
                styled = styled.background_gradient(
                    subset=["Lift"], cmap="RdYlGn", vmin=1.0, vmax=lift_max
                )
            if "Pct_in" in df_p.columns:
                styled = styled.background_gradient(subset=["Pct_in"], cmap="Blues")
            if "TFIDF" in df_p.columns:
                styled = styled.background_gradient(subset=["TFIDF"], cmap="Purples")

            styled = (
                styled.format(fmt)
                .set_properties(
                    **{
                        "font-size": "13px",
                        "border": "1px solid #e0e0e0",
                        "padding": "5px 10px",
                    }
                )
                .set_table_styles(
                    [
                        {
                            "selector": "thead th",
                            "props": [
                                ("background-color", cor),
                                ("color", "white"),
                                ("font-weight", "bold"),
                                ("font-size", "13px"),
                                ("padding", "8px 10px"),
                            ],
                        },
                        {
                            "selector": "tbody tr:hover",
                            "props": [("background-color", "#f5f5f5")],
                        },
                    ]
                )
                .hide(axis="index")
            )
            _display(styled)

        return results

    def feature_importance_summary(self) -> pd.DataFrame:
        self._check_fitted()
        rows = []
        for g in self.groups_:
            df_m = _compute_record_level_metrics(
                self.df_tokens_, self.group_series_, self.feature_cols_, g, self.add_k
            )
            if not df_m.empty:
                df_m["grupo"] = g
                rows.append(df_m[["Token_raw", "Lift", "cnt_in", "grupo"]])

        if not rows:
            return pd.DataFrame()

        all_m = pd.concat(rows, ignore_index=True)
        summary = (
            all_m.groupby("Token_raw")
            .agg(
                lift_max=("Lift", "max"),
                lift_mean=("Lift", "mean"),
                grupos_top=("grupo", lambda x: list(x[all_m.loc[x.index, "Lift"] > 2])),
            )
            .sort_values("lift_max", ascending=False)
            .reset_index()
        )
        summary["Atributo"] = (
            summary["Token_raw"]
            .str.replace("___", ": ", regex=False)
            .str.replace("_DOT_", ".", regex=False)
        )
        return summary[["Atributo", "lift_max", "lift_mean", "grupos_top"]]

    def _check_fitted(self):
        if self.df_tokens_ is None:
            raise RuntimeError(
                "Chame .fit() antes de .profile_group() ou .profile_all()."
            )


# =============================================================================
# =============================================================================
# BLOCO 4.5 — TransformPipeline (portável, sem AutoGluon, sem framework)
# =============================================================================


class TransformPipeline:
    """
    Pipeline de transformação portável — totalmente independente do
    AutoClassificationEngine e do AutoGluon.

    Exportada via:
        engine.export_transform_pipeline("minha_pipeline.pkl")

    Importada e usada em qualquer projeto externo:
        import joblib
        pipeline = joblib.load("minha_pipeline.pkl")
        df_transformado = pipeline.transform(df_bruto)

    Dependências no projeto externo: numpy, pandas, scikit-learn, joblib

    Comportamento com colunas inesperadas:
        - Coluna EXTRA (no input, não estava no treino): ignorada + warning
        - Coluna AUSENTE (estava no treino, não está no input): avisada + ignorada
          (não é inventada; simplesmente não constará no output)
        - Colunas CORRESPONDENTES: recebem exatamente as transformações do treino
    """

    _VERSION = "0.1.4"

    def __init__(self):
        self.params: dict = {}
        self.selected_features: list = []
        self._train_schema: dict = {}
        self._log_cols: list = []
        self._outlier_bounds: dict = {}
        self._rare_categories: dict = {}
        self._agg_values: dict = {}
        self._profile_features: list = []
        self.sklearn_pipeline = None
        self._framework_version: str = ""

    # ------------------------------------------------------------------
    # Helper interno (copiado do framework para independência total)
    # ------------------------------------------------------------------

    @staticmethod
    def _to_token(series: pd.Series) -> pd.Series:
        s = series.astype("string").fillna("nulo").str.strip()
        s = s.str.replace(r"^(nan|none|na|null|<na>)$", "nulo", flags=re.I, regex=True)
        s = s.str.replace(r"[\s\-]+", "_", regex=True)
        s = s.str.replace(".", "_DOT_", regex=False)
        return s

    # ------------------------------------------------------------------
    # Etapas de transformação (espelham o AutoClassificationEngine)
    # ------------------------------------------------------------------

    def _step_standardize_and_clean(self, df: pd.DataFrame) -> pd.DataFrame:
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

    def _step_extract_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        date_cols = df.select_dtypes(include=["datetime64"]).columns
        for col in date_cols:
            df[f"{col}_month"] = df[col].dt.month
            df[f"{col}_weekday"] = df[col].dt.weekday
            df[f"{col}_hour_sin"] = np.sin(2 * np.pi * df[col].dt.hour / 24)
            df = df.drop(columns=[col])
        return df

    def _step_group_aggregations(self, df: pd.DataFrame) -> pd.DataFrame:
        pairs = self.params.get("group_aggregation_pairs", [])
        for pair in pairs:
            cat_col = pair.get("cat")
            num_col = pair.get("num")
            agg_func = pair.get("agg", "mean")
            feat_name = f"{agg_func}_{num_col}_by_{cat_col}"

            if feat_name not in self._agg_values:
                continue
            if cat_col not in df.columns:
                continue

            info = self._agg_values[feat_name]
            df[feat_name] = df[cat_col].map(info["map"]).fillna(info["fallback"])
        return df

    def _step_rare_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        for col, valid_cats in self._rare_categories.items():
            if col not in df.columns:
                continue
            if isinstance(df[col].dtype, pd.CategoricalDtype):
                df[col] = df[col].astype(str)
            df[col] = df[col].where(df[col].isin(valid_cats), other="OTHER")
        return df

    def _step_outliers_and_log(self, df: pd.DataFrame) -> pd.DataFrame:
        for col, (lower, upper) in self._outlier_bounds.items():
            if col in df.columns:
                df[col] = np.clip(df[col], lower, upper)

        for col in self._log_cols:
            if col not in df.columns:
                continue
            col_min = df[col].min()
            # [FIX v0.1.4] Shift usa min(col_min, lower_bound) — cobre valores de
            # inferência que podem atingir o lower bound sem que o treino os tivesse
            lb = self._outlier_bounds.get(col, (col_min, None))[0]
            shift = min(col_min, lb) if pd.notna(col_min) else 0.0
            if shift < 0:
                df[col] = df[col] - shift
            df[col] = np.log1p(df[col])
            df[col] = df[col].replace([np.inf, -np.inf], np.nan)
        return df

    def _step_profile_features(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self._profile_features:
            return df
        df = df.copy()
        for spec in self._profile_features:
            col = spec["col"]
            val_token = spec["val_token"]
            feat_name = spec["feat_name"]
            if col not in df.columns:
                df[feat_name] = 0
                continue
            col_tokenized = self._to_token(df[col].astype("object"))
            df[feat_name] = (col_tokenized == val_token).astype(int)
        return df

    def _step_select_and_warn(self, df: pd.DataFrame) -> pd.DataFrame:
        expected = set(self.selected_features)
        input_cols = set(df.columns)

        extra = input_cols - expected
        for col in sorted(extra):
            warnings.warn(
                f"[TransformPipeline] Coluna '{col}' não fez parte do treinamento "
                f"e não será transformada por esta pipeline.",
                UserWarning,
                stacklevel=4,
            )

        missing = expected - input_cols
        for col in sorted(missing):
            warnings.warn(
                f"[TransformPipeline] Coluna '{col}' era esperada pelo pipeline "
                f"mas não foi encontrada no input — será tratada como NaN pelo imputer.",
                UserWarning,
                stacklevel=4,
            )

        # Seleciona apenas as features do treino, na mesma ordem
        # Colunas ausentes são inseridas como NaN para que os imputers atuem normalmente
        df = df.reindex(columns=self.selected_features)
        return df

    # ------------------------------------------------------------------
    # Interface pública
    # ------------------------------------------------------------------

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Aplica todas as transformações do treinamento ao DataFrame de entrada.

        Ordem das etapas:
            1. Limpeza e padronização (null patterns, force_types, exclusões)
            2. Extração de features temporais (datetime → month, weekday, hour_sin)
            3. Group aggregations (usando mapas aprendidos no treino)
            4. Rare label encoding (categorias raras → "OTHER")
            5. Outlier clipping + transformação log1p
            6. Profile features (indicadores binários TF-IDF, se ativados)
            7. Seleção das features do treino + warnings de colunas extras/ausentes
            8. Sklearn pipeline (imputers, scalers, encoders fitados no treino)

        Returns:
            pd.DataFrame transformado, pronto para uso em modelos.
        """
        df = data.copy()
        df = self._step_standardize_and_clean(df)
        df = self._step_extract_temporal_features(df)
        df = self._step_group_aggregations(df)
        df = self._step_rare_labels(df)
        df = self._step_outliers_and_log(df)
        df = self._step_profile_features(df)
        df = self._step_select_and_warn(df)

        if self.sklearn_pipeline is not None:
            df = self.sklearn_pipeline.transform(df)

        return df

    def describe(self):
        """Exibe um resumo do pipeline exportado."""
        print(
            f"\n📦 TransformPipeline v{self._VERSION}  (framework v{self._framework_version})"
        )
        print(f"   Features esperadas  : {len(self.selected_features)}")
        print(f"   Outlier bounds      : {len(self._outlier_bounds)} colunas")
        print(f"   Log transform       : {len(self._log_cols)} colunas")
        print(f"   Rare label cols     : {len(self._rare_categories)} colunas")
        print(f"   Group aggregations  : {len(self._agg_values)} features")
        print(f"   Profile features    : {len(self._profile_features)} indicadores")
        sklearn_status = (
            "✅ fitado"
            if self.sklearn_pipeline is not None
            else "❌ desativado no treino"
        )
        print(f"   Sklearn pipeline    : {sklearn_status}")
        print(f"   Features            : {self.selected_features}")


# =============================================================================
# BLOCO 5 — AutoClassificationEngine
# =============================================================================


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

        self._log_cols: list = []
        self._outlier_bounds: dict = {}
        self._rare_categories: dict = {}
        self._agg_values: dict = {}
        self._train_schema: dict = {}
        self._train_hash: str = ""

        self.eliminated_features = {
            "leakage": [],
            "alta_cardinalidade": [],
            "colinearidade": [],
            "constantes_pos_rare": [],
            "importancia_nula": [],
            "boruta": [],  # <--- ADICIONE AQUI
        }

        self.association_report: dict = {}
        self.cv_results: dict = {}
        self._profile_features: list = []  # [NEW v0.1.1]

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
        from autogluon.core.metrics import Scorer

        if isinstance(metric, str):
            if metric not in valid_metrics:
                raise ValueError(
                    f"eval_metric '{metric}' inválido. Opções: {valid_metrics}"
                )
        elif not isinstance(metric, Scorer):
            raise ValueError(
                f"eval_metric deve ser uma string de {valid_metrics} ou um objeto Scorer do AutoGluon."
            )

        valid_presets = {
            "best_quality",
            "high_quality",
            "good_quality",
            "medium_quality",
            "optimize_for_deployment",
        }
        preset = params.get("presets", "high_quality")
        if preset not in valid_presets:
            raise ValueError(f"presets '{preset}' inválido. Opções: {valid_presets}")

        valid_corr_methods = {"pearson", "spearman", "max"}
        corr_method = params.get("corr_method", "pearson")
        if corr_method not in valid_corr_methods:
            raise ValueError(
                f"corr_method '{corr_method}' inválido. Opções: {valid_corr_methods}"
            )

        valid_thr_strategies = {"youden", "f_beta", "cost_matrix"}
        thr_strategy = params.get("threshold_strategy", "youden")
        if thr_strategy not in valid_thr_strategies:
            raise ValueError(
                f"threshold_strategy '{thr_strategy}' inválido. Opções: {valid_thr_strategies}"
            )

    def _validate_group_agg_pairs(self, df_columns: list):
        pairs = self.params.get("group_aggregation_pairs", [])
        if not pairs:
            return

        valid_agg_funcs = {"mean", "median", "std", "min", "max", "sum", "count"}

        for i, pair in enumerate(pairs):
            if not isinstance(pair, dict):
                raise ValueError(
                    f"group_aggregation_pairs[{i}] deve ser um dict. Recebido: {type(pair)}"
                )
            for key in ("cat", "num"):
                if key not in pair:
                    raise ValueError(
                        f"group_aggregation_pairs[{i}] está faltando a chave '{key}'."
                    )
            agg_func = pair.get("agg", "mean")
            if agg_func not in valid_agg_funcs:
                raise ValueError(
                    f"group_aggregation_pairs[{i}]: 'agg' = '{agg_func}' inválido. Opções: {valid_agg_funcs}"
                )
            for key in ("cat", "num"):
                col = pair[key]
                if col not in df_columns:
                    warnings.warn(
                        f"group_aggregation_pairs[{i}]: coluna '{col}' (chave '{key}') "
                        f"não encontrada no DataFrame. O par será ignorado no fit().",
                        UserWarning,
                        stacklevel=3,
                    )

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
        leakage_thr = self.params.get("leakage_threshold", 0.98)
        to_drop = []
        target_series = df[self.target]
        target_is_cat = self._is_categorical_target(target_series)

        const_cols = [c for c in df.columns if df[c].nunique(dropna=False) <= 1]
        if const_cols:
            self.eliminated_features.setdefault("constantes_pos_rare", []).extend(
                const_cols
            )
            to_drop.extend(const_cols)

        features_to_check = [
            c for c in df.columns if c not in to_drop and c != self.target
        ]
        for col in features_to_check:
            col_is_num = pd.api.types.is_numeric_dtype(df[col])
            try:
                if col_is_num and not target_is_cat:
                    corr_method = self.params.get("corr_method", "pearson")
                    assoc = _numeric_correlation(
                        df[col], target_series.astype(float), method=corr_method
                    )
                elif not col_is_num and target_is_cat:
                    assoc = _theils_u(df[col], target_series)
                elif col_is_num and target_is_cat:
                    assoc = _eta_squared(target_series, df[col])
                else:
                    assoc = _eta_squared(df[col], target_series.astype(float))
            except Exception:
                assoc = 0.0

            if assoc > leakage_thr:
                self.eliminated_features.setdefault("leakage", []).append(col)
                to_drop.append(col)

        cat_cols = df.select_dtypes(include=["object", "category"]).columns
        obs_count = len(df)
        for col in cat_cols:
            if col == self.target or col in to_drop:
                continue
            if df[col].nunique() / obs_count > 0.5:
                self.eliminated_features.setdefault("alta_cardinalidade", []).append(
                    col
                )
                to_drop.append(col)

        return df.drop(columns=list(set(to_drop)))

    def _handle_multicollinearity(self, df: pd.DataFrame) -> pd.DataFrame:
        threshold = self.params.get("corr_threshold", 0.90)
        corr_method = self.params.get("corr_method", "pearson")
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

        target_is_cat = self._is_categorical_target(target_series)
        target_assoc = {}
        for col in all_feat:
            col_is_num = pd.api.types.is_numeric_dtype(df[col])
            try:
                if col_is_num and not target_is_cat:
                    target_assoc[col] = _numeric_correlation(
                        df[col], target_series.astype(float), method=corr_method
                    )
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

        to_drop = set()

        # [NEW v0.1.0] Pares numérico-numérico via df.corr() — O(n) em vez de O(n²)
        if len(num_cols) > 1:
            corr_pd_method = (
                "pearson" if corr_method in ("pearson", "max") else "spearman"
            )
            num_corr = df[num_cols].corr(method=corr_pd_method).abs()
            if corr_method == "max":
                spearman_corr = df[num_cols].corr(method="spearman").abs()
                num_corr = num_corr.combine(spearman_corr, np.maximum)

            for i, col_a in enumerate(num_cols):
                if col_a in to_drop:
                    continue
                for col_b in num_cols[i + 1 :]:
                    if col_b in to_drop:
                        continue
                    assoc = num_corr.loc[col_a, col_b]
                    if assoc > threshold:
                        assoc_a = target_assoc.get(col_a, 0.0)
                        assoc_b = target_assoc.get(col_b, 0.0)
                        if abs(assoc_a - assoc_b) < 0.01:
                            loser = (
                                col_a
                                if df[col_a].nunique() > df[col_b].nunique()
                                else col_b
                            )
                        else:
                            loser = col_b if assoc_a >= assoc_b else col_a
                        to_drop.add(loser)

        # Pares envolvendo ao menos uma coluna categórica (mantém loop, menos custoso)
        for i, col_a in enumerate(all_feat):
            if col_a in to_drop or col_a in num_cols:
                continue
            for col_b in all_feat[i + 1 :]:
                if col_b in to_drop:
                    continue
                a_is_num = pd.api.types.is_numeric_dtype(df[col_a])
                b_is_num = pd.api.types.is_numeric_dtype(df[col_b])
                if a_is_num and b_is_num:
                    continue  # já tratado acima

                try:
                    if not a_is_num and not b_is_num:
                        assoc = max(
                            _theils_u(df[col_a], df[col_b]),
                            _theils_u(df[col_b], df[col_a]),
                        )
                    else:
                        cat_col, num_col = (
                            (col_a, col_b) if not a_is_num else (col_b, col_a)
                        )
                        eta = _eta_squared(df[cat_col], df[num_col])
                        codes = df[cat_col].astype("category").cat.codes.astype(float)
                        pearson = abs(codes.corr(df[num_col]))
                        assoc = max(eta, pearson)
                except Exception:
                    assoc = 0.0

                if assoc > threshold:
                    assoc_a = target_assoc.get(col_a, 0.0)
                    assoc_b = target_assoc.get(col_b, 0.0)
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
            pairs = self.params.get("group_aggregation_pairs", [])
            agg_names = {
                f"{p.get('agg','mean')}_{p['num']}_by_{p['cat']}"
                for p in pairs
                if "cat" in p and "num" in p
            }
            dropped_agg = [c for c in to_drop if c in agg_names]
            dropped_orig = [c for c in to_drop if c not in agg_names]

            if dropped_agg:
                print(
                    f"   ⚠️  Colinearidade: {len(dropped_agg)} agg feature(s) eliminada(s): {dropped_agg}"
                )
            if dropped_orig:
                print(
                    f"   ⚠️  Colinearidade: {len(dropped_orig)} feature(s) original(is) eliminada(s): {dropped_orig}"
                )

            self.eliminated_features["colinearidade"].extend(list(to_drop))
            df = df.drop(columns=list(to_drop))

        return df

    def _run_boruta_selection(
        self,
        X_core: pd.DataFrame,
        y_core: pd.Series,
        max_iters: int = 20,
        hit_threshold_pct: float = 0.55,
    ) -> list:
        """
        Boruta-LightGBM Nativo: Filtro de elite que confronta as features
        reais contra 'Shadow Features' (cópias embaralhadas) para garantir
        que apenas o sinal sobreviva ao ruído.
        """
        print(f"\n🌪️ --- INICIANDO BORUTA-LGBM (Seleção de Elite) ---")

        from lightgbm import LGBMClassifier
        import numpy as np

        # 1. Prepara categóricas para o LightGBM
        X_proc = X_core.copy()
        cat_cols = X_proc.select_dtypes(include=["object", "category"]).columns.tolist()
        for c in cat_cols:
            X_proc[c] = X_proc[c].astype("category")

        hits = {col: 0 for col in X_proc.columns}

        for i in range(max_iters):
            # 2. Cria as Shadow Features (Embaralhadas)
            X_shadow = X_proc.copy()
            X_shadow.columns = [f"shadow_{c}" for c in X_proc.columns]

            for c in X_shadow.columns:
                # Seed variável garante um embaralhamento único por coluna e por iteração
                np.random.seed(42 + i + hash(c) % 10000)
                X_shadow[c] = np.random.permutation(X_shadow[c].values)

            # 3. Arena: Originais vs Impostoras
            X_arena = pd.concat([X_proc, X_shadow], axis=1)

            # --- FIX: Restaura o dtype 'category' que o numpy permutation destruiu ---
            arena_cat_cols = X_arena.select_dtypes(include=["object"]).columns
            for c in arena_cat_cols:
                X_arena[c] = X_arena[c].astype("category")
            # -------------------------------------------------------------------------

            # 4. Treina o juiz (Rápido, mas "amarrado" para não decorar ruído)
            clf = LGBMClassifier(
                n_estimators=50,  # Menos árvores
                max_depth=4,  # <--- FIX: Impede a árvore de crescer e decorar o ruído
                num_leaves=15,  # <--- FIX: Força árvores simples
                colsample_bytree=0.8,  # <--- FIX: Obriga a árvore a olhar para variáveis diferentes
                random_state=42 + i,
                importance_type="gain",
                verbose=-1,
                n_jobs=-1,
            )

            clf.fit(X_arena, y_core)

            # 5. Extrai as importâncias
            imp_df = pd.DataFrame(
                {"feature": X_arena.columns, "gain": clf.feature_importances_}
            )

            # 6. A Régua de Corte: O melhor impostor da rodada
            max_shadow_gain = imp_df[imp_df["feature"].str.startswith("shadow_")][
                "gain"
            ].max()

            # 7. Contabiliza os Hits (Vitórias das originais sobre a melhor impostora)
            real_imp = imp_df[~imp_df["feature"].str.startswith("shadow_")]
            for _, row in real_imp.iterrows():
                if row["gain"] > max_shadow_gain:
                    hits[row["feature"]] += 1

            print(
                f"   🔄 Iteração {i+1}/{max_iters} concluída. (Régua: {max_shadow_gain:.2f})"
            )

        # 8. Decisão Final (Tem que vencer em pelo menos X% das rodadas)
        threshold_hits = max(1, int(max_iters * hit_threshold_pct))
        good_features = [col for col, h in hits.items() if h >= threshold_hits]

        removidas = len(X_core.columns) - len(good_features)
        print(f"\n   🏆 Boruta Finalizado!")
        print(f"   => Critério de Sobrevivência: {threshold_hits} Hits")
        print(f"   => Mantidas: {len(good_features)} | Removidas: {removidas}")

        return good_features

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

    def _create_group_aggregations(
        self, df: pd.DataFrame, is_train: bool = True
    ) -> pd.DataFrame:
        pairs = self.params.get("group_aggregation_pairs", [])
        if not pairs:
            return df

        created = []
        for pair in pairs:
            cat_col = pair.get("cat")
            num_col = pair.get("num")
            agg_func = pair.get("agg", "mean")
            feat_name = f"{agg_func}_{num_col}_by_{cat_col}"

            if cat_col not in df.columns or num_col not in df.columns:
                continue

            if is_train:
                agg_map = df.groupby(cat_col)[num_col].agg(agg_func)
                global_fallback = float(df[num_col].agg(agg_func))
                self._agg_values[feat_name] = {
                    "map": agg_map.to_dict(),
                    "fallback": global_fallback,
                }
                df[feat_name] = df[cat_col].map(agg_map).fillna(global_fallback)
                created.append(feat_name)
            else:
                if feat_name in self._agg_values:
                    info = self._agg_values[feat_name]
                    df[feat_name] = (
                        df[cat_col].map(info["map"]).fillna(info["fallback"])
                    )

        if is_train and created:
            print(f"   🔧 Group aggregations criadas ({len(created)}): {created}")

        return df

    @staticmethod
    def _compute_agg_map_local(df: pd.DataFrame, pairs: list) -> dict:
        local_map = {}
        for pair in pairs:
            cat_col = pair.get("cat")
            num_col = pair.get("num")
            agg_func = pair.get("agg", "mean")
            feat_name = f"{agg_func}_{num_col}_by_{cat_col}"

            if not cat_col or not num_col:
                continue
            if cat_col not in df.columns or num_col not in df.columns:
                continue

            try:
                agg_series = df.groupby(cat_col)[num_col].agg(agg_func)
                global_fallback = float(df[num_col].agg(agg_func))
                local_map[feat_name] = {
                    "map": agg_series.to_dict(),
                    "fallback": global_fallback,
                }
            except Exception:
                pass

        return local_map

    @staticmethod
    def _apply_agg_map_local(
        df: pd.DataFrame, pairs: list, agg_map: dict
    ) -> pd.DataFrame:
        df = df.copy()
        for pair in pairs:
            cat_col = pair.get("cat")
            num_col = pair.get("num")
            agg_func = pair.get("agg", "mean")
            feat_name = f"{agg_func}_{num_col}_by_{cat_col}"

            if feat_name not in agg_map:
                continue
            if cat_col not in df.columns:
                continue

            info = agg_map[feat_name]
            df[feat_name] = df[cat_col].map(info["map"]).fillna(info["fallback"])

        return df

    def _handle_rare_labels(
        self, df: pd.DataFrame, is_train: bool = True, limit: float = None
    ) -> pd.DataFrame:
        # [NEW v0.1.0] rare_label_min_freq exposto como parâmetro do engine
        if limit is None:
            limit = self.params.get("rare_label_min_freq", 0.01)

        cat_cols = df.select_dtypes(include=["object", "category"]).columns
        for col in cat_cols:
            if col == self.target:
                continue
            if isinstance(df[col].dtype, pd.CategoricalDtype):
                df[col] = df[col].astype(str)

            if is_train:
                freq = df[col].value_counts(normalize=True)
                self._rare_categories[col] = freq[freq >= limit].index.tolist()
            if col in self._rare_categories:
                df[col] = df[col].where(
                    df[col].isin(self._rare_categories[col]), other="OTHER"
                )

        if is_train:
            const_cols = [
                c for c in cat_cols if c != self.target and df[c].nunique() <= 1
            ]
            if const_cols:
                self.eliminated_features["constantes_pos_rare"].extend(const_cols)
                df = df.drop(columns=const_cols)

        return df

    def _handle_outliers_and_log(
        self, df: pd.DataFrame, is_train: bool = True
    ) -> pd.DataFrame:
        df_num = df.select_dtypes(include=[np.number])
        for col in df_num.columns:
            if col == self.target:
                continue

            if is_train:
                Q1, Q3 = df[col].quantile([0.25, 0.75])
                IQR = Q3 - Q1
                if IQR > 0:
                    self._outlier_bounds[col] = (Q1 - 1.5 * IQR, Q3 + 1.5 * IQR)

                if abs(skew(df[col].dropna())) > 0.75:
                    self._log_cols.append(col)

            if col in self._outlier_bounds:
                lower, upper = self._outlier_bounds[col]
                df[col] = np.clip(df[col], lower, upper)

            if col in self._log_cols:
                col_min = df[col].min()
                # [FIX v0.1.4] Shift usa min(col_min, lower_bound) — cobre valores de
                # inferência que podem atingir o lower bound sem que o treino os tivesse
                lb = self._outlier_bounds.get(col, (col_min, None))[0]
                shift = min(col_min, lb) if pd.notna(col_min) else 0.0
                if shift < 0:
                    df[col] = df[col] - shift
                df[col] = np.log1p(df[col])
                df[col] = df[col].replace([np.inf, -np.inf], np.nan)

        return df

    # -----------------------------------------------------------------------
    # 3. ANÁLISE DE ASSOCIAÇÃO
    # -----------------------------------------------------------------------

    @staticmethod
    def _is_categorical_target(series: pd.Series) -> bool:
        if not pd.api.types.is_numeric_dtype(series):
            return True
        return series.nunique() <= 20

    def _compute_association_report(self, df: pd.DataFrame) -> dict:
        feat_cols = [c for c in df.columns if c != self.target]
        target_series = df[self.target]
        target_is_cat = self._is_categorical_target(target_series)
        corr_method = self.params.get("corr_method", "pearson")

        feat_target = {}
        for col in feat_cols:
            col_is_num = pd.api.types.is_numeric_dtype(df[col])
            try:
                if col_is_num and not target_is_cat:
                    v = _numeric_correlation(
                        df[col], target_series.astype(float), method=corr_method
                    )
                    metric = (
                        f"Pearson/Spearman({corr_method})"
                        if corr_method != "pearson"
                        else "Pearson"
                    )
                elif not col_is_num and target_is_cat:
                    v = _theils_u(df[col], target_series)
                    metric = "Theil's U"
                elif col_is_num and target_is_cat:
                    v = _eta_squared(target_series, df[col])
                    metric = "Eta²"
                else:
                    v = _eta_squared(df[col], target_series.astype(float))
                    metric = "Eta²"
            except Exception:
                v, metric = 0.0, "Erro"
            feat_target[col] = {"valor": round(v, 4), "metrica": metric}

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
                        v = _numeric_correlation(
                            df[col_a], df[col_b], method=corr_method
                        )
                        m = f"Pearson/Spearman({corr_method})"
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

    def _build_sklearn_pipeline(self, X: pd.DataFrame, y: pd.Series = None) -> Pipeline:
        numeric_features = X.select_dtypes(
            include=["int64", "float64"]
        ).columns.tolist()
        categorical_features = X.select_dtypes(
            include=["object", "category"]
        ).columns.tolist()

        pipeline_cfg = self.params.get("pipeline_settings", {})
        auto_numeric_prep = pipeline_cfg.get("auto_numeric_prep", True)

        transformers = []

        if auto_numeric_prep and numeric_features:
            print(
                "   🤖 Analisando features numéricas para Imputação e Scaling inteligente..."
            )
            strategy_groups = {}

            for col in numeric_features:
                col_data = X[col]
                pct_missing = col_data.isna().mean()
                imp_choice = "knn" if 0 < pct_missing <= 0.30 else "median"

                clean_data = col_data.dropna()
                if len(clean_data) > 2 and clean_data.nunique() > 1:
                    k = clean_data.kurtosis()
                    s = abs(clean_data.skew())
                    if pd.isna(k):
                        k = 0
                    if pd.isna(s):
                        s = 0
                    if k > 3 or s > 1.0:
                        scl_choice = "robust"
                    elif clean_data.min() >= 0 and clean_data.max() <= 1:
                        scl_choice = "minmax"
                    else:
                        scl_choice = "standard"
                else:
                    scl_choice = "standard"

                strategy_groups.setdefault((imp_choice, scl_choice), []).append(col)

            for (imp, scl), cols in strategy_groups.items():
                steps = []
                steps.append(
                    (
                        "imputer",
                        (
                            KNNImputer(n_neighbors=5)
                            if imp == "knn"
                            else SimpleImputer(strategy="median")
                        ),
                    )
                )
                if scl == "robust":
                    steps.append(("scaler", RobustScaler()))
                elif scl == "minmax":
                    steps.append(("scaler", MinMaxScaler()))
                else:
                    steps.append(("scaler", StandardScaler()))
                transformers.append((f"num_{imp}_{scl}", Pipeline(steps), cols))

        elif numeric_features:
            num_steps = [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]
            if pipeline_cfg.get("use_pca"):
                n_comps = pipeline_cfg.get("pca_components", 0.95)
                num_steps.append(("pca", PCA(n_components=n_comps)))
            transformers.append(("num_default", Pipeline(num_steps), numeric_features))

        if categorical_features:
            use_te = self.params.get("use_target_encoding", False)
            if use_te and _HAS_TARGET_ENCODER:
                cat_transformer = Pipeline(
                    steps=[
                        (
                            "imputer",
                            SimpleImputer(strategy="constant", fill_value="MISSING"),
                        ),
                        ("encoder", TargetEncoder(target_type="auto", smooth="auto")),
                    ]
                )
            elif use_te and not _HAS_TARGET_ENCODER:
                warnings.warn(
                    "TargetEncoder não disponível (sklearn < 1.3). Usando OrdinalEncoder.",
                    UserWarning,
                )
                cat_transformer = Pipeline(
                    steps=[
                        (
                            "imputer",
                            SimpleImputer(strategy="constant", fill_value="MISSING"),
                        ),
                        (
                            "encoder",
                            OrdinalEncoder(
                                handle_unknown="use_encoded_value", unknown_value=-1
                            ),
                        ),
                    ]
                )
            else:
                cat_transformer = Pipeline(
                    steps=[
                        (
                            "imputer",
                            SimpleImputer(strategy="constant", fill_value="MISSING"),
                        ),
                    ]
                )
            transformers.append(("cat", cat_transformer, categorical_features))

        preprocessor = ColumnTransformer(
            transformers=transformers,
            verbose_feature_names_out=False,
        ).set_output(transform="pandas")

        return Pipeline(steps=[("preprocessor", preprocessor)])

    def describe_pipeline(self):
        if self.pipeline is None:
            print("⚠️  Pipeline sklearn não está ativo (use_sklearn_pipeline=False).")
            return

        print("\n🔬 --- DESCRIÇÃO DA PIPELINE SKLEARN ---")
        preprocessor = self.pipeline.named_steps["preprocessor"]

        for name, transformer, cols in preprocessor.transformers_:
            cols = list(cols)
            if not cols:
                continue
            print(f"\n📦 Transformador: '{name}' → {len(cols)} coluna(s)")
            print(f"   Colunas: {cols}")

            if isinstance(transformer, str):
                print(f"   └── Ação: {transformer}")
                continue
            if not hasattr(transformer, "steps"):
                print(f"   └── {transformer.__class__.__name__}")
                continue

            for step_name, step in transformer.steps:
                cls_name = step.__class__.__name__
                print(f"   └── {step_name}: {cls_name}")

                if hasattr(step, "statistics_") and step.statistics_ is not None:
                    imputed = {}
                    for c, v in zip(cols, step.statistics_):
                        try:
                            f_val = float(v)
                            if not np.isnan(f_val):
                                imputed[c] = round(f_val, 4)
                        except (ValueError, TypeError):
                            imputed[c] = v
                    if imputed:
                        print(f"        strategy={step.strategy}")
                        for c, v in list(imputed.items())[:5]:
                            print(f"          {c}: {v}")
                        if len(imputed) > 5:
                            print(f"          ... (+{len(imputed) - 5} colunas)")

                if hasattr(step, "mean_") and step.mean_ is not None:
                    print(
                        f"        with_mean={step.with_mean} | with_std={step.with_std}"
                    )
                    for c, m, s in zip(cols[:3], step.mean_[:3], step.scale_[:3]):
                        print(f"          {c}: média={m:.4f}, std={s:.4f}")
                    if len(cols) > 3:
                        print(f"          ... (+{len(cols) - 3} colunas)")

                if cls_name == "RobustScaler":
                    print(f"        quantile_range={step.quantile_range}")
                elif cls_name == "MinMaxScaler":
                    print(f"        feature_range={step.feature_range}")
                elif cls_name == "KNNImputer":
                    print(f"        n_neighbors={step.n_neighbors}")

                if hasattr(step, "categories_") and step.categories_ is not None:
                    for c, cats in zip(cols[:3], step.categories_[:3]):
                        print(
                            f"          {c}: {list(cats[:5])}{'...' if len(cats) > 5 else ''}"
                        )
                    if len(cols) > 3:
                        print(f"          ... (+{len(cols) - 3} colunas)")

                if cls_name == "TargetEncoder" and hasattr(step, "encodings_"):
                    print(
                        f"          target_type={step.target_type} | smooth={step.smooth}"
                    )

                if hasattr(step, "explained_variance_ratio_"):
                    total_var = step.explained_variance_ratio_.sum()
                    print(
                        f"        n_components={step.n_components_} | variância explicada={total_var:.1%}"
                    )

        print("\n📐 Transformações do Engine (pré-pipeline):")
        if self._log_cols:
            print(f"   Log1p aplicado em {len(self._log_cols)} coluna(s):")
            for c in self._log_cols:
                print(f"     - {c}")
        else:
            print("   Log1p: nenhuma coluna")

        if self._outlier_bounds:
            print(f"   Clipping IQR aplicado em {len(self._outlier_bounds)} coluna(s):")
            for c, (lo, hi) in list(self._outlier_bounds.items())[:5]:
                print(f"     - {c}: [{lo:.4f}, {hi:.4f}]")
            if len(self._outlier_bounds) > 5:
                print(f"     ... (+{len(self._outlier_bounds) - 5} colunas)")
        else:
            print("   Clipping IQR: nenhuma coluna")

        if self._rare_categories:
            print(
                f"   Rare-label agrupado em {len(self._rare_categories)} coluna(s) categórica(s)"
            )
        else:
            print("   Rare-label: nenhuma coluna")

        print(f"\n📋 Features finais ({len(self.selected_features)}):")
        print(f"   {self.selected_features}")
        print("\n" + "─" * 60)

    # -----------------------------------------------------------------------
    # THRESHOLD HELPERS
    # -----------------------------------------------------------------------

    def get_threshold(
        self,
        fpr: np.ndarray,
        tpr: np.ndarray,
        thresholds: np.ndarray,
        y_true: np.ndarray = None,
        y_prob: np.ndarray = None,
    ) -> tuple:
        strategy = self.params.get("threshold_strategy", "youden")

        if strategy == "youden":
            idx = np.argmax(tpr - fpr)
            return thresholds[idx], fpr[idx], tpr[idx]

        elif strategy == "f_beta":
            beta = self.params.get("beta", 1.0)
            if y_true is None or y_prob is None:
                idx = np.argmax(tpr - fpr)
                return thresholds[idx], fpr[idx], tpr[idx]
            prec_arr, rec_arr, thr_pr = precision_recall_curve(y_true, y_prob)
            f_beta = (
                (1 + beta**2)
                * prec_arr[:-1]
                * rec_arr[:-1]
                / ((beta**2 * prec_arr[:-1]) + rec_arr[:-1] + 1e-9)
            )
            best_idx = np.argmax(f_beta)
            best_thr = thr_pr[best_idx]
            roc_idx = np.argmin(np.abs(thresholds - best_thr))
            return thresholds[roc_idx], fpr[roc_idx], tpr[roc_idx]

        elif strategy == "cost_matrix":
            cost_fp = self.params.get("cost_fp", 1.0)
            cost_fn = self.params.get("cost_fn", 1.0)
            cost = fpr * cost_fp + (1 - tpr) * cost_fn
            idx = np.argmin(cost)
            return thresholds[idx], fpr[idx], tpr[idx]

        else:
            idx = np.argmax(tpr - fpr)
            return thresholds[idx], fpr[idx], tpr[idx]

    # -----------------------------------------------------------------------
    # 2.5  PROFILE-BASED FEATURE ENGINEERING  [NEW v0.1.1]
    # -----------------------------------------------------------------------

    def _compute_profile_features(self, df_with_target: pd.DataFrame) -> list:
        """
        Roda ProfileAnalyzer agrupado pelo target e extrai tokens de maior lift
        por classe. Retorna lista de specs de features binárias a criar.

        Restringe análise a colunas categóricas/object — comparações de
        igualdade exata em numéricas contínuas não fazem sentido.

        Parâmetros consumidos de self.params
        ------------------------------------
        profile_min_lift         : float = 2.5
        profile_topn_per_group   : int   = 10
        profile_min_support_pct  : float = 0.05
        profile_one_per_col      : bool  = False
        profile_max_features     : int   = 50
        """
        min_lift = self.params.get("profile_min_lift", 2.5)
        topn = self.params.get("profile_topn_per_group", 10)
        min_sup = self.params.get("profile_min_support_pct", 0.05)
        one_per_col = self.params.get("profile_one_per_col", False)
        max_feats = self.params.get("profile_max_features", 50)

        cat_cols = df_with_target.select_dtypes(
            include=["object", "category"]
        ).columns.tolist()
        cat_cols = [c for c in cat_cols if c != self.target]

        if not cat_cols:
            warnings.warn(
                "[ProfileFeatures] Nenhuma coluna categórica encontrada. "
                "Nenhuma feature de perfil será criada.",
                UserWarning,
            )
            return []

        non_cat = [
            c for c in df_with_target.columns if c not in cat_cols and c != self.target
        ]

        pa = ProfileAnalyzer(
            df=df_with_target,
            group_mode="existing",
            group_col=self.target,
            label_col=self.target,
            exclude_cols=non_cat,
            binarize=False,
            max_unique_ratio=0.5,
        )
        pa.fit(verbose=False)

        seen_tokens = set()
        feature_specs = []

        for g in sorted(pa.groups_, key=str):
            if len(feature_specs) >= max_feats:
                break

            df_p = pa.profile_group(
                g,
                topn=topn,
                min_lift=min_lift,
                min_support_pct=min_sup,
                one_per_col=one_per_col,
                include_raw=True,
            )
            if df_p.empty:
                continue

            df_p = df_p.sort_values("Lift", ascending=False)

            for _, row in df_p.iterrows():
                if len(feature_specs) >= max_feats:
                    break

                raw = row.get("Token_raw", "")
                if not raw or raw in seen_tokens:
                    continue
                seen_tokens.add(raw)

                parts = raw.split("___", 1)
                if len(parts) != 2:
                    continue
                col, val_token = parts[0], parts[1]

                if col not in df_with_target.columns:
                    continue

                feat_name = _make_profile_feat_name(col, val_token)
                feature_specs.append(
                    {
                        "col": col,
                        "val_token": val_token,
                        "feat_name": feat_name,
                        "group": g,
                        "lift": round(float(row["Lift"]), 4),
                        "pct_in": round(float(row["Pct_in"]), 4),
                    }
                )

        return feature_specs

    def _apply_profile_features(
        self,
        df: pd.DataFrame,
        feature_specs: list,
    ) -> pd.DataFrame:
        """
        Cria colunas binárias {0,1} no df para cada spec de feature de perfil.
        Seguro tanto em treino quanto em inferência — não usa target.
        """
        if not feature_specs:
            return df

        df = df.copy()
        for spec in feature_specs:
            col = spec["col"]
            val_token = spec["val_token"]
            feat_name = spec["feat_name"]

            if col not in df.columns:
                df[feat_name] = 0
                continue

            col_tokenized = _to_token(df[col].astype("object"))
            df[feat_name] = (col_tokenized == val_token).astype(int)

        return df

    def _profile_based_feature_engineering(
        self,
        df: pd.DataFrame,
        is_train: bool = True,
        _external_specs: list = None,
    ) -> pd.DataFrame:
        """
        Orquestra criação/aplicação de profile features.

        is_train=True  : computa specs a partir de df (deve conter target),
                         armazena em self._profile_features e aplica.
        is_train=False : aplica specs armazenados (ou _external_specs).

        _external_specs é interno — usado pelo cross_validate() para
        recomputar specs por fold usando só dados de treino, evitando leakage.
        """
        if not self.params.get("use_profile_features", False):
            return df

        if is_train:
            self._profile_features = self._compute_profile_features(df)
            n = len(self._profile_features)
            print(f"\n   🔬 Profile Features: {n} indicadores binários criados")
            if self._profile_features:
                top5 = sorted(
                    self._profile_features, key=lambda x: x["lift"], reverse=True
                )[:5]
                print("      Top-5 por Lift:")
                for pf in top5:
                    print(
                        f"        {pf['feat_name']}"
                        f"  (lift={pf['lift']:.2f},"
                        f" pct_in={pf['pct_in']:.1%},"
                        f" grupo={pf['group']})"
                    )
            specs = self._profile_features
        else:
            specs = (
                _external_specs
                if _external_specs is not None
                else self._profile_features
            )

        return self._apply_profile_features(df, specs)

    def get_profile_features_report(self) -> pd.DataFrame:
        """
        DataFrame com todas as profile features criadas, ordenadas por lift.
        Útil para inspeção e documentação do modelo.
        """
        if not self._profile_features:
            print(
                "⚠️  Nenhuma profile feature disponível. "
                "Execute fit() com use_profile_features=True."
            )
            return pd.DataFrame()

        df_rep = pd.DataFrame(self._profile_features)
        df_rep["Atributo"] = (
            df_rep["col"]
            + ": "
            + df_rep["val_token"]
            .str.replace("_DOT_", ".", regex=False)
            .str.replace(r"(?<=\d)_(?=\d)", "–", regex=True)
        )
        return (
            df_rep[["feat_name", "Atributo", "group", "lift", "pct_in"]]
            .sort_values("lift", ascending=False)
            .reset_index(drop=True)
        )

    # -----------------------------------------------------------------------
    # 5. FIT — fit_selection / fit_model / fit (wrapper)
    # -----------------------------------------------------------------------

    def fit_selection(self, selection_data: pd.DataFrame):
        """
        [v0.1.8] Fase 1+2: Feature engineering + seleção de variáveis.
        NÃO treina o AutoGluon — chame fit_model() em seguida.

        Fluxo recomendado:
            engine.fit_selection(df_completo)    # Boruta vê todos os dados
            engine.fit_model(df_train)           # AutoGluon treina no split
            engine.plot_complete_report(df_test)
        """
        print(f"\n🔍 --- SELEÇÃO DE VARIÁVEIS (v{_FRAMEWORK_VERSION}) ---")

        # 1. Limpeza Base
        df = self._standardize_and_clean(selection_data)

        # 2. Drop Automático
        drop_cols = self.params.get("drop_features", [])
        if drop_cols:
            existing_drop = [c for c in drop_cols if c in df.columns]
            if existing_drop:
                df = df.drop(columns=existing_drop)
                print(
                    f"   🚫 Drop Automático: {existing_drop} removidas por solicitação."
                )

        # 3. Features Temporais
        df = self._extract_temporal_features(df)

        # 4. Guard Multiclasse
        n_classes = df[self.target].nunique()
        if n_classes > 2:
            raise ValueError(
                f"AutoClassificationEngine suporta apenas classificação BINÁRIA. "
                f"Target '{self.target}' possui {n_classes} classes."
            )

        self._validate_group_agg_pairs(df.columns.tolist())

        # 5. Hashing
        try:
            sample = df.head(10_000)
            hash_str = hashlib.sha256(
                pd.util.hash_pandas_object(sample, index=True).values.tobytes()
            ).hexdigest()
            self._train_hash = hash_str
            print(f"\n🔑 Hash dos dados de seleção: {hash_str[:16]}...")
        except Exception:
            self._train_hash = ""

        print(
            f"📊 Dados de seleção: {len(df)} registros × {len(df.columns)-1} features"
        )

        # ====================================================================
        # FASE 1: FEATURE ENGINEERING
        # ====================================================================
        print("\n🔧 Criando group aggregations...")
        df = self._create_group_aggregations(df, is_train=True)
        df = self._handle_rare_labels(df, is_train=True)
        if self.params.get("handle_outliers", True):
            df = self._handle_outliers_and_log(df, is_train=True)
        if self.params.get("use_profile_features", False):
            print("\n🔬 Profile-based Feature Engineering (target-guided)...")
            df = self._profile_based_feature_engineering(df, is_train=True)

        # ====================================================================
        # FASE 2: SELEÇÃO DE VARIÁVEIS
        # ====================================================================
        self.eliminated_features = {
            "leakage": [],
            "alta_cardinalidade": [],
            "colinearidade": [],
            "constantes_pos_rare": [],
            "importancia_nula": [],
            "boruta": [],
        }

        df = self._sanity_check(df)
        df = self._handle_multicollinearity(df)

        self.selected_features = [c for c in df.columns if c != self.target]
        X = df[self.selected_features]
        y = df[self.target]

        print("\n🔗 Calculando associações...")
        self.association_report = self._compute_association_report(df)
        print("   ✅ Relatório pronto.")

        # Null Importance Filter
        if self.params.get("use_importance_filter", False):
            print("\n🔍 Null Importance Filter (LightGBM)...")
            from lightgbm import LGBMClassifier

            X_imp = X.copy()
            cat_cols = X_imp.select_dtypes(
                include=["object", "category"]
            ).columns.tolist()
            for c in cat_cols:
                X_imp[c] = X_imp[c].astype("category")

            clf = LGBMClassifier(
                n_estimators=100,
                random_state=42,
                importance_type="gain",
                verbose=-1,
            )
            clf.fit(X_imp, y)
            actual_imp = clf.feature_importances_

            n_runs = 5
            null_imps = np.zeros((n_runs, X_imp.shape[1]))
            y_shuffled = y.copy().values
            null_pct = self.params.get("null_importance_percentile", 50)
            print(
                f"   🎲 Calculando ruído com {n_runs} permutações do target (percentil={null_pct})..."
            )
            for i in range(n_runs):
                np.random.seed(42 + i)
                np.random.shuffle(y_shuffled)
                clf.fit(X_imp, y_shuffled)
                null_imps[i, :] = clf.feature_importances_

            null_threshold = np.percentile(null_imps, null_pct, axis=0)
            good_features = [
                col
                for idx, col in enumerate(X_imp.columns)
                if actual_imp[idx] > null_threshold[idx] and actual_imp[idx] > 0
            ]
            removed = set(self.selected_features) - set(good_features)
            self.eliminated_features["importancia_nula"] = list(removed)
            self.selected_features = good_features
            X = X[self.selected_features]
            print(
                f"   ✅ Sinal vs Ruído (p{null_pct}): Mantidas {len(good_features)} | Removidas {len(removed)}"
            )

        # Boruta-LightGBM
        if self.params.get("use_boruta_filter", False):
            good_features = self._run_boruta_selection(
                X,
                y,
                max_iters=self.params.get("boruta_iters", 20),
                hit_threshold_pct=self.params.get("boruta_hit_pct", 0.55),
            )
            removed = set(self.selected_features) - set(good_features)
            self.eliminated_features["boruta"] = list(removed)
            self.selected_features = good_features
            X = X[self.selected_features]

        # Whitelist — resgata features protegidas que qualquer filtro tenha derrubado
        whitelist = self.params.get("feature_whitelist", [])
        rescued = []
        if whitelist:
            all_cols = df.columns.tolist()
            for feat in whitelist:
                if (
                    feat in all_cols
                    and feat != self.target
                    and feat not in self.selected_features
                ):
                    self.selected_features.append(feat)
                    rescued.append(feat)
            if rescued:
                X = df[self.selected_features]
                # Remove das listas de eliminados para não confundir o log
                for reason in self.eliminated_features:
                    self.eliminated_features[reason] = [
                        c for c in self.eliminated_features[reason] if c not in rescued
                    ]

        # Resumo
        print("\n📋 RESUMO DA FILTRAGEM DE VARIÁVEIS:")
        label_map = {
            "leakage": "Removidas por Leakage",
            "alta_cardinalidade": "Removidas por Alta Cardinalidade",
            "colinearidade": "Removidas por Colinearidade",
            "constantes_pos_rare": "Removidas por Constantes pós-RareLabel",
            "importancia_nula": "Removidas por Importância Nula",
            "boruta": "Removidas pelo Boruta-LightGBM",
        }
        for reason, cols in self.eliminated_features.items():
            print(f"   => {label_map[reason]}: {cols or 'Nenhuma'}")
        if rescued:
            print(f"   => Resgatadas pela Whitelist ({len(rescued)}): {rescued}")
        print(
            f"   => Features Finais ({len(self.selected_features)}): {self.selected_features}"
        )

        self._train_schema = {col: str(X[col].dtype) for col in self.selected_features}
        print(
            f"\n✅ Seleção concluída! {len(self.selected_features)} features prontas para fit_model()."
        )

    def fit_model(self, train_data: pd.DataFrame, time_limit: int = None):
        """
        [v0.1.8] Fase 3: Treina o AutoGluon com as features selecionadas.
        Requer que fit_selection() tenha sido executado antes.

        Fluxo recomendado:
            engine.fit_selection(df_completo)    # Boruta vê todos os dados
            engine.fit_model(df_train)           # AutoGluon treina no split
            engine.plot_complete_report(df_test)
        """
        if self.selected_features is None:
            raise RuntimeError(
                "Execute fit_selection() antes de fit_model(). "
                "Ou use fit() para rodar ambos automaticamente no mesmo dataset."
            )

        if time_limit is None:
            time_limit = self.params.get("time_limit", 300)

        print(f"\n🚀 --- TREINAMENTO DO MODELO (v{_FRAMEWORK_VERSION}) ---")
        print(f"   📊 Dados de treino: {len(train_data)} registros")
        print(f"   🎯 Features selecionadas: {len(self.selected_features)}")

        # 1. Preprocessing usando transformadores fitados pelo fit_selection
        df = self._standardize_and_clean(train_data)
        drop_cols = self.params.get("drop_features", [])
        if drop_cols:
            existing_drop = [c for c in drop_cols if c in df.columns]
            if existing_drop:
                df = df.drop(columns=existing_drop)
        df = self._extract_temporal_features(df)
        df = self._create_group_aggregations(df, is_train=False)
        df = self._handle_rare_labels(df, is_train=False)
        if self.params.get("handle_outliers", True):
            df = self._handle_outliers_and_log(df, is_train=False)
        if self.params.get("use_profile_features", False):
            df = self._profile_based_feature_engineering(df, is_train=False)

        # 2. Hashing
        try:
            sample = df.head(10_000)
            hash_str = hashlib.sha256(
                pd.util.hash_pandas_object(sample, index=True).values.tobytes()
            ).hexdigest()
            self._train_hash = hash_str
            print(f"   🔑 Hash do treino: {hash_str[:16]}...")
        except Exception:
            self._train_hash = ""

        # 3. Filtra pelas features selecionadas
        available = [c for c in self.selected_features if c in df.columns]
        missing = set(self.selected_features) - set(df.columns)
        if missing:
            warnings.warn(
                f"[fit_model] {len(missing)} feature(s) ausente(s) no train_data: {missing}",
                UserWarning,
            )

        # 4. Split de tuning para o AutoGluon
        tuning_frac = self.params.get("tuning_data_fraction", 0.15)
        df_temp = df[available + [self.target]].copy()
        df_tuning = None

        if tuning_frac > 0:
            try:
                df_core, df_tuning = train_test_split(
                    df_temp,
                    test_size=tuning_frac,
                    stratify=df_temp[self.target],
                    random_state=42,
                )
                print(
                    f"\n✂️  Split AutoGluon: train_core={len(df_core)} | "
                    f"tuning={len(df_tuning)} (fração={tuning_frac:.0%})"
                )
            except ValueError:
                print("⚠️  Split estratificado falhou. Usando dataset completo.")
                df_core = df_temp
        else:
            df_core = df_temp

        X_core = df_core[available]
        y_core = df_core[self.target]

        # 5. Sklearn pipeline
        if self.params.get("use_sklearn_pipeline", True):
            self.pipeline = self._build_sklearn_pipeline(X_core, y=y_core)
            X_core_t = self.pipeline.fit_transform(X_core, y_core)
        else:
            self.pipeline = None
            X_core_t = X_core.copy()

        train_final = X_core_t.copy()
        train_final[self.target] = y_core.values

        tuning_final = None
        if df_tuning is not None:
            X_tuning = df_tuning[[c for c in available if c in df_tuning.columns]]
            X_tuning_t = (
                self.pipeline.transform(X_tuning)
                if self.pipeline is not None
                else X_tuning.copy()
            )
            tuning_final = X_tuning_t.copy()
            tuning_final[self.target] = df_tuning[self.target].values

        # 6. AutoGluon
        chosen_metric = self.params.get("eval_metric", "f1")
        chosen_preset = self.params.get("presets", "high_quality")
        thr_strategy = self.params.get("threshold_strategy", "youden")
        corr_method = self.params.get("corr_method", "pearson")

        print(
            f"\n🎯 Métrica: {chosen_metric} | Preset: {chosen_preset} | Time: {time_limit}s"
            f" | Threshold: {thr_strategy} | corr_method: {corr_method}"
        )

        hyperparams = "default"
        if self.params.get("prune_models", False):
            hyperparams = {"GBM": {}, "CAT": {}, "XGB": {}}

        fit_kwargs = {
            "time_limit": time_limit,
            "presets": chosen_preset,
            "num_cpus": self.params.get("num_cpus", os.cpu_count() or 6),
            "ag_args_ensemble": self.params.get(
                "ag_args_ensemble", {"fold_fitting_strategy": "sequential_local"}
            ),
            "hyperparameters": hyperparams,
        }

        if tuning_final is not None:
            fit_kwargs["tuning_data"] = tuning_final
            fit_kwargs["use_bag_holdout"] = True

        for key in ["dynamic_stacking", "num_bag_folds", "num_bag_sets"]:
            if key in self.params:
                fit_kwargs[key] = self.params[key]

        if "dynamic_stacking" not in fit_kwargs:
            fit_kwargs["dynamic_stacking"] = False

        for key in ["save_space", "keep_only_best"]:
            if key in self.params:
                fit_kwargs[key] = self.params[key]

        if "feature_generator" in self.params:
            fit_kwargs["feature_generator"] = self.params["feature_generator"]
            print("   🔧 feature_generator customizado ativo.")

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.predictor = TabularPredictor(
                label=self.target,
                eval_metric=chosen_metric,
            ).fit(train_final, **fit_kwargs)

        self.compute_feature_importance(
            data=tuning_final if tuning_final is not None else train_final
        )

        print("\n✅ Modelo treinado!")
        print(f"   📐 Threshold  : {thr_strategy}")
        print(f"   📊 corr_method: {corr_method}")

    def fit(
        self,
        train_data: pd.DataFrame,
        time_limit: int = None,
        skip_selection: bool = False,
    ):
        """
        Wrapper backward-compatible.

        - skip_selection=False (padrão): chama fit_selection() + fit_model()
          no mesmo dataset. Comportamento idêntico às versões anteriores.

        - skip_selection=True: modo CV — re-fita transformadores no fold,
          pula seleção pesada, treina AutoGluon. Uso interno do cross_validate().

        Para máxima performance do Boruta, prefira o novo fluxo explícito:
            engine.fit_selection(df_completo)
            engine.fit_model(df_train)
        """
        if not skip_selection:
            self.fit_selection(train_data)
            self.fit_model(train_data, time_limit=time_limit)
            return

        # === MODO EXPRESSO (folds de CV) ===
        # Re-fita transformadores no fold e treina AutoGluon direto,
        # sem rodar seleção pesada (Boruta/Null Importance).
        if time_limit is None:
            time_limit = self.params.get("time_limit", 300)

        df = self._standardize_and_clean(train_data)
        drop_cols = self.params.get("drop_features", [])
        if drop_cols:
            existing_drop = [c for c in drop_cols if c in df.columns]
            if existing_drop:
                df = df.drop(columns=existing_drop)
        df = self._extract_temporal_features(df)

        # Hashing
        try:
            self._train_hash = hashlib.sha256(
                pd.util.hash_pandas_object(df.head(10_000), index=True).values.tobytes()
            ).hexdigest()
        except Exception:
            self._train_hash = ""

        # Feature Engineering com is_train=True no fold
        df = self._create_group_aggregations(df, is_train=True)
        df = self._handle_rare_labels(df, is_train=True)
        if self.params.get("handle_outliers", True):
            df = self._handle_outliers_and_log(df, is_train=True)
        if self.params.get("use_profile_features", False):
            df = self._profile_based_feature_engineering(df, is_train=True)

        # Zera eliminated_features sem perder o log global
        self.eliminated_features = {
            "leakage": [],
            "alta_cardinalidade": [],
            "colinearidade": [],
            "constantes_pos_rare": [],
            "importancia_nula": [],
            "boruta": [],
        }

        print("   ⏩ Pulando seleção de variáveis (modo skip_selection ativo).")
        cols_to_keep = [c for c in self.selected_features if c in df.columns]
        X_core = df[cols_to_keep]
        y_core = df[self.target]

        # Sklearn pipeline
        if self.params.get("use_sklearn_pipeline", True):
            self.pipeline = self._build_sklearn_pipeline(X_core, y=y_core)
            X_core_t = self.pipeline.fit_transform(X_core, y_core)
        else:
            self.pipeline = None
            X_core_t = X_core.copy()

        train_final = X_core_t.copy()
        train_final[self.target] = y_core.values

        chosen_metric = self.params.get("eval_metric", "f1")
        chosen_preset = self.params.get("presets", "high_quality")

        hyperparams = "default"
        if self.params.get("prune_models", False):
            hyperparams = {"GBM": {}, "CAT": {}, "XGB": {}}

        fit_kwargs = {
            "time_limit": time_limit,
            "presets": chosen_preset,
            "num_cpus": self.params.get("num_cpus", os.cpu_count() or 6),
            "ag_args_ensemble": self.params.get(
                "ag_args_ensemble", {"fold_fitting_strategy": "sequential_local"}
            ),
            "hyperparameters": hyperparams,
            "dynamic_stacking": self.params.get("dynamic_stacking", False),
        }

        for key in ["num_bag_folds", "num_bag_sets", "save_space", "keep_only_best"]:
            if key in self.params:
                fit_kwargs[key] = self.params[key]

        if "feature_generator" in self.params:
            fit_kwargs["feature_generator"] = self.params["feature_generator"]

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.predictor = TabularPredictor(
                label=self.target,
                eval_metric=chosen_metric,
            ).fit(train_final, **fit_kwargs)

        self.compute_feature_importance(data=train_final)

    def check_adversarial_drift(
        self,
        train_data: pd.DataFrame,
        test_data: pd.DataFrame,
        sample_size: int = 100000,
    ) -> pd.DataFrame:
        """
        Escudo Temporal: Verifica se os dados de teste/produção sofreram
        mutação em relação aos dados de treino.
        """
        print("\n🛡️ --- INICIANDO ESCUDO TEMPORAL (Adversarial Validation) ---")

        # 1. Prepara as bases com a flag de origem
        df_tr = train_data.copy()
        df_te = test_data.copy()

        df_tr["__is_test__"] = 0
        df_te["__is_test__"] = 1

        df_concat = pd.concat([df_tr, df_te], ignore_index=True)

        # O target do modelo real não pode entrar aqui
        if self.target in df_concat.columns:
            df_concat = df_concat.drop(columns=[self.target])

        # Amostragem inteligente para não fritar o cluster do Databricks
        if len(df_concat) > sample_size:
            df_concat = df_concat.sample(n=sample_size, random_state=42)
            print(f"   ⚖️  Amostragem aplicada: reduzido para {sample_size} registros.")

        # Usa as próprias ferramentas do seu framework para limpar o dado
        df_concat = self._standardize_and_clean(df_concat)
        df_concat = self._extract_temporal_features(df_concat)

        # Separa X e y
        X = df_concat.drop(columns=["__is_test__"])
        y = df_concat["__is_test__"]

        # Prepara categóricas pro LightGBM
        cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
        for c in cat_cols:
            X[c] = X[c].astype("category")

        # 2. Treina o Detector de Mutação usando Cross-Validation
        from lightgbm import LGBMClassifier
        from sklearn.model_selection import cross_val_predict
        from sklearn.metrics import roc_auc_score

        print("   ⚔️  Treinando detector de mutação de dados...")
        clf = LGBMClassifier(n_estimators=100, random_state=42, verbose=-1)

        # Gera predições out-of-fold para ter um AUC totalmente honesto
        y_prob = cross_val_predict(clf, X, y, cv=3, method="predict_proba")[:, 1]
        auc_score = roc_auc_score(y, y_prob)

        # 3. Laudo do Escudo
        print(f"\n📊 RESULTADO DO ESCUDO:")
        if auc_score > 0.70:
            print(f"   🚨 ALERTA VERMELHO! AUC = {auc_score:.4f}")
            print(
                "   Os dados sofreram mutação forte no tempo. Risco de quebra de modelo na produção!"
            )
        elif auc_score > 0.60:
            print(f"   ⚠️  Atenção. AUC = {auc_score:.4f}")
            print(
                "   Existe um leve drift nos dados. O modelo deve sobreviver, mas fique de olho nas features suspeitas."
            )
        else:
            print(f"   ✅ Base Segura! AUC = {auc_score:.4f}")
            print(
                "   Treino e Teste são estatisticamente indistinguíveis. Pode seguir o jogo."
            )

        # 4. Investigação Criminal: Quem está causando o drift?
        clf.fit(X, y)
        importances = clf.feature_importances_

        df_drifters = pd.DataFrame(
            {"Feature": X.columns, "Drift_Score": importances}
        ).sort_values("Drift_Score", ascending=False)

        # Normaliza o score pra ficar legível (0 a 100%)
        max_score = df_drifters["Drift_Score"].max()
        if max_score > 0:
            df_drifters["Impacto_Relativo"] = (
                df_drifters["Drift_Score"] / max_score
            ) * 100
        else:
            df_drifters["Impacto_Relativo"] = 0.0

        top_drifters = df_drifters[df_drifters["Impacto_Relativo"] > 5.0].head(10)

        if auc_score > 0.60 and not top_drifters.empty:
            print("\n🕵️ Principais suspeitas de mutação (Drifters):")
            for _, row in top_drifters.iterrows():
                print(
                    f"   - {row['Feature']}: {row['Impacto_Relativo']:.1f}% de responsabilidade"
                )

        return df_drifters[["Feature", "Drift_Score", "Impacto_Relativo"]]

    # -----------------------------------------------------------------------
    # 5b. FEATURE IMPORTANCE PÚBLICO
    # -----------------------------------------------------------------------

    def compute_feature_importance(
        self, data: pd.DataFrame = None, num_shuffle_sets: int = 5
    ):
        if self.predictor is None:
            raise RuntimeError("Execute fit() antes de compute_feature_importance().")

        print(f"\n📊 Calculando feature importance ({num_shuffle_sets} permutações)...")
        try:
            self.feature_importance = self.predictor.feature_importance(
                data=data,
                num_shuffle_sets=num_shuffle_sets,
                silent=True,
            )
            print("   ✅ Feature importance calculada.")
        except Exception as e:
            print(f"   ⚠️  Falha ao calcular importância: {e}")
            self.feature_importance = None

    def cross_validate(
        self,
        train_data: pd.DataFrame,
        n_folds: int = 5,
        n_repeats: int = 1,
        time_limit_per_fold: int = None,
        skip_selection: bool = True,
    ) -> pd.DataFrame:
        """
        Executa Cross-Validation estratificado reaproveitando o pipeline do fit().
        Versão final calibrada para os relatórios visuais (sem erros de KeyError).
        """
        if self.selected_features is None:
            raise RuntimeError(
                "Execute fit() antes de cross_validate() para selecionar as features."
            )

        # 1. Parâmetros de Tempo e Folds
        total_tl = self.params.get("time_limit", 300)
        if time_limit_per_fold is None:
            time_limit_per_fold = max(60, total_tl // n_folds)

        total_folds = n_folds * n_repeats
        print(f"\n🔄 --- CROSS-VALIDATION ({n_folds} folds) ---")
        print(
            f"   Time limit por fold: {time_limit_per_fold}s | skip_selection: {skip_selection}"
        )

        # 2. Preparação do Dataset Base
        df_base = self._standardize_and_clean(train_data)
        df_base = self._extract_temporal_features(df_base)

        y_raw = df_base[self.target]
        pos_label = self._get_positive_class()
        y_strat = (y_raw.astype(str) == str(pos_label)).astype(int)

        if n_repeats > 1:
            skf = RepeatedStratifiedKFold(
                n_splits=n_folds, n_repeats=n_repeats, random_state=42
            )
        else:
            skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

        # 3. Estruturas de Dados para Métricas
        fold_metrics = []
        fold_curves = []
        oof_probs = np.zeros(len(y_raw))
        oof_counts = np.zeros(len(y_raw))
        oof_true = np.zeros(len(y_raw))

        # 4. Loop de Folds (O Motor)
        for fold_idx, (train_idx, val_idx) in enumerate(skf.split(df_base, y_strat)):
            print(
                f"\n   📂 Fold {fold_idx + 1}/{total_folds} (treino={len(train_idx)}, val={len(val_idx)})"
            )

            df_tr_fold = df_base.iloc[train_idx].copy()
            df_vl_fold = df_base.iloc[val_idx].copy()

            # Treina o fold pulando a seleção pesada (Boruta/Null Importance)
            self.fit(
                df_tr_fold,
                time_limit=time_limit_per_fold,
                skip_selection=skip_selection,
            )

            # Predições do Fold
            y_prob_df = self.predict_proba(df_vl_fold)
            y_prob_pos = y_prob_df[pos_label]
            y_pred_raw = self.predict(df_vl_fold)

            # Formatação de alvos (binário 0/1)
            y_bin_vl = (df_vl_fold[self.target].astype(str) == str(pos_label)).astype(
                int
            )
            y_pred_bin = (y_pred_raw.astype(str) == str(pos_label)).astype(int)

            # Acumulação OOF
            oof_probs[val_idx] += y_prob_pos.values
            oof_counts[val_idx] += 1
            oof_true[val_idx] = y_bin_vl.values

            # Cálculo de Curvas e Thresholds
            fpr, tpr, thresh_roc = roc_curve(y_bin_vl, y_prob_pos)
            roc_auc = auc(fpr, tpr)
            pr_auc = average_precision_score(y_bin_vl, y_prob_pos)

            opt_t, _, _ = self.get_threshold(
                fpr, tpr, thresh_roc, y_true=y_bin_vl.values, y_prob=y_prob_pos.values
            )
            y_pred_opt = (y_prob_pos >= opt_t).astype(int)

            # Métricas do Fold (Nomes de chaves exatos para o plot_cv_report)
            fold_metrics.append(
                {
                    "Fold": fold_idx + 1,
                    "AUC-ROC": round(roc_auc, 4),
                    "Gini": round(2 * roc_auc - 1, 4),
                    "Avg Precision": round(pr_auc, 4),
                    "F1 (thr=0.5)": round(
                        f1_score(y_bin_vl, y_pred_bin, zero_division=0), 4
                    ),
                    "F1 (opt)": round(
                        f1_score(y_bin_vl, y_pred_opt, zero_division=0), 4
                    ),
                    "Accuracy": round(accuracy_score(y_bin_vl, y_pred_bin), 4),
                    "Brier": round(brier_score_loss(y_bin_vl, y_prob_pos), 4),
                    "Log-Loss": round(log_loss(y_bin_vl, y_prob_pos), 4),
                }
            )

            fold_curves.append(
                {"fpr": fpr, "tpr": tpr, "auc": roc_auc, "fold": fold_idx + 1}
            )
            print(f"      ✅ AUC={roc_auc:.4f} | Brier={fold_metrics[-1]['Brier']:.4f}")

        # 5. Consolidação Global (Onde os KeyErrors morrem)
        oof_probs_final = oof_probs / np.maximum(oof_counts, 1)

        fpr_oof, tpr_oof, _ = roc_curve(oof_true, oof_probs_final)
        oof_auc = auc(fpr_oof, tpr_oof)
        oof_ap = average_precision_score(oof_true, oof_probs_final)
        oof_brier = brier_score_loss(
            oof_true, oof_probs_final
        )  # <--- O FIX DO ERRO ATUAL

        metrics_df = pd.DataFrame(fold_metrics).set_index("Fold")
        mean_row = metrics_df.mean().rename("Média")
        std_row = metrics_df.std().rename("Std")
        summary_df = pd.concat(
            [metrics_df, mean_row.to_frame().T, std_row.to_frame().T]
        )

        # Armazenamento Final (Alimentação do Plotter)
        self.cv_results = {
            "summary_df": summary_df,
            "fold_curves": fold_curves,
            "oof_probs": oof_probs_final,
            "oof_true": oof_true,
            "fpr_oof": fpr_oof,
            "tpr_oof": tpr_oof,
            "oof_auc": oof_auc,
            "oof_ap": oof_ap,
            "oof_brier": oof_brier,  # <--- AGORA O PLOT VAI ACHAR
            "n_folds": n_folds,
            "n_repeats": n_repeats,
            "pos_label": pos_label,
            "thr_strategy": self.params.get("threshold_strategy", "youden"),
        }

        return summary_df

    # -----------------------------------------------------------------------
    # 5d. PLOT CROSS-VALIDATION
    # -----------------------------------------------------------------------

    def plot_cv_report(self):
        if not self.cv_results:
            raise RuntimeError("Execute cross_validate() antes de plot_cv_report().")

        cv = self.cv_results
        n = cv["n_folds"]
        n_repeats = cv.get("n_repeats", 1)
        total_folds = n * n_repeats
        thr_strategy = cv.get("thr_strategy", "youden")

        try:
            cmap = matplotlib.colormaps.get_cmap("tab10").resampled(total_folds)
        except AttributeError:
            cmap = plt.cm.get_cmap("tab10", total_folds)

        fig, axes = plt.subplots(2, 3, figsize=(20, 12), constrained_layout=True)

        ax = axes[0, 0]
        ax.axis("off")
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
        table.set_fontsize(8.0)
        table.scale(1, 1.5)
        n_rows = len(df_table)
        for j in range(len(df_table.columns)):
            table[n_rows - 1, j].set_facecolor("#d4edda")
            table[n_rows, j].set_facecolor("#fff3cd")
        repeat_tag = f" × {n_repeats}rep" if n_repeats > 1 else ""
        ax.set_title(
            f"Métricas por Fold (n={n}{repeat_tag})",
            fontweight="bold",
            fontsize=12,
        )

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

        ax = axes[0, 2]
        metric_cols = [
            "AUC-ROC",
            "Avg Precision",
            "F1 (thr=0.5)",
            "F1 (opt)",
            "Accuracy",
        ]
        fold_only = cv["summary_df"].iloc[:total_folds][metric_cols]
        ax.boxplot(
            [fold_only[c].values for c in metric_cols],
            labels=metric_cols,
            patch_artist=True,
            boxprops=dict(facecolor="#aec6e8", color="#1f77b4"),
            medianprops=dict(color="red", lw=2),
        )
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
        ax.tick_params(axis="x", rotation=20, labelsize=7.5)

        ax = axes[1, 0]
        oof_df = pd.DataFrame(
            {"prob": cv["oof_probs"], "label": cv["oof_true"].astype(int)}
        )
        for lbl, color, name in [
            (0, _COLOR_TRAIN, "Neg (0)"),
            (1, _COLOR_TEST, "Pos (1)"),
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

        ax = axes[1, 1]
        prob_true_oof, prob_pred_oof = calibration_curve(
            cv["oof_true"], cv["oof_probs"], n_bins=10
        )
        ax.plot(
            prob_pred_oof,
            prob_true_oof,
            "o-",
            lw=2,
            color=_COLOR_TEST,
            label="Calibração OOF",
        )
        ax.plot([0, 1], [0, 1], "k--", label="Perfeito")
        ax.set_title("Calibração OOF", fontweight="bold")
        ax.set_xlabel("Probabilidade Predita")
        ax.set_ylabel("Fração de Positivos")
        ax.legend(fontsize=9)

        ax = axes[1, 2]
        aucs = [fc["auc"] for fc in cv["fold_curves"]]
        folds = [fc["fold"] for fc in cv["fold_curves"]]
        bars = ax.bar(
            folds,
            aucs,
            color=[cmap(i) for i in range(total_folds)],
            edgecolor="white",
            width=0.6,
        )
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

        repeat_info = f" | {n_repeats} repetições" if n_repeats > 1 else ""
        plt.suptitle(
            f"Relatório CV ({n} Folds Estratificados{repeat_info})  —  "
            f"AUC OOF: {cv['oof_auc']:.4f}  |  AP OOF: {cv['oof_ap']:.4f}"
            f"  |  Brier OOF: {cv['oof_brier']:.4f}"
            f"  |  Threshold: {thr_strategy}",
            fontsize=14,
            fontweight="bold",
        )
        plt.show()

    # -----------------------------------------------------------------------
    # 5e. PROFILE ANALYZER — factory method
    # -----------------------------------------------------------------------

    def profile_analyzer(
        self,
        df: pd.DataFrame,
        score_col: str = "prob_target",
        group_mode: str = "decile",
        n_quantiles: int = 10,
        **kwargs,
    ) -> "ProfileAnalyzer":
        if self.selected_features is None:
            raise RuntimeError("Execute fit() antes de profile_analyzer().")

        default_exclude = list(kwargs.pop("exclude_cols", []))
        default_exclude += [self.target]
        if score_col not in default_exclude:
            default_exclude_clean = [c for c in default_exclude if c != score_col]
        else:
            default_exclude_clean = default_exclude

        return ProfileAnalyzer(
            df=df,
            group_mode=group_mode,
            score_col=score_col,
            n_quantiles=n_quantiles,
            label_col=kwargs.pop("label_col", self.target),
            exclude_cols=default_exclude_clean,
            **kwargs,
        )

    # -----------------------------------------------------------------------
    # 6. PREDIÇÃO
    # -----------------------------------------------------------------------

    def _get_positive_class(self):
        override = self.params.get("positive_class", None)
        if override is not None:
            return override
        return self.predictor.positive_class

    def _preprocess_for_inference(self, data: pd.DataFrame):
        df_proc = self._standardize_and_clean(data)
        df_proc = self._extract_temporal_features(df_proc)
        df_proc = self._create_group_aggregations(df_proc, is_train=False)
        df_proc = self._handle_rare_labels(df_proc, is_train=False)
        df_proc = self._handle_outliers_and_log(df_proc, is_train=False)
        # [NEW v0.1.1] Aplica profile features em inferência
        df_proc = self._profile_based_feature_engineering(df_proc, is_train=False)

        has_target = self.target in df_proc.columns
        y = df_proc[self.target] if has_target else None

        available = [c for c in self.selected_features if c in df_proc.columns]
        missing_cols = set(self.selected_features) - set(df_proc.columns)
        if missing_cols:
            warnings.warn(
                f"[Inferência] {len(missing_cols)} feature(s) ausente(s): {missing_cols}.",
                UserWarning,
                stacklevel=2,
            )
        X = df_proc[available]

        if self._train_schema:
            for col in available:
                expected = self._train_schema.get(col)
                actual = str(X[col].dtype)
                if expected and actual != expected:
                    warnings.warn(
                        f"[Schema] '{col}': dtype esperado '{expected}', recebido '{actual}'.",
                        UserWarning,
                        stacklevel=2,
                    )

        X_trans = self.pipeline.transform(X) if self.pipeline is not None else X.copy()
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

    def _get_decile_stats(self, y_true, y_prob, bins=10, cut_edges=None):
        df = pd.DataFrame({"target": y_true, "prob": y_prob})

        if cut_edges is None:
            df["decile"] = pd.qcut(
                df["prob"].rank(method="first", ascending=True),
                bins,
                labels=range(1, bins + 1),
                retbins=False,
                duplicates="drop",
            )
            cut_edges = np.quantile(df["prob"], np.linspace(0, 1, bins + 1))
            cut_edges = np.unique(cut_edges)
            cut_edges[0] -= 1e-6
            cut_edges[-1] += 1e-6
        else:
            cut_edges = np.unique(cut_edges)
            n_labels = len(cut_edges) - 1
            df["decile"] = pd.cut(
                df["prob"],
                bins=cut_edges,
                include_lowest=True,
                labels=range(1, n_labels + 1),
                ordered=True,
            )

        stats = (
            df.groupby("decile", observed=False)
            .agg(count=("target", "count"), events=("target", "sum"))
            .reset_index()
        )
        stats["event_rate"] = stats["events"] / stats["count"].replace(0, np.nan)
        global_rate = df["target"].mean()
        stats["lift"] = stats["event_rate"] / global_rate if global_rate > 0 else np.nan

        stats_sorted = stats.sort_values("decile", ascending=False).copy()
        total_pos = stats_sorted["events"].sum()
        total_neg = (stats_sorted["count"] - stats_sorted["events"]).sum()

        stats_sorted["cum_pos_rate"] = stats_sorted["events"].cumsum() / max(
            total_pos, 1
        )
        stats_sorted["cum_neg_rate"] = (
            stats_sorted["count"] - stats_sorted["events"]
        ).cumsum() / max(total_neg, 1)
        stats_sorted["ks"] = abs(
            stats_sorted["cum_pos_rate"] - stats_sorted["cum_neg_rate"]
        )
        stats = stats_sorted.sort_values("decile").reset_index(drop=True)

        score_ranges = (
            df.groupby("decile", observed=False)["prob"]
            .agg(score_min="min", score_max="max")
            .reset_index()
        )
        stats = stats.merge(score_ranges, on="decile", how="left")
        stats["intervalo"] = stats.apply(
            lambda r: (
                f"[{r['score_min']:.2f},{r['score_max']:.2f}]"
                if pd.notna(r["score_min"])
                else "—"
            ),
            axis=1,
        )
        stats["pct_positivos"] = stats["event_rate"] * 100

        return stats, cut_edges

    def _youden_threshold(self, fpr, tpr, thresholds):
        idx = np.argmax(tpr - fpr)
        return thresholds[idx], fpr[idx], tpr[idx]

    def plot_association_report(self):
        if not self.association_report:
            raise RuntimeError("Execute fit() antes de plotar associações.")

        feat_target = self.association_report["feat_target"]
        matrix = self.association_report["feat_feat_matrix"]
        corr_method = self.params.get("corr_method", "pearson")

        fig, axes = plt.subplots(1, 2, figsize=(18, max(6, len(feat_target) * 0.5)))

        ax = axes[0]
        sorted_feats = sorted(
            feat_target.items(), key=lambda x: x[1]["valor"], reverse=True
        )
        labels = [f"{k}\n({v['metrica']})" for k, v in sorted_feats]
        values = [v["valor"] for _, v in sorted_feats]
        colors_bar = [
            "#d62728" if v > 0.8 else "#ff7f0e" if v > 0.5 else "#1f77b4"
            for v in values
        ]
        bars = ax.barh(labels[::-1], values[::-1], color=colors_bar[::-1])
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
            f"Matriz de Associação entre Features\n"
            f"(Pearson/Spearman[{corr_method}] | Theil's U | Eta²)",
            fontweight="bold",
        )
        plt.suptitle("Relatório de Associações", fontsize=16, fontweight="bold")
        plt.tight_layout()
        plt.show()

    def plot_complete_report(
        self,
        test_data: pd.DataFrame,
        train_data: pd.DataFrame = None,
        use_oof: bool = False,
        bins: int = 10,
    ):
        datasets_raw = {"Teste": test_data}
        if train_data is not None and not use_oof:
            datasets_raw["Treino"] = train_data

        results = {}
        pos_label = self._get_positive_class()

        # [NEW v0.1.0] Cores com alta distinção visual treino vs. teste
        colors = {
            "Treino": _COLOR_TRAIN,
            "Treino (OOF)": _COLOR_TRAIN,
            "Teste": _COLOR_TEST,
        }
        thr_strategy = self.params.get("threshold_strategy", "youden")

        for name, data in datasets_raw.items():
            X_trans, y = self._preprocess_for_inference(data)
            y_prob_pos = self.predictor.predict_proba(X_trans)[pos_label].values
            y_bin = (y.astype(str).values == str(pos_label)).astype(int)
            y_pred = (
                self.predictor.predict(X_trans).astype(str).values == str(pos_label)
            ).astype(int)

            fpr, tpr, thresholds = roc_curve(y_bin, y_prob_pos)
            roc_auc = auc(fpr, tpr)
            prec_full, rec_full, thresh_pr = precision_recall_curve(y_bin, y_prob_pos)
            pr_auc = average_precision_score(y_bin, y_prob_pos)
            prob_true, prob_pred = calibration_curve(y_bin, y_prob_pos, n_bins=bins)

            opt_thresh, opt_fpr, opt_tpr = self.get_threshold(
                fpr, tpr, thresholds, y_true=y_bin, y_prob=y_prob_pos
            )
            y_pred_opt = (y_prob_pos >= opt_thresh).astype(int)

            # [NEW v0.1.0] Brier Score
            brier = brier_score_loss(y_bin, y_prob_pos)

            results[name] = {
                "y_true": y_bin,
                "y_prob": y_prob_pos,
                "y_pred": y_pred,
                "y_pred_youden": y_pred_opt,
                "fpr": fpr,
                "tpr": tpr,
                "thresholds": thresholds,
                "youden_thresh": opt_thresh,
                "youden_fpr": opt_fpr,
                "youden_tpr": opt_tpr,
                "metrics": {
                    "AUC-ROC": roc_auc,
                    "Gini": 2 * roc_auc - 1,
                    "F1 (thr=0.5)": f1_score(y_bin, y_pred, zero_division=0),
                    "F1 (opt)": f1_score(y_bin, y_pred_opt, zero_division=0),
                    "Recall (thr=0.5)": recall_score(y_bin, y_pred, zero_division=0),
                    "Log-Loss": log_loss(y_bin, y_prob_pos),
                    "Brier Score": brier,
                    "Accuracy": accuracy_score(y_bin, y_pred),
                    "Avg Precision": pr_auc,
                },
                "prec": prec_full[:-1],
                "rec": rec_full[:-1],
                "prec_full": prec_full,
                "rec_full": rec_full,
                "thresh_pr": thresh_pr,
                "calib_true": prob_true,
                "calib_pred": prob_pred,
            }

        if use_oof:
            if not self.cv_results:
                raise RuntimeError(
                    "use_oof=True requer que cross_validate() tenha sido executado antes."
                )

            oof_probs = self.cv_results["oof_probs"]
            oof_true = self.cv_results["oof_true"].astype(int)

            fpr_oof, tpr_oof, thr_oof = roc_curve(oof_true, oof_probs)
            roc_auc_oof = auc(fpr_oof, tpr_oof)
            prec_full_oof, rec_full_oof, thresh_pr_oof = precision_recall_curve(
                oof_true, oof_probs
            )
            pr_auc_oof = average_precision_score(oof_true, oof_probs)
            prob_true_oof, prob_pred_oof = calibration_curve(
                oof_true, oof_probs, n_bins=bins
            )

            opt_thresh_oof, opt_fpr_oof, opt_tpr_oof = self.get_threshold(
                fpr_oof, tpr_oof, thr_oof, y_true=oof_true, y_prob=oof_probs
            )
            y_pred_oof = (oof_probs >= 0.5).astype(int)
            y_pred_oof_opt = (oof_probs >= opt_thresh_oof).astype(int)
            brier_oof = brier_score_loss(oof_true, oof_probs)

            results["Treino (OOF)"] = {
                "y_true": oof_true,
                "y_prob": oof_probs,
                "y_pred": y_pred_oof,
                "y_pred_youden": y_pred_oof_opt,
                "fpr": fpr_oof,
                "tpr": tpr_oof,
                "thresholds": thr_oof,
                "youden_thresh": opt_thresh_oof,
                "youden_fpr": opt_fpr_oof,
                "youden_tpr": opt_tpr_oof,
                "metrics": {
                    "AUC-ROC": roc_auc_oof,
                    "Gini": 2 * roc_auc_oof - 1,
                    "F1 (thr=0.5)": f1_score(oof_true, y_pred_oof, zero_division=0),
                    "F1 (opt)": f1_score(oof_true, y_pred_oof_opt, zero_division=0),
                    "Recall (thr=0.5)": recall_score(
                        oof_true, y_pred_oof, zero_division=0
                    ),
                    "Log-Loss": log_loss(oof_true, oof_probs),
                    "Brier Score": brier_oof,
                    "Accuracy": accuracy_score(oof_true, y_pred_oof),
                    "Avg Precision": pr_auc_oof,
                },
                "prec": prec_full_oof[:-1],
                "rec": rec_full_oof[:-1],
                "prec_full": prec_full_oof,
                "rec_full": rec_full_oof,
                "thresh_pr": thresh_pr_oof,
                "calib_true": prob_true_oof,
                "calib_pred": prob_pred_oof,
            }

        ref_name = "Treino (OOF)" if use_oof else "Treino"

        decile_indep = {}
        train_cut_edges = None

        for name in [ref_name, "Teste"]:
            if name not in results:
                continue
            stats, edges = self._get_decile_stats(
                results[name]["y_true"], results[name]["y_prob"], bins=bins
            )
            decile_indep[name] = stats
            results[name]["decile_stats"] = stats
            results[name]["metrics"]["KS"] = stats["ks"].max()
            if name == ref_name:
                train_cut_edges = edges

        decile_fixed = {}
        if train_cut_edges is not None:
            for name in [ref_name, "Teste"]:
                if name not in results:
                    continue
                stats_fixed, _ = self._get_decile_stats(
                    results[name]["y_true"],
                    results[name]["y_prob"],
                    bins=bins,
                    cut_edges=train_cut_edges,
                )
                decile_fixed[name] = stats_fixed
        else:
            if "Teste" in results:
                decile_fixed["Teste"] = decile_indep["Teste"]

        ref_for_overfit = ref_name if ref_name in results else None

        fig = plt.figure(figsize=(24, 26), constrained_layout=True)
        gs = fig.add_gridspec(5, 4)

        ax_metrics = fig.add_subplot(gs[0, :2])
        self._plot_scorecard(ax_metrics, results)

        ax_cm_default = fig.add_subplot(gs[0, 2])
        self._plot_confusion(
            ax_cm_default, results["Teste"], pos_label, "Threshold 0.5"
        )

        ax_cm_opt = fig.add_subplot(gs[0, 3])
        self._plot_confusion(
            ax_cm_opt,
            results["Teste"],
            pos_label,
            f"{thr_strategy} ({results['Teste']['youden_thresh']:.2f})",
            use_youden=True,
        )

        ax_roc = fig.add_subplot(gs[1, 0])
        for name, res in results.items():
            ls = "-" if name == "Teste" else "--"
            color = colors.get(name, "#2ca02c")
            ax_roc.plot(
                res["fpr"],
                res["tpr"],
                ls,
                lw=2,
                color=color,
                label=f"{name} AUC={res['metrics']['AUC-ROC']:.3f}",
            )
            if name == "Teste":
                ax_roc.scatter(
                    [res["youden_fpr"]],
                    [res["youden_tpr"]],
                    color="red",
                    zorder=5,
                    s=80,
                    label=thr_strategy.title(),
                )
        ax_roc.plot([0, 1], [0, 1], "k--", alpha=0.3)
        ax_roc.set_title("Curva ROC", fontweight="bold")
        ax_roc.set_xlabel("FPR")
        ax_roc.set_ylabel("TPR")
        ax_roc.legend(fontsize=8)

        ax_pr = fig.add_subplot(gs[1, 1])
        for name, res in results.items():
            ls = "-" if name == "Teste" else "--"
            color = colors.get(name, "#2ca02c")
            ax_pr.plot(
                res["rec_full"],
                res["prec_full"],
                ls,
                lw=2,
                color=color,
                label=f"{name} AP={res['metrics']['Avg Precision']:.3f}",
            )

        if ref_for_overfit and ref_for_overfit in results:
            ap_gap = (
                results[ref_for_overfit]["metrics"]["Avg Precision"]
                - results["Teste"]["metrics"]["Avg Precision"]
            )
            label_ref = "OOF" if use_oof else "Treino"
            if ap_gap > 0.10:
                ax_pr.text(
                    0.5,
                    0.12,
                    f"⚠️ Possível Overfitting\nΔAP {label_ref}−Teste = {ap_gap:.2f}",
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
            color = colors.get(name, "#2ca02c")
            ax_calib.plot(
                res["calib_pred"],
                res["calib_true"],
                "o",
                linestyle=ls,
                color=color,
                label=name,
            )
        ax_calib.plot([0, 1], [0, 1], "k--", label="Perfeito")
        ax_calib.set_title("Curva de Calibração", fontweight="bold")
        ax_calib.set_xlabel("Prob. Predita")
        ax_calib.set_ylabel("Fração de Positivos")
        ax_calib.legend(fontsize=8)

        ax_hist = fig.add_subplot(gs[1, 3])
        for name, res in results.items():
            color = colors.get(name, "#2ca02c")
            sns.kdeplot(
                res["y_prob"], label=name, ax=ax_hist, fill=True, alpha=0.3, color=color
            )
        ax_hist.axvline(
            results["Teste"]["youden_thresh"],
            color="red",
            linestyle="--",
            label=f"{thr_strategy} ({results['Teste']['youden_thresh']:.2f})",
        )
        ax_hist.set_title("Densidade de Probabilidade", fontweight="bold")
        ax_hist.set_xlim(0, 1)
        ax_hist.legend(fontsize=8)

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
            ax_feat.set_title(
                "Top 15 Features (Importância no tuning_holdout)", fontweight="bold"
            )
        else:
            ax_feat.text(
                0.5,
                0.5,
                "Indisponível\nChame compute_feature_importance()",
                ha="center",
                va="center",
                fontsize=10,
            )

        ax_decil_perf = fig.add_subplot(gs[3, :])
        self._plot_decil(
            ax=ax_decil_perf,
            decile_data=decile_indep,
            results=results,
            colors=colors,
            bins=bins,
            title_mode="performance",
        )

        ax_decil_stab = fig.add_subplot(gs[4, :3])
        self._plot_decil(
            ax=ax_decil_stab,
            decile_data=decile_fixed,
            results=results,
            colors=colors,
            bins=bins,
            title_mode="estabilidade",
        )

        ax_ks = fig.add_subplot(gs[4, 3])
        self._plot_ks_curve(ax_ks, results["Teste"])

        oof_tag = " | Treino = OOF" if use_oof else ""
        plt.suptitle(
            f"Relatório Completo de Performance — target: {pos_label}"
            f" | Threshold: {thr_strategy}{oof_tag}",
            fontsize=17,
            weight="bold",
        )
        plt.show()

    def _plot_scorecard(self, ax, results):
        ax.axis("off")
        metric_df = pd.DataFrame({k: v["metrics"] for k, v in results.items()})

        def _color(metric, val):
            # Brier Score: menor = melhor
            if metric == "Brier Score":
                return (
                    "#d4edda"
                    if val <= 0.10
                    else "#fff3cd" if val <= 0.20 else "#f8d7da"
                )
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
                table[i + 1, j + 1].set_facecolor(_color(m, metric_df.loc[m, col]))
        ax.set_title("🏆 Scorecard do Modelo", fontweight="bold", fontsize=13)

    def _plot_confusion(self, ax, res, pos_label, title, use_youden=False):
        y_pred = res["y_pred_youden"] if use_youden else res["y_pred"]
        cm = confusion_matrix(res["y_true"], y_pred)
        neg_label = (
            f"Não {pos_label}" if isinstance(pos_label, str) else f"≠{pos_label}"
        )
        class_labels = [neg_label, str(pos_label)]
        row_totals = cm.sum(axis=1, keepdims=True)
        pct = cm / row_totals * 100
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
        thresholds = res["thresh_pr"]
        prec = res["prec"]
        rec = res["rec"]
        f1 = 2 * prec * rec / (prec + rec + 1e-9)
        thr_strategy = self.params.get("threshold_strategy", "youden")
        opt_thresh = res["youden_thresh"]

        ax.plot(thresholds, prec, label="Precision", color="#2ca02c", lw=2)
        ax.plot(thresholds, rec, label="Recall", color="#d62728", lw=2)
        ax.plot(thresholds, f1, label="F1", color="#1f77b4", lw=2)
        ax.axvline(
            opt_thresh,
            color="#9467bd",
            linestyle="-",
            lw=2,
            label=f"{thr_strategy} ({opt_thresh:.2f})",
        )
        ax.annotate(
            f"{thr_strategy}\n{opt_thresh:.2f}",
            xy=(opt_thresh, 0.97),
            xytext=(opt_thresh + 0.04, 0.97),
            fontsize=7.5,
            color="#9467bd",
            fontweight="bold",
            va="top",
            arrowprops=dict(arrowstyle="->", color="#9467bd", lw=1.2),
        )
        ax.axvline(0.5, color="#7f7f7f", linestyle="--", lw=1.5, label="Default (0.5)")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.05)
        ax.set_xlabel("Threshold de Decisão")
        ax.set_ylabel("Score")
        ax.set_title("Análise de Threshold (Teste)", fontweight="bold")
        ax.legend(fontsize=8)

    def _plot_decil(self, ax, decile_data, results, colors, bins, title_mode):
        """
        [FIX v0.1.0] Labels alinhados por iteração direta container→dados,
        offset relativo ao ylim, ylim expandido para acomodar anotações,
        clip_on=False para labels acima do topo.
        """
        all_stats = []
        for name, stats in decile_data.items():
            d = stats.copy()
            d["Dataset"] = name
            if not isinstance(d["decile"].dtype, pd.CategoricalDtype):
                d["decile"] = pd.Categorical(
                    d["decile"], categories=range(1, bins + 1), ordered=True
                )
            d = d.sort_values("decile")
            all_stats.append(d)

        if not all_stats:
            ax.set_visible(False)
            return

        plot_df = pd.concat(all_stats, ignore_index=True)
        hue_order = [h for h in colors.keys() if h in plot_df["Dataset"].unique()]

        sns.barplot(
            data=plot_df,
            x="decile",
            y="pct_positivos",
            hue="Dataset",
            hue_order=hue_order,
            ax=ax,
            palette=colors,
        )

        for ds in hue_order:
            if ds in results:
                mean_pct = results[ds]["y_true"].mean() * 100.0
                ax.axhline(
                    mean_pct,
                    color=colors[ds],
                    linestyle="--",
                    linewidth=1.5,
                    label=f"Média {ds} ({mean_pct:.1f}%)",
                )

        # [FIX v0.1.0] ① Expande ylim para dar espaço vertical às anotações (3 linhas de texto)
        ylim_lo, ylim_hi = ax.get_ylim()
        ax.set_ylim(ylim_lo, ylim_hi * 1.45)

        # [FIX v0.1.0] ② Offset relativo ao range do eixo (não absoluto)
        y_range = ax.get_ylim()[1] - ax.get_ylim()[0]
        offset = y_range * 0.012

        # [FIX v0.1.0] ③ Itera container[i] ↔ hue_order[i] e alinha diretamente com os dados
        for i, container in enumerate(ax.containers):
            if i >= len(hue_order):
                break
            ds = hue_order[i]

            # Dados do dataset ordenados por decil (mesma ordem que as barras)
            d_sorted = (
                plot_df[plot_df["Dataset"] == ds]
                .sort_values("decile")
                .reset_index(drop=True)
            )

            bar_list = list(container)

            # Garante alinhamento 1:1 — itera juntos sem filtragem prévia
            for bar, (_, row) in zip(bar_list, d_sorted.iterrows()):
                h = bar.get_height()
                if h <= 0 or row["count"] == 0:
                    continue
                x = bar.get_x() + bar.get_width() / 2
                label = (
                    f"{row['pct_positivos']:.1f}%\n"
                    f"{row['intervalo']}\n"
                    f"n={int(row['count'])}"
                )
                ax.text(
                    x,
                    h + offset,
                    label,
                    ha="center",
                    va="bottom",
                    fontsize=6.5,
                    color="black",
                    clip_on=False,  # permite que o texto ultrapasse a borda do axes
                )

        ks_txt = ""
        if title_mode == "performance" and "Teste" in results:
            ks_stat = results["Teste"]["decile_stats"]["ks"].max()
            ks_txt = f"  |  KS Máximo (Teste): {ks_stat:.4f}"

        label_map_bins = {5: "Quintil", 10: "Decil", 4: "Quartil", 20: "Vigésimo"}
        label_type = label_map_bins.get(bins, f"{bins}-Faixas")

        if title_mode == "performance":
            title = f"Taxa de Eventos por {label_type} — Performance (bins independentes){ks_txt}"
            subtitle = "Cada dataset usa seus próprios percentis. Leitura: o modelo discrimina bem?"
        else:
            title = f"Taxa de Eventos por {label_type} — Estabilidade (bins fixos do treino)"
            subtitle = "Teste mapeado nos intervalos de score do treino. Leitura: o comportamento se manteve?"

        ax.set_title(f"{title}\n{subtitle}", fontweight="bold", fontsize=11)
        ax.set_xlabel(f"{label_type} (1=Menor Risco → {bins}=Maior Risco)", fontsize=10)
        ax.set_ylabel("Taxa de Eventos (%)", fontsize=10)
        ax.legend(loc="upper left", fontsize=8)

    def _plot_ks_curve(self, ax, res):
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
            pct_pop,
            cum_pos.values,
            color=_COLOR_TEST,
            lw=2,
            label="Cumulativo Positivos",
        )
        ax.plot(
            pct_pop,
            cum_neg.values,
            color=_COLOR_TRAIN,
            lw=2,
            label="Cumulativo Negativos",
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

    def evaluate_quantile_cutoff(
        self,
        test_data: pd.DataFrame,
        top_q: int = 2,
        bins: int = 10,
        basis: str = "train",
    ):
        """
        Avalia a performance fixando o threshold no limite inferior de um quantil.

        basis="train": Usa a distribuição OOF (treino) para definir o corte (fiel à produção).
        basis="test": Usa a distribuição do próprio teste (ideal para diagnóstico de drift).
        """
        if basis == "train" and not self.cv_results:
            raise RuntimeError(
                "Para usar basis='train', execute cross_validate() antes."
            )

        pos_label = self._get_positive_class()
        neg_label = (
            f"Não {pos_label}" if isinstance(pos_label, str) else f"≠{pos_label}"
        )

        # 1. Obter probabilidades do Teste para avaliação
        X_trans, y_test = self._preprocess_for_inference(test_data)
        y_prob = self.predictor.predict_proba(X_trans)[pos_label].values

        # 2. Definir o Threshold baseado na escolha do usuário
        percentil_corte = 1.0 - (top_q / bins)

        if basis == "train":
            ref_probs = self.cv_results["oof_probs"]
            ref_name = "Treino (OOF)"
        else:
            ref_probs = y_prob
            ref_name = "Próprio Teste"

        cut_threshold = np.quantile(ref_probs, percentil_corte)

        # 3. Gerar predições baseadas no corte
        y_bin = (y_test.astype(str) == str(pos_label)).astype(int)
        y_pred_q = (y_prob >= cut_threshold).astype(int)

        # 4. Calcular Métricas
        cm = confusion_matrix(y_bin, y_pred_q)
        prec = precision_score(y_bin, y_pred_q, zero_division=0)
        rec = recall_score(y_bin, y_pred_q, zero_division=0)
        f1 = f1_score(y_bin, y_pred_q, zero_division=0)
        acc = accuracy_score(y_bin, y_pred_q)
        fracao_atingida = y_pred_q.mean() * 100

        # 5. Visualização
        fig, axes = plt.subplots(
            1, 2, figsize=(14, 5), gridspec_kw={"width_ratios": [1, 1.2]}
        )

        # --- Tabela de Métricas ---
        ax_table = axes[0]
        ax_table.axis("off")

        metrics_data = [
            ["Base de Referência", ref_name],
            ["Threshold (Score de Corte)", f"{cut_threshold:.4f}"],
            ["População de Teste Atingida", f"{fracao_atingida:.1f}%"],
            ["Precision (Acerto no alvo)", f"{prec:.4f}"],
            ["Recall (Captura do risco)", f"{rec:.4f}"],
            ["F1-Score", f"{f1:.4f}"],
            ["Accuracy", f"{acc:.4f}"],
        ]

        table = ax_table.table(
            cellText=metrics_data,
            colLabels=["Métrica", "Valor"],
            loc="center",
            cellLoc="center",
            bbox=[0, 0.1, 1, 0.8],
        )
        table.auto_set_font_size(False)
        table.set_fontsize(11)
        table.scale(1, 2)

        for i in range(len(metrics_data) + 1):
            table[i, 0].set_facecolor("#f2f2f2")
            table[i, 0].set_text_props(weight="bold")
            if i > 0:
                try:
                    val = float(metrics_data[i - 1][1].replace("%", ""))
                    if i > 3:  # Ignora referência, threshold e população atingida
                        table[i, 1].set_facecolor(
                            "#d4edda" if val >= 0.5 else "#fff3cd"
                        )
                except ValueError:
                    pass

        ax_table.set_title(
            f"🎯 Performance no Top {top_q}/{bins} Quantis\n(Base de Teste)",
            fontweight="bold",
            fontsize=13,
        )

        # --- Matriz de Confusão ---
        ax_cm = axes[1]
        class_labels = [neg_label, str(pos_label)]
        row_totals = cm.sum(axis=1, keepdims=True)
        pct = (
            np.divide(
                cm,
                row_totals,
                out=np.zeros_like(cm, dtype=float),
                where=row_totals != 0,
            )
            * 100
        )

        annot = np.array(
            [[f"{cm[i,j]}\n({pct[i,j]:.1f}%)" for j in range(2)] for i in range(2)]
        )
        sns.heatmap(
            cm,
            annot=annot,
            fmt="",
            cmap="Blues",
            ax=ax_cm,
            cbar=False,
            annot_kws={"size": 12, "weight": "bold"},
            xticklabels=class_labels,
            yticklabels=class_labels,
        )

        ax_cm.set_title(
            f"Matriz de Confusão — Corte: >= {cut_threshold:.4f}\n(Referência: {ref_name})",
            fontweight="bold",
            fontsize=12,
        )
        ax_cm.set_xlabel("Predição do Modelo", fontsize=11, fontweight="bold")
        ax_cm.set_ylabel("Realidade", fontsize=11, fontweight="bold")

        plt.tight_layout()
        plt.show()

    # -----------------------------------------------------------------------
    # 7.9. DRIFT DE SCORE E PSI
    # -----------------------------------------------------------------------

    def calculate_psi(self, train_probs, test_probs, bins=10):
        """Calcula o Population Stability Index entre duas distribuições."""
        # Criar os buckets baseados nos decis do treino
        buckets = np.linspace(0, 1, bins + 1)

        train_percents = np.histogram(train_probs, bins=buckets)[0] / len(train_probs)
        test_percents = np.histogram(test_probs, bins=buckets)[0] / len(test_probs)

        # Evitar divisão por zero
        train_percents = np.where(train_percents == 0, 0.0001, train_percents)
        test_percents = np.where(test_percents == 0, 0.0001, test_percents)

        psi_values = (test_percents - train_percents) * np.log(
            test_percents / train_percents
        )
        return np.sum(psi_values)

    def check_threshold_volatility(self, test_data, top_q=1, bins=10):
        """
        Identifica a sensibilidade do threshold.
        Se uma mudança de 1% no threshold mudar 10% do volume, o modelo é instável.
        """
        X_trans, _ = self._preprocess_for_inference(test_data)
        y_prob = self.predictor.predict_proba(X_trans)[
            self._get_positive_class()
        ].values

        # Threshold alvo (ex: 10%)
        thr_main = np.quantile(y_prob, 1.0 - (top_q / bins))

        # Threshold com leve perturbação (±0.005)
        vol_plus = (y_prob >= (thr_main + 0.005)).mean()
        vol_minus = (y_prob >= (thr_main - 0.005)).mean()

        volatility_index = (vol_minus - vol_plus) * 100
        return thr_main, volatility_index

    def get_drift_report(self, test_data, bins=10):
        """
        Gera um relatório completo de PSI e Volatilidade de Threshold.
        Ideal para identificar compressão de scores como a do Bank Marketing.
        """
        if not self.cv_results:
            raise RuntimeError(
                "Execute cross_validate() antes para ter a referência do treino (OOF)."
            )

        pos_label = self._get_positive_class()

        # 1. Obter Probabilidades
        train_probs = self.cv_results["oof_probs"]
        X_trans, _ = self._preprocess_for_inference(test_data)
        test_probs = self.predictor.predict_proba(X_trans)[pos_label].values

        # 2. Cálculo do PSI (Population Stability Index)
        # Criamos os buckets baseados nos decis do TREINO
        buckets = np.quantile(train_probs, np.linspace(0, 1, bins + 1))
        # Evitar buckets duplicados em distribuições muito comprimidas
        buckets = np.unique(buckets)
        if len(buckets) < 2:
            buckets = np.linspace(0, 1, bins + 1)

        train_dist = np.histogram(train_probs, bins=buckets)[0] / len(train_probs)
        test_dist = np.histogram(test_probs, bins=buckets)[0] / len(test_probs)

        # Ajuste para evitar log(0)
        train_dist = np.where(train_dist == 0, 0.0001, train_dist)
        test_dist = np.where(test_dist == 0, 0.0001, test_dist)

        psi_values = (test_dist - train_dist) * np.log(test_dist / train_dist)
        total_psi = np.sum(psi_values)

        # 3. Cálculo de Elasticidade (Volatilidade) do Threshold
        # Se eu mudar o threshold em 0.01, quanto o volume de aprovados muda?
        thr_atual = np.median(test_probs)
        vol_base = (test_probs >= thr_atual).mean()
        vol_up = (test_probs >= (thr_atual + 0.01)).mean()
        vol_down = (test_probs >= (thr_atual - 0.01)).mean()

        # Sensibilidade: % de mudança na população por 0.01 de mudança no score
        elasticidade = (vol_down - vol_up) * 100

        # 4. Print do Diagnóstico
        print(f"\n{'='*40}")
        print(f"📊 RELATÓRIO DE ESTABILIDADE (DRIFT)")
        print(f"{'='*40}")

        status_psi = (
            "✅ Estável"
            if total_psi < 0.1
            else "⚠️ Alerta" if total_psi < 0.25 else "🚨 Drift Crítico"
        )
        print(f"Total PSI: {total_psi:.4f} ({status_psi})")
        print(f"Elasticidade do Threshold: {elasticidade:.2f}% por ±0.01 de score")

        if elasticidade > 15:
            print(
                "🔴 ALERTA: Scores muito comprimidos! Pequenas oscilações mudarão drasticamente o volume."
            )

        # 5. Visualização do Desvio
        plt.figure(figsize=(12, 5))
        sns.kdeplot(
            train_probs, label="Treino (OOF)", fill=True, color="blue", alpha=0.3
        )
        sns.kdeplot(
            test_probs, label="Teste (Atual)", fill=True, color="orange", alpha=0.3
        )
        plt.axvline(
            thr_atual,
            color="red",
            linestyle="--",
            label=f"Mediana Teste ({thr_atual:.3f})",
        )
        plt.title(
            f"Distribuição de Scores: Treino vs Teste (PSI: {total_psi:.4f})",
            fontweight="bold",
        )
        plt.xlabel("Probabilidade Prevista")
        plt.legend()
        plt.show()

        return {"psi": total_psi, "elasticity": elasticidade}

    # -----------------------------------------------------------------------
    # 8. SERIALIZAÇÃO
    # -----------------------------------------------------------------------

    def save_bundle(self, path: str = "modelo_prod", overwrite: bool = False):
        """
        Salva o bundle do modelo em disco.

        [NEW v0.1.0] Parâmetros adicionados:
          overwrite (bool) : se False (padrão), lança FileExistsError caso o diretório já exista.
                             Use overwrite=True para substituir um bundle existente.

        O bundle inclui versioning (framework, python, sklearn) para detectar
        incompatibilidades ao carregar em ambientes diferentes.
        """
        # [NEW v0.1.0] Overwrite protection
        if os.path.exists(path) and not overwrite:
            raise FileExistsError(
                f"Bundle já existe em '{path}/'. "
                f"Use save_bundle(path, overwrite=True) para sobrescrever."
            )

        os.makedirs(path, exist_ok=True)
        joblib.dump(
            {
                # [NEW v0.1.0] Versioning para detectar incompatibilidades
                "framework_version": _FRAMEWORK_VERSION,
                "python_version": sys.version,
                "sklearn_version": sklearn.__version__,
                # Artefatos do modelo
                "pipeline": self.pipeline,
                "params": self.params,
                "log_cols": self._log_cols,
                "outlier_bounds": self._outlier_bounds,
                "rare_cats": self._rare_categories,
                "agg_values": self._agg_values,
                "selected_features": self.selected_features,
                "eliminated_features": self.eliminated_features,
                "association_report": self.association_report,
                "feature_importance": self.feature_importance,
                "train_schema": self._train_schema,
                "train_hash": self._train_hash,
                "cv_results": self.cv_results,
                "profile_features": self._profile_features,  # [NEW v0.1.1]
            },
            f"{path}/assets.pkl",
        )
        self.predictor.save(f"{path}/autogluon")
        cv_tag = (
            f"CV salvo (AUC OOF: {self.cv_results['oof_auc']:.4f})"
            if self.cv_results
            else "CV não disponível"
        )
        print(f"📦 Bundle salvo em '{path}/'")
        print(f"   🔖 Versão          : v{_FRAMEWORK_VERSION}")
        print(f"   🔑 Hash do treino  : {self._train_hash[:16]}...")
        print(f"   📐 Schema salvo    : {len(self._train_schema)} features")
        print(f"   📊 {cv_tag}")

    def export_transform_pipeline(
        self, path: str = "transform_pipeline.pkl", overwrite: bool = False
    ) -> "TransformPipeline":
        """
        Exporta a pipeline de transformação como objeto standalone — sem dependência
        do AutoClassificationEngine nem do AutoGluon.

        O arquivo exportado pode ser carregado em qualquer projeto externo com:
            import joblib
            pipeline = joblib.load("transform_pipeline.pkl")
            df_transformado = pipeline.transform(df_bruto)

        Dependências no projeto externo: numpy, pandas, scikit-learn, joblib

        Parâmetros:
            path (str)       : caminho do arquivo .pkl a ser salvo.
            overwrite (bool) : se False (padrão), lança FileExistsError caso já exista.

        Retorna:
            TransformPipeline — o objeto exportado (já instanciado e pronto para uso).
        """
        if self.selected_features is None:
            raise RuntimeError(
                "Execute fit() antes de exportar o pipeline de transformação."
            )

        if os.path.exists(path) and not overwrite:
            raise FileExistsError(
                f"Arquivo '{path}' já existe. "
                f"Use export_transform_pipeline(path, overwrite=True) para sobrescrever."
            )

        dir_path = os.path.dirname(path)
        if dir_path:
            os.makedirs(dir_path, exist_ok=True)

        tp = TransformPipeline()
        tp._framework_version = _FRAMEWORK_VERSION
        tp.params = self.params
        tp.selected_features = self.selected_features
        tp._train_schema = self._train_schema
        tp._log_cols = self._log_cols
        tp._outlier_bounds = self._outlier_bounds
        tp._rare_categories = self._rare_categories
        tp._agg_values = self._agg_values
        tp._profile_features = getattr(self, "_profile_features", [])
        tp.sklearn_pipeline = self.pipeline

        joblib.dump(tp, path)

        print(f"✅ TransformPipeline exportado em '{path}'")
        print(f"   🔖 Versão framework : v{_FRAMEWORK_VERSION}")
        print(f"   📐 Features         : {len(tp.selected_features)}")
        print(f"   📦 Uso externo:")
        print(f"      import joblib")
        print(f"      pipeline = joblib.load('{path}')")
        print(f"      df_out   = pipeline.transform(df_bruto)")

        return tp

    @staticmethod
    def load(path: str) -> "AutoClassificationEngine":
        assets_path = f"{path}/assets.pkl"
        ag_path = f"{path}/autogluon"

        if not os.path.exists(assets_path):
            raise FileNotFoundError(f"Assets não encontrados em '{assets_path}'")
        if not os.path.exists(ag_path):
            raise FileNotFoundError(f"Modelo AutoGluon não encontrado em '{ag_path}'")

        assets = joblib.load(assets_path)

        # [NEW v0.1.0] Aviso de versão incompatível
        saved_version = assets.get("framework_version", "desconhecida")
        if saved_version != _FRAMEWORK_VERSION:
            warnings.warn(
                f"Versão do bundle ({saved_version}) difere da versão atual ({_FRAMEWORK_VERSION}). "
                f"Resultados podem ser inconsistentes.",
                UserWarning,
            )
        saved_sklearn = assets.get("sklearn_version", "desconhecida")
        if saved_sklearn != sklearn.__version__:
            warnings.warn(
                f"sklearn do bundle ({saved_sklearn}) difere do atual ({sklearn.__version__}). "
                f"Transformações do pipeline podem se comportar diferente.",
                UserWarning,
            )

        engine = AutoClassificationEngine.__new__(AutoClassificationEngine)
        engine.params = assets["params"]
        engine.target = assets["params"]["target"]
        engine.pipeline = assets["pipeline"]
        engine._log_cols = assets["log_cols"]
        engine._outlier_bounds = assets["outlier_bounds"]
        engine._rare_categories = assets["rare_cats"]
        engine._agg_values = assets.get("agg_values", {})
        engine.selected_features = assets["selected_features"]
        engine.eliminated_features = assets.get("eliminated_features", {})
        engine.association_report = assets.get("association_report", {})
        engine.feature_importance = assets.get("feature_importance", None)
        engine._train_schema = assets.get("train_schema", {})
        engine._train_hash = assets.get("train_hash", "")
        engine.cv_results = assets.get("cv_results", {})
        engine._profile_features = assets.get("profile_features", [])  # [NEW v0.1.1]
        engine.predictor = TabularPredictor.load(ag_path)

        print(f"✅ Engine carregado de '{path}/'")
        print(f"   🔖 Versão  : v{saved_version}")
        print(f"   Target   : {engine.target}")
        print(f"   Features : {engine.selected_features}")
        if engine._train_hash:
            print(f"   🔑 Hash  : {engine._train_hash[:16]}...")
        if engine._train_schema:
            print(f"   📐 Schema: {len(engine._train_schema)} features esperadas")
        if engine.cv_results:
            print(f"   📊 CV disponível (AUC OOF: {engine.cv_results['oof_auc']:.4f})")
        else:
            print("   📊 CV não disponível")
        return engine
