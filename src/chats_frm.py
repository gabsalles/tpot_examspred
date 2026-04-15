# model_report.py
# Relatório completo de avaliação de modelo — standalone, sem dependência do AutoClassificationEngine.
# Compatível com qualquer modelo sklearn / AutoML Databricks que produza y_true, y_prob.
#
# USO BÁSICO:
#   from model_report import ModelReport
#
#   report = ModelReport(
#       y_true=y_test,                        # array-like binário (0/1)
#       y_prob=model.predict_proba(X_test)[:,1],  # probabilidade da classe positiva
#       pos_label=1,                          # opcional, default=1
#       threshold_strategy="youden",          # "youden" | "f1" | "default"
#       feature_importance=fi_df,             # pd.DataFrame com index=feature, coluna "importance" (opcional)
#       train_y_prob=oof_probs,               # probabilidades OOF do treino (opcional, para comparação)
#       train_y_true=oof_true,                # labels OOF do treino (opcional)
#       dataset_label="Teste",               # nome do conjunto principal
#       bins=10,                              # número de decis/quantis
#   )
#   report.plot()
#
# EXEMPLO RÁPIDO NO NOTEBOOK DO AUTOML (DATABRICKS):
#   import mlflow
#   model = mlflow.sklearn.load_model(f"runs:/{run_id}/model")
#   y_prob = model.predict_proba(X_test)[:, 1]
#
#   from model_report import ModelReport
#   report = ModelReport(y_true=y_test, y_prob=y_prob, pos_label=1)
#   report.plot()

from __future__ import annotations

import warnings
from math import ceil
from typing import Optional, Union

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

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
    brier_score_loss,
)
from sklearn.calibration import calibration_curve

# ---------------------------------------------------------------------------
# Cores canônicas (mesmas do framework)
# ---------------------------------------------------------------------------
_COLOR_TRAIN = "#F06000"  # laranja — treino / OOF
_COLOR_TEST = "#0055A8"  # azul   — teste


# ===========================================================================
# Classe principal
# ===========================================================================


class ModelReport:
    """
    Gera o relatório completo de avaliação, replicando os gráficos do
    AutoClassificationEngine para uso com modelos sklearn / AutoML Databricks.

    Parâmetros
    ----------
    y_true : array-like
        Labels reais do conjunto de teste (binário 0/1).
    y_prob : array-like
        Probabilidades preditas da classe positiva no conjunto de teste.
    pos_label : int | str
        Rótulo da classe positiva. Default=1.
    threshold_strategy : str
        Estratégia para threshold ótimo: "youden" (default), "f1" ou "default" (0.5).
    feature_importance : pd.DataFrame | None
        DataFrame com index=feature e coluna "importance". Opcional.
    train_y_prob : array-like | None
        Probabilidades OOF do treino (para curva ROC/PR comparativa). Opcional.
    train_y_true : array-like | None
        Labels reais OOF do treino. Obrigatório se train_y_prob for fornecido.
    dataset_label : str
        Nome do conjunto principal. Default="Teste".
    bins : int
        Número de quantis para tabela de decis. Default=10.
    """

    def __init__(
        self,
        y_true,
        y_prob,
        pos_label: Union[int, str] = 1,
        threshold_strategy: str = "youden",
        beta: float = 1.0,
        feature_importance: Optional[pd.DataFrame] = None,
        train_y_prob=None,
        train_y_true=None,
        dataset_label: str = "Teste",
        experiment_name: str = "experimento",
        bins: int = 10,
    ):
        self.y_true = np.asarray(y_true, dtype=int)
        self.y_prob = np.asarray(y_prob, dtype=float)
        self.pos_label = pos_label
        self.threshold_strategy = threshold_strategy
        self.beta = beta
        self.feature_importance = feature_importance
        self.bins = bins
        self.dataset_label = dataset_label
        self.experiment_name = experiment_name

        # Treino / OOF opcional
        self.train_y_true = (
            np.asarray(train_y_true, dtype=int) if train_y_true is not None else None
        )
        self.train_y_prob = (
            np.asarray(train_y_prob, dtype=float) if train_y_prob is not None else None
        )

        if self.train_y_prob is not None and self.train_y_true is None:
            raise ValueError(
                "train_y_true deve ser fornecido quando train_y_prob é passado."
            )

        # Cores por dataset
        self._colors = {
            "Treino": _COLOR_TRAIN,
            "Treino (OOF)": _COLOR_TRAIN,
            self.dataset_label: _COLOR_TEST,
        }

    # -----------------------------------------------------------------------
    # Threshold
    # -----------------------------------------------------------------------

    def _compute_threshold(self, fpr, tpr, thresholds, y_true=None, y_prob=None):
        if self.threshold_strategy == "youden":
            idx = np.argmax(tpr - fpr)
        elif (
            self.threshold_strategy in ("f1", "f_beta")
            and y_true is not None
            and y_prob is not None
        ):
            beta = self.beta
            prec, rec, thr_pr = precision_recall_curve(y_true, y_prob)
            fb = (1 + beta**2) * prec * rec / (beta**2 * prec + rec + 1e-9)
            best_idx = np.argmax(fb[:-1])
            best_thr = thr_pr[best_idx]
            idx = np.argmin(np.abs(thresholds - best_thr))
        else:
            idx = np.argmin(np.abs(thresholds - 0.5))
        return thresholds[idx], fpr[idx], tpr[idx]

    # -----------------------------------------------------------------------
    # Cálculo de estatísticas por decil
    # -----------------------------------------------------------------------

    def _get_decile_stats(self, y_true, y_prob, cut_edges=None):
        df = pd.DataFrame({"target": y_true, "prob": y_prob})

        if cut_edges is None:
            df["decile"] = pd.qcut(
                df["prob"].rank(method="first", ascending=True),
                self.bins,
                labels=range(1, self.bins + 1),
                retbins=False,
                duplicates="drop",
            )
            cut_edges = np.quantile(df["prob"], np.linspace(0, 1, self.bins + 1))
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

    # -----------------------------------------------------------------------
    # Montagem dos resultados por dataset
    # -----------------------------------------------------------------------

    def _build_result(self, y_true, y_prob, name):
        fpr, tpr, thresholds = roc_curve(y_true, y_prob)
        roc_auc = auc(fpr, tpr)
        prec_full, rec_full, thresh_pr = precision_recall_curve(y_true, y_prob)
        pr_auc = average_precision_score(y_true, y_prob)
        prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=self.bins)

        opt_thresh, opt_fpr, opt_tpr = self._compute_threshold(
            fpr, tpr, thresholds, y_true, y_prob
        )
        y_pred = (y_prob >= 0.5).astype(int)
        y_pred_opt = (y_prob >= opt_thresh).astype(int)
        brier = brier_score_loss(y_true, y_prob)

        # KS contínuo — registro a registro (fonte única de verdade, igual à curva KS)
        _df_ks = pd.DataFrame({"t": y_true, "p": y_prob}).sort_values(
            "p", ascending=False
        )
        _tp = _df_ks["t"].sum()
        _tn = len(_df_ks) - _tp
        ks_cont = float(
            (
                _df_ks["t"].cumsum() / max(_tp, 1)
                - (1 - _df_ks["t"]).cumsum() / max(_tn, 1)
            )
            .abs()
            .max()
        )

        return {
            "name": name,
            "y_true": y_true,
            "y_prob": y_prob,
            "y_pred": y_pred,
            "y_pred_youden": y_pred_opt,
            "fpr": fpr,
            "tpr": tpr,
            "thresholds": thresholds,
            "youden_thresh": opt_thresh,
            "youden_fpr": opt_fpr,
            "youden_tpr": opt_tpr,
            "ks_cont": ks_cont,
            "metrics": {
                "AUC-ROC": roc_auc,
                "Gini": 2 * roc_auc - 1,
                "KS": ks_cont,
                "F1 (thr=0.5)": f1_score(y_true, y_pred, zero_division=0),
                "F1 (opt)": f1_score(y_true, y_pred_opt, zero_division=0),
                "Recall (thr=0.5)": recall_score(y_true, y_pred, zero_division=0),
                "Log-Loss": log_loss(y_true, y_prob),
                "Brier Score": brier,
                "Accuracy": accuracy_score(y_true, y_pred),
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

    # -----------------------------------------------------------------------
    # Plots auxiliares
    # -----------------------------------------------------------------------

    def _plot_scorecard(self, ax, results):
        ax.axis("off")
        metric_df = pd.DataFrame({k: v["metrics"] for k, v in results.items()})

        def _color(metric, val):
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

    def _plot_confusion(self, ax, res, title, use_youden=False):
        y_pred = res["y_pred_youden"] if use_youden else res["y_pred"]
        cm = confusion_matrix(res["y_true"], y_pred)
        neg_label = (
            f"Não {self.pos_label}"
            if isinstance(self.pos_label, str)
            else f"≠{self.pos_label}"
        )
        class_labels = [neg_label, str(self.pos_label)]
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
        opt_thresh = res["youden_thresh"]

        ax.plot(thresholds, prec, label="Precision", color="#2ca02c", lw=2)
        ax.plot(thresholds, rec, label="Recall", color="#d62728", lw=2)
        ax.plot(thresholds, f1, label="F1", color="#1f77b4", lw=2)
        ax.axvline(
            opt_thresh,
            color="#9467bd",
            linestyle="-",
            lw=2,
            label=f"{self.threshold_strategy} ({opt_thresh:.2f})",
        )
        ax.annotate(
            f"{self.threshold_strategy}\n{opt_thresh:.2f}",
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
        ax.set_title(f"Análise de Threshold ({self.dataset_label})", fontweight="bold")
        ax.legend(fontsize=8)

    def _plot_decil(self, ax, decile_data, results, title_mode):
        all_stats = []
        for name, stats in decile_data.items():
            d = stats.copy()
            d["Dataset"] = name
            if not isinstance(d["decile"].dtype, pd.CategoricalDtype):
                d["decile"] = pd.Categorical(
                    d["decile"], categories=range(1, self.bins + 1), ordered=True
                )
            d = d.sort_values("decile")
            all_stats.append(d)

        if not all_stats:
            ax.set_visible(False)
            return

        plot_df = pd.concat(all_stats, ignore_index=True)
        hue_order = [h for h in self._colors.keys() if h in plot_df["Dataset"].unique()]

        sns.barplot(
            data=plot_df,
            x="decile",
            y="pct_positivos",
            hue="Dataset",
            hue_order=hue_order,
            ax=ax,
            palette={k: v for k, v in self._colors.items() if k in hue_order},
        )

        for ds in hue_order:
            if ds in results:
                mean_pct = results[ds]["y_true"].mean() * 100.0
                ax.axhline(
                    mean_pct,
                    color=self._colors[ds],
                    linestyle="--",
                    linewidth=1.5,
                    label=f"Média {ds} ({mean_pct:.1f}%)",
                )

        ylim_lo, ylim_hi = ax.get_ylim()
        ax.set_ylim(ylim_lo, ylim_hi * 1.45)
        y_range = ax.get_ylim()[1] - ax.get_ylim()[0]
        offset = y_range * 0.012

        for i, container in enumerate(ax.containers):
            if i >= len(hue_order):
                break
            ds = hue_order[i]
            d_sorted = (
                plot_df[plot_df["Dataset"] == ds]
                .sort_values("decile")
                .reset_index(drop=True)
            )
            for bar, (_, row) in zip(list(container), d_sorted.iterrows()):
                h = bar.get_height()
                if h <= 0 or row["count"] == 0:
                    continue
                x = bar.get_x() + bar.get_width() / 2
                label = f"{row['pct_positivos']:.1f}%\n{row['intervalo']}\nn={int(row['count'])}"
                ax.text(
                    x,
                    h + offset,
                    label,
                    ha="center",
                    va="bottom",
                    fontsize=6.5,
                    color="black",
                    clip_on=False,
                )

        ks_txt = ""
        if title_mode == "performance" and self.dataset_label in results:
            ks_stat = results[self.dataset_label]["ks_cont"]
            ks_txt = f"  |  KS ({self.dataset_label}): {ks_stat:.4f}"

        label_map = {5: "Quintil", 10: "Decil", 4: "Quartil", 20: "Vigésimo"}
        label_type = label_map.get(self.bins, f"{self.bins}-Faixas")

        if title_mode == "performance":
            title = f"Taxa de Eventos por {label_type} — Performance (bins independentes){ks_txt}"
            subtitle = "Cada dataset usa seus próprios percentis. Leitura: o modelo discrimina bem?"
        else:
            title = f"Taxa de Eventos por {label_type} — Estabilidade (bins fixos do treino/OOF)"
            subtitle = "Teste mapeado nos intervalos de score do treino. Leitura: o comportamento se manteve?"

        ax.set_title(f"{title}\n{subtitle}", fontweight="bold", fontsize=11)
        ax.set_xlabel(
            f"{label_type} (1=Menor Risco → {self.bins}=Maior Risco)", fontsize=10
        )
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
        ax.set_title(f"Curva KS ({self.dataset_label})", fontweight="bold")
        ax.set_xlabel("% População (Ordenado por Score)")
        ax.set_ylabel("% Cumulativo")
        ax.legend(fontsize=8)

    # -----------------------------------------------------------------------
    # Método principal
    # -----------------------------------------------------------------------

    def to_metrics_df(self) -> pd.DataFrame:
        """
        Retorna um DataFrame com todas as métricas do experimento em uma única linha.

        Colunas fixas de identificação:
            experiment_name, dataset_label, data, threshold_strategy,
            beta, bins, n_total, n_positivos, taxa_positivos, threshold_otimo

        Métricas calculadas (teste e treino/OOF quando disponível):
            auc_roc, gini, ks, avg_precision,
            f1_05, f1_1, f1_2,                  ← F-beta com β=0.5, 1.0 e 2.0
            f1_opt, f1_default,
            precision_opt, precision_default,
            recall_opt, recall_default,
            accuracy_opt, accuracy_default,
            log_loss, brier_score,
            mcc                                  ← Matthews Correlation Coefficient

        Se treino/OOF fornecido, repete as mesmas colunas com prefixo "train_".

        Uso:
            row = report.to_metrics_df()
            historico = pd.concat([historico, row], ignore_index=True)
        """
        from datetime import datetime
        from sklearn.metrics import matthews_corrcoef

        def _compute_all(y_true, y_prob, label):
            fpr, tpr, thresholds = roc_curve(y_true, y_prob)
            roc_auc = auc(fpr, tpr)
            prec_full, rec_full, thresh_pr = precision_recall_curve(y_true, y_prob)
            ap = average_precision_score(y_true, y_prob)
            brier = brier_score_loss(y_true, y_prob)
            ll = log_loss(y_true, y_prob)

            # KS contínuo (registro a registro)
            df_ks = pd.DataFrame({"t": y_true, "p": y_prob}).sort_values(
                "p", ascending=False
            )
            tp = df_ks["t"].sum()
            tn = len(df_ks) - tp
            ks = (
                (
                    df_ks["t"].cumsum() / max(tp, 1)
                    - (1 - df_ks["t"]).cumsum() / max(tn, 1)
                )
                .abs()
                .max()
            )

            # Threshold ótimo via estratégia configurada
            _orig_strategy = self.threshold_strategy
            opt_thr, _, _ = self._compute_threshold(
                fpr, tpr, thresholds, y_true, y_prob
            )

            y_pred_opt = (y_prob >= opt_thr).astype(int)
            y_pred_def = (y_prob >= 0.5).astype(int)

            # F-beta fixos (independente da estratégia)
            def _fbeta(beta, y_pred):
                p = precision_score(y_true, y_pred, zero_division=0)
                r = recall_score(y_true, y_pred, zero_division=0)
                return (1 + beta**2) * p * r / (beta**2 * p + r + 1e-9)

            # F-beta no threshold ótimo via pr curve para β=0.5, 1, 2
            def _fbeta_thr(beta):
                p_, r_, t_ = precision_recall_curve(y_true, y_prob)
                fb = (1 + beta**2) * p_ * r_ / (beta**2 * p_ + r_ + 1e-9)
                best = t_[np.argmax(fb[:-1])]
                yp = (y_prob >= best).astype(int)
                return _fbeta(beta, yp)

            prefix = f"{label}_" if label else ""
            return {
                f"{prefix}auc_roc": round(roc_auc, 6),
                f"{prefix}gini": round(2 * roc_auc - 1, 6),
                f"{prefix}ks": round(float(ks), 6),
                f"{prefix}avg_precision": round(ap, 6),
                f"{prefix}log_loss": round(ll, 6),
                f"{prefix}brier_score": round(brier, 6),
                f"{prefix}f1_05": round(_fbeta_thr(0.5), 6),
                f"{prefix}f1_1": round(_fbeta_thr(1.0), 6),
                f"{prefix}f1_2": round(_fbeta_thr(2.0), 6),
                f"{prefix}f1_opt": round(
                    f1_score(y_true, y_pred_opt, zero_division=0), 6
                ),
                f"{prefix}f1_default": round(
                    f1_score(y_true, y_pred_def, zero_division=0), 6
                ),
                f"{prefix}precision_opt": round(
                    precision_score(y_true, y_pred_opt, zero_division=0), 6
                ),
                f"{prefix}precision_default": round(
                    precision_score(y_true, y_pred_def, zero_division=0), 6
                ),
                f"{prefix}recall_opt": round(
                    recall_score(y_true, y_pred_opt, zero_division=0), 6
                ),
                f"{prefix}recall_default": round(
                    recall_score(y_true, y_pred_def, zero_division=0), 6
                ),
                f"{prefix}accuracy_opt": round(accuracy_score(y_true, y_pred_opt), 6),
                f"{prefix}accuracy_default": round(
                    accuracy_score(y_true, y_pred_def), 6
                ),
                f"{prefix}mcc": round(matthews_corrcoef(y_true, y_pred_opt), 6),
                f"{prefix}threshold_otimo": round(float(opt_thr), 6),
            }

        # Identificação
        row = {
            "experiment_name": self.experiment_name,
            "dataset_label": self.dataset_label,
            "data": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "threshold_strategy": self.threshold_strategy,
            "beta": self.beta,
            "bins": self.bins,
            "n_total": len(self.y_true),
            "n_positivos": int(self.y_true.sum()),
            "taxa_positivos": round(self.y_true.mean(), 6),
        }

        # Métricas do conjunto principal (sem prefixo)
        row.update(_compute_all(self.y_true, self.y_prob, label=""))

        # Métricas do treino/OOF (prefixo "train_")
        if self.train_y_prob is not None:
            row.update(
                _compute_all(self.train_y_true, self.train_y_prob, label="train")
            )

        return pd.DataFrame([row])

    def plot(self):
        """Gera o relatório completo com todos os gráficos."""

        # --- Monta resultados ---
        results = {}
        results[self.dataset_label] = self._build_result(
            self.y_true, self.y_prob, self.dataset_label
        )

        train_label = None
        if self.train_y_prob is not None:
            train_label = "Treino (OOF)"
            results[train_label] = self._build_result(
                self.train_y_true, self.train_y_prob, train_label
            )

        # --- Estatísticas por decil ---
        decile_indep = {}
        train_cut_edges = None

        ref_name = train_label if train_label else self.dataset_label

        for name in ([train_label] if train_label else []) + [self.dataset_label]:
            stats, edges = self._get_decile_stats(
                results[name]["y_true"], results[name]["y_prob"]
            )
            decile_indep[name] = stats
            results[name]["decile_stats"] = stats
            if name == ref_name:
                train_cut_edges = edges

        decile_fixed = {}
        if train_cut_edges is not None and train_label:
            for name in [train_label, self.dataset_label]:
                stats_fixed, _ = self._get_decile_stats(
                    results[name]["y_true"],
                    results[name]["y_prob"],
                    cut_edges=train_cut_edges,
                )
                decile_fixed[name] = stats_fixed
        else:
            decile_fixed[self.dataset_label] = decile_indep[self.dataset_label]

        # --- Layout ---
        has_train = train_label is not None
        has_stability = has_train  # decil estabilidade só faz sentido com treino

        fig = plt.figure(figsize=(24, 26), constrained_layout=True)
        gs = fig.add_gridspec(5, 4)

        # Linha 0 — scorecard + matrizes de confusão
        ax_metrics = fig.add_subplot(gs[0, :2])
        ax_cm_default = fig.add_subplot(gs[0, 2])
        ax_cm_opt = fig.add_subplot(gs[0, 3])

        self._plot_scorecard(ax_metrics, results)
        self._plot_confusion(
            ax_cm_default, results[self.dataset_label], "Threshold 0.5"
        )
        self._plot_confusion(
            ax_cm_opt,
            results[self.dataset_label],
            f"{self.threshold_strategy} ({results[self.dataset_label]['youden_thresh']:.2f})",
            use_youden=True,
        )

        # Linha 1 — ROC, PR, calibração, densidade
        ax_roc = fig.add_subplot(gs[1, 0])
        ax_pr = fig.add_subplot(gs[1, 1])
        ax_calib = fig.add_subplot(gs[1, 2])
        ax_hist = fig.add_subplot(gs[1, 3])

        for name, res in results.items():
            ls = "-" if name == self.dataset_label else "--"
            color = self._colors.get(name, "#2ca02c")
            ax_roc.plot(
                res["fpr"],
                res["tpr"],
                ls,
                lw=2,
                color=color,
                label=f"{name} AUC={res['metrics']['AUC-ROC']:.3f}",
            )
            if name == self.dataset_label:
                ax_roc.scatter(
                    [res["youden_fpr"]],
                    [res["youden_tpr"]],
                    color="red",
                    zorder=5,
                    s=80,
                    label=self.threshold_strategy.title(),
                )
        ax_roc.plot([0, 1], [0, 1], "k--", alpha=0.3)
        ax_roc.set_title("Curva ROC", fontweight="bold")
        ax_roc.set_xlabel("FPR")
        ax_roc.set_ylabel("TPR")
        ax_roc.legend(fontsize=8)

        for name, res in results.items():
            ls = "-" if name == self.dataset_label else "--"
            color = self._colors.get(name, "#2ca02c")
            ax_pr.plot(
                res["rec_full"],
                res["prec_full"],
                ls,
                lw=2,
                color=color,
                label=f"{name} AP={res['metrics']['Avg Precision']:.3f}",
            )
        if has_train:
            ap_gap = (
                results[train_label]["metrics"]["Avg Precision"]
                - results[self.dataset_label]["metrics"]["Avg Precision"]
            )
            if ap_gap > 0.10:
                ax_pr.text(
                    0.5,
                    0.12,
                    f"⚠️ Possível Overfitting\nΔAP OOF−Teste = {ap_gap:.2f}",
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

        for name, res in results.items():
            ls = "-" if name == self.dataset_label else "--"
            color = self._colors.get(name, "#2ca02c")
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

        for name, res in results.items():
            color = self._colors.get(name, "#2ca02c")
            sns.kdeplot(
                res["y_prob"], label=name, ax=ax_hist, fill=True, alpha=0.3, color=color
            )
        ax_hist.axvline(
            results[self.dataset_label]["youden_thresh"],
            color="red",
            linestyle="--",
            label=f"{self.threshold_strategy} ({results[self.dataset_label]['youden_thresh']:.2f})",
        )
        ax_hist.set_title("Densidade de Probabilidade", fontweight="bold")
        ax_hist.set_xlim(0, 1)
        ax_hist.legend(fontsize=8)

        # Linha 2 — threshold analysis + feature importance
        ax_thresh = fig.add_subplot(gs[2, :2])
        ax_feat = fig.add_subplot(gs[2, 2:])

        self._plot_threshold_analysis(ax_thresh, results[self.dataset_label])

        if self.feature_importance is not None:
            fi = self.feature_importance
            if not isinstance(fi.index, pd.RangeIndex):
                top_feat = fi.head(15)
                x_col = "importance"
                y_vals = top_feat.index
            else:
                # assume colunas "feature" e "importance"
                top_feat = (
                    fi.nlargest(15, "importance")
                    if "importance" in fi.columns
                    else fi.head(15)
                )
                x_col = "importance" if "importance" in fi.columns else fi.columns[1]
                y_vals = top_feat[fi.columns[0]]
            sns.barplot(x=top_feat[x_col], y=y_vals, ax=ax_feat, palette="viridis")
            ax_feat.set_title("Top 15 Features (Importância)", fontweight="bold")
        else:
            ax_feat.text(
                0.5,
                0.5,
                "Feature importance não fornecida.\nPasse feature_importance=df ao criar o ModelReport.",
                ha="center",
                va="center",
                fontsize=10,
                transform=ax_feat.transAxes,
            )
            ax_feat.axis("off")

        # Linha 3 — decil performance
        ax_decil_perf = fig.add_subplot(gs[3, :])
        self._plot_decil(ax_decil_perf, decile_indep, results, title_mode="performance")

        # Linha 4 — decil estabilidade + KS
        ax_decil_stab = fig.add_subplot(gs[4, :3])
        ax_ks = fig.add_subplot(gs[4, 3])

        self._plot_decil(
            ax_decil_stab, decile_fixed, results, title_mode="estabilidade"
        )
        self._plot_ks_curve(ax_ks, results[self.dataset_label])

        oof_tag = " | Treino = OOF" if has_train else ""
        plt.suptitle(
            f"Relatório Completo de Performance — target: {self.pos_label}"
            f" | Threshold: {self.threshold_strategy}{oof_tag}",
            fontsize=17,
            weight="bold",
        )
        plt.show()

        return results
