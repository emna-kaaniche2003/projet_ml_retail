"""
utils.py — Fonctions utilitaires partagées entre les scripts du projet.
Visualisation, analyse exploratoire, corrélation, ACP.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


# ─────────────────────────────────────────────────────────
# ANALYSE EXPLORATOIRE
# ─────────────────────────────────────────────────────────

def display_missing_values(df: pd.DataFrame, threshold: float = 0.0) -> pd.DataFrame:
    """
    Affiche le taux de valeurs manquantes par colonne.
    threshold : n'affiche que les colonnes avec un taux > threshold (0.0 = toutes).
    """
    missing = df.isnull().mean().sort_values(ascending=False)
    missing = missing[missing > threshold]

    print(f"\n[utils] Taux de valeurs manquantes (seuil > {threshold*100:.0f}%) :")
    print(missing.apply(lambda x: f"{x*100:.1f}%").to_string())
    return missing


def display_class_distribution(y: pd.Series, label: str = "Churn"):
    """Affiche la distribution de la variable cible."""
    counts = y.value_counts()
    ratios = y.value_counts(normalize=True)
    print(f"\n[utils] Distribution de '{label}' :")
    for val in counts.index:
        print(f"  {val} : {counts[val]} ({ratios[val]*100:.1f}%)")


def display_basic_stats(df: pd.DataFrame):
    """Affiche les statistiques descriptives des colonnes numériques."""
    print("\n[utils] Statistiques descriptives :")
    print(df.describe().T.to_string())


# ─────────────────────────────────────────────────────────
# ANALYSE DE CORRÉLATION
# ─────────────────────────────────────────────────────────

def plot_correlation_heatmap(df: pd.DataFrame,
                              target: str = "Churn",
                              figsize: tuple = (16, 12),
                              save_path: str = None):
    """
    Trace la heatmap de corrélation des features numériques.
    Met en évidence les corrélations avec la target.
    """
    num_df = df.select_dtypes(include=[np.number])
    corr = num_df.corr()

    plt.figure(figsize=figsize)
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(
        corr,
        mask=mask,
        annot=False,
        fmt=".2f",
        cmap="RdBu_r",
        center=0,
        linewidths=0.3,
        cbar_kws={"shrink": 0.8}
    )
    plt.title("Matrice de corrélation des features numériques", fontsize=14)
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150)
        print(f"[utils] Heatmap sauvegardée → {save_path}")
    plt.show()
    return corr


def find_high_correlations(df: pd.DataFrame,
                            threshold: float = 0.8,
                            target: str = "Churn") -> pd.DataFrame:
    """
    Identifie les paires de features fortement corrélées (multicolinéarité).
    Retourne un DataFrame des paires avec |corrélation| > threshold.
    """
    num_df = df.select_dtypes(include=[np.number]).drop(
        columns=[target], errors="ignore"
    )
    corr = num_df.corr().abs()

    # Masque triangulaire supérieur
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))

    high_corr = (
        upper.stack()
        .reset_index()
        .rename(columns={"level_0": "Feature_1", "level_1": "Feature_2", 0: "Correlation"})
    )
    high_corr = high_corr[high_corr["Correlation"] > threshold].sort_values(
        "Correlation", ascending=False
    )

    print(f"\n[utils] Paires avec |corrélation| > {threshold} :")
    if high_corr.empty:
        print("  Aucune multicolinéarité détectée.")
    else:
        print(high_corr.to_string(index=False))

    return high_corr


def plot_churn_correlations(df: pd.DataFrame,
                             target: str = "Churn",
                             top_n: int = 15,
                             save_path: str = None):
    """
    Barplot des top N features les plus corrélées avec la target.
    """
    num_df = df.select_dtypes(include=[np.number])
    if target not in num_df.columns:
        print("[utils] Target non numérique ou absente, corrélation ignorée.")
        return

    corr_target = num_df.corr()[target].drop(target).sort_values(key=abs, ascending=False)
    top = corr_target.head(top_n)

    colors = ["#e74c3c" if v > 0 else "#3498db" for v in top.values]
    plt.figure(figsize=(10, 6))
    top.sort_values().plot(kind="barh", color=colors)
    plt.axvline(0, color="black", linewidth=0.8)
    plt.title(f"Top {top_n} features corrélées avec '{target}'", fontsize=13)
    plt.xlabel("Corrélation de Pearson")
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150)
        print(f"[utils] Graphe corrélation-churn sauvegardé → {save_path}")
    plt.show()


# ─────────────────────────────────────────────────────────
# ANALYSE EN COMPOSANTES PRINCIPALES (ACP)
# ─────────────────────────────────────────────────────────

def run_pca_analysis(X: pd.DataFrame,
                     y: pd.Series = None,
                     n_components: int = None,
                     variance_threshold: float = 0.95,
                     save_path: str = None) -> PCA:
    """
    Effectue une ACP sur X (déjà normalisé).
    - Affiche la variance expliquée cumulée
    - Détermine automatiquement le nombre de composantes
      pour atteindre variance_threshold (ex: 0.95 = 95%)
    - Trace la visualisation 2D si y est fourni

    Retourne l'objet PCA fitté.
    """
    # Normalisation interne (si X n'est pas encore normalisé)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # ACP complète pour analyse de variance
    pca_full = PCA(random_state=42)
    pca_full.fit(X_scaled)

    cumvar = np.cumsum(pca_full.explained_variance_ratio_)
    n_opt  = np.argmax(cumvar >= variance_threshold) + 1

    print(f"\n[utils] ACP — {X.shape[1]} features initiales")
    print(f"[utils] Composantes pour {variance_threshold*100:.0f}% de variance : {n_opt}")

    # Plot variance expliquée
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.bar(range(1, min(31, len(pca_full.explained_variance_ratio_)+1)),
            pca_full.explained_variance_ratio_[:30] * 100,
            color="#3498db", alpha=0.8)
    plt.xlabel("Composante principale")
    plt.ylabel("Variance expliquée (%)")
    plt.title("Variance par composante (Top 30)")

    plt.subplot(1, 2, 2)
    plt.plot(range(1, len(cumvar)+1), cumvar * 100, color="#e74c3c", linewidth=2)
    plt.axhline(variance_threshold * 100, linestyle="--", color="gray",
                label=f"{variance_threshold*100:.0f}%")
    plt.axvline(n_opt, linestyle="--", color="#e74c3c",
                label=f"n={n_opt} composantes")
    plt.xlabel("Nombre de composantes")
    plt.ylabel("Variance cumulée (%)")
    plt.title("Variance cumulée expliquée")
    plt.legend()
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path.replace(".png", "_variance.png"), dpi=150)
    plt.show()

    # Visualisation 2D si y fourni
    if y is not None:
        pca_2d = PCA(n_components=2, random_state=42)
        X_2d   = pca_2d.fit_transform(X_scaled)

        plt.figure(figsize=(9, 6))
        colors = {0: "#2ecc71", 1: "#e74c3c"}
        labels = {0: "Fidèle", 1: "Churné"}
        for cls in [0, 1]:
            mask = y.values == cls
            plt.scatter(X_2d[mask, 0], X_2d[mask, 1],
                        c=colors[cls], label=labels[cls],
                        alpha=0.4, s=15)
        plt.xlabel(f"PC1 ({pca_2d.explained_variance_ratio_[0]*100:.1f}%)")
        plt.ylabel(f"PC2 ({pca_2d.explained_variance_ratio_[1]*100:.1f}%)")
        plt.title("Projection ACP 2D — Churn vs Fidèle")
        plt.legend()
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path.replace(".png", "_2d.png"), dpi=150)
        plt.show()

    # ACP finale avec le bon nombre de composantes
    n_final = n_components if n_components else n_opt
    pca     = PCA(n_components=n_final, random_state=42)
    pca.fit(X_scaled)
    return pca


# ─────────────────────────────────────────────────────────
# VISUALISATION IMPORTANCE DES FEATURES (post-entraînement)
# ─────────────────────────────────────────────────────────

def plot_feature_importances(model,
                              feature_names: list,
                              top_n: int = 20,
                              save_path: str = None):
    """
    Barplot horizontal des top N features les plus importantes du modèle.
    Compatible avec tout estimateur sklearn ayant feature_importances_.
    """
    importances = pd.Series(model.feature_importances_, index=feature_names)
    top = importances.sort_values(ascending=True).tail(top_n)

    plt.figure(figsize=(10, 7))
    top.plot(kind="barh", color="#3498db", alpha=0.85)
    plt.title(f"Top {top_n} features les plus importantes", fontsize=13)
    plt.xlabel("Importance (Gini)")
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150)
        print(f"[utils] Graphe importances sauvegardé → {save_path}")
    plt.show()


# ─────────────────────────────────────────────────────────
# DISTRIBUTION DES PRÉDICTIONS
# ─────────────────────────────────────────────────────────

def plot_churn_probability_distribution(y_proba: np.ndarray,
                                         y_true: np.ndarray = None,
                                         save_path: str = None):
    """
    Histogramme de la distribution des probabilités de churn prédites.
    Si y_true fourni, superpose les distributions par classe réelle.
    """
    plt.figure(figsize=(9, 5))

    if y_true is not None:
        plt.hist(y_proba[y_true == 0], bins=40, alpha=0.6,
                 color="#2ecc71", label="Fidèle (réel)")
        plt.hist(y_proba[y_true == 1], bins=40, alpha=0.6,
                 color="#e74c3c", label="Churné (réel)")
        plt.legend()
    else:
        plt.hist(y_proba, bins=40, color="#3498db", alpha=0.8)

    plt.axvline(0.5, linestyle="--", color="black", label="Seuil 0.5")
    plt.xlabel("Probabilité de churn prédite")
    plt.ylabel("Nombre de clients")
    plt.title("Distribution des probabilités de churn")
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150)
    plt.show()