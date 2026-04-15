import pandas as pd
import numpy as np
import joblib
import sys
from pathlib import Path


# ─────────────────────────────────────────────────────────
# CHARGEMENT DES ARTEFACTS
# ─────────────────────────────────────────────────────────

def load_all_artifacts(models_dir: str = "../models") -> dict:
    """Charge l'ensemble des modèles et outils de transformation."""
    path = Path(models_dir)
    artifacts = {}
    try:
        # CHURN (modèle principal)
        artifacts["model_churn"] = joblib.load(path / "model.pkl")
        artifacts["features_churn"] = joblib.load(path / "feature_names.pkl")

        # CLUSTERING
        artifacts["model_km"] = joblib.load(path / "kmeans_model.pkl")
        artifacts["scaler_km"] = joblib.load(path / "scaler_kmeans.pkl")
        artifacts["pca_km"] = joblib.load(path / "pca_model.pkl")
        artifacts["features_km"] = joblib.load(path / "cluster_features.pkl")

        # RÉGRESSION
        artifacts["model_reg"] = joblib.load(path / "regression_model.pkl")
        artifacts["features_reg"] = joblib.load(path / "regression_features.pkl")

        print("[predict] Tous les artefacts ont été chargés avec succès.")
    except Exception as e:
        print(f"[predict] Erreur lors du chargement des artefacts : {e}")
        artifacts = {}
    return artifacts


# ─────────────────────────────────────────────────────────
# ALIGNEMENT DES FEATURES
# ─────────────────────────────────────────────────────────

def align_to_feature_list(df: pd.DataFrame, feature_names: list) -> pd.DataFrame:
    """
    Aligne le DataFrame sur une liste de features FIXE (celle vue au fit).
    - Colonnes manquantes → ajoutées avec 0
    - Colonnes en trop    → supprimées
    """
    df_aligned = df.copy()

    for col in feature_names:
        if col not in df_aligned.columns:
            df_aligned[col] = 0

    # On se restreint STRICTEMENT à l'ordre et à la liste fournie
    df_aligned = df_aligned[feature_names]

    return df_aligned


def align_for_kmeans(df: pd.DataFrame,
                     scaler,
                     feature_names: list) -> np.ndarray:
    """
    Aligne les features pour le pipeline de clustering :
      - filtre sur `feature_names`
      - ajoute les colonnes manquantes avec 0
      - applique le scaler k-means
    Retourne un array numpy prêt pour la PCA.
    """
    X = df.copy()

    # Ajout des colonnes manquantes
    for col in feature_names:
        if col not in X.columns:
            X[col] = 0

    X = X[feature_names]

    # Gestion éventuelle de NaN avant scaler (sécurité)
    if X.isnull().values.any():
        X = X.fillna(0)

    X_scaled = scaler.transform(X)
    return X_scaled


def align_for_regression(df: pd.DataFrame, feature_names: list) -> pd.DataFrame:
    """
    Aligne les features pour la régression :
      - filtre / ajoute les colonnes
      - pas de scaler global (le RandomForestRegressor est entraîné sur des features brutes).
    """
    X = df.copy()

    for col in feature_names:
        if col not in X.columns:
            X[col] = 0

    X = X[feature_names]

    # Sécurité type / NaN
    X = X.select_dtypes(include=[np.number])
    if X.isnull().values.any():
        X = X.fillna(0)

    return X


# ─────────────────────────────────────────────────────────
# PIPELINE D'INFÉRENCE COMPLET
# ─────────────────────────────────────────────────────────

def run_full_inference(df_input: pd.DataFrame, artifacts: dict) -> pd.DataFrame:
    """
    Exécute les 3 prédictions :
      - churn (classification)
      - cluster (K-means sur espace PCA)
      - valeur monétaire (régression)
    en alignant strictement les colonnes sur celles vues à l'entraînement.
    """
    results = pd.DataFrame(index=df_input.index)

    # ── 1. PRÉDICTION CHURN ───────────────────────────────
    # IMPORTANT : X_train/X_test utilisés pour entraîner ce modèle
    # sont DEJA pré-traités + normalisés par preprocessing.py.
    # Donc ici : pas de StandardScaler supplémentaire.
    features_churn = artifacts["features_churn"]
    model_churn = artifacts["model_churn"]

    X_churn = align_to_feature_list(df_input, features_churn)
    churn_proba = model_churn.predict_proba(X_churn)[:, 1]
    churn_pred = (churn_proba > 0.5).astype(int)

    results["churn_prob"] = churn_proba
    results["churn_pred"] = churn_pred

    # ── 2. PRÉDICTION CLUSTER ─────────────────────────────
    scaler_km = artifacts["scaler_km"]
    pca_km = artifacts["pca_km"]
    kmeans_model = artifacts["model_km"]
    features_km = artifacts["features_km"]

    X_km_scaled = align_for_kmeans(df_input, scaler_km, features_km)
    X_km_pca = pca_km.transform(X_km_scaled)
    cluster_id = kmeans_model.predict(X_km_pca)

    results["cluster_id"] = cluster_id

    # ── 3. PRÉDICTION VALEUR MONÉTAIRE ─────────────────────
    model_reg = artifacts["model_reg"]
    features_reg = artifacts["features_reg"]

    X_reg = align_for_regression(df_input, features_reg)
    predicted_monetary = model_reg.predict(X_reg)

    results["predicted_monetary_total"] = predicted_monetary

    return results


# ─────────────────────────────────────────────────────────
# POINT D'ENTRÉE CLI
# ─────────────────────────────────────────────────────────

def main(input_csv: str = None):
    models_dir = "../models"
    data_dir = "../data/train_test"

    artifacts = load_all_artifacts(models_dir)
    if not artifacts:
        print("[predict] Impossible de continuer sans artefacts.")
        return

    # Choix de la source de données
    path_to_load = Path(input_csv) if input_csv else Path(data_dir) / "X_test.csv"

    try:
        df = pd.read_csv(path_to_load)
        print(f"[predict] Analyse en cours sur {len(df)} clients...")
    except Exception as e:
        print(f"[predict] Erreur lors du chargement des données ({path_to_load}) : {e}")
        return

    try:
        final_results = run_full_inference(df, artifacts)
    except Exception as e:
        print(f"[predict] Erreur lors de l'exécution des prédictions : {e}")
        return

    # Affichage synthétique
    print("\n" + "=" * 40)
    print("   RÉSULTATS DE L'ANALYSE CLIENT")
    print("=" * 40)
    cols_to_show = [c for c in ["churn_prob", "churn_pred", "cluster_id", "predicted_monetary_total"]
                    if c in final_results.columns]
    print(final_results[cols_to_show].head())

    # Sauvegarde
    output_path = Path("predictions_finales.csv")
    final_results.to_csv(output_path, index=False)
    print(f"\n[predict] Succès ! Fichier '{output_path}' créé.")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)