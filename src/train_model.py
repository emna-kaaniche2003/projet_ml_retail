import pandas as pd
import joblib
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score, f1_score
from sklearn.model_selection import cross_val_score

def load_data(data_dir: str = "../data/train_test"):
    """
    Charge les données d'entraînement et de test préalablement nettoyées
    et séparées par le script preprocessing.py.
    """
    print("Chargement des données...")
    X_train = pd.read_csv(f"{data_dir}/X_train.csv")
    X_test = pd.read_csv(f"{data_dir}/X_test.csv")
    y_train = pd.read_csv(f"{data_dir}/y_train.csv")
    y_test = pd.read_csv(f"{data_dir}/y_test.csv")
    
    # y_train et y_test sont chargés comme des DataFrames, on les convertit en Series (1D)
    return X_train, X_test, y_train.values.ravel(), y_test.values.ravel()

def train_and_evaluate():
    """
    Entraîne le modèle Random Forest robuste, évalue ses performances
    et le sauvegarde pour la mise en production.
    """
    print("=" * 60)
    print("ENTRAÎNEMENT DU MODÈLE DE PRÉDICTION DU CHURN")
    print("=" * 60)

    # 1. Chargement des données
    X_train, X_test, y_train, y_test = load_data()
    print(f"Dimensions - X_train: {X_train.shape}, X_test: {X_test.shape}\n")

    # 2. Initialisation du modèle (Configuration robuste sans triche)
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )

    # 3. Entraînement
    print("Entraînement du Random Forest en cours...")
    model.fit(X_train, y_train)

    # 4. Prédictions
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    # 5. Évaluation des performances
    print("\n--- RÉSULTATS DE L'ÉVALUATION (SUR LE SET DE TEST) ---")
    print(classification_report(y_test, y_pred, target_names=["Fidèle", "Churné"]))
    
    roc_auc = roc_auc_score(y_test, y_proba)
    f1 = f1_score(y_test, y_pred)
    print(f"Score ROC-AUC : {roc_auc:.4f}")
    print(f"Score F1 (Churn) : {f1:.4f}")

    # 6. Validation Croisée (Vérification de la stabilité)
    print("\n--- VALIDATION CROISÉE (5-Fold sur le Train Set) ---")
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring="roc_auc", n_jobs=-1)
    print(f"Scores ROC-AUC des 5 passes : {cv_scores}")
    print(f"Moyenne ROC-AUC : {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    # 7. Importance des variables
    print("\n--- TOP 5 DES VARIABLES LES PLUS IMPORTANTES ---")
    importances = pd.Series(model.feature_importances_, index=X_train.columns)
    print(importances.sort_values(ascending=False).head(5))

    # 8. Sauvegarde du modèle
    print("\n--- SAUVEGARDE ---")
    models_dir = Path("../models")
    models_dir.mkdir(parents=True, exist_ok=True)
    
    model_path = models_dir / "model.pkl"
    joblib.dump(model, model_path)
    print(f"✅ Modèle sain et prêt pour la production sauvegardé sous : {model_path}")
    print("=" * 60)

if __name__ == "__main__":
    train_and_evaluate()