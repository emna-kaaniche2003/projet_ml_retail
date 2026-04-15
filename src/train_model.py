import pandas as pd
import joblib
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report, roc_auc_score, f1_score
from sklearn.model_selection import GridSearchCV



def load_data(data_dir: str = "../data/train_test"):
    """
    Charge les données d'entraînement et de test préalablement nettoyées.
    """
    print("--- CHARGEMENT DES DONNÉES ---")
    X_train = pd.read_csv(f"{data_dir}/X_train.csv")
    X_test  = pd.read_csv(f"{data_dir}/X_test.csv")
    y_train = pd.read_csv(f"{data_dir}/y_train.csv")
    y_test  = pd.read_csv(f"{data_dir}/y_test.csv")

    # Convertir en Series 1D (ravel évite le warning sklearn)
    return X_train, X_test, y_train.values.ravel(), y_test.values.ravel()


def train_and_evaluate():
    """
    Entraîne une baseline, optimise un Random Forest et un Gradient Boosting,
    compare les résultats, évalue le meilleur et le sauvegarde.
    """
    print("=" * 60)
    print("MODÉLISATION ET RECHERCHE D'HYPERPARAMÈTRES")
    print("=" * 60)

    # 1. Chargement des données
    X_train, X_test, y_train, y_test = load_data()

    # ---------------------------------------------------------
    # ÉTAPE 1 : MODÈLE DE BASE (BASELINE)
    # ---------------------------------------------------------
    print("\n--- 1. MODÈLE DE BASE : RÉGRESSION LOGISTIQUE ---")
    baseline_model = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)
    baseline_model.fit(X_train, y_train)
    
    y_proba_base = baseline_model.predict_proba(X_test)[:, 1]
    roc_auc_base = roc_auc_score(y_test, y_proba_base)
    print(f"Score ROC-AUC de la Baseline : {roc_auc_base:.4f}")

    # ---------------------------------------------------------
    # ÉTAPE 2 : OPTIMISATION DU RANDOM FOREST
    # ---------------------------------------------------------
    print("\n--- 2. OPTIMISATION DU RANDOM FOREST ---")
    rf = RandomForestClassifier(class_weight="balanced", random_state=42)
    
    param_grid_rf = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5, 10]
    }

    grid_search_rf = GridSearchCV(
        estimator=rf, param_grid=param_grid_rf, cv=5, scoring="roc_auc", n_jobs=-1
    )
    print("Lancement de l'entraînement des combinaisons (Random Forest)...")
    grid_search_rf.fit(X_train, y_train)
    
    best_rf_score = grid_search_rf.best_score_
    print(f"Meilleurs hyperparamètres RF : {grid_search_rf.best_params_}")
    print(f"Meilleur score ROC-AUC RF (Validation Croisée) : {best_rf_score:.4f}")

    # ---------------------------------------------------------
    # ÉTAPE 3 : OPTIMISATION DU GRADIENT BOOSTING
    # ---------------------------------------------------------
    print("\n--- 3. OPTIMISATION DU GRADIENT BOOSTING ---")
    gb = GradientBoostingClassifier(random_state=42)
    
    # Grille spécifique au Gradient Boosting
    param_grid_gb = {
        'n_estimators': [50, 100, 200],
        'learning_rate': [0.01, 0.1, 0.2], # Vitesse d'apprentissage des erreurs
        'max_depth': [3, 5, 7]             # Arbres généralement plus petits que pour le RF
    }

    grid_search_gb = GridSearchCV(
        estimator=gb, param_grid=param_grid_gb, cv=5, scoring="roc_auc", n_jobs=-1
    )
    print("Lancement de l'entraînement des combinaisons (Gradient Boosting)...")
    grid_search_gb.fit(X_train, y_train)
    
    best_gb_score = grid_search_gb.best_score_
    print(f"Meilleurs hyperparamètres GB : {grid_search_gb.best_params_}")
    print(f"Meilleur score ROC-AUC GB (Validation Croisée) : {best_gb_score:.4f}")

    # ---------------------------------------------------------
    # ÉTAPE 4 : SÉLECTION DU MEILLEUR MODÈLE GLOBAL
    # ---------------------------------------------------------
    print("\n--- 4. SÉLECTION DU CHAMPION ---")
    if best_gb_score > best_rf_score:
        print(f"🏆 Le Gradient Boosting gagne avec {best_gb_score:.4f} vs {best_rf_score:.4f}")
        best_model = grid_search_gb.best_estimator_
        model_name = "GradientBoosting"
    else:
        print(f"🏆 Le Random Forest gagne avec {best_rf_score:.4f} vs {best_gb_score:.4f}")
        best_model = grid_search_rf.best_estimator_
        model_name = "RandomForest"

    # ---------------------------------------------------------
    # ÉTAPE 5 : ÉVALUATION FINALE SUR LE JEU DE TEST
    # ---------------------------------------------------------
    print(f"\n--- 5. ÉVALUATION FINALE ({model_name}) SUR X_TEST ---")
    y_pred = best_model.predict(X_test)
    y_proba = best_model.predict_proba(X_test)[:, 1]

    print("\nRapport de Classification :")
    print(classification_report(y_test, y_pred, target_names=["Fidèle", "Churné"]))

    roc_auc = roc_auc_score(y_test, y_proba)
    f1 = f1_score(y_test, y_pred)
    print(f"Score ROC-AUC Final : {roc_auc:.4f}")
    print(f"Score F1 (Churn) Final : {f1:.4f}")

    # ---------------------------------------------------------
    # ÉTAPE 6 : IMPORTANCE DES VARIABLES
    # ---------------------------------------------------------
    print("\n--- 6. TOP 10 DES VARIABLES LES PLUS IMPORTANTES ---")
    importances = pd.Series(best_model.feature_importances_, index=X_train.columns)
    top10 = importances.sort_values(ascending=False).head(10)
    print(top10.to_string())

    # ---------------------------------------------------------
    # ÉTAPE 7 : SAUVEGARDE
    # ---------------------------------------------------------
    print("\n--- 7. SAUVEGARDE DU MODÈLE EN PRODUCTION ---")
    models_dir = Path("../models")
    models_dir.mkdir(parents=True, exist_ok=True)

    joblib.dump(best_model, models_dir / "model.pkl")
    joblib.dump(list(X_train.columns), models_dir / "feature_names.pkl")

    print(f"Modèle ({model_name}) sauvegardé : {models_dir / 'model.pkl'}")
    print(f"Features sauvegardées : {models_dir / 'feature_names.pkl'}")
    print("Terminé avec succès !")

if __name__ == "__main__":
    train_and_evaluate()