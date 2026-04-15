import pandas as pd
import numpy as np
import joblib
import os
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error

def main():
    print("--- ENTRAÎNEMENT DE LA RÉGRESSION AVEC OPTIMISATION (GRIDSEARCH) ---")

    # 1. Chargement et préparation (Dynamique)
    input_path = '../data/train_test/X_train.csv'
    try:
        df = pd.read_csv(input_path, sep=None, engine='python')
    except Exception as e:
        print(f"Erreur : {e}")
        return

    # Cible : MonetaryTotal (on veut prédire la valeur du client)
    target = 'MonetaryTotal'
    X = df.drop(columns=[target, 'CustomerID', 'index', 'Churn'], errors='ignore')
    # On s'assure de ne garder que les numériques pour éviter les erreurs de type
    X = X.select_dtypes(include=[np.number])
    y = df[target]

    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 2. Définition de la grille d'hyperparamètres
    # On teste différentes profondeurs d'arbres et nombres d'estimateurs
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5],
        'bootstrap': [True]
    }

    print("Recherche des meilleurs paramètres en cours (Validation Croisée)...")
    rf = RandomForestRegressor(random_state=42)
    
    # GridSearchCV : teste toutes les combinaisons de param_grid
    grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, 
                               cv=3, n_jobs=-1, scoring='r2', verbose=1)
    
    grid_search.fit(X_train, y_train)

    # 3. Meilleur modèle
    best_model = grid_search.best_params_
    print(f"\nMeilleurs paramètres trouvés : {best_model}")

    # 4. Évaluation sur le jeu de test
    final_model = grid_search.best_estimator_
    y_pred = final_model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    print(f"\n--- PERFORMANCE FINALE ---")
    print(f"R² Score : {r2:.4f}")
    print(f"MAE (Erreur moyenne) : {mae:.2f}")
    print(f"RMSE (Écart-type des erreurs) : {rmse:.2f}")

    # 5. Sauvegarde
    model_dir = '../models/'
    os.makedirs(model_dir, exist_ok=True)
    joblib.dump(final_model, os.path.join(model_dir, 'regression_model.pkl'))
    # Sauvegarde des noms de colonnes (indispensable pour predict.py)
    joblib.dump(list(X.columns), os.path.join(model_dir, 'regression_features.pkl'))
    
    print("\nModèle et features sauvegardés avec succès.")

if __name__ == "__main__":
    main()