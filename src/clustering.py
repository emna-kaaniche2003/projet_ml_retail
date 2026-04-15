import pandas as pd
import numpy as np
import os
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

def main():
    print("--- DÉMARRAGE DU PIPELINE DE CLUSTERING (AVEC ACP) ---")

    input_path = '../data/train_test/X_train.csv'
    model_dir = '../models/'
    os.makedirs(model_dir, exist_ok=True)

    # 2. Chargement des données avec détection automatique du séparateur
    print(f"Chargement des données depuis : {input_path}")
    try:
        # sep=None avec engine='python' permet de détecter si c'est ',' ou ';' automatiquement
        df = pd.read_csv(input_path, sep=None, engine='python') 
    except Exception as e:
        print(f"ERREUR CRITIQUE : {e}")
        return

    # ==============================================================================
    # 3. SÉLECTION DES VARIABLES ET PURGE DU BIAIS DE REDONDANCE
    # ==============================================================================
    # On récupère les colonnes numériques
    X_cluster = df.select_dtypes(include=[np.number]).copy()
    
    # a. Exclusion des variables techniques et de la cible (Supervisé)
    tech_cols = ['CustomerID', 'Churn', 'index', 'Unnamed: 0']
    
    # b. Exclusion stricte des segmentations métiers préexistantes (Biais de redondance)
    # Le One-Hot Encoding a créé des colonnes du type "RFMSegment_VIP", on les repère via startswith
    business_labels = [col for col in X_cluster.columns if col.startswith(('RFMSegment_', 'CustomerType_', 'ChurnRisk_'))]
    
    # c. Application de la purge
    cols_to_drop = [c for c in tech_cols + business_labels if c in X_cluster.columns]
    X_cluster = X_cluster.drop(columns=cols_to_drop)

    # Vérification
    if X_cluster.empty:
        print("ERREUR : Aucune colonne numérique trouvée après le filtrage.")
        return

    print(f"Nombre de variables initiales soumises au clustering : {X_cluster.shape[1]}")
    print(f"Purge de {len(business_labels)} colonnes de type étiquettes métiers.")

    # ==============================================================================
    # 4. STANDARDISATION (Ajustement pour l'ACP)
    # ==============================================================================
    # Note : Même si les variables continues ont été scalées par preprocessing.py, 
    # les variables binaires (One-Hot) ne l'étaient pas. L'ACP exige que TOUTES 
    # les variables aient la même variance. On réapplique donc un scaler global ici.
    scaler_km = StandardScaler()
    X_scaled = scaler_km.fit_transform(X_cluster)

    # 5. Application de l'ACP
    print("Application de l'ACP (Cible : 95% de variance conservée)...")
    pca_model = PCA(n_components=0.95, random_state=42)
    X_pca = pca_model.fit_transform(X_scaled)
    
    print(f"-> Réduction de dimension : de {X_cluster.shape[1]} à {X_pca.shape[1]} composantes.")

    # 6. Entraînement du K-Means
    K_CHOISI = 4
    print(f"Entraînement du K-Means avec K={K_CHOISI}...")
    kmeans_final = KMeans(n_clusters=K_CHOISI, random_state=42, n_init=10)
    df['Cluster'] = kmeans_final.fit_predict(X_pca)

    # 7. Sauvegarde des artefacts
    joblib.dump(scaler_km, os.path.join(model_dir, 'scaler_kmeans.pkl'))
    joblib.dump(pca_model, os.path.join(model_dir, 'pca_model.pkl'))
    joblib.dump(kmeans_final, os.path.join(model_dir, 'kmeans_model.pkl'))
    # Sauvegarde de la liste des colonnes pour le futur predict.py
    joblib.dump(list(X_cluster.columns), os.path.join(model_dir, 'cluster_features.pkl'))

    # 8. Profilage métier dynamique
    print("\nGénération des profils métier...")
    # On prend les moyennes sur toutes les colonnes numériques utilisées
    cluster_profiles = df.groupby('Cluster')[X_cluster.columns].mean()
    cluster_profiles.to_csv(os.path.join(model_dir, 'cluster_profiles.csv'))

    print("\n--- RÉSULTATS ---")
    print(cluster_profiles.round(2))
    print("\nRépartition par cluster :")
    print(df['Cluster'].value_counts())
    print("--- TERMINÉ ---")

if __name__ == "__main__":
    main()