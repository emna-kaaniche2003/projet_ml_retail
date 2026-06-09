# Projet ML Retail — Prédiction du Churn Client

Atelier Machine Learning — Analyse Comportementale Clientèle E-commerce  
Module GI2 | Préparé par Fadoua Drira | Année 2025-2026

## Description

Chaîne complète de traitement ML pour prédire le départ (churn) de clients d'une plateforme e-commerce de cadeaux, à partir de 52 features comportementales et transactionnelles.

**Objectifs métier :**

- Personnaliser les stratégies marketing
- Réduire le taux de départ des clients (churn)
- Optimiser le chiffre d'affaires

---

## Structure du projet

```
projet_ml_retail/
├── data/
│   ├── raw/                  # Données brutes originales
│   ├── processed/            # Données nettoyées (après preprocessing)
│   └── train_test/           # Splits train/test (X_train, X_test, y_train, y_test)
├── notebooks/                # Notebooks Jupyter (exploration, prototypage)
├── src/
│   ├── preprocessing.py      # Pipeline complet de prétraitement
│   ├── train_model.py        # Entraînement et évaluation du modèle
│   ├── predict.py            # Inférence sur nouveaux clients
│   └── utils.py              # Fonctions utilitaires (EDA, corrélation, ACP)
├── models/                   # Artefacts sauvegardés (.pkl, .joblib)
├── app/
│   └── app.py                # Interface web Flask
├── reports/                  # Rapports et visualisations générés
├── requirements.txt
└── README.md
```

---

## Installation

### 1. Cloner le dépôt

```bash
git clone https://github.com/<votre-username>/projet_ml_retail.git
cd projet_ml_retail
```

### 2. Créer et activer l'environnement virtuel

```bash
# Créer
python -m venv venv

# Activer — Windows
venv\Scripts\activate

# Activer — Linux / macOS
source venv/bin/activate
```

### 3. Installer les dépendances

```bash
pip install -r requirements.txt
```

---

## Guide d'utilisation



https://github.com/user-attachments/assets/53581d60-e78e-4e77-b292-f59ba683308b



### Étape 1 — Prétraitement des données

Place le fichier brut dans `data/raw/` (CSV ou XLSX), puis :

```bash
cd src/
python preprocessing.py ../data/raw/data.csv
```

Génère dans `data/train_test/` : `X_train.csv`, `X_test.csv`, `y_train.csv`, `y_test.csv`  
Génère dans `models/` : `scaler.joblib`

### Étape 2 — Entraînement du modèle

```bash
python train_model.py
```

Génère dans `models/` : `model.pkl`, `feature_names.pkl`  
Affiche : rapport de classification, ROC-AUC, F1-score, validation croisée 5-fold, top 10 features.

### Étape 3 — Prédictions sur nouveaux clients

```bash
# Démonstration sur X_test
python predict.py

# Sur un fichier CSV externe
python predict.py chemin/vers/nouveaux_clients.csv
```

### Étape 4 — Lancer l'interface web Flask

```bash
cd app/
python app.py
```

Ouvrir http://127.0.0.1:5000 dans le navigateur.

**Routes disponibles :**

- `GET  /` → Formulaire de saisie client (interface HTML)
- `POST /predict` → Prédiction depuis le formulaire
- `POST /predict_api` → Endpoint JSON (intégration externe)
- `GET  /health` → Vérification de l'état du service

---

## Modèle

| Paramètre    | Valeur                   |
| ------------ | ------------------------ |
| Algorithme   | Random Forest Classifier |
| n_estimators | 200                      |
| max_depth    | 10                       |
| class_weight | balanced                 |
| Évaluation   | ROC-AUC, F1, CV 5-fold   |

---

## Auteur

Projet réalisé dans le cadre de l'atelier ML — GI2 | 2025-2026
