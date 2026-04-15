"""
app.py — Interface web Flask pour l'analyse comportementale clientèle retail.
Intègre : prédiction churn, clustering K-Means, régression valeur monétaire.

CORRECTION : Les données du formulaire sont maintenant transformées via le même
pipeline que preprocessing.py (feature engineering + encodage ordinal +
one-hot + target encoding + StandardScaler) avant d'être passées aux modèles.

Démarrage :
    cd app/   (ou depuis la racine)
    python app.py

Routes :
    GET  /            → Dashboard principal (formulaire + résultats)
    POST /predict     → Analyse complète d'un client
    GET  /api/health  → Vérification de l'état du service
    POST /api/predict → Endpoint JSON (intégration externe)
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../src"))

import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from flask import Flask, request, render_template_string, jsonify

app = Flask(__name__)

# ─── Répertoire des modèles ──────────────────────────────
MODELS_DIR = Path(__file__).parent.parent / "models"

# ─── Chargement de tous les artefacts au démarrage ──────
def load_artifacts():
    arts = {}
    try:
        arts["model_churn"]    = joblib.load(MODELS_DIR / "model.pkl")
        arts["features_churn"] = joblib.load(MODELS_DIR / "feature_names.pkl")
        print(f"✅ Modèle churn chargé — {len(arts['features_churn'])} features")
    except Exception as e:
        print(f"⚠️  Churn : {e}")

    try:
        arts["model_km"]    = joblib.load(MODELS_DIR / "kmeans_model.pkl")
        arts["scaler_km"]   = joblib.load(MODELS_DIR / "scaler_kmeans.pkl")
        arts["pca_km"]      = joblib.load(MODELS_DIR / "pca_model.pkl")
        arts["features_km"] = joblib.load(MODELS_DIR / "cluster_features.pkl")
        print("✅ Modèle clustering chargé")
    except Exception as e:
        print(f"⚠️  Clustering : {e}")

    try:
        arts["model_reg"]    = joblib.load(MODELS_DIR / "regression_model.pkl")
        arts["features_reg"] = joblib.load(MODELS_DIR / "regression_features.pkl")
        print("✅ Modèle régression chargé")
    except Exception as e:
        print(f"⚠️  Régression : {e}")

    # ── Scaler principal (fit sur X_train dans preprocessing.py) ──
    try:
        arts["scaler_main"] = joblib.load(MODELS_DIR / "scaler.joblib")
        print("✅ Scaler principal chargé")
    except Exception as e:
        print(f"⚠️  Scaler principal : {e}")

    return arts

ARTIFACTS = load_artifacts()


# ─── Helpers d'alignement (identiques à predict.py) ─────

def align_to_feature_list(df, feature_names):
    df_a = df.copy()
    for col in feature_names:
        if col not in df_a.columns:
            df_a[col] = 0.0
    return df_a[feature_names]

def align_for_kmeans(df, scaler, feature_names):
    X = df.copy()
    for col in feature_names:
        if col not in X.columns:
            X[col] = 0.0
    X = X[feature_names]
    if X.isnull().values.any():
        X = X.fillna(0)
    return scaler.transform(X)

def align_for_regression(df, feature_names):
    X = df.copy()
    for col in feature_names:
        if col not in X.columns:
            X[col] = 0.0
    X = X[feature_names].select_dtypes(include=[np.number])
    if X.isnull().values.any():
        X = X.fillna(0)
    return X


# ─────────────────────────────────────────────────────────
# PIPELINE DE TRANSFORMATION (reproduit preprocessing.py)
# ─────────────────────────────────────────────────────────

# Mappings ordinaux (alignés avec preprocessing.py)
ORDINAL_MAPPINGS = {
    "LoyaltyLevel":      {"Inconnu": 0, "Nouveau": 1, "Jeune": 2, "Établi": 3, "Ancien": 4},
    "AgeCategory":       {"Inconnu": 0, "18-24": 1, "25-34": 2, "35-44": 3,
                          "45-54": 4, "55-64": 5, "65+": 6},
    "SpendingCategory":  {"Low": 1, "Medium": 2, "High": 3, "VIP": 4},
    "BasketSizeCategory":{"Inconnu": 0, "Petit": 1, "Moyen": 2, "Grand": 3},
    "PreferredTimeOfDay":{"Nuit": 0, "Matin": 1, "Midi": 2, "Après-midi": 3, "Soir": 4},
}

# Colonnes supprimées pour data leakage (alignées avec preprocessing.py)
LEAKAGE_PREFIXES = ["AccountStatus", "RFMSegment", "CustomerType", "FavoriteSeason"]
LEAKAGE_EXACT    = ["FirstPurchaseDaysAgo", "CustomerTenureDays", "Recency",
                    "PreferredMonth", "RegYear", "RegMonth"]

# Colonnes one-hot (alignées avec ONE_HOT_COLS de preprocessing.py)
ONE_HOT_COLS = [
    "Gender", "WeekendPref", "ProductDiversity",
    "FavoriteSeason", "Region", "AccountStatus",
    "CustomerType", "RFMSegment",
]

# Valeur par défaut pour Country (taux de churn moyen global — fallback)
DEFAULT_COUNTRY_CHURN_RATE = 0.25


def preprocess_input(raw: dict) -> pd.DataFrame:
    """
    Applique le même pipeline que preprocessing.py sur un dict de saisie brute.
    Retourne un DataFrame d'une ligne, prêt à être passé aux modèles.
    """
    df = pd.DataFrame([raw])

    # ── 1. Conversion numérique sécurisée ──────────────────
    for col in df.columns:
        try:
            df[col] = pd.to_numeric(df[col])
        except Exception:
            pass  # laisse en string pour les colonnes catégorielles

    # ── 2. Feature Engineering ─────────────────────────────
    mt  = float(raw.get("MonetaryTotal", 0) or 0)
    rec = float(raw.get("Recency",       0) or 0)
    frq = float(raw.get("Frequency",     1) or 1)
    ten = float(raw.get("CustomerTenure",1) or 1)

    df["MonetaryPerDay"]  = mt / (rec + 1)
    df["AvgBasketValue"]  = mt / frq if frq != 0 else 0.0
    df["TenureRatio"]     = rec / ten if ten != 0 else 0.0

    # IP features (non saisies dans le formulaire → valeurs neutres)
    df["IP_IsPrivate"]  = 0
    df["IP_FirstOctet"] = 0

    # ── 3. Encodage ordinal ────────────────────────────────
    for col, mapping in ORDINAL_MAPPINGS.items():
        if col in df.columns:
            df[col] = df[col].map(mapping).fillna(0).astype(int)

    # ── 4. One-Hot Encoding ────────────────────────────────
    ohe_cols_present = [c for c in ONE_HOT_COLS if c in df.columns]
    if ohe_cols_present:
        df = pd.get_dummies(df, columns=ohe_cols_present, drop_first=False)
        bool_cols = df.select_dtypes(include=["bool"]).columns.tolist()
        df[bool_cols] = df[bool_cols].astype(int)

    # ── 5. Target Encoding Country ────────────────────────
    if "Country" in df.columns:
        df["Country"] = DEFAULT_COUNTRY_CHURN_RATE  # valeur par défaut si pays inconnu

    # ── 6. Suppression data leakage ───────────────────────
    cols_to_drop = []
    for prefix in LEAKAGE_PREFIXES:
        cols_to_drop += [c for c in df.columns if c.startswith(prefix)]
    cols_to_drop += [c for c in LEAKAGE_EXACT if c in df.columns]
    df.drop(columns=list(set(cols_to_drop)), inplace=True, errors="ignore")

    # ── 7. Suppression colonnes non numériques résiduelles ─
    df = df.select_dtypes(include=[np.number])
    df.fillna(0, inplace=True)

    # ── 8. StandardScaler (le même que fit sur X_train) ────
    if "scaler_main" in ARTIFACTS:
        scaler = ARTIFACTS["scaler_main"]
        # Le scaler ne connaît que les colonnes continues vues à l'entraînement.
        # On aligne : colonnes connues du scaler qui existent dans df → on les scale.
        try:
            scaler_cols = list(scaler.feature_names_in_)
        except AttributeError:
            # Vieux scikit-learn sans feature_names_in_ → on scale tout
            scaler_cols = df.columns.tolist()

        # Colonnes présentes dans df ET dans le scaler
        cols_to_scale = [c for c in scaler_cols if c in df.columns]
        # Colonnes manquantes → ajout avec 0 avant transform
        for c in scaler_cols:
            if c not in df.columns:
                df[c] = 0.0

        # On ne scale que les colonnes non-binaires (comme preprocessing.py)
        def is_binary(series):
            unique_vals = set(series.dropna().unique())
            return unique_vals.issubset({0, 1, 0.0, 1.0})

        binary_in_scaler = [c for c in scaler_cols if is_binary(df[c])]
        non_binary_in_scaler = [c for c in scaler_cols if c not in binary_in_scaler]

        if non_binary_in_scaler:
            df[non_binary_in_scaler] = scaler.transform(df[scaler_cols])[
                :, [scaler_cols.index(c) for c in non_binary_in_scaler]
            ]

    return df


def build_input_df(data: dict) -> pd.DataFrame:
    """Transforme le dict du formulaire en DataFrame prêt-modèle."""
    return preprocess_input(data)


def risk_level(prob: float) -> str:
    if prob < 0.25:   return "Faible"
    if prob < 0.50:   return "Moyen"
    if prob < 0.75:   return "Élevé"
    return "Critique"

CLUSTER_LABELS = {
    0: ("Champions",       "Clients haute valeur, actifs et fidèles."),
    1: ("Fidèles",         "Clients réguliers à potentiel de croissance."),
    2: ("À risque",        "Anciens bons clients qui s'éloignent."),
    3: ("Dormants",        "Clients inactifs depuis longtemps."),
}

def cluster_info(cid: int):
    return CLUSTER_LABELS.get(cid, (f"Cluster {cid}", "Profil indéterminé."))


# ─── Analyse complète ────────────────────────────────────

def full_analysis(df_input: pd.DataFrame) -> dict:
    result = {}

    # Churn
    if "model_churn" in ARTIFACTS:
        X_c = align_to_feature_list(df_input, ARTIFACTS["features_churn"])
        prob = float(ARTIFACTS["model_churn"].predict_proba(X_c)[0, 1])
        result["churn_prob"]  = round(prob * 100, 1)
        result["churn_pred"]  = int(prob > 0.5)
        result["risk_level"]  = risk_level(prob)
    else:
        result["churn_error"] = "Modèle churn non disponible."

    # Clustering
    if "model_km" in ARTIFACTS:
        X_km  = align_for_kmeans(df_input, ARTIFACTS["scaler_km"], ARTIFACTS["features_km"])
        X_pca = ARTIFACTS["pca_km"].transform(X_km)
        cid   = int(ARTIFACTS["model_km"].predict(X_pca)[0])
        label, desc = cluster_info(cid)
        result["cluster_id"]    = cid
        result["cluster_label"] = label
        result["cluster_desc"]  = desc
    else:
        result["cluster_error"] = "Modèle clustering non disponible."

    # Régression
    if "model_reg" in ARTIFACTS:
        X_r   = align_for_regression(df_input, ARTIFACTS["features_reg"])
        value = float(ARTIFACTS["model_reg"].predict(X_r)[0])
        result["predicted_value"] = round(value, 2)
    else:
        result["reg_error"] = "Modèle régression non disponible."

    return result


# ─── Template HTML ────────────────────────────────────────
HTML = """<!DOCTYPE html>
<html lang="fr">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Retail ML — Analyse Client</title>
<link href="https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600&family=DM+Mono:wght@400;500&display=swap" rel="stylesheet">
<style>
:root {
  --bg: #f7f6f2;
  --surface: #ffffff;
  --border: #e8e6df;
  --text: #1a1916;
  --muted: #706e67;
  --accent: #1a56e8;
  --accent-light: #ebf0fd;
  --green: #16a34a;
  --green-bg: #dcfce7;
  --red: #dc2626;
  --red-bg: #fee2e2;
  --amber: #d97706;
  --amber-bg: #fef3c7;
  --orange: #ea580c;
  --orange-bg: #ffedd5;
  --purple: #7c3aed;
  --purple-bg: #ede9fe;
  --radius: 10px;
  --shadow: 0 1px 3px rgba(0,0,0,.07), 0 4px 16px rgba(0,0,0,.04);
}
* { box-sizing: border-box; margin: 0; padding: 0; }
body {
  font-family: 'DM Sans', sans-serif;
  background: var(--bg);
  color: var(--text);
  min-height: 100vh;
  padding: 0 0 60px;
}
.topbar {
  background: var(--surface);
  border-bottom: 1px solid var(--border);
  padding: 16px 32px;
  display: flex;
  align-items: center;
  gap: 12px;
}
.topbar-icon {
  width: 34px; height: 34px;
  background: var(--accent);
  border-radius: 8px;
  display: flex; align-items: center; justify-content: center;
  color: #fff; font-size: 17px;
}
.topbar h1 { font-size: 16px; font-weight: 600; }
.topbar span { font-size: 13px; color: var(--muted); margin-left: 4px; }

.layout {
  max-width: 1080px;
  margin: 32px auto 0;
  padding: 0 24px;
  display: grid;
  grid-template-columns: 1fr 380px;
  gap: 24px;
}
@media (max-width: 800px) {
  .layout { grid-template-columns: 1fr; }
}

.card {
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  box-shadow: var(--shadow);
  overflow: hidden;
}
.card-header {
  padding: 18px 24px 14px;
  border-bottom: 1px solid var(--border);
  display: flex;
  align-items: center;
  gap: 8px;
}
.card-header h2 { font-size: 14px; font-weight: 600; }
.card-header .badge-section {
  margin-left: auto;
  font-size: 11px;
  background: var(--accent-light);
  color: var(--accent);
  padding: 3px 9px;
  border-radius: 20px;
  font-weight: 500;
}
.card-body { padding: 20px 24px; }

.section-label {
  font-size: 10px;
  font-weight: 600;
  letter-spacing: .08em;
  text-transform: uppercase;
  color: var(--muted);
  margin-bottom: 12px;
  margin-top: 20px;
}
.section-label:first-child { margin-top: 0; }

.form-grid {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 12px;
}
.field { display: flex; flex-direction: column; gap: 5px; }
.field label {
  font-size: 11px;
  font-weight: 500;
  color: var(--muted);
  text-transform: uppercase;
  letter-spacing: .05em;
}
.field input, .field select {
  padding: 9px 11px;
  border: 1px solid var(--border);
  border-radius: 7px;
  font-size: 14px;
  font-family: 'DM Sans', sans-serif;
  color: var(--text);
  background: var(--surface);
  transition: border-color .15s;
  outline: none;
}
.field input:focus, .field select:focus {
  border-color: var(--accent);
  box-shadow: 0 0 0 3px rgba(26,86,232,.08);
}

.submit-btn {
  margin-top: 22px;
  width: 100%;
  padding: 13px;
  background: var(--accent);
  color: #fff;
  border: none;
  border-radius: 8px;
  font-size: 14px;
  font-weight: 600;
  font-family: 'DM Sans', sans-serif;
  cursor: pointer;
  transition: background .15s, transform .1s;
  letter-spacing: .01em;
}
.submit-btn:hover { background: #1545c8; }
.submit-btn:active { transform: scale(.99); }

/* Résultats */
.results-placeholder {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  padding: 40px 20px;
  gap: 10px;
  color: var(--muted);
  font-size: 13px;
  text-align: center;
}
.placeholder-icon {
  font-size: 36px;
  opacity: .35;
}

.result-block {
  padding: 18px 0;
  border-bottom: 1px solid var(--border);
}
.result-block:last-child { border-bottom: none; padding-bottom: 0; }
.result-block:first-child { padding-top: 0; }

.result-title {
  font-size: 11px;
  font-weight: 600;
  text-transform: uppercase;
  letter-spacing: .07em;
  color: var(--muted);
  margin-bottom: 10px;
}
.churn-meter {
  margin-bottom: 10px;
}
.meter-bar {
  height: 8px;
  background: var(--border);
  border-radius: 4px;
  overflow: hidden;
  margin: 8px 0 6px;
}
.meter-fill {
  height: 100%;
  border-radius: 4px;
  transition: width .4s ease;
}
.meter-labels {
  display: flex;
  justify-content: space-between;
  font-size: 11px;
  color: var(--muted);
}
.churn-score {
  font-size: 32px;
  font-weight: 600;
  font-family: 'DM Mono', monospace;
  line-height: 1;
  margin-bottom: 4px;
}
.verdict {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  padding: 5px 12px;
  border-radius: 20px;
  font-size: 13px;
  font-weight: 500;
  margin-top: 6px;
}
.verdict.fidele  { background: var(--green-bg); color: var(--green); }
.verdict.churn   { background: var(--red-bg);   color: var(--red);   }

.risk-badge {
  display: inline-block;
  padding: 3px 10px;
  border-radius: 20px;
  font-size: 12px;
  font-weight: 500;
  margin-top: 6px;
}
.risk-Faible   { background: var(--green-bg);  color: var(--green);  }
.risk-Moyen    { background: var(--amber-bg);  color: var(--amber);  }
.risk-Élevé    { background: var(--orange-bg); color: var(--orange); }
.risk-Critique { background: var(--red-bg);    color: var(--red);    }

.cluster-card {
  display: flex;
  align-items: center;
  gap: 14px;
  padding: 14px;
  background: var(--bg);
  border-radius: 8px;
  border: 1px solid var(--border);
}
.cluster-num {
  width: 44px; height: 44px;
  border-radius: 10px;
  display: flex; align-items: center; justify-content: center;
  font-size: 18px; font-weight: 600;
  font-family: 'DM Mono', monospace;
  flex-shrink: 0;
}
.c0 { background: var(--accent-light); color: var(--accent); }
.c1 { background: var(--green-bg);     color: var(--green);  }
.c2 { background: var(--orange-bg);    color: var(--orange); }
.c3 { background: var(--purple-bg);    color: var(--purple); }

.cluster-label { font-size: 15px; font-weight: 600; }
.cluster-desc  { font-size: 12px; color: var(--muted); margin-top: 2px; }

.value-display {
  display: flex;
  align-items: baseline;
  gap: 6px;
}
.value-big {
  font-size: 28px;
  font-weight: 600;
  font-family: 'DM Mono', monospace;
}
.value-unit { font-size: 14px; color: var(--muted); }

.reco-list {
  list-style: none;
  display: flex;
  flex-direction: column;
  gap: 8px;
  margin-top: 4px;
}
.reco-list li {
  font-size: 13px;
  padding: 8px 12px;
  background: var(--bg);
  border-radius: 7px;
  border-left: 3px solid var(--accent);
  line-height: 1.45;
}

.error-box {
  padding: 12px 16px;
  background: var(--red-bg);
  color: var(--red);
  border-radius: 8px;
  font-size: 13px;
  margin-top: 12px;
}

.models-status {
  display: flex;
  flex-direction: column;
  gap: 6px;
  padding: 12px 24px 16px;
}
.model-row {
  display: flex;
  align-items: center;
  justify-content: space-between;
  font-size: 12px;
}
.model-row .name { color: var(--muted); }
.dot { width: 7px; height: 7px; border-radius: 50%; }
.dot.ok  { background: var(--green);  }
.dot.err { background: var(--red);    }
</style>
</head>
<body>

<div class="topbar">
  <div class="topbar-icon">🛍</div>
  <h1>Retail ML</h1>
  <span>— Analyse comportementale clientèle</span>
</div>

<div class="layout">

  <!-- Formulaire -->
  <div>
    <div class="card">
      <div class="card-header">
        <span>📋</span>
        <h2>Profil client</h2>
        <span class="badge-section">Saisie manuelle</span>
      </div>
      <div class="card-body">
        <form method="POST" action="/predict">

          <p class="section-label">Données RFM</p>
          <div class="form-grid">
            <div class="field">
              <label>Récence (jours)</label>
              <input type="number" name="Recency" value="{{ form.Recency or 30 }}" min="0" max="400">
            </div>
            <div class="field">
              <label>Fréquence (commandes)</label>
              <input type="number" name="Frequency" value="{{ form.Frequency or 5 }}" min="1" max="50">
            </div>
            <div class="field">
              <label>Total dépensé (£)</label>
              <input type="number" name="MonetaryTotal" value="{{ form.MonetaryTotal or 500 }}" step="0.01">
            </div>
            <div class="field">
              <label>Moyenne par commande (£)</label>
              <input type="number" name="MonetaryAvg" value="{{ form.MonetaryAvg or 100 }}" step="0.01">
            </div>
            <div class="field">
              <label>Écart-type dépenses (£)</label>
              <input type="number" name="MonetaryStd" value="{{ form.MonetaryStd or 50 }}" step="0.01">
            </div>
            <div class="field">
              <label>Total articles</label>
              <input type="number" name="TotalQuantity" value="{{ form.TotalQuantity or 100 }}" step="1">
            </div>
          </div>

          <p class="section-label">Profil client</p>
          <div class="form-grid">
            <div class="field">
              <label>Ancienneté (jours)</label>
              <input type="number" name="CustomerTenure" value="{{ form.CustomerTenure or 365 }}" min="0" max="730">
            </div>
            <div class="field">
              <label>Âge estimé</label>
              <input type="number" name="Age" value="{{ form.Age or 35 }}" min="18" max="81">
            </div>
            <div class="field">
              <label>Satisfaction (1–5)</label>
              <input type="number" name="Satisfaction" value="{{ form.Satisfaction or 4 }}" min="1" max="5" step="0.1">
            </div>
            <div class="field">
              <label>Tickets support</label>
              <input type="number" name="SupportTickets" value="{{ form.SupportTickets or 1 }}" min="0" max="20">
            </div>
            <div class="field">
              <label>Segment RFM</label>
              <select name="RFMSegment">
                <option value="0" {% if form.RFMSegment=='0' %}selected{% endif %}>Champions</option>
                <option value="1" {% if form.RFMSegment=='1' %}selected{% endif %}>Fidèles</option>
                <option value="2" {% if form.RFMSegment=='2' %}selected{% endif %}>Potentiels</option>
                <option value="3" {% if form.RFMSegment=='3' %}selected{% endif %}>Dormants</option>
              </select>
            </div>
            <div class="field">
              <label>Niveau dépense</label>
              <select name="SpendingCat">
                <option value="0" {% if form.SpendingCat=='0' %}selected{% endif %}>Low</option>
                <option value="1" {% if form.SpendingCat=='1' %}selected{% endif %}>Medium</option>
                <option value="2" {% if form.SpendingCat=='2' %}selected{% endif %}>High</option>
                <option value="3" {% if form.SpendingCat=='3' %}selected{% endif %}>VIP</option>
              </select>
            </div>
          </div>

          <p class="section-label">Comportement d'achat</p>
          <div class="form-grid">
            <div class="field">
              <label>Transactions annulées</label>
              <input type="number" name="CancelledTrans" value="{{ form.CancelledTrans or 0 }}" min="0">
            </div>
            <div class="field">
              <label>Taux de retour</label>
              <input type="number" name="ReturnRatio" value="{{ form.ReturnRatio or 0.05 }}" min="0" max="1" step="0.01">
            </div>
            <div class="field">
              <label>Produits uniques</label>
              <input type="number" name="UniqueProducts" value="{{ form.UniqueProducts or 20 }}" min="1">
            </div>
            <div class="field">
              <label>Ratio weekend</label>
              <input type="number" name="WeekendRatio" value="{{ form.WeekendRatio or 0.3 }}" min="0" max="1" step="0.01">
            </div>
            <div class="field">
              <label>Total transactions</label>
              <input type="number" name="TotalTrans" value="{{ form.TotalTrans or 50 }}" min="1">
            </div>
            <div class="field">
              <label>Factures distinctes</label>
              <input type="number" name="UniqueInvoices" value="{{ form.UniqueInvoices or 10 }}" min="1">
            </div>
          </div>

          <button type="submit" class="submit-btn">🔍 Analyser ce client</button>
        </form>
      </div>
    </div>
  </div>

  <!-- Panneau résultats -->
  <div style="display: flex; flex-direction: column; gap: 16px;">

    <!-- État des modèles -->
    <div class="card">
      <div class="card-header">
        <span>⚙️</span>
        <h2>Modèles chargés</h2>
      </div>
      <div class="models-status">
        <div class="model-row">
          <span class="name">Churn (classification)</span>
          <div class="dot {{ 'ok' if models_status.churn else 'err' }}"></div>
        </div>
        <div class="model-row">
          <span class="name">Segmentation (K-Means)</span>
          <div class="dot {{ 'ok' if models_status.clustering else 'err' }}"></div>
        </div>
        <div class="model-row">
          <span class="name">Valeur client (régression)</span>
          <div class="dot {{ 'ok' if models_status.regression else 'err' }}"></div>
        </div>
      </div>
    </div>

    <!-- Résultats -->
    <div class="card">
      <div class="card-header">
        <span>📊</span>
        <h2>Résultats de l'analyse</h2>
      </div>
      <div class="card-body">

        {% if error %}
          <div class="error-box">❌ {{ error }}</div>

        {% elif result %}

          <!-- Churn -->
          {% if result.churn_prob is defined %}
          <div class="result-block">
            <p class="result-title">Risque de churn</p>
            <div class="churn-score" style="color: {{ '#dc2626' if result.churn_pred == 1 else '#16a34a' }}">
              {{ result.churn_prob }}%
            </div>
            <div class="meter-bar">
              <div class="meter-fill" style="width: {{ result.churn_prob }}%; background: {{ '#dc2626' if result.churn_pred == 1 else '#16a34a' }};"></div>
            </div>
            <div class="meter-labels"><span>0%</span><span>50%</span><span>100%</span></div>
            <span class="verdict {{ 'churn' if result.churn_pred == 1 else 'fidele' }}">
              {{ '⚠️ Risque de départ' if result.churn_pred == 1 else '✅ Client fidèle' }}
            </span>
            <br>
            <span class="risk-badge risk-{{ result.risk_level }}">{{ result.risk_level }}</span>
          </div>
          {% endif %}

          <!-- Clustering -->
          {% if result.cluster_id is defined %}
          <div class="result-block">
            <p class="result-title">Segment client</p>
            <div class="cluster-card">
              <div class="cluster-num c{{ result.cluster_id }}">{{ result.cluster_id }}</div>
              <div>
                <div class="cluster-label">{{ result.cluster_label }}</div>
                <div class="cluster-desc">{{ result.cluster_desc }}</div>
              </div>
            </div>
          </div>
          {% endif %}

          <!-- Régression -->
          {% if result.predicted_value is defined %}
          <div class="result-block">
            <p class="result-title">Valeur monétaire prédite</p>
            <div class="value-display">
              <span class="value-big">{{ "£{:,.0f}".format(result.predicted_value) }}</span>
              <span class="value-unit">valeur totale estimée</span>
            </div>
          </div>
          {% endif %}

          <!-- Recommandations -->
          <div class="result-block">
            <p class="result-title">Recommandations marketing</p>
            <ul class="reco-list">
              {% for r in result.recommendations %}
              <li>{{ r }}</li>
              {% endfor %}
            </ul>
          </div>

        {% else %}
          <div class="results-placeholder">
            <div class="placeholder-icon">🔍</div>
            <span>Remplissez le formulaire et cliquez sur<br><strong>Analyser ce client</strong> pour obtenir les résultats.</span>
          </div>
        {% endif %}

      </div>
    </div>

  </div>
</div>

</body>
</html>"""


def build_recommendations(result: dict) -> list:
    recos = []
    pred  = result.get("churn_pred", 0)
    risk  = result.get("risk_level", "Faible")
    cid   = result.get("cluster_id")
    val   = result.get("predicted_value", 0)

    if pred == 1:
        if risk == "Critique":
            recos.append("🚨 Intervention immédiate : proposer une offre de rétention personnalisée (réduction, cadeau, appel commercial).")
            recos.append("📞 Contacter proactivement le client sous 48h via le canal préféré.")
        elif risk == "Élevé":
            recos.append("📩 Envoyer une campagne e-mail de réengagement avec code promo exclusif.")
            recos.append("🎁 Proposer un programme de fidélité adapté à son historique d'achats.")
        else:
            recos.append("👁️ Surveiller l'activité du client — activer une alerte si inactivité > 30 jours.")
    else:
        recos.append("⭐ Client fidèle — opportunité d'upselling sur les gammes premium.")
        if val and val > 1000:
            recos.append("💎 Valeur élevée : inclure dans le programme VIP et inviter aux avant-premières.")

    if cid == 0:
        recos.append("🏆 Champion : solliciter un témoignage ou un rôle d'ambassadeur de marque.")
    elif cid == 1:
        recos.append("📈 Fidèle : renforcer la relation avec des contenus personnalisés et offres anticipées.")
    elif cid == 2:
        recos.append("🔄 À risque : relancer avec une sélection basée sur ses anciens achats favoris.")
    elif cid == 3:
        recos.append("😴 Dormant : campagne de réactivation avec offre limitée dans le temps.")

    if not recos:
        recos.append("📊 Continuer le suivi standard — pas d'action urgente requise.")

    return recos


# ─── Routes ──────────────────────────────────────────────

def models_status_dict():
    return {
        "churn":      "model_churn" in ARTIFACTS,
        "clustering": "model_km" in ARTIFACTS,
        "regression": "model_reg" in ARTIFACTS,
    }

@app.route("/", methods=["GET"])
def index():
    return render_template_string(
        HTML,
        result=None,
        error=None,
        form={},
        models_status=models_status_dict()
    )

@app.route("/predict", methods=["POST"])
def predict():
    ms = models_status_dict()

    if not ms["churn"] and not ms["clustering"] and not ms["regression"]:
        return render_template_string(
            HTML, result=None, form=dict(request.form),
            error="Aucun modèle chargé. Exécutez preprocessing.py, train_model.py, clustering.py et regression.py.",
            models_status=ms
        )

    try:
        df = build_input_df(dict(request.form))
        result = full_analysis(df)
        result["recommendations"] = build_recommendations(result)
    except Exception as e:
        import traceback
        return render_template_string(
            HTML, result=None, form=dict(request.form),
            error=f"{e}\n{traceback.format_exc()}", models_status=ms
        )

    return render_template_string(
        HTML, result=result, form=dict(request.form),
        error=None, models_status=ms
    )

@app.route("/api/predict", methods=["POST"])
def api_predict():
    """Endpoint JSON pour intégration externe."""
    if not ARTIFACTS:
        return jsonify({"error": "Aucun modèle chargé"}), 503

    data = request.get_json(force=True)
    try:
        df = build_input_df(data)
        result = full_analysis(df)
        result["recommendations"] = build_recommendations(result)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route("/api/health", methods=["GET"])
def health():
    ms = models_status_dict()
    return jsonify({
        "status": "ok" if any(ms.values()) else "no_models",
        "models": ms,
    })

if __name__ == "__main__":
    print("🚀 Serveur Flask — http://127.0.0.1:5000")
    app.run(debug=True, host="0.0.0.0", port=5000)