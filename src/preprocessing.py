import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler


# ─────────────────────────────────────────────────────────
# ÉTAPE 0 : NETTOYAGE BRUT  (appliqué avant le split)
# ─────────────────────────────────────────────────────────

def clean_raw_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Nettoyage initial du dataframe brut :
    - Supprime NewsletterSubscribed (constante)
    - Corrige MonetaryTotal (valeurs datetime parasites → NaN → numérique)
    - Parse RegistrationDate, extrait RegYear/Month/Day/Weekday
    - Uniformise la casse des colonnes catégorielles textuelles
    Retourne le dataframe nettoyé (RegistrationDate supprimée après extraction).
    """
    df = df.copy()

    # --- Suppression de la colonne constante ---
    cols_to_drop = [c for c in ["NewsletterSubscribed", "Newsletter"] if c in df.columns]
    df.drop(columns=cols_to_drop, inplace=True)
    print(f"[clean] Colonnes supprimées (inutiles) : {cols_to_drop}")

    # --- Correction de MonetaryTotal ---
    if "MonetaryTotal" in df.columns:
        def to_numeric_safe(val):
            """Convertit en float, retourne NaN si c'est une date ou illisible."""
            if isinstance(val, (pd.Timestamp,)):
                return np.nan
            try:
                return float(val)
            except (ValueError, TypeError):
                return np.nan

        df["MonetaryTotal"] = df["MonetaryTotal"].apply(to_numeric_safe)
        n_nan = df["MonetaryTotal"].isna().sum()
        print(f"[clean] MonetaryTotal : {n_nan} valeurs aberrantes converties en NaN")

    # --- Correction SupportTickets : valeurs sentinelles -1 et 999 → NaN ---
    if "SupportTickets" in df.columns:
        df["SupportTickets"] = pd.to_numeric(df["SupportTickets"], errors="coerce")
        df["SupportTickets"] = df["SupportTickets"].replace([-1, 999], np.nan)
        df["SupportTickets"] = df["SupportTickets"].clip(lower=0, upper=20)
        print("[clean] SupportTickets : -1/999 → NaN, clip [0,20]")

    # --- Correction Satisfaction : valeurs sentinelles -1, 0 et 99 → NaN ---
    if "Satisfaction" in df.columns:
        df["Satisfaction"] = pd.to_numeric(df["Satisfaction"], errors="coerce")
        df["Satisfaction"] = df["Satisfaction"].replace([-1, 0, 99], np.nan)
        df["Satisfaction"] = df["Satisfaction"].clip(lower=1, upper=5)
        print("[clean] Satisfaction : -1/0/99 → NaN, clip [1,5]")

    # --- Parsing de RegistrationDate ---
    if "RegistDate" in df.columns or "RegistrationDate" in df.columns:
        date_col = "RegistDate" if "RegistDate" in df.columns else "RegistrationDate"
        if not pd.api.types.is_datetime64_any_dtype(df[date_col]):
            df[date_col] = pd.to_datetime(
                df[date_col], dayfirst=True, errors="coerce"
            )
        n_nat = df[date_col].isna().sum()
        print(f"[clean] {date_col} : {n_nat} dates non convertibles (NaT)")

        df["RegYear"]    = df[date_col].dt.year
        df["RegMonth"]   = df[date_col].dt.month
        df["RegDay"]     = df[date_col].dt.day
        df["RegWeekday"] = df[date_col].dt.weekday
        df.drop(columns=[date_col], inplace=True)
        print(f"[clean] {date_col} → RegYear, RegMonth, RegDay, RegWeekday")

    # --- Uniformisation de la casse ---
    free_text_cols = ["Country", "Gender", "AccountStatus"]
    for col in free_text_cols:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()
    print(f"[clean] Nettoyage espaces terminé pour : {free_text_cols}")

    print(f"[clean] Colonnes après nettoyage brut : {df.shape[1]}")
    return df


# ─────────────────────────────────────────────────────────
# ÉTAPE 1 : FEATURE ENGINEERING (avant le split)
# ─────────────────────────────────────────────────────────

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Crée de nouvelles features et traite LastLoginIP.
    À appliquer sur le dataset complet AVANT le split.
    """
    df = df.copy()

    # --- Feature engineering depuis IP ---
    if "LastLoginIP" in df.columns:
        def is_private(ip):
            try:
                parts = str(ip).split(".")
                if len(parts) != 4:
                    return 0
                first, second = int(parts[0]), int(parts[1])
                return int(first == 10 or
                           (first == 172 and 16 <= second <= 31) or
                           (first == 192 and second == 168))
            except Exception:
                return 0

        def first_octet(ip):
            try:
                return int(str(ip).split(".")[0])
            except Exception:
                return -1

        df["IP_IsPrivate"]  = df["LastLoginIP"].apply(is_private)
        df["IP_FirstOctet"] = df["LastLoginIP"].apply(first_octet)
        df.drop(columns=["LastLoginIP"], inplace=True)
        print("[engineer] LastLoginIP → IP_IsPrivate + IP_FirstOctet")

    # --- Ratio dépenses / récence ---
    if "MonetaryTotal" in df.columns and "Recency" in df.columns:
        df["MonetaryPerDay"] = df["MonetaryTotal"] / (df["Recency"] + 1)

    # --- Panier moyen ---
    if "MonetaryTotal" in df.columns and "Frequency" in df.columns:
        df["AvgBasketValue"] = df["MonetaryTotal"] / df["Frequency"].replace(0, np.nan)

    # --- Ancienneté vs activité récente ---
    if "Recency" in df.columns and "CustomerTenure" in df.columns:
        df["TenureRatio"] = df["Recency"] / df["CustomerTenure"].replace(0, np.nan)

    print(f"[engineer] Colonnes après feature engineering : {df.shape[1]}")
    return df


# ─────────────────────────────────────────────────────────
# ÉTAPE 2 : SÉPARATION X / y  ET  TRAIN / TEST SPLIT
# ─────────────────────────────────────────────────────────

def split_data(df: pd.DataFrame,
               target_col: str = "Churn",
               test_size: float = 0.2,
               random_state: int = 42):
    """
    Sépare features et target, puis effectue un split stratifié train/test.
    Exclut CustomerID, la target et les colonnes à risque de data leakage.
    """
    # Colonnes à exclure (identifiant + leakage direct avec le churn)
    leakage_cols = [
        "ChurnRisk",          # dérivé direct de Churn
        "ChurnRiskCategory",  # alias possible
    ]

    cols_to_exclude = ["CustomerID", target_col] + leakage_cols
    cols_to_exclude = [c for c in cols_to_exclude if c in df.columns]

    X = df.drop(columns=cols_to_exclude)
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )
    print(f"[split] Train : {X_train.shape[0]} lignes | Test : {X_test.shape[0]} lignes")
    print(f"[split] Distribution Churn train → {y_train.value_counts().to_dict()}")
    return X_train, X_test, y_train, y_test


# ─────────────────────────────────────────────────────────
# ÉTAPE 3 : SUPPRESSION COLONNES À FORT TAUX DE MANQUANTS
# ─────────────────────────────────────────────────────────

def drop_high_missing_cols(X_train: pd.DataFrame,
                            X_test: pd.DataFrame,
                            threshold: float = 0.5) -> tuple:
    """Supprime les colonnes ayant plus de `threshold` de valeurs manquantes dans X_train."""
    X_train_temp = X_train.replace("Inconnu", np.nan)
    missing_rate  = X_train_temp.isnull().mean()
    cols_to_drop  = missing_rate[missing_rate > threshold].index.tolist()

    X_train = X_train.drop(columns=cols_to_drop)
    X_test  = X_test.drop(columns=cols_to_drop)

    if cols_to_drop:
        print(f"[drop_high_missing] Colonnes supprimées (>{threshold*100:.0f}% manquants) : {cols_to_drop}")
    else:
        print(f"[drop_high_missing] Aucune colonne supprimée.")
    return X_train, X_test


# ─────────────────────────────────────────────────────────
# ÉTAPE 4 : CORRECTION DES VALEURS ABERRANTES
# ─────────────────────────────────────────────────────────

def fix_outliers(X_train: pd.DataFrame,
                 X_test: pd.DataFrame) -> tuple:
    """
    Corrige les valeurs aberrantes connues du dataset.
    Noms de colonnes alignés avec le dataset réel (SupportTickets, Satisfaction).
    """
    X_train, X_test = X_train.copy(), X_test.copy()

    # SupportTickets (peut avoir été nettoyé dans clean_raw_data, double sécurité)
    for col in ["SupportTickets", "SupportTicketsCount"]:
        if col in X_train.columns:
            X_train[col] = pd.to_numeric(X_train[col], errors="coerce").clip(lower=0, upper=20)
            X_test[col]  = pd.to_numeric(X_test[col],  errors="coerce").clip(lower=0, upper=20)
            print(f"[fix_outliers] {col} clippé [0, 20]")

    # Satisfaction (peut avoir été nettoyé dans clean_raw_data, double sécurité)
    for col in ["Satisfaction", "SatisfactionScore"]:
        if col in X_train.columns:
            X_train[col] = X_train[col].replace(0, np.nan)
            X_test[col]  = X_test[col].replace(0, np.nan)
            X_train[col] = pd.to_numeric(X_train[col], errors="coerce").clip(lower=1, upper=5)
            X_test[col]  = pd.to_numeric(X_test[col],  errors="coerce").clip(lower=1, upper=5)
            print(f"[fix_outliers] {col} : 0/-1/99 → NaN, clip [1,5]")

    return X_train, X_test


# ─────────────────────────────────────────────────────────
# ÉTAPE 5 : IMPUTATION
# ─────────────────────────────────────────────────────────

def impute_numeric(X_train: pd.DataFrame,
                   X_test: pd.DataFrame,
                   strategy_map: dict = None) -> tuple:
    """
    Impute les valeurs manquantes numériques.
    Fit sur X_train uniquement, appliqué sur X_train et X_test.
    """
    X_train, X_test = X_train.copy(), X_test.copy()

    num_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
    cols_with_nan = [c for c in num_cols if X_train[c].isna().any()]

    if not cols_with_nan:
        print("[impute] Aucun NaN numérique à imputer.")
        return X_train, X_test

    for col in cols_with_nan:
        strategy = "median"
        if strategy_map and col in strategy_map:
            strategy = strategy_map[col]

        imputer = SimpleImputer(strategy=strategy)
        X_train[[col]] = imputer.fit_transform(X_train[[col]])
        X_test[[col]]  = imputer.transform(X_test[[col]])

    print(f"[impute] Colonnes imputées : {cols_with_nan}")
    return X_train, X_test


# ─────────────────────────────────────────────────────────
# ÉTAPE 6 : ENCODAGE ORDINAL
# ─────────────────────────────────────────────────────────

# Mappings alignés avec les valeurs réelles du dataset (section 4.3-4.4 de l'énoncé)
ORDINAL_MAPPINGS = {
    # Feature 42 — LoyaltyLevel
    "LoyaltyLevel": {
        "Inconnu": 0, "Nouveau": 1, "Jeune": 2, "Établi": 3, "Ancien": 4
    },
    # Feature 36 — AgeCategory
    "AgeCategory": {
        "Inconnu": 0, "18-24": 1, "25-34": 2, "35-44": 3,
        "45-54": 4, "55-64": 5, "65+": 6
    },
    # Feature 37 — SpendingCategory
    "SpendingCategory": {
        "Low": 1, "Medium": 2, "High": 3, "VIP": 4
    },
    # Feature 45 — BasketSizeCategory
    "BasketSizeCategory": {
        "Inconnu": 0, "Petit": 1, "Moyen": 2, "Grand": 3
    },
    # Feature 40 — PreferredTimeOfDay
    "PreferredTimeOfDay": {
        "Nuit": 0, "Matin": 1, "Midi": 2, "Après-midi": 3, "Soir": 4
    },
    # Feature 43 — ChurnRisk (si présente et non exclue comme leakage)
    "ChurnRisk": {
        "Faible": 1, "Moyen": 2, "Élevé": 3, "Critique": 4
    },
}


def encode_ordinal(X_train: pd.DataFrame,
                   X_test: pd.DataFrame,
                   mappings: dict = None) -> tuple:
    """Applique l'encodage ordinal selon les mappings définis."""
    if mappings is None:
        mappings = ORDINAL_MAPPINGS

    X_train, X_test = X_train.copy(), X_test.copy()

    for col, mapping in mappings.items():
        if col not in X_train.columns:
            continue
        X_train[col] = X_train[col].map(mapping).fillna(0).astype(int)
        X_test[col]  = X_test[col].map(mapping).fillna(0).astype(int)
        print(f"[encode_ordinal] '{col}' → ordinal encodé")

    return X_train, X_test


# ─────────────────────────────────────────────────────────
# ÉTAPE 7 : ONE-HOT ENCODING
# ─────────────────────────────────────────────────────────

# Noms de colonnes alignés avec le dataset réel (section 4.3-4.4)
ONE_HOT_COLS = [
    "Gender",           # Feature 47
    "WeekendPref",      # Feature 44 (énoncé : WeekendPref)
    "ProductDiversity", # Feature 46
    "FavoriteSeason",   # Feature 39
    "Region",           # Feature 41
    "AccountStatus",    # Feature 48
    "CustomerType",     # Feature 38
    "RFMSegment",       # Feature 35 (one-hot possible selon encodage choisi)
]


def encode_one_hot(X_train: pd.DataFrame,
                   X_test: pd.DataFrame,
                   cols: list = None) -> tuple:
    """
    Applique le One-Hot Encoding.
    Aligne X_test sur la structure de X_train pour éviter les colonnes manquantes.
    """
    if cols is None:
        cols = ONE_HOT_COLS

    cols = [c for c in cols if c in X_train.columns]

    X_train = pd.get_dummies(X_train, columns=cols, drop_first=False)
    X_test  = pd.get_dummies(X_test,  columns=cols, drop_first=False)

    # Aligner : colonnes manquantes dans X_test → 0
    X_train, X_test = X_train.align(X_test, join="left", axis=1, fill_value=0)

    # pandas >= 2.0 retourne des bool → convertir en int
    bool_cols = X_train.select_dtypes(include=["bool"]).columns.tolist()
    X_train[bool_cols] = X_train[bool_cols].astype(int)
    X_test[bool_cols]  = X_test[bool_cols].astype(int)

    print(f"[encode_one_hot] One-Hot appliqué sur : {cols}")
    print(f"[encode_one_hot] Colonnes après encodage : {X_train.shape[1]}")
    return X_train, X_test


# ─────────────────────────────────────────────────────────
# ÉTAPE 8 : TARGET ENCODING pour Country
# ─────────────────────────────────────────────────────────

def encode_country_target(X_train: pd.DataFrame,
                           X_test: pd.DataFrame,
                           y_train: pd.Series,
                           country_col: str = "Country",
                           min_samples: int = 10) -> tuple:
    """
    Target Encoding pour la colonne Country (37+ modalités).
    Remplace chaque pays par son taux de churn moyen calculé sur X_train.
    Les pays rares (<min_samples) sont regroupés sous 'Other'.
    """
    if country_col not in X_train.columns:
        print(f"[encode_country] Colonne '{country_col}' absente, étape ignorée.")
        return X_train, X_test

    X_train, X_test = X_train.copy(), X_test.copy()

    # Regrouper les pays rares
    country_counts = X_train[country_col].value_counts()
    rare_countries = country_counts[country_counts < min_samples].index.tolist()

    X_train[country_col] = X_train[country_col].where(
        ~X_train[country_col].isin(rare_countries), other="Other"
    )
    X_test[country_col] = X_test[country_col].where(
        ~X_test[country_col].isin(rare_countries), other="Other"
    )

    # Taux de churn par pays (fit sur X_train uniquement — pas de leakage)
    churn_rate = (
        pd.concat([X_train[[country_col]], y_train.rename("Churn")], axis=1)
        .groupby(country_col)["Churn"]
        .mean()
    )

    global_mean = y_train.mean()

    X_train[country_col] = X_train[country_col].map(churn_rate).fillna(global_mean)
    X_test[country_col]  = X_test[country_col].map(churn_rate).fillna(global_mean)

    print(f"[encode_country] Target Encoding appliqué sur '{country_col}' "
          f"({len(churn_rate)} pays, {len(rare_countries)} pays rares → 'Other')")
    return X_train, X_test




# ─────────────────────────────────────────────────────────
# ÉTAPE : suppression active des variables colinéaires
# ─────────────────────────────────────────────────────────
def remove_highly_correlated_features(X_train: pd.DataFrame, X_test: pd.DataFrame, threshold: float = 0.8):
    """
    Identifie et supprime les variables ayant une corrélation de Pearson supérieure au seuil.
    Le calcul se fait UNIQUEMENT sur X_train pour éviter le Data Leakage.
    """
    print(f"\n[clean] Traitement de la multicolinéarité (seuil > {threshold})...")
    
    # Calcul de la matrice de corrélation absolue
    corr_matrix = X_train.corr().abs()
    
    # Sélection du triangle supérieur de la matrice pour ne pas supprimer les deux variables d'une même paire
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    
    # Identification des colonnes à supprimer
    to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
    
    # Suppression
    X_train_clean = X_train.drop(columns=to_drop)
    X_test_clean = X_test.drop(columns=to_drop)
    
    print(f"    Variables colinéaires supprimées : {to_drop if to_drop else 'Aucune'}")
    
    return X_train_clean, X_test_clean

# ─────────────────────────────────────────────────────────
# ÉTAPE : supression des colonnes qui causent une fuite de données
# ─────────────────────────────────────────────────────────
def remove_data_leakage(X_train: pd.DataFrame, X_test: pd.DataFrame):
    """
    Supprime les variables qui causent une fuite de données (Data Leakage)
    car elles contiennent directement ou indirectement la cible (Churn).
    """
    print("\n[clean] Purge des variables causant un Data Leakage...")
    
    cols_leak = []

    # 1. Le statut du compte
    cols_leak.extend([col for col in X_train.columns if 'AccountStatus' in col])
    # 2. La segmentation RFM
    cols_leak.extend([col for col in X_train.columns if 'RFMSegment' in col])
    # 3. Le type de client
    cols_leak.extend([col for col in X_train.columns if 'CustomerType' in col])
    # 4. Variables temporelles absolues et Récence
    cols_leak.extend(['FirstPurchaseDaysAgo', 'CustomerTenureDays', 'Recency'])
    # 5. Biais temporels d'extraction
    cols_leak.extend([col for col in X_train.columns if 'FavoriteSeason' in col])
    cols_leak.extend(['PreferredMonth', 'RegYear', 'RegMonth'])

    # Application de la suppression
    X_train_clean = X_train.drop(columns=cols_leak, errors='ignore')
    X_test_clean = X_test.drop(columns=cols_leak, errors='ignore')

    # Affichage propre du nombre réel de colonnes supprimées
    cols_dropped = [col for col in cols_leak if col in X_train.columns]
    print(f"    -> {len(cols_dropped)} colonnes tricheuses supprimées.")
    
    return X_train_clean, X_test_clean
# ─────────────────────────────────────────────────────────
# ÉTAPE 9 : NORMALISATION (StandardScaler)
# ─────────────────────────────────────────────────────────

def scale_features(X_train: pd.DataFrame,
                   X_test: pd.DataFrame,
                   save_path: str = None) -> tuple:
    """
    Normalise les colonnes continues (StandardScaler).
    Exclut les colonnes binaires (One-Hot).
    Fit sur X_train uniquement.
    """
    X_train, X_test = X_train.copy(), X_test.copy()

    def is_binary(series):
        unique_vals = set(series.dropna().unique())
        return unique_vals.issubset({0, 1, 0.0, 1.0})

    binary_cols = [c for c in X_train.select_dtypes(include=[np.number]).columns
                   if is_binary(X_train[c])]
    num_cols    = [c for c in X_train.select_dtypes(include=[np.number]).columns
                   if c not in binary_cols]

    print(f"[scale] Colonnes continues à normaliser : {len(num_cols)}")
    print(f"[scale] Colonnes binaires (exclues du scaling) : {len(binary_cols)}")

    scaler = StandardScaler()
    X_train[num_cols] = scaler.fit_transform(X_train[num_cols])
    X_test[num_cols]  = scaler.transform(X_test[num_cols])

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(scaler, save_path)
        print(f"[scale] Scaler sauvegardé → {save_path}")

    return X_train, X_test, scaler


# ─────────────────────────────────────────────────────────
# ÉTAPE 10 : VÉRIFICATION FINALE
# ─────────────────────────────────────────────────────────

def check_preprocessed(X_train: pd.DataFrame, X_test: pd.DataFrame):
    """Vérifie l'absence de NaN, colonnes non numériques et désalignement."""
    issues = []

    non_num_train = X_train.select_dtypes(exclude=[np.number]).columns.tolist()
    non_num_test  = X_test.select_dtypes(exclude=[np.number]).columns.tolist()
    if non_num_train:
        issues.append(f"  ⚠ Colonnes non numériques dans X_train : {non_num_train}")
    if non_num_test:
        issues.append(f"  ⚠ Colonnes non numériques dans X_test  : {non_num_test}")

    nan_train = X_train.isnull().sum().sum()
    nan_test  = X_test.isnull().sum().sum()
    if nan_train > 0:
        issues.append(f"  ⚠ NaN résiduels dans X_train : {nan_train}")
    if nan_test > 0:
        issues.append(f"  ⚠ NaN résiduels dans X_test  : {nan_test}")

    if list(X_train.columns) != list(X_test.columns):
        issues.append("  ⚠ X_train et X_test n'ont pas les mêmes colonnes !")

    if issues:
        print("\n[check] Problèmes détectés :")
        for issue in issues:
            print(issue)
    else:
        print("\n[check] ✅ Toutes les vérifications sont OK !")
        print(f"  X_train : {X_train.shape} | X_test : {X_test.shape}")
        print(f"  Aucun NaN, aucune colonne non numérique.")


# ─────────────────────────────────────────────────────────
# PIPELINE COMPLET
# ─────────────────────────────────────────────────────────

def run_full_preprocessing(
    raw_path: str = "../data/raw/data.csv",
    output_dir: str = "../data/train_test",
    models_dir: str = "../models",
    save_cleaned: str = "../data/processed/data_cleaned.csv",
    test_size: float = 0.2,
    random_state: int = 42,
):
    """
    Pipeline complet de preprocessing :
    chargement → nettoyage → feature engineering → split →
    traitement post-split → encodage → normalisation → sauvegarde.
    """
    print("=" * 60)
    print("PIPELINE DE PREPROCESSING - RETAIL ML")
    print("=" * 60)

    # 0. Chargement
    path = Path(raw_path)
    if path.suffix in [".xlsx", ".xlsm"]:
        df = pd.read_excel(raw_path)
    else:
        df = pd.read_csv(raw_path)
    print(f"\n[0] Données chargées : {df.shape}")

    # 1. Nettoyage brut
    print("\n[1] Nettoyage brut...")
    df = clean_raw_data(df)

    # 2. Feature Engineering
    print("\n[2] Feature Engineering...")
    df = engineer_features(df)

    # Sauvegarde du CSV nettoyé (avant split)
    if save_cleaned:
        Path(save_cleaned).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(save_cleaned, index=False)
        print(f"    Données nettoyées sauvegardées → {save_cleaned}")

    # 3. Split X / y + Train / Test
    print("\n[3] Séparation X/y et split Train/Test...")
    X_train, X_test, y_train, y_test = split_data(
        df, target_col="Churn", test_size=test_size, random_state=random_state
    )

    # 4. Suppression colonnes > 50% manquants
    print("\n[4] Suppression colonnes à fort taux de manquants...")
    X_train, X_test = drop_high_missing_cols(X_train, X_test, threshold=0.5)

    # 5. Correction valeurs aberrantes (double sécurité post-split)
    print("\n[5] Correction valeurs aberrantes...")
    X_train, X_test = fix_outliers(X_train, X_test)

    # 6. Imputation numériques (fit sur X_train uniquement)
    print("\n[6] Imputation valeurs manquantes numériques...")
    X_train, X_test = impute_numeric(
        X_train, X_test,
        strategy_map={
            "Age":          "median",
            "Satisfaction": "median",
            "MonetaryTotal":"median",
        }
    )

    # 7. Encodage ordinal
    print("\n[7] Encodage ordinal...")
    X_train, X_test = encode_ordinal(X_train, X_test)

    # 8. One-Hot Encoding
    print("\n[8] One-Hot Encoding...")
    X_train, X_test = encode_one_hot(X_train, X_test)

    # 9. Target Encoding Country
    print("\n[9] Target Encoding Country...")
    X_train, X_test = encode_country_target(X_train, X_test, y_train)

    #  Suppression du Data Leakage
    X_train, X_test = remove_data_leakage(X_train, X_test)

    X_train, X_test = remove_highly_correlated_features(X_train, X_test, threshold=0.8)
    # 10. Normalisation (fit sur X_train uniquement)
    print("\n[10] Normalisation (StandardScaler)...")
    scaler_path = str(Path(models_dir) / "scaler.joblib")
    X_train, X_test, scaler = scale_features(X_train, X_test, save_path=scaler_path)

    # 11. Vérification finale
    print("\n[11] Vérification finale...")
    check_preprocessed(X_train, X_test)

    # 12. Sauvegarde splits
    print("\n[12] Sauvegarde des splits...")
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    X_train.to_csv(f"{output_dir}/X_train.csv", index=False)
    X_test.to_csv( f"{output_dir}/X_test.csv",  index=False)
    y_train.to_csv(f"{output_dir}/y_train.csv", index=False)
    y_test.to_csv( f"{output_dir}/y_test.csv",  index=False)
    print(f"    Splits sauvegardés dans '{output_dir}/'")

    print("\n" + "=" * 60)
    print("PREPROCESSING TERMINÉ ✅")
    print(f"  X_train : {X_train.shape}  |  X_test : {X_test.shape}")
    print("=" * 60)

    return X_train, X_test, y_train, y_test


# ─────────────────────────────────────────────────────────
# POINT D'ENTRÉE
# ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    raw = sys.argv[1] if len(sys.argv) > 1 else "../data/raw/data.csv"
    run_full_preprocessing(raw_path=raw)