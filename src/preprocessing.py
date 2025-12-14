import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def preprocess_data(data):
    """
    Prétraite les données pour l'entraînement.
    Split des données et création du pipeline de preprocessing.

    Transformations effectuées :
    - Suppression des colonnes catégorielles encodées et identifiant du ticket
    - Encodage de la variable cible 'priority' (low=0, medium=1, high=2)
    - Imputation des valeurs manquantes de 'customer_sentiment' par 'neutral'
    - Split train/test (80/20)
    - Création d'un pipeline de preprocessing :
      Variables numériques : imputation (médiane) + standardisation
      Variables catégorielles : imputation (mode(=most_frequent)) + one-hot encoding

    Parameters
    ----------
    data : pandas.DataFrame des données brutes avec toutes les colonnes
           (y compris 'priority')

    Returns
    -------
    tuple
        (X_train, X_test, y_train, y_test, preprocessor)
        - X_train : DataFrame des features de train (non transformées)
        - X_test : DataFrame des features de test (non transformées)
        - y_train : les labels de train (encodés 0, 1, 2)
        - y_test : les labels de test (encodés 0, 1, 2)
        - preprocessor : ColumnTransformer (non fitté) pour transformer X_train/X_test

    Example
    -------
    >>> df = pd.read_csv('data.csv')
    >>> X_train, X_test, y_train, y_test, preprocessor = preprocess_data(df)
    >>> preprocessor.fit(X_train)
    >>> X_train_transformed = preprocessor.transform(X_train)
    """

    # Copie du dataframe pour ne pas modifier l'original
    data = data.copy()
    # Supp. les colonnes inutiles (cf 01-eda.ipynb)
    data = data.drop(
        columns=[
            "ticket_id",
            "day_of_week_num",
            "company_size_cat",
            "industry_cat",
            "customer_tier_cat",
            "region_cat",
            "product_area_cat",
            "booking_channel_cat",
            "reported_by_role_cat",
            "customer_sentiment_cat",
            "priority_cat",
        ]
    )
    # Encodage de la target
    data["priority"] = data["priority"].map({"low": 0, "medium": 1, "high": 2})

    # Valeurs manquantes de la colonne 'customer_sentiment'
    data["customer_sentiment"] = data["customer_sentiment"].fillna("neutral")

    ################################################
    """
    # On pourra noter qu'avec un équilibrage des classes,
    # le resultat obtenu sera le même pour le modèle gagnant.
    #### Test avec équilibrage des classes ####
    # Nombre minimum de tickets par priority
    min_count = data['priority'].value_counts().min()
    # Créer un dataset équilibré
    df_balanced = pd.DataFrame()

    for priority_level in data['priority'].unique():
        subset = data[data['priority'] == priority_level]
        # Échantillonner aléatoirement le même nombre que la classe minoritaire
        if len(subset) > min_count:
            subset_balanced = subset.sample(n=min_count, random_state=42)
        else:
            subset_balanced = subset

        df_balanced = pd.concat([df_balanced, subset_balanced], ignore_index=True)

    # Séparation des variables features et target
    X_features = df_balanced.drop(columns=['priority'])
    y_target = df_balanced['priority']
    """
    ###############################################

    # Séparation des variables features et target
    X_features = data.drop(columns=["priority"])
    y_target = data["priority"]

    # Séparation des variables catégorielles et numériques
    X_features_cat = X_features.select_dtypes(include=["object"]).columns.tolist()
    X_features_num = X_features.select_dtypes(include=["number"]).columns.tolist()

    # Split des données en train et test
    # (stratify=y_target pour garder les mêmes proportion sur la target entre train et test)
    X_train, X_test, y_train, y_test = train_test_split(
        X_features, y_target, test_size=0.2, random_state=42, stratify=y_target
    )

    # Pipeline pour les variables numériques : imputation + standardisation
    numerical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    # Pipeline pour les variables catégorielles : imputation + encodage one-hot
    # (handle_unknown => si un nouveau secteur apparait en PROD ça ne plantera pas)
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    # Pipeline de preprocessing
    # Combinaison des transformateurs pour chaque type de variable
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numerical_transformer, X_features_num),
            ("cat", categorical_transformer, X_features_cat),
        ]
    )

    return (X_train, X_test, y_train, y_test, preprocessor)
