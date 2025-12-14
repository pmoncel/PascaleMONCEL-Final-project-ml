from scipy.stats import randint
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import RandomizedSearchCV


def gb_optimisation(X_train, X_test, y_train, y_test, preprocessor):
    """
    Recherche des meilleurs hyperparamètres pour
    le modèle de Gradient Boosting, par RandomizedSearchCV
    target : priority (low/medium/hight)
    """
    print("\nOptimisation des hyperparamètres")

    # Appliquer le preprocessing
    X_train_processed = preprocessor.fit_transform(X_train)

    # Création et initialisation du modèle
    model = GradientBoostingClassifier(random_state=42)

    # Grille des paramètres aléatoires (best practices)
    param_dist = {
        "n_estimators": randint(50, 300),
        "learning_rate": [0.01, 0.05, 0.1, 0.2],
        "max_depth": randint(3, 6),
        "min_samples_split": randint(2, 10),
    }

    # Recherche aléatoire : 
    # 10 itérations sur une validation croisée à 3 sous-ensembles = 30
    random_search = RandomizedSearchCV(
        estimator=model,
        param_distributions=param_dist,
        n_iter=10,
        cv=3,
        scoring="accuracy",
        n_jobs=-1,
        random_state=42,
        verbose=2,
    )

    # Entrainement du modèle
    random_search.fit(X_train_processed, y_train)

    # Evaluation
    # Afficher les résultats
    print("-----------------------------")
    print("Meilleurs paramètres:", random_search.best_params_)
    print("Meilleur score:", random_search.best_score_)
    best_model = random_search.best_estimator_
    print(best_model)
    print("-----------------------------")
    return model
