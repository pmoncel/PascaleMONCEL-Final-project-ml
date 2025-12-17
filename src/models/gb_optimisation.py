from scipy.stats import randint
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import Pipeline


def gb_optimisation(X_train, X_test, y_train, y_test, preprocessor):
    """
    Recherche des meilleurs hyperparamètres pour
    le modèle de Gradient Boosting, par RandomizedSearchCV
    target : priority (low/medium/hight)
    """
    print("\nOptimisation des hyperparamètres")

     # Créer un Pipeline (preprocessor + modèle)
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('model', GradientBoostingClassifier(random_state=42))
    ])

    # Préfixer les paramètres avec 'model__'
    # Car ils sont maintenant dans le pipeline
    param_dist = {
        "model__n_estimators": randint(50, 300),
        "model__learning_rate": [0.01, 0.05, 0.1, 0.2],
        "model__max_depth": randint(3, 6),
        "model__min_samples_split": randint(2, 10),
    }

    # Passer le pipeline (pas le modèle seul)
    random_search = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=param_dist,
        n_iter=10,
        cv=3,
        scoring="accuracy",
        n_jobs=-1,
        random_state=42,
        verbose=2,
    )

    # Fitter sur X_train NON transformé
    # Le pipeline va s'occuper du preprocessing
    random_search.fit(X_train, y_train)

    # Afficher les résultats
    print("-----------------------------")
    print("Meilleurs paramètres:", random_search.best_params_)
    print("Meilleur score:", random_search.best_score_)
    best_pipeline = random_search.best_estimator_
    print(best_pipeline)
    print("-----------------------------")
    
    return best_pipeline
