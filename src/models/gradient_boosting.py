import os

import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    log_loss,
)


def gradient_boosting(X_train, X_test, y_train, y_test, preprocessor):
    """
    Entraîne et évalue un modèle de Gradient Boosting Classifier.
    target : priority (low/medium/hight)
    """
    print("\nModèle = gradient boosting")

    # Appliquer le preprocessing sur les données
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    # Vérification des données transformées
    print("\nDonnées après preprocessing:")
    print(X_train_processed[:5, :5])
    print("\n\nItérations d'entrainement")

    # Création du modèle
    # Initialisation du modèle
    # Les hyperparamètres sont issus de la XValidation
    # faite grâce à gb_cross_validation.py :
    # {'learning_rate': 0.2, 'max_depth': 3, 'min_samples_split': 2, 'n_estimators': 253}
    model = GradientBoostingClassifier(
        n_estimators=253,  # Nombre d'arbres (itérations)
        learning_rate=0.2,  # Taux d'apprentissage
        max_depth=3,  # Profondeur max des arbres
        min_samples_split=2,  # Nb min d'échantillons pour diviser un nœud
        random_state=42,  # Graine aléatoire pour la reproductibilité
        verbose=1,
    )

    # Entrainement du modèle
    model.fit(X_train_processed, y_train)
    # Prediction sur le jeu d'entrainement
    print("\n----- MODELE EN PREDICTION ≈ 10'' ------")
    y_train_pred = model.predict(
        X_train_processed,
    )
    # Extraire les erreurs de train
    train_scores = model.train_score_

    # Prediction sur le jeu de test et récupération du
    # score de test à chaque itération (->visualisation)
    test_scores = []
    y_pred = None
    for i, y_pred_proba in enumerate(model.staged_predict_proba(X_test_processed)):
        test_scores.append(log_loss(y_test, y_pred_proba))
        if i == len(train_scores) - 1:  # Dernière itération (253e arbre)
            y_pred = y_pred_proba.argmax(axis=1)

    # Evaluation
    train_accuracy = accuracy_score(y_train, y_train_pred)
    accuracy = accuracy_score(y_test, y_pred)

    print("\n-------------- METRIQUES ---------------")
    print(f"Accuracy TRAIN : {train_accuracy}")
    print(f"Accuracy TEST : {accuracy}")
    print("\nRapport de Classification:")
    print(classification_report(y_test, y_pred))
    print("\nMatrice de confusion:")
    print(confusion_matrix(y_test, y_pred))
    print("-----------------------------")

    # Visualisation TRAIN vs TEST
    # Création d'un graphique représentant la Courbe d'apprentissage
    os.makedirs("outputs", exist_ok=True)

    plt.figure(figsize=(10, 6))
    plt.plot(
        range(1, len(train_scores) + 1),
        train_scores,
        label="Train Score",
        linewidth=2,
        color="blue",
    )
    plt.plot(
        range(1, len(test_scores) + 1),
        test_scores,
        label="Test Score",
        linewidth=2,
        color="orange",
    )
    plt.xlabel("Nombre d'itérations (arbres)", fontsize=12)
    plt.ylabel("Train Loss", fontsize=12)
    plt.title(
        "Courbe d'apprentissage Train vs Test - Gradient Boosting",
        fontsize=14,
        fontweight="bold",
    )
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    output_path = "outputs/gradient_boosting_courbe_apprentissage.png"
    plt.savefig(output_path)
    plt.close()

    return model
