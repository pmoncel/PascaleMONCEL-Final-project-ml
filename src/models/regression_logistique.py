from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


def regression_logistique(X_train, X_test, y_train, y_test, preprocessor):
    """
    Entraîne et évalue un modèle de régression logistique
    target : priority (low/medium/hight)
    """
    print("\nModèle = Regression Logistique")

    # Appliquer le preprocessing sur les données
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    # Vérification des données transformées
    print(f"\nDonnées après preprocessing:")
    print(X_train_processed[:5, :5])

    # Création du modèle
    model = LogisticRegression(penalty=None, random_state=42)

    # Entrainement du modèle
    model.fit(X_train_processed, y_train)
    # Prediction sur le jeu d'entrainement
    y_train_pred = model.predict(X_train_processed)
    # Prediction sur le jeu de test
    y_pred = model.predict(X_test_processed)

    # Evaluation
    train_accuracy = accuracy_score(y_train, y_train_pred)
    accuracy = accuracy_score(y_test, y_pred)

    print(f"Accuracy (train) : {train_accuracy}")
    print(f"Accuracy : {accuracy}")
    print("\nRapport de Classification:")
    print(classification_report(y_test, y_pred))
    print("\nMatrice de confusion:")
    print(confusion_matrix(y_test, y_pred))
    print("-----------------------------")
    return model
