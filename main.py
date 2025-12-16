from src.data_loader import load_data
from src.models.decision_tree import decision_tree
from src.models.gb_optimisation import gb_optimisation
from src.models.gradient_boosting import gradient_boosting
from src.models.random_forest import random_forest
from src.models.regression_logistique import regression_logistique
from src.preprocessing import preprocess_data


def main():
    try:
        data = load_data()
        print("Données originales chargées :")
        print(data.head())

        # FEATURE ENGINEERING : Activer/Désactiver
        # Passez feature_engineering=True ou False (défaut car pas d'amélioration)
        X_train, X_test, y_train, y_test, preprocessor = preprocess_data(data, feature_engineering=False)

        print("\nDonnées prétraitées :")
        print(f"X_train: {X_train.shape}, X_test: {X_test.shape}")
        print(f"y_train: {y_train.shape}, y_test: {y_test.shape}")
        print("\nPréprocesseur prêt (non encore fit).")
        print("----------------------------------------")
        print("---- MODELE EN ENTRAINEMENT ≈ 40'' -----")

        # Modèle 1 - Régression logistique ( ≈ 0.8661)
        # regression_logistique(X_train, X_test, y_train, y_test, preprocessor)

        # Modèle 2 - Arbre de décision basique ( ≈ 0.9196)
        # decision_tree(X_train, X_test, y_train, y_test, preprocessor)

        # Modèle 3 - Forêt aléatoire ( ≈ 0.9028)
        # random_forest(X_train, X_test, y_train, y_test, preprocessor)

        # Modèle 4 - 🏆 Gradient Boosting ( ≈ 0.9406 au premier tour sans optimisation)
        # Etant donnée le résultat de ce dernier modèle,
        # voyons s'il est améliorable par RandomizedSearchCV.
        # gb_optimisation (X_train, X_test, y_train, y_test, preprocessor)
        # hyperparamètres optimaux :
        # {'learning_rate': 0.2, 'max_depth': 3, 'min_samples_split': 2, 'n_estimators': 253}
        # Avec ces hyperparamètres, le résultat est : 🏆0.9744🏆

        gradient_boosting(X_train, X_test, y_train, y_test, preprocessor)

        conclusion = "-----------------------------"
        conclusion += "\nConclusions: Accuracy = 0.9744"
        conclusion += (
            "\nDéséquilibre des classes, F1-score sera une métrique importante"
        )
        conclusion += "\nPrédictions - low    -> 99%"
        conclusion += "\nPrédictions - medium -> 96%"
        conclusion += "\nPrédictions - hight  -> 96%"
        conclusion += "\n-----------------------------"
        print(conclusion)

    except Exception as e:
        print(f"Erreur : {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
