# Prédiction de la Priorité des Tickets Support

Projet de Machine Learning visant à prédire automatiquement la priorité des tickets de support client (low/medium/high) en utilisant différents algorithmes de classification.

## Contexte

Dans un contexte de support client, classer automatiquement les tickets selon leur priorité permet d'optimiser les temps de réponse et d'améliorer la satisfaction client.
Ce projet explore plusieurs modèles de ML pour identifier la meilleure approche.

## Dataset

- **Source** : [Kaggle - Support Ticket Priority Dataset (50K)](https://www.kaggle.com/datasets/albertobircoci/support-ticket-priority-dataset-50k)
- **Taille** : 50 000 tickets
- **Variable cible** : `priority` (low, medium, high)
- **Features** : Informations sur les tickets (type, catégorie, canal, temps de résolution, satisfaction client, etc.)

## Structure du Projet

```
00-Final-project-ml/
│
├── main.py                          # Point d'entrée principal
├── 01-eda.ipynb                     # Analyse exploratoire des données
├── requirements.txt                 # Dépendances Python
│
├── src/
│   ├── data_loader.py               # Chargement des données depuis Kaggle
│   ├── preprocessing.py             # Prétraitement et feature engineering
│   │
│   └── models/
│       ├── regression_logistique.py # Modèle 1 : Régression logistique
│       ├── decision_tree.py         # Modèle 2 : Arbre de décision
│       ├── random_forest.py         # Modèle 3 : Forêt aléatoire
│       ├── gradient_boosting.py     # Modèle 4 : Gradient Boosting (optimal)
│       └── gb_optimisation.py       # Optimisation par validation croisée
│
└── outputs/                         # Graphiques et résultats
    └── gradient_boosting_learning_curve.png
```

## Utilisation

### Lancer le projet complet
python main.py


Le script va :
1. Télécharger automatiquement le dataset depuis Kaggle
2. Prétraiter les données
3. Entraîner le modèle Gradient Boosting optimisé
4. Générer les métriques d'évaluation
5. Sauvegarder les visualisations dans `outputs/`

### Tester d'autres modèles

Dans main.py, vous pouvez décommenter les lignes correspondantes :


# Modèle 1 - Régression logistique
regression_logistique(X_train, X_test, y_train, y_test, preprocessor)

# Modèle 2 - Arbre de décision
decision_tree(X_train, X_test, y_train, y_test, preprocessor)

# Modèle 3 - Forêt aléatoire
random_forest(X_train, X_test, y_train, y_test, preprocessor)


### Analyse exploratoire
Notebook 01-eda.ipynb


## Modèles Testés

| Modèle                  | Accuracy | Description |
|------------------------|----------|-------------|
| Régression logistique  | 86.61%   | Modèle de base linéaire |
| Arbre de décision      | 91.96%   | Modèle simple non-linéaire |
| Forêt aléatoire        | 90.28%   | Ensemble de décision trees |
| **Gradient Boosting**  | **97.44%** | **Modèle optimal avec hyperparamètres optimisés** |

**Hyperparamètres optimaux** (obtenus par validation croisée) :
- `n_estimators`: 253
- `learning_rate`: 0.2
- `max_depth`: 3
- `min_samples_split`: 2

**Performances** :
- **Accuracy globale** : 97.44%
- **F1-Score par classe** :
  - Low priority : 99%
  - Medium priority : 96%
  - High priority : 96%
