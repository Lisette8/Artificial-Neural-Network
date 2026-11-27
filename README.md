# TP Deep Learning : Classification Fashion-MNIST avec un réseau de neurones (TensorFlow / Keras)

Ce projet est un TP de deep learning. On construit un réseau de neurones entièrement connecté (ANN) avec TensorFlow/Keras pour classer des images de vêtements du dataset **Fashion-MNIST**, puis on compare plusieurs architectures.

## Contenu du projet

```
.
├── main.py             # Modèle de base : chargement, entraînement, évaluation, prédiction
├── experiments.py      # Section 3 : comparaison des 4 architectures (A, B, C, D)
├── data/               # Fichiers de données du TP
├── requirements.txt    # Dépendances Python
└── README.md
```

## Ce qui a été fait

### 1. Modèle de base (`main.py`)

- Chargement de Fashion-MNIST (60 000 images d'entraînement, 10 000 de test, 10 classes, images 28×28).
- Visualisation d'exemples d'images.
- Prétraitement : normalisation des pixels entre 0 et 1, puis aplatissement des images en vecteurs de 784 valeurs.
- Architecture : `784 → 128 → 64 → 10` (activation ReLU, softmax en sortie).
- Compilation avec l'optimiseur Adam et la loss `sparse_categorical_crossentropy`.
- Entraînement sur 10 epochs (batch size 32, 20 % des données en validation).
- Courbes d'apprentissage, évaluation sur le jeu de test et exemple de prédiction.

### 2. Expériences sur l'architecture (`experiments.py`)

Quatre architectures sont entraînées avec exactement les mêmes hyperparamètres (seule l'architecture change) :

| Modèle | Architecture | Paramètres entraînables |
|---|---|---|
| A | 784 → 32 → 10 | 25 450 |
| B | 784 → 128 → 10 | 101 770 |
| C | 784 → 128 → 64 → 10 | 109 386 |
| D | 784 → 256 → 128 → 64 → 10 | 242 762 |

Pour chaque modèle, le script compare :

- la **test accuracy**,
- le **temps d'entraînement**,
- le **nombre de paramètres entraînables**,
- le **surapprentissage** (écart entre accuracy d'entraînement et de validation, courbes de loss).

Le script affiche un tableau récapitulatif et génère trois images : `learning_curves.png`, `loss_curves.png` et `comparison.png`.


## Télécharger le projet

**Option 1 : avec Git**

```bash
git clone https://github.com/<ton-utilisateur>/<nom-du-repo>.git
cd <nom-du-repo>
```


## Technologies

- Python
- TensorFlow / Keras
- NumPy, Matplotlib, Pandas
