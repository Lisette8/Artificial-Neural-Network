import time
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input


SEED = 42
EPOCHS = 10
BATCH_SIZE = 32

(X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()

X_train = X_train.astype("float32").reshape(-1, 784) / 255.0
X_test = X_test.astype("float32").reshape(-1, 784) / 255.0



# Les 4 architectures à comparer
architectures = {
    "Model A (784-32-10)": [32],
    "Model B (784-128-10)": [128],
    "Model C (784-128-64-10)": [128, 64],
    "Model D (784-256-128-64-10)": [256, 128, 64],
}


def build_model(hidden_units):
    layers = [Input(shape=(784,))]
    for units in hidden_units:
        layers.append(Dense(units, activation="relu"))
    layers.append(Dense(10, activation="softmax"))
    return Sequential(layers)



# Entraînement
results = {}
histories = {}

for name, hidden_units in architectures.items():
    print("\n" + "=" * 60)
    print(name)
    print("=" * 60)

    # Commencer proprement pour chaque modèle
    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(SEED)

    model = build_model(hidden_units)
    model.summary()

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    start = time.time()
    history = model.fit(
        X_train,
        y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_split=0.2,
        verbose=1,
    )
    training_time = time.time() - start

    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)

    train_acc = history.history["accuracy"][-1]
    val_acc = history.history["val_accuracy"][-1]
    train_loss = history.history["loss"][-1]
    val_loss = history.history["val_loss"][-1]

    trainable_params = int(
        np.sum([np.prod(w.shape) for w in model.trainable_weights])
    )

    results[name] = {
        "test_acc": test_acc,
        "test_loss": test_loss,
        "time": training_time,
        "params": trainable_params,
        "train_acc": train_acc,
        "val_acc": val_acc,
        "acc_gap": train_acc - val_acc,
        "loss_gap": val_loss - train_loss,
    }
    histories[name] = history.history



# Résultats ( Tableau )
print("\n" + "=" * 100)
print("RÉSULTATS")
print("=" * 100)
header = (
    f"{'Model':<30}{'Test acc':>10}{'Time (s)':>10}{'Params':>10}"
    f"{'Train acc':>11}{'Val acc':>9}{'Acc gap':>9}{'Loss gap':>10}"
)
print(header)
print("-" * len(header))
for name, r in results.items():
    print(
        f"{name:<30}{r['test_acc']:>10.4f}{r['time']:>10.1f}{r['params']:>10,}"
        f"{r['train_acc']:>11.4f}{r['val_acc']:>9.4f}"
        f"{r['acc_gap']:>9.4f}{r['loss_gap']:>10.4f}"
    )


# Représentation graphique
# 1- Courbes de training
fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharey=True)
for ax, (name, h) in zip(axes.ravel(), histories.items()):
    ax.plot(h["accuracy"], label="Training accuracy")
    ax.plot(h["val_accuracy"], label="Validation accuracy")
    ax.set_title(name)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    ax.grid()
    ax.legend()
plt.tight_layout()
plt.savefig("learning_curves.png", dpi=150)
plt.show()

# 2- Courbe de loss ( le sous ou sur apprentissage )
fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharey=True)
for ax, (name, h) in zip(axes.ravel(), histories.items()):
    ax.plot(h["loss"], label="Training loss")
    ax.plot(h["val_loss"], label="Validation loss")
    ax.set_title(name)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.grid()
    ax.legend()
plt.tight_layout()
plt.savefig("loss_curves.png", dpi=150)
plt.show()

# 3- Comparaison globale : accuracy, temps, paramètres
labels = ["A", "B", "C", "D"]
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

axes[0].bar(labels, [r["test_acc"] for r in results.values()])
axes[0].set_title("Test accuracy")
axes[0].set_ylim(0.8, 0.9)

axes[1].bar(labels, [r["time"] for r in results.values()])
axes[1].set_title("Training time (s)")

axes[2].bar(labels, [r["params"] for r in results.values()])
axes[2].set_title("Trainable parameters")

plt.tight_layout()
plt.savefig("comparison.png", dpi=150)
plt.show()
