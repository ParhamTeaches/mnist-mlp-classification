import time
import pandas as pd

from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

print("Loading MNIST dataset...")

mnist = fetch_openml(
    "mnist_784",
    version=1,
    as_frame=False
)

X = mnist.data
y = mnist.target.astype(int)

X = X / 255.0

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

learning_rates = [0.0001, 0.001, 0.01]

results = []

for lr in learning_rates:
    print("\n" + "=" * 60)
    print(f"Training MLP with learning rate = {lr}")
    print("=" * 60)

    model = MLPClassifier(
        hidden_layer_sizes=(128, 64),
        activation="relu",
        solver="adam",
        learning_rate_init=lr,
        max_iter=20,
        batch_size=128,
        random_state=42,
        verbose=False
    )

    start_time = time.time()

    model.fit(X_train, y_train)

    end_time = time.time()

    training_time = end_time - start_time

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)

    final_loss = model.loss_

    iterations = model.n_iter_

    results.append({
        "Learning Rate": lr,
        "Accuracy": accuracy,
        "Final Loss": final_loss,
        "Iterations": iterations,
        "Training Time (seconds)": training_time
    })

    results_df = pd.DataFrame(results)

    print("\nFinal Comparison:")
    print(results_df)