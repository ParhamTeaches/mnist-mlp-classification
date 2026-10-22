import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

print("Loading MNIST dataset...")

mnist = fetch_openml(
    "mnist_784",
    version=1,
    as_frame=False
)

x = mnist.data
y = mnist.target.astype(int)

print("\nX shape=>")
print(x.shape)

print("\nY shape=>")
print(y.shape)\

x = x / 255.0

x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)
print("\n")
print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)

model = MLPClassifier(
    hidden_layer_sizes=(128, 64),
    activation="relu",
    solver="adam",
    learning_rate_init=0.001,
    max_iter=20,
    batch_size=128,
    random_state=42,
    verbose=True
)

print("training the model...")

model.fit(x_train, y_train)

y_prediction = model.predict(x_test)

accuracy = accuracy_score(y_test, y_prediction)

print("\nAccuracy:")
print(accuracy)

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_prediction))

print("\nClassification Report:")
print(classification_report(y_test, y_prediction))