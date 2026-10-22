# Handwritten Digit Classification Using MLP

A Python machine learning project developed as part of AI laboratory coursework. It uses a multilayer perceptron (MLP) to classify handwritten digits from the MNIST dataset and explores how different activation functions and learning rates affect model performance.

## Dataset

- MNIST handwritten digit dataset, loaded using `fetch_openml`.
- Pixel values normalized to the range [0, 1].
- Stratified 80/20 training and test split with a fixed random seed.

## Model

The classifier is implemented using scikit-learn's `MLPClassifier`.

- Hidden layers: 128 and 64 neurons
- Optimizer: Adam
- Batch size: 128
- Maximum iterations: 20
- Random state: 42

## Experiments

**1. Baseline classification — `mnist_mlp_model.py`**

Trains an MLP classifier and evaluates its performance using accuracy, a confusion matrix, and a classification report.

**2. Activation function comparison — `mnist_activation_comparison.py`**

Compares logistic, tanh, and ReLU activation functions using test accuracy, final training loss, iteration count, and training time.

**3. Learning rate comparison — `mnist_learning_rate_comparison.py`**

Compares initial learning rates of 0.0001, 0.001, and 0.01 using the same evaluation measures.

## Installation

Install the required Python packages:

```bash
pip install -r requirements.txt
```

## Usage

Run each experiment separately:

```bash
python mnist_mlp_model.py
python mnist_activation_comparison.py
python mnist_learning_rate_comparison.py
```

The MNIST dataset is downloaded automatically when needed, so an internet connection is required for the initial download.

## Notes

This is an educational machine learning project intended to explore MLP classification and the effects of selected training hyperparameters.   