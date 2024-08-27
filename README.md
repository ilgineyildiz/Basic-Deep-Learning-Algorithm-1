# Iris Dataset Classification Project
This project demonstrates the use of both a neural network using TensorFlow/Keras and an MLPClassifier from scikit-learn to classify the Iris dataset.

## Project Overview
The Iris dataset is a well-known dataset in machine learning, consisting of 150 samples of iris flowers with four features each (sepal length, sepal width, petal length, and petal width). The goal is to classify the flowers into three species: Setosa, Versicolour, and Virginica.

## 1. Data Preparation
The Iris dataset is loaded using sklearn.datasets.load_iris.
The dataset is split into training, validation, and test sets using train_test_split.

## 2. Data Preprocessing
The data is standardized using StandardScaler to improve the performance of the neural network.

## 3. Neural Network with TensorFlow/Keras
A simple neural network is built using TensorFlow/Keras with three layers:

An input layer with 10 neurons and ReLU activation.
A hidden layer with 50 neurons and ReLU activation.
An output layer with 3 neurons and softmax activation.
The model is compiled with the Adam optimizer and trained using sparse categorical crossentropy loss.

Training progress is visualized by plotting the training and validation loss over 100 epochs.

## 4. MLPClassifier with Scikit-learn
An MLPClassifier from scikit-learn is trained with a hidden layer size of 100 neurons, using the L-BFGS solver and ReLU activation function.

## 5. Evaluation
The trained models are evaluated on the test set.
The performance is measured using the confusion matrix and classification report.
The confusion matrix is visualized using ConfusionMatrixDisplay from scikit-learn.

### Dependencies
Python 3.x
TensorFlow
scikit-learn
Matplotlib
NumPy

### Running the Project
Clone the repository:
git clone https://github.com/yourusername/iris-classification.git
cd iris-classification

Install the required dependencies:
pip install -r requirements.txt

Run the Python script:
python iris_classification.py

### Results
The project includes visualizations of the model loss over epochs and the confusion matrix, helping to understand the performance of both the TensorFlow/Keras model and the MLPClassifier.

