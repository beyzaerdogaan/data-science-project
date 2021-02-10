#!/usr/bin/env python
# -*- coding: utf-8-*-
# python 2.7.16
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def sigmoid(x):
    return 1 / (1 + np.exp(-0.005*x))


def sigmoid_derivative(x):
    return 0.005 * x * (1 - x)


def read_and_divide_into_train_and_test(csv_file):
    # TODO
    missing_val = "?"
    breast_cancer_file = pd.read_csv(csv_file, na_values=missing_val)
    med = breast_cancer_file["Bare_Nuclei"].median()
    breast_cancer_file["Bare_Nuclei"].fillna(med, inplace=True)
    training_inputs = breast_cancer_file.iloc[:560, 1:10].to_numpy().astype(int)
    test_inputs = breast_cancer_file.iloc[560:, 1:10].to_numpy().astype(int)
    training_labels = breast_cancer_file.iloc[:560, 10:].to_numpy().astype(int)
    test_labels = breast_cancer_file.iloc[560:, 10:].to_numpy().astype(int)
    train_inp_cor = breast_cancer_file.iloc[:560, 1:10].corr(method="pearson")
    fig, axs = plt.subplots()
    im = axs.imshow(train_inp_cor.to_numpy())
    plt.colorbar(im)
    axs.set_xticks(np.arange(len(train_inp_cor.columns)))
    axs.set_yticks(np.arange(len(train_inp_cor.columns)))
    axs.set_xticklabels(train_inp_cor.columns)
    axs.set_yticklabels(train_inp_cor.columns)
    plt.setp(axs.get_xticklabels(), rotation=45, ha="right")
    for i in range(len(train_inp_cor.columns)):
        for j in range(len(train_inp_cor.columns)):
            axs.text(j, i, round(train_inp_cor.to_numpy()[i, j], 2),
                           ha="center", va="center", color="w")
    axs.set_title("Correlation")
    plt.show()
    return training_inputs, training_labels, test_inputs, test_labels
    

def run_on_test_set(test_inputs, test_labels, weights):
    tp = 0
    # calculate test_predictions
    test_predictions = np.dot(test_inputs, weights)
    # TODO map each prediction into either 0 or 1
    test_predictions = np.round(sigmoid(test_predictions)).astype(int)
    test_outputs = test_labels
    for predicted_val, label in zip(test_predictions, test_outputs):
        if predicted_val == label:
            tp = tp + 1
    # accuracy = tp_count / total number of samples
    accuracy = tp / float(139)
    return accuracy


def plot_loss_accuracy(accuracy_array, loss_array):
    # TODO plot loss and accuracy change for each iteration
    plt.plot(accuracy_array)
    plt.title("accuracy")
    plt.show()
    plt.plot(loss_array)
    plt.title("loss")
    plt.show()


def main():
    csv_file = './breast-cancer-wisconsin.csv'
    iteration_count = 2500
    np.random.seed(1)
    weights = 2 * np.random.random((9, 1)) - 1
    accuracy_array = []
    loss_array = []
    training_inputs, training_labels, test_inputs, test_labels = read_and_divide_into_train_and_test(csv_file)

    for iteration in range(iteration_count):
        input = training_inputs
        # calculate outputs
        outputs = input.dot(weights)
        outputs = sigmoid(outputs)
        # calculate loss
        loss = training_labels - outputs
        # calculate tuning
        tuning = loss * sigmoid_derivative(outputs)
        # update weights
        weights = weights + np.dot(input.T, tuning)
        # run_on_test_set
        run_on_test_set(test_inputs, test_labels, weights)
    # you are expected to add each accuracy value into accuracy_array
        accuracy_array.append(run_on_test_set(test_inputs, test_labels, weights))
    # you are expected to find mean value of the loss for plotting purposes and add each value into loss_array
        loss_array.append(np.mean(loss))
    plot_loss_accuracy(accuracy_array, loss_array)


if __name__ == '__main__':
    main()
