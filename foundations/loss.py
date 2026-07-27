import numpy as np
from numpy.typing import NDArray
from math import log

class Solution:

    def binary_cross_entropy(self, y_true: NDArray[np.float64], y_pred: NDArray[np.float64]) -> float:
        # y_true: true labels (0 or 1)
        # y_pred: predicted probabilities
        # Hint: add a small epsilon (1e-7) to y_pred to avoid log(0)
        # return round(your_answer, 4)
        eps = 1e-7
        l = 0
        n=len(y_true)
        for i in range(len(y_true)):
            l = l + y_true[i]* log(y_pred[i] + eps) + (1 - y_true[i]) * (log(1 - y_pred[i] + eps))
        return round((-1/n) * l,4)

        pass

    def categorical_cross_entropy(self, y_true: NDArray[np.float64], y_pred: NDArray[np.float64]) -> float:
        # y_true: one-hot encoded true labels (shape: n_samples x n_classes)
        # y_pred: predicted probabilities (shape: n_samples x n_classes)
        # Hint: add a small epsilon (1e-7) to y_pred to avoid log(0)
        # return round(your_answer, 4)
        eps = 1e-7
        l = 0
        n=len(y_true)
        for i in range(len(y_true)):
               for c in range(len(y_true[0])):
                      l+= y_true[i][c] * log(y_pred[i][c]+ eps)
               
        return round((-1/n) * l,4)



        pass
