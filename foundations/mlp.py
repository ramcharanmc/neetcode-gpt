import numpy as np
from numpy.typing import NDArray
from typing import List


class Solution:
    def forward(self, x: NDArray[np.float64], weights: List[NDArray[np.float64]], biases: List[NDArray[np.float64]]) -> NDArray[np.float64]:
        # x: 1D input array
        # weights: list of 2D weight matrices
        # biases: list of 1D bias vectors
        # Apply ReLU after each hidden layer, no activation on output layer
        # return np.round(your_answer, 5)
        x = np.array(x, dtype=np.float64)
        h = [None] * (len(weights) + 1)
        h[0] = x

        for i in range(len(weights)):
            W = np.array(weights[i], dtype=np.float64)
            b = np.array(biases[i], dtype=np.float64)
            h[i + 1] = np.matmul(h[i], W) + b
            #if i < len(weights) - 1:
            h[i + 1] = np.maximum(0, h[i + 1])

        return np.round(h[-1], 5)
        
        pass

