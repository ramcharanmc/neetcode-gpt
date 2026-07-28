import numpy as np
from typing import List


class Solution:
    def forward_and_backward(self,
                              x: List[float],
                              W1: List[List[float]], b1: List[float],
                              W2: List[List[float]], b2: List[float],
                              y_true: List[float]) -> dict:
        # Architecture: x -> Linear(W1, b1) -> ReLU -> Linear(W2, b2) -> predictions
        # Loss: MSE = mean((predictions - y_true)^2)
        #
        # Return dict with keys:
        #   'loss':  float (MSE loss, rounded to 4 decimals)
        #   'dW1':   2D list (gradient w.r.t. W1, rounded to 4 decimals)
        #   'db1':   1D list (gradient w.r.t. b1, rounded to 4 decimals)
        #   'dW2':   2D list (gradient w.r.t. W2, rounded to 4 decimals)
        #   'db2':   1D list (gradient w.r.t. b2, rounded to 4 decimals)
        W1 = np.array(W1)
        W2 = np.array(W2)
        x = np.array(x)
        b1 = np.array(b1)
        b2 = np.array(b2)
        n = len(y_true)
        z1 = np.matmul(x,W1.T) + b1
        a1 = np.maximum(0,z1)
        z2 = np.matmul(a1,W2.T) + b2
        y_hat = z2
        L = np.mean((y_hat - y_true)**2)

        dz2 = 2 * (z2 - y_true) / n
        dW2 = np.outer(dz2, a1)
        db2 = dz2
        da1 = np.dot(dz2, W2)
        # 1[z1 > 0] creates a boolean mask, cast to float (1.0 for True, 0.0 for False)
        relu_grad = (z1 > 0).astype(float)

        # Element-wise multiplication (Hadamard product: ⊙)
        dz1 = da1 * relu_grad
        dW1 = np.outer(dz1, x)
        db1 = dz1
        output = {}
        output['loss'] = float(round(L, 4))
        output['dW1'] = dW1.round(4).tolist()
        output['db1'] = db1.round(4).tolist()
        output['dW2'] = dW2.round(4).tolist()
        output['db2'] = db2.round(4).tolist()

        return output
        pass
