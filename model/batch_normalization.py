import numpy as np
from typing import Tuple, List


class Solution:
    def batch_norm(self, x: List[List[float]], gamma: List[float], beta: List[float],
                   running_mean: List[float], running_var: List[float],
                   momentum: float, eps: float, training: bool) -> Tuple[List[List[float]], List[float], List[float]]:
        # During training: normalize using batch statistics, then update running stats
        # During inference: normalize using running stats (no batch stats needed)
        # Apply affine transform: y = gamma * x_hat + beta
        # Return (y, running_mean, running_var), all rounded to 4 decimals as lists
        x = np.asarray(x)
        running_mean = np.asarray(running_mean)
        running_var = np.asarray(running_var)
        
        if len(x.shape)==2: # (batch_size, features)
            mean = np.mean(x, axis=0) # mean across Batch
            var = np.var(x, axis=0) # variance across Batch
        if training==True:
            x_hat = (x - mean)/(var + eps)**0.5
            y = np.round(x_hat * gamma + beta, 4)        
            running_mean = np.round((1.0-momentum)*running_mean+momentum*mean,4)
            running_var = np.round((1.0-momentum)*running_var+momentum*var,4)
        else: # During Inference
            y = np.round((x - running_mean)/(running_var + eps)**0.5, 4)


        return y, running_mean, running_var



        pass
