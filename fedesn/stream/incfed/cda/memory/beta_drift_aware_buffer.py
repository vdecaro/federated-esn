import numpy as np
from math import exp, log
import random
from scipy.stats import beta

from .memory_buffer import OverloadedMemoryBuffer


class BetaDriftDetectionMemory(OverloadedMemoryBuffer):

    def __init__(
        self, 
        capacity: int, 
        delta_window: int, 
        sensitivity: float, 
        threshold: float, 
        min_samples_train: int
    ) -> None:
    
        super().__init__(capacity)
        self._delta_window = delta_window
        self._lambda = sensitivity
        self._threshold = threshold
        self._min_samples_train = min_samples_train

        self._q_stats['confidence'] = []

    def test_drift(self, last_update_len: int):
        """
        Method representing the application of the drift detection method on the memory buffer.
        """
        
        if len(self) < 2 * self._delta_window or len(self) < self._min_samples_train:
            return -1
        
        confidences = self.get_stat('confidence')
        test_drift = False
        for c in confidences[-last_update_len:]:
            if exp(-2 * c) < random.random():
                test_drift = True
                break
            
        # All the confidences were high enough
        if not test_drift:
            return -1
        
        drift_idx, max_diff = -1, float('-inf')
        for p in range(self._delta_window, len(self) - self._delta_window):
            mean_a, mean_b = np.mean(confidences[p:]), np.mean(confidences[:p])
            if mean_a <= (1 - self._lambda) * mean_b:
                var_a, var_b = np.std(confidences[p:])**2, np.std(confidences[:p])**2
                if var_b < mean_b*(1-mean_b) and var_a < mean_a*(1-mean_a):
                    alpha_b = mean_b * ((mean_b*(1 - mean_b) / var_b) - 1)
                    beta_b = (1 - mean_b) * ((mean_b*(1 - mean_b) / var_b) - 1)
                    alpha_a = mean_a * ((mean_a*(1 - mean_a) / var_a) - 1)
                    beta_a = (1 - mean_a) * ((mean_a*(1 - mean_a) / var_a) - 1)

                    partial_diff = 0
                    for k in range(p, len(self) - last_update_len):
                        partial_diff += log(beta.pdf(confidences[k], alpha_a, beta_a) +
                                            beta.pdf(confidences[k], alpha_b, beta_b))
                    
                    if partial_diff > max_diff:
                        drift_idx, max_diff = p, partial_diff
        
        if max_diff > self._threshold:
            print(f"Drift occurred at index {drift_idx} with threshold value {max_diff}")
            return drift_idx
        
        return -1
        