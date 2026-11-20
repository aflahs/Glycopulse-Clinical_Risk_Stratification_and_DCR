"""
Confidence Calibration & Temperature Scaling
Author: MUHAMMED AFLAH S (RA2411026050147)
Reference: Guo et al., "On Calibration of Modern Neural Networks", ICML 2017.
"""

from typing import Tuple, Dict, List
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


def calculate_ece(probs: np.ndarray, labels: np.ndarray, n_bins: int = 15) -> float:
    """
    Computes Expected Calibration Error (ECE) for binary classification:
        ECE = sum_m (|B_m| / N) * |acc(B_m) - conf(B_m)|
    
    Args:
        probs: 1D array of predicted probabilities in [0, 1].
        labels: 1D array of binary ground truth labels in {0, 1}.
        n_bins: Number of equal-width confidence bins (default: 15).
    Returns:
        ece: Float value in [0, 1].
    """
    assert len(probs) == len(labels), "Length mismatch between probs and labels"
    bin_boundaries = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(probs)

    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]

        # Samples within bin [bin_lower, bin_upper]
        if i == n_bins - 1:
            in_bin = (probs >= bin_lower) & (probs <= bin_upper)
        else:
            in_bin = (probs >= bin_lower) & (probs < bin_upper)

        bin_size = np.sum(in_bin)
        if bin_size > 0:
            bin_acc = np.mean(labels[in_bin])
            bin_conf = np.mean(probs[in_bin])
            ece += (bin_size / n) * np.abs(bin_acc - bin_conf)

    return float(ece)


class TemperatureScalingCalibrator(nn.Module):
    """
    Post-hoc confidence calibrator using Temperature Scaling.
    Optimizes a single positive scalar temperature T_k per task:
        p_calibrated = sigmoid(logit / T_k)
    Preserves model ranking (AUROC unchanged) while minimizing NLL and ECE.
    """
    def __init__(self, num_tasks: int = 3):
        super().__init__()
        # Initialize temperature parameters to 1.0 (log_temp = 0.0)
        self.log_temperatures = nn.Parameter(torch.zeros(num_tasks))

    @property
    def temperatures(self) -> torch.Tensor:
        return torch.exp(self.log_temperatures)

    def calibrate_logits(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Scales logits by temperature: logits / T
        """
        return logits / self.temperatures.to(logits.device)

    def fit(
        self,
        val_logits: torch.Tensor,
        val_targets: torch.Tensor,
        lr: float = 0.01,
        max_iter: int = 100
    ) -> Dict[str, float]:
        """
        Fits temperature scaling parameters using L-BFGS / Adam on validation set.
        
        Args:
            val_logits: Uncalibrated logits (N_val, num_tasks).
            val_targets: Ground truth binary labels (N_val, num_tasks).
        Returns:
            Dictionary of fitted temperatures per task.
        """
        device = val_logits.device
        self.to(device)
        optimizer = optim.LBFGS([self.log_temperatures], lr=lr, max_iter=max_iter)

        def eval_loss():
            optimizer.zero_grad()
            scaled_logits = self.calibrate_logits(val_logits)
            loss = nn.functional.binary_cross_entropy_with_logits(
                scaled_logits,
                val_targets.float()
            )
            loss.backward()
            return loss

        optimizer.step(eval_loss)

        fitted_temps = {}
        with torch.no_grad():
            temps = self.temperatures.cpu().numpy()
            for k in range(len(temps)):
                fitted_temps[f"task_{k}_temperature"] = float(temps[k])

        return fitted_temps
