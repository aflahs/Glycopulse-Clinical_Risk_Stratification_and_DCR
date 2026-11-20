"""
Homoscedastic Multi-Task Uncertainty Loss Module
Author: MUHAMMED AFLAH S (RA2411026050147)
Reference: Kendall & Gal, "Multi-Task Learning Using Uncertainty to Weigh Losses", CVPR 2018.
"""

from typing import Dict, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class HomoscedasticUncertaintyLoss(nn.Module):
    """
    Dynamically balances multi-task loss gradients by learning homoscedastic task variances:
        L_total = sum_k [ 0.5 * exp(-s_k) * L_k + 0.5 * s_k ]
    where s_k = log(sigma_k^2) represents the log-variance for task k.
    
    This eliminates manual hyperparameter tuning of task weights and prevents any
    single high-gradient task from dominating latent representations.
    """
    def __init__(self, num_tasks: int = 3, pos_weights: torch.Tensor = None):
        super().__init__()
        self.num_tasks = num_tasks

        # Learnable log-variances initialized to 0 (sigma^2 = 1.0)
        self.log_vars = nn.Parameter(torch.zeros(num_tasks))

        # Binary Cross Entropy with Logits for each classification task
        if pos_weights is not None:
            self.criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weights, reduction="mean")
        else:
            self.criterion = nn.BCEWithLogitsLoss(reduction="mean")

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Args:
            logits: Predicted logits of shape (B, num_tasks).
            targets: Binary ground truth labels of shape (B, num_tasks).
        Returns:
            total_loss: Scalar weighted multi-task loss with gradient tracking.
            metrics: Dictionary of individual task losses and learned task standard deviations.
        """
        assert logits.shape == targets.shape, f"Logits shape {logits.shape} != targets shape {targets.shape}"

        total_loss = torch.tensor(0.0, device=logits.device)
        metrics: Dict[str, float] = {}

        for k in range(self.num_tasks):
            task_logit = logits[:, k]
            task_target = targets[:, k].float()

            task_loss = F.binary_cross_entropy_with_logits(task_logit, task_target)

            # Precision factor exp(-s_k) and regularization penalty 0.5 * s_k
            precision = torch.exp(-self.log_vars[k])
            weighted_task_loss = 0.5 * precision * task_loss + 0.5 * self.log_vars[k]

            total_loss = total_loss + weighted_task_loss

            # Track task standard deviation sigma_k = exp(0.5 * s_k)
            sigma_k = torch.exp(0.5 * self.log_vars[k]).item()
            metrics[f"task_{k}_loss"] = task_loss.item()
            metrics[f"task_{k}_sigma"] = sigma_k

        metrics["total_loss"] = total_loss.item()
        return total_loss, metrics
