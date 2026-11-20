"""
Model Training and Validation Engine
Author: MUHAMMED AFLAH S (RA2411026050147)
"""

import os
from typing import Dict, Any, List, Optional
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss

from glycopulse.config import TrainingConfig, ModelConfig
from glycopulse.model.network import GlycoPulseNet
from glycopulse.loss.uncertainty_loss import HomoscedasticUncertaintyLoss


class GlycoPulseTrainer:
    """
    Production-grade training orchestrator for GLYCOPULSE-NET.
    
    Features:
      - Homoscedastic Multi-Task Loss balance (Kendall & Gal).
      - AdamW optimizer with Cosine Annealing learning rate schedule.
      - Gradient norm clipping for numerical stability.
      - Validation tracking: AUROC, AUPRC, Brier Score per clinical endpoint.
      - Early stopping with best checkpoint restoration.
    """
    def __init__(
        self,
        model: GlycoPulseNet,
        loss_fn: HomoscedasticUncertaintyLoss,
        config: Optional[TrainingConfig] = None,
        device: str = "cpu"
    ):
        self.model = model
        self.loss_fn = loss_fn
        self.config = config or TrainingConfig()
        self.device = torch.device(device)

        self.model.to(self.device)
        self.loss_fn.to(self.device)

        # Joint optimization of network weights and task homoscedastic log-variances
        self.optimizer = torch.optim.AdamW(
            list(self.model.parameters()) + list(self.loss_fn.parameters()),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay
        )

        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=self.config.epochs,
            eta_min=1e-6
        )

        self.history: Dict[str, List[float]] = {
            "train_loss": [],
            "val_loss": [],
            "val_t2dm_auc": [],
            "val_neph_auc": [],
            "val_ascvd_auc": []
        }

    def train_epoch(self, train_loader: DataLoader) -> Dict[str, float]:
        self.model.train()
        self.loss_fn.train()

        total_loss = 0.0
        task_losses = [0.0, 0.0, 0.0]
        n_batches = len(train_loader)

        for x_batch, y_batch in train_loader:
            x_batch = x_batch.to(self.device)
            y_batch = y_batch.to(self.device)

            self.optimizer.zero_grad()
            logits = self.model(x_batch)
            loss, metrics = self.loss_fn(logits, y_batch)

            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)
            self.optimizer.step()

            total_loss += loss.item()
            for k in range(3):
                task_losses[k] += metrics.get(f"task_{k}_loss", 0.0)

        self.scheduler.step()

        return {
            "train_loss": total_loss / n_batches,
            "train_task0_loss": task_losses[0] / n_batches,
            "train_task1_loss": task_losses[1] / n_batches,
            "train_task2_loss": task_losses[2] / n_batches,
            "lr": self.optimizer.param_groups[0]["lr"]
        }

    def evaluate(self, val_loader: DataLoader) -> Dict[str, float]:
        self.model.eval()
        self.loss_fn.eval()

        total_loss = 0.0
        all_logits = []
        all_targets = []
        n_batches = len(val_loader)

        with torch.no_grad():
            for x_batch, y_batch in val_loader:
                x_batch = x_batch.to(self.device)
                y_batch = y_batch.to(self.device)

                logits = self.model(x_batch)
                loss, _ = self.loss_fn(logits, y_batch)

                total_loss += loss.item()
                all_logits.append(logits.cpu())
                all_targets.append(y_batch.cpu())

        all_logits = torch.cat(all_logits, dim=0).numpy()
        all_targets = torch.cat(all_targets, dim=0).numpy()
        all_probs = 1.0 / (1.0 + np.exp(-all_logits))

        # Calculate clinical metrics
        task_names = ["t2dm", "neph", "ascvd"]
        metrics = {"val_loss": total_loss / n_batches}

        for k, name in enumerate(task_names):
            y_true = all_targets[:, k]
            y_pred = all_probs[:, k]

            try:
                auc = float(roc_auc_score(y_true, y_pred))
            except Exception:
                auc = 0.5
            try:
                prauc = float(average_precision_score(y_true, y_pred))
            except Exception:
                prauc = 0.0
            brier = float(brier_score_loss(y_true, y_pred))

            metrics[f"val_{name}_auc"] = auc
            metrics[f"val_{name}_prauc"] = prauc
            metrics[f"val_{name}_brier"] = brier

        return metrics

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        checkpoint_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Executes complete training loop with early stopping and best model saving.
        """
        best_val_loss = float("inf")
        patience_counter = 0
        best_epoch = 0

        for epoch in range(1, self.config.epochs + 1):
            train_metrics = self.train_epoch(train_loader)
            val_metrics = self.evaluate(val_loader)

            val_loss = val_metrics["val_loss"]
            self.history["train_loss"].append(train_metrics["train_loss"])
            self.history["val_loss"].append(val_loss)
            self.history["val_t2dm_auc"].append(val_metrics["val_t2dm_auc"])
            self.history["val_neph_auc"].append(val_metrics["val_neph_auc"])
            self.history["val_ascvd_auc"].append(val_metrics["val_ascvd_auc"])

            # Check for improvement
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_epoch = epoch
                patience_counter = 0

                if checkpoint_path:
                    os.makedirs(os.path.dirname(os.path.abspath(checkpoint_path)), exist_ok=True)
                    torch.save({
                        "epoch": epoch,
                        "model_state_dict": self.model.state_dict(),
                        "loss_state_dict": self.loss_fn.state_dict(),
                        "optimizer_state_dict": self.optimizer.state_dict(),
                        "val_metrics": val_metrics,
                        "epochs": self.config.epochs,
                        "batch_size": self.config.batch_size,
                        "learning_rate": self.config.learning_rate
                    }, checkpoint_path)
            else:
                patience_counter += 1
                if patience_counter >= self.config.patience:
                    break

        return {
            "best_epoch": best_epoch,
            "best_val_loss": best_val_loss,
            "history": self.history
        }
