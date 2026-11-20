"""
Inverse Differentiable Counterfactual Recourse Engine
Author: MUHAMMED AFLAH S (RA2411026050147)
"""

from typing import Dict, Any, List, Optional
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from glycopulse.config import (
    BIOMARKER_REGISTRY,
    FEATURE_NAMES,
    ACTIONABLE_MASK,
    DIRECTION_VECTOR,
    RecourseConfig
)


class CounterfactualRecourseEngine:
    """
    Inverse Differentiable Counterfactual Recourse Engine.
    
    Given a high-risk patient biomarker profile x_0, optimizes an actionable perturbation
    vector delta_x via Projected Gradient Descent:
        x* = x_0 + delta_x
    subject to:
      1. Immutability constraints: delta_x[immutable] = 0 (e.g. Age, Family History).
      2. Clinical plausibility bounds: physio_min <= x_0 + delta_x <= physio_max.
      3. Directional validity: delta_x follows valid pathophysiological intervention paths.
      4. Sparse L1 regularization (minimal number of altered lifestyle/medication biomarkers).
      5. Mahalanobis / Normalized L2 distance to avoid unrealistic covariate shifts.
    """
    def __init__(
        self,
        model: nn.Module,
        config: Optional[RecourseConfig] = None,
        feature_means: Optional[np.ndarray] = None,
        feature_stds: Optional[np.ndarray] = None,
        device: str = "cpu"
    ):
        self.model = model
        self.model.eval()
        self.config = config or RecourseConfig()
        self.device = torch.device(device)
        self.model.to(self.device)

        # Actionable mask: 1.0 for actionable, 0.0 for immutable
        self.actionable_mask = torch.tensor(
            [1.0 if a else 0.0 for a in ACTIONABLE_MASK],
            dtype=torch.float32,
            device=self.device
        )

        # Directional constraints (-1 = must decrease or zero, +1 = must increase or zero, 0 = free/immutable)
        self.directions = torch.tensor(
            DIRECTION_VECTOR,
            dtype=torch.float32,
            device=self.device
        )

        # Physiological bounds
        self.physio_min = torch.tensor(
            [m.physio_min for m in BIOMARKER_REGISTRY],
            dtype=torch.float32,
            device=self.device
        )
        self.physio_max = torch.tensor(
            [m.physio_max for m in BIOMARKER_REGISTRY],
            dtype=torch.float32,
            device=self.device
        )

        # Feature normalization scales
        if feature_means is not None and feature_stds is not None:
            self.means = torch.tensor(feature_means, dtype=torch.float32, device=self.device)
            self.stds = torch.tensor(feature_stds, dtype=torch.float32, device=self.device)
        else:
            self.means = (self.physio_min + self.physio_max) / 2.0
            self.stds = (self.physio_max - self.physio_min) / 4.0

        self.scale = self.stds.clone()

    def optimize_recourse(
        self,
        patient_biomarkers: np.ndarray,
        target_task_idx: int = 0,
        target_risk_threshold: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Finds optimal counterfactual intervention for an individual patient.
        
        Args:
            patient_biomarkers: 1D array of shape (24,) containing raw patient lab values in clinical units.
            target_task_idx: Index of clinical endpoint to reduce (0 = T2DM, 1 = Nephropathy, 2 = ASCVD).
            target_risk_threshold: Desired maximum post-intervention risk probability (default from config).
        """
        target_threshold = target_risk_threshold or self.config.target_risk_threshold

        x_raw = torch.tensor(
            patient_biomarkers,
            dtype=torch.float32,
            device=self.device
        ).unsqueeze(0)  # Shape (1, 24)

        # Baseline evaluation (raw -> normalized -> forward)
        with torch.no_grad():
            x_norm_0 = (x_raw - self.means) / (self.stds + 1e-7)
            initial_logits = self.model(x_norm_0)
            initial_probs = torch.sigmoid(initial_logits)[0]
            initial_risk = initial_probs[target_task_idx].item()

        if initial_risk <= target_threshold:
            return {
                "initial_risk": initial_risk,
                "optimized_risk": initial_risk,
                "target_risk": target_threshold,
                "converged": True,
                "num_actionable_changes": 0,
                "original_biomarkers": {FEATURE_NAMES[i]: float(patient_biomarkers[i]) for i in range(len(FEATURE_NAMES))},
                "counterfactual_biomarkers": {FEATURE_NAMES[i]: float(patient_biomarkers[i]) for i in range(len(FEATURE_NAMES))},
                "biomarker_deltas": {FEATURE_NAMES[i]: 0.0 for i in range(len(FEATURE_NAMES))},
                "intervention_summary": ["Patient risk is already below the target clinical threshold. No intervention required."]
            }

        # Learnable delta in raw feature units
        delta = nn.Parameter(torch.zeros_like(x_raw))
        optimizer = torch.optim.Adam([delta], lr=self.config.learning_rate * 5.0)

        best_delta = delta.detach().clone()
        best_risk = initial_risk
        lowest_risk = initial_risk
        lowest_risk_delta = delta.detach().clone()
        best_cost = float("inf")
        converged = False

        for step in range(self.config.max_steps):
            optimizer.zero_grad()

            # Candidate counterfactual in raw scale
            x_cf_raw = x_raw + delta * self.actionable_mask

            # Normalized for neural network input
            x_cf_norm = (x_cf_raw - self.means) / (self.stds + 1e-7)

            logits = self.model(x_cf_norm)
            prob = torch.sigmoid(logits[0, target_task_idx])

            # 1. Target prediction risk loss
            task_loss = F.relu(prob - target_threshold) ** 2

            # 2. Normalized sparsity and proximity penalties
            norm_delta = (delta * self.actionable_mask) / (self.scale + 1e-6)
            sparsity_loss = torch.sum(torch.abs(norm_delta))
            proximity_loss = torch.sum(norm_delta ** 2)

            # 3. Directional penalty
            direction_penalty = torch.tensor(0.0, device=self.device)
            for i in range(len(DIRECTION_VECTOR)):
                dir_i = self.directions[i]
                if dir_i == -1.0:
                    direction_penalty = direction_penalty + F.relu(delta[0, i]) ** 2
                elif dir_i == 1.0:
                    direction_penalty = direction_penalty + F.relu(-delta[0, i]) ** 2

            total_loss = (
                task_loss * 25.0
                + self.config.sparsity_lambda * sparsity_loss
                + self.config.manifold_lambda * proximity_loss
                + 5.0 * direction_penalty
            )

            total_loss.backward()
            optimizer.step()

            # Projection step
            with torch.no_grad():
                # Enforce immutable zeros
                delta.data.mul_(self.actionable_mask)

                # Directional projection
                for i in range(len(DIRECTION_VECTOR)):
                    dir_i = self.directions[i]
                    if dir_i == -1.0:
                        delta.data[0, i].clamp_(max=0.0)
                    elif dir_i == 1.0:
                        delta.data[0, i].clamp_(min=0.0)

                # Clamp x_raw + delta to valid physiological ranges
                clamped_x = torch.clamp(x_raw + delta.data, min=self.physio_min, max=self.physio_max)
                delta.data.copy_(clamped_x - x_raw)

                # Evaluate current state
                curr_norm = (x_raw + delta.data - self.means) / (self.stds + 1e-7)
                curr_prob = torch.sigmoid(self.model(curr_norm)[0, target_task_idx]).item()
                curr_cost = total_loss.item()

                if curr_prob < lowest_risk:
                    lowest_risk = curr_prob
                    lowest_risk_delta = delta.data.clone()

                if curr_prob <= target_threshold:
                    if curr_cost < best_cost:
                        best_cost = curr_cost
                        best_delta = delta.data.clone()
                        best_risk = curr_prob
                        converged = True

        if not converged:
            best_delta = lowest_risk_delta
            best_risk = lowest_risk

        final_delta = best_delta.squeeze(0).cpu().numpy()
        final_x = patient_biomarkers + final_delta

        # Generate clinical intervention summary
        summary = []
        actionable_changes = 0
        deltas_dict = {}
        cf_dict = {}
        orig_dict = {}

        for i, name in enumerate(FEATURE_NAMES):
            orig_val = float(patient_biomarkers[i])
            cf_val = float(final_x[i])
            d_val = float(final_delta[i])
            orig_dict[name] = round(orig_val, 2)
            cf_dict[name] = round(cf_val, 2)
            deltas_dict[name] = round(d_val, 2)

            if abs(d_val) > 1e-2 and ACTIONABLE_MASK[i]:
                actionable_changes += 1
                meta = BIOMARKER_REGISTRY[i]
                action = "Reduce" if d_val < 0 else "Increase"
                summary.append(
                    f"{action} {meta.name} ({meta.description}) by {abs(d_val):.2f} {meta.unit} "
                    f"(from {orig_val:.2f} to {cf_val:.2f} {meta.unit})"
                )

        if not summary:
            summary.append("No actionable changes satisfied the target risk threshold.")

        return {
            "initial_risk": round(initial_risk, 4),
            "optimized_risk": round(best_risk, 4),
            "target_risk": round(target_threshold, 4),
            "converged": converged,
            "num_actionable_changes": actionable_changes,
            "original_biomarkers": orig_dict,
            "counterfactual_biomarkers": cf_dict,
            "biomarker_deltas": deltas_dict,
            "intervention_summary": summary
        }
