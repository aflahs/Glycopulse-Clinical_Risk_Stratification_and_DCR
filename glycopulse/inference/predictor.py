"""
Clinical Inference, Risk Stratification and Uncertainty Estimation Pipeline
Author: MUHAMMED AFLAH S (RA2411026050147)
"""

from typing import Dict, Any, List, Optional, Union
import numpy as np
import torch
import torch.nn as nn

from glycopulse.config import FEATURE_NAMES, NUM_FEATURES, BIOMARKER_REGISTRY
from glycopulse.model.network import GlycoPulseNet
from glycopulse.calibration.temperature import TemperatureScalingCalibrator
from glycopulse.calibration.ood import MahalanobisOODDetector
from glycopulse.recourse.optimizer import CounterfactualRecourseEngine


class GlycoPulsePredictor:
    """
    Unified High-Level Clinical Inference Pipeline.
    
    Integrates:
      1. Clinical biomarker preprocessing & z-score standardization.
      2. Bayesian Monte Carlo Variational Dropout for epistemic uncertainty quantification.
      3. Temperature Scaling confidence calibration.
      4. Mahalanobis latent manifold Out-of-Distribution (OOD) screening.
      5. Automated trigger for Differentiable Counterfactual Recourse on elevated risk.
    """
    def __init__(
        self,
        model: GlycoPulseNet,
        feature_means: np.ndarray,
        feature_stds: np.ndarray,
        calibrator: Optional[TemperatureScalingCalibrator] = None,
        ood_detector: Optional[MahalanobisOODDetector] = None,
        device: str = "cpu"
    ):
        self.device = torch.device(device)
        self.model = model.to(self.device)
        self.model.eval()

        self.feature_means = np.asarray(feature_means, dtype=np.float32)
        self.feature_stds = np.asarray(feature_stds, dtype=np.float32)
        self.calibrator = calibrator
        self.ood_detector = ood_detector

        # Recourse engine initialized on demand
        self.recourse_engine = CounterfactualRecourseEngine(
            model=self.model,
            feature_means=self.feature_means,
            feature_stds=self.feature_stds,
            device=device
        )

    def _parse_input(self, raw_input: Union[Dict[str, float], List[float], np.ndarray]) -> np.ndarray:
        """
        Validates and converts diverse input formats to canonical 1D array of length 24.
        """
        if isinstance(raw_input, dict):
            values = []
            for name in FEATURE_NAMES:
                if name not in raw_input:
                    # Fallback to healthy midpoint
                    meta = next(m for m in BIOMARKER_REGISTRY if m.name == name)
                    values.append((meta.healthy_min + meta.healthy_max) / 2.0)
                else:
                    values.append(float(raw_input[name]))
            arr = np.array(values, dtype=np.float32)
        else:
            arr = np.asarray(raw_input, dtype=np.float32)

        if arr.shape != (NUM_FEATURES,):
            raise ValueError(f"Expected array of shape ({NUM_FEATURES},), got {arr.shape}")
        return arr

    def predict(
        self,
        raw_biomarkers: Union[Dict[str, float], List[float], np.ndarray],
        mc_samples: int = 30,
        generate_recourse: bool = True,
        recourse_task_idx: int = 0
    ) -> Dict[str, Any]:
        """
        Executes end-to-end clinical diagnostic triage.
        
        Args:
            raw_biomarkers: Dict or array of 24 patient biomarkers.
            mc_samples: Number of stochastic MC dropout passes.
            generate_recourse: Whether to compute counterfactual intervention if risk > 20%.
            recourse_task_idx: Endpoint for recourse (0 = T2DM, 1 = Nephropathy, 2 = ASCVD).
        Returns:
            Structured clinical report dictionary.
        """
        x_raw = self._parse_input(raw_biomarkers)
        x_norm = (x_raw - self.feature_means) / (self.feature_stds + 1e-7)
        x_tensor = torch.tensor(x_norm, dtype=torch.float32, device=self.device).unsqueeze(0)

        # 1. Latent bottleneck representation & OOD check
        with torch.no_grad():
            h_latent = self.model.extract_latent(x_tensor).cpu().numpy()

        ood_info = {"is_ood": False, "mahalanobis_distance": 0.0, "threshold": 0.0}
        if self.ood_detector is not None and self.ood_detector.is_fitted:
            ood_res = self.ood_detector.predict(h_latent)
            ood_info = {
                "is_ood": ood_res["is_ood"][0],
                "mahalanobis_distance": round(ood_res["distances"][0], 3),
                "threshold": round(ood_res["threshold"], 3)
            }

        # 2. Bayesian Monte Carlo Dropout Uncertainty Quantification
        temperatures = self.calibrator.temperatures.to(self.device) if self.calibrator else None
        mc_res = self.model.predict_monte_carlo(
            x_tensor,
            num_samples=mc_samples,
            temperature=temperatures
        )

        mean_probs = mc_res["mean_probs"][0].cpu().numpy()
        epistemic_std = mc_res["epistemic_std"][0].cpu().numpy()

        # Risk categories & clinical alerts
        risk_stratification = {
            "T2DM_Onset_5Year": {
                "probability": round(float(mean_probs[0]), 4),
                "epistemic_uncertainty": round(float(epistemic_std[0]), 4),
                "tier": "High Risk" if mean_probs[0] >= 0.50 else ("Moderate Risk" if mean_probs[0] >= 0.20 else "Low Risk")
            },
            "Nephropathy_Progression": {
                "probability": round(float(mean_probs[1]), 4),
                "epistemic_uncertainty": round(float(epistemic_std[1]), 4),
                "tier": "High Risk" if mean_probs[1] >= 0.50 else ("Moderate Risk" if mean_probs[1] >= 0.20 else "Low Risk")
            },
            "ASCVD_5Year_Risk": {
                "probability": round(float(mean_probs[2]), 4),
                "epistemic_uncertainty": round(float(epistemic_std[2]), 4),
                "tier": "High Risk" if mean_probs[2] >= 0.50 else ("Moderate Risk" if mean_probs[2] >= 0.20 else "Low Risk")
            }
        }

        # 3. Optional Counterfactual Recourse
        recourse_plan = None
        if generate_recourse and mean_probs[recourse_task_idx] >= 0.20:
            recourse_plan = self.recourse_engine.optimize_recourse(
                patient_biomarkers=x_raw,
                target_task_idx=recourse_task_idx,
                target_risk_threshold=0.20
            )

        return {
            "patient_biomarkers": {FEATURE_NAMES[i]: float(x_raw[i]) for i in range(NUM_FEATURES)},
            "risk_stratification": risk_stratification,
            "ood_screening": ood_info,
            "counterfactual_recourse": recourse_plan
        }
