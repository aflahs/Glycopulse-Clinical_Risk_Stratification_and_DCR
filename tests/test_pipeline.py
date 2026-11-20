"""
GLYCOPULSE-NET Automated Pytest Verification Suite
Author: MUHAMMED AFLAH S (RA2411026050147)
"""

import pytest
import numpy as np
import torch
import torch.nn as nn

from glycopulse.config import ModelConfig, FEATURE_NAMES, NUM_FEATURES, BIOMARKER_REGISTRY
from glycopulse.model.tokenization import BiomarkerGatedTokenizer, SwiGLUGating
from glycopulse.model.attention import ResidualCrossAttentionBlock, LatentBottleneckPooling
from glycopulse.model.network import GlycoPulseNet
from glycopulse.loss.uncertainty_loss import HomoscedasticUncertaintyLoss
from glycopulse.data.dataset import NHANESCohortGenerator, ClinicalDataModule
from glycopulse.calibration.temperature import TemperatureScalingCalibrator, calculate_ece
from glycopulse.calibration.ood import MahalanobisOODDetector
from glycopulse.recourse.optimizer import CounterfactualRecourseEngine


def test_swiglu_gating():
    batch_size = 8
    dim = 64
    x = torch.randn(batch_size, dim)
    layer = SwiGLUGating(in_features=dim, hidden_features=dim * 2, out_features=dim)
    out = layer(x)
    assert out.shape == (batch_size, dim)
    assert not torch.isnan(out).any()


def test_biomarker_gated_tokenizer():
    batch_size = 4
    x = torch.randn(batch_size, NUM_FEATURES)
    tokenizer = BiomarkerGatedTokenizer(num_features=NUM_FEATURES, d_model=128)
    tokens = tokenizer(x)
    assert tokens.shape == (batch_size, NUM_FEATURES, 128)
    assert tokens.requires_grad or x.requires_grad is False


def test_residual_cross_attention_block():
    batch_size = 4
    tokens = torch.randn(batch_size, NUM_FEATURES, 128)
    block = ResidualCrossAttentionBlock(d_model=128, n_heads=4, dim_feedforward=256)
    out = block(tokens)
    assert out.shape == (batch_size, NUM_FEATURES, 128)
    assert not torch.isnan(out).any()


def test_latent_bottleneck_pooling():
    batch_size = 6
    tokens = torch.randn(batch_size, NUM_FEATURES, 128)
    pool = LatentBottleneckPooling(d_model=128, n_heads=4)
    patient_h = pool(tokens)
    assert patient_h.shape == (batch_size, 128)


def test_glycopulse_net_full_forward():
    batch_size = 5
    x = torch.randn(batch_size, NUM_FEATURES)
    cfg = ModelConfig()
    model = GlycoPulseNet(cfg)

    # Test forward returning logits
    logits = model(x)
    assert logits.shape == (batch_size, 3)

    # Test forward with latent extraction
    logits, h = model(x, return_latent=True)
    assert logits.shape == (batch_size, 3)
    assert h.shape == (batch_size, cfg.d_model)

    # Test Bayesian Monte Carlo inference
    mc_res = model.predict_monte_carlo(x, num_samples=10)
    assert mc_res["mean_probs"].shape == (batch_size, 3)
    assert mc_res["epistemic_std"].shape == (batch_size, 3)
    assert (mc_res["mean_probs"] >= 0.0).all() and (mc_res["mean_probs"] <= 1.0).all()


def test_homoscedastic_uncertainty_loss():
    batch_size = 8
    logits = torch.randn(batch_size, 3, requires_grad=True)
    targets = torch.randint(0, 2, (batch_size, 3)).float()

    loss_fn = HomoscedasticUncertaintyLoss(num_tasks=3)
    loss, metrics = loss_fn(logits, targets)

    assert loss.dim() == 0  # scalar
    assert loss.item() > 0
    assert "total_loss" in metrics
    assert "task_0_sigma" in metrics

    # Backward pass verification
    loss.backward()
    assert logits.grad is not None
    assert loss_fn.log_vars.grad is not None


def test_nhanes_cohort_generator():
    generator = NHANESCohortGenerator(random_seed=123)
    X, Y, df = generator.generate_cohort(n_samples=200)

    assert X.shape == (200, NUM_FEATURES)
    assert Y.shape == (200, 3)
    assert len(df) == 200
    assert "HbA1c" in df.columns
    assert "T2DM_Onset_Label" in df.columns

    # Verify no NaN or Inf
    assert not np.isnan(X).any()
    assert not np.isnan(Y).any()

    # DataModule partitioning test
    dm = ClinicalDataModule(X, Y, val_split=0.2, test_split=0.2, batch_size=32)
    train_loader, val_loader, test_loader = dm.get_dataloaders()
    assert len(train_loader) > 0
    assert len(val_loader) > 0
    assert len(test_loader) > 0


def test_counterfactual_recourse_engine_constraints():
    # Setup model and sample input
    model = GlycoPulseNet(ModelConfig())
    patient = np.array([(m.healthy_min + m.healthy_max) / 2.0 for m in BIOMARKER_REGISTRY], dtype=np.float32)

    engine = CounterfactualRecourseEngine(
        model=model,
        device="cpu"
    )

    result = engine.optimize_recourse(
        patient_biomarkers=patient,
        target_task_idx=0,
        target_risk_threshold=0.20
    )

    assert "initial_risk" in result
    assert "optimized_risk" in result
    assert "intervention_summary" in result

    # Check strict immutability: Age (index 20) and Family History (index 23) must have delta = 0
    age_delta = result["biomarker_deltas"]["Age"]
    fam_delta = result["biomarker_deltas"]["Family_History_Score"]
    assert age_delta == 0.0, f"Immutable Age was perturbed by {age_delta}"
    assert fam_delta == 0.0, f"Immutable Family History was perturbed by {fam_delta}"


def test_temperature_scaling_and_ece():
    # Synthetic predictions
    y_true = np.array([1, 1, 1, 0, 0, 1, 0, 0, 1, 0])
    y_prob = np.array([0.9, 0.8, 0.85, 0.1, 0.2, 0.7, 0.3, 0.15, 0.65, 0.25])

    ece = calculate_ece(y_prob, y_true, n_bins=5)
    assert 0.0 <= ece <= 1.0

    # Calibrator fitting
    calibrator = TemperatureScalingCalibrator(num_tasks=3)
    logits = torch.randn(50, 3)
    targets = torch.randint(0, 2, (50, 3))
    fitted = calibrator.fit(logits, targets)
    assert len(fitted) == 3
    assert all(t > 0 for t in calibrator.temperatures.detach().numpy())


def test_mahalanobis_ood_detector():
    d_model = 16
    n_train = 100
    train_latents = np.random.normal(0.0, 1.0, size=(n_train, d_model))

    detector = MahalanobisOODDetector(alpha=0.95)
    detector.fit(train_latents)

    assert detector.is_fitted
    assert detector.threshold > 0.0

    # In-distribution sample
    in_dist = np.random.normal(0.0, 1.0, size=(5, d_model))
    in_res = detector.predict(in_dist)
    assert len(in_res["distances"]) == 5

    # Extreme out-of-distribution anomaly
    extreme_ood = np.ones((2, d_model)) * 50.0
    out_res = detector.predict(extreme_ood)
    assert all(out_res["is_ood"])
