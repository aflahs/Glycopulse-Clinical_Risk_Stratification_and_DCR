"""
Benchmark Evaluation CLI Script: GLYCOPULSE-NET vs Baselines
Author: MUHAMMED AFLAH S (RA2411026050147)
"""

import argparse
import os
import sys

# Ensure repository root is in python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss

from glycopulse.config import ModelConfig
from glycopulse.model.network import GlycoPulseNet
from glycopulse.calibration.temperature import TemperatureScalingCalibrator, calculate_ece
from glycopulse.data.dataset import NHANESCohortGenerator, ClinicalDataModule


def evaluate_binary_predictions(y_true, y_prob):
    auc = roc_auc_score(y_true, y_prob)
    prauc = average_precision_score(y_true, y_prob)
    brier = brier_score_loss(y_true, y_prob)
    ece = calculate_ece(y_prob, y_true, n_bins=15)
    return auc, prauc, brier, ece


def main():
    parser = argparse.ArgumentParser(description="Evaluate GLYCOPULSE-NET vs Machine Learning Baselines")
    parser.add_argument("--samples", type=int, default=3000, help="Test cohort size")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    print("=" * 80)
    print(" BENCHMARK COMPARISON: GLYCOPULSE-NET vs CLINICAL BASELINES")
    print(" Lead Researcher: MUHAMMED AFLAH S | Reg No: RA2411026050147")
    print(" SRM Institute of Science and Technology, Tiruchirappalli")
    print("=" * 80)

    # Generate synthetic NHANES cohort
    gen = NHANESCohortGenerator(random_seed=args.seed)
    X, Y, _ = gen.generate_cohort(n_samples=args.samples)
    dm = ClinicalDataModule(X, Y, random_seed=args.seed)

    print(f"\nCohort Size: {args.samples} patients | Train: {len(dm.X_train)} | Test: {len(dm.X_test)}")
    print("Evaluating 4 Competitive Architectures on 3 Endpoints...\n")

    # Train baselines per task
    results = {}

    # 1. Logistic Regression
    lr_preds = []
    for k in range(3):
        clf = LogisticRegression(max_iter=500, random_state=args.seed)
        clf.fit(dm.X_train, dm.Y_train[:, k])
        p = clf.predict_proba(dm.X_test)[:, 1]
        lr_preds.append(p)
    results["Logistic Regression"] = np.stack(lr_preds, axis=1)

    # 2. Standard MLP (2 Hidden Layers)
    mlp_preds = []
    for k in range(3):
        mlp = MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=200, random_state=args.seed)
        mlp.fit(dm.X_train, dm.Y_train[:, k])
        p = mlp.predict_proba(dm.X_test)[:, 1]
        mlp_preds.append(p)
    results["Standard MLP (Deep)"] = np.stack(mlp_preds, axis=1)

    # 3. Gradient Boosting Trees (LightGBM/HistGradient)
    hgb_preds = []
    for k in range(3):
        hgb = HistGradientBoostingClassifier(max_iter=100, random_state=args.seed)
        hgb.fit(dm.X_train, dm.Y_train[:, k])
        p = hgb.predict_proba(dm.X_test)[:, 1]
        hgb_preds.append(p)
    results["HistGradientBoost (GBDT)"] = np.stack(hgb_preds, axis=1)

    # 4. GLYCOPULSE-NET
    model = GlycoPulseNet(ModelConfig())
    # Lightweight quick fit on train batches
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    train_loader, _, _ = dm.get_dataloaders()
    model.train()
    for epoch in range(12):
        for xb, yb in train_loader:
            optimizer.zero_grad()
            logits = model(xb)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, yb)
            loss.backward()
            optimizer.step()

    model.eval()
    with torch.no_grad():
        test_logits = model(torch.tensor(dm.X_test, dtype=torch.float32))
        glyco_probs = torch.sigmoid(test_logits).numpy()
    results["GLYCOPULSE-NET (Ours)"] = glyco_probs

    # Output Benchmark Table
    task_names = ["T2DM Onset (5-Yr)", "Nephropathy eGFR Decline", "ASCVD Risk (5-Yr)"]

    for k, task in enumerate(task_names):
        print(f"\n--- [Task {k+1}/3: {task}] ---")
        print(f"{'Model Architecture':<28} | {'AUROC':<9} | {'AUPRC':<9} | {'ECE':<9} | {'Brier':<9}")
        print("-" * 72)
        y_true = dm.Y_test[:, k]
        for name, preds in results.items():
            auc, prauc, brier, ece = evaluate_binary_predictions(y_true, preds[:, k])
            print(f"{name:<28} | {auc:<9.4f} | {prauc:<9.4f} | {ece:<9.4f} | {brier:<9.4f}")

    print("\n" + "=" * 80)
    print(" SYSTEM CAPABILITY MATRIX:")
    print(f"{'Feature / Capability':<35} | {'GBDT / Tabular':<18} | {'GLYCOPULSE-NET':<18}")
    print("-" * 80)
    print(f"{'Biomarker Gated Tokenization (SwiGLU)':<35} | {'No':<18} | {'Yes (BGT)':<18}")
    print(f"{'Homoscedastic Multi-Task Loss':<35} | {'No (Disjoint)':<18} | {'Yes (Kendall-Gal)':<18}")
    print(f"{'Bayesian Monte Carlo Uncertainty':<35} | {'No':<18} | {'Yes (Variational)':<18}")
    print(f"{'Differentiable Recourse Optimizer':<35} | {'No (Heuristic)':<18} | {'Yes (Inverse Grad)':<18}")
    print(f"{'Mahalanobis Latent OOD Screening':<35} | {'No':<18} | {'Yes (Bottleneck)':<18}")
    print("=" * 80)


if __name__ == "__main__":
    main()
