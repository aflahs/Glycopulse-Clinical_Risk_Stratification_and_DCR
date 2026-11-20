"""
GLYCOPULSE-NET Model Training CLI Script
Author: MUHAMMED AFLAH S (RA2411026050147)
"""

import argparse
import os
import sys

# Ensure repository root is in python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import numpy as np

from glycopulse.config import ModelConfig, TrainingConfig
from glycopulse.model.network import GlycoPulseNet
from glycopulse.loss.uncertainty_loss import HomoscedasticUncertaintyLoss
from glycopulse.training.trainer import GlycoPulseTrainer
from glycopulse.data.dataset import NHANESCohortGenerator, ClinicalDataModule
from glycopulse.calibration.temperature import TemperatureScalingCalibrator, calculate_ece
from glycopulse.calibration.ood import MahalanobisOODDetector


def parse_args():
    parser = argparse.ArgumentParser(description="Train GLYCOPULSE-NET Multi-Task Model")
    parser.add_argument("--epochs", type=int, default=25, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--samples", type=int, default=4000, help="Cohort dataset size")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints", help="Directory to save checkpoints")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--quick-test", action="store_true", help="Run rapid 2-epoch test")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.quick_test:
        args.epochs = 2
        args.samples = 600

    print("=" * 70)
    print(" GLYCOPULSE-NET: MULTI-TASK CLINICAL RISK STRATIFICATION")
    print(" Author: MUHAMMED AFLAH S | Register: RA2411026050147")
    print(" SRM Institute of Science and Technology, Tiruchirappalli")
    print("=" * 70)

    # 1. Synthesize NHANES-style multi-wave clinical cohort
    print(f"\n[1/5] Synthesizing clinical cohort ({args.samples} patients, 24 lab biomarkers)...")
    generator = NHANESCohortGenerator(random_seed=args.seed)
    X, Y, df = generator.generate_cohort(n_samples=args.samples)
    print(f"      Feature matrix: {X.shape} | Ground truth multi-task labels: {Y.shape}")
    print(f"      Prevalence -> T2DM: {np.mean(Y[:, 0])*100:.1f}%, Nephropathy: {np.mean(Y[:, 1])*100:.1f}%, ASCVD: {np.mean(Y[:, 2])*100:.1f}%")

    # 2. DataModule splits and normalization
    print("\n[2/5] Partitioning train/val/test splits and computing standardization parameters...")
    datamodule = ClinicalDataModule(
        X, Y,
        val_split=0.15,
        test_split=0.15,
        batch_size=args.batch_size,
        random_seed=args.seed
    )
    train_loader, val_loader, test_loader = datamodule.get_dataloaders()
    print(f"      Train samples: {len(datamodule.X_train)} | Val: {len(datamodule.X_val)} | Test: {len(datamodule.X_test)}")

    # 3. Model & Loss initialization
    print("\n[3/5] Initializing GLYCOPULSE-NET with Homoscedastic Uncertainty Loss...")
    model_cfg = ModelConfig()
    train_cfg = TrainingConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        random_seed=args.seed
    )

    model = GlycoPulseNet(model_cfg)
    loss_fn = HomoscedasticUncertaintyLoss(num_tasks=3)

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    best_ckpt_path = os.path.join(args.checkpoint_dir, "glycopulse_best.pt")

    trainer = GlycoPulseTrainer(
        model=model,
        loss_fn=loss_fn,
        config=train_cfg,
        device="cpu"
    )

    # 4. Training loop
    print(f"\n[4/5] Training for {args.epochs} epochs with AdamW & Cosine Annealing...")
    fit_res = trainer.fit(train_loader, val_loader, checkpoint_path=best_ckpt_path)

    print(f"      Training finished! Best validation loss: {fit_res['best_val_loss']:.4f} at epoch {fit_res['best_epoch']}")

    # 5. Temperature Calibration & Latent Manifold OOD Fitting
    print("\n[5/5] Calibrating probabilities & fitting Mahalanobis OOD manifold...")
    # Load best weights
    best_state = torch.load(best_ckpt_path, weights_only=False)
    model.load_state_dict(best_state["model_state_dict"])
    model.eval()

    # Extract validation logits for temperature scaling
    with torch.no_grad():
        val_x_t = torch.tensor(datamodule.X_val, dtype=torch.float32)
        val_y_t = torch.tensor(datamodule.Y_val, dtype=torch.float32)
        val_logits = model(val_x_t)

    calibrator = TemperatureScalingCalibrator(num_tasks=3)
    fitted_temps = calibrator.fit(val_logits, val_y_t)
    print(f"      Fitted Temperatures: T2DM: {fitted_temps['task_0_temperature']:.3f}, Nephropathy: {fitted_temps['task_1_temperature']:.3f}, ASCVD: {fitted_temps['task_2_temperature']:.3f}")

    # Latent OOD Detector
    with torch.no_grad():
        train_x_t = torch.tensor(datamodule.X_train, dtype=torch.float32)
        train_latents = model.extract_latent(train_x_t).numpy()

    ood_detector = MahalanobisOODDetector(alpha=0.95)
    ood_detector.fit(train_latents)
    print(f"      Mahalanobis OOD threshold (95th percentile): {ood_detector.threshold:.3f}")

    # Evaluate test set
    with torch.no_grad():
        test_x_t = torch.tensor(datamodule.X_test, dtype=torch.float32)
        test_logits = model(test_x_t)
        calibrated_test_logits = calibrator.calibrate_logits(test_logits)
        test_probs = torch.sigmoid(calibrated_test_logits).numpy()

    print("\n" + "=" * 70)
    print(f"{'ENDPOINT':<28} | {'AUROC':<10} | {'AUPRC':<10} | {'ECE (Calibrated)':<15}")
    print("-" * 70)
    task_labels = ["T2DM Onset (5-Yr)", "Nephropathy eGFR Decline", "ASCVD Risk (5-Yr)"]
    from sklearn.metrics import roc_auc_score, average_precision_score
    for k in range(3):
        y_true = datamodule.Y_test[:, k]
        p_pred = test_probs[:, k]
        auc = roc_auc_score(y_true, p_pred)
        prauc = average_precision_score(y_true, p_pred)
        ece = calculate_ece(p_pred, y_true, n_bins=15)
        print(f"{task_labels[k]:<28} | {auc:<10.4f} | {prauc:<10.4f} | {ece:<15.4f}")
    print("=" * 70)

    # Save complete production artifacts
    final_artifact_path = os.path.join(args.checkpoint_dir, "glycopulse_full_model.pt")
    torch.save({
        "model_state": model.state_dict(),
        "calibrator_state": calibrator.state_dict(),
        "ood_mean": ood_detector.mean,
        "ood_inv_cov": ood_detector.inv_cov,
        "ood_threshold": ood_detector.threshold,
        "feature_means": datamodule.mean,
        "feature_stds": datamodule.std,
        "d_model": model_cfg.d_model,
        "n_heads": model_cfg.n_heads,
        "num_layers": model_cfg.num_layers,
        "author": "MUHAMMED AFLAH S (RA2411026050147)"
    }, final_artifact_path)
    print(f"\n[OK] Model successfully packaged and saved to: {final_artifact_path}\n")


if __name__ == "__main__":
    main()
