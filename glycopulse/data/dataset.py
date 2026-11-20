"""
Clinical Cohort Dataset and NHANES Data Generator
Author: MUHAMMED AFLAH S (RA2411026050147)
"""

from typing import Tuple, Dict, Optional
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

from glycopulse.config import BIOMARKER_REGISTRY, FEATURE_NAMES, NUM_FEATURES


class ClinicalDataset(Dataset):
    """
    PyTorch Dataset wrapper for multi-task clinical biomarker tensors.
    """
    def __init__(self, x: np.ndarray, y: np.ndarray):
        assert len(x) == len(y), "Features and labels must have equal length"
        self.x = torch.tensor(x, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.x[idx], self.y[idx]


class NHANESCohortGenerator:
    """
    Generates synthetic, clinically realistic epidemiological cohorts mimicking
    the National Health and Nutrition Examination Survey (NHANES).
    
    Incorporates physiological correlations across glycemic, renal, lipid,
    and vascular biomarker clusters, generating multi-task risk outcomes.
    """
    def __init__(self, random_seed: int = 42):
        self.rng = np.random.default_rng(random_seed)

    def generate_cohort(
        self,
        n_samples: int = 5000,
        t2dm_prevalence: float = 0.28
    ) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
        """
        Synthesizes n_samples clinical records with 24 continuous biomarkers
        and 3 multi-task binary ground truth outcomes.
        
        Returns:
            X: Raw biomarker matrix of shape (n_samples, 24).
            Y: Binary outcome matrix of shape (n_samples, 3) [T2DM, Nephropathy, ASCVD].
            df: Human-readable pandas DataFrame.
        """
        # Demographic priors
        age = self.rng.uniform(22.0, 80.0, size=n_samples)
        family_hist = self.rng.choice([0.0, 1.0, 2.0], size=n_samples, p=[0.55, 0.35, 0.10])
        bmi = self.rng.normal(27.5, 5.5, size=n_samples).clip(16.0, 52.0)
        waist = bmi * 2.8 + self.rng.normal(12.0, 6.0, size=n_samples)

        # Latent metabolic syndrome factor
        latent_metabolic = (
            0.03 * (age - 50.0)
            + 0.08 * (bmi - 25.0)
            + 0.35 * family_hist
            + self.rng.normal(0.0, 0.8, size=n_samples)
        )

        # Glycemic cluster
        hba1c = 5.2 + 0.6 * np.maximum(0.0, latent_metabolic) + self.rng.normal(0.0, 0.4, size=n_samples)
        hba1c = hba1c.clip(4.2, 14.5)

        fasting_glucose = 85.0 + 15.0 * (hba1c - 5.0) + self.rng.normal(0.0, 8.0, size=n_samples)
        fasting_glucose = fasting_glucose.clip(60.0, 380.0)

        fasting_insulin = 8.0 + 2.5 * np.maximum(0.0, latent_metabolic) + self.rng.normal(0.0, 3.0, size=n_samples)
        fasting_insulin = fasting_insulin.clip(2.0, 95.0)

        homa_ir = (fasting_glucose * fasting_insulin) / 405.0
        glycated_albumin = 11.5 + 2.2 * (hba1c - 4.5) + self.rng.normal(0.0, 1.2, size=n_samples)

        # Renal cluster
        egfr = 105.0 - 0.7 * (age - 30.0) - 2.5 * np.maximum(0.0, latent_metabolic) + self.rng.normal(0.0, 8.0, size=n_samples)
        egfr = egfr.clip(15.0, 135.0)

        serum_creatinine = 0.9 + 18.0 / (egfr + 5.0) + self.rng.normal(0.0, 0.15, size=n_samples)
        serum_creatinine = serum_creatinine.clip(0.4, 6.5)

        uacr = np.exp(2.2 + 0.35 * np.maximum(0.0, latent_metabolic) + 0.02 * (age - 40.0) + self.rng.normal(0.0, 0.6, size=n_samples))
        uacr = uacr.clip(3.0, 2500.0)

        bun = 12.0 + 3.5 * (serum_creatinine - 0.8) + self.rng.normal(0.0, 2.5, size=n_samples)
        bun = bun.clip(5.0, 75.0)

        uric_acid = 4.5 + 0.05 * bmi + 0.3 * (serum_creatinine - 0.8) + self.rng.normal(0.0, 0.8, size=n_samples)

        # Lipid cluster
        triglycerides = 110.0 + 25.0 * np.maximum(0.0, latent_metabolic) + self.rng.normal(0.0, 30.0, size=n_samples)
        triglycerides = triglycerides.clip(45.0, 750.0)

        hdl = 60.0 - 0.3 * (triglycerides - 100.0) * 0.1 - 0.2 * bmi + self.rng.normal(0.0, 6.0, size=n_samples)
        hdl = hdl.clip(20.0, 105.0)

        ldl = 105.0 + 12.0 * np.maximum(0.0, latent_metabolic) + self.rng.normal(0.0, 22.0, size=n_samples)
        ldl = ldl.clip(40.0, 260.0)

        tot_chol = hdl + ldl + 0.2 * triglycerides
        apob = 0.8 * ldl + 0.1 * triglycerides + self.rng.normal(0.0, 8.0, size=n_samples)
        apob = apob.clip(35.0, 200.0)

        # Vascular cluster
        sbp = 115.0 + 0.45 * (age - 30.0) + 0.7 * (bmi - 22.0) + self.rng.normal(0.0, 10.0, size=n_samples)
        sbp = sbp.clip(85.0, 210.0)

        dbp = 72.0 + 0.2 * (age - 30.0) + 0.4 * (bmi - 22.0) + self.rng.normal(0.0, 7.0, size=n_samples)
        dbp = dbp.clip(50.0, 120.0)

        pulse_pressure = sbp - dbp
        hs_crp = np.exp(-0.2 + 0.06 * bmi + 0.02 * (sbp - 120.0) + self.rng.normal(0.0, 0.5, size=n_samples))
        hs_crp = hs_crp.clip(0.1, 35.0)

        wbc = 6.2 + 0.05 * bmi + 0.1 * hs_crp + self.rng.normal(0.0, 1.2, size=n_samples)
        wbc = wbc.clip(3.0, 20.0)

        # Assemble full 24-feature matrix strictly matching FEATURE_NAMES order
        features_dict = {
            "HbA1c": hba1c,
            "Fasting_Glucose": fasting_glucose,
            "Fasting_Insulin": fasting_insulin,
            "HOMA_IR": homa_ir,
            "Glycated_Albumin": glycated_albumin,
            "Serum_Creatinine": serum_creatinine,
            "eGFR_CKD_EPI": egfr,
            "UACR": uacr,
            "BUN": bun,
            "Uric_Acid": uric_acid,
            "Triglycerides": triglycerides,
            "HDL_C": hdl,
            "LDL_C": ldl,
            "Total_Cholesterol": tot_chol,
            "ApoB": apob,
            "Systolic_BP": sbp,
            "Diastolic_BP": dbp,
            "Pulse_Pressure": pulse_pressure,
            "hs_CRP": hs_crp,
            "WBC_Count": wbc,
            "Age": age,
            "BMI": bmi,
            "Waist_Circumference": waist,
            "Family_History_Score": family_hist
        }

        df = pd.DataFrame(features_dict)
        X = df[FEATURE_NAMES].to_numpy(dtype=np.float32)

        # Generate Ground Truth Clinical Outcomes
        # Task 1: 5-Year T2DM Onset (HbA1c >= 6.5 or fasting glucose >= 126 or metabolic risk)
        t2dm_logits = (
            2.4 * (hba1c - 6.2)
            + 0.03 * (fasting_glucose - 110.0)
            + 0.06 * (bmi - 28.0)
            + 0.4 * family_hist
            - 0.5
        )
        t2dm_prob = 1.0 / (1.0 + np.exp(-t2dm_logits))
        y_t2dm = (self.rng.uniform(size=n_samples) < t2dm_prob).astype(np.float32)

        # Task 2: Diabetic Nephropathy eGFR Decline (eGFR decline > 3 ml/min/year)
        neph_logits = (
            -0.05 * (egfr - 80.0)
            + 0.002 * (uacr - 50.0)
            + 1.2 * (hba1c - 6.5)
            + 0.02 * (sbp - 130.0)
            - 1.2
        )
        neph_prob = 1.0 / (1.0 + np.exp(-neph_logits))
        y_neph = (self.rng.uniform(size=n_samples) < neph_prob).astype(np.float32)

        # Task 3: 5-Year ASCVD Risk (Atherosclerotic Event)
        ascvd_logits = (
            0.05 * (age - 55.0)
            + 0.025 * (sbp - 130.0)
            + 0.015 * (apob - 95.0)
            - 0.03 * (hdl - 45.0)
            + 0.12 * hs_crp
            + 0.8 * y_t2dm
            - 1.4
        )
        ascvd_prob = 1.0 / (1.0 + np.exp(-ascvd_logits))
        y_ascvd = (self.rng.uniform(size=n_samples) < ascvd_prob).astype(np.float32)

        Y = np.stack([y_t2dm, y_neph, y_ascvd], axis=1)

        df["T2DM_Onset_Label"] = y_t2dm
        df["Nephropathy_Progression_Label"] = y_neph
        df["ASCVD_5Year_Label"] = y_ascvd

        return X, Y, df


class ClinicalDataModule:
    """
    Standardized data preparation pipeline providing train, validation, and test
    DataLoaders with z-score normalization parameters.
    """
    def __init__(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        val_split: float = 0.15,
        test_split: float = 0.15,
        batch_size: int = 64,
        random_seed: int = 42
    ):
        self.batch_size = batch_size
        n = len(X)
        rng = np.random.default_rng(random_seed)
        indices = rng.permutation(n)

        n_val = int(n * val_split)
        n_test = int(n * test_split)
        n_train = n - n_val - n_test

        train_idx = indices[:n_train]
        val_idx = indices[n_train:n_train + n_val]
        test_idx = indices[n_train + n_val:]

        # Raw partition splits
        self.X_train_raw = X[train_idx]
        self.Y_train = Y[train_idx]
        self.X_val_raw = X[val_idx]
        self.Y_val = Y[val_idx]
        self.X_test_raw = X[test_idx]
        self.Y_test = Y[test_idx]

        # Compute training mean and std for z-score normalization
        self.mean = np.mean(self.X_train_raw, axis=0)
        self.std = np.std(self.X_train_raw, axis=0) + 1e-7

        # Standardized feature matrices
        self.X_train = (self.X_train_raw - self.mean) / self.std
        self.X_val = (self.X_val_raw - self.mean) / self.std
        self.X_test = (self.X_test_raw - self.mean) / self.std

    def get_dataloaders(self) -> Tuple[DataLoader, DataLoader, DataLoader]:
        train_ds = ClinicalDataset(self.X_train, self.Y_train)
        val_ds = ClinicalDataset(self.X_val, self.Y_val)
        test_ds = ClinicalDataset(self.X_test, self.Y_test)

        train_loader = DataLoader(train_ds, batch_size=self.batch_size, shuffle=True, drop_last=True)
        val_loader = DataLoader(val_ds, batch_size=self.batch_size, shuffle=False)
        test_loader = DataLoader(test_ds, batch_size=self.batch_size, shuffle=False)

        return train_loader, val_loader, test_loader
