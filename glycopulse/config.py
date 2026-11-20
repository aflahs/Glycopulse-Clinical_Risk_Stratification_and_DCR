"""
GLYCOPULSE-NET Configuration and Clinical Biomarker Schema
Author: MUHAMMED AFLAH S (RA2411026050147)
SRM Institute of Science and Technology, Tiruchirappalli
Department of Computer Science and Engineering (AIML)
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any


@dataclass
class BiomarkerMetadata:
    name: str
    category: str
    unit: str
    healthy_min: float
    healthy_max: float
    physio_min: float
    physio_max: float
    is_actionable: bool
    desired_direction: int  # -1 = decrease desirable, +1 = increase desirable, 0 = stable/immutable
    description: str


# 24 Continuous Clinical Biomarkers categorized by pathophysiological domain
BIOMARKER_REGISTRY: List[BiomarkerMetadata] = [
    # --- Glycemic & Metabolic Regulation ---
    BiomarkerMetadata("HbA1c", "Glycemic", "%", 4.0, 5.6, 3.5, 16.0, True, -1, "Glycated Hemoglobin A1c"),
    BiomarkerMetadata("Fasting_Glucose", "Glycemic", "mg/dL", 70.0, 99.0, 40.0, 450.0, True, -1, "Fasting Blood Glucose"),
    BiomarkerMetadata("Fasting_Insulin", "Glycemic", "uIU/mL", 2.6, 24.9, 1.0, 150.0, True, -1, "Fasting Serum Insulin"),
    BiomarkerMetadata("HOMA_IR", "Glycemic", "index", 0.5, 1.9, 0.2, 25.0, True, -1, "Homeostatic Model Assessment of Insulin Resistance"),
    BiomarkerMetadata("Glycated_Albumin", "Glycemic", "%", 11.0, 16.0, 8.0, 45.0, True, -1, "Intermediate-term Glycemic Indicator"),

    # --- Renal Function & Microvascular Dynamics ---
    BiomarkerMetadata("Serum_Creatinine", "Renal", "mg/dL", 0.6, 1.2, 0.3, 12.0, True, -1, "Serum Creatinine Level"),
    BiomarkerMetadata("eGFR_CKD_EPI", "Renal", "mL/min/1.73m2", 90.0, 120.0, 10.0, 140.0, True, 1, "Estimated Glomerular Filtration Rate"),
    BiomarkerMetadata("UACR", "Renal", "mg/g", 0.0, 30.0, 1.0, 5000.0, True, -1, "Urinary Albumin-to-Creatinine Ratio"),
    BiomarkerMetadata("BUN", "Renal", "mg/dL", 7.0, 20.0, 3.0, 100.0, True, -1, "Blood Urea Nitrogen"),
    BiomarkerMetadata("Uric_Acid", "Renal", "mg/dL", 3.5, 7.2, 1.5, 16.0, True, -1, "Serum Uric Acid"),

    # --- Lipid Metabolism & Atherogenic Profiling ---
    BiomarkerMetadata("Triglycerides", "Lipid", "mg/dL", 50.0, 150.0, 30.0, 1000.0, True, -1, "Serum Triglycerides"),
    BiomarkerMetadata("HDL_C", "Lipid", "mg/dL", 40.0, 60.0, 15.0, 120.0, True, 1, "High-Density Lipoprotein Cholesterol"),
    BiomarkerMetadata("LDL_C", "Lipid", "mg/dL", 50.0, 100.0, 20.0, 300.0, True, -1, "Low-Density Lipoprotein Cholesterol"),
    BiomarkerMetadata("Total_Cholesterol", "Lipid", "mg/dL", 120.0, 200.0, 70.0, 450.0, True, -1, "Total Serum Cholesterol"),
    BiomarkerMetadata("ApoB", "Lipid", "mg/dL", 55.0, 90.0, 30.0, 220.0, True, -1, "Apolipoprotein B"),

    # --- Hemodynamic & Chronic Inflammation ---
    BiomarkerMetadata("Systolic_BP", "Vascular", "mmHg", 90.0, 120.0, 70.0, 220.0, True, -1, "Systolic Blood Pressure"),
    BiomarkerMetadata("Diastolic_BP", "Vascular", "mmHg", 60.0, 80.0, 40.0, 130.0, True, -1, "Diastolic Blood Pressure"),
    BiomarkerMetadata("Pulse_Pressure", "Vascular", "mmHg", 30.0, 50.0, 15.0, 110.0, True, -1, "Pulse Pressure (SBP - DBP)"),
    BiomarkerMetadata("hs_CRP", "Vascular", "mg/L", 0.1, 1.0, 0.05, 50.0, True, -1, "High-Sensitivity C-Reactive Protein"),
    BiomarkerMetadata("WBC_Count", "Vascular", "10^3/uL", 4.5, 11.0, 2.0, 30.0, True, -1, "White Blood Cell Count"),

    # --- Anthropometric & Demographic (Contextual) ---
    BiomarkerMetadata("Age", "Demographic", "years", 18.0, 80.0, 18.0, 95.0, False, 0, "Patient Age (Immutable)"),
    BiomarkerMetadata("BMI", "Anthropometric", "kg/m2", 18.5, 24.9, 14.0, 60.0, True, -1, "Body Mass Index"),
    BiomarkerMetadata("Waist_Circumference", "Anthropometric", "cm", 70.0, 94.0, 50.0, 160.0, True, -1, "Abdominal Adiposity"),
    BiomarkerMetadata("Family_History_Score", "Demographic", "score", 0.0, 1.0, 0.0, 3.0, False, 0, "Familial Diabetes Burden (Immutable)")
]

FEATURE_NAMES: List[str] = [m.name for m in BIOMARKER_REGISTRY]
NUM_FEATURES: int = len(FEATURE_NAMES)  # 24

# Boolean mask indicating actionable features (modifiable via clinical recourse)
ACTIONABLE_MASK: List[bool] = [m.is_actionable for m in BIOMARKER_REGISTRY]

# Desired direction vector (-1, 0, +1)
DIRECTION_VECTOR: List[int] = [m.desired_direction for m in BIOMARKER_REGISTRY]


@dataclass
class ModelConfig:
    num_features: int = NUM_FEATURES
    d_model: int = 128
    n_heads: int = 4
    num_layers: int = 4
    dim_feedforward: int = 256
    dropout: float = 0.15
    mc_dropout: float = 0.20
    num_tasks: int = 3
    task_names: List[str] = field(default_factory=lambda: [
        "T2DM_Onset_Risk",
        "Nephropathy_Progression",
        "ASCVD_5Year_Risk"
    ])


@dataclass
class TrainingConfig:
    batch_size: int = 64
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    epochs: int = 50
    patience: int = 10
    grad_clip: float = 1.0
    val_split: float = 0.15
    test_split: float = 0.15
    random_seed: int = 42


@dataclass
class RecourseConfig:
    learning_rate: float = 0.05
    max_steps: int = 200
    target_risk_threshold: float = 0.20
    sparsity_lambda: float = 0.08
    manifold_lambda: float = 0.04
    tolerance: float = 1e-4
