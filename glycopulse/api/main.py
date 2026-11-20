"""
GLYCOPULSE-NET Clinical Microservice API (FastAPI)
Author: MUHAMMED AFLAH S (RA2411026050147)
"""

from typing import Dict, Any, List, Optional
import numpy as np
import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from glycopulse.config import (
    FEATURE_NAMES,
    NUM_FEATURES,
    BIOMARKER_REGISTRY,
    ModelConfig
)
from glycopulse.model.network import GlycoPulseNet
from glycopulse.calibration.temperature import TemperatureScalingCalibrator
from glycopulse.calibration.ood import MahalanobisOODDetector
from glycopulse.inference.predictor import GlycoPulsePredictor
from glycopulse.data.dataset import NHANESCohortGenerator


app = FastAPI(
    title="GLYCOPULSE-NET Clinical Decision Support API",
    description="Multi-Task Deep Learning for Type 2 Diabetes Stratification, Multi-Organ Microvascular Complications, and Counterfactual Recourse.",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model instance
predictor: Optional[GlycoPulsePredictor] = None


class PatientBiomarkersSchema(BaseModel):
    HbA1c: float = Field(7.4, description="Glycated Hemoglobin (%)", ge=3.5, le=16.0)
    Fasting_Glucose: float = Field(142.0, description="Fasting Blood Glucose (mg/dL)", ge=40.0, le=450.0)
    Fasting_Insulin: float = Field(18.5, description="Fasting Serum Insulin (uIU/mL)", ge=1.0, le=150.0)
    HOMA_IR: float = Field(6.47, description="Insulin Resistance Index", ge=0.2, le=25.0)
    Glycated_Albumin: float = Field(17.8, description="Glycated Albumin (%)", ge=8.0, le=45.0)
    Serum_Creatinine: float = Field(1.15, description="Serum Creatinine (mg/dL)", ge=0.3, le=12.0)
    eGFR_CKD_EPI: float = Field(74.0, description="eGFR (mL/min/1.73m2)", ge=10.0, le=140.0)
    UACR: float = Field(68.0, description="Urinary Albumin-Creatinine Ratio (mg/g)", ge=1.0, le=5000.0)
    BUN: float = Field(18.0, description="Blood Urea Nitrogen (mg/dL)", ge=3.0, le=100.0)
    Uric_Acid: float = Field(6.4, description="Serum Uric Acid (mg/dL)", ge=1.5, le=16.0)
    Triglycerides: float = Field(195.0, description="Serum Triglycerides (mg/dL)", ge=30.0, le=1000.0)
    HDL_C: float = Field(38.0, description="HDL Cholesterol (mg/dL)", ge=15.0, le=120.0)
    LDL_C: float = Field(138.0, description="LDL Cholesterol (mg/dL)", ge=20.0, le=300.0)
    Total_Cholesterol: float = Field(215.0, description="Total Cholesterol (mg/dL)", ge=70.0, le=450.0)
    ApoB: float = Field(112.0, description="Apolipoprotein B (mg/dL)", ge=30.0, le=220.0)
    Systolic_BP: float = Field(138.0, description="Systolic Blood Pressure (mmHg)", ge=70.0, le=220.0)
    Diastolic_BP: float = Field(88.0, description="Diastolic Blood Pressure (mmHg)", ge=40.0, le=130.0)
    Pulse_Pressure: float = Field(50.0, description="Pulse Pressure (mmHg)", ge=15.0, le=110.0)
    hs_CRP: float = Field(3.8, description="hs-CRP (mg/L)", ge=0.05, le=50.0)
    WBC_Count: float = Field(8.1, description="White Blood Cell Count (10^3/uL)", ge=2.0, le=30.0)
    Age: float = Field(52.0, description="Patient Age (years) [Immutable]", ge=18.0, le=95.0)
    BMI: float = Field(31.2, description="Body Mass Index (kg/m2)", ge=14.0, le=60.0)
    Waist_Circumference: float = Field(102.0, description="Waist Circumference (cm)", ge=50.0, le=160.0)
    Family_History_Score: float = Field(1.0, description="Familial Risk Score [Immutable]", ge=0.0, le=3.0)


class RecourseRequestSchema(BaseModel):
    biomarkers: PatientBiomarkersSchema
    target_risk_threshold: float = Field(0.20, ge=0.05, le=0.50)
    target_endpoint: str = Field("T2DM", description="Endpoint to target: 'T2DM', 'Nephropathy', or 'ASCVD'")


@app.on_event("startup")
def startup_event():
    global predictor
    # Initialize model with calibrated cohort priors
    torch.manual_seed(42)
    np.random.seed(42)

    gen = NHANESCohortGenerator(random_seed=42)
    X, Y, _ = gen.generate_cohort(n_samples=500)

    means = np.mean(X, axis=0)
    stds = np.std(X, axis=0) + 1e-7

    model = GlycoPulseNet(ModelConfig())
    calibrator = TemperatureScalingCalibrator(num_tasks=3)
    ood_detector = MahalanobisOODDetector()

    # Pre-fit latent OOD detector on normalized sample
    X_norm = (X - means) / stds
    with torch.no_grad():
        latents = model.extract_latent(torch.tensor(X_norm, dtype=torch.float32)).numpy()
    ood_detector.fit(latents)

    predictor = GlycoPulsePredictor(
        model=model,
        feature_means=means,
        feature_stds=stds,
        calibrator=calibrator,
        ood_detector=ood_detector
    )


@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "service": "GLYCOPULSE-NET Inference Engine",
        "version": "1.0.0",
        "author": "MUHAMMED AFLAH S (RA2411026050147)",
        "features_tracked": NUM_FEATURES,
        "device": str(predictor.device) if predictor else "uninitialized"
    }


@app.get("/biomarkers")
def get_biomarker_catalog():
    """
    Returns clinical biomarker schema including physiological ranges, units, and actionability.
    """
    return [
        {
            "name": m.name,
            "category": m.category,
            "unit": m.unit,
            "reference_range": f"{m.healthy_min} - {m.healthy_max} {m.unit}",
            "is_actionable": m.is_actionable,
            "description": m.description
        }
        for m in BIOMARKER_REGISTRY
    ]


@app.post("/predict")
def predict_patient_risk(biomarkers: PatientBiomarkersSchema, mc_samples: int = 25):
    """
    Computes multi-task clinical risk scores with Bayesian epistemic uncertainty and OOD check.
    """
    if predictor is None:
        raise HTTPException(status_code=503, detail="Predictor engine not initialized.")

    raw_dict = biomarkers.dict()
    result = predictor.predict(
        raw_biomarkers=raw_dict,
        mc_samples=mc_samples,
        generate_recourse=False
    )
    return result


@app.post("/recourse")
def compute_counterfactual_recourse(request: RecourseRequestSchema):
    """
    Computes optimal actionable counterfactual intervention trajectory to reduce clinical risk.
    """
    if predictor is None:
        raise HTTPException(status_code=503, detail="Predictor engine not initialized.")

    endpoint_map = {"T2DM": 0, "Nephropathy": 1, "ASCVD": 2}
    task_idx = endpoint_map.get(request.target_endpoint, 0)

    raw_dict = request.biomarkers.dict()
    x_raw = predictor._parse_input(raw_dict)

    recourse_plan = predictor.recourse_engine.optimize_recourse(
        patient_biomarkers=x_raw,
        target_task_idx=task_idx,
        target_risk_threshold=request.target_risk_threshold
    )
    return recourse_plan
