# GLYCOPULSE-NET

> **Multi-Task Clinical Risk Stratification & Differentiable Counterfactual Recourse**  
> **Author**: MUHAMMED AFLAH S  
> **Registration Number**: RA2411026050147  
> **Department**: Computer Science and Engineering (AIML)  
> **Institution**: SRM Institute of Science and Technology, Tiruchirappalli  
> **Repository**: [https://github.com/aflahs/ANN_Project_Submission](https://github.com/aflahs/ANN_Project_Submission)

---

## 1. Project Overview

GLYCOPULSE-NET is a multi-task deep neural network architecture designed for tabular clinical biomarker analysis. It addresses key clinical requirements:

1. **Multi-Task Risk Prediction**: Predicts 5-year Type 2 Diabetes (T2DM) onset, Diabetic Nephropathy progression, and Atherosclerotic Cardiovascular Disease (ASCVD) risk simultaneously.
2. **Homoscedastic Uncertainty Balancing**: Dynamically balances multi-task loss gradients using learnable observation uncertainties.
3. **Counterfactual Clinical Recourse**: Computes actionable, patient-specific lifestyle and biomarker changes to reduce predicted risk while respecting immutable physical constraints.
4. **Reliability & OOD Screening**: Integrates post-hoc Temperature Scaling calibration and Mahalanobis latent space anomaly detection.

---

## 2. Repository Layout

```text
ANN_Project_Submission/
├── glycopulse/
│   ├── __init__.py               # Package definitions and exports
│   ├── config.py                 # 24 Clinical biomarkers and system hyperparameters
│   ├── model/
│   │   ├── tokenization.py       # Biomarker Gated Tokenization (BGT) with SwiGLU gating
│   │   ├── attention.py          # Residual cross-attention & latent bottleneck pooling
│   │   └── network.py            # GlycoPulseNet master model with decoupled heads
│   ├── loss/
│   │   └── uncertainty_loss.py   # Multi-task loss with learnable task log-variances
│   ├── recourse/
│   │   └── optimizer.py          # Differentiable counterfactual recourse optimizer
│   ├── calibration/
│   │   ├── temperature.py        # Temperature scaling & Expected Calibration Error (ECE)
│   │   └── ood.py                # Mahalanobis out-of-distribution detector
│   ├── data/
│   │   └── dataset.py            # Synthetic clinical cohort generator and PyTorch DataLoader
│   ├── training/
│   │   └── trainer.py            # Model training pipeline with early stopping and metrics
│   ├── inference/
│   │   └── predictor.py          # Clinical inference pipeline with uncertainty quantification
│   └── api/
│       └── main.py               # FastAPI microservice exposing prediction and recourse endpoints
├── scripts/
│   ├── train.py                  # CLI training script
│   ├── evaluate.py               # Benchmark comparison script
│   └── run_recourse.py           # Counterfactual recourse demo
├── tests/
│   └── test_pipeline.py          # Automated PyTest unit tests
├── requirements.txt              # Core dependencies
├── setup.py                      # Package installation script
└── pyproject.toml                # Build configuration
```

---

## 3. Installation

Clone this repository and install the dependencies:

```bash
git clone https://github.com/aflahs/ANN_Project_Submission.git
cd ANN_Project_Submission
pip install -r requirements.txt
pip install -e .
```

---

## 4. Running the Tests

To run the automated test suite verifying all tokenizers, attention layers, losses, and constraints:

```bash
pytest tests/test_pipeline.py -v
```

---

## 5. Usage

### 5.1 Training the Model

To train the multi-task model on synthetic cohort data:

```bash
python scripts/train.py --epochs 30 --batch-size 64 --lr 0.0003
```

For a rapid verification run:

```bash
python scripts/train.py --quick-test
```

### 5.2 Benchmarking Against Baselines

Compare GLYCOPULSE-NET against Logistic Regression, Deep MLP, and Gradient Boosting:

```bash
python scripts/evaluate.py --samples 1000
```

### 5.3 Generating Counterfactual Recourse

To compute personalized biomarker interventions for a high-risk patient profile:

```bash
python scripts/run_recourse.py --endpoint T2DM --target-risk 0.20
```

### 5.4 Starting the FastAPI Clinical Microservice

Launch the REST API server:

```bash
uvicorn glycopulse.api.main:app --host 0.0.0.0 --port 8000 --reload
```

Interactive documentation will be available at:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

---

## 6. Academic Attribution

- **Student Name**: MUHAMMED AFLAH S
- **Registration Number**: RA2411026050147
- **Program**: B.Tech Computer Science and Engineering (Artificial Intelligence and Machine Learning)
- **Institution**: SRM Institute of Science and Technology, Tiruchirappalli
