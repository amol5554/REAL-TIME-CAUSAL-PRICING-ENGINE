# 🚀 Real-Time Causal Pricing Engine

### End-to-End Intelligent Pricing System for E-Commerce

A production-style pricing intelligence system that combines **Machine Learning**, **Causal Inference**, **Online Learning**, and **Real-Time Serving** to generate adaptive and revenue-optimized product prices.

---

## 📌 Project Overview

Traditional pricing systems rely on static rules or purely correlation-based machine learning models.

This project combines:

- **XGBoost** for price prediction
- **DoWhy** for causal demand modeling
- **River UCB** for online learning
- **FastAPI** for real-time inference
- **Redis** for low-latency caching
- **Evidently** for drift monitoring
- **Streamlit** for visualization

The system estimates how price changes affect demand and continuously adapts pricing decisions using feedback-driven optimization.

---

## 🏗️ System Architecture

<img src="assets/SYSTEM OVERVIEW - REAL-TIME CAUSAL PRICING ENGINE.png" width="100%">

---

## 🔄 Pricing Pipeline

```text
Competitor Data
        │
        ▼
Feature Engineering
        │
        ▼
XGBoost Price Prediction
        │
        ▼
DoWhy Causal Adjustment
        │
        ▼
Business Constraints
        │
        ▼
River UCB Optimization
        │
        ▼
Final Recommended Price
```

---

## ⚙️ Technology Stack

| Component | Technology |
|------------|------------|
| Machine Learning | XGBoost |
| Causal Inference | DoWhy |
| Online Learning | River UCB |
| Backend API | FastAPI |
| Dashboard | Streamlit |
| Caching | Redis |
| Drift Monitoring | Evidently |
| Explainability | SHAP |
| Data Processing | Pandas, NumPy |

---

## 📂 Project Structure

```text
REAL-TIME-CAUSAL-PRICING-ENGINE
│
├── assets/
│   ├── Presentation Slides
│   └── Architecture Images
│
├── data/
│   └── Dataset Files
│
├── models/
│   └── Trained Model Artifacts
│
├── notebooks/
│   ├── 01_causal_eda.ipynb
│   ├── 02_xgb_model.ipynb
│   ├── 03_causal_analysis.ipynb
│   └── 04_online_bandit.ipynb
│
├── api.py
├── scraper.py
├── dashboard.py
├── drift_monitor.py
│
└── README.md
```

---

## 🧠 Training Pipeline

| Notebook | Purpose |
|-----------|----------|
| 01_causal_eda.ipynb | Data cleaning, feature engineering, competitor pricing generation |
| 02_xgb_model.ipynb | XGBoost model training and SHAP explainability |
| 03_causal_analysis.ipynb | DoWhy causal inference and elasticity estimation |
| 04_online_bandit.ipynb | River UCB online learning and adaptive pricing |

---

## 📦 Model Artifacts

The training pipeline generates the following artifacts:

```text
models/
│
├── xgb_model.pkl
├── xgb_model.json
├── causal_state.json
├── bandit_state.json
├── category_map.json
└── model_meta.json
```

These artifacts are loaded by the FastAPI inference engine during runtime.

---

## ✨ Key Features

### 🤖 XGBoost Price Prediction

Predicts the baseline product price using engineered competitor and product features.

### 📈 DoWhy Causal Inference

Estimates price elasticity and models the causal effect of price changes on demand.

### 🎰 River UCB Online Learning

Continuously adapts pricing strategy using revenue-based feedback.

### ⚡ FastAPI Inference Engine

Serves pricing recommendations through REST APIs.

### 🧠 Redis Cache

Reduces latency by caching live product data and prediction results.

### 📉 Evidently Drift Monitoring

Detects feature distribution shifts and supports retraining workflows.

### 📊 Streamlit Dashboard

Provides interactive pricing analysis and visualization.

---

## 🚀 Running the Project

### 1. Start Redis

```bash
redis-server
```

### 2. Start Live Scraper

```bash
python scraper.py
```

### 3. Start FastAPI Server

```bash
uvicorn api:app --reload --port 8000
```

API Documentation:

```text
http://localhost:8000/docs
```

### 4. Launch Dashboard

```bash
streamlit run dashboard.py
```

Dashboard:

```text
http://localhost:8501
```

---

## 📌 Repository Notes

The following are intentionally excluded from the repository:

```text
venv/
Redis installation binaries
__pycache__/
Temporary cache files
```

Redis must be installed separately if caching functionality is required.

---

# 📑 Project Presentation

<details>
<summary><b>Click to View Complete Presentation</b></summary>

<br>

<img src="assets/Slide1.JPG" width="100%">
<img src="assets/Slide2.JPG" width="100%">
<img src="assets/Slide3.JPG" width="100%">
<img src="assets/Slide4.JPG" width="100%">
<img src="assets/Slide5.JPG" width="100%">
<img src="assets/Slide6.JPG" width="100%">
<img src="assets/Slide7.JPG" width="100%">
<img src="assets/Slide8.JPG" width="100%">
<img src="assets/Slide9.JPG" width="100%">
<img src="assets/Slide10.JPG" width="100%">
<img src="assets/Slide11.JPG" width="100%">
<img src="assets/Slide12.JPG" width="100%">
<img src="assets/Slide13.JPG" width="100%">
<img src="assets/Slide14.JPG" width="100%">
<img src="assets/Slide15.JPG" width="100%">
<img src="assets/Slide16.JPG" width="100%">
<img src="assets/Slide17.JPG" width="100%">
<img src="assets/Slide18.JPG" width="100%">
<img src="assets/Slide19.JPG" width="100%">
<img src="assets/Slide20.JPG" width="100%">
<img src="assets/Slide21.JPG" width="100%">

</details>

---

## 🎯 Skills Demonstrated

- Machine Learning
- Causal Inference
- Online Learning
- Feature Engineering
- FastAPI
- Redis
- Streamlit
- SHAP Explainability
- Evidently Drift Detection
- MLOps Concepts
- End-to-End ML System Design

---

## 📄 License

This project is intended for educational, research, and portfolio purposes.
