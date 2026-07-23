# Privacy-Preserving Cloud Intelligence Platform using Federated Learning
### Applied to: Insider Threat / Access Anomaly Detection

BITE412L Cloud Computing — Individual Project (Instructor: Dr. Priya V)

## Overview
A federated learning pipeline that detects insider-threat access anomalies across simulated organizational departments without centralizing raw access logs. Each department trains locally on its own Azure Blob Storage data; only model weight updates (with differential-privacy noise) are shared for aggregation.

## Structure
- `docs/` - abstract, objectives, novelty summary
- `literature_survey/` - 15-paper survey + research gap analysis
- `architecture/` - architecture diagrams
- `dataset/` - CERT r4.2 partitioning scripts + dataset details
- `src/ai_model/` - feature engineering, Isolation Forest, FedAvg, DP, SHAP
- `src/azure/` - Blob/RBAC/Key Vault setup scripts
- `src/dashboard/` - Power BI files
- `results/` - exported metrics and reports
- `presentation/` - review slides
- `references/` - citation exports

## Tech Stack
Azure Blob Storage, Azure RBAC (Entra ID), Azure Key Vault, Azure ML Notebook, Power BI
