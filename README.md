# Privacy-Preserving Cloud Intelligence Platform using Federated Learning
### Applied to: Insider Threat / Access Anomaly Detection

## Project Title
Privacy-Preserving Cloud Intelligence Platform using Federated Learning — Applied to Insider Threat / Access Anomaly Detection

## Team Members
Individual Project (BITE412L — Cloud Computing, Instructor: Dr. Priya V)
- Paul — sole contributor (literature survey, research gap analysis, Azure infrastructure, AI/ML pipeline, documentation)

## Problem Statement
Insider threats — authorized users misusing legitimate access — account for a large share of cloud data breaches, yet most existing detection systems require centralizing sensitive access logs, which itself creates a privacy risk when multiple departments are unwilling to share raw behavioral data. Most published federated-learning security research also targets IoT/network traffic rather than insider-threat detection on cloud document systems, and largely assumes privacy is achieved simply by keeping data local rather than engineering it in.

## Objectives
1. Design a federated learning architecture that trains insider-threat detection models across simulated departments without centralizing raw Azure Blob Storage access logs.
2. Implement differential-privacy-based aggregation to provide measurable privacy guarantees beyond simple data locality.
3. Achieve reliable anomaly detection (target: recall >= 85% on labeled malicious-insider events) using Isolation Forest on engineered behavioral features.
4. Handle non-IID department behavior through personalized local models alongside the shared global model.
5. Provide explainable, per-event justifications for flagged anomalies using SHAP.
6. Visualize risk-ranked results through a lightweight, read-only Power BI dashboard.

## Proposed Architecture / Framework
Two architecture diagrams are provided in the `architecture/` folder:
- **Diagram 1 — Azure Cloud Architecture:** shows how Azure services interact (data flow, storage, authentication, processing, monitoring, notifications).
- **Diagram 2 — Complete System Architecture:** shows the full federated learning workflow, from raw dataset partitioning through local training, privacy-preserving aggregation, and explainability output.

## Technology Stack
- **Cloud:** Microsoft Azure — Blob Storage, Microsoft Entra ID / RBAC, Key Vault, Azure ML Notebook, Azure Functions, Azure Monitor
- **AI/ML:** Python, scikit-learn (Isolation Forest), federated averaging (FedAvg), differential privacy, SHAP
- **Visualization:** Power BI
- **Dataset:** CMU CERT Insider Threat Test Dataset r4.2

## Dataset Details
CMU CERT Insider Threat Test Dataset (r4.2) — ~32.7 million activity events across 1,000 synthetic users over 17 months, including 70 pre-labeled malicious insiders. Full details in `dataset/README.md`.

## Repository Structure
See folder-level README files for details on each directory:
- `docs/` - abstract, objectives, novelty summary
- `literature_survey/` - 15-paper survey and research gap analysis
- `architecture/` - architecture diagrams
- `dataset/` - CERT r4.2 partitioning scripts and dataset details
- `src/ai_model/` - feature engineering, Isolation Forest, FedAvg, DP, SHAP
- `src/azure/` - Blob/RBAC/Key Vault setup scripts
- `src/dashboard/` - Power BI files
- `results/` - exported metrics and reports
- `presentation/` - review slides
- `references/` - citation exports
