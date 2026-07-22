# econometric-game

# Predicting the Hotspots of Child Malnutrition

Submission for the **VSAE Econometric Game 2023** ("Case A"), organized by the University
of Amsterdam and VSAE. The task is to predict the level ("Low", "Medium", "High", "Very
High") of the **burden of wasting** (severe/moderate acute malnutrition) across
administrative regions of the G5 Sahel countries plus Senegal, using nutrition-survey and
INFORM risk-index data.

## Repository contents

| File | Description |
|---|---|
| `CASE_A.pdf` | Original case statement from the organizers: background on wasting/malnutrition, the prediction task, and data dictionaries (Tables 1–2). |
| `Econometric_Game___AMSE.pdf` | Final report submitted for the case, including methodology, exploratory analysis, model comparison, and SHAP-based interpretation. |
| `Data_Case_A.csv` | Raw dataset as provided by the organizers (70 columns, incl. `target`, admin-level identifiers, nutrition-survey and INFORM features). |
| `Database_V6.csv` | Cleaned/preprocessed version of the dataset (167 columns after feature engineering / one-hot encoding, used as input to the modelling notebook). |
| `0__NOTEBOOK.ipynb` | Main analysis notebook: dimensionality reduction (Lasso / Elastic-Net), model training (Random Forest, SVM, KNN, XGBoost), evaluation, SHAP interpretation, and ROC curves. |
| `gamePresentation.pptx` | Presentation given by the organizers. |

## Task summary

- **Target**: multiclass malnutrition burden level — `Low` (0), `Medium` (1), `High` (2), `Very High` (3).
- **Unit of observation**: administrative region (admin1/admin2) × year, across Burkina Faso, Chad, Mali, Mauritania, Niger, and Senegal.
- **Features**: nutrition-survey indicators (GAM/SAM/MAM prevalence, stunting, deworming, measles, diarrhea, etc.) and INFORM risk-index indicators (access to health care, food insecurity, internally displaced persons, flood/drought exposure, physical infrastructure, etc.).

## Methodology

1. **Preprocessing**: missing-value handling (variables with >50% missing dropped; remaining gaps imputed via KNN hot-deck), removal of countries with insufficient data, one-hot encoding, and scaling.
2. **Dimensionality reduction**: Lasso and Elastic-Net regularization (hyperparameters tuned via `GridSearchCV`) to address multicollinearity and reduce the feature set; Elastic-Net was selected as the final reduction method.
3. **Modelling**: four classifiers trained and tuned on an 80/20 train-test split:
   - Random Forest
   - Support Vector Machine (RBF kernel)
   - K-Nearest Neighbors
   - XGBoost
4. **Evaluation**: accuracy, MSE, confusion matrix, and one-vs-rest ROC/AUC curves.
5. **Interpretation**: SHAP values (variable importance, beeswarm, force, and waterfall plots) to identify the key drivers of malnutrition risk.

## Results

| Model | Test MSE | Accuracy |
|---|---|---|
| **XGBoost** | **0.118** | **89.5%** |
| Random Forest | 0.147 | 86.6% |
| K-Nearest Neighbors | 0.189 | 86.1% |
| Support Vector Machine | 0.189 | 83.6% |

XGBoost was selected as the final model. SHAP analysis identified **access to health care,
food insecurity, internal displacement, and malaria-related mortality** as the most
influential predictors of malnutrition burden.

## Reproducing the analysis

The notebook expects a cleaned dataset (`Database_V6.csv`) and relies on:

pandas, numpy, scikit-learn, xgboost, shap, yellowbrick, matplotlib

Run the cells in order: data loading → Lasso/Elastic-Net feature selection → train/test
split → model training and tuning (Random Forest, SVM, KNN, XGBoost) → evaluation → SHAP
plots → ROC curves.

## Note

Per competition rules, the report deliberately omits author names and university
affiliation to preserve jury anonymity.

