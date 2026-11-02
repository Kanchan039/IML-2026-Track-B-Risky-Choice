# IML Midterm Project: Track B (Behavioural Economics), predicting risky choices

CS-3410 / MAT-3211-1, Ashoka University, Monsoon 2026
**Name:** Kanchan Bharti | **Kaggle username:** kanchanbharti039 | **Track:** B (simulated risky choice)

> All data are **simulated**.

## 1. What this project does
For each decision, predict the probability that a simulated participant chooses the risky option (`chose_risky`). The metric is ROC-AUC. Validation always holds out whole participants (`GroupKFold` on `participant_id`), because the test set contains 90 participants not seen in training. The final model is an L2-regularised logistic regression (`C=0.1`) on nominal, expected-value, domain/frame and prospect-theory features.

Results:

| Model | ROC-AUC |
|---|---|
| Expected-value rule | 0.727 |
| Logistic regression, nominal features | 0.721 |
| + EV features, domain, frame | 0.812 |
| Final model (+ prospect-theory features) | **0.817** (fold SD 0.018) |


## 2. Files

| File | Content |
|---|---|
| `IML_Mid_term_Project_Track_B.ipynb` | All code: data checks, features, baselines, experiments, plots, final model, prediction file |
| `train.csv`, `test.csv`, `sample_submission.csv` | Competition data (not modified; put them in the same folder as the notebook) |
| `Track_B_submission.csv` | Prediction file, produced by the notebook, same format as `sample_submission.csv` (probabilities in `chose_risky`) |
| `Track_b_experimental_results.csv` | Every run reported (one row per run), produced by the notebook |
| `figures/` | Figures produced by the notebook (see map in section 6) |
| `report.tex`, `report.pdf` | Report  |
| `requirements.txt` | Python packages |

## 3. Environment
- Python 3.10.11
- Packages as in `requirements.txt` (numpy 2.2.6, pandas 2.3.3, scikit-learn 1.7.2 as printed by the notebook, plus matplotlib and Jupyter)
- CPU only; no GPU, no network access needed.
- Only classical methods from the lectures are used (logistic regression, LDA, Gaussian Naive Bayes, linear SVM). No neural networks, no pretrained models.

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## 4. How to reproduce (one command)
Put `train.csv`, `test.csv`, `sample_submission.csv` next to `IML_Mid_term_Project_Track_B.ipynb`, then run:

```bash
jupyter nbconvert --to notebook --execute --IML_Mid_term_Project_Track_B.ipynb
```

This regenerates `Track_b_submission.csv`, `Track_b_experimental_results.csv`. (Alternatively open the notebook and use *Run All*.) To rebuild the report PDF:

```bash
pdflatex report.tex && pdflatex report.tex
```

The data folder is set in the first code cell (`folder = "."`, i.e. the notebook's own folder).

## 5. Seeds and determinism
- `np.random.seed(0)` in the first code cell.
- Random split in Experiment 2: `KFold(n_splits=8, shuffle=True, random_state=0)`.
- `GroupKFold`, `LogisticRegression` (lbfgs), LDA, Gaussian NB and `LinearSVC` have no random component here, so reruns give the same numbers.
- Expected runtime: about 15-30 seconds on a laptop-class CPU.

## 6. Map from report tables and figures to code
Section numbers are the headings of the notebook.

| Report item | Notebook section | Output file |
|---|---|---|
| Data statistics (Section 3 of the report: sizes, 210/90 participants, 48.2% risky, leakage audit) | 2. Check the data and remove leakage | printed output |
| Figure 1 (left): risky share by domain/frame | 2. cell 3 | `figures/fig1_choice_by_cell.png` |
| Features, including prospect-theory features | 3. Make new features | none |
| Table 1: baselines | 5. Baselines | `experimental_results.csv` (Baseline rows) |
| Table 2, runs 1a--1f | 6. Experiment 1 | `experimental_results.csv` (Exp1 rows) |
| Table 2, runs 2a--2d | 7. Experiment 2 | `experimental_results.csv` (Exp2 rows) |
| Table 2, run 3 and response-time means | 8. Experiment 3 | `experimental_results.csv` (Exp3 row) |
| Figure 2 (left): ROC curve; per-fold AUC (mean 0.817, SD 0.018) | 9. ROC curve and uncertainty | `figures/fig2_roc.png` |
| Figure 2 (right): calibration; AUC paid vs hypothetical | 10. Calibration | `figures/fig3_calibration.png` |
| Figure 1 (right): AUC by domain/frame cell; per-participant accuracy | 11. Error analysis | `figures/fig4_auc_by_cell.png`, `figures/fig5_accuracy_per_participant.png` |
| Prediction file | 12. Final model and submission file | `Track_b_submission.csv` |

## 7. Leakage and validation checklist
- `realised_payout_inr`: only in `train.csv`, known only after the choice, so it is dropped.
- `participant_id`: used only as the CV group. It is added as a feature only in the deliberate Experiment 2 comparison, never in the final model.
- `response_time_ms`: recorded with the choice, so it is excluded from the final model (Experiment 3 checks that it would not help).
- Scaling and one-hot encoding are inside the scikit-learn pipeline and fitted on training folds only.
- All new features are fixed formulas of the problem numbers (nothing learned from labels). The prospect-theory parameters are fixed literature-style values, not fitted.
- The test file is used once, in the final cell. Final model choice used the same CV, so the CV estimate may be slightly optimistic.

## 8. AI-assistance declaration
"I used Claude for help with code structure and for drafting the report and README. I ran all code myself and can explain."

## 9. References
See the bibliography in `report.pdf` (Tversky and Kahneman 1992 and 1981; Roberts et al. 2017; Kaufman et al. 2012; Niculescu-Mizil and Caruana 2005; Pedregosa et al. 2011). The competition data are the course's simulated Track B dataset on Kaggle.
