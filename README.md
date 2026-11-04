# Kanchan Bharti — IML Midterm Project, Track B

> Individual submission for CS-3410 / MAT-3211-1, Ashoka University.
> The dataset is **simulated**; see the track's Kaggle page.

| Field | Value |
|---|---|
| Student | Kanchan Bharti, [student ID: 5020261053] |
| Kaggle username | kanchanbharti039 |
| Track | B Behavioural Economics |
| Team  | [TDI Lake Grove] |
| Research question | For new participants, do the gain/loss frame and expected-value features predict risky choice better than a model that only sees the raw problem numbers? |

## 1. Environment
- Python 3.12 (3.10+ should work); Jupyter.
- Install: `pip install -r requirements.txt` (every version pinned: numpy 2.2.6, pandas 2.3.3, scikit-learn 1.7.2, scipy 1.15.3, matplotlib 3.10.7).
- Hardware used and approximate runtime: one CPU core, no GPU; a few minutes for the whole notebook.

## 2. Data
Download `train.csv`, `test.csv` and `sample_submission.csv` from the Track B Kaggle competition
(`iml-2026-track-b-risky-choice`) into one existing folder.

## 3. Reproduce everything
Open `Kanchan_Bharti_IML_Mid_term_Project.ipynb`, set the single variable `folder` in the first code cell to that
existing folder, and run all cells (or `jupyter nbconvert --to notebook --execute IML_Mid_term_Project_Track_B_improved.ipynb`).
No new folder is created: the notebook saves its results **into the same `folder`**:
`experimental_results.csv`, `submission.csv` (the Kaggle prediction file), `fig_overview.png`,
`fig_participant_acc.png` and `report_numbers.json`. Re-running gives a byte-identical prediction file.

## 4. Map from report to code
| Report item | Produced by (notebook section) |
|---|---|
| Table 1 (main results), Table 2 (Exp. 2 and 3), Table 3 (C sweep) | sections 5–8 → `experimental_results.csv` |
| Table 4 (paired differences) | section 9 → `report_numbers.json` |
| Table 5 (AUC by cell), session AUC, per-participant accuracy, calibration gap | section 10 → `report_numbers.json` |
| Figure 1 / Figure 2 (appendix) | section 10 → `fig_overview.png` / `fig_participant_acc.png` |
| Prediction file | section 11 → `submission.csv` (final fit on all training data) |

## 5. Seeds and determinism
`SEED = 0` (section 1) sets `np.random.seed`, the shuffled `GroupKFold`/`KFold` splits and the ExtraTrees `random_state`.
Logistic regression is deterministic. Only the `runtime_s` column of `experimental_results.csv` changes between runs.

## 6. Methods used
Classical methods only; no neural networks, deep learning, pretrained or foundation models.
- Expected-value rule (no training), constant prior predictor.
- Logistic regression, linear discriminant analysis, Gaussian Naive Bayes, linear SVM (taught in the course lectures).
- ExtraTrees (Geurts et al., 2006; not taught in the lectures) and a rank-average blend of logistic regression and ExtraTrees (final model).
- Fixed prospect-theory feature formulas (Tversky & Kahneman, 1992); no parameter is fitted to data.
- Validation: `GroupKFold` on `participant_id`, 5 shuffled folds. Metrics: ROC-AUC (primary), log-loss, accuracy at 0.5.

## 7. Sources and adapted code
- scikit-learn (Pedregosa et al., 2011), pandas, numpy, scipy (`rankdata`), matplotlib; code written against their documentation.
- Kahneman & Tversky (1979); Tversky & Kahneman (1981, 1992); Roberts et al. (2017); Geurts et al. (2006); Fawcett (2006) — full references in the report.
- No code was copied from other repositories or tutorials.

## 8. Generative-AI declaration
I used Claude during this project: for help with the notebook, the report text and this README.

## 9. Declaration
I confirm that this submission is my own individual work with the help of Generative AI and that I can explain every part of it.
