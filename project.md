# Mathematics Behind AI — AI-based Technology for Diabetic Retinopathy

Supervisor: Dr. Rohit Ahuja · Evaluation: after Diwali break

## Deliverables (from sir's message)
- [ ] GitHub repository with the working project (implementation verifiable)
- [ ] Research paper: Overleaf file/link
- [ ] Paper's experimental setup contains:
  - [ ] Ablation study *(experiments done; to be written into the paper)*
  - [ ] Statistical analysis *(experiments done; to be written into the paper)*
  - [ ] Cross-dataset validation *(experiments done; to be written into the paper)*

## Phase 0 — Decisions (blocking)
- [x] Dataset approved by user (APTOS + IDRiD)
- [x] Model family approved (proposal: CNN baseline vs. EfficientNet/ResNet transfer learning, 5-class DR grading + binary referable-DR)
- [x] Repo name + GitHub account: `NikhilD2003/AI-based-Technology-for-Diabetic-Retinopathy` (private, empty)

## Phase 1 — Repo & data
- [x] Create repo skeleton (`src/`, `notebooks/`, `docs/`, `paper/`, `results/`, `requirements.txt`, `README.md`)
- [x] Download primary dataset (training/validation/test split, stratified by grade, fixed seed) — 3,662 → 3,504 after removing exact duplicates; 70/15/15, seed 42, `results/splits.csv`
- [x] Download external dataset(s) for cross-dataset validation — IDRiD Disease Grading in `data/IDRiD/` (413 train + 103 test = 516, CC BY 4.0, [Zenodo 17219542](https://zenodo.org/records/17219542) mirror, 2026-11-27)
- [x] Preprocessing: Ben Graham Gaussian-blur contrast enhancement (applied to all colour channels), crop black borders, resize, augmentation
- [x] Class-imbalance handling (weighted loss)

## Phase 2 — Models & maths
- [x] Baseline (logistic regression / SVM on handcrafted features) — shows the maths foundation
- [x] CNN from scratch
- [x] Transfer-learning model (EfficientNet-B0 or ResNet50)
- [x] Metrics: accuracy, quadratic weighted kappa, AUC, sensitivity/specificity, F1, confusion matrix
- [x] Explainability: Grad-CAM
- [ ] Maths write-up: convolution, backprop/gradients, cross-entropy & weighted loss, optimiser (Adam), softmax, kappa definition *(moved to the paper author)*

## Phase 3 — Required experiments
- [x] **Ablation study**: preprocessing on/off, augmentation on/off, loss variant, backbone, image size
- [x] **Statistical analysis**: multiple seeds (mean ± SD), bootstrap 95% CIs, McNemar / paired t-test between models, DeLong for AUC
- [x] **Cross-dataset validation**: train on dataset A → test on B (and reverse); report performance drop
- [x] **Main model: original APTOS-trained EfficientNet-B0** (`full`)

## Phase 4 — Research paper (Overleaf / LaTeX)

**Handled by another project member (2026-11-27), including the maths write-up.**
- [ ] Replicate layout of `docs/DE-REPORT-8026340040.pdf` (Calibri-like, numbered sections, table of contents, figure captions, tables)
- [ ] Cover page: only change title → *AI-Based Technology for Diabetic Retinopathy* and supervisor → *DR. Rohit Ahuja* (rest identical: M.Tech in AI, Nikhilesh(8026340040), TIET logo, Department line)
- [ ] Remove all other content, keep section/TOC/table/figure styling
- [ ] Fill with this project: overview & architecture, dataset, methodology, experimental setup (ablation, statistics, cross-dataset), results, discussion, conclusion, references
- [ ] Generate figures from `results/` (no hand-drawn numbers)
- [ ] Final PDF compile + Overleaf link

## Phase 5 — Wrap-up
- [x] README: setup, how to reproduce each table/figure
- [ ] Final check: every number in the paper traceable to a script in the repo
- [ ] Push to GitHub, send repo + Overleaf links to sir

---

## Dataset proposal (approved)

| Role | Dataset | Size | Labels | Notes |
|---|---|---|---|---|
| **Primary (train/val/test)** | APTOS 2019 Blindness Detection (Kaggle) | ~3,662 labelled fundus images | 5 grades (0–4) | Small enough for a laptop/Colab, widely cited |
| **External (cross-dataset)** | IDRiD (Grading challenge) | 516 images | 5 DR grades + DME | Different camera/clinic; free |
| Optional 2nd external | Messidor-2 | 1,748 images | DR grades (adjudicated) | Needs registration |
| Optional bigger set | EyePACS / Kaggle DR 2015 | ~35k train | 5 grades | Large (tens of GB) — only if GPU available |

Recommendation: APTOS 2019 + IDRiD. Sizes and licences to be re-verified at download time.
