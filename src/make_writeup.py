import json
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle
from reportlab.lib import colors

m = json.load(open("results/metrics.json"))
ci, nb = m["cindex"], m["nb_balanced_acc"]
ss = getSampleStyleSheet()
B = ParagraphStyle("b", parent=ss["Normal"], fontSize=9, leading=11.5, alignment=4)
H = ParagraphStyle("h", parent=ss["Heading2"], fontSize=11, spaceBefore=5, spaceAfter=2)
T = ParagraphStyle("t", parent=ss["Title"], fontSize=15, spaceAfter=2)
S = ParagraphStyle("s", parent=ss["Normal"], fontSize=9, alignment=1, textColor=colors.grey)

doc = SimpleDocTemplate("MiniProject_Writeup.pdf", pagesize=A4, leftMargin=1.8*cm, rightMargin=1.8*cm,
                        topMargin=1.4*cm, bottomMargin=1.3*cm, title="UE24CS352A ML Mini-Project Write-up: Pancreatic Cancer Prognosis",
                        author="Team (Problem #6)", subject="Survival analysis with clinical and radiomic data")
P = lambda t, st=B: Paragraph(t, st)
st = []
st += [P("Pancreatic Cancer Prognosis using Clinical and Radiomic Data", T),
       P("UE24CS352A Machine Learning &middot; Mini-Project &middot; Problem #6 &middot; Team: [Member 1 (SRN)], [Member 2 (SRN)]", S)]

st += [P("1. Problem Statement", H),
       P("Most pancreatic cancer patients are diagnosed late, so prognosis matters for borderline-resectable and locally advanced cases. "
         "We build survival-analysis models that predict patient risk from <b>clinical</b> features, <b>radiomic</b> (MRI-derived) features, and both combined, "
         "and test whether radiomics adds prognostic value. The task follows the reference report by A. Jamalian (Stanford CS229, 2020).")]

st += [P("2. Dataset", H),
       P("The reference cohort (80 patients, 900 QIFP radiomic features, plus clinical fields such as stage, KPS, resectability, chemotherapy and tumour size) "
         "was provided by Stanford Cancer Center under an NDA and is <b>not public</b>. We therefore generated a <b>synthetic cohort with the same structure</b> "
         "(<font face='Courier'>src/generate_data.py</font>): 80 patients, 12 clinical features, 900 correlated radiomic features (30 latent factors), "
         "right-censored survival times (57 events, median 336 days) drawn from a hidden Cox-style hazard. "
         "<b>Results are illustrative of the pipeline, not clinical findings</b>, and are not directly comparable to the paper's numbers.")]

st += [P("3. Approach", H),
       P("<b>(i) Risk-threshold selection:</b> Naive Bayes (Laplace-smoothed, discretised clinical features) classifies high/low risk for survival-day thresholds 240-360; "
         "balanced accuracy under 10-fold CV picks the threshold. "
         "<b>(ii) Augmentation:</b> SMOTE oversamples the small high-risk group (time and event are interpolated with the features). "
         "<b>(iii) Cox proportional hazards</b> with L2 penalty (0.1, l1_ratio=0) on clinical (12), radiomic (150 highest-variance) and combined feature sets; "
         "radiomic features with a 95% CI width on log(HR) above 1 are dropped to cut variance. "
         "<b>(iv) Evaluation:</b> 10-fold CV Harrell C-index; Kaplan-Meier curves for patients split at the median hazard, with a log-rank test. "
         "<b>Improvement over the paper:</b> SMOTE, standardisation and feature filtering are fitted <i>inside each training fold only</i>, avoiding leakage into held-out folds.")]

st += [P("4. Implementation Overview", H),
       P("Python 3 with <i>lifelines</i> (Cox, KM, log-rank), <i>imbalanced-learn</i> (SMOTE), <i>scikit-learn</i> (Naive Bayes, CV) and <i>matplotlib</i>. "
         "<font face='Courier'>generate_data.py</font> builds the cohort; <font face='Courier'>pipeline.py</font> runs all stages and saves metrics and figures to <font face='Courier'>results/</font>; "
         "<font face='Courier'>demo.py</font> scores a patient for the live demo. Everything runs with <font face='Courier'>python src/pipeline.py</font> (see README).")]

rows = [["Model", "Features in", "Used (avg/fold)", "10-fold C-index"]] + \
       [[k, v["n_features_in"], v["n_features_used"], f"{v['mean']:.3f} ± {v['std']:.3f}"] for k, v in ci.items()]
t = Table(rows, colWidths=[5*cm, 3*cm, 3.5*cm, 4*cm])
t.setStyle(TableStyle([("FONTSIZE", (0, 0), (-1, -1), 8.5), ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#DDE6F0")),
                       ("GRID", (0, 0), (-1, -1), .4, colors.grey), ("ALIGN", (1, 0), (-1, -1), "CENTER")]))
nbtxt = ", ".join(f"{k} d: {v:.2f}" for k, v in nb.items())
st += [P("5. Results", H), t, Spacer(1, 4),
       P(f"Naive Bayes balanced accuracy by threshold: {nbtxt}. The best threshold was <b>{m['best_threshold']} days</b>, but accuracy stays near chance, "
         "so with 80 patients the threshold choice is weakly supported. "
         f"The log-rank test between predicted high- and low-risk groups gave a statistic of {m['logrank']['statistic']:.1f} (p = {m['logrank']['p']:.1e}); "
         "this final-model test is computed in-sample and is therefore optimistic.")]
imgs = Table([[Image("results/km_curves.png", 8.3*cm, 5.65*cm), Image("results/cindex_comparison.png", 8.3*cm, 5.65*cm)]])
st += [Spacer(1, 3), imgs]

st += [P("6. Conclusions", H),
       P("The full pipeline works end to end: Naive Bayes threshold selection, SMOTE, penalised Cox regression and risk stratification. "
         f"Combining clinical and radiomic features (C-index {ci['Clinical + Radiomic']['mean']:.2f}) beat clinical alone ({ci['Clinical']['mean']:.2f}), "
         "consistent in direction with the paper, while the radiomic-only model was near chance because 150 correlated features overfit ~70 training patients. "
         "Fold variance is large (±0.13-0.17), so these differences are not conclusive. "
         "<b>Limitations:</b> synthetic data, tiny sample, and synthetic survival times from SMOTE interpolation. "
         "<b>Future work:</b> real data, stronger regularisation or PCA on radiomics, and DeepSurv/DeepHit models.")]
st += [P("<b>Reference:</b> A. Jamalian, <i>Pancreatic cancer prognosis using clinical and radiomic data</i>, CS229 Final Report, Stanford, 2020. "
         "https://cs229.stanford.edu/proj2020spr/report/Jamalian.pdf", ParagraphStyle("r", parent=B, fontSize=8))]
doc.build(st)
