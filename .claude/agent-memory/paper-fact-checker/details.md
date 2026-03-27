# Detailed Fact-Check Notes — Elkhallouki Bachelor Thesis

## Session: 2026-03-02 (methodologie.tex fact-check)

### Issue A: EDDM drift threshold wrong (FIXED)
- Original text: "DDM and EDDM warning and drift thresholds follow the commonly cited values of α = 0.95 and α = 0.99 respectively."
- Problem: DDM does not use α notation natively; its River defaults are numeric multipliers (2.0 warning, 3.0 drift). More critically, EDDM's drift threshold is 0.9, NOT 0.99. The text falsely implies both detectors share identical threshold semantics.
- Corrected text: "DDM uses a warning multiplier of 2.0 and a drift multiplier of 3.0, corresponding to confidence levels of approximately 95% and 99% (Gama2004). EDDM uses a warning threshold of α = 0.95 and a drift threshold of β = 0.9 (baena2006early)."
- Sources: River DDM docs (warning_threshold=2.0, drift_threshold=3.0); River EDDM docs (alpha=0.95, beta=0.9).

### Issue B: Grad-CAM transformer claim overstated (FIXED)
- Original text: "a constraint that rules out most newer transformer-based architectures."
- Problem: Grad-CAM CAN be applied to Vision Transformers using reshape transforms. The pytorch-grad-cam library explicitly documents this. The claim that it "rules out" transformers is factually incorrect.
- Corrected text: "While Grad-CAM can be adapted to Vision Transformers via reshape transforms (Selvaraju2017), doing so requires non-trivial modifications and careful layer selection; ResNet-50 supports it directly without any such adaptation overhead."

### Issue C: 76.1% accuracy cited to He2016 (FIXED)
- Original text: "ImageNet top-1 accuracy: 76.1%~\autocite{He2016}"
- Problem: He et al. 2016 reports 75.3% for ResNet-50. The 76.1% figure is from torchvision's IMAGENET1K_V1 pre-trained weights (ResNet V1.5, slightly different stride placement). Citing He2016 for this figure is wrong.
- Corrected text: Architecture cited to He2016; accuracy figure 76.1% left without citation but weight name (IMAGENET1K_V1) added for traceability.

### Issue D: Pesaranghader cite key year wrong (FIXED)
- Changed cite key from Pesaranghader2017 to Pesaranghader2018. MDDM paper is IJCNN 2018.
- Added bib entry with correct DOI 10.1109/IJCNN.2018.8489260.

### Issue E: Missing bib entries (FIXED in this session)
- Added to bachproef.bib: Selvaraju2017, baena2006early, Pesaranghader2018, Hendrycks2019, Ribeiro2016.

### Confirmed correct (no changes):
- ADWIN δ=0.002 — confirmed River default.
- AUROC probabilistic interpretation — confirmed correct (Mann-Whitney equivalence).
- imagecorruptions library — confirmed 15 corruption types.
- MVTec AD categories carpet/bottle/metal_nut/transistor/leather — all confirmed valid.
- MDDM window size 100 — plausible per Pesaranghader2018 experiments.

---

## Session: 2026-02-27

### Issue 1: Grad-CAM citation year
- Thesis uses cite key `Selvaraju2019` and states published in IJCV.
- ORIGINAL paper: Selvaraju et al., "Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization," ICCV 2017.
- EXTENDED journal paper: Selvaraju et al., IJCV 2020 (vol. 128, pp. 336–359).
- Neither publication year is 2019. The cite key `Selvaraju2019` matches no real publication date.
- The bib file does not contain this entry at all.

### Issue 2: SHAP original paper attribution
- Thesis attributes SHAP tractability to Van den Broeck et al. (2022) — this is a paper about SHAP tractability/complexity, not the original SHAP paper.
- Original SHAP paper: Lundberg & Lee, "A Unified Approach to Interpreting Model Predictions," NeurIPS 2017.
- The thesis never cites Lundberg & Lee for SHAP — it only cites Van den Broeck for tractability, which is acceptable if framed correctly, but the description blends the introduction of SHAP with the tractability paper.

### Issue 3: MDDM — Pesaranghader citation year
- Thesis cites MDDM as Pesaranghader et al. (2017).
- The main MDDM paper is: Pesaranghader & Viktor, "Fast Hoeffding Drift Detection Method for Evolving Data Streams," ECML-PKDD 2016 (not 2017).
- McDiarmid-based variant: Pesaranghader et al., "McDiarmid Drift Detection Methods for Evolving Data Streams," IJCAI 2018.
- The 2017 date in the thesis matches neither publication.

### Issue 4: River library citation
- Thesis cites River as `\autocite{Bifet2007}` — this is the ADWIN paper, not the River library.
- River (formerly Creme) is a separate Python library. Correct citation: Montiel et al., "River: machine learning for streaming data in Python," JMLR 2021.
- Citing the 2007 ADWIN paper for the River library is incorrect; ADWIN is one algorithm in the library, not the library itself.

### Issue 5: MVTec AD total image count and train/test split (UPDATED Session 2)
- **5,354 total** is CORRECT — confirmed by multiple independent sources including the official MVTec website, IEEE DataPort, and the original CVPR 2019 paper.
- **Correct split: 3,629 training images + 1,725 test images** (per Bergmann CVPR 2019 and multiple independent aggregators).
- The thesis (after revision) now states "4,096 training images and 1,258 test images" — BOTH figures are WRONG.
  - 4,096 refers to the number of *defect-free (good/normal) images* in the training split, not the total training set. The full training set (including some anomalous samples used for reference in certain methods) is 3,629.
  - 1,258 test images is also incorrect; the correct figure is 1,725.
- The 15 categories (5 texture: carpet, grid, leather, tile, wood; 10 object: bottle, cable, capsule, hazelnut, metal nut, pill, screw, toothbrush, transistor, zipper) are CORRECT per the paper.
- Source: Multiple independent aggregators and original MVTec CVPR 2019 paper. [MVTec official](https://www.mvtec.com/company/research/datasets/mvtec-ad), [IEEE DataPort](https://ieee-dataport.org/documents/mvtec-ad).

### Issue 6: ImageNet-C corruption categories — "weather" subcategory
- Thesis lists weather corruptions as: frost, fog, brightness, snow.
- Hendrycks & Dietterich 2019 list weather corruptions as: snow, frost, fog, brightness — same four, just different order. This is correct.
- Digital/other corruptions listed as: contrast, elastic, pixelate, JPEG. The thesis says "elastic" — the full name in the paper is "elastic transform." Not an error.
- Total: 15 corruption types. Thesis states 15. Correct.

### Issue 7: DDM — description of statistical bounds
- Thesis says: "When p_i + s_i reaches a pre-defined warning level, DDM enters a warning state; when it reaches the drift level, an alarm is triggered."
- This is accurate. In the Gama 2004 paper, DDM uses the Hoeffding/binomial bound. The warning level is p_min + s_min + 2*s_min (effectively) and drift level is p_min + 3*s_min. The description is a simplification but not incorrect.

### Issue 8: EDDM — claim it detects gradual drift "earlier than DDM"
- Thesis claims EDDM detects gradual drift earlier than DDM.
- Per Baena-García et al. 2006, EDDM is specifically designed to improve detection of gradual drift compared to DDM, and experimental results in the paper support this claim.
- The claim is accurate.

### Issue 9: ADWIN — description
- Thesis says ADWIN "tests whether any sub-window has a significantly different mean from the rest."
- More precisely, ADWIN splits the window into two sub-windows and uses a bound based on Hoeffding's inequality to test whether the means of the two halves differ significantly.
- The thesis description is a reasonable simplification and not materially wrong.

### Issue 10: Garreau & Luxburg citation
- Cited as "Garreau and Luxburg (2020)" — the correct spelling is "Luxburg" (Ulrike von Luxburg). However the standard academic citation omits "von." The 2020 date is plausible for this paper. The actual paper is Garreau & von Luxburg, "Explaining the Explainer: A First Theoretical Analysis of LIME," AISTATS 2020. Year is correct. Minor: "Luxburg" vs "von Luxburg."

### Issue 11: LIME paper title and venue
- Cited as Ribeiro et al. (2016) — correct. Paper: "Why Should I Trust You?: Explaining the Predictions of Any Classifier," KDD 2016. Year is correct.

### Issue 12: ResNet-50 top-1 accuracy 76.1%
- He et al. 2016 report top-1 accuracy on ImageNet validation of 75.3% for ResNet-50 in the original paper.
- The commonly cited figure of 76.1% comes from PyTorch's pre-trained model weights (torchvision ResNet50_Weights.IMAGENET1K_V1), which uses the same architecture but with slightly different training.
- This is a minor discrepancy. The 76.1% figure is widely cited in practice and is accurate for the standard PyTorch pre-trained weights. Author should clarify which weights are used.

### Issue 13: Bergmann 2021 IJCV paper author list
- Thesis body (standvanzaken.tex line 119) does not spell out authors.
- Bib entry for Bergmann2021 includes Batzner as a new co-author not in Bergmann2019. This is correct.
