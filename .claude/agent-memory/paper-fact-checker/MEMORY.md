# Fact-Checker Agent Memory

## Author / Project
- Bachelor thesis by Marwan Elkhallouki, HOGENT
- Topic: Drift detection + XAI for industrial visual inspection (MVTec AD + ImageNet-C)
- File: `/home/marwan/School/3de/BAP/BAP/bachproef/`

## Common Error Patterns Observed (Session 1, 2026-02-27)
1. **Citation year errors**: Grad-CAM cited as Selvaraju et al. (2019) — correct year is 2017 (ICCV); journal version in IJCV is 2020.
2. **Missing bib entries**: Many sources cited in body text have no .bib entry (EDDM, Grad-CAM, LIME, SHAP, MDDM, ImageNet-C, Bayram 2022, Radhakrishnan 2020, Garreau 2020). This is a structural bibliography problem.
3. **MVTec AD image count**: Thesis states 5,354 images — 5,354 IS the correct total (training + test) per Bergmann CVPR 2019. However the split is 3,629 training + 1,725 test, NOT 4,096 + 1,258 as the thesis now claims. The 4,096 figure refers only to defect-free (good/normal) training images.
4. **River library citation**: Thesis cites River as (Bifet and Gavalda, 2007) — River is a separate modern library; correct citation is Montiel et al. (2021), JMLR.
5. **ResNet-50 ImageNet top-1 accuracy**: Thesis states 76.1% — this is correct for the original He et al. 2016 paper.
6. **SHAP axiomatic properties**: Thesis lists "efficiency, symmetry, and the dummy axiom" — correct but incomplete; Shapley values also satisfy linearity/additivity. Not an error, just imprecise.
7. **DDM description**: Paper says "p_i + s_i reaches a pre-defined warning level" — accurate description of DDM mechanism.
8. **EDDM description**: Described as Baena-García et al. (2006) — correct year and authors.
9. **ImageNet-C corruption count**: Thesis says 15 types — correct per Hendrycks & Dietterich 2019.
10. **Grad-CAM venue**: Thesis (standvanzaken.tex) originally said "published in IJCV" — FIXED in session 2 to ICCV 2017, which is correct. DOI 10.1109/ICCV.2017.74 confirmed correct.

## Confirmed Fixes Applied (Session 2, 2026-02-27)
- Grad-CAM citation changed to Selvaraju et al. (2017, ICCV) — CORRECT.
- Bayram 2022 arXiv:2203.11070 IS published in Knowledge-Based Systems, Vol. 245 (2022), article 108632, DOI: 10.1016/j.knosys.2022.108632 — CONFIRMED.
- MVTec AD 5,354 total confirmed. But new thesis split (4,096 + 1,258) is WRONG. Correct split: 3,629 train + 1,725 test. 4,096 = defect-free images only.

## New Findings (Session 4, 2026-03-02) — methodologie.tex fact-check
- **EDDM drift threshold**: thesis originally claimed both DDM and EDDM use α=0.95/α=0.99. WRONG for EDDM. River EDDM defaults: `alpha=0.95` (warning), `beta=0.9` (drift). DDM defaults: `warning_threshold=2.0` (~95%), `drift_threshold=3.0` (~99%). FIXED.
- **Grad-CAM transformer claim**: thesis said Grad-CAM "rules out most newer transformer-based architectures." INACCURATE — Grad-CAM can be adapted to Vision Transformers via reshape transforms. FIXED to state ResNet-50 is supported directly without adaptation overhead.
- **76.1% citation**: figure is from torchvision IMAGENET1K_V1 weights (76.130%), NOT He 2016 paper (75.3%). Citation moved to He2016 for the architecture; accuracy figure left uncited. FIXED.
- **Pesaranghader cite key**: changed from Pesaranghader2017 to Pesaranghader2018. MDDM paper is IJCNN 2018. Added bib entry.
- **Missing bib entries added** (Session 4): Selvaraju2017, baena2006early, Pesaranghader2018, Hendrycks2019, Ribeiro2016. All five now in bachproef.bib.
- **ADWIN δ=0.002**: CONFIRMED correct (River default).
- **AUROC interpretation**: "probability that defective image scores higher than normal image" — CONFIRMED correct (Mann–Whitney interpretation).
- **imagecorruptions library**: CONFIRMED implements all 15 ImageNet-C corruption types.
- **MVTec AD categories**: carpet, bottle, metal_nut, transistor, leather — ALL CONFIRMED valid MVTec AD categories.

## New Findings (Session 3, 2026-02-27) — bachproef_extra/ files
- New chapter files in bachproef_extra/ (inleiding, standvanzaken, methodologie, samenvatting) are substantially cleaner than previous versions.
- MVTec AD: new standvanzaken correctly states 5,354 total, 15 categories, 5 texture + 10 object — ALL CORRECT. No erroneous split stated.
- Grad-CAM: cite key is still `Selvaraju2019` (wrong year) — should be `Selvaraju2017`.
- MDDM: cite key is `Pesaranghader2017` — actual publication is 2018 IJCNN. Year is wrong.
- ResNet-50 76.1%: this matches torchvision IMAGENET1K_V1 weights (76.130%), not the original He 2016 paper figure (75.3%). Since the thesis explicitly says weights loaded from torchvision, the figure is effectively correct but should clarify it refers to torchvision weights.
- SHAP axiomatic properties: thesis lists "efficiency, symmetry, and dummy axiom" — confirmed correct (these are three of the four Shapley axioms; linearity is the fourth). Imprecise but not wrong.
- ImageNet normalisation values: mean [0.485, 0.456, 0.406], std [0.229, 0.224, 0.225] — CONFIRMED correct per official torchvision docs.
- LIME superpixel perturbation with neutral colour: CONFIRMED correct per Ribeiro 2016 KDD paper.
- River library: now correctly cited as Montiel2021 (JMLR 2021) — CORRECT.
- ADWIN O(log n) complexity: CONFIRMED correct per Bifet & Gavalda 2007.
- bib file (bachproef.bib): contains He2016, Gama2004, Bifet2007, Gama2014, Lundberg2017, Montiel2021, Bergmann2019. MISSING: Selvaraju (any year), Pesaranghader (any year), Bayram2022, Hendrycks2019, Ribeiro2016, baena2006early, garreau2020explaining, Radhakrishnan2020, VandenBroeck2022.

## Key Sources in This Domain
- MVTec AD: Bergmann et al. CVPR 2019 + IJCV 2021 (in bib, correct). Total: 5,354 images = 3,629 train + 1,725 test. 4,096 = defect-free subset only.
- DDM: Gama et al. SBIA 2004 (in bib, correct)
- ADWIN: Bifet & Gavalda SIAM SDM 2007 (in bib, correct)
- ResNet-50: He et al. CVPR 2016 (in bib, correct)
- Grad-CAM: Selvaraju et al. ICCV 2017, DOI 10.1109/ICCV.2017.74, pages 618-626 (CONFIRMED). IJCV 2020 is the extended version.
- Bayram 2022: Knowledge-Based Systems Vol. 245, article 108632, DOI 10.1016/j.knosys.2022.108632 (CONFIRMED). arXiv:2203.11070 is the preprint.
- MISSING from bib: EDDM (Baena-García 2006, ECML workshop), LIME (Ribeiro 2016 KDD), SHAP (Lundberg & Lee 2017 NeurIPS), MDDM (Pesaranghader, Viktor & Paquet 2018 IJCNN — NOT IJCAI), ImageNet-C (Hendrycks & Dietterich 2019 ICLR), Garreau & Luxburg 2020

## Details File
See `details.md` for extended notes on each error.
