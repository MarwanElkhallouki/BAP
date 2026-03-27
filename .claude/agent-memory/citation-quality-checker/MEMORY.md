# Citation Quality Checker — Agent Memory

## Project context
Thesis: XAI-Supported Evaluation of Model Degradation in Industrial Visual Inspection (HOGENT, Marwan Elkhallouki).
Main bib files audited:
- `/home/marwan/School/3de/BAP/BAP/bachproef_extra/bachproef.bib`
- `/home/marwan/School/3de/BAP/BAP/proposal/proposal.bib`
- `/home/marwan/School/3de/BAP/BAP/voorstel/voorstel.bib`
- `/home/marwan/School/3de/BAP/BAP/bachproef/bachproef.bib`

## Venue open-access patterns confirmed in this project
- CVPR proceedings: fully open access via CVF (openaccess.thecvf.com). Both He2016 and Bergmann2019 have direct CVF PDF links.
- ICCV proceedings: fully open access via CVF. Selvaraju2017 confirmed open via CVF.
- NeurIPS proceedings: fully open access via proceedings.neurips.cc. Lundberg2017 confirmed.
- ICLR proceedings: fully open access via OpenReview.net. Hendrycks2019 confirmed published there.
- JMLR: gold open access. Montiel2021 confirmed via jmlr.org.
- JAIR (Journal of Artificial Intelligence Research): gold open access (jair.org). VandenBroeck2022 confirmed.
- AISTATS/PMLR: open access via proceedings.mlr.press. garreau2020explaining confirmed.
- ACM Computing Surveys: paywalled, but author preprints are typically available. Gama2014 has a preprint URL.
- Springer LNCS (SBIA): paywalled, but author preprints often available. Gama2004 has a preprint URL.
- SIAM proceedings: paywalled, but author preprints often available. Bifet2007 has a preprint URL.
- Knowledge-Based Systems (Elsevier): paywalled. Bayram2022 published version is paywalled (no open URL in bib).
- Springer edited book chapters: paywalled. Radhakrishnan2020 is in a Springer IFIP volume.

## Known entry-level issues in this bibliography (first audit, 2026-02-27)
- Ribeiro2016 (LIME): entered as @Article with publisher=arXiv. The published version is KDD 2016 (@InProceedings). Bib type and fields need correction.
- Pesaranghader2017 (MDDM): entered as @Article with publisher=arXiv. A published version exists in IJCAI 2018 proceedings. Needs correction.
- Hendrycks2019: entered as @Misc (arXiv) in both proposal.bib and voorstel.bib. The published ICLR 2019 version should be used (@InProceedings). The bachproef_extra version is already correct as @InProceedings.
- Bayram2022: INCONSISTENT across files. In proposal.bib it is @Misc (arXiv only). In voorstel.bib it is @Article in Knowledge-Based Systems (Elsevier, 2022, vol 245, doi 10.1016/j.knosys.2022.108632) — this is the correct published version.
- baena2006early (EDDM): missing DOI. Workshop paper, lighter peer review. A DOI exists via Semantic Scholar corpus ID 15672006. No URL in the main entry (only in the duplicate BaenaGarc2005EarlyDD).
- BaenaGarc2005EarlyDD: duplicate entry for EDDM, year listed as 2005, contradicting baena2006early (year 2006). Should be removed.
- Selvaraju2019: cite key says 2019 but the ICCV paper is from 2017. Key should be Selvaraju2017.
- Lundberg2017: missing DOI field. The NeurIPS DOI is 10.5555/3294996.3295150 (ACM DL) or the paper can be cited via proceedings.neurips.cc URL which is already present.
- Montiel2021: missing DOI. JMLR DOI is not always required given the URL, but could be added.
- VandenBroeck2022: missing URL field in voorstel.bib and proposal.bib. JAIR is open access; the paper is freely available at https://jair.org/index.php/jair/article/view/13283.
- Radhakrishnan2020: the HAL preprint URL (hal.science) is only in proposal.bib, not in voorstel.bib. Both should have it.

## Patterns to watch for in future audits
- arXiv DOIs (doi: 10.48550/ARXIV.*) used on @Article or @InProceedings entries signal the arXiv preprint is being cited instead of the published version.
- @Misc with eprinttype=arXiv signals an unreviewed preprint — always check whether a published version exists.
- Cite key year mismatches (e.g., Selvaraju2019 for a 2017 paper) are a common error in this bib.
- Duplicate entries for the same paper with different keys appear in this bib (baena2006early vs BaenaGarc2005EarlyDD).

See `venue-patterns.md` for extended venue open-access notes.
