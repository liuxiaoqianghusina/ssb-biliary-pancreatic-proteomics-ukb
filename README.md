# Sugar-sweetened beverages, plasma proteomic signatures and biliary–pancreatic disease (UK Biobank)

Analysis code for the manuscript:

> **Sugar-sweetened beverages and risk of gallstone disease and pancreatitis: a prospective cohort study with proteomic and genetic analyses.**

The study relates intake of sugar-sweetened beverages (SSBs) and artificially sweetened beverages (ASBs), assessed with Oxford WebQ 24-h dietary recalls, to incident gallstone disease, cholecystitis, acute pancreatitis (gallstone-related and non-gallstone) and chronic pancreatitis. It also derives elastic-net proteomic signatures of SSB and ASB intake from 2,919 Olink plasma proteins, tests them in an independent sample, assesses reverse causation, relates intake and signatures to abdominal MRI measures of organ fat, and uses cis-pQTL Mendelian randomization with colocalization (UKB-PPP, FinnGen R12) to examine the signature proteins.

**No data are included in this repository.** UK Biobank data are available to approved researchers on application (https://www.ukbiobank.ac.uk); this work used application number 1099827. FinnGen and UKB-PPP summary statistics are publicly available.

## Repository layout

| Folder | Script | Purpose |
|---|---|---|
| `data_prep/` | `01_beverages_outcomes_covariates.py` | SSB/ASB and other beverages (servings/day over valid recalls), hospital ICD-10 outcomes, self-reported conditions |
| | `02_first_occurrence_outcomes.py` | Outcomes from UK Biobank First Occurrence fields (primary care, hospital, self-report, death) |
| `analysis/` | `common.py` | Paths, covariate sets (Models 1–3), outcome definitions, Cox helper |
| | `P01_prepare.py` | Person-level analysis table (exposures, outcomes incl. gallstone-related vs non-gallstone acute pancreatitis, covariates) |
| | `S02_intake.py` | Intake–disease Cox models: Models 1–3, categories, splines, sensitivity, subgroup and substitution analyses |
| | `S03_proteins.py` | Protein-wise associations, replication of published proteins, elastic-net and residualized signatures |
| | `S04_signature_cox.py` | Signature–disease Cox models in the independent sample (Models 1–3, mutual adjustment, tertiles, sensitivity, individual proteins) |
| | `S05_extra.py`, `S05b_finegray.R` | Blood biomarkers, E-values, heterogeneity tests, pathway enrichment (g:Profiler), Fine–Gray competing-risk models |
| | `S06_ph.py`, `S06b_ph_timesplit.py` | Proportional-hazards tests and time-split estimates |
| | `S10_cis_mr.py`, `S11_coloc.py`, `S11b_coloc_window.py` | cis-pQTL Mendelian randomization (UKB-PPP → FinnGen R12) and colocalization (coloc.abf) |
| | `S12_reverse_causation.py`, `S12b_substitution_restricted.py` | Restrictions to stable diet/weight, no diabetes and BMI <30; habitual consumption patterns |
| | `S14_abdominal_mri.py` | Intake and signatures versus pancreatic and liver PDFF, visceral and subcutaneous fat and pancreas volume at the imaging visit |
| | `S07_figures.py`, `S08_tables.py`, `S13_graphical_abstract.py`, `S15_toc_FoodFunction.py` | Figures, tables, supplementary tables and the graphical abstract |

## Configuration

Directories are set with environment variables (defaults in brackets):

- `SSB_PROJECT_DIR` – output root; `cache/`, `results/` and `figures/` are created inside (`.`)
- `SCREEN_DIR` – cache written by `data_prep/` (`./data_prep_cache`)
- `UKB_DATA_DIR` – UK Biobank dataset CSV extracts and data dictionary (`../ukb_data`)
- `UKB_EXTRACT_DIR` – cached raw extracts of 24-h recall nutrients and items and of health outcomes (`../ukb_extracts`)
- `DIET_PROJECT_DIR` – companion diet–proteome project (`../diet-proteome`)
- `MS_DIR`, `SI_TITLE`, `MAIN_TABLES` – output folder and title for `S08_tables.py`; `MAIN_TABLES=2` leaves the signature table out of the main text, as in the submitted version (`3`)

Inputs from the companion diet–proteome project (to be released with that paper): the person-level covariate table (`cache/person.pkl`), the quality-controlled, plate-adjusted and standardized Olink matrix (`cache/olink_qc_plateadj.pkl`), the 70/30 signature training split and BMI proteomic signature (`cache/signatures.pkl`), gallstone-related covariates (`cache/extra_tier1.pkl`), and FinnGen R12 / UKB-PPP files under `data/mr/`.

## Running order

```
data_prep/01_beverages_outcomes_covariates.py
data_prep/02_first_occurrence_outcomes.py
analysis/P01_prepare.py
analysis/S02_intake.py
analysis/S03_proteins.py
analysis/S04_signature_cox.py
analysis/S05_extra.py
Rscript analysis/S05b_finegray.R
analysis/S06_ph.py
analysis/S06b_ph_timesplit.py
analysis/S10_cis_mr.py
analysis/S11_coloc.py            # needs UKB-PPP regional summary statistics from Synapse
analysis/S11b_coloc_window.py
analysis/S12_reverse_causation.py
analysis/S12b_substitution_restricted.py
analysis/S14_abdominal_mri.py        # after S12 (uses cache/diet_change.pkl)
analysis/S07_figures.py
analysis/S08_tables.py
analysis/S13_graphical_abstract.py
```

Run the `analysis/` scripts from inside `analysis/` (they import `common.py`).

## Software

Python 3.14 (pandas 2.3, numpy 2.4, scipy 1.17, lifelines 0.30, scikit-learn 1.9, statsmodels 0.14, matplotlib 3.11, joblib 1.5, openpyxl 3.1, python-docx) and R 4.2.2 (survival, data.table). See `requirements.txt`.

## License

MIT (see `LICENSE`).
