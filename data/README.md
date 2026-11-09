# Ames Mutagenicity Harmonized Benchmark Dataset

This directory contains the curated and harmonized Ames mutagenicity dataset, exact scaffold split assignments, and standardized constituent source files used throughout this benchmark.

---

## 1. Dataset overview

- **Primary benchmark file:** [`ames_final.csv`](ames_final.csv) (canonical SMILES and binary target $y \in \{0, 1\}$)
- **Comprehensive registry:** [`harmonized_compounds_with_identifiers.csv`](harmonized_compounds_with_identifiers.csv) (SMILES, standard InChI, InChIKeys, molecular formula, MW, LogP, Murcko scaffold, acyclic indicator, and source presence flags)
- **Split memberships:** [`scaffold_split_memberships.csv.gz`](scaffold_split_memberships.csv.gz) (exact `train`, `val`, `test` assignments across all 25 repeated splits)
- **Total unique compounds:** 11,805
- **Outcome balance:**
  - **Mutagenic (**$y = 1$**):** 5,789 (49.04%)
  - **Non-mutagenic (**$y = 0$**):** 6,016 (50.96%)
- **Task formulation:** Binary classification for *Salmonella typhimurium* reverse mutation assay (Ames test) outcome.
- **Evaluation scheme:** 25 repeated scaffold-balanced 80/10/10 splits partitioned by Bemis–Murcko scaffold clustering.

---

## 2. Constituent source datasets

The benchmark unifies five prominent public Ames collections. Standardized per-source records are provided in [`sources/`](sources/):


| Source dataset       | File in `sources/`                               | Raw records  | Standardized records | Mutagenic ($y=1$) | Non-mutagenic ($y=0$) | Positive (%) |
| :-------------------- | :------------------------------------------------ | :------------: | :--------------------: | :-----------------: | :---------------------: | :------------: |
| **Hansen**           | [`sources/ds_hansen.csv`](sources/ds_hansen.csv) | 6,512        | 6,423                | 3,489             | 2,934                 | 54.3%        |
| **TDC (Xu et al.)**  | [`sources/ds_tdc.csv`](sources/ds_tdc.csv)       | 7,255 unique | 7,160                | 3,935             | 3,225                 | 55.0%        |
| **Deep-PK**          | [`sources/ds_deepk.csv`](sources/ds_deepk.csv)   | 7,617        | 7,557                | 4,279             | 3,278                 | 56.6%        |
| **PharmaBench**      | [`sources/ds_pharma.csv`](sources/ds_pharma.csv) | 9,145        | 9,029                | 4,407             | 4,622                 | 48.8%        |
| **ISSTOX (ISSSTY)**  | [`sources/ds_isstox.csv`](sources/ds_isstox.csv) | 7,367        | 6,492                | 3,255             | 3,237                 | 50.1%        |
| **Harmonized final** | [`ames_final.csv`](ames_final.csv)               | —            | **11,805**           | **5,789**         | **6,016**             | **49.0%**    |


---

## 3. Data harmonization and preprocessing pipeline

The curation workflow is implemented in [`../notebooks/data_prep.ipynb`](../notebooks/data_prep.ipynb) (note that [`sources/`](sources/) contains post-standardization per-source CSVs, while the notebook originally ran on `data/ames-combined/`):

1. **Standardization:** Structures were parsed and sanitized with Datamol/RDKit (repaired valence and formal-charge errors, stripped recognized salts/solvents, normalized and reionized structures moving negative charge to the most acidic group in already ionized molecules, without assuming a pH or neutralizing charges, duplicates removed) to yield canonical SMILES.
2. **Consensus voting:** Overlapping compounds (306 discordant cases, 4.1% of multi-source records) were resolved by majority voting. Ties (9 cases, 0.08%) and negative majorities were mapped to non-mutagenic ($y = 0$).
3. **ISSTOX mapping:** Raw records with summary Ames call `OVERALL = 3` were mapped to mutagenic ($y = 1$), whereas negative (score 1) or equivocal (score 2) calls and inconclusive records were assigned to non-mutagenic ($y = 0$).
4. **Ambiguity & sensitivity analysis:** Tied voting outcomes ($n = 9$, 0.08%) and sole-source ambiguous ISSTOX records ($n = 352$, 2.98%) together accounted for 361 compounds (3.06% of the dataset). Sensitivity analyses confirm that filtering these ambiguous records leaves the relative performance ranking of pipelines completely invariant (Supporting Information Table S5).

---

## 4. Scaffold split protocol

- **Splitting scheme:** 25 repeated scaffold-balanced 80/10/10 splits partitioned by Bemis–Murcko scaffold clustering (`tox_ames.data`).
- **Acyclic compounds:** Because the 1,821 acyclic compounds share an empty Bemis–Murcko scaffold, their combined cluster exceeds holdout quotas and resides in the training set across all 25 splits to prevent data leakage.
- **Partition matrix:** Exact split memberships for all 11,805 compounds are recorded in [`scaffold_split_memberships.csv.gz`](scaffold_split_memberships.csv.gz).

---

## 5. Structural alert rulebase (applicability domain)

For mechanistic sub-group evaluation and applicability domain mapping, compounds are profiled against the **Benigni–Bossa rulebase**:

- Implemented in `tox_ames.utils.structural_alerts`.
- Encodes 44 SMARTS patterns covering mutagenic and carcinogenic structural alerts derived from the European Commission Joint Research Centre (JRC) / Toxtree module.

---

---

## 6. Citations and references

1. **Hansen Dataset:**
   > Hansen, K.; Mika, S.; Schroeter, T.; Sutter, A.; ter Laak, A.; Steger-Hartmann, T.; Heinrich, N.; Müller, K.-R. Benchmark Data Set for *in Silico* Prediction of Ames Mutagenicity. *J. Chem. Inf. Model.* **2009**, *49* (9), 2077–2081. DOI: [10.1021/ci900161g](https://doi.org/10.1021/ci900161g).
2. **Therapeutics Data Commons (TDC) Ames Task:**
   > Huang, K.; Fu, T.; Gao, W.; Zhao, Y.; Roohani, Y.; Leskovec, J.; Coley, C. W.; Xiao, C.; Sun, J.; Zitnik, M. Therapeutics Data Commons: Machine Learning Datasets and Tasks for Drug Discovery and Development. *Proc. Neural Inf. Process. Syst. Track Datasets Benchmarks* **2021**, *1*.
3. **Xu et al. Ames Dataset (Underlying Source for TDC Ames):**
   > Xu, C.; Cheng, F.; Chen, L.; Du, Z.; Li, W.; Liu, G.; Lee, P. W.; Tang, Y. In Silico Prediction of Chemical Ames Mutagenicity. *J. Chem. Inf. Model.* **2012**, *52* (11), 2840–2847. DOI: [10.1021/ci300400a](https://doi.org/10.1021/ci300400a).
4. **Deep-PK Dataset:**
   > Myung, Y.; de Sá, A. G. C.; Ascher, D. B. Deep-PK: Deep Learning for Small Molecule Pharmacokinetic and Toxicity Prediction. *Nucleic Acids Res.* **2024**, *52* (W1), W469–W475. DOI: [10.1093/nar/gkae254](https://doi.org/10.1093/nar/gkae254).
5. **PharmaBench Dataset:**
   > Niu, Z.; Xiao, X.; Wu, W.; Cai, Q.; Jiang, Y.; Jin, W.; Wang, M.; Yang, G.; Kong, L.; Jin, X.; et al. PharmaBench: Enhancing ADMET Benchmarks with Large Language Models. *Sci. Data* **2024**, *11*, 985. DOI: [10.1038/s41597-024-03793-0](https://doi.org/10.1038/s41597-024-03793-0).
6. **ISSTOX / ISSSTY Database:**
   > Benigni, R.; Battistelli, C. L.; Bossa, C.; Tcheremenskaia, O.; Crettaz, P. New Perspectives in Toxicological Information Management, and the Role of ISSTOX Databases in Assessing Chemical Mutagenicity and Carcinogenicity. *Mutagenesis* **2013**, *28* (4), 401–409. DOI: [10.1093/mutage/get016](https://doi.org/10.1093/mutage/get016).
7. **Benigni–Bossa Structural Alerts:**
   > Benigni, R.; Bossa, C. Structural Alerts of Mutagens and Carcinogens. *Curr. Comput.-Aided Drug Des.* **2006**, *2* (2), 169–176. DOI: [10.2174/157340906777441663](https://doi.org/10.2174/157340906777441663).  
   > Benigni, R.; Bossa, C.; Jeliazkova, N.; Netzeva, T.; Worth, A. *The Benigni/Bossa Rulebase for Mutagenicity and Carcinogenicity: A Module of Toxtree*; Report EUR 23241 EN, JRC43157; Office for Official Publications of the European Communities: Luxembourg, **2008**.

---

## 7. License and attribution policy

 

Individual constituent datasets remain subject to their respective original source licenses and usage policies. If you use these datasets or their derived split subsets in your research, please follow their original licensing terms and appropriately cite the respective primary publications listed above.