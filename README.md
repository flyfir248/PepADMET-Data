# PepADMET-Data
PepADMET Data

Yes — there's no single "master index," but the peptide-ADMET literature repeatedly draws on a fairly stable set of experimentally validated databases, each covering a different property. Here's a consolidated list organized by ADMET category, pulled from the papers you shared plus current search results. This should give Aurigene a practical starting map for data sourcing.

## Absorption / Permeability

| Database | Content | Size | URL/Access |
|---|---|---|---|
| **CycPeptMPDB** | Cyclic peptide membrane permeability (PAMPA, Caco-2, RRCK) with SMILES/HELM | 7,334–8,466 peptides (v1.2) | http://cycpeptmpdb.com |
| **B3Pdb** | Blood-brain barrier penetrating peptides | — | archive of BBB peptides |
| **CPPsite 2.0** | Experimentally validated cell-penetrating peptides w/ structures | — | published database |

## Distribution / BBB

| Database | Content | URL/Access |
|---|---|---|
| **BBPpred / BBPpredict / SCMB3PP / DeepB3P** | BBB-penetrating peptide predictors, each with own curated datasets (238–7,269 entries) | See respective publications |
| **Brainpeps** | BBB peptide database | — |

## Metabolism / Half-life

| Database | Content | Size | URL/Access |
|---|---|---|---|
| **PEPlife / PEPlife2** | Peptide half-life, experimental methods, modifications | 2,229 → 4,412 entries (402 proteins, 1,781 peptides) | published, Sci. Rep. |
| **PepTherDia** | Approved peptide drugs/diagnostics — half-life, plasma protein binding, PK | 105 peptides | http://peptherdia.herokuapp.com |
| **THPdb / THPdb2** | FDA-approved therapeutic peptides & proteins | THPdb: 852 entries (239 peptides+proteins); THPdb2: 894 unique therapeutics | http://crdd.osdd.net/raghava/thpdb/ |

## Toxicity (general, cytotoxicity, neurotoxicity, hemolysis)

| Database | Content | Size | URL/Access |
|---|---|---|---|
| **DBAASP v3** | Antimicrobial/cytotoxic activity + structure of peptides | large, continuously updated | http://dbaasp.org |
| **Hemolytik / Hemolytik2** | Experimentally determined hemolytic & non-hemolytic peptides | ~1,166 (orig.) → expanded in v2 | http://crdd.osdd.net/servers/hemolytik (check current mirror) |
| **ToxinPred 3.0** | Peptide toxicity (binary) | 11,036 peptides | webserver + dataset |
| **HemoPI-MOD** | Hemolytic potency of chemically modified peptides | 1,166 | published dataset |
| **EnDL-HemoLyt** | Hemolytic activity (natural + modified) | 4,339 | published dataset |
| **CancerPPD** | Anticancer peptides | — | database |
| **ApInAPDB** | Apoptosis-inducing anticancer peptides | 818 | published |
| **AVPdb / DRAVP / HIPdb** | Antiviral / HIV-inhibiting peptides (experimentally validated) | 981 (HIPdb) etc. | http://crdd.osdd.net/servers/hipdb (HIPdb) |
| **AHTPDB** | Antihypertensive peptides | — | database |

## Antimicrobial (source of much toxicity/hemolysis training data)

| Database | Content | URL/Access |
|---|---|---|
| **APD3/APD6 (Antimicrobial Peptide Database)** | AMP sequences, structures, activity | https://aps.unmc.edu |
| **CAMPR4** | Natural & synthetic antimicrobial peptides | database |
| **BaAMPs** | Biofilm-active antimicrobial peptides | database |

## Multi-property / aggregated resources

| Resource | Content | Size | Access |
|---|---|---|---|
| **pepADMET** (Tan et al., 2026) | 19 ADMET endpoints unified (permeability, half-life, LogD, F, BBB, 12 toxicity types) | 36,643 entries | https://pepadmet.ddai.tech; code: https://github.com/ifyoungnet/pepADMET |
| **PeptiVerse** (Zhang et al., 2026) | Hemolysis, solubility, non-fouling, toxicity, permeability, half-life, binding affinity — both sequence & SMILES | varies by property | https://huggingface.co/spaces/ChatterjeeLab/PeptiVerse; data: https://huggingface.co/datasets/ChatterjeeLab/PeptiVerse_data |
| **Scientific Data comprehensive therapeutic peptide dataset** (2025) | 58,583 experimentally validated therapeutic peptides across 47 function classes, 54,722 with structures (PDB + AlphaFold2) | 58,583 | https://www.nature.com/articles/s41597-025-05528-1 |
| **ChEMBL v35** | General bioactivity DB, includes ~22,045 HELM-notated linear/cyclic peptide entries | 22,045 | https://www.ebi.ac.uk/chembl |
| **Propedia v2.3** | Peptide-protein complex structures from PDB | 49,297 | published database |

## Small-molecule ADMET tools sometimes cross-validated against (not peptide-native, but referenced as comparators)
ADMETlab 3.0, admetSAR3.0, ProTox-II, SwissADME, pkCSM, deep-PK, ADMET-AI, ADMETboost — useful for benchmarking but built on small-molecule chemical space, so their accuracy on peptides is generally poor (this is explicitly flagged in the pepADMET Table S7 comparison you already have).

---

### A practical note for building your data pipeline
None of these individual source databases cover the full ADMET spectrum alone — that's precisely the gap pepADMET and PeptiVerse were built to fill by merging them. If Aurigene's goal is training in-house models, the efficient path is likely:
1. Pull the **already-merged, deduplicated datasets** from pepADMET (GitHub) and PeptiVerse (HuggingFace) rather than re-scraping each primary source separately — they've already done deduplication (InChIKey-based) and unit standardization.
2. Supplement gaps specific to your compound classes (e.g., if Aurigene targets a specific modality like cyclic peptides or ADCs) by going to the primary source (CycPeptMPDB, DBAASP v3) directly for the freshest experimental records.
3. Watch for licensing terms — some of these (e.g., CycPeptMPDB, DBAASP) are academic/research-use databases; check terms before commercial/industrial use at Aurigene.

Want me to check current access/licensing status for any of these specific databases (e.g., whether Hemolytik's original OSDD server is still live, since several CSIR-hosted tools have migrated over the years)?


Yes — there's no single "master index," but the peptide-ADMET literature repeatedly draws on a fairly stable set of experimentally validated databases, each covering a different property. Here's a consolidated list organized by ADMET category, pulled from the papers you shared plus current search results. This should give Aurigene a practical starting map for data sourcing.

## Absorption / Permeability

| Database | Content | Size | URL/Access |
|---|---|---|---|
| **CycPeptMPDB** | Cyclic peptide membrane permeability (PAMPA, Caco-2, RRCK) with SMILES/HELM | 7,334–8,466 peptides (v1.2) | http://cycpeptmpdb.com |
| **B3Pdb** | Blood-brain barrier penetrating peptides | — | archive of BBB peptides |
| **CPPsite 2.0** | Experimentally validated cell-penetrating peptides w/ structures | — | published database |

## Distribution / BBB

| Database | Content | URL/Access |
|---|---|---|
| **BBPpred / BBPpredict / SCMB3PP / DeepB3P** | BBB-penetrating peptide predictors, each with own curated datasets (238–7,269 entries) | See respective publications |
| **Brainpeps** | BBB peptide database | — |

## Metabolism / Half-life

| Database | Content | Size | URL/Access |
|---|---|---|---|
| **PEPlife / PEPlife2** | Peptide half-life, experimental methods, modifications | 2,229 → 4,412 entries (402 proteins, 1,781 peptides) | published, Sci. Rep. |
| **PepTherDia** | Approved peptide drugs/diagnostics — half-life, plasma protein binding, PK | 105 peptides | http://peptherdia.herokuapp.com |
| **THPdb / THPdb2** | FDA-approved therapeutic peptides & proteins | THPdb: 852 entries (239 peptides+proteins); THPdb2: 894 unique therapeutics | http://crdd.osdd.net/raghava/thpdb/ |

## Toxicity (general, cytotoxicity, neurotoxicity, hemolysis)

| Database | Content | Size | URL/Access |
|---|---|---|---|
| **DBAASP v3** | Antimicrobial/cytotoxic activity + structure of peptides | large, continuously updated | http://dbaasp.org |
| **Hemolytik / Hemolytik2** | Experimentally determined hemolytic & non-hemolytic peptides | ~1,166 (orig.) → expanded in v2 | http://crdd.osdd.net/servers/hemolytik (check current mirror) |
| **ToxinPred 3.0** | Peptide toxicity (binary) | 11,036 peptides | webserver + dataset |
| **HemoPI-MOD** | Hemolytic potency of chemically modified peptides | 1,166 | published dataset |
| **EnDL-HemoLyt** | Hemolytic activity (natural + modified) | 4,339 | published dataset |
| **CancerPPD** | Anticancer peptides | — | database |
| **ApInAPDB** | Apoptosis-inducing anticancer peptides | 818 | published |
| **AVPdb / DRAVP / HIPdb** | Antiviral / HIV-inhibiting peptides (experimentally validated) | 981 (HIPdb) etc. | http://crdd.osdd.net/servers/hipdb (HIPdb) |
| **AHTPDB** | Antihypertensive peptides | — | database |

## Antimicrobial (source of much toxicity/hemolysis training data)

| Database | Content | URL/Access |
|---|---|---|
| **APD3/APD6 (Antimicrobial Peptide Database)** | AMP sequences, structures, activity | https://aps.unmc.edu |
| **CAMPR4** | Natural & synthetic antimicrobial peptides | database |
| **BaAMPs** | Biofilm-active antimicrobial peptides | database |

## Multi-property / aggregated resources

| Resource | Content | Size | Access |
|---|---|---|---|
| **pepADMET** (Tan et al., 2026) | 19 ADMET endpoints unified (permeability, half-life, LogD, F, BBB, 12 toxicity types) | 36,643 entries | https://pepadmet.ddai.tech; code: https://github.com/ifyoungnet/pepADMET |
| **PeptiVerse** (Zhang et al., 2026) | Hemolysis, solubility, non-fouling, toxicity, permeability, half-life, binding affinity — both sequence & SMILES | varies by property | https://huggingface.co/spaces/ChatterjeeLab/PeptiVerse; data: https://huggingface.co/datasets/ChatterjeeLab/PeptiVerse_data |
| **Scientific Data comprehensive therapeutic peptide dataset** (2025) | 58,583 experimentally validated therapeutic peptides across 47 function classes, 54,722 with structures (PDB + AlphaFold2) | 58,583 | https://www.nature.com/articles/s41597-025-05528-1 |
| **ChEMBL v35** | General bioactivity DB, includes ~22,045 HELM-notated linear/cyclic peptide entries | 22,045 | https://www.ebi.ac.uk/chembl |
| **Propedia v2.3** | Peptide-protein complex structures from PDB | 49,297 | published database |

## Small-molecule ADMET tools sometimes cross-validated against (not peptide-native, but referenced as comparators)
ADMETlab 3.0, admetSAR3.0, ProTox-II, SwissADME, pkCSM, deep-PK, ADMET-AI, ADMETboost — useful for benchmarking but built on small-molecule chemical space, so their accuracy on peptides is generally poor (this is explicitly flagged in the pepADMET Table S7 comparison you already have).

---

### A practical note for building your data pipeline
None of these individual source databases cover the full ADMET spectrum alone — that's precisely the gap pepADMET and PeptiVerse were built to fill by merging them. If Aurigene's goal is training in-house models, the efficient path is likely:
1. Pull the **already-merged, deduplicated datasets** from pepADMET (GitHub) and PeptiVerse (HuggingFace) rather than re-scraping each primary source separately — they've already done deduplication (InChIKey-based) and unit standardization.
2. Supplement gaps specific to your compound classes (e.g., if Aurigene targets a specific modality like cyclic peptides or ADCs) by going to the primary source (CycPeptMPDB, DBAASP v3) directly for the freshest experimental records.
3. Watch for licensing terms — some of these (e.g., CycPeptMPDB, DBAASP) are academic/research-use databases; check terms before commercial/industrial use at Aurigene.

Want me to check current access/licensing status for any of these specific databases (e.g., whether Hemolytik's original OSDD server is still live, since several CSIR-hosted tools have migrated over the years)?