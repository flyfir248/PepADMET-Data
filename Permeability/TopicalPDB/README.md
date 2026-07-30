# TopicalPdb: A Database of Topically Delivered Peptides

Welcome to the official repository and documentation overview for **TopicalPdb**, a specialized repository of experimentally verified topically delivered peptides. This resource is designed to facilitate the scientific community in developing non-invasive peptide drug delivery systems.

**Web Server:** [http://crdd.osdd.net/raghava/topicalpdb/](http://crdd.osdd.net/raghava/topicalpdb/)

## Citation

Mathur, D., Mehta, A., Firmal, P., Bedi, G., Sood, C., Gautam, A., ... & Raghava, G. P. S. (2018). 
**TopicalPdb: A database of topically delivered peptides.** *PLoS ONE*, 13(2): e0190134. 
[https://doi.org/10.1371/journal.pone.0190134](https://doi.org/10.1371/journal.pone.0190134)

## About the Database

TopicalPdb was developed to address the lack of a dedicated repository for topically administered peptides. It consolidates information that was previously scattered throughout various literature sources, making it easily accessible for researchers working on peptide therapeutics.

The database integrates data from:
* **Primary Literature:** Data manually collected and curated from 135 research articles identified via systematic PubMed searches.
* **Structural Databases:** Tertiary structures mapped from the **Protein Data Bank (PDB)** or predicted using state-of-the-art tools.

## Key Features

### Comprehensive Dataset
* **657 Unique Entries:** Experimentally validated peptides categorized by their route of administration.
* **Route Distribution:** Includes 462 peptides delivered through the skin, 173 through the eye, and 22 through the nose.
* **Diverse Peptide Types:** Contains 584 linear and 73 cyclic peptides, including those with natural, non-natural, and modified residues.

### Rich Annotations
Each record includes:
* **Primary Data:** Sequence, length, N- and C-terminal modifications, chirality, and source of origin.
* **Experimental Conditions:** Mechanism of penetration, type of assay (in vitro, in vivo, or ex vivo), cargo properties, and tissue samples used.
* **Secondary Information:** Predicted tertiary structures and peptide sequences in SMILES format.

### Built-in Tools
* **Search Tools:** Simple keyword search and compounded search facilitating complex queries with logical operators (AND, OR).
* **Browsing Facility:** Retrieve information classified by conformation, peptide length, and route of administration.
* **Analysis Tools:** Integrated **BLAST** and **Smith-Waterman** for similarity searches, along with **HAlign** for multiple sequence alignment and **MUSCLE** for structural alignment.


## Overview

The architecture of TopicalPdb is designed for high data integrity and user interaction:
1.  **Data Curation:** Systematic manual curation ensuring only experimentally verified peptides are included.
2.  **Structural Prediction:** Utilization of **PEPstrMod** for small peptides (5-25 residues) and **I-TASSER** for longer natural peptides (>40 residues).
3.  **Multiple Entries:** Some peptides have multiple entries to reflect permeability studied under different experimental conditions (e.g., varying pH or concentration).


## Applications

* **Drug Formulation:** Identifying peptides that can penetrate barriers like the stratum corneum to develop novel topical drug formulations.
* **Molecular Interaction:** Using provided structural data for docking studies to understand interactions with skin proteins like keratin.
* **Computational Modeling:** Providing datasets for the analysis and development of prediction methods for skin-penetrating peptides.

## Contact & Authors

**Prof. G.P.S. Raghava**
raghava@imtech.res.in | raghava@iiitd.ac.in
Bioinformatics Centre, CSIR-Institute of Microbial Technology, Chandigarh, India.

## License

This database is distributed under the **Creative Commons Attribution License (CC BY 4.0)**.
