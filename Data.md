Good question to get from your PI — and worth being precise about, since this is a core pharmacokinetics concept. Here's the breakdown, tied to what's actually in your file.

**What A→B and B→A mean in a Caco-2 assay**

Caco-2 cells are grown as a polarized monolayer on a permeable membrane insert, creating two compartments:
- **Apical (A)** side — mimics the intestinal lumen
- **Basolateral (B)** side — mimics the blood-facing side

- **A→B permeability**: compound dosed on the apical side, measured appearing on the basolateral side. This estimates **absorptive** transport — roughly, "how well would this be absorbed from gut into blood."
- **B→A permeability**: compound dosed on the basolateral side, measured appearing apically. This captures **secretory/efflux** transport — how much the compound gets pumped back out, usually via efflux transporters like P-glycoprotein (P-gp/ABCB1) or BCRP, which are expressed on the apical membrane of Caco-2 cells.

From these two values, the standard derived metric is the **efflux ratio (ER)**:

ER = Papp(B→A) / Papp(A→B)

- ER ≈ 1 → passive diffusion dominates, no significant efflux
- ER > 2 (commonly used cutoff) → compound is likely an efflux transporter substrate, meaning its *apparent* oral absorption could be much lower in vivo than passive permeability alone would suggest — this matters a lot for cyclic/peptide-like molecules, which are frequent P-gp substrates

**What's actually in this file**

I checked the CSV — it has a single `Caco2` column (log Papp, cm/s) per compound, plus `R_Caco2` (recovery %) and detection-limit flags. There's no separate A→B / B→A split or efflux-ratio column here. This is the aggregated/curated value CycPeptMPDB pulled from the source literature — most likely representing the A→B (absorptive) direction, since that's the default readout reported in most cyclic-peptide permeability papers, but the dataset doesn't tell us that explicitly per row.

**What I'd tell your PI**

1. This particular Caco2 assay table gives you one permeability value per peptide — treat it as A→B (absorptive) unless a specific source paper says otherwise.
2. If the dissertation needs efflux behavior (B→A, ER) explicitly, that data isn't in this file — you'd need to go back to the original source papers (the `Source`/`Year` columns tell you which) and check their supplementary tables, since many Caco-2 papers on cyclic peptides do report bidirectional Papp and ER for P-gp liability discussion.
3. If you want, I can help you pull the `Source` list from this file and flag which original papers are likely to have bidirectional data worth citing.

Want me to extract that source list, or draft a short paragraph explaining this A→B/B→A distinction for the dissertation write-up?

__________

The Caco-2 monolayer assay is a widely used in vitro model for estimating intestinal permeability, exploiting the polarized nature of the cell line when cultured on a permeable membrane insert. Apparent permeability can be measured in two directions: apical-to-basolateral (A→B), which reflects the absorptive transport of a compound from the intestinal lumen toward systemic circulation, and basolateral-to-apical (B→A), which captures secretory transport driven largely by apically expressed efflux transporters such as P-glycoprotein (P-gp) and BCRP. The ratio of these two values, the efflux ratio (ER = Papp,B→A / Papp,A→B), is commonly used to flag compounds whose net absorption is limited by active efflux rather than passive diffusion alone (ER > 2 is a frequently applied cutoff). This distinction is particularly relevant for cyclic peptides, which are structurally prone to recognition by efflux transporters despite favorable passive membrane permeability. The Caco-2 permeability values used in the present dataset (CycPeptMPDB) represent a single curated log Papp value per compound, most consistent with the A→B (absorptive) direction as typically reported in the source literature; bidirectional (B→A) values and efflux ratios were not consistently available across sources and are therefore not included as a distinct feature in this analysis.

____________

Checked both files — same structure as the Caco2 one: a single permeability value per compound, no explicit A→B/B→A split. But the two assays are conceptually different from each other, which matters for your PI's question:

**PAMPA** (`CycPeptMPDB_Peptide_Assay_PAMPA.csv`, ~7,298 rows)
- This is a **cell-free** assay — compound diffuses across an artificial lipid membrane, no living cells involved.
- Because there are no transporters or efflux pumps present, **the A→B / B→A / efflux-ratio concept doesn't really apply here** — PAMPA only measures passive transcellular diffusion. A single Papp value is the whole story.
- Useful for isolating "pure" passive permeability, stripped of any transporter effects — often used alongside Caco-2/MDCK specifically to detect whether a *difference* between the cell-based and PAMPA values implies transporter involvement.

**MDCK** (`CycPeptMPDB_Peptide_Assay_MDCK.csv`, only 64 rows here)
- This *is* a cell-based monolayer assay, same logic as Caco-2 — polarized epithelial cells (often the MDCK-MDR1 variant, which overexpresses P-glycoprotein).
- So A→B / B→A and efflux ratio are conceptually valid here too, for the same reason as Caco-2. I noticed two rows (`1017`, `1018`) actually have **both a PAMPA and an MDCK value populated with an `R_MDCK` recovery number** — that's just recovery %, not a second direction, so still no bidirectional split in this file.
- Same caveat as before: this table gives one curated value per compound, presumed A→B, not the raw bidirectional pair.

**One-line summary for your PI:** Caco-2 and MDCK are both directional cell-based assays where A→B vs B→A (and efflux ratio) are meaningful, but none of these three curated files retain that split — they report one value each. PAMPA is fundamentally non-directional since it's not a cellular system at all, so the A→B/B→A question doesn't apply to it in the same way.

___________

**RRCK** (`CycPeptMPDB_Peptide_Assay_RRCK.csv`, 186 rows) — same pattern again: one Papp value per compound, no bidirectional split.

Conceptually, RRCK sits with Caco-2 and MDCK, not with PAMPA:

- RRCK cells are a **low-efflux MDCK subclone** — engineered specifically to have minimal endogenous transporter expression. It's often used as a "cleaner" cell-based passive permeability readout, sitting somewhere between PAMPA (no cells, no transporters at all) and standard Caco-2/MDCK-MDR1 (cells + active efflux).
- Because RRCK is selected for *low* efflux, A→B and B→A values are typically close to each other (efflux ratio near 1) — that's actually the point of using this cell line, to approximate passive permeability while still being a real cell membrane.
- Same caveat as the others: this file gives you one curated value, so you can't compute ER from it directly even though the underlying assay concept supports it.

So across your four assay files, here's the full picture for your PI:

| Assay | Cell-based? | Efflux-relevant? | A→B/B→A meaningful? |
|---|---|---|---|
| PAMPA | No | No | Not applicable |
| RRCK | Yes (low-efflux) | Minimal | Yes, in principle — but ER expected ≈1 |
| MDCK | Yes (often MDR1) | Yes | Yes |
| Caco-2 | Yes | Yes | Yes |

None of the four curated CSVs retain the raw bidirectional pair — all report a single value per compound. If the dissertation needs actual A→B/B→A numbers or efflux ratios, you'd have to trace back to the specific source papers (via the `Source`/`Year` columns) for the assays where that distinction is biologically meaningful (Caco-2 and MDCK especially).

_________

Here's the overall take, pulled together:

**The core answer to your PI's question:** none of these four CycPeptMPDB assay files (PAMPA, RRCK, MDCK, Caco-2) give you the raw A→B and B→A permeability values separately — each reports one curated Papp value per compound. So you can't compute efflux ratios directly from this dataset as-is.

**But the A→B/B→A concept applies differently across the four, and that's the useful story for your PI:**

- **PAMPA** — no cells, no transporters. A→B/B→A doesn't apply at all; it's pure passive diffusion through an artificial membrane. This is your "baseline" passive permeability.
- **RRCK** — real cells, but engineered to have minimal efflux transporter expression. A→B/B→A is conceptually meaningful but expected to be *similar* (efflux ratio ≈1), so it's used as a cleaner proxy for passive permeability in a cellular context.
- **MDCK** — real cells, often the MDR1-transfected variant that overexpresses P-gp. A→B/B→A and efflux ratio matter here — this is where efflux liability actually shows up.
- **Caco-2** — the gold-standard human intestinal model, naturally expressing multiple efflux transporters (P-gp, BCRP, etc.). Same story as MDCK — A→B/B→A is highly relevant, and it's the assay most predictive of real oral absorption behavior.

**The narrative you can hand your PI:** these four assays form a gradient of increasing biological complexity — PAMPA (no membrane transporters at all) → RRCK (cells, low efflux) → MDCK/Caco-2 (cells with active efflux machinery). Comparing a compound's permeability *across* these assays (rather than needing bidirectional data within one) already tells you a lot: if PAMPA and RRCK permeability are high but Caco-2/MDCK permeability is much lower, that gap itself is indirect evidence of efflux transporter involvement — even without a formal A→B/B→A split.

That's honestly a solid answer on its own, and it uses data you already have rather than requiring you to go dig up bidirectional numbers from 30+ source papers. If your PI specifically wants efflux ratios though, that would require pulling supplementary tables from the individual Caco-2/MDCK source papers.

_______

Good — solid, citable sources exist for all of this. Here's exactly what to hand your PI when he asks "where's this from":

**1. The database itself**
Li, Q. et al. *"CycPeptMPDB: A Comprehensive Database of Membrane Permeability of Cyclic Peptides"*, **J. Chem. Inf. Model.** 2023, 63(7), 2240–2251. (PubMed: 36930969)
— This paper describes CycPeptMPDB as a comprehensive database recording membrane permeability of cyclic peptides compiled from published papers and pharmaceutical patents, covering 7334 cyclic peptides collected from 45 published papers and 2 pharmaceutical patents. This is your primary citation for "where did this dataset come from."

**2. Why PAMPA has no A→B/B→A concept**
The parallel artificial membrane permeability assay (PAMPA) is one of the traditional experimental approaches for assessing membrane permeability — the key point being it uses an artificial (non-cellular) membrane, so there's no transporter machinery to create directional asymmetry. Original PAMPA method: Ottaviani et al., 2006 (commonly cited alongside CycPeptMPDB descriptions).

**3. Why RRCK is the "low-efflux" control**
The low-efflux MDCK clone Ralph-Russ canine kidney (RRCK) is typically used to provide assessments of passive permeability across the intestinal epithelium, since MDCK and RRCK cells typically have low expression levels of prototypical drug transporters compared to Caco-2 cells. Also: RRCK cells were developed from a subpopulation of low-efflux cells from MDCKII-WT, and are also called the MDCKII-LE (low efflux) cell line.

**4. Why MDCK (especially MDR1-transfected) is efflux-relevant**
MDCK-MDR1 cells originate from transfection of Madin-Darby canine kidney cells with the MDR1 gene encoding P-glycoprotein, making this cell line ideal for identifying P-gp substrates; studying drug efflux requires measuring transport from the basolateral to apical compartment and calculating an efflux ratio.

**5. Why Caco-2 is the gold standard for A→B/B→A + efflux ratio**
Caco-2 is most often used for evaluating overall permeability and involvement of efflux transporters or active uptake in a bidirectional experiment, and Caco-2 and MDCK assays are available as mono- and bidirectional formats to evaluate epithelial permeability and involvement of active transport.

**One-line citation list you can drop straight into the thesis:**
- Li et al., *J. Chem. Inf. Model.* 2023 — CycPeptMPDB source
- Ottaviani et al., 2006 — PAMPA method
- Di et al., 2011 — RRCK assay
- Irvine et al., 1999 — MDCK assay
- van Breemen & Li, 2005 — Caco-2 assay

Want me to pull exact DOIs/full citations for Ottaviani, Di, Irvine, and van Breemen so you have a ready-made reference list?


________________________

Let me verify the two links in the chain I haven't independently confirmed yet — ToxinPred 3.0's toxicity data source and the Barrett et al. non-fouling dataset — since PeptiVerse's paper cites these as secondary sources rather than describing the original assays itself.I went through the PeptiVerse paper's Methods section property-by-property, and additionally verified the two links that PeptiVerse cites but doesn't itself describe in detail (ToxinPred3.0 and the Barrett et al. non-fouling dataset) by pulling their original methodology. Here's the exact picture — including where "experimentally verified" needs a caveat.

## Property-by-property: what it measures, where it comes from, how solid the experimental grounding is

| Property | What it measures | Root experimental source | Verification status |
|---|---|---|---|
| **Permeability (PAMPA)** | Effective membrane permeability coefficient (logPexp) from cell-free parallel artificial membrane assay | CycPeptMPDB → 45 published papers + 2 pharma patents, each reporting real PAMPA assay results | **Fully experimental.** Confirmed directly from the CycPeptMPDB source paper: every value is a wet-lab-measured logPexp, no synthetic/predicted values mixed in. |
| **Permeability (Caco-2)** | Apparent permeability across Caco-2 cell monolayer (intestinal absorption proxy) | Same CycPeptMPDB lineage, subset measured by Caco-2 assay | **Fully experimental**, same provenance as PAMPA. |
| **Permeability_CPP** | Binary: cell-penetrating vs. non-penetrating (canonical sequences) | Positives: 22 independent experimental CPP uptake studies. Negatives: sourced from UniProt | **Positives experimental; negatives are assumed, not tested.** UniProt sequences are presumed non-CPP because they weren't reported as CPPs — they were never assayed for penetration and confirmed negative. This is an inferred negative class, not an experimentally-confirmed one. |
| **Hemolysis** | Binary: hemolytic vs. non-hemolytic | PeptideBERT/Peptide-Dashboard, cross-validated against DBAASP v3.0 original experimental records | **Experimental**, and explicitly cross-checked against DBAASP's primary experimental data (DBAASP itself compiles real red-blood-cell lysis assay results from literature). |
| **Non-fouling** | Binary: resists vs. permits non-specific protein adsorption on a surface | Barrett et al. 2018, which itself splits into two sub-sources | **Mixed.** Part of the positive class traces to White/Jiang 2013 (real surface-adsorption assay data — SAM/zwitterionic-coating experiments). But Barrett et al.'s expanded "Human" dataset defines additional non-fouling peptides by similarity to human protein surface composition — a QSPR-style inference, not a per-peptide wet-lab measurement. So: **experimentally rooted, but partly extrapolated, not 100% assayed per peptide.** |
| **Toxicity** | Binary: toxic vs. non-toxic | ToxinPred3.0, which pools toxic peptides from Conoserver, DRAMP, CAMPR3, dbAMP2.0, YADAMP, DBAASP-v3 | **Positive class is experimental** (these source databases hold characterized toxins — e.g., Conoserver = experimentally characterized conotoxins). **Negative class is not experimentally confirmed** — ToxinPred3.0's own paper states the non-toxic peptides were "gathered from SwissProt," i.e., presumed non-toxic by absence of toxin annotation, not tested and shown non-toxic. |
| **Solubility** | Binary: soluble vs. insoluble | PROSO II protocol (pepcDB pipeline stage tracking) + SoluProtMutDB | **Experimentally grounded but as a categorical proxy, not a quantitative assay.** The label reflects whether a real expression/purification construct reached the "Soluble" stage in a structural genomics pipeline (or stalled ≥8 months) — a real lab outcome, but binary pipeline status, not a measured solubility limit (e.g., mg/mL). Also a domain caveat: this data is largely full proteins from structural genomics, not short peptides specifically. |
| **Binding affinity** | Kd, Ki, or IC50 → converted to −log10(Kd/i in M) | PepLand → RCSB PDB + DrugBank entries labeled "peptide" | **Fully experimental** — these are real biophysical/biochemical binding measurements deposited with structures or drug records. |
| **Half-life** | Circulating half-life in human serum, converted to hours | THPdb2, PEPlife, PepTherDia — "only human serum measurements were retained" | **Fully experimental**, but the paper itself flags it as "sparse, heterogeneous in experimental protocol, and often reported in coarse or qualitative units" — real data, but noisy/inconsistent precision across sources. |

## Not experimental at all (important to separate out)
- **Physicochemical properties** (MW, net charge, pI, hydrophobicity) — purely calculated from sequence via Biopython. No experimental measurement involved; these are descriptors, not assay results.
- **ipTM scores** — computational structure-prediction confidence values from OpenFold3, used only to test whether structure confidence correlates with binding affinity (it didn't, ρ≈0.1). Not one of the trained/exposed properties.
- **Stability (TAPE/SaProt)** — used only as a pretraining corpus for half-life models, not a standalone exposed property. Itself experimentally derived (deep mutational scanning-type assays), but peptide-half-life-specific "authentication" doesn't apply to it directly.

## Cross-check against your HuggingFace screenshots
Your 46 subsets map exactly onto Table 2 of the paper — 6.08k hemolysis rows (4765+1311), 17.2k non-fouling (13580+3600), 606 Caco-2, 6.87k PAMPA, 2.32k Permeability_CPP (1162+1162), 18.5k solubility (9668+8785), 11k toxicity (5518+5518), 1.43k/1.7k binding affinity (AA/SMILES), 245/130 half-life (SMILES/AA). Every number lines up, which confirms the HF dataset is a faithful, unaltered mirror of what's documented in the paper — nothing appears to have been silently modified or expanded between publication and the hosted data. The "chemberta/peptideclm/wt" and "pooled/unpooled" suffixes are just different **embedding representations of the same underlying labels** — not different datasets or different experimental sources.

## Bottom line for your manager
Every property in PeptiVerse traces back to a real experimental source at its root — nothing is purely synthetic. But **"experimentally verified" is not uniform in strength across properties**:
- **Strong/direct** (every row = a real measured value): Permeability (PAMPA, Caco-2), Binding affinity, Half-life
- **Strong but binarized from a real measurement**: Hemolysis
- **Experimental positive class, but inferred/unconfirmed negative class**: Toxicity, Permeability_CPP
- **Experimental root, but partly proxy-extrapolated**: Non-fouling, Solubility

If Aurigene's models depend on trusting the negative class specifically (e.g., "this peptide is confirmed non-toxic" rather than "this peptide just isn't in a toxin database"), that's the one thing I'd flag as needing your own scrutiny before use — it's a common and known limitation in this type of binary bioactivity dataset, not unique to PeptiVerse.

Want me to now build the master tracker spreadsheet with this exact provenance/confidence grading built in as its own column?