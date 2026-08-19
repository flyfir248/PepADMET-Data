#readme

Confirmed the repo exists. Let me actually fetch the code files, not just the README.

Fetched: GitHub - panda1103/CPMP

This is huge — the data folder is literally named pampa_uff_ig_true, confirming UFF with ignoreInterfragInteractions=True (their "-NB" variant) for PAMPA, matching the paper's Table S1. And critically: CPMP is built directly on ardigen/MAT, the original MAT reference implementation — that's almost certainly where the real hyperparameters live, not something the paper had to spell out because it's inherited from that codebase. Let me pull the actual source files from both repos.

Searched the web · 1 note

This changes everything. train_pampa.py in the actual repo isn't running a grid search at runtime — it's using fixed, already-determined hyperparameters, and they're substantially different from what we guessed. Key findings, straight from their source:

Parameter	Real CPMP value	Ours (before)
d_model	64	128
encoder layers N	6	4
attention heads h	64 (d_k=1 per head — unusual but literal)	8
readout dense layers	2	2 (ok)
activation (FF/encoder)	LeakyReLU(slope=0.16)	plain ReLU
λ (attention, distance, graph)	(0.1, 0.6, 0.3)	guessed presets
distance kernel	g(d) = exp(-d), element-wise, NOT row-normalized	softmax(-D) (row-normalized) — fundamentally different!
dummy node distance	1e6 (confirmed against the original MAT paper text)	matches our fix ✓
learning rate	1e-3	1e-4
loss	MSELoss(reduction='sum') per batch	mean-reduction
weight decay	none	1e-5
epochs	600, no early stopping	100, patience 15
PAMPA force field	UFF, ignoreInterfragInteractions=True (their "ig_true" = our -NB)	MMFF
checkpoint selection	best test-set loss over 600 epochs (train+val merged for this final run)	best val R² (proper practice)

The distance kernel difference is the big one — our normalized softmax spreads attention weight thin across all neighbors; their raw exp(-d) doesn't normalize, so absolute distance magnitude matters directly, and combined with λ_d=0.6 it dominates far more than our tuning ever converged on. This is a real architecture mismatch, not just a hyperparameter gap.

One methodological note before I implement this: their script merges train+val and picks the best checkpoint by test-set loss across all 600 epochs — that's a soft form of test leakage (optimistic bias), not something I'd recommend doing for genuine model selection. I'll implement the faithful architecture/training recipe but keep checkpoint selection on validation (proper practice), and add an explicit opt-in flag if you want the literal test-peeking behavior too.

Edited 3 files, ran 3 commands, viewed a file · 2 notes

Confirmed — the real architecture instantiates cleanly (305K params vs. our old ~1M+ guess — much smaller model, matching CPMP), and the two kernels behave exactly as documented: exp_elementwise doesn't normalize (row sums vary freely), softmax_neg always sums to 1.

Now let's wire this into train.py as a --paper_faithful preset, plus the sum-reduction loss and lr=1e-3:

Edited 14 files, ran 9 commands, viewed 7 files · 9 notes
The command to actually run now
bash
python train.py --train splits/PAMPA_train.csv --val splits/PAMPA_val.csv \
    --test splits/PAMPA_test.csv --out_dir cpmp_runs/pampa_mat_faithful \
    --paper_faithful --repeats 3

That's it — everything else (--epochs 600, --lr 1e-3, the exact λ values, the real distance kernel, the real architecture) is set automatically by the flag. Console output will echo every override so you can confirm it matches the table above before it starts training.

What I'd genuinely expect from this: given we now match the confirmed architecture, λ values, distance kernel, force field, and training length — plus checkpoint selection on validation (not their test-peeking) — I'd expect this to land meaningfully closer to 0.671 than anything we've run so far. Whether it lands exactly there or a bit under (say high 0.5s–0.6x) partly hinges on that test-peeking gap, which is a genuine, quantifiable source of optimistic bias in their reported number, not something wrong with our reproduction.

Two honest caveats:

n_heads=64 with d_model=64 gives 1 dimension per attention head — real and correctly implemented above, just an unusual design choice on their part, not a bug on ours.
This is PAMPA-specific. Caco-2 almost certainly uses different λ/force-field values (the paper's own Table S1 says MMFF-NB wins Caco-2, not UFF-NB) — I haven't fetched train_caco2.py yet. Want me to pull that next so Caco-2 gets the same treatment before you commit GPU time to it?
