"""
Unified Peptide ADMET / Toxicity prediction app.
=================================================

ONE SCRIPT. Run with:  python app.py
Serves a tabbed web UI at http://localhost:5000

How it works
------------
Every endpoint (toxicity, hemolysis, half_life, nf, binding_affinity,
permeability_caco2, permeability_pampa, permeability_penetrance, solubility)
is described by an entry in ENDPOINTS below. Each entry says:
  - where its trained model (.joblib) and metadata (.json) live
  - what kind of task it is (classification / regression)
  - what kind of embedding it needs (chemberta / wt / pair)
  - how to turn raw text input (SMILES or sequence) into features

Only "toxicity" is wired to a model that I confirmed exists on your machine
(models/toxicity_chemberta_xgboost.joblib). The others point at the paths
implied by your train_*.py scripts' RESULTS_DIR / ENDPOINT_DIR conventions,
but I never saw a saved .joblib for them -- only leaderboard CSVs from
training runs. If a model file is missing, that tab still renders and
explains clearly what's missing instead of crashing the whole app or
silently mispredicting.

TO ACTIVATE AN ENDPOINT:
  1. Make sure MODEL_PATH and METADATA_PATH below point at real files.
  2. Make sure the embedding function for that endpoint's EMBED_TYPE is
     correctly imported (see the EMBEDDERS section) -- these currently
     assume a `03_generate_embeddings.py` module next to this file with
     functions `embed_smiles(list[str]) -> np.ndarray` (ChemBERTa) and
     `embed_wt(list[str]) -> np.ndarray` (WT/handcrafted peptide features).
     Adjust the import / function names to match your actual module.
  3. Restart the app.

Directory layout this script expects (adjust ROOT below if different):

  ROOT/
    models/
      toxicity_chemberta_xgboost.joblib
      toxicity_chemberta_xgboost_metadata.json
      hemolysis_xgboost.joblib                 (example -- adjust)
      ...
    results/
      leaderboard_binding_affinity.csv
      leaderboard_hemolysis.csv
      leaderboard_nf.csv
      leaderboard_permeability_caco2.csv
      leaderboard_permeability_pampa.csv
      leaderboard_permeability_penetrance.csv
      leaderboard_permeability_regression.csv
      leaderboard_solubility.csv
    src/  (or repo root)
      03_generate_embeddings.py
"""

import sys
import io
import json
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
from flask import Flask, render_template_string, request, jsonify, send_file

# --------------------------------------------------------------------------
# PATH CONFIG -- adjust these three lines to match your machine
# --------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent
MODELS_DIR = REPO_ROOT / "models"
RESULTS_DIR = REPO_ROOT / "results"
EMBEDDINGS_MODULE_DIR = REPO_ROOT / "src"   # folder containing 03_generate_embeddings.py

sys.path.insert(0, str(EMBEDDINGS_MODULE_DIR))
sys.path.insert(0, str(REPO_ROOT))

app = Flask(__name__)

# --------------------------------------------------------------------------
# ENDPOINT REGISTRY
# --------------------------------------------------------------------------
# task: "classification" | "regression"
# embed_type: "chemberta" | "wt" | "pair_chemberta_wt" | "pair_wt_wt"
# input_kind: "single" (one SMILES/sequence) | "pair" (binder + target)
ENDPOINTS = {
    "toxicity": {
        "label": "Toxicity",
        "task": "classification",
        "embed_type": "chemberta",
        "input_kind": "single",
        "model_path": MODELS_DIR / "toxicity_chemberta_xgboost.joblib",
        "metadata_path": MODELS_DIR / "toxicity_chemberta_xgboost_metadata.json",
        "leaderboard_csv": RESULTS_DIR / "leaderboard_hemolysis.csv",  # placeholder if no tox leaderboard saved
        "positive_label": "Toxic",
        "negative_label": "Non-toxic",
    },
    "hemolysis": {
        "label": "Hemolysis",
        "task": "classification",
        "embed_type": "chemberta",
        "input_kind": "single",
        "model_path": MODELS_DIR / "hemolysis_chemberta_xgboost.joblib",
        "metadata_path": MODELS_DIR / "hemolysis_chemberta_xgboost_metadata.json",
        "leaderboard_csv": RESULTS_DIR / "leaderboard_hemolysis.csv",
        "positive_label": "Hemolytic",
        "negative_label": "Non-hemolytic",
    },
    "half_life": {
        "label": "Half-life",
        "task": "regression",
        "embed_type": "chemberta",
        "input_kind": "single",
        "model_path": MODELS_DIR / "halflife_chemberta_xgboost.joblib",
        "metadata_path": MODELS_DIR / "halflife_chemberta_xgboost_metadata.json",
        "leaderboard_csv": RESULTS_DIR / "leaderboard_nf.csv",  # adjust if you have a dedicated half-life leaderboard
        "unit": "hours",
    },
    "nf": {
        "label": "NF",
        "task": "regression",
        "embed_type": "chemberta",
        "input_kind": "single",
        "model_path": MODELS_DIR / "nf_chemberta_xgboost.joblib",
        "metadata_path": MODELS_DIR / "nf_chemberta_xgboost_metadata.json",
        "leaderboard_csv": RESULTS_DIR / "leaderboard_nf.csv",
        "unit": "",
    },
    "binding_affinity": {
        "label": "Binding Affinity",
        "task": "regression",
        "embed_type": "pair_wt_wt",
        "input_kind": "pair",
        "model_path": MODELS_DIR / "binding_affinity_xgboost.joblib",
        "metadata_path": MODELS_DIR / "binding_affinity_xgboost_metadata.json",
        "leaderboard_csv": RESULTS_DIR / "leaderboard_binding_affinity.csv",
        "unit": "affinity units (see metadata)",
    },
    "permeability_caco2": {
        "label": "Permeability (Caco-2)",
        "task": "regression",
        "embed_type": "chemberta",
        "input_kind": "single",
        "model_path": MODELS_DIR / "caco2_chemberta_xgboost.joblib",
        "metadata_path": MODELS_DIR / "caco2_chemberta_xgboost_metadata.json",
        "leaderboard_csv": RESULTS_DIR / "leaderboard_permeability_caco2.csv",
        "unit": "log Papp",
    },
    "permeability_pampa": {
        "label": "Permeability (PAMPA)",
        "task": "regression",
        "embed_type": "chemberta",
        "input_kind": "single",
        "model_path": MODELS_DIR / "pampa_chemberta_xgboost.joblib",
        "metadata_path": MODELS_DIR / "pampa_chemberta_xgboost_metadata.json",
        "leaderboard_csv": RESULTS_DIR / "leaderboard_permeability_pampa.csv",
        "unit": "log Pe",
    },
    "permeability_penetrance": {
        "label": "Permeability (Penetrance)",
        "task": "classification",
        "embed_type": "chemberta",
        "input_kind": "single",
        "model_path": MODELS_DIR / "penetrance_chemberta_xgboost.joblib",
        "metadata_path": MODELS_DIR / "penetrance_chemberta_xgboost_metadata.json",
        "leaderboard_csv": RESULTS_DIR / "leaderboard_permeability_penetrance.csv",
        "positive_label": "Penetrant",
        "negative_label": "Non-penetrant",
    },
    "solubility": {
        "label": "Solubility",
        "task": "classification",
        "embed_type": "chemberta",
        "input_kind": "single",
        "model_path": MODELS_DIR / "solubility_chemberta_xgboost.joblib",
        "metadata_path": MODELS_DIR / "solubility_chemberta_xgboost_metadata.json",
        "leaderboard_csv": RESULTS_DIR / "leaderboard_solubility.csv",
        "positive_label": "Soluble",
        "negative_label": "Insoluble",
    },
}

# --------------------------------------------------------------------------
# EMBEDDING BACKEND -- lazy imported so the app still starts if this module
# isn't present yet; individual endpoints will just report a clear error.
# --------------------------------------------------------------------------
_gen_embeddings = None


def _load_embedding_module():
    global _gen_embeddings
    if _gen_embeddings is None:
        try:
            from importlib import import_module
            _gen_embeddings = import_module("03_generate_embeddings")
        except Exception as e:
            raise RuntimeError(
                f"Could not import embedding module '03_generate_embeddings.py' "
                f"from {EMBEDDINGS_MODULE_DIR}. Error: {e}"
            )
    return _gen_embeddings


def embed_single(text: str, embed_type: str) -> np.ndarray:
    """Turn one SMILES/sequence string into a feature vector for the given embed_type."""
    mod = _load_embedding_module()
    if embed_type == "chemberta":
        if not hasattr(mod, "embed_smiles"):
            raise RuntimeError("03_generate_embeddings.py has no embed_smiles() function.")
        return np.asarray(mod.embed_smiles([text]))[0]
    if embed_type == "wt":
        if not hasattr(mod, "embed_wt"):
            raise RuntimeError("03_generate_embeddings.py has no embed_wt() function.")
        return np.asarray(mod.embed_wt([text]))[0]
    raise RuntimeError(f"Unsupported embed_type '{embed_type}' for single-input embedding.")


def embed_pair(binder_text: str, target_text: str, embed_type: str) -> np.ndarray:
    """Build the concatenated [target_embedding, binder_embedding] feature vector."""
    mod = _load_embedding_module()
    if embed_type == "pair_wt_wt":
        target_emb = np.asarray(mod.embed_wt([target_text]))[0]
        binder_emb = np.asarray(mod.embed_wt([binder_text]))[0]
    elif embed_type == "pair_chemberta_wt":
        target_emb = np.asarray(mod.embed_smiles([target_text]))[0]
        binder_emb = np.asarray(mod.embed_wt([binder_text]))[0]
    else:
        raise RuntimeError(f"Unsupported pair embed_type '{embed_type}'.")
    return np.concatenate([target_emb, binder_emb])


# --------------------------------------------------------------------------
# MODEL CACHE
# --------------------------------------------------------------------------
_model_cache = {}
_metadata_cache = {}


def get_model_and_metadata(endpoint_key: str):
    if endpoint_key in _model_cache:
        return _model_cache[endpoint_key], _metadata_cache.get(endpoint_key)

    cfg = ENDPOINTS[endpoint_key]
    model_path = cfg["model_path"]
    metadata_path = cfg["metadata_path"]

    if not model_path.exists():
        raise FileNotFoundError(
            f"No trained model found at {model_path}. "
            f"Train and save a model for '{endpoint_key}' first."
        )

    model = joblib.load(model_path)
    metadata = None
    if metadata_path.exists():
        with open(metadata_path) as f:
            metadata = json.load(f)

    _model_cache[endpoint_key] = model
    _metadata_cache[endpoint_key] = metadata
    return model, metadata


def endpoint_status(endpoint_key: str) -> dict:
    cfg = ENDPOINTS[endpoint_key]
    return {
        "key": endpoint_key,
        "label": cfg["label"],
        "task": cfg["task"],
        "input_kind": cfg["input_kind"],
        "model_ready": cfg["model_path"].exists(),
        "model_path": str(cfg["model_path"]),
        "leaderboard_ready": cfg["leaderboard_csv"].exists(),
    }


# --------------------------------------------------------------------------
# PREDICTION CORE
# --------------------------------------------------------------------------
def predict_single(endpoint_key: str, text: str, text2: str = None) -> dict:
    cfg = ENDPOINTS[endpoint_key]
    if not text or not str(text).strip():
        return {"error": "No input provided."}

    try:
        model, metadata = get_model_and_metadata(endpoint_key)
    except FileNotFoundError as e:
        return {"error": str(e)}

    try:
        if cfg["input_kind"] == "pair":
            if not text2 or not str(text2).strip():
                return {"error": "This endpoint needs both a binder and a target sequence."}
            X = embed_pair(text.strip(), text2.strip(), cfg["embed_type"]).reshape(1, -1)
        else:
            X = embed_single(text.strip(), cfg["embed_type"]).reshape(1, -1)
    except Exception as e:
        return {"error": f"Feature extraction failed: {e}"}

    try:
        if cfg["task"] == "classification":
            prob = float(model.predict_proba(X)[:, 1][0])
            threshold = (metadata or {}).get("classification_threshold", 0.5)
            pred_class = int(prob >= threshold)
            label = cfg["positive_label"] if pred_class == 1 else cfg["negative_label"]
            dist = abs(prob - threshold)
            confidence = "High" if dist > 0.35 else "Moderate" if dist > 0.15 else "Low"
            return {
                "input": text if cfg["input_kind"] != "pair" else {"binder": text, "target": text2},
                "prediction": label,
                "probability": round(prob, 4),
                "class": pred_class,
                "confidence": confidence,
                "error": None,
            }
        else:
            value = float(model.predict(X)[0])
            return {
                "input": text if cfg["input_kind"] != "pair" else {"binder": text, "target": text2},
                "prediction": round(value, 4),
                "unit": cfg.get("unit", ""),
                "error": None,
            }
    except Exception as e:
        return {"error": f"Prediction failed: {e}"}


def predict_batch(endpoint_key: str, items: list) -> list:
    """items: list of str (single input) or list of [binder, target] pairs."""
    cfg = ENDPOINTS[endpoint_key]
    results = []
    for item in items:
        if cfg["input_kind"] == "pair":
            binder, target = (item + [None, None])[:2] if isinstance(item, list) else (item, None)
            results.append(predict_single(endpoint_key, binder, target))
        else:
            results.append(predict_single(endpoint_key, item))
    return results


# --------------------------------------------------------------------------
# ROUTES
# --------------------------------------------------------------------------
@app.route("/")
def index():
    return render_template_string(INDEX_HTML, endpoints=ENDPOINTS)


@app.route("/api/status")
def api_status():
    return jsonify({k: endpoint_status(k) for k in ENDPOINTS})


@app.route("/api/predict/<endpoint_key>", methods=["POST"])
def api_predict(endpoint_key):
    if endpoint_key not in ENDPOINTS:
        return jsonify({"error": f"Unknown endpoint '{endpoint_key}'."}), 404
    data = request.get_json(force=True, silent=True) or {}
    text = (data.get("input") or "").strip()
    text2 = (data.get("input2") or "").strip() or None
    result = predict_single(endpoint_key, text, text2)
    status = 400 if result.get("error") else 200
    return jsonify(result), status


@app.route("/api/predict_batch/<endpoint_key>", methods=["POST"])
def api_predict_batch(endpoint_key):
    if endpoint_key not in ENDPOINTS:
        return jsonify({"error": f"Unknown endpoint '{endpoint_key}'."}), 404
    data = request.get_json(force=True, silent=True) or {}
    raw = data.get("input_block", "")
    cfg = ENDPOINTS[endpoint_key]
    if cfg["input_kind"] == "pair":
        # Expect "binder,target" per line
        items = []
        for line in raw.splitlines():
            line = line.strip()
            if not line:
                continue
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 2:
                items.append([parts[0], parts[1]])
    else:
        items = [ln.strip() for ln in raw.splitlines() if ln.strip()]

    if not items:
        return jsonify({"error": "No valid input rows found."}), 400
    if len(items) > 500:
        return jsonify({"error": "Max 500 rows via paste. Use CSV upload for more."}), 400

    return jsonify({"results": predict_batch(endpoint_key, items)})


@app.route("/api/predict_csv/<endpoint_key>", methods=["POST"])
def api_predict_csv(endpoint_key):
    if endpoint_key not in ENDPOINTS:
        return jsonify({"error": f"Unknown endpoint '{endpoint_key}'."}), 404
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded."}), 400
    cfg = ENDPOINTS[endpoint_key]
    file = request.files["file"]
    col = request.form.get("input_col", "SMILES")
    col2 = request.form.get("input_col2", "target")

    try:
        df = pd.read_csv(file)
    except Exception as e:
        return jsonify({"error": f"Could not read CSV: {e}"}), 400

    if col not in df.columns:
        return jsonify({"error": f"Column '{col}' not found. Available: {list(df.columns)}"}), 400
    if len(df) > 5000:
        return jsonify({"error": "Max 5000 rows per CSV upload."}), 400

    if cfg["input_kind"] == "pair":
        if col2 not in df.columns:
            return jsonify({"error": f"Column '{col2}' not found. Available: {list(df.columns)}"}), 400
        items = list(zip(df[col].astype(str), df[col2].astype(str)))
        items = [list(x) for x in items]
    else:
        items = df[col].astype(str).tolist()

    predictions = predict_batch(endpoint_key, items)
    pred_df = pd.DataFrame(predictions)
    out = pd.concat([df.reset_index(drop=True), pred_df], axis=1)

    buf = io.StringIO()
    out.to_csv(buf, index=False)
    buf.seek(0)
    mem = io.BytesIO(buf.getvalue().encode("utf-8"))
    return send_file(
        mem,
        mimetype="text/csv",
        as_attachment=True,
        download_name=f"{endpoint_key}_predictions.csv",
    )


@app.route("/api/leaderboard/<endpoint_key>")
def api_leaderboard(endpoint_key):
    if endpoint_key not in ENDPOINTS:
        return jsonify({"error": f"Unknown endpoint '{endpoint_key}'."}), 404
    path = ENDPOINTS[endpoint_key]["leaderboard_csv"]
    if not path.exists():
        return jsonify({"error": f"No leaderboard found at {path}."}), 404
    try:
        df = pd.read_csv(path)
    except Exception as e:
        return jsonify({"error": f"Could not read leaderboard: {e}"}), 400
    return jsonify({"columns": list(df.columns), "rows": df.to_dict(orient="records")})


# --------------------------------------------------------------------------
# FRONTEND (single-file, inlined so the whole app stays one script)
# --------------------------------------------------------------------------
INDEX_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Peptide ADMET / Toxicity Predictor</title>
<style>
  :root{
    --bg:#0f1117; --panel:#161923; --panel2:#1d2130; --border:#2a2f42;
    --text:#e8e9ee; --muted:#9aa0b4; --accent:#6ea8fe; --good:#4fd18b; --bad:#ff6b6b;
  }
  *{box-sizing:border-box;}
  body{margin:0;font-family:-apple-system,Segoe UI,Roboto,sans-serif;background:var(--bg);color:var(--text);}
  header{padding:20px 28px;border-bottom:1px solid var(--border);}
  header h1{margin:0;font-size:20px;}
  header p{margin:4px 0 0;color:var(--muted);font-size:13px;}
  .tabs{display:flex;flex-wrap:wrap;gap:4px;padding:12px 24px 0;border-bottom:1px solid var(--border);}
  .tab{padding:9px 14px;border-radius:8px 8px 0 0;cursor:pointer;color:var(--muted);font-size:13px;user-select:none;}
  .tab.active{background:var(--panel);color:var(--text);font-weight:600;}
  .tab .dot{display:inline-block;width:7px;height:7px;border-radius:50%;margin-right:6px;}
  .dot.ready{background:var(--good);} .dot.missing{background:var(--bad);}
  .content{padding:24px;max-width:1000px;margin:0 auto;}
  .panel{background:var(--panel);border:1px solid var(--border);border-radius:12px;padding:20px;margin-bottom:16px;}
  .panel h2{margin-top:0;font-size:16px;}
  .subtabs{display:flex;gap:8px;margin-bottom:14px;}
  .subtab{padding:6px 12px;border-radius:6px;background:var(--panel2);cursor:pointer;font-size:13px;color:var(--muted);}
  .subtab.active{background:var(--accent);color:#06101f;font-weight:600;}
  textarea,input[type=text]{width:100%;background:var(--panel2);border:1px solid var(--border);color:var(--text);
    border-radius:8px;padding:10px;font-family:monospace;font-size:13px;}
  textarea{min-height:90px;resize:vertical;}
  button{background:var(--accent);color:#06101f;border:none;padding:9px 16px;border-radius:8px;
    font-weight:600;cursor:pointer;font-size:13px;margin-top:10px;}
  button:hover{opacity:.9;}
  button.secondary{background:var(--panel2);color:var(--text);border:1px solid var(--border);}
  .row{display:flex;gap:12px;flex-wrap:wrap;}
  .row>div{flex:1;min-width:220px;}
  label{font-size:12px;color:var(--muted);display:block;margin-bottom:4px;}
  .result{margin-top:14px;padding:14px;border-radius:8px;background:var(--panel2);font-size:13px;white-space:pre-wrap;}
  .result.err{border:1px solid var(--bad);color:var(--bad);}
  .badge{display:inline-block;padding:2px 8px;border-radius:12px;font-size:12px;font-weight:600;}
  .badge.pos{background:rgba(255,107,107,.15);color:var(--bad);}
  .badge.neg{background:rgba(79,209,139,.15);color:var(--good);}
  table{width:100%;border-collapse:collapse;font-size:12px;margin-top:10px;}
  th,td{padding:6px 8px;border-bottom:1px solid var(--border);text-align:left;white-space:nowrap;}
  th{color:var(--muted);font-weight:600;}
  .muted{color:var(--muted);font-size:12px;}
  .missing-note{color:var(--bad);font-size:13px;}
</style>
</head>
<body>
<header>
  <h1>Peptide ADMET / Toxicity Predictor</h1>
  <p>One model per endpoint &middot; ChemBERTa / WT embeddings &middot; single script</p>
</header>

<div class="tabs" id="tabs"></div>
<div class="content" id="content"></div>

<script>
const ENDPOINTS = {{ endpoints_json | safe }};
let status = {};
let activeTab = Object.keys(ENDPOINTS)[0];
let activeSub = {}; // per-endpoint sub-tab: predict | metrics

async function loadStatus(){
  const res = await fetch('/api/status');
  status = await res.json();
  renderTabs();
  renderContent();
}

function renderTabs(){
  const tabs = document.getElementById('tabs');
  tabs.innerHTML = '';
  Object.entries(ENDPOINTS).forEach(([key, cfg]) => {
    const s = status[key] || {};
    const div = document.createElement('div');
    div.className = 'tab' + (key === activeTab ? ' active' : '');
    div.innerHTML = `<span class="dot ${s.model_ready ? 'ready' : 'missing'}"></span>${cfg.label}`;
    div.onclick = () => { activeTab = key; renderTabs(); renderContent(); };
    tabs.appendChild(div);
  });
}

function renderContent(){
  const cfg = ENDPOINTS[activeTab];
  const s = status[activeTab] || {};
  const sub = activeSub[activeTab] || 'predict';
  const c = document.getElementById('content');

  let html = `<div class="panel">
    <div class="subtabs">
      <div class="subtab ${sub==='predict'?'active':''}" onclick="setSub('predict')">Predict</div>
      <div class="subtab ${sub==='metrics'?'active':''}" onclick="setSub('metrics')">Metrics / Leaderboard</div>
    </div>`;

  if (!s.model_ready) {
    html += `<p class="missing-note">No trained model found at <code>${s.model_path}</code>.
      Train and save a model for this endpoint, then restart the app.</p></div>`;
    c.innerHTML = html;
    return;
  }

  if (sub === 'predict') {
    html += renderPredictPanel(cfg);
  } else {
    html += `<div id="metrics-area">Loading metrics&hellip;</div>`;
  }
  html += `</div>`;
  c.innerHTML = html;

  if (sub === 'predict') {
    wirePredictHandlers(cfg);
  } else {
    loadMetrics(activeTab);
  }
}

function setSub(name){
  activeSub[activeTab] = name;
  renderContent();
}

function renderPredictPanel(cfg){
  if (cfg.input_kind === 'pair') {
    return `
      <h2>Single prediction</h2>
      <div class="row">
        <div><label>Binder sequence</label><input type="text" id="single-input2a" placeholder="e.g. KLALKLALKAWKAALKLA"></div>
        <div><label>Target sequence</label><input type="text" id="single-input2b" placeholder="e.g. target protein sequence"></div>
      </div>
      <button onclick="runSingle()">Predict</button>
      <div id="single-result"></div>
      <h2 style="margin-top:24px;">Batch (paste, one pair per line: binder,target)</h2>
      <textarea id="batch-input" placeholder="binder1,target1&#10;binder2,target2"></textarea>
      <button onclick="runBatch()">Predict batch</button>
      <div id="batch-result"></div>
      <h2 style="margin-top:24px;">CSV upload</h2>
      <div class="row">
        <div><label>CSV file</label><input type="file" id="csv-file"></div>
        <div><label>Binder column name</label><input type="text" id="csv-col" value="SMILES"></div>
        <div><label>Target column name</label><input type="text" id="csv-col2" value="target"></div>
      </div>
      <button onclick="runCsv()">Upload &amp; download predictions</button>
      <div id="csv-result"></div>
    `;
  }
  return `
    <h2>Single prediction</h2>
    <label>SMILES / sequence</label>
    <input type="text" id="single-input" placeholder="e.g. CC(=O)Oc1ccccc1C(=O)O or KLALKLALKAWKAALKLA">
    <button onclick="runSingle()">Predict</button>
    <div id="single-result"></div>
    <h2 style="margin-top:24px;">Batch (paste, one per line)</h2>
    <textarea id="batch-input" placeholder="one SMILES/sequence per line"></textarea>
    <button onclick="runBatch()">Predict batch</button>
    <div id="batch-result"></div>
    <h2 style="margin-top:24px;">CSV upload</h2>
    <div class="row">
      <div><label>CSV file</label><input type="file" id="csv-file"></div>
      <div><label>Column name</label><input type="text" id="csv-col" value="SMILES"></div>
    </div>
    <button onclick="runCsv()">Upload &amp; download predictions</button>
    <div id="csv-result"></div>
  `;
}

function wirePredictHandlers(cfg){ /* handlers are global fns below, referencing activeTab */ }

function formatResult(r){
  if (r.error) return `<div class="result err">${r.error}</div>`;
  if ('probability' in r) {
    const badgeClass = r.class === 1 ? 'pos' : 'neg';
    return `<div class="result"><span class="badge ${badgeClass}">${r.prediction}</span>
      &nbsp; probability: ${r.probability} &nbsp; confidence: ${r.confidence}</div>`;
  }
  return `<div class="result">Prediction: <b>${r.prediction}</b> ${r.unit || ''}</div>`;
}

async function runSingle(){
  const cfg = ENDPOINTS[activeTab];
  let body;
  if (cfg.input_kind === 'pair') {
    body = { input: document.getElementById('single-input2a').value,
             input2: document.getElementById('single-input2b').value };
  } else {
    body = { input: document.getElementById('single-input').value };
  }
  const res = await fetch(`/api/predict/${activeTab}`, {method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify(body)});
  const data = await res.json();
  document.getElementById('single-result').innerHTML = formatResult(data);
}

async function runBatch(){
  const raw = document.getElementById('batch-input').value;
  const res = await fetch(`/api/predict_batch/${activeTab}`, {method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({input_block: raw})});
  const data = await res.json();
  const el = document.getElementById('batch-result');
  if (data.error) { el.innerHTML = `<div class="result err">${data.error}</div>`; return; }
  let rows = data.results.map(r => `<tr><td>${JSON.stringify(r.input)}</td><td>${r.error ? r.error : (r.prediction)}</td><td>${r.probability ?? ''}</td></tr>`).join('');
  el.innerHTML = `<table><tr><th>Input</th><th>Prediction</th><th>Probability</th></tr>${rows}</table>`;
}

async function runCsv(){
  const fileInput = document.getElementById('csv-file');
  if (!fileInput.files.length) { alert('Choose a CSV file first.'); return; }
  const cfg = ENDPOINTS[activeTab];
  const fd = new FormData();
  fd.append('file', fileInput.files[0]);
  fd.append('input_col', document.getElementById('csv-col').value);
  if (cfg.input_kind === 'pair') fd.append('input_col2', document.getElementById('csv-col2').value);
  const res = await fetch(`/api/predict_csv/${activeTab}`, {method:'POST', body: fd});
  if (!res.ok) {
    const data = await res.json();
    document.getElementById('csv-result').innerHTML = `<div class="result err">${data.error || 'Failed.'}</div>`;
    return;
  }
  const blob = await res.blob();
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url; a.download = `${activeTab}_predictions.csv`; a.click();
  document.getElementById('csv-result').innerHTML = `<div class="result">Downloaded.</div>`;
}

async function loadMetrics(endpointKey){
  const res = await fetch(`/api/leaderboard/${endpointKey}`);
  const data = await res.json();
  const el = document.getElementById('metrics-area');
  if (!el) return;
  if (data.error) { el.innerHTML = `<p class="missing-note">${data.error}</p>`; return; }
  let head = data.columns.map(c => `<th>${c}</th>`).join('');
  let rows = data.rows.map(r => `<tr>${data.columns.map(c => `<td>${typeof r[c] === 'number' ? r[c].toFixed(4) : r[c]}</td>`).join('')}</tr>`).join('');
  el.innerHTML = `<table><tr>${head}</tr>${rows}</table>`;
}

loadStatus();
</script>
</body>
</html>
"""


def _endpoints_json_safe():
    """Serialize ENDPOINTS config (label/task/input_kind only) for the frontend."""
    out = {}
    for k, cfg in ENDPOINTS.items():
        out[k] = {
            "label": cfg["label"],
            "task": cfg["task"],
            "input_kind": cfg["input_kind"],
        }
    return out


# Inject endpoints_json into the template context at render time
_orig_index_html = INDEX_HTML
INDEX_HTML = _orig_index_html.replace(
    "{{ endpoints_json | safe }}", "{{ endpoints_json | safe }}"
)


@app.context_processor
def inject_globals():
    return {}


# Patch render call to include endpoints_json
def render_index():
    return render_template_string(INDEX_HTML, endpoints_json=json.dumps(_endpoints_json_safe()))


app.view_functions["index"] = render_index


if __name__ == "__main__":
    print("Starting unified predictor app...")
    print(f"Models dir:   {MODELS_DIR}")
    print(f"Results dir:  {RESULTS_DIR}")
    for key in ENDPOINTS:
        s = endpoint_status(key)
        flag = "OK" if s["model_ready"] else "MISSING MODEL"
        print(f"  [{flag:14s}] {key:28s} -> {s['model_path']}")
    app.run(debug=True, host="0.0.0.0", port=5000)
