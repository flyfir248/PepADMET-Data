"""
app.py
------
Flask app serving side-by-side LogP(exp) predictions from the RFR, SVR, and
MAT models exported by export_models.py. All three models are loaded once
at startup (not per-request) via ModelRegistry.

Usage:
  export MODELS_DIR=exported_models/pampa   # dir containing models_manifest.json
  python app.py
Then open http://localhost:5000

Note: MAT predictions run synchronously in the same request (as requested) --
3D conformer generation for a single peptide is typically sub-second to a few
seconds, unlike the thousands-of-peptides batch case in the training pipeline,
but very large/unusual macrocycles can still take a moment. There's no
per-request timeout here; add one (e.g. via a worker timeout in gunicorn) if
you deploy this behind a web server that expects fast responses.
"""
import os
from flask import Flask, jsonify, render_template, request

from model_registry import ModelRegistry

MODELS_DIR = os.environ.get("MODELS_DIR", "exported_models/pampa")
# MAT runs synchronously per compound (conformer generation, one at a time),
# so a batch takes roughly this many times as long as a single prediction --
# capped to keep a single request from running unboundedly long.
MAX_BATCH = int(os.environ.get("MAX_BATCH", 25))

app = Flask(__name__)
registry = ModelRegistry(MODELS_DIR)


def _parse_smiles_field(data) -> list:
    """Accepts either a JSON list under "smiles", or a single string that may
    contain multiple SMILES separated by newlines and/or commas (covers both
    the JS client and a manual curl/form submission)."""
    raw = data.get("smiles")
    if raw is None:
        return []
    if isinstance(raw, list):
        items = raw
    else:
        items = [tok for line in str(raw).splitlines() for tok in line.split(",")]
    return [s.strip() for s in items if s and s.strip()]


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(silent=True) or request.form
    smiles_list = _parse_smiles_field(data)

    if not smiles_list:
        return jsonify({"error": "at least one smiles string is required"}), 400
    if len(smiles_list) > MAX_BATCH:
        return jsonify({"error": f"too many SMILES ({len(smiles_list)}); max is {MAX_BATCH} per request"}), 400

    try:
        results = registry.predict_batch(smiles_list)
    except Exception as e:
        app.logger.exception("batch prediction failed")
        return jsonify({"error": f"prediction failed: {e}"}), 500

    return jsonify({"results": results, "count": len(results)})


@app.route("/health")
def health():
    return jsonify({"status": "ok", "models_dir": MODELS_DIR})


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
