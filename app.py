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

app = Flask(__name__)
registry = ModelRegistry(MODELS_DIR)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(silent=True) or request.form
    smiles = (data.get("smiles") or "").strip()
    if not smiles:
        return jsonify({"error": "smiles is required"}), 400

    try:
        result = registry.predict_all(smiles)
    except Exception as e:
        app.logger.exception("prediction failed")
        return jsonify({"error": f"prediction failed: {e}"}), 500

    return jsonify(result)


@app.route("/health")
def health():
    return jsonify({"status": "ok", "models_dir": MODELS_DIR})


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
