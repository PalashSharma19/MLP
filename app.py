from __future__ import annotations

import tempfile
from pathlib import Path

from flask import Flask, jsonify, request

from predict import predict_file


app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 50 * 1024 * 1024


@app.after_request
def _add_cors_headers(response):
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    return response


@app.get("/api/health")
def health():
    return jsonify({"status": "ok"})


@app.route("/api/predict", methods=["POST", "OPTIONS"])
def predict_route():
    if request.method == "OPTIONS":
        return ("", 204)

    if "file" not in request.files:
        return jsonify({"error": "Missing file upload."}), 400

    uploaded_file = request.files["file"]
    if not uploaded_file.filename:
        return jsonify({"error": "Empty file name."}), 400

    model_name = request.form.get("model", "cnn").strip().lower()

    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            suffix = Path(uploaded_file.filename).suffix or ".bin"
            temp_path = Path(temp_dir) / f"upload{suffix}"
            uploaded_file.save(temp_path)

            result = predict_file(temp_path, model_name=model_name)

        return jsonify({"success": True, "result": result})
    except Exception as exc:
        return jsonify({"error": str(exc)}), 400


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5001, debug=False)