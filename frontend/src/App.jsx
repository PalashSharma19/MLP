import { useEffect, useMemo, useState } from "react";

const MODEL_OPTIONS = [
  { value: "cnn", label: "CNN ResNet-18" },
  { value: "rf", label: "Random Forest" },
  { value: "svm", label: "SVM (RBF)" },
  { value: "knn", label: "k-NN" },
];

function formatPercent(value) {
  if (value === null || value === undefined) {
    return "N/A";
  }

  const numericValue = Number(value);
  if (Number.isNaN(numericValue)) {
    return "N/A";
  }

  return `${(numericValue * 100).toFixed(2)}%`;
}

export default function App() {
  const [selectedFile, setSelectedFile] = useState(null);
  const [model, setModel] = useState("cnn");
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [prediction, setPrediction] = useState(null);
  const [error, setError] = useState("");
  const [previewUrl, setPreviewUrl] = useState("");

  useEffect(() => {
    if (!selectedFile) {
      setPreviewUrl("");
      return undefined;
    }

    if (!selectedFile.type.startsWith("image/")) {
      setPreviewUrl("");
      return undefined;
    }

    const objectUrl = URL.createObjectURL(selectedFile);
    setPreviewUrl(objectUrl);
    return () => URL.revokeObjectURL(objectUrl);
  }, [selectedFile]);

  const selectedModelLabel = useMemo(
    () => MODEL_OPTIONS.find((option) => option.value === model)?.label ?? model,
    [model],
  );

  const handleFileChange = (event) => {
    const file = event.target.files?.[0] ?? null;
    setSelectedFile(file);
    setPrediction(null);
    setError("");
  };

  const handleSubmit = async (event) => {
    event.preventDefault();

    if (!selectedFile) {
      setError("Choose a file or image first.");
      return;
    }

    setIsSubmitting(true);
    setError("");
    setPrediction(null);

    try {
      const formData = new FormData();
      formData.append("file", selectedFile);
      formData.append("model", model);

      const response = await fetch("/api/predict", {
        method: "POST",
        body: formData,
      });

      const payload = await response.json();

      if (!response.ok) {
        throw new Error(payload.error || "Prediction request failed.");
      }

      setPrediction(payload.result);
    } catch (submissionError) {
      setError(submissionError.message || "Prediction request failed.");
    } finally {
      setIsSubmitting(false);
    }
  };

  return (
    <main className="shell">
      <section className="hero">
        <div className="hero-copy">
          <p className="eyebrow">Malware Scan Demo</p>
          <h1>Upload a file and see the model prediction.</h1>
          <p className="lede">
            This demo predicts the family of uploaded files.
            It accepts executable samples and image files, then returns the predicted family.
          </p>
          <div className="pill-row">
            <span className="pill">CNN ready</span>
            <span className="pill">RF / SVM / k-NN</span>
            <span className="pill">Local Flask API</span>
          </div>
        </div>

        <form className="panel upload-card" onSubmit={handleSubmit}>
          <label className="file-dropzone">
            <input
              type="file"
              accept="image/*,.exe,.dll,.bat"
              onChange={handleFileChange}
            />
            <div>
              <strong>Drop a file here or click to browse.</strong>
              <p>Supports `.exe`, `.dll`, `.bat`, and image files.</p>
            </div>
          </label>

          <label className="field">
            <span>Model</span>
            <select value={model} onChange={(event) => setModel(event.target.value)}>
              {MODEL_OPTIONS.map((option) => (
                <option key={option.value} value={option.value}>
                  {option.label}
                </option>
              ))}
            </select>
          </label>

          <button type="submit" className="primary-button" disabled={isSubmitting}>
            {isSubmitting ? "Analyzing..." : "Predict"}
          </button>

          {selectedFile && (
            <div className="file-meta">
              <strong>{selectedFile.name}</strong>
              <span>{selectedModelLabel}</span>
            </div>
          )}
        </form>
      </section>

      <section className="content-grid">
        <article className="panel result-card">
          <div className="section-heading">
            <p>Result</p>
            <span>{prediction ? prediction.model.toUpperCase() : "Waiting"}</span>
          </div>

          {error && <div className="error-box">{error}</div>}

          {!error && !prediction && (
            <div className="empty-state">
              Upload a sample to see the prediction card fill in.
            </div>
          )}

          {prediction && (
            <div className="result-stack">
              <div className="result-badge">{prediction.verdict}</div>
              <div className="result-row">
                <span>Predicted family</span>
                <strong>{prediction.predicted_class}</strong>
              </div>
              <div className="result-row">
                <span>Confidence</span>
                <strong>{formatPercent(prediction.confidence)}</strong>
              </div>
              <div className="result-row">
                <span>Input type</span>
                <strong>{prediction.input_kind}</strong>
              </div>
              <div className="note-box">{prediction.note}</div>
            </div>
          )}
        </article>

        <article className="panel preview-card">
          <div className="section-heading">
            <p>Preview</p>
            <span>Optional</span>
          </div>

          {previewUrl ? (
            <img className="preview-image" src={previewUrl} alt="Uploaded preview" />
          ) : (
            <div className="empty-state compact">
              Image previews appear here. Non-image files are still accepted and scored.
            </div>
          )}

          {prediction?.top_predictions?.length ? (
            <div className="top-list">
              <h3>Top predictions</h3>
              {prediction.top_predictions.map((item) => (
                <div className="top-item" key={`${item.rank}-${item.label}`}>
                  <span>{item.rank}. {item.label}</span>
                  <strong>{formatPercent(item.probability)}</strong>
                </div>
              ))}
            </div>
          ) : null}
        </article>
      </section>
    </main>
  );
}