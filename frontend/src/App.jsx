import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import axios from "axios";

// Use window.location.hostname to automatically use the correct IP (works for localhost and network access)
const API_BASE = import.meta.env.VITE_API_BASE || `http://${window.location.hostname}:8000`;

function App() {
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState("");
  const [result, setResult] = useState(null);
  const [error, setError] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [apiHealth, setApiHealth] = useState(null);
  const [healthError, setHealthError] = useState("");

  // Camera state
  const [cameraActive, setCameraActive] = useState(false);
  const [cameraError, setCameraError] = useState("");
  const videoRef = useRef(null);
  const streamRef = useRef(null);
  const canvasRef = useRef(null);

  useEffect(() => {
    let mounted = true;
    async function loadHealth() {
      try {
        const response = await axios.get(`${API_BASE}/health`);
        if (mounted) { setApiHealth(response.data); setHealthError(""); }
      } catch (err) {
        if (mounted) setHealthError(err.message || "API unavailable");
      }
    }
    loadHealth();
    const interval = setInterval(loadHealth, 12000);
    return () => { mounted = false; clearInterval(interval); };
  }, []);

  // Attach stream to video element AFTER it mounts (cameraActive triggers render)
  useEffect(() => {
    if (cameraActive && videoRef.current && streamRef.current) {
      videoRef.current.srcObject = streamRef.current;
      videoRef.current.play().catch(() => {});
    }
  }, [cameraActive]);

  const startCamera = useCallback(async () => {
    setCameraError("");
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: "environment", width: { ideal: 1280 }, height: { ideal: 960 } },
      });
      streamRef.current = stream;
      setFile(null);
      setPreview("");
      setResult(null);
      setError("");
      setCameraActive(true); // video element renders after this, then useEffect attaches stream
    } catch (err) {
      setCameraError("Camera access denied: " + err.message);
    }
  }, []);

  const stopCamera = useCallback(() => {
    if (streamRef.current) {
      streamRef.current.getTracks().forEach((t) => t.stop());
      streamRef.current = null;
    }
    setCameraActive(false);
  }, []);

  const captureFrame = useCallback(() => {
    if (!videoRef.current) return;
    const canvas = document.createElement("canvas");
    canvas.width = videoRef.current.videoWidth;
    canvas.height = videoRef.current.videoHeight;
    canvas.getContext("2d").drawImage(videoRef.current, 0, 0);
    canvas.toBlob((blob) => {
      const captured = new File([blob], "capture.jpg", { type: "image/jpeg" });
      setFile(captured);
      setPreview(canvas.toDataURL("image/jpeg"));
      setResult(null);
      setError("");
      stopCamera();
    }, "image/jpeg", 0.92);
  }, [stopCamera]);

  useEffect(() => () => stopCamera(), [stopCamera]);

  useEffect(() => {
    const root = document.documentElement;
    const canvas = canvasRef.current;
    const cursor = document.querySelector(".cursor-ring");
    const cursorTrail = document.querySelector(".cursor-trail");
    const ctx = canvas?.getContext("2d");
    if (!canvas || !ctx || !cursor || !cursorTrail) return undefined;

    let mouseX = window.innerWidth / 2;
    let mouseY = window.innerHeight / 2;
    let trailX = mouseX;
    let trailY = mouseY;
    let particles = [];
    let frameId;

    function resize() {
      const ratio = Math.min(window.devicePixelRatio || 1, 2);
      canvas.width = window.innerWidth * ratio;
      canvas.height = window.innerHeight * ratio;
      canvas.style.width = `${window.innerWidth}px`;
      canvas.style.height = `${window.innerHeight}px`;
      ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
      particles = Array.from({ length: Math.min(76, Math.floor(window.innerWidth / 20)) }, () => ({
        x: Math.random() * window.innerWidth,
        y: Math.random() * window.innerHeight,
        vx: (Math.random() - 0.5) * 0.18,
        vy: (Math.random() - 0.5) * 0.18,
        size: Math.random() * 1.8 + 0.6,
        hue: Math.random() > 0.62 ? 44 : (Math.random() > 0.5 ? 178 : 236),
      }));
    }

    function onMouseMove(event) {
      mouseX = event.clientX;
      mouseY = event.clientY;
      root.style.setProperty("--mx", `${(mouseX - window.innerWidth / 2) / 34}px`);
      root.style.setProperty("--my", `${(mouseY - window.innerHeight / 2) / 34}px`);
    }

    function onPointerOver(event) {
      if (event.target.closest("button, a, label, input, .card, .status-card, .panel")) {
        cursor.classList.add("is-hovering");
      }
    }

    function onPointerOut(event) {
      if (event.target.closest("button, a, label, input, .card, .status-card, .panel")) {
        cursor.classList.remove("is-hovering");
      }
    }

    function onClick(event) {
      const ripple = document.createElement("span");
      ripple.className = "cursor-ripple";
      ripple.style.left = `${event.clientX}px`;
      ripple.style.top = `${event.clientY}px`;
      document.body.appendChild(ripple);
      ripple.addEventListener("animationend", () => ripple.remove(), { once: true });
    }

    function draw() {
      trailX += (mouseX - trailX) * 0.16;
      trailY += (mouseY - trailY) * 0.16;
      cursor.style.transform = `translate3d(${mouseX}px, ${mouseY}px, 0) translate(-50%, -50%)`;
      cursorTrail.style.transform = `translate3d(${trailX}px, ${trailY}px, 0) translate(-50%, -50%)`;

      ctx.clearRect(0, 0, window.innerWidth, window.innerHeight);
      particles.forEach((p, index) => {
        p.x += p.vx + (mouseX - p.x) * 0.00014;
        p.y += p.vy + (mouseY - p.y) * 0.00014;
        if (p.x < -20) p.x = window.innerWidth + 20;
        if (p.x > window.innerWidth + 20) p.x = -20;
        if (p.y < -20) p.y = window.innerHeight + 20;
        if (p.y > window.innerHeight + 20) p.y = -20;

        ctx.beginPath();
        ctx.fillStyle = `hsla(${p.hue}, 54%, 45%, 0.24)`;
        ctx.shadowColor = `hsla(${p.hue}, 54%, 45%, 0.24)`;
        ctx.shadowBlur = 10;
        ctx.arc(p.x, p.y, p.size, 0, Math.PI * 2);
        ctx.fill();

        for (let j = index + 1; j < particles.length; j += 1) {
          const q = particles[j];
          const distance = Math.hypot(p.x - q.x, p.y - q.y);
          if (distance < 125) {
            ctx.beginPath();
            ctx.strokeStyle = `rgba(48, 58, 140, ${0.055 * (1 - distance / 125)})`;
            ctx.lineWidth = 1;
            ctx.moveTo(p.x, p.y);
            ctx.lineTo(q.x, q.y);
            ctx.stroke();
          }
        }
      });

      frameId = requestAnimationFrame(draw);
    }

    resize();
    draw();
    window.addEventListener("resize", resize);
    window.addEventListener("mousemove", onMouseMove, { passive: true });
    window.addEventListener("pointerover", onPointerOver);
    window.addEventListener("pointerout", onPointerOut);
    window.addEventListener("click", onClick);

    return () => {
      cancelAnimationFrame(frameId);
      window.removeEventListener("resize", resize);
      window.removeEventListener("mousemove", onMouseMove);
      window.removeEventListener("pointerover", onPointerOver);
      window.removeEventListener("pointerout", onPointerOut);
      window.removeEventListener("click", onClick);
    };
  }, []);

  const confidencePct = useMemo(() => {
    if (!result) return 0;
    return Math.round(result.confidence * 1000) / 10;
  }, [result]);

  const sortedProbabilities = useMemo(() => {
    if (!result?.probabilities) return [];
    return Object.entries(result.probabilities)
      .map(([label, value]) => ({ label, disease: value.disease, probability: value.probability, std: value.std }))
      .sort((a, b) => b.probability - a.probability);
  }, [result]);

  const confidenceBand = useMemo(() => {
    if (!result) return "";
    if (result.confidence >= 0.70) return "High confidence";
    if (result.confidence >= 0.45) return "Moderate confidence";
    return "Low confidence";
  }, [result]);

  const uncertaintyColor = useMemo(() => {
    if (!result?.uncertainty) return "";
    const level = result.uncertainty.level;
    if (level === "low") return "uncertainty-low";
    if (level === "moderate") return "uncertainty-moderate";
    return "uncertainty-high";
  }, [result]);

  function onFileChange(event) {
    const nextFile = event.target.files?.[0] || null;
    setResult(null); setError("");
    setFile(nextFile);
    if (!nextFile) { setPreview(""); return; }
    const fileReader = new FileReader();
    fileReader.onload = () => setPreview(fileReader.result?.toString() || "");
    fileReader.readAsDataURL(nextFile);
  }

  async function onSubmit(event) {
    event.preventDefault();
    if (!file) { setError("Upload or capture a lesion image first."); return; }
    setError(""); setIsLoading(true);
    try {
      const formData = new FormData();
      formData.append("file", file);
      const response = await axios.post(
        `${API_BASE}/predict?explain=true&tta_runs=1&mc_runs=5`,
        formData,
        { headers: { "Content-Type": "multipart/form-data" } },
      );
      setResult(response.data);
    } catch (err) {
      setError(err.response?.data?.detail || err.message || "Prediction failed");
    } finally {
      setIsLoading(false);
    }
  }

  return (
    <main className="page">
      <canvas ref={canvasRef} className="particle-field" aria-hidden="true" />
      <div className="cursor-ring" aria-hidden="true" />
      <div className="cursor-trail" aria-hidden="true" />
      <section className="hero">
        <p className="eyebrow">Clinical AI Workspace</p>
        <h1>DermaAegis AI</h1>
        <p className="hero-subtitle">Skin Lesion Intelligence Workspace</p>
        <p>Upload or capture a lesion image, run the trained model, inspect calibrated probabilities, and verify attention focus with Grad-CAM.</p>
      </section>

      <section className="status-strip">
        <div className="status-card">
          <p className="status-kicker">API</p>
          <p className={`status-value ${apiHealth?.status === "ok" ? "ok" : "bad"}`}>
            {apiHealth?.status === "ok" ? "Connected" : "Disconnected"}
          </p>
        </div>
        <div className="status-card">
          <p className="status-kicker">Model</p>
          <p className={`status-value ${apiHealth?.model_loaded ? "ok" : "bad"}`}>
            {apiHealth?.model_loaded ? "Loaded" : "Not loaded"}
          </p>
        </div>
      </section>

      {healthError && <p className="error">API check failed: {healthError}</p>}

      <section className="panel">
        <form onSubmit={onSubmit} className="upload-form">
          <div className="input-row">
            <label htmlFor="image" className="upload-box">
              <span>Choose image (JPG, PNG, WEBP)</span>
              <input id="image" type="file" accept="image/*" onChange={onFileChange} />
            </label>
            <button
              type="button"
              className={`camera-btn ${cameraActive ? "camera-btn--active" : ""}`}
              onClick={cameraActive ? stopCamera : startCamera}
            >
              {cameraActive ? "Stop Camera" : "Use Camera"}
            </button>
          </div>
          {cameraError && <p className="error">{cameraError}</p>}
          {cameraActive && (
            <div className="camera-wrap">
              <video ref={videoRef} autoPlay playsInline muted className="camera-feed" />
              <button type="button" className="capture-btn" onClick={captureFrame}>Capture</button>
            </div>
          )}
          <button disabled={isLoading} type="submit">
            {isLoading ? "Analyzing..." : "Run Prediction"}
          </button>
        </form>

        {error && <p className="error">{error}</p>}

        <div className="grid">
          <article className="card">
            <h2>Input</h2>
            {preview
              ? <img src={preview} alt="Uploaded lesion" className="preview" />
              : <p>No image selected.</p>}
          </article>

          <article className="card">
            <h2>Prediction</h2>
            {isLoading ? (
              <div className="skeleton-content">
                <div className="skeleton skeleton-text skeleton-text-large" />
                <div className="skeleton skeleton-text" />
                <div className="skeleton skeleton-text skeleton-text-short" />
                <div className="skeleton skeleton-box" />
                <div className="skeleton skeleton-text" />
                <div className="skeleton skeleton-text" />
                <div className="skeleton skeleton-text" />
              </div>
            ) : result ? (
              <>
                <p className="prediction-label">{result.predicted_disease}</p>
                <p className="prediction-meta">Label: {result.predicted_label}</p>
                <p className="prediction-meta">Confidence: {confidencePct}%</p>
                <p className="prediction-band">{confidenceBand}</p>

                {result.uncertainty && (
                  <div className={`uncertainty-box ${uncertaintyColor}`}>
                    <p className="uncertainty-label">Model Uncertainty</p>
                    <div className="uncertainty-bar-track">
                      <div className="uncertainty-bar-fill" style={{ width: `${result.uncertainty.score * 100}%` }} />
                    </div>
                    <p className="uncertainty-meta">
                      {(result.uncertainty.score * 100).toFixed(1)}% — {result.uncertainty.level} uncertainty
                    </p>
                    {result.uncertainty.level === "high" && (
                      <p className="uncertainty-warning">
                        High uncertainty — specialist review recommended.
                      </p>
                    )}
                  </div>
                )}

                {sortedProbabilities.length > 0 && (
                  <ul className="prob-list">
                    {sortedProbabilities.map((item) => (
                      <li key={item.label}>
                        <div className="prob-head">
                          <span>{item.disease}</span>
                          <span>{(item.probability * 100).toFixed(1)}%{item.std != null ? ` ±${(item.std * 100).toFixed(1)}` : ""}</span>
                        </div>
                        <div className="bar-track">
                          <div className="bar-fill" style={{ width: `${item.probability * 100}%` }} />
                        </div>
                      </li>
                    ))}
                  </ul>
                )}
              </>
            ) : (
              <p>Run inference to view results.</p>
            )}
          </article>

          <article className="card card-wide">
            <h2>Grad-CAM Explainability</h2>
            {isLoading ? (
              <div className="skeleton skeleton-image" />
            ) : result?.gradcam_base64 ? (
              <img className="preview" src={`data:image/png;base64,${result.gradcam_base64}`} alt="Grad CAM" />
            ) : result && !result.gradcam_base64 ? (
              <p>Grad-CAM unavailable for this image.</p>
            ) : (
              <p>Heatmap will appear after prediction.</p>
            )}
          </article>

          {(result?.fitzpatrick_analysis || isLoading) && (
            <article className="card card-wide">
              <h2>Fitzpatrick Skin Tone Analysis</h2>
              {isLoading ? (
                <div className="skeleton-content">
                  <div className="skeleton skeleton-text" />
                  <div className="skeleton skeleton-text" />
                  <div className="skeleton skeleton-text skeleton-text-short" />
                </div>
              ) : result?.fitzpatrick_analysis ? (
                <div className="fitzpatrick-content">
                  <div className="fitzpatrick-category">
                    <p className="fitzpatrick-label">Detected Skin Tone</p>
                    <p className="fitzpatrick-value">{result.fitzpatrick_analysis.skin_tone_category}</p>
                  </div>
                  {result.fitzpatrick_analysis.bias_warning && (
                    <div className="fitzpatrick-warning">
                      <p className="warning-icon">⚠️</p>
                      <div>
                        <p className="warning-title">Bias Warning</p>
                        <p className="warning-text">{result.fitzpatrick_analysis.bias_warning}</p>
                      </div>
                    </div>
                  )}
                  {result.fitzpatrick_analysis.recommendation && (
                    <div className="fitzpatrick-recommendation">
                      <p className="recommendation-title">Recommendation</p>
                      <p className="recommendation-text">{result.fitzpatrick_analysis.recommendation}</p>
                    </div>
                  )}
                  {result.fitzpatrick_analysis.dataset_representation && (
                    <div className="fitzpatrick-stats">
                      <p className="stats-label">Dataset Representation</p>
                      <p className="stats-value">{(result.fitzpatrick_analysis.dataset_representation * 100).toFixed(1)}%</p>
                    </div>
                  )}
                </div>
              ) : null}
            </article>
          )}
        </div>

        <div className="notice">
          <strong>Clinical Safety Notice:</strong> This tool is for research and educational purposes only.
          It is not a medical device. Final diagnosis must always be performed by a qualified dermatologist.
        </div>
      </section>
    </main>
  );
}

export default App;
