/**
 * ExplainPanel — modal for Explainable AI visualisations.
 *
 * Crops the selected detection bounding box from the full image,
 * sends it to POST /api/explain, and renders the returned heatmap images.
 *
 * Appears when the user clicks "Explain" on a traffic-sign detection.
 */
import React, { useCallback, useEffect, useRef, useState } from 'react'
import { DetectionBox, ExplainResult, XaiMethod } from '../types/detection'
import { ExplainService } from '../services/apiClient'
import './ExplainPanel.css'

// ── helpers ──────────────────────────────────────────────────────────────────

/** Crop a region from a data-URL image and return it as a PNG Blob. */
function cropToBlob(
  imageDataUrl: string,
  x: number, y: number, w: number, h: number,
): Promise<Blob> {
  return new Promise((resolve, reject) => {
    const img = new Image()
    img.onload = () => {
      const canvas = document.createElement('canvas')
      canvas.width  = Math.max(1, w)
      canvas.height = Math.max(1, h)
      const ctx = canvas.getContext('2d')!
      ctx.drawImage(img, x, y, w, h, 0, 0, w, h)
      canvas.toBlob((blob) => {
        if (blob) resolve(blob)
        else reject(new Error('canvas.toBlob returned null'))
      }, 'image/png')
    }
    img.onerror = () => reject(new Error('Failed to load image for crop'))
    img.src = imageDataUrl
  })
}

const METHOD_LABELS: Record<XaiMethod, string> = {
  grad_cam: 'Grad-CAM',
  shap:     'SHAP',
  zennit:   'Zennit (LRP)',
}

const METHOD_DESCRIPTIONS: Record<XaiMethod, string> = {
  grad_cam: 'Gradient-weighted class activation map — highlights which regions pushed the prediction.',
  shap:     'SHAP pixel importance via blur masking — slow (~30 s), thorough explanation.',
  zennit:   'Layer-wise Relevance Propagation — shows how relevance flows through the network.',
}

// ── component ────────────────────────────────────────────────────────────────

interface Props {
  imageDataUrl: string
  detection: DetectionBox
  imageWidth: number
  imageHeight: number
  onClose: () => void
}

export const ExplainPanel: React.FC<Props> = ({
  imageDataUrl, detection, imageWidth, imageHeight, onClose,
}) => {
  const [method, setMethod]   = useState<XaiMethod>('grad_cam')
  const [results, setResults] = useState<ExplainResult[] | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError]     = useState<string | null>(null)
  const [elapsedMs, setElapsedMs] = useState<number | null>(null)

  // Preview canvas for the crop thumbnail
  const previewRef = useRef<HTMLCanvasElement>(null)

  // Draw crop preview whenever the modal opens
  useEffect(() => {
    const canvas = previewRef.current
    if (!canvas) return
    const img = new Image()
    img.onload = () => {
      const { x_min, y_min, x_max, y_max } = detection
      const w = Math.max(1, x_max - x_min)
      const h = Math.max(1, y_max - y_min)
      const aspect = w / h
      const maxSide = 160
      let dw = maxSide, dh = maxSide
      if (aspect > 1) dh = maxSide / aspect
      else            dw = maxSide * aspect
      canvas.width  = Math.round(dw)
      canvas.height = Math.round(dh)
      canvas.getContext('2d')!.drawImage(img, x_min, y_min, w, h, 0, 0, dw, dh)
    }
    img.src = imageDataUrl
  }, [imageDataUrl, detection])

  // Run explanation when method changes (auto-run)
  const runExplanation = useCallback(async () => {
    setLoading(true)
    setError(null)
    setResults(null)
    setElapsedMs(null)
    try {
      const { x_min, y_min, x_max, y_max } = detection
      const blob = await cropToBlob(
        imageDataUrl,
        Math.round(x_min), Math.round(y_min),
        Math.round(x_max - x_min), Math.round(y_max - y_min),
      )
      const response = await ExplainService.explainFromBlob(blob, method)
      setResults(response.results)
      setElapsedMs(response.inference_time_ms)
    } catch (err: any) {
      const msg =
        err?.response?.data?.detail ??
        err?.message ??
        'Explanation failed. Check the backend logs.'
      setError(msg)
    } finally {
      setLoading(false)
    }
  }, [imageDataUrl, detection, method])

  // Auto-run on method change
  useEffect(() => { runExplanation() }, [runExplanation])

  // Close on Escape key
  useEffect(() => {
    const handler = (e: KeyboardEvent) => { if (e.key === 'Escape') onClose() }
    window.addEventListener('keydown', handler)
    return () => window.removeEventListener('keydown', handler)
  }, [onClose])

  // ── labels ──
  const signLabel = detection.label.includes(' | ')
    ? detection.label.split(' | ')[0]
    : detection.label
  const damageLabel = detection.label.includes(' | ')
    ? detection.label.split(' | ')[1]
    : ''

  return (
    <div
      className="xai-overlay"
      role="dialog"
      aria-modal="true"
      aria-label="Explainability panel"
      onClick={(e) => { if (e.target === e.currentTarget) onClose() }}
    >
      <div className="xai-panel">

        {/* ── Header ── */}
        <div className="xai-header">
          <div className="xai-header-info">
            <span className="xai-header-title">Explain Detection</span>
            {signLabel && (
              <span className="xai-header-sign">{signLabel}</span>
            )}
            {damageLabel && (
              <span className="xai-header-damage">{damageLabel}</span>
            )}
          </div>
          <button className="xai-close" onClick={onClose} aria-label="Close">✕</button>
        </div>

        {/* ── Body ── */}
        <div className="xai-body">

          {/* Left column: crop + method selector */}
          <div className="xai-sidebar">
            <div className="xai-crop-wrap">
              <p className="xai-sidebar-label">Sign crop</p>
              <canvas ref={previewRef} className="xai-crop-preview" />
            </div>

            <div className="xai-method-wrap">
              <p className="xai-sidebar-label">Method</p>
              <div className="xai-method-tabs" role="tablist">
                {(Object.keys(METHOD_LABELS) as XaiMethod[]).map((m) => (
                  <button
                    key={m}
                    role="tab"
                    aria-selected={method === m}
                    className={`xai-method-tab${method === m ? ' active' : ''}`}
                    onClick={() => setMethod(m)}
                    disabled={loading}
                  >
                    {METHOD_LABELS[m]}
                  </button>
                ))}
              </div>
              <p className="xai-method-desc">{METHOD_DESCRIPTIONS[method]}</p>
            </div>

            {elapsedMs !== null && !loading && (
              <span className="xai-elapsed">⏱ {elapsedMs.toFixed(0)} ms</span>
            )}
          </div>

          {/* Right column: results */}
          <div className="xai-results">
            {loading && (
              <div className="xai-loading">
                <div className="xai-spinner" />
                <p>Running {METHOD_LABELS[method]}…</p>
                {method === 'shap' && (
                  <p className="xai-loading-note">SHAP may take up to 30 seconds</p>
                )}
              </div>
            )}

            {error && !loading && (
              <div className="xai-error">
                <span className="xai-error-icon">⚠</span>
                <p>{error}</p>
                <button className="xai-retry-btn" onClick={runExplanation}>
                  Retry
                </button>
              </div>
            )}

            {results && !loading && results.length === 0 && (
              <div className="xai-empty">
                <span className="xai-empty-icon">◎</span>
                <p>No damage classes predicted above threshold.</p>
              </div>
            )}

            {results && !loading && results.length > 0 && (
              <div className="xai-cards">
                {results
                  .filter((r) => r.image_base64)
                  .map((r) => (
                    <div key={r.class_name} className="xai-card">
                      <div className="xai-card-header">
                        <span className="xai-card-class">{r.class_name.replace(/_/g, ' ')}</span>
                        <span className="xai-card-prob">
                          {(r.probability * 100).toFixed(1)}%
                        </span>
                      </div>
                      <img
                        className="xai-card-img"
                        src={`data:image/png;base64,${r.image_base64}`}
                        alt={`${METHOD_LABELS[method]} heatmap for ${r.class_name}`}
                      />
                    </div>
                  ))}
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  )
}
