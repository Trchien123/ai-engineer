/**
 * Detection Results Component
 * Renders detection bounding boxes on a canvas and a grouped summary panel
 * with cropped thumbnails per detection. Clicking a thumbnail zooms the
 * main canvas into that object. "View all" resets to the full image.
 */
import React, { useEffect, useRef, useState, useCallback } from 'react'
import { DetectionBox } from '../types/detection'
import './DetectionResults.css'

interface Props {
  imageDataUrl: string
  detections: DetectionBox[]
  imageWidth: number
  imageHeight: number
  inferenceTime: number
}

const CLASS_COLORS: Record<string, string> = {
  plastic:         '#EF4444',
  plastic_bottle:  '#EF4444',
  paper:           '#14B8A6',
  cardboard:       '#10B981',
  metal:           '#F59E0B',
  metal_can:       '#F59E0B',
  glass:           '#06B6D4',
  glass_bottle:    '#0EA5E9',
  organic:         '#22C55E',
  cigarette:       '#A78BFA',
  styrofoam:       '#FB923C',
  trash:           '#F87171',
  rubbish_area:    '#FBBF24',
  damaged_sign:    '#EC4899',
  normal_sign:     '#8B5CF6',
}

const getColor = (label: string) =>
  CLASS_COLORS[label.toLowerCase()] ?? '#3b82f6'

const prettyLabel = (label: string) =>
  label.replace(/_/g, ' ').replace(/\b\w/g, (c) => c.toUpperCase())

// ─── Crop Thumbnail ──────────────────────────────────────────────────────────

interface CropProps {
  imageDataUrl: string
  x_min: number
  y_min: number
  x_max: number
  y_max: number
  size?: number
  selected?: boolean
  color?: string
  onClick?: () => void
}

const CropThumbnail: React.FC<CropProps> = ({
  imageDataUrl, x_min, y_min, x_max, y_max, size = 60, selected, color, onClick,
}) => {
  const ref = useRef<HTMLCanvasElement>(null)

  useEffect(() => {
    const canvas = ref.current
    if (!canvas) return
    const ctx = canvas.getContext('2d')
    if (!ctx) return

    const img = new Image()
    img.onload = () => {
      const sw = Math.max(1, x_max - x_min)
      const sh = Math.max(1, y_max - y_min)
      const aspect = sw / sh
      let dw = size, dh = size
      if (aspect > 1) { dh = size / aspect } else { dw = size * aspect }
      canvas.width  = Math.round(dw)
      canvas.height = Math.round(dh)
      ctx.drawImage(img, x_min, y_min, sw, sh, 0, 0, canvas.width, canvas.height)
    }
    img.src = imageDataUrl
  }, [imageDataUrl, x_min, y_min, x_max, y_max, size])

  return (
    <canvas
      ref={ref}
      className={[
        'crop-thumbnail',
        onClick ? 'crop-thumbnail--clickable' : '',
        selected ? 'crop-thumbnail--selected' : '',
      ].filter(Boolean).join(' ')}
      style={selected && color ? { boxShadow: `0 0 0 2px ${color}` } : undefined}
      title={selected ? 'Viewing this detection' : 'Click to focus'}
      onClick={onClick}
    />
  )
}

// ─── Main Component ───────────────────────────────────────────────────────────

interface FocusedDet {
  det: DetectionBox
  label: string
}

export const DetectionResults: React.FC<Props> = ({
  imageDataUrl,
  detections,
  imageWidth,
  imageHeight,
  inferenceTime,
}) => {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const [focused, setFocused] = useState<FocusedDet | null>(null)

  // ── Draw helpers ────────────────────────────────────────────

  const drawAllBoxes = useCallback(
    (ctx: CanvasRenderingContext2D, img: HTMLImageElement) => {
      const canvas = ctx.canvas
      canvas.width  = imageWidth
      canvas.height = imageHeight
      ctx.drawImage(img, 0, 0, imageWidth, imageHeight)

      const fontSize  = Math.max(12, Math.min(imageWidth / 60, 18))
      const lineWidth = Math.max(2, imageWidth / 500)
      ctx.font = `500 ${fontSize}px Inter, sans-serif`

      const areaBoxes = detections.filter(d => d.label.toLowerCase() === 'rubbish_area')
      const typeBoxes = detections.filter(d => d.label.toLowerCase() !== 'rubbish_area')

      areaBoxes.forEach((det) => {
        const color = getColor(det.label)
        const x = det.x_min, y = det.y_min
        const w = det.x_max - det.x_min, h = det.y_max - det.y_min
        ctx.save()
        ctx.setLineDash([8, 5])
        ctx.strokeStyle = color
        ctx.lineWidth = lineWidth * 1.5
        ctx.strokeRect(x, y, w, h)
        ctx.fillStyle = color + '18'
        ctx.fillRect(x, y, w, h)
        ctx.restore()

        const label = `area ${(det.confidence * 100).toFixed(0)}%`
        const textW = ctx.measureText(label).width
        const chipH = fontSize + 8
        const chipY = y - chipH < 0 ? y + 2 : y - chipH
        ctx.fillStyle = color
        ctx.beginPath()
        ctx.roundRect(x, chipY, textW + 12, chipH, 3)
        ctx.fill()
        ctx.fillStyle = '#000'
        ctx.fillText(label, x + 6, chipY + chipH - 5)
      })

      typeBoxes.forEach((det) => {
        const color = getColor(det.label)
        const x = det.x_min, y = det.y_min
        const w = det.x_max - det.x_min, h = det.y_max - det.y_min
        const label = `${det.label} ${(det.confidence * 100).toFixed(0)}%`

        ctx.strokeStyle = color
        ctx.lineWidth = lineWidth
        ctx.strokeRect(x, y, w, h)

        const textW = ctx.measureText(label).width
        const chipH = fontSize + 8
        const chipY = y - chipH < 0 ? y + 2 : y - chipH
        ctx.fillStyle = color
        ctx.beginPath()
        ctx.roundRect(x, chipY, textW + 12, chipH, 3)
        ctx.fill()
        ctx.fillStyle = '#fff'
        ctx.fillText(label, x + 6, chipY + chipH - 5)
      })
    },
    [detections, imageWidth, imageHeight]
  )

  const drawFocused = useCallback(
    (ctx: CanvasRenderingContext2D, img: HTMLImageElement, det: DetectionBox, label: string) => {
      const canvas = ctx.canvas
      const color = getColor(label)

      // Full image — same dimensions as "all" mode
      canvas.width  = imageWidth
      canvas.height = imageHeight
      ctx.drawImage(img, 0, 0, imageWidth, imageHeight)

      const x = det.x_min
      const y = det.y_min
      const w = det.x_max - det.x_min
      const h = det.y_max - det.y_min

      // Subtle fill inside the box
      ctx.fillStyle = color + '28'
      ctx.fillRect(x, y, w, h)

      // Box border
      const lineWidth = Math.max(2, imageWidth / 400)
      ctx.strokeStyle = color
      ctx.lineWidth = lineWidth
      ctx.strokeRect(x, y, w, h)

      // Label badge
      const fontSize = Math.max(12, Math.min(imageWidth / 55, 20))
      ctx.font = `600 ${fontSize}px Inter, sans-serif`
      const labelText = `${prettyLabel(label)}  ·  ${(det.confidence * 100).toFixed(0)}%`
      const textW = ctx.measureText(labelText).width
      const chipH = fontSize + 10
      const chipY = y - chipH - 3 < 0 ? y + 3 : y - chipH - 3

      ctx.fillStyle = color
      ctx.beginPath()
      ctx.roundRect(x, chipY, textW + 14, chipH, 4)
      ctx.fill()
      ctx.fillStyle = '#fff'
      ctx.fillText(labelText, x + 7, chipY + chipH - 5)
    },
    [imageWidth, imageHeight]
  )

  // ── Re-draw canvas whenever focused state or image changes ──

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return
    const ctx = canvas.getContext('2d')
    if (!ctx) return

    const img = new Image()
    img.onload = () => {
      if (focused) {
        drawFocused(ctx, img, focused.det, focused.label)
      } else {
        drawAllBoxes(ctx, img)
      }
    }
    img.src = imageDataUrl
  }, [imageDataUrl, focused, drawAllBoxes, drawFocused])

  // ── Helpers ─────────────────────────────────────────────────

  const handleThumbnailClick = useCallback((det: DetectionBox, label: string) => {
    setFocused((prev) =>
      prev?.det === det && prev?.label === label ? null : { det, label }
    )
  }, [])

  const handleShowAll = useCallback(() => setFocused(null), [])

  // ── Group detections ─────────────────────────────────────────

  const areaDetections = detections.filter(d => d.label.toLowerCase() === 'rubbish_area')
  const typeDetections = detections.filter(d => d.label.toLowerCase() !== 'rubbish_area')
  const isRubbish = areaDetections.length > 0 || typeDetections.some(d =>
    ['plastic','paper','cardboard','metal','glass','organic','cigarette','styrofoam','trash',
     'plastic_bottle','metal_can','glass_bottle'].includes(d.label.toLowerCase())
  )

  const grouped = typeDetections.reduce<Record<string, DetectionBox[]>>((acc, det) => {
    if (!acc[det.label]) acc[det.label] = []
    acc[det.label].push(det)
    return acc
  }, {})

  return (
    <section className="detection-results" aria-label="Detection results">
      {/* ── Header ── */}
      <div className="results-header">
        <h3>Results</h3>
        <span className="inference-badge">⏱ {inferenceTime.toFixed(0)} ms</span>
      </div>

      <div className="results-grid">
        {/* ── Left: annotated canvas ── */}
        <div className="canvas-col">
          {/* Show-all toolbar — only visible when a detection is focused */}
          {focused && (
            <div className="canvas-toolbar">
              <div className="canvas-toolbar-info">
                <span
                  className="canvas-toolbar-dot"
                  style={{ background: getColor(focused.label) }}
                />
                <span className="canvas-toolbar-label">
                  {prettyLabel(focused.label)}
                </span>
                <span className="canvas-toolbar-conf">
                  {(focused.det.confidence * 100).toFixed(0)}%
                </span>
              </div>
              <button
                className="btn-show-all"
                onClick={handleShowAll}
                aria-label="Show all detections"
              >
                ← View all
              </button>
            </div>
          )}

          <div className="canvas-wrapper">
            <canvas
              ref={canvasRef}
              className="detection-canvas"
              aria-label={focused ? `Focused: ${focused.label}` : 'Annotated detection image'}
            />
          </div>
        </div>

        {/* ── Right: summary panel ── */}
        <aside className="detections-summary" aria-label="Detection summary">
          <div className="summary-header">
            <h4>Detections</h4>
            {detections.length > 0 && (
              <span className="summary-count">{detections.length}</span>
            )}
          </div>

          <div className="summary-body">
            {detections.length === 0 ? (
              <div className="no-detections" role="status">
                <span className="no-detections-icon">◎</span>
                No objects detected
              </div>
            ) : (
              <ul className="detection-list" role="list">

                {/* Rubbish area group */}
                {isRubbish && areaDetections.length > 0 && (
                  <li className="detection-group">
                    <div className="detection-group-label"
                      style={{ borderLeft: `3px solid ${getColor('rubbish_area')}` }}>
                      <strong style={{ color: getColor('rubbish_area') }}>Rubbish Areas</strong>
                      <span className="detection-group-badge">{areaDetections.length}</span>
                    </div>
                    <ul className="detection-item-list">
                      {areaDetections.map((det, i) => {
                        const isFocused = focused?.det === det
                        return (
                          <li key={i} className={`detection-item${isFocused ? ' detection-item--focused' : ''}`}>
                            <CropThumbnail
                              imageDataUrl={imageDataUrl}
                              x_min={det.x_min} y_min={det.y_min}
                              x_max={det.x_max} y_max={det.y_max}
                              selected={isFocused}
                              color={getColor('rubbish_area')}
                              onClick={() => handleThumbnailClick(det, 'rubbish_area')}
                            />
                            <div className="detection-item-info">
                              <span className="confidence-pill"
                                style={{ background: getColor('rubbish_area') }}>
                                {(det.confidence * 100).toFixed(0)}%
                              </span>
                              <span className="bbox-coords">
                                {Math.round(det.x_min)},{Math.round(det.y_min)} →{' '}
                                {Math.round(det.x_max)},{Math.round(det.y_max)}
                              </span>
                            </div>
                          </li>
                        )
                      })}
                    </ul>
                  </li>
                )}

                {/* Type detection groups */}
                {Object.entries(grouped).map(([label, dets]) => (
                  <li key={label} className="detection-group">
                    <div className="detection-group-label"
                      style={{ borderLeft: `3px solid ${getColor(label)}` }}>
                      <strong style={{ color: getColor(label) }}>{prettyLabel(label)}</strong>
                      <span className="detection-group-badge">{dets.length}</span>
                    </div>
                    <ul className="detection-item-list">
                      {dets.map((det, i) => {
                        const isFocused = focused?.det === det
                        return (
                          <li key={i} className={`detection-item${isFocused ? ' detection-item--focused' : ''}`}>
                            <CropThumbnail
                              imageDataUrl={imageDataUrl}
                              x_min={det.x_min} y_min={det.y_min}
                              x_max={det.x_max} y_max={det.y_max}
                              selected={isFocused}
                              color={getColor(label)}
                              onClick={() => handleThumbnailClick(det, label)}
                            />
                            <div className="detection-item-info">
                              <span className="confidence-pill"
                                style={{ background: getColor(label) }}>
                                {(det.confidence * 100).toFixed(0)}%
                              </span>
                              <span className="bbox-coords">
                                {Math.round(det.x_min)},{Math.round(det.y_min)} →{' '}
                                {Math.round(det.x_max)},{Math.round(det.y_max)}
                              </span>
                            </div>
                          </li>
                        )
                      })}
                    </ul>
                  </li>
                ))}

              </ul>
            )}
          </div>
        </aside>
      </div>
    </section>
  )
}
