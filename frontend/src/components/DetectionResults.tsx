/**
 * Detection Results Component
 * Displays detection results with bounding boxes overlay
 */
import React, { useEffect, useRef } from 'react'
import { DetectionBox } from '../types/detection'
import './DetectionResults.css'

interface Props {
  imageDataUrl: string
  detections: DetectionBox[]
  imageWidth: number
  imageHeight: number
  inferenceTime: number
}

export const DetectionResults: React.FC<Props> = ({
  imageDataUrl,
  detections,
  imageWidth,
  imageHeight,
  inferenceTime,
}) => {
  const canvasRef = useRef<HTMLCanvasElement>(null)

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return

    const ctx = canvas.getContext('2d')
    if (!ctx) return

    // Load image
    const img = new Image()
    img.onload = () => {
      // Set canvas size to match image
      canvas.width = imageWidth
      canvas.height = imageHeight

      // Draw image
      ctx.drawImage(img, 0, 0, imageWidth, imageHeight)

      // Draw detections
      drawDetections(ctx, detections, imageWidth, imageHeight)
    }
    img.src = imageDataUrl
  }, [imageDataUrl, detections, imageWidth, imageHeight])

  const drawDetections = (
    ctx: CanvasRenderingContext2D,
    detections: DetectionBox[],
    width: number,
    height: number
  ) => {
    // Define colors for different classes
    const colorMap: { [key: string]: string } = {
      plastic: '#FF6B6B',
      paper: '#4ECDC4',
      metal: '#FFE66D',
      glass: '#95E1D3',
      damaged_sign: '#FF6B9D',
      normal_sign: '#C5B358',
    }

    detections.forEach((det, index) => {
      const color = colorMap[det.label] || '#007BFF'

      // Draw bounding box
      ctx.strokeStyle = color
      ctx.lineWidth = 3
      ctx.strokeRect(det.x_min, det.y_min, det.x_max - det.x_min, det.y_max - det.y_min)

      // Draw label background
      const label = `${det.label} (${(det.confidence * 100).toFixed(1)}%)`
      const fontSize = 14
      ctx.font = `${fontSize}px Arial`
      const textWidth = ctx.measureText(label).width
      const textHeight = fontSize + 4

      ctx.fillStyle = color
      ctx.fillRect(det.x_min, det.y_min - textHeight - 4, textWidth + 4, textHeight)

      // Draw label text
      ctx.fillStyle = 'white'
      ctx.fillText(label, det.x_min + 2, det.y_min - 6)
    })
  }

  // Group detections by label for summary
  const groupedDetections = detections.reduce(
    (acc, det) => {
      if (!acc[det.label]) {
        acc[det.label] = []
      }
      acc[det.label].push(det)
      return acc
    },
    {} as { [key: string]: DetectionBox[] }
  )

  return (
    <div className="detection-results">
      <div className="results-header">
        <h3>Detection Results</h3>
        <div className="inference-time">Inference time: {inferenceTime.toFixed(2)}ms</div>
      </div>

      <div className="results-container">
        <div className="canvas-wrapper">
          <canvas
            ref={canvasRef}
            className="detection-canvas"
            style={{
              maxWidth: '100%',
              height: 'auto',
              border: '2px solid #ddd',
              borderRadius: '4px',
            }}
          />
        </div>

        <div className="detections-summary">
          <h4>Summary:</h4>
          {detections.length === 0 ? (
            <p className="no-detections">No objects detected</p>
          ) : (
            <>
              <p className="total-detections">Total detections: {detections.length}</p>
              <div className="detection-list">
                {Object.entries(groupedDetections).map(([label, dets]) => (
                  <div key={label} className="detection-group">
                    <strong>{label}</strong>
                    <ul>
                      {dets.map((det, idx) => (
                        <li key={idx}>
                          <span className="confidence">
                            {(det.confidence * 100).toFixed(1)}%
                          </span>
                          <span className="bbox">
                            ({Math.round(det.x_min)}, {Math.round(det.y_min)}) -{' '}
                            ({Math.round(det.x_max)}, {Math.round(det.y_max)})
                          </span>
                        </li>
                      ))}
                    </ul>
                  </div>
                ))}
              </div>
            </>
          )}
        </div>
      </div>
    </div>
  )
}
