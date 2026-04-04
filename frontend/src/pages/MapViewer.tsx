/**
 * Map Viewer Component
 * Displays Google Maps and captures views for detection
 */
import React, { useCallback } from 'react'
import { useMapCapture } from '../hooks/useMapCapture'
import { useDetectionStore } from '../hooks/useDetectionState'
import { DetectionService } from '../services/apiClient'
import { LoadingSpinner, ErrorMessage } from '../components'
import './MapViewer.css'

export const MapViewer: React.FC = () => {
  const {
    mapContainerRef,
    isCapturing,
    currentCaptureDataUrl,
    captureMapView,
  } = useMapCapture()

  const selectedModel = useDetectionStore((state) => state.selectedModel)
  const detecting = useDetectionStore((state) => state.detecting)
  const detectionError = useDetectionStore((state) => state.detectionError)
  const setDetecting = useDetectionStore((state) => state.setDetecting)
  const setDetectionResult = useDetectionStore((state) => state.setDetectionResult)
  const setDetectionError = useDetectionStore((state) => state.setDetectionError)
  const setCurrentImage = useDetectionStore((state) => state.setCurrentImage)

  const handleCaptureAndDetect = useCallback(async () => {
    const mapFrame = await captureMapView()
    if (!mapFrame) {
      setDetectionError('Failed to capture map view')
      return
    }

    // Store the captured image
    setCurrentImage(mapFrame.dataUrl, { width: mapFrame.width, height: mapFrame.height })

    // Run detection
    setDetecting(true)
    setDetectionError(null)

    try {
      const result = await DetectionService.detectFromBase64(mapFrame.dataUrl, selectedModel)
      setDetectionResult(result)
    } catch (error: any) {
      setDetectionError(error.message || 'Detection failed')
    } finally {
      setDetecting(false)
    }
  }, [captureMapView, selectedModel, setDetecting, setDetectionResult, setDetectionError, setCurrentImage])

  return (
    <div className="map-viewer">
      {detectionError && (
        <ErrorMessage
          error={detectionError}
          onDismiss={() => setDetectionError(null)}
        />
      )}

      <div className="map-header">
        <h3>Street Map View Detector</h3>
        <p>Click to place markers and capture for detection</p>
      </div>

      <div ref={mapContainerRef} className="map-container">
        {/* Google Maps will be embedded here. For now, placeholder */}
        <div className="map-placeholder">
          <p>📍 Google Maps integration coming soon</p>
          <p style={{ fontSize: '12px', marginTop: '10px' }}>
            This section will display an interactive Google Map where you can:
          </p>
          <ul style={{ fontSize: '12px', textAlign: 'left', display: 'inline-block' }}>
            <li>Navigate to any location</li>
            <li>Capture the current view</li>
            <li>Run object detection on the view</li>
          </ul>
        </div>
      </div>

      <div className="map-controls">
        <button
          onClick={handleCaptureAndDetect}
          disabled={isCapturing || detecting}
          className="capture-btn"
        >
          📷 {detecting ? 'Processing...' : isCapturing ? 'Capturing...' : 'Capture & Detect'}
        </button>
      </div>

      {currentCaptureDataUrl && (
        <div className="capture-preview">
          <h4>Captured Map View</h4>
          <img src={currentCaptureDataUrl} alt="Captured map view" />
        </div>
      )}

      {detecting && <LoadingSpinner message="Running detection on map view..." />}
    </div>
  )
}
