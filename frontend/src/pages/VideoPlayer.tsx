/**
 * Video Player Component
 * Plays video and captures frames for detection
 */
import React, { useCallback } from 'react'
import { useVideoCapture } from '../hooks/useVideoCapture'
import { useDetectionStore } from '../hooks/useDetectionState'
import { DetectionService } from '../services/apiClient'
import { LoadingSpinner, ErrorMessage } from '../components'
import './VideoPlayer.css'

interface Props {
  videoFile: File | null
}

export const VideoPlayer: React.FC<Props> = ({ videoFile }) => {
  const {
    videoRef,
    canvasRef,
    isPlaying,
    currentFrameDataUrl,
    playVideo,
    pauseVideo,
    stopVideo,
    captureFrame,
    loadVideo,
  } = useVideoCapture()

  const selectedModel = useDetectionStore((state) => state.selectedModel)
  const detecting = useDetectionStore((state) => state.detecting)
  const detectionError = useDetectionStore((state) => state.detectionError)
  const setDetecting = useDetectionStore((state) => state.setDetecting)
  const setDetectionResult = useDetectionStore((state) => state.setDetectionResult)
  const setDetectionError = useDetectionStore((state) => state.setDetectionError)
  const setCurrentImage = useDetectionStore((state) => state.setCurrentImage)

  React.useEffect(() => {
    if (videoFile) {
      loadVideo(videoFile)
    }
  }, [videoFile, loadVideo])

  const handleCaptureAndDetect = useCallback(async () => {
    const frame = captureFrame()
    if (!frame) {
      setDetectionError('Failed to capture frame')
      return
    }

    // Store the captured image
    setCurrentImage(frame.dataUrl, { width: frame.width, height: frame.height })

    // Pause video
    pauseVideo()

    // Run detection
    setDetecting(true)
    setDetectionError(null)

    try {
      const result = await DetectionService.detectFromBase64(frame.dataUrl, selectedModel)
      setDetectionResult(result)
    } catch (error: any) {
      setDetectionError(error.message || 'Detection failed')
    } finally {
      setDetecting(false)
    }
  }, [captureFrame, pauseVideo, selectedModel, setDetecting, setDetectionResult, setDetectionError, setCurrentImage])

  if (!videoFile) {
    return (
      <div className="video-player-empty">
        <p>Upload a video to get started</p>
      </div>
    )
  }

  return (
    <div className="video-player">
      {detectionError && (
        <ErrorMessage
          error={detectionError}
          onDismiss={() => setDetectionError(null)}
        />
      )}

      <div className="video-container">
        <video
          ref={videoRef}
          controls
          className="video-element"
          onLoadedMetadata={() => console.log('Video loaded')}
        />
        <canvas ref={canvasRef} style={{ display: 'none' }} />
      </div>

      <div className="player-controls">
        <button
          onClick={playVideo}
          disabled={isPlaying || detecting}
          className="control-btn"
        >
          ▶ Play
        </button>
        <button
          onClick={pauseVideo}
          disabled={!isPlaying || detecting}
          className="control-btn"
        >
          ⏸ Pause
        </button>
        <button
          onClick={stopVideo}
          disabled={detecting}
          className="control-btn"
        >
          ⏹ Stop
        </button>
        <button
          onClick={handleCaptureAndDetect}
          disabled={!isPlaying || detecting}
          className="control-btn capture-btn"
        >
          📷 {detecting ? 'Processing...' : 'Capture & Detect'}
        </button>
      </div>

      {currentFrameDataUrl && (
        <div className="frame-preview">
          <h4>Captured Frame</h4>
          <img src={currentFrameDataUrl} alt="Captured frame" />
        </div>
      )}

      {detecting && <LoadingSpinner message="Running detection..." />}
    </div>
  )
}
