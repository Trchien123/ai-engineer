/**
 * Media Player Component
 * Plays video, previews images, and runs detections on uploaded media.
 */
import React, { useCallback, useEffect, useState } from 'react'
import { useVideoCapture } from '../hooks/useVideoCapture'
import { useDetectionStore } from '../hooks/useDetectionState'
import { DetectionService } from '../services/apiClient'
import { LoadingSpinner, ErrorMessage } from '../components'
import './VideoPlayer.css'

interface Props {
  mediaFile: File | null
}

export const VideoPlayer: React.FC<Props> = ({ mediaFile }) => {
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

  const selectedModel    = useDetectionStore((s) => s.selectedModel)
  const detecting        = useDetectionStore((s) => s.detecting)
  const detectionError   = useDetectionStore((s) => s.detectionError)
  const setDetecting     = useDetectionStore((s) => s.setDetecting)
  const setDetectionResult = useDetectionStore((s) => s.setDetectionResult)
  const setDetectionError  = useDetectionStore((s) => s.setDetectionError)
  const setCurrentImage  = useDetectionStore((s) => s.setCurrentImage)

  const [previewUrl, setPreviewUrl] = useState<string | null>(null)
  const isVideo = mediaFile?.type.startsWith('video/') ?? false

  useEffect(() => {
    if (!mediaFile) return
    if (isVideo) {
      loadVideo(mediaFile)
    } else {
      const url = URL.createObjectURL(mediaFile)
      setPreviewUrl(url)
      setCurrentImage(url)
    }
    return () => {
      if (previewUrl) {
        URL.revokeObjectURL(previewUrl)
        setPreviewUrl(null)
      }
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [mediaFile, isVideo])

  const handleCaptureAndDetect = useCallback(async () => {
    const frame = captureFrame()
    if (!frame) { setDetectionError('Failed to capture frame'); return }

    setCurrentImage(frame.dataUrl, { width: frame.width, height: frame.height })
    pauseVideo()
    setDetecting(true)
    setDetectionError(null)
    try {
      const result = await DetectionService.detectFromBase64(frame.dataUrl, selectedModel)
      setDetectionResult(result)
    } catch (err: any) {
      setDetectionError(err.message ?? 'Detection failed')
    } finally {
      setDetecting(false)
    }
  }, [captureFrame, pauseVideo, selectedModel, setDetecting, setDetectionResult, setDetectionError, setCurrentImage])

  const handleDetectImage = useCallback(async () => {
    if (!mediaFile) { setDetectionError('No image selected'); return }

    setDetecting(true)
    setDetectionError(null)
    try {
      if (previewUrl) setCurrentImage(previewUrl)
      const result = await DetectionService.detectFromFile(mediaFile, selectedModel)
      setDetectionResult(result)
    } catch (err: any) {
      setDetectionError(err.message ?? 'Detection failed')
    } finally {
      setDetecting(false)
    }
  }, [mediaFile, previewUrl, selectedModel, setDetecting, setDetectionResult, setDetectionError, setCurrentImage])

  if (!mediaFile) {
    return (
      <div className="video-player-empty" role="status">
        Upload a video or image to get started
      </div>
    )
  }

  return (
    <div className="video-player">
      {detectionError && (
        <ErrorMessage error={detectionError} onDismiss={() => setDetectionError(null)} />
      )}

      {isVideo ? (
        <>
          <div className="video-container">
            <video
              ref={videoRef}
              controls
              className="video-element"
              aria-label="Uploaded video"
            />
            <canvas ref={canvasRef} style={{ display: 'none' }} aria-hidden="true" />
          </div>

          <div className="player-controls" role="toolbar" aria-label="Video controls">
            <button
              onClick={playVideo}
              disabled={isPlaying || detecting}
              className="control-btn"
              aria-label="Play"
            >
              ▶ Play
            </button>
            <button
              onClick={pauseVideo}
              disabled={!isPlaying || detecting}
              className="control-btn"
              aria-label="Pause"
            >
              ⏸ Pause
            </button>
            <button
              onClick={stopVideo}
              disabled={detecting}
              className="control-btn"
              aria-label="Stop"
            >
              ⏹ Stop
            </button>
            <button
              onClick={handleCaptureAndDetect}
              disabled={!isPlaying || detecting}
              className="control-btn capture-btn"
              aria-label="Capture current frame and run detection"
            >
              {detecting ? 'Detecting…' : '⊙ Capture & Detect'}
            </button>
          </div>

          {currentFrameDataUrl && !detecting && (
            <div className="frame-preview">
              <span className="frame-preview-label">Captured frame</span>
              <img
                src={currentFrameDataUrl}
                alt="Captured video frame"
                onLoad={(e) => (e.currentTarget as HTMLImageElement).classList.add('loaded')}
                style={{ opacity: 0, transition: 'opacity 0.25s ease' }}
              />
            </div>
          )}
        </>
      ) : (
        <div className="image-preview-section">
          <div className="image-preview">
            {previewUrl && (
              <img
                src={previewUrl}
                alt="Uploaded image preview"
                onLoad={(e) => (e.currentTarget as HTMLImageElement).classList.add('loaded')}
              />
            )}
          </div>
          <button
            onClick={handleDetectImage}
            disabled={detecting}
            className="detect-image-btn"
            aria-label="Run detection on uploaded image"
          >
            {detecting ? 'Detecting…' : 'Run Detection'}
          </button>
        </div>
      )}

      {detecting && (
        <LoadingSpinner message="Running detection…" size="medium" />
      )}
    </div>
  )
}
