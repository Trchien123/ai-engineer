/**
 * Custom hook for video capture functionality
 */
import { useRef, useCallback, useState } from 'react'

export interface VideoFrame {
  dataUrl: string
  width: number
  height: number
}

export const useVideoCapture = () => {
  const videoRef = useRef<HTMLVideoElement>(null)
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const [isPlaying, setIsPlaying] = useState(false)
  const [currentFrameDataUrl, setCurrentFrameDataUrl] = useState<string | null>(null)

  const playVideo = useCallback(() => {
    if (videoRef.current) {
      videoRef.current.play()
      setIsPlaying(true)
    }
  }, [])

  const pauseVideo = useCallback(() => {
    if (videoRef.current) {
      videoRef.current.pause()
      setIsPlaying(false)
    }
  }, [])

  const captureFrame = useCallback((): VideoFrame | null => {
    const video = videoRef.current
    const canvas = canvasRef.current

    if (!video || !canvas) {
      console.error('Video or canvas ref not available')
      return null
    }

    const ctx = canvas.getContext('2d')
    if (!ctx) {
      console.error('Cannot get canvas context')
      return null
    }

    // Set canvas size to match video
    canvas.width = video.videoWidth
    canvas.height = video.videoHeight

    // Draw current video frame to canvas
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height)

    // Convert to data URL
    const dataUrl = canvas.toDataURL('image/jpeg', 0.95)
    setCurrentFrameDataUrl(dataUrl)

    return {
      dataUrl,
      width: canvas.width,
      height: canvas.height,
    }
  }, [])

  const loadVideo = useCallback((file: File) => {
    if (videoRef.current) {
      const url = URL.createObjectURL(file)
      videoRef.current.src = url
    }
  }, [])

  const stopVideo = useCallback(() => {
    if (videoRef.current) {
      videoRef.current.pause()
      videoRef.current.currentTime = 0
      setIsPlaying(false)
    }
  }, [])

  return {
    videoRef,
    canvasRef,
    isPlaying,
    currentFrameDataUrl,
    playVideo,
    pauseVideo,
    stopVideo,
    captureFrame,
    loadVideo,
  }
}
