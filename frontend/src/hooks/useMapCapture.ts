/**
 * Custom hook for map capture functionality
 */
import { useRef, useCallback, useState } from 'react'
import html2canvas from 'html2canvas'

export interface MapFrame {
  dataUrl: string
  width: number
  height: number
}

export const useMapCapture = () => {
  const mapContainerRef = useRef<HTMLDivElement>(null)
  const [isCapturing, setIsCapturing] = useState(false)
  const [currentCaptureDataUrl, setCurrentCaptureDataUrl] = useState<string | null>(null)

  const captureMapView = useCallback(async (): Promise<MapFrame | null> => {
    if (!mapContainerRef.current) {
      console.error('Map container ref not available')
      return null
    }

    try {
      setIsCapturing(true)

      // Use html2canvas to capture the map view
      const canvas = await html2canvas(mapContainerRef.current, {
        allowTaint: true,
        useCORS: true,
        backgroundColor: '#ffffff',
      })

      const dataUrl = canvas.toDataURL('image/jpeg', 0.95)
      setCurrentCaptureDataUrl(dataUrl)

      return {
        dataUrl,
        width: canvas.width,
        height: canvas.height,
      }
    } catch (error) {
      console.error('Failed to capture map view:', error)
      return null
    } finally {
      setIsCapturing(false)
    }
  }, [])

  return {
    mapContainerRef,
    isCapturing,
    currentCaptureDataUrl,
    captureMapView,
  }
}
