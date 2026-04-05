/**
 * Map Viewer Component
 * Displays Google Maps and captures views for detection
 */
import React, { useCallback, useEffect, useRef, useState } from 'react'
import { useMapCapture } from '../hooks/useMapCapture'
import { useDetectionStore } from '../hooks/useDetectionState'
import { DetectionService } from '../services/apiClient'
import { LoadingSpinner, ErrorMessage } from '../components'
import './MapViewer.css'

declare global {
  interface Window {
    google: any;
  }
}

export const MapViewer: React.FC = () => {
  const mapRef = useRef<HTMLDivElement | null>(null)
  const mapInstanceRef = useRef<any>(null)
  const [mapLoaded, setMapLoaded] = useState(false)
  const [mapError, setMapError] = useState<string | null>(null)
  const [scriptLoaded, setScriptLoaded] = useState(false)
  // ✅ REMOVED: mapInitialized state — was causing re-init loops

  const {
    isCapturing,
    currentCaptureDataUrl,
  } = useMapCapture(mapRef)

  const selectedModel = useDetectionStore((state) => state.selectedModel)
  const detecting = useDetectionStore((state) => state.detecting)
  const detectionError = useDetectionStore((state) => state.detectionError)
  const setDetecting = useDetectionStore((state) => state.setDetecting)
  const setDetectionResult = useDetectionStore((state) => state.setDetectionResult)
  const setDetectionError = useDetectionStore((state) => state.setDetectionError)
  const setCurrentImage = useDetectionStore((state) => state.setCurrentImage)

  // ✅ REMOVED: setMapRef callback — was resetting state on every render
  // mapRef is now used directly as a plain ref on the div

  // Load Google Maps API script
  useEffect(() => {
    const apiKey = import.meta.env.VITE_GOOGLE_MAPS_API_KEY

    if (!apiKey || apiKey === 'YOUR_DEMO_KEY') {
      setMapError('Google Maps API key not configured. Please set VITE_GOOGLE_MAPS_API_KEY in .env.local')
      return
    }

    if (window.google?.maps) {
      setScriptLoaded(true)
      return
    }

    if (scriptLoaded) return

    const existingScript = document.querySelector('script[src*="maps.googleapis.com"]')
    if (existingScript) {
      setScriptLoaded(true)
      return
    }

    const script = document.createElement('script')
    script.src = `https://maps.googleapis.com/maps/api/js?key=${apiKey}&libraries=geometry&loading=async`
    script.async = true
    script.defer = true
    document.head.appendChild(script)

    script.onload = () => setScriptLoaded(true)
    script.onerror = () => setMapError('Failed to load Google Maps. Check your internet connection.')
  }, [scriptLoaded])

  // Initialize Google Map
  useEffect(() => {
    // ✅ Guard with mapInstanceRef.current directly — no mapInitialized state needed
    if (!scriptLoaded || !mapRef.current || mapInstanceRef.current) return

    let attempts = 0
    const maxAttempts = 100
    let timeoutId: ReturnType<typeof setTimeout>

    const initMap = () => {
      if (!window.google?.maps) {
        if (++attempts >= maxAttempts) {
          setMapError('Google Maps API failed to load within timeout')
          return
        }
        timeoutId = setTimeout(initMap, 100)
        return
      }

      // ✅ Double-check ref and instance are still valid before initializing
      if (!mapRef.current || mapInstanceRef.current) return

      try {
        mapInstanceRef.current = new window.google.maps.Map(mapRef.current, {
          center: { lat: 40.7128, lng: -74.0060 },
          zoom: 15,
          mapTypeId: window.google.maps.MapTypeId.ROADMAP,
          streetViewControl: true,
          mapTypeControl: true,
          fullscreenControl: true,
          // ✅ Tells Google Maps where the control buttons should sit
          gestureHandling: 'greedy',
        })

        setMapLoaded(true)
        setMapError(null)
      } catch (error: any) {
        console.error('Error initializing Google Map:', error)
        setMapError(`Failed to initialize map: ${error.message || 'Unknown error'}`)
      }
    }

    initMap()

    // ✅ Clean up the pending timeout if effect re-runs before map loads
    return () => clearTimeout(timeoutId)
  }, [scriptLoaded]) // ✅ Only depends on scriptLoaded — not mapInitialized

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      if (mapInstanceRef.current) {
        window.google?.maps?.event?.clearInstanceListeners?.(mapInstanceRef.current)
        mapInstanceRef.current = null
      }
    }
  }, [])

  const handleCaptureAndDetect = useCallback(async () => {
    if (!mapInstanceRef.current || !mapRef.current) {
      setDetectionError('Map not initialized')
      return
    }

    setDetecting(true)
    setDetectionError(null)

    try {
      const apiKey = import.meta.env.VITE_GOOGLE_MAPS_API_KEY;
      
      // 1. Calculate the exact aspect ratio of your UI container
      const rect = mapRef.current.getBoundingClientRect();
      const aspectRatio = rect.width / rect.height;

      // 2. Scale it to fit within Google's 640x640 free tier limit 
      // while PRESERVING the exact shape of your viewer
      let width = 640;
      let height = 640;
      
      if (aspectRatio > 1) {
        // Landscape (wider than it is tall)
        width = 640;
        height = Math.round(640 / aspectRatio);
      } else {
        // Portrait (taller than it is wide)
        height = 640;
        width = Math.round(640 * aspectRatio);
      }
      
      let imageUrl = '';

      const streetView = mapInstanceRef.current.getStreetView();
      
      if (streetView && streetView.getVisible()) {
        const pov = streetView.getPov(); 
        const zoom = streetView.getZoom() || 1;
        
        // 3. Use the more accurate exponential FOV formula
        const fov = 180 / Math.pow(2, zoom); 
        
        const panoId = streetView.getPano();
        if (!panoId) throw new Error("Could not find Street View Panorama ID.");

        imageUrl = `https://maps.googleapis.com/maps/api/streetview?size=${width}x${height}&pano=${panoId}&heading=${pov.heading}&pitch=${pov.pitch}&fov=${fov}&key=${apiKey}`;
        
      } else {
        const center = mapInstanceRef.current.getCenter();
        const zoom = mapInstanceRef.current.getZoom();
        imageUrl = `https://maps.googleapis.com/maps/api/staticmap?center=${center.lat()},${center.lng()}&zoom=${zoom}&size=${width}x${height}&maptype=roadmap&key=${apiKey}`;
      }

      const response = await fetch(imageUrl);
      if (!response.ok) throw new Error('Failed to fetch image from Google API');
      
      const blob = await response.blob();
      const reader = new FileReader();
      
      reader.readAsDataURL(blob);
      reader.onloadend = async () => {
        const base64data = reader.result as string;
        
        setCurrentImage(base64data, { width, height });
        
        try {
          const result = await DetectionService.detectFromBase64(base64data, selectedModel);
          setDetectionResult(result);
        } catch (error: any) {
          setDetectionError(error.message || 'Detection failed');
        } finally {
          setDetecting(false);
        }
      };
      
    } catch (error: any) {
      setDetectionError(error.message || 'Failed to capture view');
      setDetecting(false);
    }
  }, [selectedModel, setDetecting, setDetectionResult, setDetectionError, setCurrentImage]);

  return (
    <div className="map-viewer">
      {detectionError && (
        <ErrorMessage error={detectionError} onDismiss={() => setDetectionError(null)} />
      )}
      {mapError && (
        <ErrorMessage error={mapError} onDismiss={() => setMapError(null)} />
      )}

      <div className="map-header">
        <h3>Street Map View Detector</h3>
        <p>Navigate the map and capture views for detection</p>
      </div>

      <div className="map-container-wrapper">
        {!mapLoaded && !mapError && (
          <div className="map-overlay">
            <LoadingSpinner message="Loading Google Maps..." />
          </div>
        )}
        {mapError && (
          <div className="map-overlay map-placeholder">
            <p style={{ fontWeight: 600, color: 'var(--text-primary)' }}>Map unavailable</p>
            <ul>
              <li>Check your API key in .env.local</li>
              <li>Ensure Maps JavaScript API is enabled</li>
              <li>Check browser console for details</li>
            </ul>
          </div>
        )}
        {/* ✅ Plain ref — React never renders children here, Google Maps owns this node */}
        <div ref={mapRef} className="map-container" />
      </div>

      <div className="map-controls">
        <button
          onClick={handleCaptureAndDetect}
          disabled={!mapLoaded || isCapturing || detecting}
          className="map-capture-btn"
          aria-label="Capture current map view and run detection"
        >
          {detecting ? 'Detecting…' : isCapturing ? 'Capturing…' : '⊙ Capture & Detect'}
        </button>
      </div>

      {currentCaptureDataUrl && (
        <div className="capture-preview">
          <span className="capture-preview-label">Captured map view</span>
          <img src={currentCaptureDataUrl} alt="Captured map view" loading="lazy" />
        </div>
      )}

      {detecting && <LoadingSpinner message="Running detection on map view..." />}
    </div>
  )
}