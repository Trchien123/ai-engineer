/**
 * Main Application Component
 * Orchestrates all features: video upload, map view, model selection, and detection
 */
import { useEffect, useState } from 'react'
import { useDetectionStore } from './hooks/useDetectionState'
import { ModelService, HealthService } from './services/apiClient'
import { ModelSelector, DetectionResults, LoadingSpinner, ErrorMessage, ErrorBoundary } from './components'
import { VideoUploader, VideoPlayer, MapViewer } from './pages'
import './App.css'

function App() {
  const [mediaFile, setMediaFile] = useState<File | null>(null)
  const [appError, setAppError] = useState<string | null>(null)
  const [isInitializing, setIsInitializing] = useState(true)

  const availableModels   = useDetectionStore((s) => s.availableModels)
  const currentImage      = useDetectionStore((s) => s.currentImage)
  const detectionResult   = useDetectionStore((s) => s.detectionResult)
  const activeTab         = useDetectionStore((s) => s.activeTab)
  const detecting         = useDetectionStore((s) => s.detecting)

  const setAvailableModels = useDetectionStore((s) => s.setAvailableModels)
  const setActiveTab       = useDetectionStore((s) => s.setActiveTab)
  const resetDetection     = useDetectionStore((s) => s.resetDetection)

  useEffect(() => {
    const initializeApp = async () => {
      try {
        setIsInitializing(true)
        const health = await HealthService.check()
        console.log('🟢 API Health:', health)
        const modelsResponse = await ModelService.listModels()
        console.log('📦 Models loaded:', modelsResponse)
        setAvailableModels(modelsResponse)
        if (modelsResponse.count === 0) {
          setAppError('No detection models available. Please check the backend.')
        }
      } catch (error: any) {
        console.error('❌ Initialization error:', error)
        setAppError(`Failed to initialize: ${error.message}. Make sure the backend is running on http://localhost:8000`)
      } finally {
        setIsInitializing(false)
      }
    }
    initializeApp()
  }, [setAvailableModels])

  if (isInitializing) {
    return (
      <div className="app app--loading">
        <LoadingSpinner message="Initializing Detection Studio…" size="large" />
      </div>
    )
  }

  return (
    <div className="app">
      {/* ── Header ── */}
      <header className="app-header">
        <div className="header-inner">
          {/* Brand */}
          <div className="header-brand">
            <div className="brand-dot" aria-hidden="true" />
            <span className="brand-name">Detection Studio</span>
          </div>

          {/* Tab navigation */}
          <nav className="header-tabs" role="tablist" aria-label="Analysis mode">
            <button
              role="tab"
              aria-selected={activeTab === 'video'}
              className={`tab-btn${activeTab === 'video' ? ' active' : ''}`}
              onClick={() => { setActiveTab('video'); resetDetection() }}
            >
              Media
            </button>
            <button
              role="tab"
              aria-selected={activeTab === 'map'}
              className={`tab-btn${activeTab === 'map' ? ' active' : ''}`}
              onClick={() => { setActiveTab('map'); resetDetection() }}
            >
              Map
            </button>
          </nav>

          {/* Model selector */}
          <div className="header-actions">
            {availableModels && (
              <ModelSelector
                models={availableModels.models}
                disabled={detecting}
                compact
              />
            )}
          </div>
        </div>
      </header>

      {/* ── Main ── */}
      <main className="app-main">
        {appError && (
          <ErrorMessage error={appError} onDismiss={() => setAppError(null)} />
        )}

        {/* Media tab */}
        {activeTab === 'video' && (
          <section className="feature-section">
            {!mediaFile ? (
              <VideoUploader
                onVideoSelected={(file) => { setMediaFile(file); resetDetection() }}
                disabled={detecting}
              />
            ) : (
              <div className="video-section">
                <div className="section-header">
                  <h3>
                    {mediaFile.type.startsWith('video/') ? 'Video' : 'Image'}
                    {' — '}
                    {mediaFile.name}
                  </h3>
                  <button
                    className="btn-secondary"
                    onClick={() => { setMediaFile(null); resetDetection() }}
                    disabled={detecting}
                  >
                    Change file
                  </button>
                </div>
                <VideoPlayer mediaFile={mediaFile} />
              </div>
            )}
          </section>
        )}

        {/* Map tab */}
        {activeTab === 'map' && (
          <section className="feature-section">
            <ErrorBoundary
              fallback={
                <div className="error-fallback-card">
                  <p className="error-fallback-title">Map Unavailable</p>
                  <p className="error-fallback-desc">
                    Check your Google Maps API key, network connection, and browser console for details.
                  </p>
                </div>
              }
            >
              <MapViewer />
            </ErrorBoundary>
          </section>
        )}

        {/* Detection results */}
        {currentImage && detectionResult && !detecting && (
          <section className="results-section">
            <DetectionResults
              imageDataUrl={currentImage}
              detections={detectionResult.detections}
              imageWidth={detectionResult.image_width}
              imageHeight={detectionResult.image_height}
              inferenceTime={detectionResult.inference_time_ms}
              modelType={detectionResult.model_type}
            />
          </section>
        )}

        {/* Detecting */}
        {detecting && (
          <section className="loading-section">
            <LoadingSpinner message="Running detection inference…" size="large" />
          </section>
        )}
      </main>

      <footer className="app-footer">
        <p>Detection Studio &nbsp;·&nbsp; React + FastAPI</p>
      </footer>
    </div>
  )
}

export default App
