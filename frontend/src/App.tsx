/**
 * Main Application Component
 * Orchestrates all features: video upload, map view, model selection, and detection
 */
import React, { useEffect, useState } from 'react'
import { useDetectionStore } from './hooks/useDetectionState'
import { ModelService, HealthService } from './services/apiClient'
import { ModelSelector, DetectionResults, LoadingSpinner, ErrorMessage } from './components'
import { VideoUploader, VideoPlayer, MapViewer } from './pages'
import './App.css'

function App() {
  const [videoFile, setVideoFile] = useState<File | null>(null)
  const [appError, setAppError] = useState<string | null>(null)
  const [isInitializing, setIsInitializing] = useState(true)

  // Global state
  const availableModels = useDetectionStore((state) => state.availableModels)
  const currentImage = useDetectionStore((state) => state.currentImage)
  const detectionResult = useDetectionStore((state) => state.detectionResult)
  const activeTab = useDetectionStore((state) => state.activeTab)
  const detecting = useDetectionStore((state) => state.detecting)

  const setAvailableModels = useDetectionStore((state) => state.setAvailableModels)
  const setActiveTab = useDetectionStore((state) => state.setActiveTab)
  const resetDetection = useDetectionStore((state) => state.resetDetection)

  // Initialize app on mount
  useEffect(() => {
    const initializeApp = async () => {
      try {
        setIsInitializing(true)
        
        // Check API health
        const health = await HealthService.check()
        console.log('🟢 API Health:', health)

        // Load models
        const modelsResponse = await ModelService.listModels()
        console.log('📦 Models loaded:', modelsResponse)
        setAvailableModels(modelsResponse)

        if (modelsResponse.count === 0) {
          setAppError('No detection models available. Please check the backend.')
        }
      } catch (error: any) {
        console.error('❌ Initialization error:', error)
        setAppError(`Failed to initialize app: ${error.message}. Make sure the backend is running on http://localhost:8000`)
      } finally {
        setIsInitializing(false)
      }
    }

    initializeApp()
  }, [setAvailableModels])

  if (isInitializing) {
    return (
      <div className="app">
        <LoadingSpinner message="Initializing application..." size="large" />
      </div>
    )
  }

  return (
    <div className="app">
      <header className="app-header">
        <div className="header-content">
          <h1>🎬 Object Detection System</h1>
          <p>Detailed analysis for rubbish detection and traffic signs</p>
        </div>
      </header>

      <main className="app-main">
        {appError && (
          <ErrorMessage
            error={appError}
            onDismiss={() => setAppError(null)}
          />
        )}

        {/* Model Selector */}
        {availableModels && (
          <section className="config-section">
            <ModelSelector
              models={availableModels.models}
              disabled={detecting}
            />
          </section>
        )}

        {/* Tab Navigation */}
        <div className="tab-navigation">
          <button
            className={`tab-button ${activeTab === 'video' ? 'active' : ''}`}
            onClick={() => {
              setActiveTab('video')
              resetDetection()
            }}
          >
            🎥 Video Analysis
          </button>
          <button
            className={`tab-button ${activeTab === 'map' ? 'active' : ''}`}
            onClick={() => {
              setActiveTab('map')
              resetDetection()
            }}
          >
            🗺️ Map Analysis
          </button>
        </div>

        {/* Video Tab */}
        {activeTab === 'video' && (
          <section className="feature-section">
            {!videoFile ? (
              <VideoUploader
                onVideoSelected={(file) => {
                  setVideoFile(file)
                  resetDetection()
                }}
                disabled={detecting}
              />
            ) : (
              <div className="video-section">
                <div className="section-header">
                  <h3>Video: {videoFile.name}</h3>
                  <button
                    onClick={() => {
                      setVideoFile(null)
                      resetDetection()
                    }}
                    className="btn-secondary"
                    disabled={detecting}
                  >
                    Upload Different Video
                  </button>
                </div>
                <VideoPlayer videoFile={videoFile} />
              </div>
            )}
          </section>
        )}

        {/* Map Tab */}
        {activeTab === 'map' && (
          <section className="feature-section">
            <MapViewer />
          </section>
        )}

        {/* Detection Results */}
        {currentImage && detectionResult && !detecting && (
          <section className="results-section">
            <DetectionResults
              imageDataUrl={currentImage}
              detections={detectionResult.detections}
              imageWidth={detectionResult.image_width}
              imageHeight={detectionResult.image_height}
              inferenceTime={detectionResult.inference_time_ms}
            />
          </section>
        )}

        {/* Loading indicator */}
        {detecting && (
          <section className="loading-section">
            <LoadingSpinner message="Running detection inference..." size="large" />
          </section>
        )}
      </main>

      <footer className="app-footer">
        <p>Object Detection System • Built with React + FastAPI</p>
      </footer>
    </div>
  )
}

export default App
