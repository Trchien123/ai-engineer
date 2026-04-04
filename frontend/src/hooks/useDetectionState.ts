/**
 * Global detection state management using Zustand
 */
import { create } from 'zustand'
import { DetectionResult, ModelType, ModelsListResponse } from '../types/detection'

export interface DetectionState {
  // Model management
  availableModels: ModelsListResponse | null
  selectedModel: ModelType
  modelsLoading: boolean
  modelsError: string | null

  // Current detection
  currentImage: string | null // base64 encoded
  currentImageSize: { width: number; height: number } | null
  detectionResult: DetectionResult | null
  detecting: boolean
  detectionError: string | null

  // UI state
  showHistory: boolean
  activeTab: 'video' | 'map'

  // Actions
  setSelectedModel: (model: ModelType) => void
  setAvailableModels: (models: ModelsListResponse) => void
  setModelsLoading: (loading: boolean) => void
  setModelsError: (error: string | null) => void

  setCurrentImage: (image: string | null, size?: { width: number; height: number }) => void
  setDetectionResult: (result: DetectionResult | null) => void
  setDetecting: (detecting: boolean) => void
  setDetectionError: (error: string | null) => void

  setShowHistory: (show: boolean) => void
  setActiveTab: (tab: 'video' | 'map') => void

  // Reset methods
  resetDetection: () => void
  resetState: () => void
}

const initialState = {
  availableModels: null,
  selectedModel: 'rubbish_area' as ModelType,
  modelsLoading: false,
  modelsError: null,

  currentImage: null,
  currentImageSize: null,
  detectionResult: null,
  detecting: false,
  detectionError: null,

  showHistory: false,
  activeTab: 'video' as const,
}

export const useDetectionStore = create<DetectionState>((set) => ({
  ...initialState,

  setSelectedModel: (model) => set({ selectedModel: model }),
  setAvailableModels: (models) => set({ availableModels: models }),
  setModelsLoading: (loading) => set({ modelsLoading: loading }),
  setModelsError: (error) => set({ modelsError: error }),

  setCurrentImage: (image, size) =>
    set({ currentImage: image, currentImageSize: size || null }),
  setDetectionResult: (result) => set({ detectionResult: result }),
  setDetecting: (detecting) => set({ detecting }),
  setDetectionError: (error) => set({ detectionError: error }),

  setShowHistory: (show) => set({ showHistory: show }),
  setActiveTab: (tab) => set({ activeTab: tab }),

  resetDetection: () =>
    set({
      currentImage: null,
      currentImageSize: null,
      detectionResult: null,
      detecting: false,
      detectionError: null,
    }),

  resetState: () => set(initialState),
}))
