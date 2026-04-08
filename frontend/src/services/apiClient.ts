/**
 * API client for communication with backend
 */
import axios from 'axios'
import {
  DetectionResult,
  ModelsListResponse,
  DetectionHistoryResponse,
  HealthCheckResponse,
  ModelType,
  ExplainResponse,
  XaiMethod,
} from '../types/detection'

const API_BASE_URL = '/api'

const apiClient = axios.create({
  baseURL: API_BASE_URL,
  timeout: 30000,
  headers: {
    'Content-Type': 'application/json',
  },
})

/**
 * Services for detecting objects in images
 */
export const DetectionService = {
  /**
   * Send image as base64 and run detection
   */
  detectFromBase64: async (
    imageBase64: string,
    modelType: ModelType
  ): Promise<DetectionResult> => {
    try {
      const response = await apiClient.post<DetectionResult>('/detect-base64', {
        image_base64: imageBase64,
        model_type: modelType,
      })
      return response.data
    } catch (error) {
      console.error('Detection error:', error)
      throw error
    }
  },

  /**
   * Send image file and run detection
   */
  detectFromFile: async (
    file: File,
    modelType: ModelType
  ): Promise<DetectionResult> => {
    try {
      const formData = new FormData()
      formData.append('file', file)
      formData.append('model_type', modelType)

      const response = await apiClient.post<DetectionResult>('/detect', formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      })
      return response.data
    } catch (error) {
      console.error('File detection error:', error)
      throw error
    }
  },
}

/**
 * Services for model management
 */
export const ModelService = {
  /**
   * Get list of available models
   */
  listModels: async (): Promise<ModelsListResponse> => {
    try {
      const response = await apiClient.get<ModelsListResponse>('/models')
      return response.data
    } catch (error) {
      console.error('Error fetching models:', error)
      throw error
    }
  },
}

/**
 * Services for detection results history
 */
export const HistoryService = {
  /**
   * Get detection history with pagination
   */
  getHistory: async (
    modelType?: string,
    limit: number = 50,
    offset: number = 0
  ): Promise<DetectionHistoryResponse> => {
    try {
      const response = await apiClient.get<DetectionHistoryResponse>('/results', {
        params: {
          model_type: modelType,
          limit,
          offset,
        },
      })
      return response.data
    } catch (error) {
      console.error('Error fetching history:', error)
      throw error
    }
  },

  /**
   * Get specific detection result
   */
  getResult: async (resultId: number): Promise<DetectionResult> => {
    try {
      const response = await apiClient.get<DetectionResult>(`/results/${resultId}`)
      return response.data
    } catch (error) {
      console.error(`Error fetching result ${resultId}:`, error)
      throw error
    }
  },
}

/**
 * Health check service
 */
export const HealthService = {
  /**
   * Check API health
   */
  check: async (): Promise<HealthCheckResponse> => {
    try {
      const response = await apiClient.get<HealthCheckResponse>('/health')
      return response.data
    } catch (error) {
      console.error('Health check error:', error)
      throw error
    }
  },
}

/**
 * Service for Explainable AI (XAI) visualisations
 */
export const ExplainService = {
  /**
   * Send a cropped sign image blob and get XAI heatmap visualisations back.
   *
   * @param blob   PNG/JPEG blob of the sign crop
   * @param method 'grad_cam' | 'shap' | 'zennit'
   */
  explainFromBlob: async (blob: Blob, method: XaiMethod): Promise<ExplainResponse> => {
    const formData = new FormData()
    formData.append('method', method)
    formData.append('file', blob, 'crop.png')

    const response = await apiClient.post<ExplainResponse>('/explain', formData, {
      headers: { 'Content-Type': 'multipart/form-data' },
      // SHAP can take up to ~30s — use a longer timeout for that method
      timeout: method === 'shap' ? 120_000 : 30_000,
    })
    return response.data
  },
}

/**
 * Service for fetching traffic sign metadata
 */
export const SignInfoService = {
  /**
   * Look up sign metadata by its Descriptions text (the sign label from detection).
   */
  lookup: async (name: string): Promise<Record<string, string>> => {
    const response = await apiClient.get<Record<string, string>>('/sign-info', {
      params: { name },
      timeout: 10_000,
    })
    return response.data
  },
}

export default apiClient
