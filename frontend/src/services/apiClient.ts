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

export default apiClient
