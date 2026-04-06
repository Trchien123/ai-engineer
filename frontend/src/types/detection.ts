/**
 * TypeScript types for detection system
 */

export interface DetectionBox {
  x_min: number
  y_min: number
  x_max: number
  y_max: number
  label: string
  confidence: number
}

export interface DetectionResult {
  id: number
  model_type: string
  detections: DetectionBox[]
  image_height: number
  image_width: number
  inference_time_ms: number
  created_at: string
}

export interface ModelInfo {
  name: string
  type: string
  description: string
  loaded: boolean
}

export interface ModelsListResponse {
  models: ModelInfo[]
  count: number
}

export interface DetectionHistoryResponse {
  results: DetectionResult[]
  total: number
  limit: number
}

export type ModelType = 'rubbish' | 'traffic_sign'

export interface HealthCheckResponse {
  status: string
  models_loaded: number
  total_models: number
}

export interface CapturedImage {
  dataUrl: string
  width: number
  height: number
}

// ── Explainable AI ────────────────────────────────────────────

export type XaiMethod = 'grad_cam' | 'shap' | 'zennit'

export interface ExplainResult {
  class_name: string
  probability: number
  /** Base64-encoded PNG. Render as: <img src={`data:image/png;base64,${image_base64}`} /> */
  image_base64: string
}

export interface ExplainResponse {
  method: XaiMethod
  results: ExplainResult[]
  inference_time_ms: number
}
