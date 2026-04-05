/**
 * Model Selector Component
 * Allows user to choose which detection model to use
 */
import React from 'react'
import { useDetectionStore } from '../hooks/useDetectionState'
import { ModelInfo } from '../types/detection'
import './ModelSelector.css'

interface Props {
  models: ModelInfo[]
  disabled?: boolean
  compact?: boolean
}

export const ModelSelector: React.FC<Props> = ({ models, disabled = false, compact = false }) => {
  const selectedModel = useDetectionStore((state) => state.selectedModel)
  const setSelectedModel = useDetectionStore((state) => state.setSelectedModel)

  return (
    <div className={`model-selector${compact ? ' model-selector--compact' : ''}`}>
      {!compact && <label htmlFor="model-select">Detection Model:</label>}
      <select
        id="model-select"
        value={selectedModel}
        onChange={(e) => setSelectedModel(e.target.value as any)}
        disabled={disabled}
        className="model-select"
        aria-label="Detection model"
      >
        {models.map((model) => (
          <option key={model.name} value={model.name} disabled={!model.loaded}>
            {model.description} {!model.loaded ? '(Not loaded)' : ''}
          </option>
        ))}
      </select>
      {!compact && (
        <div className="model-info">
          {models.find((m) => m.name === selectedModel)?.description}
        </div>
      )}
    </div>
  )
}
