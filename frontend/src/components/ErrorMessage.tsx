/**
 * Error Message Component
 */
import React from 'react'
import './ErrorMessage.css'

interface Props {
  error: string
  onDismiss?: () => void
}

export const ErrorMessage: React.FC<Props> = ({ error, onDismiss }) => {
  return (
    <div className="error-message">
      <div className="error-content">
        <span className="error-icon">⚠️</span>
        <span className="error-text">{error}</span>
      </div>
      {onDismiss && (
        <button className="error-close" onClick={onDismiss}>
          ✕
        </button>
      )}
    </div>
  )
}
