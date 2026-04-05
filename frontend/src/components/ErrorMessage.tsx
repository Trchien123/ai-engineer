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
    <div className="error-message" role="alert">
      <span className="error-icon" aria-hidden="true">⚠</span>
      <span className="error-text">{error}</span>
      {onDismiss && (
        <button
          className="error-close"
          onClick={onDismiss}
          aria-label="Dismiss error"
        >
          ✕
        </button>
      )}
    </div>
  )
}
