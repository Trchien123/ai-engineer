/**
 * Loading Spinner Component
 */
import React from 'react'
import './LoadingSpinner.css'

interface Props {
  message?: string
  size?: 'small' | 'medium' | 'large'
}

export const LoadingSpinner: React.FC<Props> = ({ message, size = 'medium' }) => {
  return (
    <div className={`loading-container loading-${size}`}>
      <div className="spinner"></div>
      {message && <p className="loading-message">{message}</p>}
    </div>
  )
}
