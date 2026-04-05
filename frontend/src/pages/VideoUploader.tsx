/**
 * Media Uploader Component
 * Handles image and video file selection with drag-and-drop support.
 */
import React, { useRef, useState } from 'react'
import './VideoUploader.css'

interface Props {
  onVideoSelected: (file: File) => void
  disabled?: boolean
}

const ALLOWED_FORMATS = [
  'image/jpeg', 'image/png', 'image/bmp', 'image/webp',
  'video/mp4', 'video/avi', 'video/quicktime', 'video/webm', 'video/x-matroska',
]
const MAX_FILE_SIZE = 500 * 1024 * 1024 // 500 MB

export const VideoUploader: React.FC<Props> = ({ onVideoSelected, disabled = false }) => {
  const fileInputRef = useRef<HTMLInputElement>(null)
  const [isDragging, setIsDragging] = useState(false)

  const handleFileSelect = (file: File) => {
    if (!ALLOWED_FORMATS.includes(file.type)) {
      alert('Unsupported format. Accepted: JPG, PNG, BMP, WebP, MP4, AVI, MOV, WebM')
      return
    }
    if (file.size > MAX_FILE_SIZE) {
      alert(`File too large (${(file.size / 1024 / 1024).toFixed(1)} MB). Max 500 MB.`)
      return
    }
    onVideoSelected(file)
  }

  const handleInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    if (file) handleFileSelect(file)
  }

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault()
    e.stopPropagation()
    if (!disabled) setIsDragging(true)
  }

  const handleDragLeave = (e: React.DragEvent) => {
    e.preventDefault()
    setIsDragging(false)
  }

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault()
    e.stopPropagation()
    setIsDragging(false)
    if (disabled) return
    const file = e.dataTransfer.files?.[0]
    if (file) handleFileSelect(file)
  }

  const zoneClasses = [
    'upload-area',
    isDragging ? 'drag-active' : '',
    disabled ? 'upload-area--disabled' : '',
  ].filter(Boolean).join(' ')

  return (
    <div className="video-uploader">
      <div
        className={zoneClasses}
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        onDrop={handleDrop}
        onClick={() => !disabled && fileInputRef.current?.click()}
        role="button"
        tabIndex={disabled ? -1 : 0}
        aria-label="Upload file — click or drag and drop"
        onKeyDown={(e) => e.key === 'Enter' && !disabled && fileInputRef.current?.click()}
      >
        <div className="upload-icon" aria-hidden="true">
          <svg viewBox="0 0 24 24" fill="none" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
            <path d="M4 16v2a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2v-2" />
            <polyline points="16 10 12 6 8 10" />
            <line x1="12" y1="6" x2="12" y2="16" />
          </svg>
        </div>
        <p className="upload-title">
          {isDragging ? 'Release to upload' : 'Drop a file here'}
        </p>
        <p className="upload-subtitle">Images and videos supported</p>

        <input
          ref={fileInputRef}
          type="file"
          accept="image/*,video/*"
          onChange={handleInputChange}
          disabled={disabled}
          style={{ display: 'none' }}
          aria-hidden="true"
          tabIndex={-1}
        />

        <button
          type="button"
          className="upload-cta"
          disabled={disabled}
          onClick={(e) => { e.stopPropagation(); fileInputRef.current?.click() }}
        >
          Browse files
        </button>

        <p className="upload-hint">or drag &amp; drop</p>
      </div>

      <div className="upload-meta" aria-label="File requirements">
        <span className="upload-meta-item">
          <strong>Formats:</strong> JPG, PNG, BMP, WebP, MP4, AVI, MOV, WebM
        </span>
        <span className="upload-meta-item">
          <strong>Max size:</strong> 500 MB
        </span>
      </div>
    </div>
  )
}
