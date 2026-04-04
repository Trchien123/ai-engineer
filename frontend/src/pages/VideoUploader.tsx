/**
 * Video Uploader Component
 * Handles video file selection and validation
 */
import React, { useRef } from 'react'
import './VideoUploader.css'

interface Props {
  onVideoSelected: (file: File) => void
  disabled?: boolean
}

export const VideoUploader: React.FC<Props> = ({ onVideoSelected, disabled = false }) => {
  const fileInputRef = useRef<HTMLInputElement>(null)

  const ALLOWED_FORMATS = ['video/mp4', 'video/avi', 'video/quicktime', 'video/webm']
  const MAX_FILE_SIZE = 500 * 1024 * 1024 // 500MB

  const handleFileSelect = (file: File) => {
    // Validate format
    if (!ALLOWED_FORMATS.includes(file.type)) {
      alert('Unsupported video format. Supported formats: MP4, AVI, MOV, WebM')
      return
    }

    // Validate file size
    if (file.size > MAX_FILE_SIZE) {
      alert(`File too large. Maximum size: 500MB. Your file: ${(file.size / 1024 / 1024).toFixed(2)}MB`)
      return
    }

    onVideoSelected(file)
  }

  const handleInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    if (file) {
      handleFileSelect(file)
    }
  }

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault()
    e.stopPropagation()
  }

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault()
    e.stopPropagation()

    const file = e.dataTransfer.files?.[0]
    if (file) {
      handleFileSelect(file)
    }
  }

  return (
    <div className="video-uploader">
      <div className="upload-area" onDragOver={handleDragOver} onDrop={handleDrop}>
        <div className="upload-icon">🎬</div>
        <h3>Upload Video</h3>
        <p>Drag and drop your video here, or click to select</p>
        <input
          ref={fileInputRef}
          type="file"
          accept="video/*"
          onChange={handleInputChange}
          disabled={disabled}
          style={{ display: 'none' }}
        />
        <button
          onClick={() => fileInputRef.current?.click()}
          disabled={disabled}
          className="upload-button"
        >
          Select Video File
        </button>
      </div>
      <div className="upload-info">
        <p><strong>Supported formats:</strong> MP4, AVI, MOV, WebM</p>
        <p><strong>Maximum size:</strong> 500MB</p>
      </div>
    </div>
  )
}
