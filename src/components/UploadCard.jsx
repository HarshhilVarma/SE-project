import { useCallback, useState } from 'react';
import { useDropzone } from 'react-dropzone';
import './UploadCard.css';

const ACCEPTED_TYPES = {
  'image/jpeg': ['.jpg', '.jpeg'],
  'image/png': ['.png'],
  'image/webp': ['.webp'],
};

const MAX_SIZE_MB = 20;
const MAX_SIZE_BYTES = MAX_SIZE_MB * 1024 * 1024;

export default function UploadCard({ onFileSelected }) {
  const [validationError, setValidationError] = useState(null);

  const onDrop = useCallback(
    (acceptedFiles, rejectedFiles) => {
      setValidationError(null);

      if (rejectedFiles.length > 0) {
        const error = rejectedFiles[0].errors[0];
        if (error.code === 'file-too-large') {
          setValidationError(`File is too large. Maximum size is ${MAX_SIZE_MB} MB.`);
        } else if (error.code === 'file-invalid-type') {
          setValidationError('Unsupported file format. Accepted: JPG · PNG · WEBP');
        } else {
          setValidationError('Could not process this file. Please try another.');
        }
        return;
      }

      if (acceptedFiles.length > 0) {
        onFileSelected(acceptedFiles[0]);
      }
    },
    [onFileSelected]
  );

  const { getRootProps, getInputProps, isDragActive, isDragReject } = useDropzone({
    onDrop,
    accept: ACCEPTED_TYPES,
    maxSize: MAX_SIZE_BYTES,
    multiple: false,
  });

  return (
    <div className="upload-wrapper">
      <div className="upload-hero">
        <h1 className="upload-title">Verify the origin of an image</h1>
        <p className="upload-subtitle">
          We check provenance, metadata, and watermark information before running AI analysis.
        </p>
      </div>

      <div
        {...getRootProps()}
        className={`upload-dropzone ${isDragActive ? 'drag-active' : ''} ${isDragReject ? 'drag-reject' : ''} ${validationError ? 'has-error' : ''}`}
        role="button"
        aria-label="Upload image — drag and drop or click to browse"
        tabIndex={0}
      >
        <input {...getInputProps()} aria-label="File input" />

        <div className="upload-icon-wrap" aria-hidden="true">
          {isDragActive && !isDragReject ? (
            <svg className="upload-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
              <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/>
              <polyline points="17 8 12 3 7 8"/>
              <line x1="12" y1="3" x2="12" y2="15"/>
            </svg>
          ) : (
            <svg className="upload-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
              <rect x="3" y="3" width="18" height="18" rx="2" ry="2"/>
              <circle cx="8.5" cy="8.5" r="1.5"/>
              <polyline points="21 15 16 10 5 21"/>
            </svg>
          )}
        </div>

        {isDragActive && !isDragReject ? (
          <p className="upload-drop-hint active">Drop image here</p>
        ) : isDragReject ? (
          <p className="upload-drop-hint reject">This file type is not supported</p>
        ) : (
          <>
            <p className="upload-drop-hint">Drop image here</p>
            <span className="upload-or">or</span>
            <button
              type="button"
              className="btn btn-primary btn-lg upload-btn"
              onClick={(e) => e.stopPropagation()}
            >
              Choose an image
            </button>
            <p className="upload-formats">JPG · PNG · WEBP · max {MAX_SIZE_MB} MB</p>
          </>
        )}
      </div>

      {validationError && (
        <div className="upload-error-box" role="alert" aria-live="polite">
          <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
            <circle cx="12" cy="12" r="10"/>
            <line x1="12" y1="8" x2="12" y2="12"/>
            <line x1="12" y1="16" x2="12.01" y2="16"/>
          </svg>
          <span>{validationError}</span>
        </div>
      )}

      <p className="upload-privacy-note">
        Images are processed locally and not stored on any server.
      </p>
    </div>
  );
}
