import { useEffect } from 'react';
import { addSystemLogApi } from '../api/verifyApi';
import './FileFormatErrorModal.css';

export default function FileFormatErrorModal({ isOpen, fileInfo, onClose, onRetry }) {
  useEffect(() => {
    if (isOpen && fileInfo) {
      // Log the <<extend>> format error event to system audit logs
      addSystemLogApi(
        'WARN',
        'User',
        'FORMAT_ERROR',
        `Show File Format Error: Rejected ${fileInfo.name || 'unnamed'} (${fileInfo.type || 'unknown format'})`
      );
    }
  }, [isOpen, fileInfo]);

  if (!isOpen || !fileInfo) return null;

  return (
    <div className="format-error-backdrop" onClick={onClose} role="alertdialog" aria-modal="true">
      <div className="format-error-card" onClick={(e) => e.stopPropagation()}>
        <div className="format-error-header">
          <div className="format-error-icon-wrap" aria-hidden="true">
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"/>
              <line x1="12" y1="9" x2="12" y2="13"/>
              <line x1="12" y1="17" x2="12.01" y2="17"/>
            </svg>
          </div>
          <div className="format-error-title-wrap">
            <span className="format-error-badge">&lt;&lt;extend&gt;&gt; Exception Triggered</span>
            <h2 className="format-error-title">Show File Format Error</h2>
          </div>
        </div>

        <div className="format-error-body">
          <p style={{ fontSize: 'var(--text-sm)', color: 'var(--color-text-muted)' }}>
            The uploaded image cannot be processed by the forensic pipeline because its file format or encoding is not supported.
          </p>

          <div className="format-error-meta">
            <div className="format-error-row">
              <span className="format-error-key">Rejected File:</span>
              <span className="format-error-val">{fileInfo.name}</span>
            </div>
            <div className="format-error-row">
              <span className="format-error-key">Detected Type:</span>
              <span className="format-error-val" style={{ color: 'var(--color-error)' }}>
                {fileInfo.type || 'Unknown MIME / Non-Image'}
              </span>
            </div>
            <div className="format-error-row">
              <span className="format-error-key">File Size:</span>
              <span className="format-error-val">
                {fileInfo.size ? `${(fileInfo.size / (1024 * 1024)).toFixed(2)} MB` : 'N/A'}
              </span>
            </div>
          </div>

          <div className="format-error-accepted">
            <strong>Supported Forensic Formats:</strong>
            <p style={{ marginTop: '4px' }}>
              The Classical Forensics (187 features) and ResNet-18 neural models accept raster image formats containing verifiable pixel grids:
            </p>
            <div className="format-error-pill-list">
              <span className="format-error-pill">✓ JPEG (.jpg, .jpeg)</span>
              <span className="format-error-pill">✓ PNG (.png)</span>
              <span className="format-error-pill">✓ WEBP (.webp)</span>
            </div>
          </div>
        </div>

        <div className="format-error-actions">
          <button type="button" className="btn btn-ghost" onClick={onClose}>
            Dismiss
          </button>
          <button
            type="button"
            className="btn btn-primary"
            onClick={() => {
              onClose();
              if (onRetry) onRetry();
            }}
          >
            Upload Valid Image
          </button>
        </div>
      </div>
    </div>
  );
}

