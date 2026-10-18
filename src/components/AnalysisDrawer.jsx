import './AnalysisDrawer.css';

export default function AnalysisDrawer({ open, onClose, provenance, metadata, watermark, aiResult }) {
  if (!open) return null;

  return (
    <>
      {/* Backdrop */}
      <div
        className="drawer-backdrop"
        onClick={onClose}
        aria-hidden="true"
      />

      {/* Panel */}
      <aside
        className="drawer-panel"
        role="dialog"
        aria-modal="true"
        aria-label="Analysis details"
      >
        <div className="drawer-header">
          <h2 className="drawer-title">Analysis Details</h2>
          <button
            className="drawer-close btn btn-ghost btn-sm"
            onClick={onClose}
            aria-label="Close details panel"
          >
            ✕
          </button>
        </div>

        <div className="drawer-body">

          {/* Provenance */}
          <section className="drawer-section">
            <h3 className="drawer-section-title">Provenance</h3>
            {provenance?.found && provenance?.status === 'verified' ? (
              <dl className="drawer-dl">
                <dt>Content Credentials</dt><dd>Verified</dd>
                {provenance.source      && <><dt>Source</dt><dd>{provenance.source}</dd></>}
                {provenance.created     && <><dt>Created</dt><dd>{provenance.created}</dd></>}
                {provenance.editedActions != null && <><dt>Edits</dt><dd>{provenance.editedActions} recorded</dd></>}
                {provenance.aiDisclosure && <><dt>AI Disclosure</dt><dd>{provenance.aiDisclosure}</dd></>}
              </dl>
            ) : (
              <p className="drawer-note">{provenance?.reason ?? 'No verified provenance found.'}</p>
            )}
          </section>

          <div className="drawer-divider" />

          {/* Metadata */}
          <section className="drawer-section">
            <h3 className="drawer-section-title">Metadata</h3>
            {metadata?.found ? (
              <dl className="drawer-dl">
                {metadata.fields?.map(({ key, value }) => (
                  <><dt key={`dt-${key}`}>{key}</dt><dd key={`dd-${key}`}>{value}</dd></>
                ))}
              </dl>
            ) : (
              <p className="drawer-note">No metadata fields were found.</p>
            )}
          </section>

          <div className="drawer-divider" />

          {/* Watermark */}
          <section className="drawer-section">
            <h3 className="drawer-section-title">Watermark / Signals</h3>
            <dl className="drawer-dl">
              <dt>Status</dt>
              <dd>
                {watermark?.status === 'found'
                  ? `Signal detected — ${watermark.signal ?? 'Unknown'}`
                  : watermark?.status === 'undetermined'
                  ? 'Undetermined'
                  : 'No signal detected'}
              </dd>
              {watermark?.provider && <><dt>Provider</dt><dd>{watermark.provider}</dd></>}
            </dl>
          </section>

          {aiResult && (
            <>
              <div className="drawer-divider" />
              <section className="drawer-section">
                <h3 className="drawer-section-title">AI Analysis</h3>
                <dl className="drawer-dl">
                  <dt>Classification</dt>
                  <dd>{aiResult.classification.replace('_', ' ')}</dd>
                  <dt>Confidence</dt>
                  <dd>{aiResult.confidence}%</dd>
                  <dt>Model Version</dt>
                  <dd>{aiResult.modelVersion || 'aid-fusion-v1'}</dd>
                  {aiResult.branchScores?.classical_probability != null && (
                    <>
                      <dt>Classical Forensics</dt>
                      <dd>{(aiResult.branchScores.classical_probability * 100).toFixed(1)}% AI</dd>
                    </>
                  )}
                  {aiResult.branchScores?.cnn_probability != null && (
                    <>
                      <dt>Neural ResNet-18</dt>
                      <dd>{(aiResult.branchScores.cnn_probability * 100).toFixed(1)}% AI</dd>
                    </>
                  )}
                  {aiResult.branchScores?.fusion_probability != null && (
                    <>
                      <dt>Calibrated Fusion</dt>
                      <dd>{(aiResult.branchScores.fusion_probability * 100).toFixed(1)}% AI</dd>
                    </>
                  )}
                  <dt>Grad-CAM Overlay</dt>
                  <dd>{aiResult.heatmapAvailable ? 'Generated' : 'Not available'}</dd>
                  <dt>ELA Forensic Map</dt>
                  <dd>{aiResult.elaUrl ? 'Generated' : 'Not available'}</dd>
                </dl>
                <details className="drawer-tech-details">
                  <summary>Technical details</summary>
                  <p>
                    <strong>Architecture:</strong> Dual-branch ensemble combining 187 Classical Forensic features
                    (ELA recompression, 2D FFT spectral radial energy & entropy, Jia et al. CVPR 2025 color quantization,
                    Sobel & Laplacian residuals) with a fine-tuned ResNet-18 neural classifier. Predictions are calibrated
                    and fused via Platt scaling.
                  </p>
                </details>
              </section>
            </>
          )}

        </div>
      </aside>
    </>
  );
}
