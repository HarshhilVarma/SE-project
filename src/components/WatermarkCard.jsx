import './WatermarkCard.css';

export default function WatermarkCard({ watermark }) {
  if (!watermark) return null;

  return (
    <div className="wm-card card">
      <span className="section-label">Watermark / Embedded Signals</span>

      {watermark.status === 'found' && (
        <div className="wm-body wm-found">
          <div className="wm-status-row">
            <span className="wm-status-icon found" aria-hidden="true">✓</span>
            <span className="wm-status-text">Signal detected</span>
          </div>
          <p className="wm-detail">
            A recognized watermark or provenance signal was detected.
            {watermark.signal && <> Signal: <strong>{watermark.signal}</strong>.</>}
            {watermark.provider && <> Provider: <strong>{watermark.provider}</strong>.</>}
          </p>
        </div>
      )}

      {watermark.status === 'not_found' && (
        <div className="wm-body wm-not-found">
          <div className="wm-status-row">
            <span className="wm-status-icon neutral" aria-hidden="true">—</span>
            <span className="wm-status-text">No recognized signal found</span>
          </div>
          <p className="wm-detail">
            No supported watermark signal was detected. This does not indicate that the image is
            authentic or AI-generated.
          </p>
        </div>
      )}

      {watermark.status === 'undetermined' && (
        <div className="wm-body wm-unknown">
          <div className="wm-status-row">
            <span className="wm-status-icon unknown" aria-hidden="true">?</span>
            <span className="wm-status-text">Unable to determine</span>
          </div>
          <p className="wm-detail">
            {watermark.reason ?? 'The available information was insufficient for watermark verification.'}
          </p>
        </div>
      )}
    </div>
  );
}
