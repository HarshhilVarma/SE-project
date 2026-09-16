import './VerificationProgress.css';

const STEPS = [
  { key: 'validating',    label: 'File validated' },
  { key: 'provenance',    label: 'Checking content credentials' },
  { key: 'metadata',      label: 'Inspecting image metadata' },
  { key: 'watermark',     label: 'Checking watermark signals' },
  { key: 'ai_detection',  label: 'Running visual AI analysis' },
];

const ICON_DONE = (
  <svg className="step-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
    <polyline points="20 6 9 17 4 12"/>
  </svg>
);

const ICON_ACTIVE = (
  <span className="step-spinner" aria-hidden="true" />
);

const ICON_PENDING = (
  <span className="step-circle" aria-hidden="true" />
);

export default function VerificationProgress({ steps, filename }) {
  return (
    <div className="vp-wrapper" role="status" aria-live="polite" aria-label="Verification progress">
      <div className="vp-header">
        <div className="vp-file-icon" aria-hidden="true">
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
            <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/>
            <polyline points="14 2 14 8 20 8"/>
            <line x1="16" y1="13" x2="8" y2="13"/>
            <line x1="16" y1="17" x2="8" y2="17"/>
            <polyline points="10 9 9 9 8 9"/>
          </svg>
        </div>
        <div>
          <h2 className="vp-title">Verifying image</h2>
          {filename && <p className="vp-filename" title={filename}>{filename}</p>}
        </div>
      </div>

      <ol className="vp-steps" aria-label="Verification steps">
        {STEPS.map(({ key, label }) => {
          const status = steps[key] ?? 'pending';
          // skip ai_detection if it stays pending (wasn't needed)
          if (key === 'ai_detection' && status === 'pending' && steps.ai_detection === 'pending') {
            // still show it as pending
          }
          return (
            <li key={key} className={`vp-step ${status}`} aria-label={`${label}: ${status}`}>
              <span className="vp-step-icon-wrap">
                {status === 'done'   && ICON_DONE}
                {status === 'active' && ICON_ACTIVE}
                {status === 'pending' && ICON_PENDING}
              </span>
              <span className="vp-step-label">{label}</span>
              {status === 'active' && (
                <span className="vp-step-current-tag">In progress…</span>
              )}
            </li>
          );
        })}
      </ol>
    </div>
  );
}
