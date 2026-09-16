import './DetectionResult.css';

const CONFIG = {
  AI_GENERATED: {
    label: 'AI-Generated',
    className: 'result-ai',
    icon: (
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
        <path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"/>
        <line x1="12" y1="9" x2="12" y2="13"/>
        <line x1="12" y1="17" x2="12.01" y2="17"/>
      </svg>
    ),
  },
  AUTHENTIC: {
    label: 'Authentic',
    className: 'result-authentic',
    icon: (
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
        <path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"/>
        <polyline points="22 4 12 14.01 9 11.01"/>
      </svg>
    ),
  },
  INCONCLUSIVE: {
    label: 'Inconclusive',
    className: 'result-inconclusive',
    icon: (
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
        <circle cx="12" cy="12" r="10"/>
        <line x1="12" y1="8" x2="12" y2="12"/>
        <line x1="12" y1="16" x2="12.01" y2="16"/>
      </svg>
    ),
  },
};

export default function DetectionResult({ aiResult, hasUsableEvidence, provenance }) {
  // Determine display based on what we have
  if (!aiResult && hasUsableEvidence && provenance?.found) {
    // Provenance-based result
    const isAiDisclosed =
      provenance.aiDisclosure?.toLowerCase().includes('ai');
    return (
      <div className={`detection-result ${isAiDisclosed ? 'result-ai' : 'result-authentic'}`}>
        <div className="dr-icon-wrap" aria-hidden="true">
          {isAiDisclosed ? CONFIG.AI_GENERATED.icon : CONFIG.AUTHENTIC.icon}
        </div>
        <div className="dr-body">
          <span className="dr-label">
            {isAiDisclosed ? 'AI Disclosed' : 'Provenance Verified'}
          </span>
          <p className="dr-summary">
            {isAiDisclosed
              ? 'The image\'s content credentials disclose AI involvement in its creation or editing.'
              : 'Verified content credentials were found. The image has traceable provenance.'}
          </p>
          <span className="dr-basis">Based on content credentials</span>
        </div>
      </div>
    );
  }

  if (!aiResult) return null;

  const cfg = CONFIG[aiResult.classification] ?? CONFIG.INCONCLUSIVE;

  return (
    <div className={`detection-result ${cfg.className}`} role="region" aria-label="Verification result">
      <div className="dr-icon-wrap" aria-hidden="true">{cfg.icon}</div>
      <div className="dr-body">
        <span className="dr-label">{cfg.label}</span>
        {aiResult.classification !== 'INCONCLUSIVE' && (
          <div className="dr-confidence" aria-label={`Confidence: ${aiResult.confidence}%`}>
            <ConfidenceBar value={aiResult.confidence} cls={aiResult.classification} />
            <span className="dr-confidence-pct">{aiResult.confidence}%</span>
            <span className="dr-confidence-word">confidence</span>
          </div>
        )}
        <p className="dr-summary">{aiResult.summary}</p>
        <span className="dr-basis">Based on visual AI analysis</span>
      </div>
    </div>
  );
}

function ConfidenceBar({ value, cls }) {
  const color =
    cls === 'AI_GENERATED' ? 'var(--color-error)' :
    cls === 'AUTHENTIC'    ? 'var(--color-success)' :
                             'var(--color-warning)';
  return (
    <div className="confidence-bar-wrap" aria-hidden="true">
      <div
        className="confidence-bar-fill"
        style={{ width: `${value}%`, background: color }}
      />
    </div>
  );
}
