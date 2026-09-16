import { useState } from 'react';
import './EvidenceSummary.css';

function EvidenceRow({ icon, iconClass, label, value, detail, onToggle, expanded }) {
  return (
    <div className="ev-row">
      <button
        className="ev-row-btn"
        onClick={onToggle}
        aria-expanded={expanded}
        aria-label={`${label}: ${value} — ${expanded ? 'collapse' : 'expand'} details`}
      >
        <span className={`ev-icon ${iconClass}`} aria-hidden="true">{icon}</span>
        <span className="ev-label">{label}</span>
        <span className="ev-value">{value}</span>
        <span className="ev-chevron" aria-hidden="true">{expanded ? '▲' : '▼'}</span>
      </button>
      {expanded && detail && (
        <div className="ev-detail" role="region" aria-label={`${label} details`}>
          {detail}
        </div>
      )}
    </div>
  );
}

export default function EvidenceSummary({ provenance, metadata, watermark, aiResult }) {
  const [openRow, setOpenRow] = useState(null);
  const toggle = (key) => setOpenRow((v) => (v === key ? null : key));

  const rows = [];

  // Provenance
  rows.push({
    key: 'provenance',
    icon: provenance?.found && provenance?.status === 'verified' ? '✓' : '—',
    iconClass: provenance?.found && provenance?.status === 'verified' ? 'ev-success' : 'ev-neutral',
    label: 'Provenance',
    value: provenance?.found && provenance?.status === 'verified'
      ? 'Available'
      : provenance?.status === 'unverifiable' ? 'Unverifiable' : 'Not found',
    detail: provenance?.found
      ? `Content Credentials found. ${provenance.aiDisclosure ? `AI Disclosure: ${provenance.aiDisclosure}.` : ''}`
      : 'No verifiable Content Credentials were present in this file.',
  });

  // Metadata
  rows.push({
    key: 'metadata',
    icon: metadata?.found ? '✓' : '—',
    iconClass: metadata?.found ? 'ev-success' : 'ev-neutral',
    label: 'Metadata',
    value: metadata?.found ? `${metadata.fieldCount} fields detected` : 'Not found',
    detail: metadata?.found
      ? `${metadata.fieldCount} metadata fields were extracted from the file. ${metadata.hasAiSignals ? 'Software metadata suggests possible AI-assisted editing.' : ''}`
      : 'No EXIF or file metadata was found.',
  });

  // Watermark
  rows.push({
    key: 'watermark',
    icon: watermark?.status === 'found' ? '✓' : watermark?.status === 'undetermined' ? '?' : '—',
    iconClass: watermark?.status === 'found' ? 'ev-success' : watermark?.status === 'undetermined' ? 'ev-warning' : 'ev-neutral',
    label: 'Watermark',
    value: watermark?.status === 'found'
      ? 'Signal detected'
      : watermark?.status === 'undetermined' ? 'Undetermined' : 'Not detected',
    detail: watermark?.status === 'found'
      ? `Signal: ${watermark.signal ?? 'Unknown'}. Provider: ${watermark.provider ?? 'Unknown'}.`
      : 'No supported watermark or embedded signal was detected. Absence does not indicate AI generation.',
  });

  // AI analysis (only if run)
  if (aiResult) {
    const cls = aiResult.classification;
    rows.push({
      key: 'ai',
      icon: cls === 'AI_GENERATED' ? '!' : cls === 'AUTHENTIC' ? '✓' : '?',
      iconClass: cls === 'AI_GENERATED' ? 'ev-error' : cls === 'AUTHENTIC' ? 'ev-success' : 'ev-warning',
      label: 'AI Analysis',
      value: cls === 'AI_GENERATED'
        ? `${aiResult.confidence}% AI-generated`
        : cls === 'AUTHENTIC' ? `${aiResult.confidence}% authentic` : 'Inconclusive',
      detail: aiResult.summary,
    });
  }

  return (
    <div className="evidence-summary card" role="region" aria-label="Evidence summary">
      <span className="section-label">Evidence Summary</span>
      <div className="ev-list">
        {rows.map((row) => (
          <EvidenceRow
            key={row.key}
            {...row}
            expanded={openRow === row.key}
            onToggle={() => toggle(row.key)}
          />
        ))}
      </div>
    </div>
  );
}
