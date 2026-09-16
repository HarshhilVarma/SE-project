import { useState } from 'react';
import './ProvenanceCard.css';

const STATUS_CONFIG = {
  verified: {
    icon: '✓',
    className: 'prov-verified',
    headline: 'Content Credentials found',
  },
  unavailable: {
    icon: '—',
    className: 'prov-unavailable',
    headline: 'No Content Credentials found',
  },
  unverifiable: {
    icon: '!',
    className: 'prov-unverifiable',
    headline: 'Credentials could not be verified',
  },
};

export default function ProvenanceCard({ provenance }) {
  const [expanded, setExpanded] = useState(false);
  if (!provenance) return null;

  const cfg = STATUS_CONFIG[provenance.status] ?? STATUS_CONFIG.unavailable;

  return (
    <div className={`prov-card card ${cfg.className}`}>
      <div className="prov-header">
        <div className="prov-title-row">
          <span className="prov-section-label section-label">Content Provenance</span>
          <span className={`prov-status-badge ${cfg.className}`} aria-label={`Status: ${provenance.status}`}>
            <span aria-hidden="true">{cfg.icon}</span>
            {cfg.headline}
          </span>
        </div>
      </div>

      {provenance.found && provenance.status === 'verified' ? (
        <>
          <dl className="prov-fields">
            {provenance.source && (
              <div className="prov-field">
                <dt>Source</dt>
                <dd>{provenance.source}</dd>
              </div>
            )}
            {provenance.created && (
              <div className="prov-field">
                <dt>Created</dt>
                <dd>{provenance.created}</dd>
              </div>
            )}
            {provenance.editedActions != null && (
              <div className="prov-field">
                <dt>Edits</dt>
                <dd>{provenance.editedActions} recorded action{provenance.editedActions !== 1 ? 's' : ''}</dd>
              </div>
            )}
            {provenance.aiDisclosure && (
              <div className="prov-field">
                <dt>AI Disclosure</dt>
                <dd className="prov-ai-disclosure">{provenance.aiDisclosure}</dd>
              </div>
            )}
          </dl>
          <button
            className="btn btn-ghost btn-sm prov-toggle"
            onClick={() => setExpanded((v) => !v)}
            aria-expanded={expanded}
          >
            {expanded ? 'Hide details' : 'View provenance details'}
          </button>
          {expanded && (
            <div className="prov-detail-box">
              <p className="prov-detail-note">
                Content Credentials are metadata standards (C2PA/IPTC) that record the origin and editing history of media. Their presence indicates an auditable chain of custody but does not guarantee the image is unaltered.
              </p>
            </div>
          )}
        </>
      ) : (
        <p className="prov-not-found-note">
          {provenance.reason ?? 'No verifiable provenance information was found in this file.'}
          {' '}This does not determine whether the image is authentic or AI-generated.
        </p>
      )}
    </div>
  );
}
