import { useState } from 'react';
import './MetadataCard.css';

export default function MetadataCard({ metadata }) {
  const [showAll, setShowAll] = useState(false);
  if (!metadata) return null;

  const primaryFields = metadata.fields?.slice(0, 4) ?? [];
  const extraFields   = metadata.fields?.slice(4) ?? [];
  const displayFields = showAll ? metadata.fields : primaryFields;

  return (
    <div className="meta-card card">
      <span className="section-label">Image Metadata</span>

      {metadata.found ? (
        <>
          <dl className="meta-fields">
            {displayFields.map(({ key, value }) => (
              <div className="meta-field" key={key}>
                <dt>{key}</dt>
                <dd>{value}</dd>
              </div>
            ))}
          </dl>

          {metadata.hasAiSignals && (
            <div className="meta-ai-signal" role="note">
              <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
                <circle cx="12" cy="12" r="10"/>
                <line x1="12" y1="8" x2="12" y2="12"/>
                <line x1="12" y1="16" x2="12.01" y2="16"/>
              </svg>
              Software metadata suggests possible AI-assisted editing. This is evidence, not proof.
            </div>
          )}

          {extraFields.length > 0 && (
            <button
              className="btn btn-ghost btn-sm meta-toggle"
              onClick={() => setShowAll((v) => !v)}
              aria-expanded={showAll}
            >
              {showAll
                ? 'Show less'
                : `Show all ${metadata.fieldCount} fields`}
            </button>
          )}
        </>
      ) : (
        <p className="meta-not-found">No metadata fields were found in this file.</p>
      )}
    </div>
  );
}
