import { useState } from 'react';
import DetectionResult from './DetectionResult';
import HeatmapViewer from './HeatmapViewer';
import EvidenceSummary from './EvidenceSummary';
import ProvenanceCard from './ProvenanceCard';
import MetadataCard from './MetadataCard';
import WatermarkCard from './WatermarkCard';
import AnalysisDrawer from './AnalysisDrawer';
import ReportActions from './ReportActions';
import './ResultScreen.css';

export default function ResultScreen({ results, imageUrl, filename, onReset }) {
  const [drawerOpen, setDrawerOpen] = useState(false);
  const { provenance, metadata, watermark, aiResult, hasUsableEvidence } = results;

  return (
    <div className="result-screen page-container">
      {/* ── Top action bar ── */}
      <div className="result-topbar">
        <div className="result-topbar-left">
          <button
            className="btn btn-ghost btn-sm"
            onClick={onReset}
            aria-label="Verify another image"
          >
            ← Verify another image
          </button>
        </div>
        <button
          className="btn btn-ghost btn-sm"
          onClick={() => setDrawerOpen(true)}
          aria-label="Open full analysis details"
        >
          View full details
        </button>
      </div>

      {/* ── Main layout ── */}
      <div className="result-layout">

        {/* LEFT: Image + heatmap */}
        <aside className="result-image-col">
          <div className="result-image-sticky">
            <div className="result-img-label section-label">
              {filename ?? 'Uploaded image'}
            </div>
            <HeatmapViewer
              imageUrl={imageUrl}
              aiResult={aiResult}
            />
          </div>
        </aside>

        {/* RIGHT: Results */}
        <main className="result-detail-col">

          {/* Detection result card */}
          <DetectionResult
            aiResult={aiResult}
            hasUsableEvidence={hasUsableEvidence}
            provenance={provenance}
          />

          {/* Evidence summary */}
          <EvidenceSummary
            provenance={provenance}
            metadata={metadata}
            watermark={watermark}
            aiResult={aiResult}
          />

          <div className="result-divider" />

          {/* Individual evidence cards */}
          <h2 className="result-cards-heading">Evidence Details</h2>
          <ProvenanceCard provenance={provenance} />
          <MetadataCard metadata={metadata} />
          <WatermarkCard watermark={watermark} />

          <div className="result-divider" />

          {/* Report download */}
          <ReportActions
            filename={filename}
            provenance={provenance}
            metadata={metadata}
            watermark={watermark}
            aiResult={aiResult}
          />

          {/* Disclaimer */}
          <p className="result-disclaimer">
            Automated analysis is not a substitute for expert judgment. Results should be considered
            alongside other evidence. Absence of provenance or watermarks does not confirm AI generation.
          </p>
        </main>
      </div>

      {/* Analysis drawer */}
      <AnalysisDrawer
        open={drawerOpen}
        onClose={() => setDrawerOpen(false)}
        provenance={provenance}
        metadata={metadata}
        watermark={watermark}
        aiResult={aiResult}
      />
    </div>
  );
}
