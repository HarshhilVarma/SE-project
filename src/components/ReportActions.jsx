import './ReportActions.css';

export default function ReportActions({ filename, provenance, metadata, watermark, aiResult }) {
  async function handleDownload() {
    // Dynamically import jsPDF to avoid increasing initial bundle
    const { jsPDF } = await import('jspdf');
    const doc = new jsPDF({ unit: 'pt', format: 'a4' });
    const W = doc.internal.pageSize.getWidth();
    const margin = 48;
    let y = 56;

    // Helper
    const line = (text, x = margin, size = 10, weight = 'normal', color = '#1c1c1e') => {
      doc.setFontSize(size);
      doc.setFont('helvetica', weight);
      doc.setTextColor(color);
      doc.text(text, x, y);
    };
    const nl = (n = 14) => { y += n; };
    const hr = () => {
      doc.setDrawColor('#e3e2de');
      doc.line(margin, y, W - margin, y);
      nl(18);
    };

    // Title
    line('AI IMAGE VERIFICATION REPORT', margin, 20, 'bold', '#1a1a2e');
    nl(28);
    line(`File: ${filename ?? 'Unknown'}`, margin, 10, 'normal', '#6b6b70');
    nl(14);
    line(`Generated: ${new Date().toLocaleString()}`, margin, 10, 'normal', '#6b6b70');
    nl(22);
    hr();

    // Classification
    line('RESULT', margin, 13, 'bold');
    nl(18);
    if (aiResult) {
      const cls = { AI_GENERATED: 'AI-Generated', AUTHENTIC: 'Authentic', INCONCLUSIVE: 'Inconclusive' }[aiResult.classification] ?? aiResult.classification;
      line(cls, margin, 18, 'bold', aiResult.classification === 'AI_GENERATED' ? '#dc2626' : aiResult.classification === 'AUTHENTIC' ? '#16a34a' : '#b45309');
      nl(20);
      if (aiResult.classification !== 'INCONCLUSIVE') {
        line(`Confidence: ${aiResult.confidence}%`, margin, 11);
        nl(16);
      }
      const summaryLines = doc.splitTextToSize(aiResult.summary, W - margin * 2);
      summaryLines.forEach((l) => { line(l, margin, 10, 'normal', '#6b6b70'); nl(14); });
    } else if (provenance?.found) {
      line(provenance.aiDisclosure ? 'AI Disclosed' : 'Provenance Verified', margin, 18, 'bold', '#16a34a');
      nl(20);
      line('Based on verified Content Credentials.', margin, 10, 'normal', '#6b6b70');
      nl(16);
    }
    nl(6);
    hr();

    // Provenance
    line('PROVENANCE', margin, 13, 'bold');
    nl(18);
    line(provenance?.found && provenance?.status === 'verified' ? 'Content Credentials: Verified' : 'No verified provenance found', margin, 10);
    if (provenance?.aiDisclosure) { nl(14); line(`AI Disclosure: ${provenance.aiDisclosure}`, margin, 10); }
    nl(22);
    hr();

    // Metadata
    line('METADATA', margin, 13, 'bold');
    nl(18);
    if (metadata?.found) {
      metadata.fields?.forEach(({ key, value }) => {
        line(`${key}: ${value}`, margin, 10);
        nl(14);
      });
    } else {
      line('No metadata found.', margin, 10, 'normal', '#6b6b70');
      nl(14);
    }
    nl(8);
    hr();

    // Watermark
    line('WATERMARK', margin, 13, 'bold');
    nl(18);
    const wmText = watermark?.status === 'found'
      ? `Signal detected — ${watermark.signal ?? 'Unknown'}`
      : watermark?.status === 'undetermined' ? 'Undetermined' : 'No signal detected';
    line(wmText, margin, 10);
    nl(22);
    hr();

    // Disclaimer
    line('IMPORTANT LIMITATIONS', margin, 11, 'bold');
    nl(16);
    const disclaimer =
      'This report summarises automated analysis results. No automated system can guarantee 100% accuracy in detecting AI-generated content. Results should be treated as one input among multiple forms of evidence. Absence of metadata or watermarks does not confirm that an image is AI-generated or authentic.';
    const disclaimerLines = doc.splitTextToSize(disclaimer, W - margin * 2);
    disclaimerLines.forEach((l) => { line(l, margin, 9, 'normal', '#6b6b70'); nl(13); });

    doc.save(`ai-verify-report-${Date.now()}.pdf`);
  }

  return (
    <div className="report-actions">
      <button
        id="download-report-btn"
        className="btn btn-primary btn-lg report-btn"
        onClick={handleDownload}
        aria-label="Download verification report as PDF"
      >
        <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
          <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/>
          <polyline points="7 10 12 15 17 10"/>
          <line x1="12" y1="15" x2="12" y2="3"/>
        </svg>
        Download Report
      </button>
      <p className="report-note">Downloads a PDF summary of this verification.</p>
    </div>
  );
}
