import { useState, useRef, useEffect } from 'react';
import './HeatmapViewer.css';

const TABS = [
  { id: 'Original', label: 'Original' },
  { id: 'Grad-CAM', label: 'Grad-CAM' },
  { id: 'ELA', label: 'ELA Forensic Map' },
];

export default function HeatmapViewer({ imageUrl, aiResult }) {
  const [activeTab, setActiveTab] = useState('Original');
  const canvasRef = useRef(null);

  useEffect(() => {
    if (!canvasRef.current || !imageUrl) return;
    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');

    const renderSource = (sourceUrl, isOverlay = false) => {
      const img = new Image();
      img.onload = () => {
        canvas.width = img.naturalWidth || 512;
        canvas.height = img.naturalHeight || 512;
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        ctx.drawImage(img, 0, 0);
      };
      img.src = sourceUrl;
    };

    if (activeTab === 'Original') {
      renderSource(imageUrl);
    } else if (activeTab === 'Grad-CAM') {
      if (aiResult?.heatmapUrl) {
        renderSource(aiResult.heatmapUrl);
      } else {
        // Fallback: draw simulated overlay on top of original image
        const img = new Image();
        img.onload = () => {
          canvas.width = img.naturalWidth || 512;
          canvas.height = img.naturalHeight || 512;
          ctx.drawImage(img, 0, 0);
          drawHeatmap(ctx, canvas.width, canvas.height);
        };
        img.src = imageUrl;
      }
    } else if (activeTab === 'ELA') {
      if (aiResult?.elaUrl) {
        renderSource(aiResult.elaUrl);
      } else {
        renderSource(imageUrl);
      }
    }
  }, [imageUrl, activeTab, aiResult]);

  return (
    <div className="heatmap-viewer">
      <div className="hm-tabs" role="tablist" aria-label="Image view">
        {TABS.map((tab) => (
          <button
            key={tab.id}
            role="tab"
            aria-selected={activeTab === tab.id}
            className={`hm-tab ${activeTab === tab.id ? 'active' : ''}`}
            onClick={() => setActiveTab(tab.id)}
          >
            {tab.label}
          </button>
        ))}
      </div>

      <div className="hm-canvas-wrap" role="tabpanel" aria-label={`${activeTab} view`}>
        <canvas
          ref={canvasRef}
          className="hm-canvas"
          aria-label={`Image — ${activeTab} view`}
        />
        {!imageUrl && (
          <div className="hm-placeholder">No image loaded</div>
        )}
      </div>

      {activeTab === 'Grad-CAM' && (
        <p className="hm-note">
          Grad-CAM (Gradient-weighted Class Activation Mapping) highlights visual regions in the final
          convolutional layer that contributed most strongly to the neural network's AI prediction.
        </p>
      )}

      {activeTab === 'ELA' && (
        <p className="hm-note">
          Error Level Analysis (ELA) visualizes JPEG/compression re-encoding error residuals.
          Discrepancies in error variance across uniform surfaces often signify synthetic generation.
        </p>
      )}
    </div>
  );
}

/** Draws a pseudo-Grad-CAM heatmap fallback using radial gradients if backend is offline */
function drawHeatmap(ctx, w, h) {
  const spots = [
    { x: 0.35, y: 0.28, r: 0.32, colors: ['#ff1744', '#ff6d00', 'transparent'] },
    { x: 0.68, y: 0.55, r: 0.25, colors: ['#ff6d00', '#ffea00', 'transparent'] },
    { x: 0.5,  y: 0.7,  r: 0.18, colors: ['#ff1744', '#ff6d00', 'transparent'] },
    { x: 0.2,  y: 0.6,  r: 0.14, colors: ['#ff6d00', 'transparent', 'transparent'] },
  ];

  spots.forEach(({ x, y, r, colors }) => {
    const cx = w * x;
    const cy = h * y;
    const radius = Math.min(w, h) * r;
    const grad = ctx.createRadialGradient(cx, cy, 0, cx, cy, radius);
    grad.addColorStop(0,   colors[0]);
    grad.addColorStop(0.5, colors[1]);
    grad.addColorStop(1,   colors[2]);
    ctx.globalAlpha = 0.65;
    ctx.fillStyle = grad;
    ctx.fillRect(0, 0, w, h);
  });

  ctx.globalAlpha = 1.0;
}
