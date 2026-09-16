import { useState, useRef, useEffect } from 'react';
import './HeatmapViewer.css';

const TABS = ['Original', 'Heatmap', 'Overlay'];

export default function HeatmapViewer({ imageUrl, aiResult }) {
  const [activeTab, setActiveTab] = useState('Original');
  const canvasRef = useRef(null);

  // Draw mock heatmap on canvas whenever tab changes
  useEffect(() => {
    if (!canvasRef.current || !imageUrl) return;
    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    const img = new Image();
    img.onload = () => {
      canvas.width  = img.naturalWidth;
      canvas.height = img.naturalHeight;

      // Always draw original image
      ctx.drawImage(img, 0, 0);

      if (activeTab === 'Heatmap') {
        // Replace with hot-map gradient overlay (simulated)
        ctx.globalAlpha = 0.0;
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        drawHeatmap(ctx, canvas.width, canvas.height, 1.0);
      } else if (activeTab === 'Overlay') {
        ctx.drawImage(img, 0, 0);
        drawHeatmap(ctx, canvas.width, canvas.height, 0.55);
      }
    };
    img.src = imageUrl;
  }, [imageUrl, activeTab]);

  return (
    <div className="heatmap-viewer">
      <div className="hm-tabs" role="tablist" aria-label="Image view">
        {TABS.map((tab) => (
          <button
            key={tab}
            role="tab"
            aria-selected={activeTab === tab}
            className={`hm-tab ${activeTab === tab ? 'active' : ''}`}
            onClick={() => setActiveTab(tab)}
          >
            {tab}
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

      {activeTab !== 'Original' && (
        <p className="hm-note">
          Highlighted regions contributed to the model's prediction. Intensity indicates relative
          importance — not definitive proof of manipulation.
        </p>
      )}
    </div>
  );
}

/** Draws a pseudo-Grad-CAM heatmap using radial gradients */
function drawHeatmap(ctx, w, h) {
  // Background (cool blue for low activation)
  ctx.fillStyle = '#1a237e';
  ctx.fillRect(0, 0, w, h);

  // Several "hot" spots using radial gradients
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
    ctx.globalAlpha = 0.82;
    ctx.fillStyle = grad;
    ctx.fillRect(0, 0, w, h);
  });

  ctx.globalAlpha = 1.0;
}
