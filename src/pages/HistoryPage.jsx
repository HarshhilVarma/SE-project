import { useEffect, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import './HistoryPage.css';

const RESULT_CONFIG = {
  AI_GENERATED: { label: 'AI-Generated', cls: 'hist-ai' },
  AUTHENTIC:    { label: 'Authentic',    cls: 'hist-auth' },
  INCONCLUSIVE: { label: 'Inconclusive', cls: 'hist-inconcl' },
  pending:      { label: 'Pending',      cls: 'hist-pending' },
};

function formatDate(iso) {
  return new Date(iso).toLocaleDateString('en-GB', {
    day: 'numeric', month: 'short', year: 'numeric',
  });
}

export default function HistoryPage() {
  const [history, setHistory] = useState([]);
  const navigate = useNavigate();

  useEffect(() => {
    const stored = JSON.parse(localStorage.getItem('av-history') ?? '[]');
    setHistory(stored);
  }, []);

  function clearHistory() {
    if (window.confirm('Clear all verification history?')) {
      localStorage.removeItem('av-history');
      setHistory([]);
    }
  }

  return (
    <div className="history-page page-container">
      <div className="hist-header">
        <div>
          <h1 className="hist-title">History</h1>
          <p className="hist-subtitle">Your recent image verifications</p>
        </div>
        {history.length > 0 && (
          <button className="btn btn-ghost btn-sm" onClick={clearHistory}>
            Clear history
          </button>
        )}
      </div>

      {history.length === 0 ? (
        <div className="hist-empty">
          <div className="hist-empty-icon" aria-hidden="true">
            <svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
              <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/>
              <polyline points="14 2 14 8 20 8"/>
            </svg>
          </div>
          <p>No verifications yet.</p>
          <button className="btn btn-primary" onClick={() => navigate('/')}>
            Verify an image
          </button>
        </div>
      ) : (
        <div className="hist-table-wrap">
          <table className="hist-table" aria-label="Verification history">
            <thead>
              <tr>
                <th scope="col">Image</th>
                <th scope="col">Result</th>
                <th scope="col">Confidence</th>
                <th scope="col">Date</th>
              </tr>
            </thead>
            <tbody>
              {history.map((entry) => {
                const cfg = RESULT_CONFIG[entry.result] ?? RESULT_CONFIG.INCONCLUSIVE;
                return (
                  <tr key={entry.id} className="hist-row">
                    <td className="hist-filename" title={entry.filename}>
                      <div className="hist-file-cell">
                        <span className="hist-file-icon" aria-hidden="true">
                          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                            <rect x="3" y="3" width="18" height="18" rx="2" ry="2"/>
                            <circle cx="8.5" cy="8.5" r="1.5"/>
                            <polyline points="21 15 16 10 5 21"/>
                          </svg>
                        </span>
                        <span className="hist-fname">{entry.filename}</span>
                      </div>
                    </td>
                    <td>
                      <span className={`hist-badge ${cfg.cls}`}>{cfg.label}</span>
                    </td>
                    <td className="hist-confidence">
                      {entry.confidence != null ? `${entry.confidence}%` : '—'}
                    </td>
                    <td className="hist-date">{formatDate(entry.date)}</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}
