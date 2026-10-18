import { useEffect, useState } from 'react';
import { NavLink } from 'react-router-dom';
import { checkBackendHealth } from '../api/verifyApi';
import './Header.css';

export default function Header() {
  const [backendStatus, setBackendStatus] = useState(null);

  useEffect(() => {
    checkBackendHealth().then((res) => {
      setBackendStatus(res);
    });
  }, []);

  return (
    <header className="header" role="banner">
      <div className="header-inner page-container">
        <div className="header-left">
          <NavLink to="/" className="header-logo" aria-label="AI Verify — Home">
            <span className="header-logo-icon" aria-hidden="true">
              <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round">
                <path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/>
                <path d="M9 12l2 2 4-4"/>
              </svg>
            </span>
            <span className="header-logo-text">AI Verify</span>
          </NavLink>

          {backendStatus?.online && (
            <span className="header-status-badge online" title={`Device: ${backendStatus.device || 'CPU'}`}>
              <span className="header-status-dot" />
              Detector Online {backendStatus.device ? `(${backendStatus.device})` : ''}
            </span>
          )}
        </div>

        <nav className="header-nav" aria-label="Main navigation">
          <NavLink
            to="/"
            end
            className={({ isActive }) => `header-nav-link ${isActive ? 'active' : ''}`}
          >
            Verify
          </NavLink>
          <NavLink
            to="/history"
            className={({ isActive }) => `header-nav-link ${isActive ? 'active' : ''}`}
          >
            History
          </NavLink>
          <NavLink
            to="/about"
            className={({ isActive }) => `header-nav-link ${isActive ? 'active' : ''}`}
          >
            About
          </NavLink>
        </nav>
      </div>
    </header>
  );
}
