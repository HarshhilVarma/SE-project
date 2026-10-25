import { useEffect, useState } from 'react';
import { NavLink } from 'react-router-dom';
import { checkBackendHealth } from '../api/verifyApi';
import { useAuth } from '../context/AuthContext';
import './Header.css';

export default function Header() {
  const [backendStatus, setBackendStatus] = useState(null);
  const { currentUser, isAuthenticated, userRole, switchRole, logout, openLogin } = useAuth();

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
              Detector Online
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

          {/* Admin link for Admin role */}
          {(userRole === 'Admin' || userRole === 'Security Service') && (
            <NavLink
              to="/admin"
              className={({ isActive }) => `header-nav-link ${isActive ? 'active' : ''}`}
            >
              🛡️ Admin
            </NavLink>
          )}

          {/* Security Service link */}
          {(userRole === 'Security Service' || userRole === 'Admin') && (
            <NavLink
              to="/security"
              className={({ isActive }) => `header-nav-link ${isActive ? 'active' : ''}`}
            >
              🔒 Security
            </NavLink>
          )}

          <NavLink
            to="/about"
            className={({ isActive }) => `header-nav-link ${isActive ? 'active' : ''}`}
          >
            About
          </NavLink>
        </nav>

        {/* Right side: UML Actor Switcher & Auth actions */}
        <div className="header-right">
          <div className="role-switcher-wrap" title="Switch UML Actor View">
            <span className="role-switcher-label">Actor:</span>
            <select
              className="role-select"
              value={userRole}
              onChange={(e) => switchRole(e.target.value)}
              aria-label="Select UML Actor Role"
            >
              <option value="User">👤 User</option>
              <option value="Admin">🛡️ Admin</option>
              <option value="Security Service">🔒 Security Service</option>
            </select>
          </div>

          {isAuthenticated ? (
            <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
              <span className="header-user-pill" title={`Logged in as ${currentUser.email}`}>
                <span className="header-user-icon">
                  {currentUser.username ? currentUser.username[0].toUpperCase() : 'U'}
                </span>
                <span>{currentUser.username}</span>
              </span>
              <button
                type="button"
                className="btn btn-ghost btn-sm"
                onClick={logout}
                title="Use Case: Logout"
              >
                Logout
              </button>
            </div>
          ) : (
            <button
              type="button"
              className="btn btn-primary btn-sm"
              onClick={openLogin}
              title="Use Cases: Login & Register (included)"
            >
              Sign In
            </button>
          )}
        </div>
      </div>
    </header>
  );
}
