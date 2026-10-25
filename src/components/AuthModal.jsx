import { useState } from 'react';
import { useAuth } from '../context/AuthContext';
import './AuthModal.css';

export default function AuthModal() {
  const {
    authModalOpen,
    authModalMode,
    setAuthModalMode,
    closeAuthModal,
    login,
    register,
    authError,
    switchRole,
  } = useAuth();

  const [username, setUsername] = useState('');
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('Password123!');
  const [role, setRole] = useState('User');
  const [loading, setLoading] = useState(false);

  if (!authModalOpen) return null;

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    try {
      if (authModalMode === 'login') {
        await login(email || username, password, role);
      } else {
        await register(username || email.split('@')[0], email, password, role);
      }
    } catch (err) {
      // error handled in context
    } finally {
      setLoading(false);
    }
  };

  const handleDemoLogin = (demoRole, demoEmail, demoUsername) => {
    switchRole(demoRole);
    closeAuthModal();
  };

  return (
    <div className="auth-modal-backdrop" onClick={closeAuthModal} role="dialog" aria-modal="true">
      <div className="auth-modal-card" onClick={(e) => e.stopPropagation()}>
        <div className="auth-modal-header">
          <h2 className="auth-modal-title">
            {authModalMode === 'login' ? 'Account Login' : 'User Registration'}
          </h2>
          <button className="auth-modal-close" onClick={closeAuthModal} aria-label="Close modal">
            ✕
          </button>
        </div>

        {/* Tab switch between Login and Register (<<include>>) */}
        <div className="auth-modal-tabs">
          <button
            type="button"
            className={`auth-tab-btn ${authModalMode === 'login' ? 'active' : ''}`}
            onClick={() => setAuthModalMode('login')}
          >
            Sign In
          </button>
          <button
            type="button"
            className={`auth-tab-btn ${authModalMode === 'register' ? 'active' : ''}`}
            onClick={() => setAuthModalMode('register')}
          >
            Register (included)
          </button>
        </div>

        <form className="auth-form" onSubmit={handleSubmit}>
          {authError && (
            <div className="auth-error-alert" role="alert">
              {authError}
            </div>
          )}

          {authModalMode === 'register' && (
            <div className="auth-input-group">
              <label className="auth-label">Username</label>
              <input
                type="text"
                className="auth-input"
                placeholder="e.g. jsmith"
                value={username}
                onChange={(e) => setUsername(e.target.value)}
                required
              />
            </div>
          )}

          <div className="auth-input-group">
            <label className="auth-label">
              {authModalMode === 'login' ? 'Email or Username' : 'Email Address'}
            </label>
            <input
              type="text"
              className="auth-input"
              placeholder="e.g. user@aiverify.internal"
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              required
            />
          </div>

          <div className="auth-input-group">
            <label className="auth-label">Password</label>
            <input
              type="password"
              className="auth-input"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              required
            />
          </div>

          <div className="auth-input-group">
            <label className="auth-label">System Actor / Role</label>
            <select
              className="auth-select"
              value={role}
              onChange={(e) => setRole(e.target.value)}
            >
              <option value="User">User (Image Verification Analyst)</option>
              <option value="Admin">Admin (System Administrator)</option>
              <option value="Security Service">Security Service (Authentication & Audit)</option>
            </select>
          </div>

          <button type="submit" className="auth-submit-btn" disabled={loading}>
            {loading ? 'Authenticating...' : authModalMode === 'login' ? 'Sign In' : 'Register Account'}
          </button>
        </form>

        {/* Demo Fast Login Switcher for UML Actors */}
        <div className="auth-demo-section">
          <span className="auth-demo-label">⚡ Fast Switcher (UML Actors):</span>
          <div className="auth-demo-btns">
            <button
              type="button"
              className="auth-demo-pill"
              onClick={() => handleDemoLogin('User', 'harshi@aiverify.internal', 'harshi')}
            >
              👤 User
            </button>
            <button
              type="button"
              className="auth-demo-pill"
              onClick={() => handleDemoLogin('Admin', 'admin@aiverify.internal', 'admin')}
            >
              🛡️ Admin
            </button>
            <button
              type="button"
              className="auth-demo-pill"
              onClick={() => handleDemoLogin('Security Service', 'security@aiverify.internal', 'secops')}
            >
              🔒 Security Service
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}
