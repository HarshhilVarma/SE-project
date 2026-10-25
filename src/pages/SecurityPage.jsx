import { useState, useEffect } from 'react';
import {
  fetchSecuritySessionsApi,
  verifySecuritySessionApi,
  revokeSecuritySessionApi,
  fetchSystemLogsApi,
} from '../api/verifyApi';
import './AdminPage.css';

export default function SecurityPage() {
  const [sessions, setSessions] = useState([]);
  const [logs, setLogs] = useState([]);
  const [loading, setLoading] = useState(false);
  const [filter, setFilter] = useState('');

  useEffect(() => {
    loadSecurityData();
  }, []);

  const loadSecurityData = async () => {
    setLoading(true);
    try {
      const sessData = await fetchSecuritySessionsApi();
      setSessions(sessData.sessions || []);

      const logsData = await fetchSystemLogsApi();
      // Filter for security-relevant actions
      const securityLogs = (logsData.logs || []).filter((l) =>
        ['AUTHENTICATE', 'LOGIN', 'LOGOUT', 'REVOKED', 'BLOCKED', 'FORMAT_ERROR', 'USER'].some((kw) =>
          l.action.toUpperCase().includes(kw)
        )
      );
      setLogs(securityLogs);
    } catch (e) {
      console.error('Failed to load security data:', e);
    } finally {
      setLoading(false);
    }
  };

  const handleVerifySession = async (sessionId) => {
    await verifySecuritySessionApi(sessionId);
    await loadSecurityData();
  };

  const handleRevokeSession = async (sessionId) => {
    if (window.confirm('Revoke and invalidate this user authentication session?')) {
      await revokeSecuritySessionApi(sessionId);
      await loadSecurityData();
    }
  };

  const filteredLogs = logs.filter(
    (l) =>
      !filter ||
      l.details.toLowerCase().includes(filter.toLowerCase()) ||
      l.actor.toLowerCase().includes(filter.toLowerCase())
  );

  return (
    <div className="admin-page page-container">
      {/* Header */}
      <div className="admin-header">
        <div>
          <div className="admin-title-row">
            <h1 style={{ fontSize: 'var(--text-2xl)', fontWeight: 'var(--weight-bold)' }}>
              Security Operations Center
            </h1>
            <span className="admin-role-badge" style={{ background: '#0f172a' }}>
              Actor: Security Service
            </span>
          </div>
          <p style={{ fontSize: 'var(--text-sm)', color: 'var(--color-text-muted)' }}>
            UML Use Cases: Authenticate User & View System Security Logs
          </p>
        </div>

        <button className="btn btn-primary btn-sm" onClick={loadSecurityData} disabled={loading}>
          {loading ? 'Refreshing...' : '↻ Refresh Sessions'}
        </button>
      </div>

      {/* ── Section 1: Authenticate User ── */}
      <div className="admin-card">
        <div className="admin-card-header">
          <div>
            <h2 className="admin-card-title">User Authentication & Active Sessions</h2>
            <p style={{ fontSize: 'var(--text-xs)', color: 'var(--color-text-muted)' }}>
              UML Use Case: Authenticate User — Cryptographically validate access tokens and monitor authentication requests.
            </p>
          </div>
          <span className="status-badge active">
            {sessions.filter((s) => s.status === 'valid').length} Active Authenticated Sessions
          </span>
        </div>

        <table className="data-table">
          <thead>
            <tr>
              <th>User</th>
              <th>System Role</th>
              <th>Client IP</th>
              <th>Login Time</th>
              <th>Session Token</th>
              <th>Security State</th>
              <th>Actions</th>
            </tr>
          </thead>
          <tbody>
            {sessions.map((sess) => (
              <tr key={sess.id}>
                <td>
                  <strong>{sess.username}</strong>
                  <div style={{ fontSize: '11px', color: 'var(--color-text-muted)' }}>{sess.email}</div>
                </td>
                <td>{sess.role}</td>
                <td>
                  <code style={{ fontSize: '12px', background: 'var(--color-surface-2)', padding: '2px 6px', borderRadius: '4px' }}>
                    {sess.ipAddress || '127.0.0.1'}
                  </code>
                </td>
                <td style={{ fontSize: '12px' }}>
                  {sess.loginTime ? new Date(sess.loginTime).toLocaleTimeString() : 'Recent'}
                </td>
                <td>
                  <code style={{ fontSize: '11px', fontFamily: 'var(--font-mono, monospace)' }}>
                    {sess.token || 'tok_sec_valid'}
                  </code>
                </td>
                <td>
                  <span className={`status-badge ${sess.status}`}>
                    {sess.verifiedBySecurity ? '✓ Authenticated' : sess.status === 'revoked' ? '✕ Revoked' : '⏳ Pending'}
                  </span>
                </td>
                <td>
                  <div className="table-actions">
                    {!sess.verifiedBySecurity && sess.status !== 'revoked' && (
                      <button
                        className="btn btn-primary btn-sm"
                        style={{ fontSize: '11px', padding: '3px 8px' }}
                        onClick={() => handleVerifySession(sess.id)}
                      >
                        ✓ Authenticate
                      </button>
                    )}
                    {sess.status !== 'revoked' && (
                      <button
                        className="btn btn-ghost btn-sm"
                        style={{ color: 'var(--color-error)', fontSize: '11px' }}
                        onClick={() => handleRevokeSession(sess.id)}
                      >
                        Revoke Token
                      </button>
                    )}
                  </div>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {/* ── Section 2: View System Logs (Security View) ── */}
      <div className="admin-card">
        <div className="admin-card-header">
          <div>
            <h2 className="admin-card-title">Security Audit & Threat Logs</h2>
            <p style={{ fontSize: 'var(--text-xs)', color: 'var(--color-text-muted)' }}>
              UML Use Case: View System Logs — Real-time security events, authentication verifications, and format errors.
            </p>
          </div>
        </div>

        <div className="log-filter-bar">
          <input
            type="text"
            className="log-search-input"
            placeholder="Search security events..."
            value={filter}
            onChange={(e) => setFilter(e.target.value)}
          />
        </div>

        <table className="data-table">
          <thead>
            <tr>
              <th>Timestamp</th>
              <th>Level</th>
              <th>Actor</th>
              <th>Security Action</th>
              <th>Audit Narrative</th>
            </tr>
          </thead>
          <tbody>
            {filteredLogs.map((l) => (
              <tr key={l.id}>
                <td style={{ fontSize: '11px', color: 'var(--color-text-muted)', whiteSpace: 'nowrap' }}>
                  {new Date(l.timestamp).toLocaleTimeString()}
                </td>
                <td>
                  <span className={`log-level-badge ${l.level}`}>{l.level}</span>
                </td>
                <td><strong>{l.actor}</strong></td>
                <td>
                  <code style={{ fontSize: '11px', background: 'var(--color-surface-2)', padding: '2px 4px', borderRadius: '3px' }}>
                    {l.action}
                  </code>
                </td>
                <td>{l.details}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
