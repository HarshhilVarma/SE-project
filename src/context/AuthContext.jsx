import { createContext, useContext, useState, useEffect } from 'react';
import { loginApi, registerApi, logoutApi, addSystemLogApi } from '../api/verifyApi';

const AuthContext = createContext(null);

const DEFAULT_USER = {
  id: 'u-user1',
  username: 'harshi',
  email: 'harshi@aiverify.internal',
  role: 'User',
  status: 'active',
  scanCount: 156,
};

export function AuthProvider({ children }) {
  const [currentUser, setCurrentUser] = useState(() => {
    const saved = localStorage.getItem('av-current-user');
    if (saved) {
      try {
        return JSON.parse(saved);
      } catch {}
    }
    return DEFAULT_USER;
  });

  const [authModalOpen, setAuthModalOpen] = useState(false);
  const [authModalMode, setAuthModalMode] = useState('login'); // 'login' | 'register'
  const [authError, setAuthError] = useState(null);

  useEffect(() => {
    if (currentUser) {
      localStorage.setItem('av-current-user', JSON.stringify(currentUser));
    } else {
      localStorage.removeItem('av-current-user');
    }
  }, [currentUser]);

  const login = async (username_or_email, password, role = 'User') => {
    setAuthError(null);
    try {
      const res = await loginApi({ username_or_email, password, role });
      if (res.user) {
        setCurrentUser(res.user);
        setAuthModalOpen(false);
        return res.user;
      }
    } catch (err) {
      setAuthError(err.message || 'Login failed');
      throw err;
    }
  };

  const register = async (username, email, password, role = 'User') => {
    setAuthError(null);
    try {
      const res = await registerApi({ username, email, password, role });
      if (res.user) {
        setCurrentUser(res.user);
        setAuthModalOpen(false);
        return res.user;
      }
    } catch (err) {
      setAuthError(err.message || 'Registration failed');
      throw err;
    }
  };

  const logout = async () => {
    if (currentUser) {
      await logoutApi(currentUser.username, currentUser.role);
    }
    setCurrentUser(null);
  };

  const switchRole = (newRole) => {
    let target = {
      ...currentUser,
      role: newRole,
    };
    if (newRole === 'Admin') {
      target = {
        id: 'u-admin',
        username: 'admin',
        email: 'admin@aiverify.internal',
        role: 'Admin',
        status: 'active',
        scanCount: 42,
      };
    } else if (newRole === 'Security Service') {
      target = {
        id: 'u-sec',
        username: 'secops',
        email: 'security@aiverify.internal',
        role: 'Security Service',
        status: 'active',
        scanCount: 18,
      };
    } else {
      target = {
        id: 'u-user1',
        username: 'harshi',
        email: 'harshi@aiverify.internal',
        role: 'User',
        status: 'active',
        scanCount: 156,
      };
    }
    setCurrentUser(target);
    addSystemLogApi('INFO', newRole, 'ROLE_SWITCH', `Switched active session view to ${newRole}`);
  };

  const openLogin = () => {
    setAuthError(null);
    setAuthModalMode('login');
    setAuthModalOpen(true);
  };

  const openRegister = () => {
    setAuthError(null);
    setAuthModalMode('register');
    setAuthModalOpen(true);
  };

  const closeAuthModal = () => {
    setAuthError(null);
    setAuthModalOpen(false);
  };

  return (
    <AuthContext.Provider
      value={{
        currentUser,
        isAuthenticated: !!currentUser,
        userRole: currentUser?.role || 'Guest',
        login,
        register,
        logout,
        switchRole,
        authModalOpen,
        authModalMode,
        setAuthModalMode,
        authError,
        setAuthError,
        openLogin,
        openRegister,
        closeAuthModal,
      }}
    >
      {children}
    </AuthContext.Provider>
  );
}

export function useAuth() {
  const context = useContext(AuthContext);
  if (!context) {
    throw new Error('useAuth must be used within an AuthProvider');
  }
  return context;
}
