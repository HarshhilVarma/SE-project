import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import Header from './components/Header';
import VerifyPage from './pages/VerifyPage';
import HistoryPage from './pages/HistoryPage';
import AboutPage from './pages/AboutPage';
import AdminPage from './pages/AdminPage';
import SecurityPage from './pages/SecurityPage';
import AuthModal from './components/AuthModal';
import { AuthProvider } from './context/AuthContext';
import './styles/tokens.css';
import './styles/global.css';
import './App.css';

export default function App() {
  return (
    <AuthProvider>
      <Router>
        <div className="app-root">
          <Header />
          <main id="main-content" className="app-main" tabIndex={-1}>
            <Routes>
              <Route path="/" element={<VerifyPage />} />
              <Route path="/history" element={<HistoryPage />} />
              <Route path="/admin" element={<AdminPage />} />
              <Route path="/security" element={<SecurityPage />} />
              <Route path="/about" element={<AboutPage />} />
            </Routes>
          </main>
          <AuthModal />
        </div>
      </Router>
    </AuthProvider>
  );
}
