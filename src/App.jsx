import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import Header from './components/Header';
import VerifyPage from './pages/VerifyPage';
import HistoryPage from './pages/HistoryPage';
import './styles/tokens.css';
import './styles/global.css';
import './App.css';

export default function App() {
  return (
    <Router>
      <div className="app-root">
        <Header />
        <main id="main-content" className="app-main" tabIndex={-1}>
          <Routes>
            <Route path="/" element={<VerifyPage />} />
            <Route path="/history" element={<HistoryPage />} />
          </Routes>
        </main>
      </div>
    </Router>
  );
}
