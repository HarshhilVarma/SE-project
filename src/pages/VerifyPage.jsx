import { useState, useRef, useEffect, useCallback } from 'react';
import UploadCard from '../components/UploadCard';
import VerificationProgress from '../components/VerificationProgress';
import ResultScreen from '../components/ResultScreen';
import { useVerification } from '../hooks/useVerification';
import './VerifyPage.css';

const SUPPORTED_TYPES = ['image/jpeg', 'image/png', 'image/webp'];
const MAX_BYTES = 20 * 1024 * 1024;

export default function VerifyPage() {
  const { stage, steps, results, startVerification, reset } = useVerification();
  const [file, setFile] = useState(null);
  const [imageUrl, setImageUrl] = useState(null);
  const [validationError, setValidationError] = useState(null);

  // Cleanup object URL on unmount or new file
  const prevUrl = useRef(null);
  useEffect(() => {
    return () => {
      if (prevUrl.current) URL.revokeObjectURL(prevUrl.current);
    };
  }, []);

  const handleFileSelected = useCallback((selectedFile) => {
    setValidationError(null);

    // Validate
    if (!SUPPORTED_TYPES.includes(selectedFile.type)) {
      setValidationError('Unsupported file format. Accepted: JPG · PNG · WEBP');
      return;
    }
    if (selectedFile.size > MAX_BYTES) {
      setValidationError('File is too large. Maximum size is 20 MB.');
      return;
    }

    // Create preview URL
    if (prevUrl.current) URL.revokeObjectURL(prevUrl.current);
    const url = URL.createObjectURL(selectedFile);
    prevUrl.current = url;

    setFile(selectedFile);
    setImageUrl(url);

    // Persist to history (localStorage) — will be updated with result later
    const entry = {
      id: Date.now(),
      filename: selectedFile.name,
      date: new Date().toISOString(),
      result: 'pending',
    };
    const existing = JSON.parse(localStorage.getItem('av-history') ?? '[]');
    localStorage.setItem('av-history', JSON.stringify([entry, ...existing].slice(0, 50)));

    startVerification(selectedFile);
  }, [startVerification]);

  // Save result to history when done
  useEffect(() => {
    if (stage === 'done' && results && file) {
      const existing = JSON.parse(localStorage.getItem('av-history') ?? '[]');
      const updated = existing.map((entry, i) =>
        i === 0
          ? {
              ...entry,
              result: results.aiResult?.classification ?? (results.provenance?.found ? 'AUTHENTIC' : 'INCONCLUSIVE'),
              confidence: results.aiResult?.confidence ?? null,
            }
          : entry
      );
      localStorage.setItem('av-history', JSON.stringify(updated));
    }
  }, [stage, results, file]);

  const handleReset = useCallback(() => {
    reset();
    setFile(null);
    if (prevUrl.current) {
      URL.revokeObjectURL(prevUrl.current);
      prevUrl.current = null;
    }
    setImageUrl(null);
    setValidationError(null);
  }, [reset]);

  return (
    <div className="verify-page">
      {stage === 'idle' && (
        <UploadCard onFileSelected={handleFileSelected} />
      )}

      {(stage === 'running') && (
        <div className="page-container verify-progress-wrap">
          <VerificationProgress steps={steps} filename={file?.name} />
        </div>
      )}

      {stage === 'done' && results && (
        <ResultScreen
          results={results}
          imageUrl={imageUrl}
          filename={file?.name}
          onReset={handleReset}
        />
      )}

      {stage === 'error' && (
        <div className="page-container verify-error">
          <div className="verify-error-card card">
            <h2>Something went wrong</h2>
            <p>The verification could not be completed. Please try again.</p>
            <button className="btn btn-primary" onClick={handleReset}>Try again</button>
          </div>
        </div>
      )}
    </div>
  );
}
