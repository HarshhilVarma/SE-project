/* ============================================================
   Verification API Client
   Connects to FastAPI backend (/api/verify) with mock fallback
   ============================================================ */

import { runVerification as runMockVerification } from './mockVerify';

const delay = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

/**
 * Health check to inspect backend status and model readiness.
 */
export async function checkBackendHealth() {
  try {
    const res = await fetch('/api/health');
    if (!res.ok) return { online: false };
    const data = await res.json();
    return { online: true, ...data };
  } catch {
    return { online: false };
  }
}

/**
 * Executes full verification against the FastAPI backend.
 * Provides progressive step updates to match the UI pipeline.
 */
export async function runVerification(file, onStep) {
  onStep('validating', null);
  await delay(150);
  onStep('validating_done', null);

  const formData = new FormData();
  formData.append('file', file);

  try {
    // Check if backend is available
    const apiCallPromise = fetch('/api/verify', {
      method: 'POST',
      body: formData,
    });

    // Provide progressive UI animation while detector runs on server
    onStep('provenance', null);
    await delay(250);
    onStep('provenance_done', null);

    onStep('metadata', null);
    await delay(250);
    onStep('metadata_done', null);

    onStep('watermark', null);
    await delay(250);
    onStep('watermark_done', null);

    onStep('ai_detection', null);

    const response = await apiCallPromise;
    if (!response.ok) {
      const errText = await response.text();
      throw new Error(`Server returned ${response.status}: ${errText}`);
    }

    const data = await response.json();
    onStep('ai_detection_done', data.aiResult);
    onStep('done', data);
    return data;
  } catch (err) {
    console.warn('Backend API unavailable or errored; falling back to mock pipeline:', err);
    // Graceful fallback to mock verification
    return await runMockVerification(file, onStep);
  }
}
