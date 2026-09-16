import { useState, useEffect, useRef } from 'react';
import { runVerification } from '../api/mockVerify';

/**
 * Custom hook that drives the full verification pipeline.
 * Returns current stage, per-step data, and a start function.
 */
export function useVerification() {
  const [stage, setStage] = useState('idle'); // idle | running | done | error
  const [steps, setSteps] = useState({
    validating: 'pending',    // pending | active | done
    provenance: 'pending',
    metadata: 'pending',
    watermark: 'pending',
    ai_detection: 'pending',
  });
  const [results, setResults] = useState(null);
  const abortRef = useRef(false);

  function updateStep(step, status) {
    setSteps((prev) => ({ ...prev, [step]: status }));
  }

  async function startVerification(file) {
    abortRef.current = false;
    setStage('running');
    setResults(null);
    setSteps({
      validating: 'pending',
      provenance: 'pending',
      metadata: 'pending',
      watermark: 'pending',
      ai_detection: 'pending',
    });

    try {
      await runVerification(file, (step, data) => {
        if (abortRef.current) return;
        switch (step) {
          case 'validating':      updateStep('validating', 'active'); break;
          case 'validating_done': updateStep('validating', 'done'); break;
          case 'provenance':      updateStep('provenance', 'active'); break;
          case 'provenance_done': updateStep('provenance', 'done'); break;
          case 'metadata':        updateStep('metadata', 'active'); break;
          case 'metadata_done':   updateStep('metadata', 'done'); break;
          case 'watermark':       updateStep('watermark', 'active'); break;
          case 'watermark_done':  updateStep('watermark', 'done'); break;
          case 'ai_detection':    updateStep('ai_detection', 'active'); break;
          case 'ai_detection_done': updateStep('ai_detection', 'done'); break;
          case 'done':
            setResults(data);
            setStage('done');
            break;
          default: break;
        }
      });
    } catch (err) {
      if (!abortRef.current) setStage('error');
    }
  }

  function reset() {
    abortRef.current = true;
    setStage('idle');
    setResults(null);
    setSteps({
      validating: 'pending',
      provenance: 'pending',
      metadata: 'pending',
      watermark: 'pending',
      ai_detection: 'pending',
    });
  }

  return { stage, steps, results, startVerification, reset };
}
