/* ============================================================
   Mock Verification API
   Simulates progressive backend calls with realistic delays
   ============================================================ */

/**
 * Generates a pseudo-random number seeded from a string.
 * This ensures the same filename always gives the same "result".
 */
function seededRand(seed) {
  let h = 2166136261;
  for (let i = 0; i < seed.length; i++) {
    h ^= seed.charCodeAt(i);
    h = (h * 16777619) >>> 0;
  }
  return (h % 1000) / 1000; // 0..1
}

function delay(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

/**
 * Step 1: Check content provenance / C2PA
 */
export async function checkProvenance(file) {
  await delay(1200 + Math.random() * 400);
  const r = seededRand(file.name + 'prov');

  if (r < 0.35) {
    return {
      found: true,
      status: 'verified',
      source: 'Available',
      created: 'Available',
      editedActions: Math.floor(r * 10) + 1,
      aiDisclosure: r < 0.18 ? 'AI creation disclosed' : 'AI modification disclosed',
    };
  } else if (r < 0.55) {
    return {
      found: false,
      status: 'unavailable',
      reason: 'No Content Credentials found in this file.',
    };
  } else {
    return {
      found: false,
      status: 'unverifiable',
      reason: 'Content Credentials were present but could not be verified.',
    };
  }
}

/**
 * Step 2: Read EXIF / file metadata
 */
export async function checkMetadata(file) {
  await delay(800 + Math.random() * 300);
  const r = seededRand(file.name + 'meta');

  const cameras = ['Sony α7R V', 'Canon EOS R5', 'Nikon Z9', null, null];
  const softwares = ['Adobe Photoshop 25.0', 'Lightroom 7.2', 'GIMP 2.10', 'Unknown', null];

  const camera = cameras[Math.floor(r * cameras.length)];
  const software = softwares[Math.floor((r * 7) % softwares.length)];

  const fields = [];
  if (camera) fields.push({ key: 'Camera', value: camera });
  if (software) fields.push({ key: 'Software', value: software });
  fields.push({ key: 'Dimensions', value: `${1280 + Math.floor(r * 1920)} × ${720 + Math.floor(r * 1080)}` });
  fields.push({ key: 'Color Space', value: r > 0.5 ? 'sRGB' : 'Adobe RGB' });
  fields.push({ key: 'File Size', value: `${(file.size / 1024 / 1024).toFixed(2)} MB` });

  const now = new Date();
  now.setDate(now.getDate() - Math.floor(r * 30));
  fields.push({
    key: 'Date (file)',
    value: now.toLocaleDateString('en-GB', { day: 'numeric', month: 'short', year: 'numeric' }),
  });

  return {
    found: fields.length > 0,
    fieldCount: fields.length,
    fields,
    hasAiSignals: software?.toLowerCase().includes('photoshop') && r < 0.4,
  };
}

/**
 * Step 3: Check for digital watermarks / embedded signals
 */
export async function checkWatermark(file) {
  await delay(1000 + Math.random() * 500);
  const r = seededRand(file.name + 'wm');

  if (r < 0.2) {
    return {
      status: 'found',
      signal: r < 0.1 ? 'C2PA content binding' : 'Provider watermark signal',
      provider: r < 0.1 ? 'Adobe Content Authenticity Initiative' : 'Proprietary',
    };
  } else if (r < 0.85) {
    return { status: 'not_found' };
  } else {
    return { status: 'undetermined', reason: 'Signal detection was inconclusive.' };
  }
}

/**
 * Step 4: Run AI visual detection (only if provenance/metadata is insufficient)
 */
export async function runAIDetection(file) {
  await delay(2000 + Math.random() * 1000);
  const r = seededRand(file.name + 'ai');

  let classification;
  let confidence;

  if (r < 0.42) {
    classification = 'AI_GENERATED';
    confidence = 0.72 + r * 0.3; // 72–100%
  } else if (r < 0.72) {
    classification = 'AUTHENTIC';
    confidence = 0.65 + r * 0.28;
  } else {
    classification = 'INCONCLUSIVE';
    confidence = 0.4 + r * 0.2; // 40–60%
  }

  confidence = Math.min(confidence, 0.99);

  // Mock heatmap: a descriptive placeholder (in real app this would be base64 from backend)
  return {
    classification,
    confidence: Math.round(confidence * 100),
    heatmapAvailable: classification !== 'AUTHENTIC',
    summary: getSummary(classification, Math.round(confidence * 100)),
  };
}

function getSummary(cls, pct) {
  if (cls === 'AI_GENERATED')
    return `Visual analysis identified patterns associated with AI-generated imagery (${pct}% confidence).`;
  if (cls === 'AUTHENTIC')
    return `Visual analysis found patterns consistent with authentic photographic content (${pct}% confidence).`;
  return `The available evidence does not provide sufficient confidence for classification.`;
}

/**
 * Master flow orchestrator — drives the sequential verification stages.
 * Calls onStep(step, data) after each stage completes.
 *
 * Steps: 'validating' | 'provenance' | 'metadata' | 'watermark' | 'ai_detection' | 'done'
 */
export async function runVerification(file, onStep) {
  onStep('validating', null);
  await delay(400);
  onStep('validating_done', null);

  onStep('provenance', null);
  const provenance = await checkProvenance(file);
  onStep('provenance_done', provenance);

  onStep('metadata', null);
  const metadata = await checkMetadata(file);
  onStep('metadata_done', metadata);

  onStep('watermark', null);
  const watermark = await checkWatermark(file);
  onStep('watermark_done', watermark);

  // Standards-first: only run AI detector if evidence is insufficient
  const hasUsableEvidence =
    (provenance.found && provenance.status === 'verified') ||
    watermark.status === 'found';

  let aiResult = null;
  if (!hasUsableEvidence) {
    onStep('ai_detection', null);
    aiResult = await runAIDetection(file);
    onStep('ai_detection_done', aiResult);
  }

  onStep('done', { provenance, metadata, watermark, aiResult, hasUsableEvidence });
}
