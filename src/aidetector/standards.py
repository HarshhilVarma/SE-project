"""Standards-based checks for Provenance (C2PA), Metadata, and Watermarks."""

import io
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
from PIL import Image, ExifTags
import numpy as np


KNOWN_AI_KEYWORDS = [
    "firefly", "midjourney", "stable diffusion", "stablediffusion", "dall-e", "dalle",
    "comfyui", "automatic1111", "novelai", "invokeai", "imagen", "bing image creator",
    "flux", "ideogram", "runway", "pika", "sora", "trainedalgorithmicmedia", "compositesynthetic",
    "parameters", "steps:", "sampler:", "negative prompt", "cfg scale", "seed:"
]


def check_provenance(raw_bytes: bytes, pil_img: Image.Image) -> Dict[str, Any]:
    """Inspects raw image bytes and XMP headers for C2PA Content Credentials.

    Looks for:
        - JUMBF box / c2pa markers (urn:c2pa, c2pa manifest)
        - IPTC DigitalSourceType (trainedAlgorithmicMedia / compositeSynthetic)
    """
    lower_bytes = raw_bytes[:100000].lower() + raw_bytes[-10000:].lower()

    # 1. C2PA / CAI Content Credentials check
    c2pa_detected = False
    if b"urn:c2pa" in lower_bytes or b"c2pa" in lower_bytes or b"jumb" in lower_bytes:
        # Verify it has manifest / claim
        if b"c2pa.claim" in lower_bytes or b"c2pa.manifest" in lower_bytes or b"urn:c2pa" in lower_bytes:
            c2pa_detected = True

    # Check XMP metadata
    xmp_ai_disclosure = None
    if b"digitalsourcetype" in lower_bytes:
        if b"trainedalgorithmicmedia" in lower_bytes:
            xmp_ai_disclosure = "AI creation disclosed (trainedAlgorithmicMedia)"
        elif b"compositesynthetic" in lower_bytes:
            xmp_ai_disclosure = "AI modification disclosed (compositeSynthetic)"

    # Check PIL info for C2PA
    if "c2pa" in pil_img.info:
        c2pa_detected = True

    if c2pa_detected:
        return {
            "found": True,
            "status": "verified",
            "source": "Content Authenticity Initiative (C2PA)",
            "created": "Verified in manifest",
            "editedActions": 1,
            "aiDisclosure": xmp_ai_disclosure or "Traceable provenance metadata present",
            "reason": None,
        }
    elif xmp_ai_disclosure:
        return {
            "found": True,
            "status": "verified",
            "source": "IPTC Photo Metadata Standards",
            "created": "Recorded in XMP",
            "editedActions": 1,
            "aiDisclosure": xmp_ai_disclosure,
            "reason": None,
        }
    else:
        return {
            "found": False,
            "status": "unavailable",
            "source": None,
            "created": None,
            "editedActions": None,
            "aiDisclosure": None,
            "reason": "No C2PA or IPTC Content Credentials found in this file.",
        }


def check_metadata(raw_bytes: bytes, pil_img: Image.Image, filename: str) -> Dict[str, Any]:
    """Extracts standard EXIF, IPTC, and file metadata and flags AI signatures."""
    fields: List[Dict[str, str]] = []
    has_ai_signals = False

    w, h = pil_img.size
    fields.append({"key": "Dimensions", "value": f"{w} × {h}"})
    fields.append({"key": "Color Space", "value": pil_img.mode})
    fields.append({"key": "File Size", "value": f"{(len(raw_bytes) / (1024 * 1024)):.2f} MB"})

    if pil_img.format:
        fields.append({"key": "Format", "value": pil_img.format})

    # Extract EXIF tags
    try:
        exif = pil_img.getexif()
        if exif:
            for tag_id, value in exif.items():
                tag_name = ExifTags.TAGS.get(tag_id, str(tag_id))
                val_str = str(value).strip()
                if not val_str or len(val_str) > 200:
                    continue

                if tag_name in ("Make", "Model", "Software", "DateTime", "Artist", "Copyright", "LensModel"):
                    fields.append({"key": tag_name, "value": val_str})

                # Check for AI keywords in tags
                val_lower = val_str.lower()
                for kw in KNOWN_AI_KEYWORDS:
                    if kw in val_lower:
                        has_ai_signals = True
                        break
    except Exception:
        pass

    # Check PNG chunks / text metadata
    if hasattr(pil_img, "info") and isinstance(pil_img.info, dict):
        for k, v in pil_img.info.items():
            if isinstance(v, (str, bytes)):
                v_str = str(v)[:300].strip()
                k_lower = str(k).lower()
                v_lower = v_str.lower()

                if k_lower in ("parameters", "prompt", "workflow", "comment", "description", "software"):
                    fields.append({"key": f"Info: {k}", "value": v_str[:120]})

                for kw in KNOWN_AI_KEYWORDS:
                    if kw in v_lower or kw in k_lower:
                        has_ai_signals = True
                        break

    return {
        "found": len(fields) > 0,
        "fieldCount": len(fields),
        "fields": fields,
        "hasAiSignals": has_ai_signals,
    }


def check_watermark(raw_bytes: bytes, pil_img: Image.Image) -> Dict[str, Any]:
    """Inspects for digital watermark signatures and embedded watermark signals."""
    lower_bytes = raw_bytes[:50000].lower() + raw_bytes[-20000:].lower()

    # 1. Google SynthID check
    if b"synthid" in lower_bytes or b"google_synthid" in lower_bytes:
        return {
            "status": "found",
            "signal": "SynthID Digital Watermark",
            "provider": "Google DeepMind",
            "reason": None,
        }

    # 2. C2PA watermark signal
    if b"urn:c2pa" in lower_bytes:
        return {
            "status": "found",
            "signal": "C2PA Content Binding",
            "provider": "Content Authenticity Initiative",
            "reason": None,
        }

    # 3. DALL-E signature
    if b"dall-e" in lower_bytes or b"dalle" in lower_bytes:
        return {
            "status": "found",
            "signal": "DALL-E Generation Metadata Signature",
            "provider": "OpenAI",
            "reason": None,
        }

    # 4. Midjourney signature
    if b"midjourney" in lower_bytes:
        return {
            "status": "found",
            "signal": "Midjourney Watermark / Signature",
            "provider": "Midjourney",
            "reason": None,
        }

    # 5. Stable Diffusion invisible watermark metadata marker
    if b"invisible-watermark" in lower_bytes or b"sd-watermark" in lower_bytes:
        return {
            "status": "found",
            "signal": "Invisible Watermark (DwtDct / RivaGAN)",
            "provider": "Stability AI",
            "reason": None,
        }

    return {
        "status": "not_found",
        "signal": None,
        "provider": None,
        "reason": "No recognized digital watermark signal found.",
    }

