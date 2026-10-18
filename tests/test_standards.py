"""Tests for aidetector.standards module."""

import io
from PIL import Image, PngImagePlugin
from aidetector.standards import check_metadata, check_provenance, check_watermark


def test_check_metadata_clean_image():
    img = Image.new("RGB", (200, 150), color=(100, 100, 100))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    raw_bytes = buf.getvalue()

    meta = check_metadata(raw_bytes, img, "test.png")
    assert meta["found"] is True
    assert meta["fieldCount"] >= 3
    assert meta["hasAiSignals"] is False


def test_check_metadata_with_ai_prompt():
    img = Image.new("RGB", (200, 150), color=(100, 100, 100))
    png_info = PngImagePlugin.PngInfo()
    png_info.add_text("parameters", "A stunning portrait of a cat, photorealistic. Steps: 20, Sampler: Euler a")

    buf = io.BytesIO()
    img.save(buf, format="PNG", pnginfo=png_info)
    raw_bytes = buf.getvalue()

    # Re-open with PIL to read chunks
    re_img = Image.open(io.BytesIO(raw_bytes))
    meta = check_metadata(raw_bytes, re_img, "ai_cat.png")

    assert meta["found"] is True
    assert meta["hasAiSignals"] is True


def test_check_watermark_detection():
    # SynthID tag in raw bytes
    raw_bytes = b"\xff\xd8\xff\xe0...synthid_watermark_signal_tag...\xff\xd9"
    img = Image.new("RGB", (50, 50))
    wm = check_watermark(raw_bytes, img)

    assert wm["status"] == "found"
    assert "SynthID" in wm["signal"]
    assert wm["provider"] == "Google DeepMind"


def test_check_watermark_none():
    raw_bytes = b"clean_ordinary_unwatermarked_image_bytes"
    img = Image.new("RGB", (50, 50))
    wm = check_watermark(raw_bytes, img)

    assert wm["status"] == "not_found"


def test_check_provenance_c2pa():
    raw_bytes = b"header...urn:c2pa...c2pa.claim...footer"
    img = Image.new("RGB", (50, 50))
    prov = check_provenance(raw_bytes, img)

    assert prov["found"] is True
    assert prov["status"] == "verified"

