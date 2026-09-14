"""Fast smoke tests — no model downloads. Run: pytest -q

Covers the logic that must be correct regardless of backend:
  * edge_infer degrades gracefully and reports a well-formed status
  * LLM JSON parsing handles clean JSON, JSON-with-prose, and garbage
  * the WAV loader reads a real 16 kHz mono file
  * the FastAPI status endpoint responds without loading any model
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from edge_infer import has_npu, npu_report
from engine.audio import load_wav
from engine.llm import _parse_notes


def test_npu_report_shape():
    rep = npu_report()
    assert set(rep) >= {"onnxruntime_available", "available_providers", "npu_available", "device"}
    assert isinstance(rep["available_providers"], list)
    assert isinstance(has_npu(), bool)


def test_parse_notes_clean_json():
    raw = '{"summary":"S","decisions":["d1"],"action_items":["[Ana] ship"]}'
    n = _parse_notes(raw)
    assert n.summary == "S"
    assert n.decisions == ["d1"]
    assert n.action_items == ["[Ana] ship"]


def test_parse_notes_json_with_prose():
    raw = 'Here is the result:\n{"summary":"S2","decisions":[],"action_items":[]}\nThanks!'
    n = _parse_notes(raw)
    assert n.summary == "S2"
    assert n.decisions == []


def test_parse_notes_garbage_falls_back_to_summary():
    raw = "no json here, just text"
    n = _parse_notes(raw)
    assert n.summary == raw
    assert n.action_items == []


def test_load_wav_jfk():
    wav = Path(__file__).resolve().parents[1] / "recordings" / "jfk.wav"
    if not wav.exists():
        return  # sample not present in this checkout
    audio, rate = load_wav(str(wav))
    assert rate == 16000
    assert audio.ndim == 1 and audio.dtype.name == "float32"
    assert 10.0 < len(audio) / rate < 12.0  # ~11s clip


def test_status_endpoint_no_model_load():
    from fastapi.testclient import TestClient

    from app.server import app
    client = TestClient(app)
    r = client.get("/api/status")
    assert r.status_code == 200
    body = r.json()
    assert body["offline"] is True
    assert "compute" in body
