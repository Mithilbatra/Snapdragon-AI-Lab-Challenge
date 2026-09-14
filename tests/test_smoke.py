import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from edge_infer import has_npu, npu_report
from engine.audio import load_wav
from engine.llm import parse_notes

SAMPLE = Path(__file__).resolve().parents[1] / "recordings" / "jfk.wav"


def test_npu_report_shape():
    report = npu_report()
    assert {"onnxruntime_available", "available_providers", "npu_available", "device"} <= report.keys()
    assert isinstance(report["available_providers"], list)
    assert isinstance(has_npu(), bool)


def test_parse_clean_json():
    notes = parse_notes('{"summary":"S","decisions":["d1"],"action_items":["Ana: ship"]}')
    assert notes.summary == "S"
    assert notes.decisions == ["d1"]
    assert notes.action_items == ["Ana: ship"]


def test_parse_json_in_prose():
    raw = 'Sure, here you go:\n{"summary":"S2","decisions":[],"action_items":[]}\nHope that helps!'
    assert parse_notes(raw).summary == "S2"


def test_parse_json_in_code_fence():
    raw = '```json\n{"summary":"S3","decisions":["keep"],"action_items":[]}\n```'
    notes = parse_notes(raw)
    assert notes.summary == "S3"
    assert notes.decisions == ["keep"]


def test_parse_nested_braces():
    raw = '{"summary":"uses {braces} inside","decisions":[],"action_items":["a {b} c"]}'
    notes = parse_notes(raw)
    assert notes.summary == "uses {braces} inside"
    assert notes.action_items == ["a {b} c"]


def test_parse_garbage_falls_back():
    notes = parse_notes("no json at all")
    assert notes.summary == "no json at all"
    assert notes.action_items == []


def test_parse_drops_blank_items():
    notes = parse_notes('{"summary":"s","decisions":["", "  "],"action_items":["real"]}')
    assert notes.decisions == []
    assert notes.action_items == ["real"]


def test_load_wav():
    if not SAMPLE.exists():
        return
    audio, rate = load_wav(str(SAMPLE))
    assert rate == 16000
    assert audio.dtype.name == "float32" and audio.ndim == 1
    assert 10 < len(audio) / rate < 12


def test_status_endpoint_skips_model_load():
    from fastapi.testclient import TestClient

    from app.server import app
    body = TestClient(app).get("/api/status").json()
    assert body["offline"] is True
    assert "compute" in body
