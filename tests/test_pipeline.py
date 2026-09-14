import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from engine.llm import MeetingNotes
from engine.pipeline import MeetingPipeline


class FakeLLM:
    def __init__(self):
        self.seen = None

    def summarize(self, transcript, language="English"):
        self.seen = (transcript, language)
        return MeetingNotes("summary", ["decided"], ["Ana: do it"])

    def draft_reply(self, summary, language="English"):
        return f"email about: {summary}"


def _pipeline_with_fake_llm(language="English"):
    pipeline = MeetingPipeline(language=language)  # constructors don't load models
    pipeline.llm = FakeLLM()
    return pipeline


def test_process_text_runs_llm_only():
    pipeline = _pipeline_with_fake_llm()
    result = pipeline.process_text("Ana: let's ship on Friday.")
    assert result.notes.summary == "summary"
    assert result.notes.action_items == ["Ana: do it"]
    assert result.on_npu is False
    assert result.asr_secs == 0.0 and result.llm_secs >= 0.0


def test_language_is_passed_through():
    pipeline = _pipeline_with_fake_llm(language="Hindi")
    pipeline.process_text("some transcript")
    assert pipeline.llm.seen[1] == "Hindi"


def test_draft_followup_uses_summary():
    pipeline = _pipeline_with_fake_llm()
    result = pipeline.process_text("x")
    assert pipeline.draft_followup(result.notes) == "email about: summary"


def test_unknown_backend_rejected():
    import pytest

    from engine.asr import Whisper
    from engine.llm import LocalLLM
    with pytest.raises(ValueError):
        Whisper(backend="gpu")
    with pytest.raises(ValueError):
        LocalLLM(backend="tpu")
