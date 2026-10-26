"""End-to-end meeting flow used by both the app and the command line.

    python -m engine.pipeline recordings/meeting.wav --reply
    python -m engine.pipeline --text recordings/sample_meeting.txt --reply
"""

from __future__ import annotations

import argparse
import os
import time
from dataclasses import dataclass

from .asr import Whisper
from .llm import LocalLLM, MeetingNotes


@dataclass
class MeetingResult:
    transcript: str
    notes: MeetingNotes
    on_npu: bool
    asr_secs: float
    llm_secs: float


class MeetingPipeline:
    def __init__(self, asr_backend: str = "cpu", llm_backend: str = "cpu",
                 language: str = "English", models_dir: str = "models",
                 asr_model: str = "tiny", llm_model: str = "Qwen/Qwen2.5-1.5B-Instruct"):
        self.language = language
        self.asr = Whisper(backend=asr_backend, model_size=asr_model, models_dir=models_dir)
        self.llm = LocalLLM(backend=llm_backend, model_id=llm_model)

    def process_recording(self, wav_path: str) -> MeetingResult:
        start = time.perf_counter()
        transcript = self.asr.transcribe_file(wav_path).text
        asr_secs = time.perf_counter() - start

        start = time.perf_counter()
        notes = self.llm.summarize(transcript, self.language)
        return MeetingResult(transcript, notes, self.asr.on_npu, asr_secs,
                             time.perf_counter() - start)

    def process_text(self, transcript: str) -> MeetingResult:
        start = time.perf_counter()
        notes = self.llm.summarize(transcript, self.language)
        return MeetingResult(transcript, notes, False, 0.0, time.perf_counter() - start)

    def draft_followup(self, notes: MeetingNotes) -> str:
        return self.llm.draft_reply(notes.summary, self.language)


def _print(result: MeetingResult, reply: str | None) -> None:
    rule = "=" * 68
    print(f"\n{rule}\nSnapAssist  [on_npu={result.on_npu}]  "
          f"ASR {result.asr_secs:.1f}s · LLM {result.llm_secs:.1f}s\n{rule}")
    print(f"\nTranscript:\n{result.transcript}")
    print(f"\nSummary:\n{result.notes.summary or '(none)'}")
    for label, items in (("Decisions", result.notes.decisions),
                         ("Action items", result.notes.action_items)):
        if items:
            print(f"\n{label}:")
            for item in items:
                print(f"  - {item}")
    if reply is not None:
        print(f"\nDraft follow-up:\n{reply}")
    print(rule)


def main() -> int:
    # Environment variables give the defaults so the CLI and the server behave identically.
    parser = argparse.ArgumentParser(description="SnapAssist meeting pipeline")
    parser.add_argument("wav", nargs="?", help="16-bit PCM WAV recording")
    parser.add_argument("--text", help="transcript/notes file (skips ASR)")
    parser.add_argument("--asr-backend", default=os.getenv("SNAP_ASR_BACKEND", "cpu"),
                        choices=["cpu", "npu"])
    parser.add_argument("--llm-backend", default=os.getenv("SNAP_LLM_BACKEND", "cpu"),
                        choices=["cpu", "genie"])
    parser.add_argument("--asr-model", default=os.getenv("SNAP_ASR_MODEL", "tiny"))
    parser.add_argument("--llm-model",
                        default=os.getenv("SNAP_LLM_MODEL", "Qwen/Qwen2.5-1.5B-Instruct"))
    parser.add_argument("--language", default="English")
    parser.add_argument("--reply", action="store_true", help="also draft a follow-up email")
    args = parser.parse_args()

    if not args.wav and not args.text:
        parser.error("pass a WAV path or --text <file>")

    pipeline = MeetingPipeline(args.asr_backend, args.llm_backend, args.language,
                               asr_model=args.asr_model, llm_model=args.llm_model)
    if args.text:
        with open(args.text, encoding="utf-8") as fh:
            result = pipeline.process_text(fh.read())
    else:
        result = pipeline.process_recording(args.wav)

    _print(result, pipeline.draft_followup(result.notes) if args.reply else None)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
