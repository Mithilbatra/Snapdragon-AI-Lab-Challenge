"""End-to-end meeting flow: audio -> transcript -> notes -> draft reply. All on-device.

Phase 1 milestone: a headless run that takes a recorded clip and prints a summary + action
items, fully offline. Uses CPU backends on a dev machine; flip to NPU/Genie on the HP PC.

    python -m engine.pipeline recordings/jfk.wav                # CPU dev backends
    python -m engine.pipeline recordings/meeting.wav --reply    # also draft a follow-up email
"""

from __future__ import annotations

import argparse
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
                 language: str = "English", models_dir: str = "models"):
        self.language = language
        self.asr = Whisper(backend=asr_backend, models_dir=models_dir)
        self.llm = LocalLLM(backend=llm_backend)

    def process_recording(self, wav_path: str) -> MeetingResult:
        t0 = time.perf_counter()
        transcript = self.asr.transcribe_file(wav_path, language="en").text
        t1 = time.perf_counter()
        notes = self.llm.summarize(transcript, language=self.language)
        t2 = time.perf_counter()
        return MeetingResult(transcript=transcript, notes=notes, on_npu=self.asr.on_npu,
                             asr_secs=t1 - t0, llm_secs=t2 - t1)

    def process_text(self, transcript: str) -> MeetingResult:
        """Skip ASR — summarize an existing transcript / pasted notes / email body."""
        t0 = time.perf_counter()
        notes = self.llm.summarize(transcript, language=self.language)
        return MeetingResult(transcript=transcript, notes=notes, on_npu=False,
                             asr_secs=0.0, llm_secs=time.perf_counter() - t0)

    def draft_followup(self, notes: MeetingNotes) -> str:
        return self.llm.draft_reply(notes.summary, language=self.language)


def _print(result: MeetingResult, reply: str | None) -> None:
    line = "=" * 68
    print(f"\n{line}\nSnapAssist — meeting result   [on_npu={result.on_npu}]")
    print(f"ASR {result.asr_secs:.1f}s · LLM {result.llm_secs:.1f}s (offline)\n{line}")
    print("\nTRANSCRIPT:\n" + result.transcript)
    print("\nSUMMARY:\n" + (result.notes.summary or "(none)"))
    if result.notes.decisions:
        print("\nDECISIONS:")
        for d in result.notes.decisions:
            print(f"  • {d}")
    if result.notes.action_items:
        print("\nACTION ITEMS:")
        for a in result.notes.action_items:
            print(f"  - {a}")
    if reply is not None:
        print("\nDRAFT FOLLOW-UP EMAIL:\n" + reply)
    print(line)


def main() -> int:
    ap = argparse.ArgumentParser(description="SnapAssist Phase 1 pipeline")
    ap.add_argument("wav", nargs="?", help="path to a 16-bit PCM WAV recording")
    ap.add_argument("--text", help="path to a transcript/notes text file (skips ASR)")
    ap.add_argument("--asr-backend", default="cpu", choices=["cpu", "npu"])
    ap.add_argument("--llm-backend", default="cpu", choices=["cpu", "genie"])
    ap.add_argument("--language", default="English")
    ap.add_argument("--reply", action="store_true", help="also draft a follow-up email")
    args = ap.parse_args()

    if not args.wav and not args.text:
        ap.error("provide a WAV path or --text <file>")

    pipe = MeetingPipeline(asr_backend=args.asr_backend, llm_backend=args.llm_backend,
                           language=args.language)
    if args.text:
        with open(args.text, encoding="utf-8") as f:
            result = pipe.process_text(f.read())
    else:
        result = pipe.process_recording(args.wav)
    reply = pipe.draft_followup(result.notes) if args.reply else None
    _print(result, reply)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
