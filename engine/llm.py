"""Meeting summarisation, follow-up drafting and Q&A over the transcript.

CPU backend: a small instruct model via transformers, for development. Genie backend: the
on-device LLM (Phi-3.5-mini / Llama-3.2-3B) through Qualcomm's Genie runtime on the NPU.
Prompts and parsing are shared; only generation differs.
"""

from __future__ import annotations

import json
import re
import subprocess
from dataclasses import dataclass, field

SYSTEM = "You are SnapAssist, a concise meeting assistant running fully offline on the user's PC."

_SUMMARY = """Summarise the meeting transcript below. Reply in {language}.
Base everything only on what the transcript says, and name the responsible person on each action
item when it is stated. Return one JSON object with the keys "summary" (about three sentences),
"decisions" (a list of strings) and "action_items" (a list of strings). Use empty lists where
there is nothing to report, and output only the JSON.

Transcript:
{transcript}"""

_DRAFT = """Write a follow-up email (under 150 words, {language}) from this meeting summary.
Return only the email body.

Summary:
{summary}"""

_QA = """Answer the question using only the meeting notes. If the notes don't cover it, say so.

Question: {question}

Notes:
{context}"""


@dataclass
class MeetingNotes:
    summary: str
    decisions: list[str] = field(default_factory=list)
    action_items: list[str] = field(default_factory=list)


class LocalLLM:
    def __init__(self, backend: str = "cpu", model_id: str = "Qwen/Qwen2.5-0.5B-Instruct",
                 genie_dir: str = "models/genie"):
        if backend not in ("cpu", "genie"):
            raise ValueError(f"unknown backend {backend!r}")
        self.backend = backend
        self.model_id = model_id
        self.genie_dir = genie_dir
        self._tokenizer = None
        self._model = None

    def summarize(self, transcript: str, language: str = "English") -> MeetingNotes:
        raw = self._generate(_SUMMARY.format(transcript=transcript, language=language), 400)
        return parse_notes(raw)

    def draft_reply(self, summary: str, language: str = "English") -> str:
        return self._generate(_DRAFT.format(summary=summary, language=language), 256).strip()

    def answer(self, question: str, context: str) -> str:
        return self._generate(_QA.format(question=question, context=context), 256).strip()

    def _generate(self, prompt: str, max_tokens: int) -> str:
        if self.backend == "genie":
            return self._generate_genie(prompt, max_tokens)
        return self._generate_cpu(prompt, max_tokens)

    # CPU -------------------------------------------------------------------
    def _generate_cpu(self, prompt: str, max_tokens: int) -> str:
        import torch
        if self._model is None:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_id)
            self._model = AutoModelForCausalLM.from_pretrained(self.model_id, dtype=torch.float32)
            self._model.eval()

        chat = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}]
        text = self._tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
        inputs = self._tokenizer(text, return_tensors="pt")
        with torch.no_grad():
            out = self._model.generate(**inputs, max_new_tokens=max_tokens, do_sample=False,
                                       pad_token_id=self._tokenizer.eos_token_id)
        new_tokens = out[0][inputs["input_ids"].shape[1]:]
        return self._tokenizer.decode(new_tokens, skip_special_tokens=True)

    # Genie (NPU) -----------------------------------------------------------
    def _generate_genie(self, prompt: str, max_tokens: int) -> str:
        # Runs the Genie CLI against the bundle produced for the device (see docs/DEVICE_SETUP.md).
        config = f"{self.genie_dir}/genie_config.json"
        full = f"<|system|>\n{SYSTEM}<|end|>\n<|user|>\n{prompt}<|end|>\n<|assistant|>\n"
        proc = subprocess.run(
            ["genie-t2t-run", "-c", config, "-p", full, "--max-length", str(max_tokens)],
            capture_output=True, text=True,
        )
        if proc.returncode != 0:
            raise RuntimeError(f"genie-t2t-run failed: {proc.stderr.strip()}")
        return _clean_genie_output(proc.stdout)


def parse_notes(raw: str) -> MeetingNotes:
    data = _extract_json(raw)
    if data is None:
        return MeetingNotes(summary=raw.strip())
    return MeetingNotes(
        summary=str(data.get("summary", "")).strip(),
        decisions=_string_list(data.get("decisions")),
        action_items=_string_list(data.get("action_items")),
    )


def _string_list(value) -> list[str]:
    """Coerce a list of items to clean strings, tolerating a model that wraps them in dicts."""
    if not isinstance(value, list):
        return []
    out = []
    for item in value:
        if isinstance(item, dict):
            item = item.get("text") or item.get("task") or item.get("sentence") or next(
                (str(v) for v in item.values() if isinstance(v, str)), "")
        text = str(item).strip()
        if text:
            out.append(text)
    return out


def _extract_json(text: str) -> dict | None:
    """Find the first valid JSON object in the model output, ignoring fences and surrounding prose."""
    text = re.sub(r"```(?:json)?|```", "", text)
    decoder = json.JSONDecoder()
    idx = text.find("{")
    while idx != -1:
        try:
            obj, _ = decoder.raw_decode(text, idx)
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            pass
        idx = text.find("{", idx + 1)
    return None


def _clean_genie_output(stdout: str) -> str:
    # Genie echoes a few banner lines around the completion; keep the assistant turn.
    if "<|assistant|>" in stdout:
        stdout = stdout.split("<|assistant|>", 1)[1]
    return stdout.replace("<|end|>", "").strip()
