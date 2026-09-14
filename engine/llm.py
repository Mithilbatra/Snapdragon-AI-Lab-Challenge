"""On-device LLM tasks: summaries, action items, draft replies, Q&A.

Two backends behind ONE interface:
  * backend="cpu" — reference via HuggingFace transformers with a small instruct model
    (default Qwen2.5-0.5B-Instruct). Runs on any dev machine so the pipeline is testable today.
  * backend="genie" — Phi-3.5-mini / Llama-3.2-3B on the Hexagon NPU via Qualcomm Genie.
    Wired on the Snapdragon HP PC in Phase 1b.

The prompt templates are the product's real IP — they define SnapAssist's behavior and are
backend-independent.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

SYSTEM = "You are SnapAssist, a concise on-device meeting assistant. You run fully offline."

SUMMARY_PROMPT = """Summarize the meeting transcript below. Respond in {language}.
Base every field ONLY on the transcript's actual content. Do NOT copy words from these
instructions. Each action item should name the responsible person if stated.
Return ONLY valid JSON with keys: "summary" (3 sentences), "decisions" (list of strings),
"action_items" (list of strings). Use empty lists if none.

Transcript:
---
{transcript}
---"""

DRAFT_REPLY_PROMPT = """Write a concise, professional follow-up email (< 150 words) in {language}
based on this meeting summary. Return only the email body.
Summary:
---
{summary}
---"""

QA_PROMPT = """Answer the question using ONLY the meeting notes below. If the answer is not
present, say "Not covered in the notes." Question: {question}
Notes:
---
{context}
---"""


@dataclass
class MeetingNotes:
    summary: str
    decisions: list[str] = field(default_factory=list)
    action_items: list[str] = field(default_factory=list)


class LocalLLM:
    def __init__(self, backend: str = "cpu", model_id: str = "Qwen/Qwen2.5-0.5B-Instruct",
                 model_dir: str = "models/genie"):
        self.backend = backend
        self.model_id = model_id
        self.model_dir = model_dir
        self._tok = None
        self._model = None

    # --- public API -------------------------------------------------------
    def summarize(self, transcript: str, language: str = "English") -> MeetingNotes:
        raw = self._generate(SUMMARY_PROMPT.format(transcript=transcript, language=language),
                             max_tokens=400)
        return _parse_notes(raw)

    def draft_reply(self, summary: str, language: str = "English") -> str:
        return self._generate(DRAFT_REPLY_PROMPT.format(summary=summary, language=language),
                              max_tokens=256).strip()

    def answer(self, question: str, context: str) -> str:
        return self._generate(QA_PROMPT.format(question=question, context=context),
                              max_tokens=256).strip()

    # --- generation dispatch ---------------------------------------------
    def _generate(self, prompt: str, max_tokens: int = 400) -> str:
        if self.backend == "genie":
            return self._generate_genie(prompt, max_tokens)
        return self._generate_cpu(prompt, max_tokens)

    # --- CPU reference backend (transformers) -----------------------------
    def _ensure_cpu(self):
        if self._model is not None:
            return
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self._tok = AutoTokenizer.from_pretrained(self.model_id)
        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_id, torch_dtype=torch.float32
        )
        self._model.eval()

    def _generate_cpu(self, prompt: str, max_tokens: int) -> str:
        import torch
        self._ensure_cpu()
        messages = [{"role": "system", "content": SYSTEM},
                    {"role": "user", "content": prompt}]
        text = self._tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self._tok(text, return_tensors="pt")
        with torch.no_grad():
            out = self._model.generate(
                **inputs, max_new_tokens=max_tokens, do_sample=False,
                pad_token_id=self._tok.eos_token_id,
            )
        gen = out[0][inputs["input_ids"].shape[1]:]
        return self._tok.decode(gen, skip_special_tokens=True)

    # --- Genie backend (NPU) ---------------------------------------------
    def _generate_genie(self, prompt: str, max_tokens: int) -> str:
        # Phase 1b (on device): call Genie's local generate() with the exported context binaries.
        # Stream tokens to the UI so it feels instant.
        raise NotImplementedError(
            "Genie backend: initialize on the Snapdragon device. Use backend='cpu' for development."
        )


def _parse_notes(raw: str) -> MeetingNotes:
    """Parse the model's JSON. Falls back to treating the whole output as the summary."""
    text = raw.strip()
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            data = json.loads(text[start:end + 1])
            return MeetingNotes(
                summary=str(data.get("summary", "")).strip(),
                decisions=[str(d) for d in data.get("decisions", [])],
                action_items=[str(a) for a in data.get("action_items", [])],
            )
        except json.JSONDecodeError:
            pass
    return MeetingNotes(summary=text)
