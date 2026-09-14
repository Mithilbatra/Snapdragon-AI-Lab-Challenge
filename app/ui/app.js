// SnapAssist UI logic — vanilla JS, no external dependencies (must work offline).
const $ = (s) => document.querySelector(s);
const state = { transcript: "", summary: "", actionItems: [], wavBlob: null };

// ---- status / badges -------------------------------------------------------
async function refreshStatus() {
  try {
    const r = await fetch("/api/status");
    const s = await r.json();
    $("#computeText").textContent = s.compute + " · " + (s.llm_backend === "cpu" ? "dev LLM" : "Genie");
    $("#computeChip").classList.toggle("npu", s.compute.includes("NPU"));
  } catch { $("#computeText").textContent = "engine offline"; }
}
function refreshOnline() {
  const on = navigator.onLine;
  $("#offlineText").textContent = on ? "On-device · online" : "On-device · OFFLINE ✓";
  $("#offlineChip").classList.toggle("ok", true);
}
window.addEventListener("online", refreshOnline);
window.addEventListener("offline", refreshOnline);

// ---- tabs ------------------------------------------------------------------
document.querySelectorAll(".tab").forEach((t) => t.addEventListener("click", () => {
  document.querySelectorAll(".tab").forEach((x) => x.classList.remove("active"));
  t.classList.add("active");
  const tab = t.dataset.tab;
  $("#pane-paste").hidden = tab !== "paste";
  $("#pane-upload").hidden = tab !== "upload";
  $("#pane-record").hidden = tab !== "record";
  $("#transcribeBtn").hidden = tab === "paste";
}));

// ---- helpers ---------------------------------------------------------------
function busy(btn, on, label) {
  btn.disabled = on;
  btn.dataset.label = btn.dataset.label || btn.textContent;
  btn.innerHTML = on ? `<span class="spinner"></span> ${label || "Working…"}` : btn.dataset.label;
}
async function postJSON(url, body) {
  const r = await fetch(url, { method: "POST", headers: { "Content-Type": "application/json" },
                               body: JSON.stringify(body) });
  if (!r.ok) throw new Error((await r.json().catch(() => ({}))).detail || r.statusText);
  return r.json();
}

// ---- transcription (upload or recording) -----------------------------------
$("#wavfile").addEventListener("change", (e) => { state.wavBlob = e.target.files[0] || null; });

$("#transcribeBtn").addEventListener("click", async () => {
  if (!state.wavBlob) { alert("Choose or record a WAV first."); return; }
  const btn = $("#transcribeBtn"); busy(btn, true, "Transcribing…");
  try {
    const fd = new FormData();
    fd.append("file", state.wavBlob, "audio.wav");
    const r = await fetch("/api/transcribe", { method: "POST", body: fd });
    const d = await r.json();
    if (!r.ok) throw new Error(d.detail || "transcribe failed");
    $("#transcript").value = d.transcript;
    state.transcript = d.transcript;
    $("#asrLat").textContent = `ASR ${d.asr_secs}s · ${d.audio_secs}s audio · ${d.on_npu ? "NPU" : "CPU"}`;
    document.querySelector('.tab[data-tab="paste"]').click();
  } catch (e) { alert("Transcription error: " + e.message); }
  finally { busy(btn, false); }
});

// ---- summarize -------------------------------------------------------------
$("#summarizeBtn").addEventListener("click", async () => {
  const transcript = $("#transcript").value.trim();
  if (!transcript) { alert("Add a transcript first (paste, upload, or record)."); return; }
  state.transcript = transcript;
  const btn = $("#summarizeBtn"); busy(btn, true, "Summarizing…");
  try {
    const d = await postJSON("/api/summarize", { transcript, language: $("#lang").value });
    state.summary = d.summary; state.actionItems = d.action_items || [];
    renderSummary(d);
    $("#llmLat").textContent = `LLM ${d.llm_secs}s`;
    $("#draftBtn").disabled = false; $("#saveBtn").disabled = false;
  } catch (e) { alert("Summarize error: " + e.message); }
  finally { busy(btn, false); }
});

function renderSummary(d) {
  const box = $("#summaryBox");
  let html = `<p>${escapeHtml(d.summary || "(no summary)")}</p>`;
  if (d.decisions && d.decisions.length) {
    html += `<h2 style="margin-top:14px">Decisions</h2>` +
      d.decisions.map((x) => `<span class="pill">✓ ${escapeHtml(x)}</span>`).join("");
  }
  if (d.action_items && d.action_items.length) {
    html += `<h2 style="margin-top:14px">Action items</h2><ul class="items">` +
      d.action_items.map((x) => `<li>${escapeHtml(x)}</li>`).join("") + `</ul>`;
  }
  box.innerHTML = html;
}

// ---- draft email -----------------------------------------------------------
$("#draftBtn").addEventListener("click", async () => {
  const btn = $("#draftBtn"); busy(btn, true, "Drafting…");
  try {
    const d = await postJSON("/api/draft", { summary: state.summary, language: $("#lang").value });
    $("#emailCard").hidden = false; $("#emailBox").textContent = d.email;
  } catch (e) { alert("Draft error: " + e.message); }
  finally { busy(btn, false); }
});

// ---- ask -------------------------------------------------------------------
$("#askBtn").addEventListener("click", async () => {
  const q = $("#askInput").value.trim();
  const ctx = state.transcript || $("#transcript").value;
  if (!q || !ctx) { alert("Add a transcript and a question."); return; }
  const btn = $("#askBtn"); busy(btn, true, "…");
  try {
    const d = await postJSON("/api/ask", { question: q, context: ctx });
    $("#answerBox").textContent = "→ " + d.answer;
  } catch (e) { alert("Ask error: " + e.message); }
  finally { busy(btn, false); }
});

// ---- save / history --------------------------------------------------------
$("#saveBtn").addEventListener("click", async () => {
  const title = (state.transcript.split(/\s+/).slice(0, 6).join(" ") || "Meeting") + "…";
  await postJSON("/api/save", { title, language: $("#lang").value, transcript: state.transcript,
                                summary: state.summary, action_items: state.actionItems });
  loadHistory();
});
async function loadHistory() {
  try {
    const rows = await (await fetch("/api/history")).json();
    const list = $("#historyList");
    if (!rows.length) { list.innerHTML = '<p class="empty">Nothing saved yet.</p>'; return; }
    list.innerHTML = rows.map((r) => `<div class="hist-item" data-id="${r.id}">
        <div class="t">${escapeHtml((r.title || "Meeting").slice(0, 40))}</div>
        <div class="d">${(r.created_at || "").slice(0, 16).replace("T", " ")} · ${escapeHtml(r.language || "")}</div>
      </div>`).join("");
    list.querySelectorAll(".hist-item").forEach((el, i) => el.addEventListener("click", () => {
      const r = rows[i];
      $("#transcript").value = r.transcript || "";
      state.transcript = r.transcript || "";
      renderSummary({ summary: r.summary, decisions: [], action_items: (r.action_items || "").split("\n").filter(Boolean) });
    }));
  } catch {}
}

// ---- mic recording → 16 kHz mono 16-bit WAV (client-side, no upload of raw audio) ----
let mediaRec, audioCtx, recChunks = [], recStart = 0, recTimer;
$("#recBtn").addEventListener("click", async () => {
  const btn = $("#recBtn");
  if (btn.classList.contains("active")) { stopRecording(); return; }
  try {
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    audioCtx = new (window.AudioContext || window.webkitAudioContext)();
    const src = audioCtx.createMediaStreamSource(stream);
    const proc = audioCtx.createScriptProcessor(4096, 1, 1);
    recChunks = [];
    proc.onaudioprocess = (e) => recChunks.push(new Float32Array(e.inputBuffer.getChannelData(0)));
    src.connect(proc); proc.connect(audioCtx.destination);
    btn.dataset.stream = "1"; window._recStream = stream; window._recProc = proc; window._recSrc = src;
    btn.classList.add("active"); btn.textContent = "■ Stop";
    recStart = Date.now();
    recTimer = setInterval(() => { $("#recTime").textContent = ((Date.now() - recStart) / 1000).toFixed(0) + "s"; }, 250);
  } catch (e) { alert("Mic error: " + e.message); }
});

function stopRecording() {
  const btn = $("#recBtn");
  clearInterval(recTimer);
  window._recProc.disconnect(); window._recSrc.disconnect();
  window._recStream.getTracks().forEach((t) => t.stop());
  const rate = audioCtx.sampleRate;
  audioCtx.close();
  const merged = mergeChunks(recChunks);
  const down = downsample(merged, rate, 16000);
  state.wavBlob = encodeWav(down, 16000);
  btn.classList.remove("active"); btn.textContent = "● Record meeting";
  $("#transcribeBtn").hidden = false;
  $("#recTime").textContent = "recorded ✓ — click Transcribe audio";
}
function mergeChunks(chunks) {
  const len = chunks.reduce((a, c) => a + c.length, 0);
  const out = new Float32Array(len); let o = 0;
  for (const c of chunks) { out.set(c, o); o += c.length; }
  return out;
}
function downsample(buf, src, dst) {
  if (dst >= src) return buf;
  const ratio = src / dst, out = new Float32Array(Math.round(buf.length / ratio));
  for (let i = 0; i < out.length; i++) out[i] = buf[Math.floor(i * ratio)];
  return out;
}
function encodeWav(samples, rate) {
  const buf = new ArrayBuffer(44 + samples.length * 2), view = new DataView(buf);
  const w = (o, s) => { for (let i = 0; i < s.length; i++) view.setUint8(o + i, s.charCodeAt(i)); };
  w(0, "RIFF"); view.setUint32(4, 36 + samples.length * 2, true); w(8, "WAVE"); w(12, "fmt ");
  view.setUint32(16, 16, true); view.setUint16(20, 1, true); view.setUint16(22, 1, true);
  view.setUint32(24, rate, true); view.setUint32(28, rate * 2, true);
  view.setUint16(32, 2, true); view.setUint16(34, 16, true); w(36, "data");
  view.setUint32(40, samples.length * 2, true);
  let o = 44;
  for (let i = 0; i < samples.length; i++, o += 2) {
    const s = Math.max(-1, Math.min(1, samples[i]));
    view.setInt16(o, s < 0 ? s * 0x8000 : s * 0x7fff, true);
  }
  return new Blob([view], { type: "audio/wav" });
}

function escapeHtml(s) { return (s || "").replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c])); }

// init
refreshStatus(); refreshOnline(); loadHistory();
