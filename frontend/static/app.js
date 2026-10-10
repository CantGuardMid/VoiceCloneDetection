const statusPill = document.getElementById("statusPill");

let enrollRecorder = null;
let analyzeRecorder = null;
let enrollBlob = null;
let analyzeBlob = null;
let enrollTimerInt = null;
let analyzeTimerInt = null;

function fmtTime(sec) {
  const m = Math.floor(sec / 60).toString().padStart(2, "0");
  const s = Math.floor(sec % 60).toString().padStart(2, "0");
  return `${m}:${s}`;
}

function startTimer(el, startedAt) {
  return setInterval(() => {
    el.textContent = fmtTime((Date.now() - startedAt) / 1000);
  }, 200);
}

// ---------------------------------------------------------------- Enrollment
const enrollRecordBtn = document.getElementById("enrollRecordBtn");
const enrollTimer = document.getElementById("enrollTimer");
const enrollFileInput = document.getElementById("enrollFile");
const enrollResult = document.getElementById("enrollResult");
const resetBtn = document.getElementById("resetBtn");

enrollRecordBtn.addEventListener("click", async () => {
  if (!enrollRecorder || !enrollRecorder.recording) {
    try {
      enrollRecorder = new WavRecorder();
      await enrollRecorder.start();
    } catch (err) {
      enrollResult.textContent = "Microphone access denied or unavailable: " + err.message;
      enrollResult.style.color = "var(--red)";
      return;
    }
    enrollRecordBtn.textContent = "■ Stop Recording";
    enrollRecordBtn.classList.add("recording");
    const t0 = Date.now();
    enrollTimerInt = startTimer(enrollTimer, t0);
  } else {
    const { blob, durationSec } = enrollRecorder.stop();
    clearInterval(enrollTimerInt);
    enrollRecordBtn.textContent = "● Record Reference";
    enrollRecordBtn.classList.remove("recording");
    enrollBlob = blob;
    await submitEnroll(blob);
  }
});

enrollFileInput.addEventListener("change", async (e) => {
  const file = e.target.files[0];
  if (!file) return;
  enrollBlob = file;
  await submitEnroll(file);
});

resetBtn.addEventListener("click", async () => {
  await fetch("/api/reset", { method: "POST" });
  statusPill.textContent = "Speaker not enrolled";
  statusPill.classList.remove("enrolled");
  enrollResult.textContent = "";
});

async function submitEnroll(blob) {
  enrollResult.style.color = "var(--teal)";
  enrollResult.textContent = "Extracting voice reference...";
  const form = new FormData();
  form.append("audio", blob, "enroll.wav");
  try {
    const res = await fetch("/api/enroll", { method: "POST", body: form });
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || "enrollment failed");
    enrollResult.style.color = "var(--teal)";
    enrollResult.textContent = `Enrolled ✓  (${data.duration_sec}s clip, detected pitch ≈ ${data.pitch_hz} Hz)`;
    statusPill.textContent = "Speaker enrolled";
    statusPill.classList.add("enrolled");
  } catch (err) {
    enrollResult.style.color = "var(--red)";
    enrollResult.textContent = "Error: " + err.message;
  }
}

// ---------------------------------------------------------------- Analyze
const analyzeRecordBtn = document.getElementById("analyzeRecordBtn");
const analyzeTimer = document.getElementById("analyzeTimer");
const analyzeFileInput = document.getElementById("analyzeFile");
const runAnalysisBtn = document.getElementById("runAnalysisBtn");
const analyzeError = document.getElementById("analyzeError");

analyzeRecordBtn.addEventListener("click", async () => {
  if (!analyzeRecorder || !analyzeRecorder.recording) {
    try {
      analyzeRecorder = new WavRecorder();
      await analyzeRecorder.start();
      startSpeechToText();
    } catch (err) {
      analyzeError.textContent = "Microphone access denied or unavailable: " + err.message;
      return;
    }
    analyzeRecordBtn.textContent = "■ Stop Recording";
    analyzeRecordBtn.classList.add("recording");
    const t0 = Date.now();
    analyzeTimerInt = startTimer(analyzeTimer, t0);
  } else {
    const { blob } = analyzeRecorder.stop();
    stopSpeechToText();
    clearInterval(analyzeTimerInt);
    analyzeRecordBtn.textContent = "● Record Call Audio";
    analyzeRecordBtn.classList.remove("recording");
    analyzeBlob = blob;
    runAnalysisBtn.disabled = false;
    analyzeError.textContent = "";
  }
});

analyzeFileInput.addEventListener("change", (e) => {
  const file = e.target.files[0];
  if (!file) return;
  analyzeBlob = file;
  runAnalysisBtn.disabled = false;
  analyzeError.textContent = "";
});

runAnalysisBtn.addEventListener("click", async () => {
  if (!analyzeBlob) return;
  runAnalysisBtn.disabled = true;
  runAnalysisBtn.textContent = "Analyzing...";
  analyzeError.textContent = "";

  const form = new FormData();
  form.append("audio", analyzeBlob, "call.wav");
  try {
    const res = await fetch("/api/analyze", { method: "POST", body: form });
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || "analysis failed");
    const textData = await autoAnalyzeTranscript();
    renderResults(data, textData);
  } catch (err) {
    analyzeError.textContent = "Error: " + err.message;
  } finally {
    runAnalysisBtn.disabled = false;
    runAnalysisBtn.textContent = "Analyze";
  }
});

// ---------------------------------------------------------------- Speech-to-text
let speechRecognition = null;
let speechTranscript = "";
let speechSupported = false;
let speechRecordingActive = false;
let speechFinalized = false;

const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
if (SpeechRecognition) {
  speechSupported = true;
  speechRecognition = new SpeechRecognition();
  speechRecognition.continuous = true;
  speechRecognition.interimResults = true;
  speechRecognition.lang = "en-IN";
  speechRecognition.onresult = (event) => {
    let finalText = "";
    let interimText = "";
    for (let i = event.resultIndex; i < event.results.length; i++) {
      const text = event.results[i][0].transcript;
      if (event.results[i].isFinal) finalText += text + " ";
      else interimText += text;
    }
    if (finalText) speechTranscript += finalText;
    const box = document.getElementById("transcriptInput");
    if (box) box.value = (speechTranscript + interimText).trim();
  };
  speechRecognition.onend = () => {
    // Chrome can stop recognition by itself. Restart while the audio recorder is active.
    if (speechRecordingActive && !speechFinalized) {
      try { speechRecognition.start(); } catch (_) {}
    }
  };
  speechRecognition.onerror = (event) => {
    if (event.error !== "no-speech" && event.error !== "aborted") {
      const err = document.getElementById("textError");
      if (err) err.textContent = "Speech-to-text: " + event.error;
    }
  };
}

function startSpeechToText() {
  if (!speechRecognition) {
    const err = document.getElementById("textError");
    if (err) err.textContent = "Speech-to-text is not supported by this browser. Use Google Chrome/Edge or type the transcript.";
    return;
  }
  speechTranscript = "";
  speechRecordingActive = true;
  speechFinalized = false;
  const box = document.getElementById("transcriptInput");
  if (box) box.value = "";
  try { speechRecognition.start(); } catch (_) {}
}

function stopSpeechToText() {
  if (!speechRecognition) return;
  speechRecordingActive = false;
  speechFinalized = true;
  try { speechRecognition.stop(); } catch (_) {}
}

async function autoAnalyzeTranscript() {
  const text = transcriptInput.value.trim();
  if (!text) return null;
  if (text.split(/\s+/).filter(Boolean).length < 15) return null;
  try {
    const res = await fetch("/api/text-check", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text }),
    });
    const data = await res.json();
    if (res.ok) {
      renderTextResult(data);
      return data;
    }
  } catch (_) {}
  return null;
}

// ---------------------------------------------------------------- Text authenticity check
const checkTextBtn = document.getElementById("checkTextBtn");
const transcriptInput = document.getElementById("transcriptInput");
const textError = document.getElementById("textError");
const textResult = document.getElementById("textResult");

const BREAKDOWN_LABELS = {
  sentence_length_uniformity: "Sentence-length uniformity",
  low_lexical_diversity: "Low vocabulary diversity",
  stock_phrase_density: "Stock AI/scam phrase density",
  lack_of_disfluency: "Lack of natural disfluency (um, like...)",
  repeated_sentence_openers: "Repeated sentence openers",
};

checkTextBtn.addEventListener("click", async () => {
  const text = transcriptInput.value.trim();
  textError.textContent = "";
  if (!text) {
    textError.textContent = speechSupported ? "Record a call first or enter a transcript." : "Paste or type a transcript first.";
    return;
  }
  checkTextBtn.disabled = true;
  checkTextBtn.textContent = "Analyzing...";
  try {
    const res = await fetch("/api/text-check", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text }),
    });
    const data = await res.json();
    if (!res.ok) {
      textError.textContent = data.message || data.error || "could not analyze text";
      textResult.style.display = "none";
      return;
    }
    renderTextResult(data);
  } catch (err) {
    textError.textContent = "Error: " + err.message;
  } finally {
    checkTextBtn.disabled = false;
    checkTextBtn.textContent = "Analyze Text";
  }
});

function renderTextResult(data) {
  textResult.style.display = "block";

  const banner = document.getElementById("textVerdictBanner");
  const label = document.getElementById("textVerdictLabel");
  const score = document.getElementById("textVerdictScore");

  let cls = "safe";
  if (data.ai_likelihood >= 65) cls = "alert";
  else if (data.ai_likelihood >= 35) cls = "suspicious";
  banner.className = "verdict-banner " + cls;

  const icon = { safe: "🟢", suspicious: "🟡", alert: "🔴" }[cls];
  label.textContent = `${icon} ${data.label}`;
  score.textContent = `AI-likelihood ${data.ai_likelihood}/100 · ${data.word_count} words, ${data.sentence_count} sentences`;

  const bars = document.getElementById("breakdownBars");
  bars.innerHTML = "";
  Object.entries(data.breakdown).forEach(([key, val]) => {
    const row = document.createElement("div");
    row.className = "breakdown-row";
    row.innerHTML = `
      <div class="label">${BREAKDOWN_LABELS[key] || key}</div>
      <div class="bar-track"><div class="bar-fill" style="width:${val}%"></div></div>
      <div class="pct">${val}%</div>
    `;
    bars.appendChild(row);
  });

  const phraseLine = document.getElementById("stockPhrasesLine");
  phraseLine.textContent = data.matched_stock_phrases.length
    ? "Matched stock phrases: " + data.matched_stock_phrases.join(", ")
    : "No stock AI/scam phrases matched.";
}

// ---------------------------------------------------------------- Rendering
function renderResults(data, textData = null) {
  const panel = document.getElementById("resultsPanel");
  panel.style.display = "block";

  const banner = document.getElementById("verdictBanner");
  const label = document.getElementById("verdictLabel");
  const score = document.getElementById("verdictScore");
  banner.className = "verdict-banner " + data.overall_risk_label.toLowerCase();

  const icon = { SAFE: "🟢", SUSPICIOUS: "🟡", ALERT: "🔴" }[data.overall_risk_label];
  label.textContent = `${icon} ${data.overall_risk_label}`;
  score.textContent = `risk score ${(data.overall_risk_score * 100).toFixed(1)}% · ${data.duration_sec}s analyzed` +
    (data.speaker_enrolled ? "" : " · no reference speaker enrolled (synthetic-speech signal only)");

  const timeline = document.getElementById("timeline");
  timeline.innerHTML = "";
  data.windows.forEach((w) => {
    const div = document.createElement("div");
    div.className = "chunk " + w.risk_label.toLowerCase();
    div.innerHTML = `<div>${w.risk_label[0]}</div><div class="t">${w.start_sec}s</div>`;
    div.title = `Window ${w.index}: synthetic ${w.synthetic_prob}, similarity ${w.speaker_similarity ?? "n/a"}`;
    timeline.appendChild(div);
  });

  const tbody = document.getElementById("detailTableBody");
  tbody.innerHTML = "";
  data.windows.forEach((w) => {
    const tr = document.createElement("tr");
    tr.innerHTML = `
      <td>${w.start_sec}s – ${w.end_sec}s</td>
      <td>${(w.synthetic_prob * 100).toFixed(1)}%</td>
      <td>${w.speaker_similarity !== null ? (w.speaker_similarity * 100).toFixed(1) + "%" : "—"}</td>
      <td>${w.pitch_hz || "—"}</td>
      <td><span class="badge ${w.risk_label.toLowerCase()}">${w.risk_label}</span></td>
      <td>${textData ? `<span class="badge ${textData.ai_likelihood >= 65 ? "alert" : textData.ai_likelihood >= 35 ? "suspicious" : "safe"}">${textData.ai_likelihood >= 65 ? "FAIL" : textData.ai_likelihood >= 35 ? "REVIEW" : "PASS"}</span>` : "—"}</td>
    `;
    tbody.appendChild(tr);
  });

  if (textData) {
    const tr = document.createElement("tr");
    const status = textData.ai_likelihood >= 65 ? "FAIL" : textData.ai_likelihood >= 35 ? "REVIEW" : "PASS";
    const cls = status === "FAIL" ? "alert" : status === "REVIEW" ? "suspicious" : "safe";
    tr.innerHTML = `<td colspan="5"><strong>Transcript AI-script check</strong> — ${textData.ai_likelihood.toFixed(1)}% AI-likelihood</td><td><span class="badge ${cls}">${status}</span></td>`;
    tbody.appendChild(tr);
  }

  document.getElementById("melspecImg").src = "data:image/png;base64," + data.melspec_png_base64;
  panel.scrollIntoView({ behavior: "smooth", block: "start" });
}
