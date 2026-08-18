const $ = (selector) => document.querySelector(selector);
const $$ = (selector) => [...document.querySelectorAll(selector)];

let API_BASE = window.APP_CONFIG?.API_BASE || "http://127.0.0.1:5000/api";
let students = [];
let currentFilter = "all";
let currentSearch = "";
let lastPrediction = null;
let toastTimer;
let usingDemoData = false;

const pageMeta = {
  dashboard: ["ACADEMIC MONITORING", "Dashboard", "A clear view of student performance and academic risk."],
  predictor: ["MODEL-POWERED ASSESSMENT", "Risk Predictor", "Estimate pass probability and identify current academic risk."],
  students: ["STUDENT RECORDS", "Students", "Search, filter and review saved prediction records."],
  analytics: ["ACADEMIC INSIGHTS", "Analytics", "A compact view of the indicators available to the model."],
  evaluation: ["QA & ANALYST", "Model Evaluation", "Cross-validated evidence focused on missed at-risk students."]
};

function normalizeBaseUrl(url) {
  return url.trim().replace(/\/+$/, "");
}

function showToast(message, type = "default") {
  const toast = $("#toast");
  toast.textContent = message;
  toast.className = `toast show ${type}`;
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => toast.classList.remove("show"), 3200);
}

function escapeHtml(value) {
  return String(value).replace(/[&<>'"]/g, (char) => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", "'": "&#039;", '"': "&quot;"
  }[char]));
}

function formatPercent(value) {
  return `${(Number(value) * 100).toFixed(1)}%`;
}

function riskBadge(risk) {
  const safe = ["high", "medium", "low"].includes(risk) ? risk : "low";
  return `<span class="risk-badge ${safe}"><i></i>${safe}</span>`;
}

async function api(path, options = {}) {
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), 20000);
  try {
    const response = await fetch(`${API_BASE}${path}`, {
      ...options,
      signal: controller.signal,
      headers: { ...(options.body ? { "Content-Type": "application/json" } : {}), ...(options.headers || {}) }
    });
    const data = await response.json().catch(() => ({}));
    if (!response.ok) throw new Error(data.error || `Request failed (${response.status})`);
    return data;
  } catch (error) {
    if (error.name === "AbortError") throw new Error("The prediction service timed out. Please try again.");
    throw error;
  } finally {
    clearTimeout(timeout);
  }
}

function switchSection(sectionId) {
  if (!$("#" + sectionId)) return;
  $$(".section").forEach(section => section.classList.toggle("active", section.id === sectionId));
  $$(".nav-item").forEach(item => item.classList.toggle("active", item.dataset.section === sectionId));

  const [eyebrow, title, subtitle] = pageMeta[sectionId] || pageMeta.dashboard;
  $("#pageEyebrow").textContent = eyebrow;
  $("#pageTitle").textContent = title;
  $("#pageSubtitle").textContent = subtitle;
  closeSidebar();

  if (sectionId === "dashboard" || sectionId === "analytics") refreshData();
  if (sectionId === "students") renderStudentsTable();
  if (sectionId === "evaluation") loadModelMetrics();
}

function closeSidebar() {
  $("#sidebar").classList.remove("open");
  $("#mobileBackdrop").classList.remove("open");
}

$$(".nav-item").forEach(button => button.addEventListener("click", () => switchSection(button.dataset.section)));
$$("[data-go]").forEach(button => button.addEventListener("click", () => switchSection(button.dataset.go)));

$("#mobileMenu").addEventListener("click", () => {
  $("#sidebar").classList.add("open");
  $("#mobileBackdrop").classList.add("open");
});
$("#mobileBackdrop").addEventListener("click", closeSidebar);
$("#refreshBtn").addEventListener("click", refreshData);

function setConnection(online, count = null) {
  $("#statusDot").classList.toggle("offline", !online);
  $("#connectionText").textContent = online
    ? `API connected${count === null ? "" : ` · ${count} loaded`}`
    : usingDemoData ? "Demo data · API offline" : "API offline";
  $("#dataModePill").innerHTML = online
    ? "<span></span> Live API"
    : usingDemoData ? "<span></span> Demo data" : "<span></span> API offline";
  $("#dataModePill").classList.toggle("demo", !online);
}

async function checkHealth() {
  try {
    const data = await api("/health");
    setConnection(true, data.students_loaded);
  } catch {
    setConnection(false);
  }
}

function calculateAverages(data) {
  if (!data.length) return { attendance: 0, homework: 0, midterm: 0, study: 0, performance: 0 };
  const sum = key => data.reduce((total, row) => total + Number(row[key] || 0), 0) / data.length;
  return {
    attendance: sum("attendance_pct"),
    homework: sum("homework_pct"),
    midterm: sum("midterm_score"),
    study: sum("study_hours_per_week"),
    performance: sum("performance_score")
  };
}

function updateRiskOverview(data) {
  const total = data.length;
  const high = data.filter(s => s.risk_flag === "high").length;
  const medium = data.filter(s => s.risk_flag === "medium").length;
  const low = data.filter(s => s.risk_flag === "low").length;
  const pct = value => total ? (value / total * 100) : 0;

  $("#totalStudents").textContent = total;
  $("#highRisk").textContent = high;
  $("#mediumRisk").textContent = medium;
  $("#lowRisk").textContent = low;
  $("#riskDonutTotal").textContent = total;
  $("#riskTotalLabel").textContent = `${total} student${total === 1 ? "" : "s"}`;

  $("#legendHigh").textContent = `${pct(high).toFixed(0)}%`;
  $("#legendMedium").textContent = `${pct(medium).toFixed(0)}%`;
  $("#legendLow").textContent = `${pct(low).toFixed(0)}%`;

  const highPct = pct(high), mediumPct = pct(medium);
  $("#riskDonut").style.background = `conic-gradient(#ef5a67 0 ${highPct}%, #eab04d ${highPct}% ${highPct + mediumPct}%, #35b77a ${highPct + mediumPct}% 100%)`;

  $("#analyticsTotal").textContent = total;
  $("#analyticsHigh").textContent = high;
  $("#analyticsMedium").textContent = medium;
  $("#analyticsLow").textContent = low;
  $("#stackHigh").style.width = `${highPct}%`;
  $("#stackMedium").style.width = `${mediumPct}%`;
  $("#stackLow").style.width = `${pct(low)}%`;
}

function updateAverages(data) {
  const avg = calculateAverages(data);
  const values = [
    ["avgAttendance", "attendanceBar", avg.attendance, 100],
    ["avgHomework", "homeworkBar", avg.homework, 100],
    ["avgMidterm", "midtermBar", avg.midterm, 100],
    ["avgStudyHours", "studyBar", avg.study, 20]
  ];
  values.forEach(([label, bar, value, max]) => {
    $("#" + label).textContent = label === "avgStudyHours" ? `${value.toFixed(1)} hrs` : `${value.toFixed(1)}%`;
    $("#" + bar).style.width = `${Math.min(100, value / max * 100)}%`;
  });

  $("#analyticsAttendance").textContent = `${avg.attendance.toFixed(1)}%`;
  $("#analyticsHomework").textContent = `${avg.homework.toFixed(1)}%`;
  $("#analyticsMidterm").textContent = `${avg.midterm.toFixed(1)}%`;
  $("#analyticsStudy").textContent = `${avg.study.toFixed(1)} hrs`;
}

function renderRecent(data) {
  const rows = [...data].sort((a, b) => Number(b.student_id) - Number(a.student_id)).slice(0, 6);
  $("#recentTable").innerHTML = rows.length ? rows.map(student => `
    <tr data-id="${student.student_id}" class="clickable-row">
      <td><strong class="student-number">${escapeHtml(student.student_name || `Student ${student.student_id}`)}</strong><small class="student-id">#${escapeHtml(student.student_id)}</small></td>
      <td>${Number(student.attendance_pct).toFixed(1)}%</td>
      <td>${Number(student.homework_pct).toFixed(1)}%</td>
      <td>${Number(student.midterm_score).toFixed(1)}%</td>
      <td>${Number(student.study_hours_per_week).toFixed(1)} hrs</td>
      <td>${riskBadge(student.risk_flag)}</td>
    </tr>`).join("") : `<tr><td colspan="6" class="empty-cell">No student records available.</td></tr>`;
  $$("#recentTable .clickable-row").forEach(row => row.addEventListener("click", () => openStudentDetail(row.dataset.id)));
}

async function refreshData() {
  try {
    const data = await api("/students");
    usingDemoData = false;
    students = Array.isArray(data) ? data : [];
    updateRiskOverview(students);
    updateAverages(students);
    renderRecent(students);
    renderStudentsTable();
    await checkHealth();
  } catch (error) {
    try {
      const response = await fetch("demo_students.json");
      if (!response.ok) throw new Error("Demo data is unavailable.");
      students = await response.json();
      usingDemoData = true;
      updateRiskOverview(students);
      updateAverages(students);
      renderRecent(students);
      renderStudentsTable();
      setConnection(false);
      showToast("Live API unavailable — showing read-only demo data.");
    } catch {
      usingDemoData = false;
      setConnection(false);
      showToast(error.message || "Unable to load student data.", "error");
    }
  }
}

function filteredStudents() {
  return students.filter(student => {
    const matchesRisk = currentFilter === "all" || student.risk_flag === currentFilter;
    const haystack = `${student.student_id} ${student.student_name || ""}`.toLowerCase();
    const matchesSearch = !currentSearch || haystack.includes(currentSearch.toLowerCase());
    return matchesRisk && matchesSearch;
  }).sort((a, b) => Number(b.student_id) - Number(a.student_id));
}

function renderStudentsTable() {
  if (!$("#studentsTable")) return;
  const data = filteredStudents();
  $("#studentCountLabel").textContent = `${data.length} of ${students.length} record${students.length === 1 ? "" : "s"}`;
  $("#studentsTable").innerHTML = data.length ? data.map(student => `
    <tr>
      <td><button class="student-link" data-id="${student.student_id}">${escapeHtml(student.student_name || `Student ${student.student_id}`)}</button><small class="student-id">#${escapeHtml(student.student_id)}</small></td>
      <td>${Number(student.attendance_pct).toFixed(1)}%</td>
      <td>${Number(student.homework_pct).toFixed(1)}%</td>
      <td>${Number(student.midterm_score).toFixed(1)}%</td>
      <td>${Number(student.study_hours_per_week).toFixed(1)} hrs</td>
      <td>${Number(student.performance_score).toFixed(1)}</td>
      <td>${formatPercent(student.pass_probability ?? (1 - Number(student.risk_probability)))}</td>
      <td>${riskBadge(student.risk_flag)}</td>
      <td><button class="delete-btn" data-id="${student.student_id}" title="Delete student">⌫</button></td>
    </tr>`).join("") : `<tr><td colspan="9" class="empty-cell">No records match your search.</td></tr>`;

  $$(".student-link").forEach(button => button.addEventListener("click", () => openStudentDetail(button.dataset.id)));
  $$(".delete-btn").forEach(button => button.addEventListener("click", () => deleteStudent(button.dataset.id)));
}

$("#studentSearch").addEventListener("input", event => {
  currentSearch = event.target.value.trim();
  renderStudentsTable();
});
$("#riskFilters").addEventListener("click", event => {
  const button = event.target.closest(".filter-btn");
  if (!button) return;
  currentFilter = button.dataset.filter;
  $$(".filter-btn").forEach(item => item.classList.toggle("active", item === button));
  renderStudentsTable();
});

async function deleteStudent(id) {
  const student = students.find(item => String(item.student_id) === String(id));
  if (!student) return;
  if (usingDemoData) return showToast("Demo records are read-only. Reconnect the API to make changes.", "error");
  if (!window.confirm(`Delete Student #${id}? This action cannot be undone.`)) return;
  try {
    await api(`/students/${id}`, { method: "DELETE" });
    showToast(`Student #${id} deleted.`, "success");
    await refreshData();
  } catch (error) {
    showToast(error.message, "error");
  }
}

function openStudentDetail(id) {
  const student = students.find(item => String(item.student_id) === String(id));
  if (!student) return;
  $("#detailTitle").textContent = student.student_name || `Student #${student.student_id}`;
  $("#detailGrid").innerHTML = [
    ["Student ID", `#${student.student_id}`],
    ["Risk level", riskBadge(student.risk_flag)],
    ["At-risk probability", formatPercent(student.risk_probability)],
    ["Pass probability", formatPercent(student.pass_probability ?? (1 - Number(student.risk_probability)))],
    ["Prediction", student.predicted_outcome === "fail" || student.prediction === 0 ? "At risk of failing" : "Likely to pass"],
    ["Attendance", `${Number(student.attendance_pct).toFixed(1)}%`],
    ["Homework", `${Number(student.homework_pct).toFixed(1)}%`],
    ["Midterm", `${Number(student.midterm_score).toFixed(1)}%`],
    ["Study hours", `${Number(student.study_hours_per_week).toFixed(1)} hrs/week`],
    ["Performance score", Number(student.performance_score).toFixed(1)]
  ].map(([label, value]) => `<div><span>${label}</span><strong>${value}</strong></div>`).join("");
  openModal("detailModal");
}

function openModal(id) { $("#" + id).classList.add("open"); $("#" + id).setAttribute("aria-hidden", "false"); }
function closeModal(id) { $("#" + id).classList.remove("open"); $("#" + id).setAttribute("aria-hidden", "true"); }
$$("[data-close]").forEach(button => button.addEventListener("click", () => closeModal(button.dataset.close)));
$$(".modal-overlay").forEach(overlay => overlay.addEventListener("click", event => { if (event.target === overlay) closeModal(overlay.id); }));

document.addEventListener("keydown", event => {
  if (event.key === "Escape") $$(".modal-overlay.open").forEach(modal => closeModal(modal.id));
});

// Predictor range controls
const rangePairs = [
  ["attendanceInput", "attendanceRange"],
  ["homeworkInput", "homeworkRange"],
  ["midtermInput", "midtermRange"],
  ["studyInput", "studyRange"]
];
rangePairs.forEach(([inputId, rangeId]) => {
  const input = $("#" + inputId), range = $("#" + rangeId);
  input.addEventListener("input", () => { if (input.value !== "") range.value = Math.min(Number(range.max), Number(input.value)); });
  range.addEventListener("input", () => { input.value = range.value; });
});

function getPredictionPayload() {
  const form = new FormData($("#predictForm"));
  return {
    student_name: String(form.get("student_name") || "").trim(),
    attendance_pct: Number(form.get("attendance_pct")),
    homework_pct: Number(form.get("homework_pct")),
    midterm_score: Number(form.get("midterm_score")),
    study_hours_per_week: Number(form.get("study_hours_per_week"))
  };
}

function validatePayload(payload) {
  if (![payload.attendance_pct, payload.homework_pct, payload.midterm_score, payload.study_hours_per_week].every(Number.isFinite)) return "Please enter all four academic indicators.";
  if (payload.attendance_pct < 0 || payload.attendance_pct > 100) return "Attendance must be between 0 and 100.";
  if (payload.homework_pct < 0 || payload.homework_pct > 100) return "Homework completion must be between 0 and 100.";
  if (payload.midterm_score < 0 || payload.midterm_score > 100) return "Midterm score must be between 0 and 100.";
  if (payload.study_hours_per_week < 0) return "Study hours cannot be negative.";
  return "";
}

function showPrediction(result) {
  lastPrediction = { ...result, payload: getPredictionPayload() };
  const probability = Number(result.risk_probability);
  const risk = ["high", "medium", "low"].includes(result.risk_flag) ? result.risk_flag : "low";
  const likelyPass = result.predicted_outcome ? result.predicted_outcome === "pass" : Number(result.prediction) === 1;

  $("#resultEmpty").classList.add("hidden");
  $("#resultContent").classList.remove("hidden");
  $("#resultTitle").textContent = likelyPass ? "Likely to pass" : "At risk of failing";
  $("#probability").textContent = formatPercent(probability);
  $("#riskBadge").className = `risk-badge ${risk}`;
  $("#riskBadge").innerHTML = `<i></i>${risk} risk`;
  $("#probabilityRing").style.background = `conic-gradient(var(--${risk}) ${Math.min(100, probability * 100)}%, #edf0f5 0)`;
  $("#resultMessage").textContent = result.recommendation || (risk === "high"
    ? "This student should receive prompt academic attention and support."
    : risk === "medium"
      ? "This student should be monitored and supported before performance declines further."
      : "The current indicators suggest a lower academic risk.");
}

$("#predictForm").addEventListener("submit", async event => {
  event.preventDefault();
  $("#predictError").textContent = "";
  const payload = getPredictionPayload();
  const validationError = validatePayload(payload);
  if (validationError) { $("#predictError").textContent = validationError; return; }

  const button = $("#predictBtn");
  button.disabled = true;
  button.innerHTML = "Analyzing…";
  try {
    const result = await api("/predict", { method: "POST", body: JSON.stringify(payload) });
    showPrediction(result);
  } catch (error) {
    $("#predictError").textContent = error.message || "Unable to reach the prediction service.";
  } finally {
    button.disabled = false;
    button.innerHTML = "Analyze risk <span>→</span>";
  }
});

$("#predictForm").addEventListener("reset", () => {
  setTimeout(() => {
    rangePairs.forEach(([inputId, rangeId]) => { $("#" + rangeId).value = 0; });
    $("#resultEmpty").classList.remove("hidden");
    $("#resultContent").classList.add("hidden");
    $("#predictError").textContent = "";
    lastPrediction = null;
  }, 0);
});

$("#newPredictionBtn").addEventListener("click", () => $("#predictForm").reset());

$("#saveStudentBtn").addEventListener("click", async () => {
  if (!lastPrediction?.payload) return;
  if (usingDemoData) return showToast("Reconnect the API before saving a student.", "error");
  const button = $("#saveStudentBtn");
  button.disabled = true;
  button.innerHTML = "Saving…";
  try {
    const saved = await api("/students", { method: "POST", body: JSON.stringify(lastPrediction.payload) });
    showToast(`Student #${saved.student_id} saved successfully.`, "success");
    await refreshData();
    switchSection("students");
  } catch (error) {
    showToast(error.message || "Unable to save student.", "error");
  } finally {
    button.disabled = false;
    button.innerHTML = "Save student <span>＋</span>";
  }
});

// API settings
$("#settingsBtn").addEventListener("click", () => {
  $("#apiBaseInput").value = API_BASE;
  openModal("settingsModal");
});
$("#saveApiBtn").addEventListener("click", async () => {
  const value = normalizeBaseUrl($("#apiBaseInput").value);
  if (!value) return showToast("Enter a valid API base URL.", "error");
  API_BASE = value;
  localStorage.setItem("studentIQApiBase", API_BASE);
  closeModal("settingsModal");
  showToast("API URL updated. Reconnecting…");
  await refreshData();
});

const savedApiBase = localStorage.getItem("studentIQApiBase");
const localHostnames = ["localhost", "127.0.0.1", "::1"];
if (savedApiBase && !localHostnames.includes(window.location.hostname)) {
  API_BASE = normalizeBaseUrl(savedApiBase);
}

function renderModelMetrics(metrics) {
  if (!metrics?.confusion_matrix) return;
  const cm = metrics.confusion_matrix;
  $("#metricRecall").textContent = formatPercent(metrics.at_risk_recall);
  $("#metricAccuracy").textContent = formatPercent(metrics.accuracy);
  $("#metricFalseNegatives").textContent = cm.false_negative;
  $("#metricThreshold").textContent = Number(metrics.threshold).toFixed(2);
  $("#metricSample").textContent = `${metrics.students_evaluated} students`;
  $("#matrixTN").textContent = cm.true_negative;
  $("#matrixFP").textContent = cm.false_positive;
  $("#matrixFN").textContent = cm.false_negative;
  $("#matrixTP").textContent = cm.true_positive;
  $("#modelType").textContent = `${metrics.model_type || "Logistic Regression"} · v${metrics.model_version || "2.0"}`;
  $("#evaluationMethod").textContent = metrics.evaluation_method || "5-fold stratified cross-validation";
  $("#datasetNote").textContent = metrics.dataset_note || "This predictor supports staff decisions; it does not replace teacher judgment.";
}

async function loadModelMetrics() {
  try {
    renderModelMetrics(await api("/model-metrics"));
  } catch {
    try {
      const response = await fetch("model_metrics.json");
      renderModelMetrics(await response.json());
    } catch {
      // The rest of the dashboard remains usable if the QA report is unavailable.
    }
  }
}

Promise.allSettled([refreshData(), loadModelMetrics()]);
