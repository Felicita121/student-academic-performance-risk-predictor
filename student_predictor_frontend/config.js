const isLocal = ["localhost", "127.0.0.1", "::1"].includes(window.location.hostname);

window.APP_CONFIG = {
  API_BASE: isLocal
    ? "http://127.0.0.1:5000/api"
    : "https://student-academic-performance-risk-api.onrender.com/api"
};
