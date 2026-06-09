// Extracts readable text from the current page — called by popup.js via chrome.scripting
function extractPageText() {
  const noise = document.querySelectorAll("script, style, nav, footer, header, aside, .ad, .sidebar");
  noise.forEach(el => el.remove());
  const text = document.body.innerText || document.body.textContent || "";
  // Truncate to ~3000 chars to stay within API context limits
  return text.replace(/\s+/g, " ").trim().slice(0, 3000);
}
