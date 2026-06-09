// popup.js — JRC AI Summarizer
// MOCK = true  → hardcoded summary (no API needed)
// MOCK = false → calls Claude API with page text
const MOCK = true;

const summarizeBtn = document.getElementById("summarize-btn");
const saveKeyBtn   = document.getElementById("save-key-btn");
const apiKeyInput  = document.getElementById("api-key-input");
const modeBadge    = document.getElementById("mode-badge");
const statusEl     = document.getElementById("status");
const resultBox    = document.getElementById("result-box");
const summaryText  = document.getElementById("summary-text");
const copyBtn      = document.getElementById("copy-btn");

// ── Init: load saved API key ──────────────────────────────────────────────────
chrome.storage.sync.get(["apiKey"], ({ apiKey }) => {
  if (apiKey) {
    apiKeyInput.value = apiKey;
    setModeBadge();
  }
});

// ── Save API key ──────────────────────────────────────────────────────────────
saveKeyBtn.addEventListener("click", () => {
  const key = apiKeyInput.value.trim();
  if (!key) { showStatus("Paste your ANTHROPIC_API_KEY first.", "error"); return; }
  chrome.storage.sync.set({ apiKey: key }, () => {
    setModeBadge();
    showStatus("API key saved. Set MOCK = false in popup.js to enable live mode.", "");
  });
});

function setModeBadge() {
  modeBadge.textContent = MOCK ? "MOCK" : "LIVE";
  modeBadge.className   = "badge " + (MOCK ? "mock" : "live");
}

// ── Summarize ─────────────────────────────────────────────────────────────────
summarizeBtn.addEventListener("click", async () => {
  showStatus("Extracting page text...", "");
  resultBox.classList.add("hidden");

  try {
    const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });

    const [{ result: pageText }] = await chrome.scripting.executeScript({
      target: { tabId: tab.id },
      func: () => {
        document.querySelectorAll("script,style,nav,footer,header,aside").forEach(el => el.remove());
        return (document.body.innerText || "").replace(/\s+/g, " ").trim().slice(0, 3000);
      },
    });

    if (!pageText || pageText.length < 50) {
      showStatus("Not enough text on this page to summarize.", "error");
      return;
    }

    showStatus("Summarizing...", "");
    const summary = MOCK
      ? getMockSummary(tab.title || "this page", pageText)
      : await callClaudeAPI(pageText, tab.title);

    summaryText.textContent = summary;
    resultBox.classList.remove("hidden");
    statusEl.classList.add("hidden");

  } catch (err) {
    showStatus("Error: " + err.message, "error");
  }
});

// ── Mock summary ──────────────────────────────────────────────────────────────
function getMockSummary(title, text) {
  const wordCount = text.split(" ").length;
  return (
    `[MOCK Summary of "${title}"]\n\n` +
    `• Page contains ~${wordCount} words of content.\n` +
    `• Preview: "${text.slice(0, 150).replace(/\n/g, " ")}..."\n\n` +
    `In live mode, Claude returns:\n` +
    `  → 4–6 bullet point summary of main ideas\n` +
    `  → Key takeaways and actionable insights\n\n` +
    `To enable: paste ANTHROPIC_API_KEY above → Save Key → set MOCK = false in popup.js.`
  );
}

// ── Real Claude API call ──────────────────────────────────────────────────────
async function callClaudeAPI(pageText, pageTitle) {
  const { apiKey } = await chrome.storage.sync.get(["apiKey"]);
  if (!apiKey) throw new Error("No API key saved. Paste it above and click Save Key.");

  const response = await fetch("https://api.anthropic.com/v1/messages", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      "x-api-key": apiKey,
      "anthropic-version": "2023-06-01",
    },
    body: JSON.stringify({
      model: "claude-haiku-4-5-20251001",
      max_tokens: 512,
      messages: [{
        role: "user",
        content: `Summarize this webpage in 4–6 bullet points. Be concise and highlight key insights.\n\nTitle: ${pageTitle}\n\nContent:\n${pageText}`,
      }],
    }),
  });

  if (!response.ok) {
    const err = await response.json();
    throw new Error(err.error?.message || "API error " + response.status);
  }

  const data = await response.json();
  return data.content[0].text;
}

// ── Copy button ───────────────────────────────────────────────────────────────
copyBtn.addEventListener("click", () => {
  navigator.clipboard.writeText(summaryText.textContent).then(() => {
    copyBtn.textContent = "Copied!";
    setTimeout(() => { copyBtn.textContent = "Copy"; }, 1500);
  });
});

// ── Helpers ───────────────────────────────────────────────────────────────────
function showStatus(msg, type) {
  statusEl.textContent = msg;
  statusEl.className   = "status" + (type === "error" ? " error" : "");
  statusEl.classList.remove("hidden");
}
