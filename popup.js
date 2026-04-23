const API_BASE_URL = "https://youtube-ai-yos4vguqva-el.a.run.app";

const loginBtn = document.getElementById("login-btn");
const logoutBtn = document.getElementById("logout-btn");
const askBtn = document.getElementById("ask-button");

const questionInput = document.getElementById("question");
const chatDiv = document.getElementById("chat");
const youtubeUrlInput = document.getElementById("youtube-url");
const userEmailSpan = document.getElementById("user-email");

const sessionListDiv = document.getElementById("sessionList");

// ==============================
// SESSION MANAGEMENT (NEW)
// ==============================
async function getSessionId() {
  const videoUrl = document.getElementById("youtube-url").value;
  const { email } = await chrome.storage.local.get(["email"]);

  const videoIdMatch = videoUrl.match(/(?:v=|youtu.be\/)([^&?/]+)/);
  const videoId = videoIdMatch ? videoIdMatch[1] : null;

  // EDGE CASE FIX
  if (!videoId) {
    const fallbackId = Date.now().toString();
    chrome.storage.local.set({ session_id: fallbackId });
    return fallbackId;
  }

  const sessionId = `${email}_${videoId}`;

  chrome.storage.local.set({ session_id: sessionId });

  return sessionId;
}

async function getVideoTitle(videoUrl) {
  try {
    const res = await fetch(`https://www.youtube.com/oembed?url=${videoUrl}&format=json`);
    const data = await res.json();
    return data.title;
  } catch (e) {
    console.error("❌ Failed to fetch title");
    return "YouTube Video";
  }
}


async function saveSessionMeta(sessionId, question) {
  chrome.storage.local.get(["sessions"], async (res) => {
    let sessions = res.sessions || [];

    const exists = sessions.find(s => s.id === sessionId);

    if (!exists) {

      const videoUrl = document.getElementById("youtube-url").value;

      // FETCH VIDEO TITLE
      const title = await getVideoTitle(videoUrl);

      sessions.unshift({
        id: sessionId,
        title: title.slice(0, 50)
      });

      chrome.storage.local.set({ sessions });
    }
  });
}

function loadSessionsUI() {
  chrome.storage.local.get(["sessions", "session_id"], (res) => {
    const sessions = res.sessions || [];
    const activeId = res.session_id;

    sessionListDiv.innerHTML = "";

    sessions.forEach((s, index) => {
      const div = document.createElement("div");
      div.className = "session";

      // Highlight active session
      if (s.id === activeId) {
        div.style.background = "#4CAF50";
      }

      // ===== TEXT =====
      const text = document.createElement("span");
      text.textContent = s.title;

      // ===== DELETE BUTTON =====
      const del = document.createElement("span");
      del.textContent = " ❌";
      del.style.cursor = "pointer";
      del.style.float = "right";



      del.onclick = async (e) => {
          e.stopPropagation();

          const sessionId = s.id;

          // CONFIRMATION POPUP
          const confirmDelete = confirm("Are you sure you want to delete this chat?");
          if (!confirmDelete) return;

          try {
            // FIX 1: Prevent unnecessary API call (only valid sessions)
            if (sessionId.includes("_")) {
              await fetch(`${API_BASE_URL}/api/history/${sessionId}`, {
                method: "DELETE"
              });
            }

            console.log("✅ Deleted from backend:", sessionId);

          } catch (err) {
            console.error("❌ Backend delete failed:", err);
          }

          // DELETE FROM LOCAL STORAGE
          sessions.splice(index, 1);

          // FIX 2: Auto-switch to another session (better UX)
          const newActiveSession = sessions.length ? sessions[0].id : null;

          chrome.storage.local.set(
            { sessions, session_id: newActiveSession },
            () => {

              // CLEAR CHAT UI IF ACTIVE SESSION DELETED
              if (sessionId === activeId) {
                chatDiv.innerHTML = "";
              }

              // FIX 3: Update status message
              const statusDiv = document.getElementById("status");
              if (statusDiv) {
                statusDiv.textContent = "Chat deleted";
              }

              loadSessionsUI();
            }
          );
      };


      // ===== RENAME (DOUBLE CLICK) =====

       text.ondblclick = () => {
          const newName = prompt("Rename chat:", s.title);

          // VALIDATION
          if (!newName || !newName.trim()) {
            return; // ignore empty or cancel
          }

          const trimmed = newName.trim();

          // OPTIONAL: LIMIT LENGTH
          s.title = trimmed.slice(0, 50);

          chrome.storage.local.set({ sessions }, () => {
            loadSessionsUI();
          });
       };


      // ===== CLICK LOAD =====
      div.onclick = () => loadSession(s.id);

      div.appendChild(text);
      div.appendChild(del);

      sessionListDiv.appendChild(div);
    });
  });
}

// ==============================
// LOAD CHAT HISTORY (UPDATED FIX)
// ==============================
async function loadSession(sessionId) {
  chatDiv.innerHTML = "";

  // SET ACTIVE SESSION
  chrome.storage.local.set({ session_id: sessionId });

  // IMPORTANT FIX (ADDED)
  loadSessionsUI();

  try {
    const res = await fetch(`${API_BASE_URL}/api/history/${sessionId}`);
    const data = await res.json();


    if (data.history && Array.isArray(data.history)) {
      data.history.forEach((item) => {
      if (item.question) addMessage(item.question, "user");
      if (item.answer) addMessage(item.answer, "bot");
    });
    } else {
      console.log("⚠️ No history found for session:", sessionId);
    }


  } catch (e) {
    console.error("❌ Failed to load history", e);
  }
}

// ==============================
// AUTO LOAD YOUTUBE URL
// ==============================
async function loadYoutubeUrl() {
  const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });

  if (tab?.url && tab.url.includes("youtube.com/watch")) {
    youtubeUrlInput.value = tab.url;
  }
}

// ==============================
// LOGIN
// ==============================
loginBtn.onclick = () => {
  chrome.identity.getAuthToken({ interactive: true }, async (token) => {

    if (chrome.runtime.lastError || !token) {
      alert("❌ Login failed");
      return;
    }

    try {
      const res = await fetch("https://www.googleapis.com/oauth2/v2/userinfo", {
        headers: { Authorization: "Bearer " + token }
      });

      const user = await res.json();

      chrome.storage.local.set({
        token,
        email: user.email
      });

      updateUI(user.email);

    } catch (e) {
      console.error(e);
      alert("❌ Login failed");
    }
  });
};

// ==============================
// LOGOUT
// ==============================
logoutBtn.onclick = () => {
  chrome.storage.local.clear(() => {
    updateUI(null);
    chatDiv.innerHTML = "";
  });
};

// ==============================
// LOAD USER
// ==============================
function loadUser() {
  chrome.storage.local.get(["token", "email"], (data) => {
    if (data.token) {
      updateUI(data.email);
    } else {
      updateUI(null);
    }
  });
}

// ==============================
// UI UPDATE
// ==============================
function updateUI(email) {
  if (email) {
    userEmailSpan.textContent = email;
    loginBtn.style.display = "none";
    logoutBtn.style.display = "inline";
  } else {
    userEmailSpan.textContent = "Not logged in";
    loginBtn.style.display = "block";
    logoutBtn.style.display = "none";
  }
}

// ==============================
// ADD MESSAGE
// ==============================
function addMessage(text, type) {
  const div = document.createElement("div");
  div.className = `msg ${type}`;
  div.textContent = text;
  chatDiv.appendChild(div);
  chatDiv.scrollTop = chatDiv.scrollHeight;
  return div;
}

// ==============================
// BUTTON ENABLE
// ==============================
questionInput.addEventListener("input", () => {
  askBtn.disabled = !questionInput.value.trim();
});

// ==============================
// ASK BUTTON (UPDATED)
// ==============================
askBtn.onclick = async () => {
  const question = questionInput.value.trim();

  const { token } = await chrome.storage.local.get(["token"]);

  if (!token) {
    alert("Please login first");
    return;
  }

  const videoUrl = youtubeUrlInput.value;

  if (!videoUrl) {
    alert("Open a YouTube video first");
    return;
  }

  const sessionId = await getSessionId();

  saveSessionMeta(sessionId, question);
  loadSessionsUI();

  addMessage(question, "user");
  questionInput.value = "";

  const botMsg = addMessage("⏳ Processing video...", "bot");

  try {
    const response = await fetch(`${API_BASE_URL}/api/ask-stream`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "Authorization": "Bearer " + token
      },
      body: JSON.stringify({
        video_url: videoUrl,
        question,
        session_id: sessionId
      })
    });

    if (!response.ok) {
      botMsg.textContent = "❌ Server error";
      return;
    }

    const reader = response.body.getReader();
    const decoder = new TextDecoder();

    let buffer = "";
    let finalAnswer = "";
    let metricsData = null;

    botMsg.innerHTML = "";

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      buffer += decoder.decode(value, { stream: true });

      // process FULL event only when \n\n appears
      if (!buffer.includes("\n\n")) continue;

      const parts = buffer.split("\n\n");
      buffer = parts.pop(); // keep incomplete

      for (let part of parts) {
        if (!part.trim()) continue;

        // ===== METRICS =====
        if (part.includes("event: metrics")) {
          try {
            const dataLine = part.split("\n").find(l => l.startsWith("data:"));
            if (dataLine) {
              metricsData = JSON.parse(dataLine.replace("data:", "").trim());
            }
          } catch {}
          continue;
        }

        // ===== ANSWER (FIXED) =====

        let lines = part.split("\n");

        // remove "data:" from first line
        if (lines[0].startsWith("data:")) {
          lines[0] = lines[0].replace("data:", "");
        }

        const chunkText = lines.join("\n");

        // CRITICAL FIX: APPEND (NOT REPLACE)
        finalAnswer += chunkText;

        botMsg.innerHTML = finalAnswer.replace(/\n/g, "<br>");
        chatDiv.scrollTop = chatDiv.scrollHeight;
      }
    }

    // FINAL BUFFER PROCESS (VERY IMPORTANT)
    if (buffer.trim()) {
      let lines = buffer.split("\n");

      if (lines[0].startsWith("data:")) {
        lines[0] = lines[0].replace("data:", "");
      }

      finalAnswer += lines.join("\n");

      botMsg.innerHTML = finalAnswer.replace(/\n/g, "<br>");
    }

    // ===== METRICS =====
    if (metricsData) {
      const evalBox = document.createElement("div");

      evalBox.style.fontSize = "12px";
      evalBox.style.padding = "6px";
      evalBox.style.borderRadius = "6px";
      evalBox.style.marginTop = "8px";

      evalBox.style.background =
        (metricsData.confidence ?? 0) > 0.7
          ? "#d4edda"
          : "#f8d7da";

      evalBox.innerText =
        "📊 Evaluation:\n" +
        "Confidence: " + (metricsData.confidence ?? "N/A") + "\n" +
        "Context Length: " + (metricsData.context_length ?? "N/A") + "\n" +
        "Context Quality: " + (metricsData.context_quality ?? "N/A") + "\n" +
        "Final Score: " + (metricsData.final_score ?? "N/A") + "\n" +
        "Answer Length: " + (metricsData.answer_length ?? "N/A") + "\n" +
        "Retrieval Count: " + (metricsData.retrieval_count ?? "N/A") + "\n" +
        "Source: " + (metricsData.source ?? "N/A");

      chatDiv.appendChild(evalBox);

      setTimeout(() => evalBox.remove(), 40000);
    }

  } catch (err) {
    console.error(err);
    botMsg.textContent = "❌ Failed";
  }
};

// ==============================
// 🔹 INIT
// ==============================
document.addEventListener("DOMContentLoaded", async () => {
  loadUser();
  await loadYoutubeUrl();
  loadSessionsUI();

  // LOAD LAST CHAT AUTOMATICALLY
  chrome.storage.local.get(["session_id"], (res) => {
    if (res.session_id) {
      loadSession(res.session_id);
    }
  });
});
