const API_BASE_URL = "https://youtube-ai-yos4vguqva-el.a.run.app";

const loginBtn = document.getElementById("login-btn");
const logoutBtn = document.getElementById("logout-btn");
const askBtn = document.getElementById("ask-button");

const questionInput = document.getElementById("question");
const chatDiv = document.getElementById("chat");
const youtubeUrlInput = document.getElementById("youtube-url");
const userEmailSpan = document.getElementById("user-email");

const sessionListDiv = document.getElementById("sessionList");

// SIDEBAR TOGGLE (NEW)
const toggleBtn = document.getElementById("toggle-sidebar");
const sidebar = document.getElementById("sidebar");

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
    console.error(" Failed to fetch title");
    return "YouTube Video";
  }
}


async function saveSessionMeta(sessionId, question) {
  const { email } = await chrome.storage.local.get(["email"]);
  const key = `sessions_${email}`;

  chrome.storage.local.get([key], async (res) => {
    let sessions = res[key] || [];

    const exists = sessions.find(s => s.id === sessionId);

    if (!exists) {
      const videoUrl = document.getElementById("youtube-url").value;
      const title = await getVideoTitle(videoUrl);

      sessions.unshift({
        id: sessionId,
        title: title.slice(0, 50)
      });

      chrome.storage.local.set({ [key]: sessions });
    }
  });
}


function loadSessionsUI() {
  chrome.storage.local.get(["email", "session_id"], (meta) => {
    const email = meta.email;
    const activeId = meta.session_id;

    if (!email) {
      sessionListDiv.innerHTML = "";
      return;
    }

    const key = `sessions_${email}`;

    chrome.storage.local.get([key], (res) => {
      const sessions = res[key] || [];

      sessionListDiv.innerHTML = "";

      sessions.forEach((s, index) => {
        const div = document.createElement("div");
        div.className = "session";

        if (s.id === activeId) {
          div.style.background = "#4CAF50";
        }

        const text = document.createElement("span");
        text.textContent = s.title;

        const del = document.createElement("span");
        del.textContent = " ❌";
        del.style.cursor = "pointer";
        del.style.float = "right";

        // DELETE FIX
        del.onclick = async (e) => {
          e.stopPropagation();

          const confirmDelete = confirm("Are you sure?");
          if (!confirmDelete) return;

          sessions.splice(index, 1);

          chrome.storage.local.set({ [key]: sessions }, () => {
            loadSessionsUI();
          });
        };

        text.ondblclick = () => {
          const newName = prompt("Rename chat:", s.title);
          if (!newName || !newName.trim()) return;

          s.title = newName.trim().slice(0, 50);

          chrome.storage.local.set({ [key]: sessions }, () => {
            loadSessionsUI();
          });
        };

        div.onclick = () => loadSession(s.id);

        div.appendChild(text);
        div.appendChild(del);

        sessionListDiv.appendChild(div);
      });
    });
  });
}



// ==============================
// LOAD CHAT HISTORY
// ==============================
async function loadSession(sessionId) {
  chatDiv.innerHTML = "";

  // SET ACTIVE SESSION
  chrome.storage.local.set({ session_id: sessionId });


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
      console.log(" No history found for session:", sessionId);
    }


  } catch (e) {
    console.error(" Failed to load history", e);
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
      alert(" Login failed");
      return;
    }

    try {
      const res = await fetch("https://www.googleapis.com/oauth2/v2/userinfo", {
        headers: { Authorization: "Bearer " + token }
      });

      const user = await res.json();

      // SAVE USER DATA
      chrome.storage.local.set({
        token,
        email: user.email
      }, () => {

        // UPDATE UI
        updateUI(user.email);

        // LOAD SIDEBAR SESSIONS (IMPORTANT FIX)
        loadSessionsUI();

        // LOAD LAST OPENED CHAT
        chrome.storage.local.get(["session_id"], (res) => {
          if (res.session_id) {
            loadSession(res.session_id);
          }
        });

      });

    } catch (e) {
      console.error(e);
      alert(" Login failed");
    }
  });
};

// ==============================
// LOGOUT
// ==============================
logoutBtn.onclick = () => {
  chrome.storage.local.remove(["token", "email"], () => {
    updateUI(null);
    chatDiv.innerHTML = "";
    sessionListDiv.innerHTML = "";
  });
};

// ==============================
// SIDEBAR TOGGLE LOGIC
// ==============================
if (toggleBtn && sidebar) {
  toggleBtn.onclick = () => {
    sidebar.classList.toggle("hidden");

    // Change icon
    if (sidebar.classList.contains("hidden")) {
      toggleBtn.textContent = "→";
    } else {
      toggleBtn.textContent = "☰";
    }

    // Save state
    chrome.storage.local.set({
      sidebar_hidden: sidebar.classList.contains("hidden")
    });
  };
}

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

    // IMPORTANT: update tooltip
    userEmailSpan.title = email;

    loginBtn.style.display = "none";
    logoutBtn.style.display = "inline";
  } else {
    userEmailSpan.textContent = "Not logged in";

    // IMPORTANT: update tooltip
    userEmailSpan.title = "Not logged in";

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
  chatDiv.scrollTo({ top: chatDiv.scrollHeight, behavior: "smooth" });
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

  const botMsg = addMessage(" Processing video...", "bot");

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
      botMsg.textContent = " Server error";
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

        // APPEND (NOT REPLACE)
        finalAnswer += chunkText;

        botMsg.innerHTML = finalAnswer.replace(/\n/g, "<br>");
        chatDiv.scrollTo({ top: chatDiv.scrollHeight,behavior: "smooth" });
      }
    }

    // FINAL BUFFER PROCESS
    if (buffer.trim()) {
      let lines = buffer.split("\n");

      if (lines[0].startsWith("data:")) {
        lines[0] = lines[0].replace("data:", "");
      }

      finalAnswer += lines.join("\n");

      botMsg.innerHTML = finalAnswer.replace(/\n/g, "<br>");
    }

    // ===== FINAL METRICS UI (WITH STRONG FINAL SCORE) =====
    if (metricsData) {

      const oldBox = botMsg.querySelector(".eval-box");
      if (oldBox) oldBox.remove();

      const evalBox = document.createElement("div");
      evalBox.className = "eval-box";

      evalBox.style.fontSize = "12px";
      evalBox.style.padding = "12px";
      evalBox.style.borderRadius = "10px";
      evalBox.style.marginTop = "10px";
      evalBox.style.lineHeight = "1.6";
      evalBox.style.borderLeft = "5px solid #333";

      // ===== VALUES =====
      const confidence = metricsData.confidence ?? 0;
      const llmConfidence = metricsData.llm_confidence ?? 0;
      const hallucination = metricsData.hallucination_score ?? 0;
      const contextQuality = metricsData.context_quality ?? 0;
      const finalScore = metricsData.final_score ?? 0;

      const contextLength = metricsData.context_length ?? 0;
      const retrievalCount = metricsData.retrieval_count ?? 0;
      const answerLength = metricsData.answer_length ?? 0;
      const source = (metricsData.source || "Transcript").toLowerCase();

      const fmt = (v) => (typeof v === "number" ? v.toFixed(2) : "N/A");

      // ===== STATUS =====
      let status = "Poor";
      let statusColor = "#dc3545"; // Red (Bootstrap "danger")

      if (finalScore >= 0.7) {
        status = "Good";
        statusColor = "#28a745";  // Green (Bootstrap "success")
      } else if (finalScore >= 0.4) {
        status = "Medium";
        statusColor = "#ffc107";  // Yellow / Amber (Bootstrap "warning")
      }

      // ===== SOURCE =====
    let sourceLabel = "Transcript";

    if (source.includes("wikipedia") && source.includes("tavily")) {
      sourceLabel = "Wikipedia + Tavily";
    }
    else if (source.includes("wikipedia")) {
      sourceLabel = "Wikipedia";
    }
    else if (source.includes("tavily")) {
      sourceLabel = "Tavily";
    }

      // ===== FINAL SCORE BAR =====
      const safeScore = Math.max(0, Math.min(finalScore, 1));
      const scorePercent = Math.round(safeScore * 100);

      // ===== UI =====
      evalBox.innerHTML = `
        <b>📊 Evaluation (${status})</b><br><br>

        <!-- 🔥 FINAL SCORE (BIG + VISUAL) -->
        <div style="font-size:16px; font-weight:bold; color:${statusColor};">
          ⭐ Final Score: ${fmt(finalScore)} (${scorePercent}%)
        </div>

        <!-- 🔥 PROGRESS BAR -->
        <div style="background:#eee; border-radius:6px; overflow:hidden; margin:6px 0 10px 0;">
          <div style="
            width:${scorePercent}%;
            background:${statusColor};
            height:8px;
          "></div>
        </div>

        🔹 Confidence: ${fmt(confidence)}<br>
        🔹 LLM Match: ${fmt(llmConfidence)}<br>
        🔹 Hallucination Risk: ${fmt(hallucination)}<br>
        🔹 Context Quality: ${fmt(contextQuality)}<br><br>

        🔹 Context Length: ${contextLength}<br>
        🔹 Retrieved Chunks: ${retrievalCount}<br>
        🔹 Answer Length: ${answerLength}<br>
        🔹 Source: ${sourceLabel}
      `;

      // ===== BACKGROUND =====
      if (finalScore >= 0.7) {
        evalBox.style.background = "#d4edda";
      } else if (finalScore >= 0.4) {
        evalBox.style.background = "#fff3cd";
      } else {
        evalBox.style.background = "#f8d7da";
      }

      botMsg.appendChild(evalBox);

      chatDiv.scrollTo({ top: chatDiv.scrollHeight, behavior: "smooth" });

      setTimeout(() => {
        if (evalBox) evalBox.remove();
      }, 40000);
    }

  } catch (err) {
    console.error(err);
    botMsg.textContent = " Failed";
  }
};

// ==============================
// INIT
// ==============================
document.addEventListener("DOMContentLoaded", async () => {
  loadUser();
  await loadYoutubeUrl();

  chrome.storage.local.get(["email"], (res) => {
    if (res.email) {
      loadSessionsUI();
    }
  });

  // RESTORE SIDEBAR STATE
  chrome.storage.local.get(["sidebar_hidden"], (res) => {
    if (res.sidebar_hidden && sidebar && toggleBtn) {
      sidebar.classList.add("hidden");
      toggleBtn.textContent = "→";
   }
  });

  // LOAD LAST CHAT AUTOMATICALLY
  chrome.storage.local.get(["session_id"], (res) => {
    if (res.session_id) {
      loadSession(res.session_id);
    }
  });
});
