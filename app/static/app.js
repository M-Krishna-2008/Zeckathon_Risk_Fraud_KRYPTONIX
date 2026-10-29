/**
 * Distress Watch - Frontend Controller
 * Pure Vanilla JavaScript with LocalStorage Persistence & Micro Markdown Renderer.
 */

(function () {
  "use strict";

  // Elements
  const chatInput = document.getElementById("chatInput");
  const sendBtn = document.getElementById("sendBtn");
  const messagesContainer = document.getElementById("messagesContainer");
  const messageStream = document.getElementById("messageStream");
  const emptyState = document.getElementById("emptyState");
  const newChatBtn = document.getElementById("newChatBtn");
  const chatHistoryList = document.getElementById("chatHistoryList");
  const themeToggleBtn = document.getElementById("themeToggleBtn");
  const themeToggleText = document.getElementById("themeToggleText");
  const mobileMenuBtn = document.getElementById("mobileMenuBtn");
  const sidebar = document.getElementById("sidebar");
  const sidebarCloseBtn = document.getElementById("sidebarCloseBtn");
  const quickWatchlistBtn = document.getElementById("quickWatchlistBtn");
  const quickMetricsBtn = document.getElementById("quickMetricsBtn");

  // State
  let conversations = [];
  let currentChatId = null;
  let isGenerating = false;

  // Local Storage Keys
  const STORAGE_KEY_CHATS = "distress_watch_chats_v1";
  const STORAGE_KEY_ACTIVE = "distress_watch_active_id";
  const STORAGE_KEY_THEME = "distress_watch_theme";

  // ==========================================================================
  // Initialization & Theme
  // ==========================================================================
  function init() {
    initTheme();
    loadConversations();
    attachEventListeners();

    if (conversations.length > 0) {
      const activeId = localStorage.getItem(STORAGE_KEY_ACTIVE);
      const exists = conversations.some((c) => c.id === activeId);
      switchChat(exists ? activeId : conversations[0].id);
    } else {
      createNewChat();
    }
  }

  function initTheme() {
    const savedTheme = localStorage.getItem(STORAGE_KEY_THEME) || "dark";
    setTheme(savedTheme);
  }

  function setTheme(theme) {
    document.documentElement.setAttribute("data-theme", theme);
    localStorage.setItem(STORAGE_KEY_THEME, theme);
    if (themeToggleText) {
      themeToggleText.textContent = theme === "dark" ? "Light mode" : "Dark mode";
    }
  }

  function toggleTheme() {
    const current = document.documentElement.getAttribute("data-theme") || "dark";
    setTheme(current === "dark" ? "light" : "dark");
  }

  // ==========================================================================
  // Conversation & LocalStorage Management
  // ==========================================================================
  function loadConversations() {
    try {
      const raw = localStorage.getItem(STORAGE_KEY_CHATS);
      conversations = raw ? JSON.parse(raw) : [];
    } catch (e) {
      conversations = [];
    }
    renderHistoryList();
  }

  function saveConversations() {
    localStorage.setItem(STORAGE_KEY_CHATS, JSON.stringify(conversations));
    localStorage.setItem(STORAGE_KEY_ACTIVE, currentChatId);
    renderHistoryList();
  }

  function createNewChat() {
    const newId = "chat_" + Date.now();
    const newChat = {
      id: newId,
      title: "New Investigation",
      messages: [],
      createdAt: new Date().toISOString(),
    };
    conversations.unshift(newChat);
    currentChatId = newId;
    saveConversations();
    renderChatMessages();
    chatInput.focus();
  }

  function switchChat(chatId) {
    currentChatId = chatId;
    localStorage.setItem(STORAGE_KEY_ACTIVE, chatId);
    renderHistoryList();
    renderChatMessages();
    // Close sidebar on mobile
    if (window.innerWidth <= 768) {
      sidebar.classList.remove("open");
    }
  }

  function deleteChat(chatId, e) {
    if (e) e.stopPropagation();
    conversations = conversations.filter((c) => c.id !== chatId);
    if (conversations.length === 0) {
      createNewChat();
    } else if (currentChatId === chatId) {
      switchChat(conversations[0].id);
    } else {
      saveConversations();
    }
  }

  function renameChat(chatId, e) {
    if (e) e.stopPropagation();
    const chat = conversations.find((c) => c.id === chatId);
    if (!chat) return;
    const newTitle = prompt("Rename chat title:", chat.title);
    if (newTitle && newTitle.trim()) {
      chat.title = newTitle.trim();
      saveConversations();
    }
  }

  function renderHistoryList() {
    chatHistoryList.innerHTML = "";
    conversations.forEach((chat) => {
      const item = document.createElement("div");
      item.className = "history-item" + (chat.id === currentChatId ? " active" : "");
      item.onclick = () => switchChat(chat.id);

      const title = document.createElement("span");
      title.className = "history-title";
      title.textContent = chat.title;

      const actions = document.createElement("div");
      actions.className = "history-actions";

      const renameBtn = document.createElement("button");
      renameBtn.className = "history-btn";
      renameBtn.title = "Rename";
      renameBtn.innerHTML = `
        <svg viewBox="0 0 24 24" width="13" height="13" fill="none" stroke="currentColor" stroke-width="2">
          <path d="M11 4H4a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h14a2 2 0 0 0 2-2v-7"></path>
          <path d="M18.5 2.5a2.121 2.121 0 0 1 3 3L12 15l-4 1 1-4 9.5-9.5z"></path>
        </svg>
      `;
      renameBtn.onclick = (e) => renameChat(chat.id, e);

      const delBtn = document.createElement("button");
      delBtn.className = "history-btn";
      delBtn.title = "Delete";
      delBtn.innerHTML = `
        <svg viewBox="0 0 24 24" width="13" height="13" fill="none" stroke="currentColor" stroke-width="2">
          <polyline points="3 6 5 6 21 6"></polyline>
          <path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"></path>
        </svg>
      `;
      delBtn.onclick = (e) => deleteChat(chat.id, e);

      actions.appendChild(renameBtn);
      actions.appendChild(delBtn);

      item.appendChild(title);
      item.appendChild(actions);
      chatHistoryList.appendChild(item);
    });
  }

  // ==========================================================================
  // Micro Markdown Renderer (Zero Dependencies)
  // ==========================================================================
  function renderMarkdown(rawText) {
    if (!rawText) return "";

    let lines = rawText.split("\n");
    let out = [];
    let inTable = false;
    let tableHeaders = [];
    let tableRows = [];
    let inList = false;
    let inCodeBlock = false;
    let codeContent = [];

    function flushTable() {
      if (!inTable) return;
      let html = '<div class="table-wrapper"><table><thead><tr>';
      tableHeaders.forEach((h) => {
        html += `<th>${h}</th>`;
      });
      html += "</tr></thead><tbody>";
      tableRows.forEach((row) => {
        html += "<tr>";
        row.forEach((cell) => {
          let badge = cell;
          if (cell.includes("distress") || cell.includes("True") || cell.includes("99.") || cell.includes("95.")) {
            badge = `<span class="badge-distress">${cell}</span>`;
          } else if (cell.includes("safe") || cell.includes("False")) {
            badge = `<span class="badge-safe">${cell}</span>`;
          } else if (cell.includes("grey")) {
            badge = `<span class="badge-grey">${cell}</span>`;
          }
          html += `<td>${badge}</td>`;
        });
        html += "</tr>";
      });
      html += "</tbody></table></div>";
      out.push(html);
      inTable = false;
      tableHeaders = [];
      tableRows = [];
    }

    function flushList() {
      if (inList) {
        out.push("</ul>");
        inList = false;
      }
    }

    for (let i = 0; i < lines.length; i++) {
      let line = lines[i];

      // Code blocks
      if (line.trim().startsWith("```")) {
        if (!inCodeBlock) {
          flushTable();
          flushList();
          inCodeBlock = true;
          codeContent = [];
        } else {
          inCodeBlock = false;
          out.push(`<pre><code>${escapeHtml(codeContent.join("\n"))}</code></pre>`);
        }
        continue;
      }
      if (inCodeBlock) {
        codeContent.push(line);
        continue;
      }

      // Tables
      if (line.trim().startsWith("|") && line.trim().endsWith("|")) {
        flushList();
        let cells = line
          .trim()
          .slice(1, -1)
          .split("|")
          .map((c) => formatInline(c.trim()));

        // Check if separator line (|---|---|)
        if (cells.every((c) => /^[-:\s]+$/.test(c))) {
          continue;
        }

        if (!inTable) {
          inTable = true;
          tableHeaders = cells;
        } else {
          tableRows.push(cells);
        }
        continue;
      } else {
        flushTable();
      }

      // Blockquotes
      if (line.trim().startsWith(">")) {
        flushList();
        let content = line.trim().replace(/^>\s*/, "");
        out.push(`<blockquote>${formatInline(content)}</blockquote>`);
        continue;
      }

      // Headers
      if (line.startsWith("#### ")) {
        flushList();
        out.push(`<h4>${formatInline(line.slice(5))}</h4>`);
        continue;
      }
      if (line.startsWith("### ")) {
        flushList();
        out.push(`<h3>${formatInline(line.slice(4))}</h3>`);
        continue;
      }
      if (line.startsWith("## ")) {
        flushList();
        out.push(`<h2>${formatInline(line.slice(3))}</h2>`);
        continue;
      }
      if (line.startsWith("# ")) {
        flushList();
        out.push(`<h1>${formatInline(line.slice(2))}</h1>`);
        continue;
      }

      // Unordered lists (- or *)
      if (/^[-*]\s+/.test(line.trim())) {
        if (!inList) {
          inList = true;
          out.push("<ul>");
        }
        let item = line.trim().replace(/^[-*]\s+/, "");
        out.push(`<li>${formatInline(item)}</li>`);
        continue;
      } else {
        flushList();
      }

      // Empty lines
      if (!line.trim()) {
        continue;
      }

      // Standard paragraphs
      out.push(`<p>${formatInline(line)}</p>`);
    }

    flushTable();
    flushList();
    return out.join("");
  }

  function formatInline(str) {
    if (!str) return "";
    return str
      .replace(/\*\*(.*?)\*\*/g, "<strong>$1</strong>")
      .replace(/\*(.*?)\*/g, "<em>$1</em>")
      .replace(/`([^`]+)`/g, "<code>$1</code>")
      .replace(/\[(.*?)\]\((.*?)\)/g, '<a href="$2" target="_blank" rel="noopener">$1</a>');
  }

  function escapeHtml(text) {
    return text
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;");
  }

  // ==========================================================================
  // Render Messages in Active Chat
  // ==========================================================================
  function renderChatMessages() {
    messageStream.innerHTML = "";
    const activeChat = conversations.find((c) => c.id === currentChatId);

    if (!activeChat || activeChat.messages.length === 0) {
      emptyState.style.display = "flex";
      messageStream.style.display = "none";
      return;
    }

    emptyState.style.display = "none";
    messageStream.style.display = "flex";

    activeChat.messages.forEach((msg, idx) => {
      appendMessageToDOM(msg, idx === activeChat.messages.length - 1 && msg.role === "assistant");
    });

    scrollToBottom();
  }

  function appendMessageToDOM(msg, isLast) {
    const row = document.createElement("div");
    row.className = "message-row " + (msg.role === "user" ? "user-row" : "assistant-row");

    if (msg.role === "user") {
      const bubble = document.createElement("div");
      bubble.className = "user-bubble";
      bubble.textContent = msg.content;
      row.appendChild(bubble);
    } else {
      const avatar = document.createElement("div");
      avatar.className = "assistant-avatar";
      avatar.textContent = "DW";

      const contentBox = document.createElement("div");
      contentBox.className = "assistant-body";
      contentBox.innerHTML = renderMarkdown(msg.content);

      const actions = document.createElement("div");
      actions.className = "assistant-actions";

      const copyBtn = document.createElement("button");
      copyBtn.className = "msg-action-btn";
      copyBtn.innerHTML = `
        <svg viewBox="0 0 24 24" width="13" height="13" fill="none" stroke="currentColor" stroke-width="2">
          <rect x="9" y="9" width="13" height="13" rx="2" ry="2"></rect>
          <path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"></path>
        </svg>
        <span>Copy</span>
      `;
      copyBtn.onclick = () => {
        navigator.clipboard.writeText(msg.content);
        copyBtn.querySelector("span").textContent = "Copied!";
        setTimeout(() => (copyBtn.querySelector("span").textContent = "Copy"), 2000);
      };
      actions.appendChild(copyBtn);

      if (isLast) {
        const regenBtn = document.createElement("button");
        regenBtn.className = "msg-action-btn";
        regenBtn.innerHTML = `
          <svg viewBox="0 0 24 24" width="13" height="13" fill="none" stroke="currentColor" stroke-width="2">
            <polyline points="1 4 1 10 7 10"></polyline>
            <path d="M3.51 15a9 9 0 1 0 2.13-9.36L1 10"></path>
          </svg>
          <span>Regenerate</span>
        `;
        regenBtn.onclick = regenerateLastMessage;
        actions.appendChild(regenBtn);
      }

      contentBox.appendChild(actions);
      row.appendChild(avatar);
      row.appendChild(contentBox);
    }

    messageStream.appendChild(row);
  }

  function scrollToBottom() {
    messagesContainer.scrollTop = messagesContainer.scrollHeight;
  }

  // ==========================================================================
  // Send Message & Typewriter Animation
  // ==========================================================================
  async function handleSendMessage(promptText) {
    const text = (promptText || chatInput.value).trim();
    if (!text || isGenerating) return;

    chatInput.value = "";
    autoResizeTextarea();
    sendBtn.disabled = true;
    isGenerating = true;

    const activeChat = conversations.find((c) => c.id === currentChatId);
    if (!activeChat) return;

    // Set title on first message
    if (activeChat.messages.length === 0) {
      activeChat.title = text.length > 28 ? text.slice(0, 26) + "..." : text;
    }

    // Append User Message
    activeChat.messages.push({ role: "user", content: text });
    saveConversations();
    renderChatMessages();

    // Placeholder Assistant Message
    const assistantMsg = { role: "assistant", content: "" };
    activeChat.messages.push(assistantMsg);

    // Create DOM row for typewriter effect
    emptyState.style.display = "none";
    messageStream.style.display = "flex";

    const row = document.createElement("div");
    row.className = "message-row assistant-row";
    const avatar = document.createElement("div");
    avatar.className = "assistant-avatar";
    avatar.textContent = "DW";

    const contentBox = document.createElement("div");
    contentBox.className = "assistant-body";
    contentBox.innerHTML = '<span class="typing-cursor">Searching regulatory panel...</span>';

    row.appendChild(avatar);
    row.appendChild(contentBox);
    messageStream.appendChild(row);
    scrollToBottom();

    try {
      const resp = await fetch("/api/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message: text }),
      });

      if (!resp.ok) {
        throw new Error("HTTP error " + resp.status);
      }

      const data = await resp.json();
      const fullText = data.response || "No response received.";

      // Typewriter streaming effect
      await typeWriterEffect(contentBox, fullText, (currentChunk) => {
        assistantMsg.content = currentChunk;
        scrollToBottom();
      });

      assistantMsg.content = fullText;
      saveConversations();
      renderChatMessages();
    } catch (err) {
      const errorText = `**Error retrieving data**: ${err.message}. Ensure backend server is running on port 8000.`;
      assistantMsg.content = errorText;
      contentBox.innerHTML = renderMarkdown(errorText);
      saveConversations();
    } finally {
      isGenerating = false;
      sendBtn.disabled = chatInput.value.trim().length === 0;
      chatInput.focus();
    }
  }

  function typeWriterEffect(targetElement, fullText, onUpdate) {
    return new Promise((resolve) => {
      // Chunk-based typing for fast, fluid rendering
      const words = fullText.split(" ");
      let currentIdx = 0;
      let displayed = "";
      const step = Math.max(1, Math.floor(words.length / 35));

      const interval = setInterval(() => {
        if (currentIdx >= words.length) {
          clearInterval(interval);
          targetElement.innerHTML = renderMarkdown(fullText);
          onUpdate(fullText);
          resolve();
          return;
        }

        const nextWords = words.slice(currentIdx, currentIdx + step);
        displayed += (displayed ? " " : "") + nextWords.join(" ");
        currentIdx += step;

        targetElement.innerHTML = renderMarkdown(displayed);
        onUpdate(displayed);
      }, 18);
    });
  }

  function regenerateLastMessage() {
    const activeChat = conversations.find((c) => c.id === currentChatId);
    if (!activeChat || activeChat.messages.length < 2) return;

    // Pop the assistant message and resend the previous user message
    activeChat.messages.pop();
    const lastUserMsg = activeChat.messages.pop();
    saveConversations();
    handleSendMessage(lastUserMsg.content);
  }

  function autoResizeTextarea() {
    chatInput.style.height = "auto";
    chatInput.style.height = Math.min(chatInput.scrollHeight, 180) + "px";
    sendBtn.disabled = chatInput.value.trim().length === 0;
  }

  // ==========================================================================
  // Event Listeners
  // ==========================================================================
  function attachEventListeners() {
    newChatBtn.onclick = createNewChat;
    themeToggleBtn.onclick = toggleTheme;

    // Textarea input
    chatInput.addEventListener("input", autoResizeTextarea);
    chatInput.addEventListener("keydown", (e) => {
      if (e.key === "Enter" && !e.shiftKey) {
        e.preventDefault();
        handleSendMessage();
      }
    });

    sendBtn.onclick = () => handleSendMessage();

    // Suggestion cards click-to-send
    document.querySelectorAll(".suggestion-card").forEach((card) => {
      card.onclick = () => {
        const prompt = card.getAttribute("data-prompt");
        if (prompt) handleSendMessage(prompt);
      };
    });

    // Mobile sidebar toggling
    if (mobileMenuBtn) {
      mobileMenuBtn.onclick = () => sidebar.classList.add("open");
    }
    if (sidebarCloseBtn) {
      sidebarCloseBtn.onclick = () => sidebar.classList.remove("open");
    }

    // Top action buttons
    if (quickWatchlistBtn) {
      quickWatchlistBtn.onclick = () => handleSendMessage("show watch-list");
    }
    if (quickMetricsBtn) {
      quickMetricsBtn.onclick = () => handleSendMessage("recall and model performance");
    }
  }

  // Run
  document.addEventListener("DOMContentLoaded", init);
})();
