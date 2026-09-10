const form = document.querySelector("#question-form");
const input = document.querySelector("#question-input");
const sendButton = document.querySelector("#send-button");
const clearButton = document.querySelector("#clear-button");
const messages = document.querySelector("#messages");
const serviceStatus = document.querySelector("#service-status");

function scrollToLatest() {
  messages.scrollTop = messages.scrollHeight;
}

function appendMessage(role, text, result = null) {
  const article = document.createElement("article");
  article.className = `message ${role}-message`;
  if (role === "assistant") {
    const avatar = document.createElement("div");
    avatar.className = "avatar";
    avatar.setAttribute("aria-hidden", "true");
    avatar.textContent = "知";
    article.append(avatar);
  }

  const content = document.createElement("div");
  content.className = "message-content";
  const paragraph = document.createElement("p");
  paragraph.textContent = text;
  content.append(paragraph);

  // Key step: render citations from structured JSON, never from HTML returned by the model.
  if (result?.retrieved_sources?.length) {
    const sources = document.createElement("div");
    sources.className = "sources";
    const title = document.createElement("strong");
    title.textContent = result.citation_valid ? "已验证来源" : "检索来源";
    const list = document.createElement("div");
    list.className = "source-list";
    result.retrieved_sources.forEach((source) => {
      const chip = document.createElement("span");
      chip.className = `source-chip ${result.citations.includes(source) ? "used" : ""}`;
      chip.textContent = source;
      list.append(chip);
    });
    sources.append(title, list);
    content.append(sources);
  }

  article.append(content);
  messages.append(article);
  scrollToLatest();
  return article;
}

function appendLoading() {
  const article = appendMessage("assistant", "");
  const paragraph = article.querySelector("p");
  paragraph.innerHTML = '<span class="loading-dots"><span></span><span></span><span></span></span>';
  return article;
}

async function checkHealth() {
  try {
    const response = await fetch("/health");
    if (!response.ok) throw new Error("unhealthy");
    serviceStatus.textContent = "已连接";
    serviceStatus.className = "status online";
  } catch {
    serviceStatus.textContent = "未连接";
    serviceStatus.className = "status offline";
  }
}

async function submitQuestion(question) {
  appendMessage("user", question);
  const loading = appendLoading();
  sendButton.disabled = true;
  input.disabled = true;
  try {
    // Step 1: call the typed backend API instead of talking to the model directly.
    const response = await fetch("/api/v1/knowledge/ask", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question, top_k: 4 }),
    });
    const result = await response.json();
    loading.remove();
    if (!response.ok) throw new Error(result.message || "请求失败");
    // Step 2: show the answer and citation validation returned by the service.
    appendMessage("assistant", result.answer, result);
  } catch (error) {
    loading.remove();
    const message = appendMessage("assistant", `暂时无法回答：${error.message}`);
    message.classList.add("error-message");
  } finally {
    sendButton.disabled = false;
    input.disabled = false;
    input.focus();
  }
}

form.addEventListener("submit", (event) => {
  event.preventDefault();
  const question = input.value.trim();
  if (!question) return;
  input.value = "";
  input.style.height = "auto";
  submitQuestion(question);
});

input.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault();
    form.requestSubmit();
  }
});

input.addEventListener("input", () => {
  input.style.height = "auto";
  input.style.height = `${Math.min(input.scrollHeight, 140)}px`;
});

clearButton.addEventListener("click", () => {
  messages.replaceChildren();
  appendMessage("assistant", "今天想复习哪个知识点？");
  input.focus();
});

checkHealth();
input.focus();
