const API_BASE = localStorage.getItem('ragApiBase') || 'http://localhost:8000';
const SESSION_STORAGE_KEY = 'ragSessionId';
document.querySelectorAll('#apiHost').forEach((el) => { el.textContent = API_BASE.replace(/^https?:\/\//, ''); });

function createSessionId() {
  if (window.crypto && typeof window.crypto.randomUUID === 'function') {
    return window.crypto.randomUUID();
  }
  return `session-${Date.now()}-${Math.random().toString(16).slice(2)}`;
}

function getSessionId() {
  let sessionId = localStorage.getItem(SESSION_STORAGE_KEY);
  if (!sessionId) {
    sessionId = createSessionId();
    localStorage.setItem(SESSION_STORAGE_KEY, sessionId);
  }
  return sessionId;
}

async function checkHealth() {
  const textEl = document.querySelector('#healthText');
  const badge = document.querySelector('#healthBadge');
  if (!textEl || !badge) return;
  try {
    const response = await fetch(`${API_BASE}/`);
    if (!response.ok) throw new Error('unhealthy');
    textEl.textContent = 'FastAPI 服务已连接，可以开始提问。';
    badge.textContent = '在线';
  } catch (error) {
    textEl.textContent = '暂时无法连接，请确认后端已启动。';
    badge.textContent = '离线';
    badge.style.color = '#9b6d5d';
  }
}

function initChat() {
  const form = document.querySelector('#questionForm');
  const input = document.querySelector('#questionInput');
  const messages = document.querySelector('#messages');
  const count = document.querySelector('#charCount');
  const button = document.querySelector('#submitButton');
  const suggestions = document.querySelectorAll('.suggestion');
  const sessionId = getSessionId();
  input.addEventListener('input', () => { count.textContent = `${input.value.length} / 1000`; });
  input.addEventListener('keydown', (event) => { if ((event.ctrlKey || event.metaKey) && event.key === 'Enter') form.requestSubmit(); });
  suggestions.forEach((item) => item.addEventListener('click', () => { input.value = item.dataset.question; input.dispatchEvent(new Event('input')); input.focus(); }));
  form.addEventListener('submit', async (event) => {
    event.preventDefault();
    const question = input.value.trim();
    if (!question) return;
    const empty = messages.querySelector('.empty-message');
    if (empty) empty.remove();
    messages.insertAdjacentHTML('beforeend', `<div class="message"><div class="message-question">${escapeHtml(question)}</div><div class="message-answer loading">正在检索知识库并生成回答...</div></div>`);
    const current = messages.lastElementChild.querySelector('.message-answer');
    input.value = ''; input.dispatchEvent(new Event('input')); button.disabled = true; button.style.opacity = '.55'; messages.scrollTop = messages.scrollHeight;
    const abortController = new AbortController();
    let streamFinished = false;
    let answerStarted = false;
    try {
      const response = await fetch(`${API_BASE}/v1/chat/stream`, {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({question, session_id: sessionId}),
        signal: abortController.signal,
      });
      if (!response.ok) throw new Error(`HTTP ${response.status}`);

      if (!response.body) throw new Error('浏览器不支持流式响应');
      await readSse(response, (eventName, data) => {
        if (eventName === 'token') {
          if (!answerStarted) {
            current.classList.remove('loading');
            current.textContent = '';
            answerStarted = true;
          }
          current.textContent += String(data.text || '');
          messages.scrollTop = messages.scrollHeight;
          return;
        }

        if (eventName === 'done') {
          streamFinished = true;
          current.classList.remove('loading');
          if (!answerStarted) current.textContent = data.answer || '接口没有返回回答。';
          appendAnswerMeta(current, data);
          messages.scrollTop = messages.scrollHeight;
          return;
        }

        if (eventName === 'error') {
          throw new Error(data.message || '回答生成失败');
        }
      });

      if (!streamFinished) throw new Error('流式响应未正常结束');
    } catch (error) {
      current.classList.remove('loading');
      current.textContent = `请求失败：${error.message}。请确认 FastAPI 已运行在 ${API_BASE}`;
      current.style.color = '#a56755';
    }
    finally { button.disabled = false; button.style.opacity = '1'; messages.scrollTop = messages.scrollHeight; }
  });
}

async function readSse(response, onEvent) {
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = '';

  const processBuffer = () => {
    buffer = buffer.replace(/\r\n/g, '\n');
    let boundary = buffer.indexOf('\n\n');
    while (boundary !== -1) {
      const block = buffer.slice(0, boundary);
      buffer = buffer.slice(boundary + 2);
      dispatchSseBlock(block, onEvent);
      boundary = buffer.indexOf('\n\n');
    }
  };

  while (true) {
    const {value, done} = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, {stream: true});
    processBuffer();
  }

  buffer += decoder.decode();
  processBuffer();
  if (buffer.trim()) dispatchSseBlock(buffer, onEvent);
}

function dispatchSseBlock(block, onEvent) {
  let eventName = 'message';
  const dataLines = [];

  block.split('\n').forEach((line) => {
    if (line.startsWith(':')) return;
    if (line.startsWith('event:')) {
      eventName = line.slice(6).trim();
    } else if (line.startsWith('data:')) {
      dataLines.push(line.slice(5).trimStart());
    }
  });

  if (dataLines.length === 0) return;
  onEvent(eventName, JSON.parse(dataLines.join('\n')));
}

function appendAnswerMeta(current, data) {
  if (current.nextElementSibling && current.nextElementSibling.classList.contains('answer-meta')) return;
  const hit = String(data.is_included || '').toLowerCase();
  const hitClass = hit.includes('否') || hit.includes('no') ? 'hit-no' : 'hit-yes';
  current.insertAdjacentHTML('afterend', `<div class="answer-meta"><span class="${hitClass}">知识库：${escapeHtml(data.is_included || '未说明')}</span><span>${escapeHtml(data.time || '')}</span></div>`);
}

function escapeHtml(value) { return String(value).replace(/[&<>"']/g, (char) => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#039;'}[char])); }
