const API_BASE = localStorage.getItem('ragApiBase') || 'http://localhost:8000';
document.querySelectorAll('#apiHost').forEach((el) => { el.textContent = API_BASE.replace(/^https?:\/\//, ''); });

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
    try {
      const response = await fetch(`${API_BASE}/v1`, { method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({question}) });
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      const data = await response.json();
      current.classList.remove('loading'); current.textContent = data.answer || '接口没有返回回答。';
      const hit = String(data.is_included || '').toLowerCase();
      const hitClass = hit.includes('否') || hit.includes('no') ? 'hit-no' : 'hit-yes';
      current.insertAdjacentHTML('afterend', `<div class="answer-meta"><span class="${hitClass}">知识库：${escapeHtml(data.is_included || '未说明')}</span><span>${escapeHtml(data.time || '')}</span></div>`);
    } catch (error) { current.classList.remove('loading'); current.textContent = `请求失败：${error.message}。请确认 FastAPI 已运行在 ${API_BASE}`; current.style.color = '#a56755'; }
    finally { button.disabled = false; button.style.opacity = '1'; messages.scrollTop = messages.scrollHeight; }
  });
}

function escapeHtml(value) { return String(value).replace(/[&<>"']/g, (char) => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#039;'}[char])); }
