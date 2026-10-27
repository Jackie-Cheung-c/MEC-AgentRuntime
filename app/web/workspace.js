const byId = (id) => document.getElementById(id);

const state = {
  view: 'chat', attachment: null, attachments: [], extraction: null,
  evidenceById: new Map(), originalValues: new Map(), dirtyPaths: new Set(),
  zoom: 100, busy: false, threadId: crypto.randomUUID(), localPreviewUrl: null,
  selectedFile: null, threads: [], historyRequestId: 0,
  historyReadyForUser: null, threadOpenRequestId: 0,
  trip: null, memories: [], memoryRequestId: 0,
  user: null, authMode: 'login',
  initialized: false, protectedImageUrl: null,
  pendingRequest: null, pendingUi: null,
  capabilities: null, documentTypes: new Map(), capabilityRequestId: 0,
};

const viewMeta = {
  chat: ['AGENT WORKSPACE', '智能对话'],
  upload: ['ATTACHMENT INTAKE', '附件上传'],
  review: ['HUMAN IN THE LOOP', '证据审核'],
  memory: ['USER MEMORY', '长期记忆'],
};
const statusLabels = { pending: '等待解析', completed: '已完成', needs_review: '待人工确认', failed: '解析失败' };
const THREAD_STORAGE_PREFIX = 'travelops.active_thread_id.';
const AUTH_TOKEN_STORAGE_KEY = 'travelops.auth_token';

function authToken() {
  return localStorage.getItem(AUTH_TOKEN_STORAGE_KEY) || '';
}
function userId() {
  return state.user?.user_id || '';
}
function activeThreadStorageKey(user = userId()) { return `${THREAD_STORAGE_PREFIX}${user}`; }
function rememberActiveThread(threadId) {
  try { localStorage.setItem(activeThreadStorageKey(), threadId); } catch (_) { /* 隐私模式忽略 */ }
}
function forgetActiveThread() {
  try { localStorage.removeItem(activeThreadStorageKey()); } catch (_) { /* 隐私模式忽略 */ }
}
async function api(url, options = {}) {
  const headers = new Headers(options.headers || {});
  const token = authToken();
  if (token) headers.set('Authorization', `Bearer ${token}`);
  const response = await fetch(url, { ...options, headers });
  let data = null;
  try { data = await response.json(); } catch (_) { data = null; }
  if (response.status === 401 && !url.startsWith('/api/auth/')) {
    clearAuthentication();
    const error = new Error(data?.detail || '登录已失效，请重新登录。');
    error.status = response.status;
    throw error;
  }
  if (!response.ok) {
    const error = new Error(responseErrorMessage(data, response.status));
    error.status = response.status;
    throw error;
  }
  return data;
}
function responseErrorMessage(data, status) {
  const detail = data?.detail;
  if (Array.isArray(detail)) {
    const messages = detail
      .map((item) => item?.msg || item?.message)
      .filter(Boolean);
    if (messages.length) return messages.join('；');
  }
  if (typeof detail === 'string' && detail.trim()) return detail;
  if (typeof data?.error === 'string' && data.error.trim()) return data.error;
  return `请求失败 (${status})`;
}
function documentTypeLabel(documentType) {
  return state.documentTypes.get(documentType) || documentType;
}
function setChatPrompt(prompt) {
  if (!prompt) return;
  byId('chat-input').value = prompt;
  byId('chat-input').focus();
}
function renderQuickPrompts(prompts) {
  const grid = byId('quick-prompt-grid');
  grid.replaceChildren();
  grid.hidden = !prompts.length;
  prompts.forEach((item) => {
    const button = document.createElement('button');
    button.type = 'button';
    const title = document.createElement('strong');
    title.textContent = item.title;
    const description = document.createElement('small');
    description.textContent = item.description || '';
    button.append(title);
    if (description.textContent) button.append(description);
    button.addEventListener('click', () => setChatPrompt(item.prompt));
    grid.append(button);
  });
}
function renderToolEntries(entries) {
  const grid = byId('tool-entry-grid');
  grid.replaceChildren();
  grid.hidden = !entries.length;
  entries.forEach((item) => {
    const button = document.createElement('button');
    button.type = 'button';
    button.className = 'tool-entry';
    button.disabled = item.available === false;
    if (item.unavailable_reason) {
      button.title = item.unavailable_reason;
    }
    const title = document.createElement('strong');
    title.textContent = item.title;
    const description = document.createElement('small');
    description.textContent = item.description || '';
    button.append(title);
    if (description.textContent) button.append(description);
    button.addEventListener('click', () => setChatPrompt(item.prompt));
    grid.append(button);
  });
}
function renderDocumentTypeHint(documentTypes) {
  const labels = Array.from(
    new Set(
      documentTypes
        .map((item) => item.label || item.id)
        .filter(Boolean),
    ),
  );
  byId('upload-document-types').textContent = labels.length
    ? `支持：${labels.join('、')}`
    : '将文档拖到这里';
}
function applyCapabilities(payload) {
  state.capabilities = payload || {};
  state.documentTypes = new Map(
    (payload?.document_types || []).map(
      (item) => [item.id, item.label || item.id],
    ),
  );
  renderQuickPrompts(payload?.quick_prompts || []);
  renderToolEntries(payload?.tool_entries || []);
  renderDocumentTypeHint(payload?.document_types || []);
}
async function loadCapabilities() {
  const requestId = ++state.capabilityRequestId;
  try {
    const payload = await api('/api/capabilities');
    if (requestId !== state.capabilityRequestId) return;
    applyCapabilities(payload);
  } catch (error) {
    if (requestId !== state.capabilityRequestId) return;
    showMessage(`能力元数据加载失败：${error.message}`, true);
  }
}
function parseSseFrame(frame) {
  const message = { event: 'message', data: [] };
  frame.split(/\r?\n/).forEach((line) => {
    if (!line || line.startsWith(':')) return;
    const separator = line.indexOf(':');
    const field = separator < 0 ? line : line.slice(0, separator);
    let value = separator < 0 ? '' : line.slice(separator + 1);
    if (value.startsWith(' ')) value = value.slice(1);
    if (field === 'event') message.event = value;
    if (field === 'data') message.data.push(value);
  });
  return message;
}
async function streamAgent(payload, onDelta, onReset) {
  const headers = { 'Content-Type': 'application/json' };
  const token = authToken();
  if (token) headers.Authorization = `Bearer ${token}`;
  const response = await fetch('/api/agent/stream', {
    method: 'POST',
    headers,
    body: JSON.stringify(payload),
  });
  if (!response.ok) {
    let data = null;
    try { data = await response.json(); } catch (_) { data = null; }
    if (response.status === 401) clearAuthentication();
    const error = new Error(responseErrorMessage(data, response.status));
    error.status = response.status;
    throw error;
  }
  if (!response.body) throw new Error('当前浏览器不支持流式响应。');

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = '';
  let result = null;
  let streamError = null;

  const handleFrame = (frame) => {
    const message = parseSseFrame(frame);
    if (!message.data.length) return;
    let data;
    try { data = JSON.parse(message.data.join('\n')); } catch (_) {
      throw new Error('流式响应格式错误。');
    }
    if (message.event === 'delta' && data.text) {
      onDelta(data.text);
    } else if (message.event === 'reset') {
      if (onReset) onReset(data);
    } else if (message.event === 'done') {
      result = data.result;
    } else if (message.event === 'error') {
      streamError = new Error(data.error || data.detail || '流式请求失败。');
      streamError.terminal = data.terminal !== false;
    }
  };

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    let match = /\r?\n\r?\n/.exec(buffer);
    while (match) {
      handleFrame(buffer.slice(0, match.index));
      buffer = buffer.slice(match.index + match[0].length);
      match = /\r?\n\r?\n/.exec(buffer);
    }
  }
  buffer += decoder.decode();
  if (buffer.trim()) handleFrame(buffer);
  if (streamError) throw streamError;
  if (!result) throw new Error('流式响应未返回完整结果。');
  return result;
}
function showMessage(text, error = false) {
  const node = byId('global-message');
  node.textContent = text; node.className = `global-message visible${error ? ' error' : ''}`;
  clearTimeout(showMessage.timer); showMessage.timer = setTimeout(() => node.className = 'global-message', 4500);
}
function setAuthMode(mode) {
  state.authMode = mode;
  const register = mode === 'register';
  byId('auth-name-field').hidden = !register;
  byId('auth-name').required = register;
  byId('auth-title').textContent = register ? '创建 TravelTrace 账号' : '登录 TravelTrace';
  byId('auth-copy').textContent = register
    ? '创建账号后，数据会按账号身份隔离。'
    : '登录后，会话、附件、行程和长期记忆都会绑定到你的账号。';
  byId('auth-submit').textContent = register ? '创建账号' : '登录';
  byId('auth-toggle').textContent = register
    ? '已有账号？返回登录'
    : '没有账号？创建账号';
  showAuthError('');
}
function showAuthError(message) {
  const node = byId('auth-error');
  node.textContent = message;
  node.classList.toggle('visible', Boolean(message));
}
function showAuthGate() {
  byId('auth-gate').hidden = false;
  byId('auth-email').focus();
}
function clearAuthentication() {
  localStorage.removeItem(AUTH_TOKEN_STORAGE_KEY);
  state.user = null;
  state.initialized = false;
  byId('user-display').textContent = '未登录';
  showAuthGate();
}
async function submitAuthentication(event) {
  event.preventDefault();
  const button = byId('auth-submit');
  const email = byId('auth-email').value.trim();
  const password = byId('auth-password').value;
  const displayName = byId('auth-name').value.trim();
  const register = state.authMode === 'register';
  if (register && !displayName) {
    showAuthError('请输入显示名称。');
    return;
  }
  if (password.length < 3) {
    showAuthError('密码至少需要 3 个字符。');
    return;
  }
  button.disabled = true;
  button.textContent = register ? '正在创建...' : '正在登录...';
  showAuthError('');
  try {
    const response = await api(
      register ? '/api/auth/register' : '/api/auth/login',
      {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(
          register
            ? { email, password, display_name: displayName }
            : { email, password },
        ),
      },
    );
    localStorage.setItem(AUTH_TOKEN_STORAGE_KEY, response.access_token);
    state.user = response.user;
    byId('user-display').textContent =
      response.user.display_name || response.user.email;
    byId('auth-password').value = '';
    byId('auth-gate').hidden = true;
    initializeApp();
  } catch (error) {
    showAuthError(error.message);
  } finally {
    button.disabled = false;
    button.textContent = register ? '创建账号' : '登录';
  }
}
async function restoreAuthentication() {
  if (!authToken()) {
    showAuthGate();
    return;
  }
  try {
    const user = await api('/api/auth/me');
    state.user = user;
    byId('user-display').textContent = user.display_name || user.email;
    byId('auth-gate').hidden = true;
    initializeApp();
  } catch (_error) {
    clearAuthentication();
  }
}
function logout() {
  clearAuthentication();
  state.initialized = false;
  state.threads = [];
  state.attachments = [];
  state.memories = [];
  state.pendingRequest = null;
  state.pendingUi = null;
  byId('thread-list').innerHTML = '<p>暂无会话</p>';
  byId('messages').replaceChildren();
  byId('memory-list').innerHTML = '<p class="empty-copy">暂无长期记忆</p>';
}
function switchView(view) {
  state.view = view;
  document.querySelectorAll('[data-view-panel]').forEach((panel) => { panel.hidden = panel.dataset.viewPanel !== view; panel.classList.toggle('active', panel.dataset.viewPanel === view); });
  document.querySelectorAll('.nav-item').forEach((item) => item.classList.toggle('active', item.dataset.view === view));
  [byId('view-kicker').textContent, byId('view-title').textContent] = viewMeta[view];
  if (view === 'review') renderReviewShell();
  if (view === 'memory') loadMemories();
  if (view === 'chat') requestAnimationFrame(scrollChat);
}
function formatBytes(bytes) { return bytes < 1024 ? `${bytes} B` : bytes < 1048576 ? `${(bytes / 1024).toFixed(1)} KiB` : `${(bytes / 1048576).toFixed(1)} MiB`; }
function attachmentUrl(item) { return `/api/attachments/${item.attachment_id}/content`; }
async function loadProtectedDocumentImage(item) {
  const requestAttachmentId = item.attachment_id;
  const headers = new Headers();
  const token = authToken();
  if (token) headers.set('Authorization', `Bearer ${token}`);
  const response = await fetch(attachmentUrl(item), { headers });
  if (response.status === 401) {
    clearAuthentication();
    return;
  }
  if (!response.ok) throw new Error(`图片读取失败 (${response.status})`);
  const content = await response.blob();
  if (state.attachment?.attachment_id !== requestAttachmentId) return;
  if (state.protectedImageUrl) URL.revokeObjectURL(state.protectedImageUrl);
  state.protectedImageUrl = URL.createObjectURL(content);
  byId('document-image').src = state.protectedImageUrl;
}
function setActiveAttachment(item) {
  state.attachment = item; state.extraction = null; state.dirtyPaths.clear();
  const active = byId('active-attachment');
  active.className = 'active-attachment'; active.replaceChildren();
  if (!item) { active.classList.add('empty'); active.textContent = '尚未选择附件'; }
  else {
    const name = document.createElement('strong'); name.textContent = item.filename;
    const detail = document.createElement('small'); detail.textContent = `${item.status} · ${item.attachment_id.slice(0, 8)}`;
    active.append(name, detail);
  }
  byId('composer-attachment').hidden = !item;
  if (item) { byId('composer-file').textContent = item.filename; byId('composer-status').textContent = '将使用已审核的结构化证据'; }
  renderAttachmentLists(); renderReviewShell();
}
function renderAttachmentLists() {
  const recent = byId('recent-list'), library = byId('library-list');
  recent.replaceChildren(); library.replaceChildren(); byId('attachment-count').textContent = String(state.attachments.length);
  if (!state.attachments.length) { recent.innerHTML = '<p>暂无附件</p>'; library.innerHTML = '<p class="empty-copy">暂无附件</p>'; return; }
  state.attachments.forEach((item) => {
    const side = document.createElement('button'); side.type = 'button'; side.className = `recent-item${item.attachment_id === state.attachment?.attachment_id ? ' active' : ''}`;
    const sideName = document.createElement('strong'); sideName.textContent = item.filename;
    const sideMeta = document.createElement('small'); sideMeta.textContent = `${item.status} · ${item.attachment_id.slice(0, 8)}`;
    side.append(sideName, sideMeta); side.addEventListener('click', () => openAttachment(item, 'review')); recent.append(side);
    const row = document.createElement('button'); row.type = 'button'; row.className = `library-item${item.attachment_id === state.attachment?.attachment_id ? ' active' : ''}`;
    const thumb = document.createElement('span'); thumb.className = 'library-thumb'; thumb.textContent = item.media_type.split('/')[1].toUpperCase();
    const copy = document.createElement('span'); copy.className = 'library-copy';
    const name = document.createElement('strong'); name.textContent = item.filename;
    const meta = document.createElement('small'); meta.textContent = `${formatBytes(item.size_bytes)} · ${item.status}`; copy.append(name, meta);
    const action = document.createElement('span'); action.className = 'library-action'; action.textContent = '审核';
    row.append(thumb, copy, action); row.addEventListener('click', () => openAttachment(item, 'review')); library.append(row);
  });
}
async function loadAttachments() {
  if (!state.user) return;
  try {
    state.attachments = await api('/api/attachments?limit=50');
    byId('service-state').classList.remove('error'); byId('service-state').querySelector('span').textContent = '服务已连接'; renderAttachmentLists();
  } catch (error) {
    byId('service-state').classList.add('error'); byId('service-state').querySelector('span').textContent = '附件服务不可用'; showMessage(error.message, true);
  }
}
async function openAttachment(item, view = 'review') { setActiveAttachment(item); switchView(view); await loadLatestExtraction(); }

function pickFile(file) {
  if (!file) return;
  if (!['image/jpeg', 'image/png', 'image/webp'].includes(file.type)) { showMessage('仅支持 JPEG、PNG 和 WebP 图片。', true); return; }
  if (state.localPreviewUrl) URL.revokeObjectURL(state.localPreviewUrl);
  state.selectedFile = file;
  state.localPreviewUrl = URL.createObjectURL(file); byId('local-preview').src = state.localPreviewUrl;
  byId('selected-file-name').textContent = file.name; byId('selected-file-meta').textContent = `${file.type} · ${formatBytes(file.size)}`;
  byId('drop-zone').hidden = true; byId('upload-selection').hidden = false;
}
function resetFile() { byId('file-input').value = ''; byId('drop-zone').hidden = false; byId('upload-selection').hidden = true; if (state.localPreviewUrl) URL.revokeObjectURL(state.localPreviewUrl); state.localPreviewUrl = null; state.selectedFile = null; }
async function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

async function uploadFileWithRetry(file, maxRetries = 3) {
  let lastError = null;

  for (let attempt = 0; attempt <= maxRetries; attempt += 1) {
    try {
      const form = new FormData();
      form.append('user_id', userId());
      form.append('file', file);

      return await api('/api/attachments', {
        method: 'POST',
        body: form,
      });
    } catch (error) {
      lastError = error;

      if (attempt >= maxRetries) {
        throw lastError;
      }

      const delay = Math.min(1000 * (2 ** attempt), 8000);

      byId('upload-button').textContent =
        `网络异常，${Math.ceil(delay / 1000)} 秒后重试 (${attempt + 1}/${maxRetries})`;

      await sleep(delay);
    }
  }

  throw lastError || new Error('上传失败');
}
window.addEventListener('offline', () => {
  showMessage('网络已断开，文件仍保留在当前页面，恢复网络后可继续上传', true);
});

window.addEventListener('online', () => {
  showMessage('网络已恢复，可以重新上传');
  retryPendingRequest();
});
async function uploadFile() {
  const file = state.selectedFile;

  if (!file || state.busy) {
    return;
  }
  if (!navigator.onLine) {
    showMessage('当前没有网络连接，请恢复网络后再上传', true);
    return;
  }
  state.busy = true;
  byId('upload-button').disabled = true;
  byId('upload-button').textContent = '正在上传...';

  try {
    const attachment = await uploadFileWithRetry(file, 3);

    setActiveAttachment(attachment);
    await loadAttachments();
    switchView('review');

    state.busy = false;
    byId('upload-button').disabled = false;
    byId('upload-button').textContent = '上传并解析';

    // 先读取已有解析结果
    const existingExtraction = await loadLatestExtraction();

    if (existingExtraction) {
      // 已经解析过，直接使用数据库中的结果
      showMessage('检测到相同图片，已复用已有解析结果');
    } else {
      // 只有数据库中没有解析结果时，才调用 VLM
      await extractCurrent();
    }

    resetFile();

    showMessage('图片上传成功');
  } catch (error) {
    showMessage(
      `上传失败，请检查网络后点击“重试”：${error.message}`,
      true,
    );

    // 不清除 selectedFile，允许用户手动重试
    byId('drop-zone').hidden = true;
    byId('upload-selection').hidden = false;
  } finally {
    state.busy = false;
    byId('upload-button').disabled = false;
    byId('upload-button').textContent = '上传并解析';
  }
}
function renderReviewShell() {
  const has = Boolean(state.attachment); byId('review-empty').hidden = has; byId('review-workspace').hidden = !has; if (!has) return;
  const item = state.attachment; byId('document-title').textContent = item.filename; byId('meta-filename').textContent = item.filename;
  byId('meta-dimensions').textContent = `${item.width || '-'} × ${item.height || '-'} · ${formatBytes(item.size_bytes)}`; byId('meta-id').textContent = item.attachment_id;
  byId('image-loading').hidden = false; byId('image-loading').textContent = '正在读取图片...'; loadProtectedDocumentImage(item).catch((error) => {
    byId('image-loading').textContent = error.message;
  }); byId('evidence-box').hidden = true;
  if (!state.extraction) resetExtractionView();
}
function resetExtractionView() { byId('document-type').textContent = '-'; byId('field-count').textContent = '0'; byId('review-count').textContent = '0'; byId('raw-confidence').textContent = '-'; byId('raw-text').textContent = '暂无识别文本'; byId('fields-list').innerHTML = '<p class="no-fields">该附件还没有结构化字段</p>'; setDocumentStatus('pending'); byId('save-button').disabled = true; byId('save-hint').textContent = '点击重新解析生成字段'; }
function setDocumentStatus(status) { const badge = byId('document-status'); badge.textContent = statusLabels[status] || status; badge.className = `status-badge ${status === 'completed' ? 'done' : status === 'needs_review' ? 'review' : status === 'failed' ? 'failed' : 'neutral'}`; }
function valueForInput(value) { return value == null ? '' : typeof value === 'object' ? JSON.stringify(value) : String(value); }
function typedValue(input, original) { const value = input.value; if (original === null) return value || null; if (typeof original === 'number') { const number = Number(value); return Number.isFinite(number) ? number : value; } if (typeof original === 'boolean') return value === 'true'; return value; }
function renderExtraction(record) {
  state.extraction = record; state.evidenceById = new Map(record.evidence.map((item) => [item.evidence_id, item])); state.originalValues.clear(); state.dirtyPaths.clear();
  const doc = record.document, reviewCount = doc.fields.filter((field) => field.requires_confirmation).length;
  byId('document-type').textContent = documentTypeLabel(doc.document_type); byId('field-count').textContent = String(doc.fields.length); byId('review-count').textContent = String(reviewCount); byId('raw-text').textContent = doc.raw_text || '暂无识别文本';
  const raw = record.evidence.find((item) => item.metadata?.kind === 'vlm_transcription'); byId('raw-confidence').textContent = raw ? `${Math.round(raw.confidence * 100)}%` : '-'; setDocumentStatus(doc.status);
  const list = byId('fields-list'); list.hidden = false; list.replaceChildren(); if (!doc.fields.length) list.innerHTML = '<p class="no-fields">没有提取到可验证字段</p>';
  doc.fields.forEach((field) => {
    state.originalValues.set(field.path, structuredClone(field.value)); const evidence = state.evidenceById.get(field.evidence_ids[0]);
    const item = document.createElement('article'); item.className = 'field-item'; item.dataset.path = field.path;
    const head = document.createElement('div'); head.className = 'field-head'; const path = document.createElement('span'); path.className = 'field-path'; path.textContent = field.path;
    const confidence = document.createElement('div'); confidence.className = `confidence${field.confidence < .8 ? ' low' : ''}`; const percent = Math.round(field.confidence * 100); confidence.innerHTML = `<span>${percent}%</span><span class="confidence-track"><i style="width:${percent}%"></i></span>`; head.append(path, confidence);
    let input; if (typeof field.value === 'boolean') { input = document.createElement('select'); input.innerHTML = '<option value="true">是</option><option value="false">否</option>'; input.value = String(field.value); } else { input = document.createElement('input'); input.type = typeof field.value === 'number' ? 'number' : 'text'; input.value = valueForInput(field.value); }
    input.className = 'field-input'; input.dataset.path = field.path; input.addEventListener('input', () => markChanged(item, input, field)); input.addEventListener('focus', () => selectEvidence(item, evidence)); item.append(head, input);
    if (evidence) { const quote = document.createElement('p'); quote.className = 'evidence-quote'; quote.textContent = evidence.content; item.append(quote); }
    if (field.requires_confirmation) { const flag = document.createElement('span'); flag.className = 'review-flag'; flag.textContent = '需要确认'; item.append(flag); }
    item.addEventListener('click', () => selectEvidence(item, evidence)); list.append(item);
  });
  byId('save-button').disabled = true; byId('save-hint').textContent = record.corrections.length ? `已保存 ${record.corrections.length} 条纠错记录` : '选择字段可定位证据';
}
function markChanged(item, input, field) { const original = state.originalValues.get(field.path); const changed = JSON.stringify(typedValue(input, original)) !== JSON.stringify(original); item.classList.toggle('changed', changed); changed ? state.dirtyPaths.add(field.path) : state.dirtyPaths.delete(field.path); byId('save-button').disabled = state.busy || !state.dirtyPaths.size; byId('save-hint').textContent = state.dirtyPaths.size ? `${state.dirtyPaths.size} 个字段尚未保存` : '所有更改已保存'; }
function selectEvidence(item, evidence) { document.querySelectorAll('.field-item.active').forEach((node) => node.classList.remove('active')); item.classList.add('active'); const box = evidence?.locator?.bounding_box, overlay = byId('evidence-box'); if (!box) { overlay.hidden = true; return; } overlay.style.left = `${box.x * 100}%`; overlay.style.top = `${box.y * 100}%`; overlay.style.width = `${box.width * 100}%`; overlay.style.height = `${box.height * 100}%`; overlay.hidden = false; }
function setReviewBusy(busy, label = '') { state.busy = busy; byId('rerun-button').disabled = busy; byId('processing-state').hidden = !busy; byId('fields-list').hidden = busy; if (label) byId('save-hint').textContent = label; }
async function extractCurrent(force = false) {
  if (!state.attachment || state.busy) {
    return;
  }

  setReviewBusy(true, 'VLM 正在生成结构化证据');

  try {
    const params = new URLSearchParams();

    if (force) {
      params.set('force', 'true');
    }

    const record = await api(
      `/api/attachments/${state.attachment.attachment_id}/extract?${params}`,
      {
        method: 'POST',
      },
    );

    renderExtraction(record);
    showMessage(
      force
        ? '已重新解析图片'
        : '已加载解析结果',
    );

    await loadAttachments();
  } catch (error) {
    setDocumentStatus('failed');
    showMessage(error.message, true);
  } finally {
    setReviewBusy(false);
  }
}
async function loadLatestExtraction() {
  if (!state.attachment) {
    return null;
  }

  try {
    const record = await api(
      `/api/attachments/${state.attachment.attachment_id}/extraction`
    );

    renderExtraction(record);
    return record;
  } catch (error) {
    if (!/not found|未找到/i.test(error.message)) {
      showMessage(error.message, true);
    }

    return null;
  }
}
async function saveCorrections() {
  if (!state.extraction || !state.dirtyPaths.size || state.busy) return;
  const corrections = [...state.dirtyPaths].map((path) => { const input = document.querySelector(`.field-input[data-path="${CSS.escape(path)}"]`), original = state.originalValues.get(path); return { field_path: path, original_value: original, corrected_value: typedValue(input, original), note: 'human_review' }; });
  setReviewBusy(true, '正在保存人工纠错');
  try {
    const record = await api(`/api/attachments/${state.attachment.attachment_id}/corrections`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ document_id: state.extraction.document.document_id, corrections }) });
    renderExtraction(record);
  } catch (error) {
    showMessage(error.message, true);
    setReviewBusy(false);
    return;
  }
  setReviewBusy(false);
  switchView('chat');
  await sendChat('我已完成当前凭证的人工纠错。请基于已确认的结构化字段重新分析，并说明对行程安排的影响。', true);
}

function addMessage(role, content, pending = false) { byId('chat-empty').hidden = true; const message = document.createElement('article'); message.className = `message ${role}${pending ? ' pending' : ''}`; const avatar = document.createElement('div'); avatar.className = 'message-avatar'; avatar.textContent = role === 'user' ? '你' : 'TT'; const body = document.createElement('div'); body.className = 'message-body'; body.textContent = content; message.append(avatar, body); byId('messages').append(message); scrollChat(); return { message, body }; }
function scrollChat() { const area = byId('chat-scroll'); area.scrollTop = area.scrollHeight; }
function resetChatInput() {
  const input = byId('chat-input');
  input.value = '';
  input.style.height = 'auto';
}
function formatThreadTime(value) {
  if (!value) return '';
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return '';
  return new Intl.DateTimeFormat('zh-CN', {
    month: '2-digit', day: '2-digit', hour: '2-digit', minute: '2-digit',
  }).format(date);
}
function formatMemoryValue(value) {
  if (typeof value === 'string') return value;
  return JSON.stringify(value);
}
function formatMemoryTime(value) {
  if (!value) return '-';
  const date = new Date(value);
  return Number.isNaN(date.getTime())
    ? value
    : new Intl.DateTimeFormat('zh-CN', {
      year: 'numeric', month: '2-digit', day: '2-digit',
      hour: '2-digit', minute: '2-digit',
    }).format(date);
}
function renderMemoryList() {
  const list = byId('memory-list');
  const memories = state.memories || [];
  byId('memory-count').textContent = String(memories.length);
  byId('memory-global-count').textContent = String(
    memories.filter((item) => item.scope === 'global').length
  );
  byId('memory-trip-count').textContent = String(
    memories.filter((item) => item.scope === 'context').length
  );
  byId('memory-thread-count').textContent = String(
    memories.filter((item) => item.scope === 'thread').length
  );
  list.replaceChildren();
  if (!memories.length) {
    list.innerHTML = '<p class="empty-copy">暂无长期记忆</p>';
    return;
  }

  const categoryLabels = {
    profile: '资料', preference: '偏好', constraint: '约束', episodic: '经历',
  };
  memories.forEach((memory) => {
    const row = document.createElement('article');
    row.className = 'memory-item';
    const main = document.createElement('div');
    main.className = 'memory-main';
    const heading = document.createElement('div');
    heading.className = 'memory-heading';
    const category = document.createElement('span');
    category.className = `memory-category ${memory.category}`;
    category.textContent = categoryLabels[memory.category] || memory.category;
    const key = document.createElement('strong');
    key.textContent = memory.memory_key;
    heading.append(category, key);
    const value = document.createElement('p');
    value.className = 'memory-value';
    value.textContent = formatMemoryValue(memory.value);
    const meta = document.createElement('div');
    meta.className = 'memory-meta';
    meta.textContent = [
      memory.domain || 'general',
      memory.scope,
      memory.workflow_id || '',
      memory.scope_id ? `scope ${memory.scope_id.slice(0, 8)}` : '',
      memory.thread_id ? `thread ${memory.thread_id.slice(0, 8)}` : '',
      `置信度 ${Math.round((memory.confidence || 0) * 100)}%`,
      formatMemoryTime(memory.updated_at),
    ].filter(Boolean).join(' · ');
    main.append(heading, value, meta);
    const remove = document.createElement('button');
    remove.type = 'button';
    remove.className = 'memory-delete';
    remove.textContent = '删除';
    remove.addEventListener('click', () => deleteMemory(memory.memory_id));
    row.append(main, remove);
    list.append(row);
  });
}
async function loadMemories() {
  if (!state.user) return;
  const requestId = ++state.memoryRequestId;
  const requestUser = userId();
  try {
    const params = new URLSearchParams({
      status: 'active',
      limit: '500',
    });
    const memories = await api(`/api/memories?${params}`);
    if (requestId !== state.memoryRequestId || requestUser !== userId()) return;
    state.memories = Array.isArray(memories) ? memories : [];
    renderMemoryList();
  } catch (error) {
    if (requestId !== state.memoryRequestId) return;
    showMessage(`长期记忆加载失败：${error.message}`, true);
  }
}
async function deleteMemory(memoryId) {
  if (!window.confirm('删除这条长期记忆？')) return;
  try {
    await api(`/api/memories/${memoryId}`, { method: 'DELETE' });
    showMessage('长期记忆已删除');
    await loadMemories();
  } catch (error) {
    showMessage(error.message, true);
  }
}
async function exportMemories() {
  if (!state.user) return;
  const params = new URLSearchParams();
  const headers = new Headers();
  const token = authToken();
  if (token) headers.set('Authorization', `Bearer ${token}`);
  try {
    const response = await fetch(`/api/memories/export?${params}`, { headers });
    if (response.status === 401) {
      clearAuthentication();
      return;
    }
    if (!response.ok) throw new Error(`导出失败 (${response.status})`);
    const content = await response.blob();
    const url = URL.createObjectURL(content);
    const link = document.createElement('a');
    link.href = url;
    link.download = `memories-${userId()}.json`;
    document.body.append(link);
    link.click();
    link.remove();
    URL.revokeObjectURL(url);
  } catch (error) {
    showMessage(error.message, true);
  }
}
function renderThreadList() {
  const list = byId('thread-list');
  list.replaceChildren();
  if (!state.threads.length) {
    list.innerHTML = '<p>暂无会话</p>';
    return;
  }
  state.threads.forEach((thread) => {
    const row = document.createElement('div');
    row.className = 'thread-row';
    const item = document.createElement('button');
    item.type = 'button';
    item.className = `thread-item${thread.thread_id === state.threadId ? ' active' : ''}`;
    item.dataset.threadId = thread.thread_id;
    const title = document.createElement('strong');
    title.textContent = thread.title || '未命名会话';
    const time = formatThreadTime(thread.updated_at);
    const meta = document.createElement('small');
    meta.textContent = `${thread.turns || 0} 轮${time ? ` · ${time}` : ''}`;
    item.append(title, meta);
    item.addEventListener('click', () => openThread(thread.thread_id));
    const remove = document.createElement('button');
    remove.type = 'button';
    remove.className = 'thread-delete';
    remove.title = '删除会话';
    remove.setAttribute('aria-label', `删除会话 ${thread.title || '未命名会话'}`);
    remove.innerHTML = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.9" stroke-linecap="round" stroke-linejoin="round"><path d="M4 7h16M9 7V4h6v3m-8 0 1 13h8l1-13M10 11v5m4-5v5"/></svg>';
    remove.addEventListener('click', () => deleteThread(thread));
    row.append(item, remove);
    list.append(row);
  });
}
async function deleteThread(thread) {
  if (state.busy) {
    showMessage('请等待当前回答完成后再删除会话。', true);
    return;
  }
  const title = thread.title || '未命名会话';
  if (!window.confirm(`删除会话“${title}”？对应会话记忆也会一并删除。`)) return;
  try {
    await api(
      `/api/threads/${encodeURIComponent(thread.thread_id)}`,
      { method: 'DELETE' },
    );
    state.threads = state.threads.filter(
      (item) => item.thread_id !== thread.thread_id
    );
    if (thread.thread_id === state.threadId) {
      forgetActiveThread();
      state.threadOpenRequestId += 1;
      state.threadId = crypto.randomUUID();
      state.trip = null;
      byId('messages').replaceChildren();
      byId('chat-empty').hidden = false;
    }
    renderThreadList();
    showMessage(`已删除会话“${title}”`);
  } catch (error) {
    showMessage(`会话删除失败：${error.message}`, true);
  }
}
async function loadThreads() {
  if (!state.user) return;
  const requestId = ++state.historyRequestId;
  const requestUser = userId();
  const threadIdBeforeRequest = state.threadId;
  try {
    const threads = await api('/api/threads?limit=50');
    if (requestId !== state.historyRequestId || requestUser !== userId()) return;
    state.threads = Array.isArray(threads) ? threads : [];
    renderThreadList();
    if (
      state.historyReadyForUser !== requestUser
      && state.threadId === threadIdBeforeRequest
      && !state.busy
    ) {
      state.historyReadyForUser = requestUser;
      const savedThreadId = localStorage.getItem(activeThreadStorageKey(requestUser));
      if (savedThreadId && state.threads.some((thread) => thread.thread_id === savedThreadId)) {
        await openThread(savedThreadId, { persist: false });
      } else if (savedThreadId) {
        localStorage.removeItem(activeThreadStorageKey(requestUser));
      }
    }
  } catch (error) {
    if (requestId !== state.historyRequestId) return;
    showMessage(`会话历史加载失败：${error.message}`, true);
  }
}
async function openThread(threadId, options = {}) {
  const thread = state.threads.find((item) => item.thread_id === threadId);
  if (!thread) return;
  if (state.busy) {
    showMessage('请等待当前回答完成后再切换会话。', true);
    return;
  }

  const requestId = ++state.threadOpenRequestId;
  state.threadId = threadId;
  state.trip = thread.trip_id ? { trip_id: thread.trip_id } : null;
  if (options.persist !== false) rememberActiveThread(threadId);
  renderThreadList();
  byId('messages').replaceChildren();
  byId('chat-empty').hidden = true;
  addMessage('assistant', '正在载入会话', true);
  switchView('chat');

  try {
    const turns = await api(`/api/threads/${encodeURIComponent(threadId)}/turns?limit=200`);
    if (requestId !== state.threadOpenRequestId || threadId !== state.threadId) return;
    byId('messages').replaceChildren();
    turns.forEach((turn) => {
      if (turn.query) addMessage('user', turn.query);
      const answer = turn.answer || (turn.status === 'failed' ? '本轮没有生成回答。' : '');
      if (answer) {
        const rendered = addMessage('assistant', answer);
        if (turn.status === 'failed') rendered.message.classList.add('error');
      }
    });
    byId('chat-empty').hidden = turns.length > 0;
    scrollChat();
  } catch (error) {
    if (requestId !== state.threadOpenRequestId) return;
    byId('messages').replaceChildren();
    byId('chat-empty').hidden = false;
    showMessage(`会话读取失败：${error.message}`, true);
  }
}
function pendingRequestStorageKey(user = userId()) {
  return `travelops.pending_request.${user}`;
}
function savePendingRequest(pending) {
  try {
    localStorage.setItem(
      pendingRequestStorageKey(pending.payload.user_id),
      JSON.stringify(pending),
    );
  } catch (_) { /* 隐私模式忽略 */ }
}
function clearPendingRequest() {
  if (state.pendingRequest) {
    localStorage.removeItem(
      pendingRequestStorageKey(state.pendingRequest.payload.user_id),
    );
  }
  state.pendingRequest = null;
  state.pendingUi = null;
}
function loadPendingRequest() {
  try {
    const raw = localStorage.getItem(pendingRequestStorageKey());
    return raw ? JSON.parse(raw) : null;
  } catch (_) {
    return null;
  }
}
function isRetryableRequestError(error) {
  if (error?.terminal) return false;
  if (!navigator.onLine) return true;
  const status = Number(error?.status || 0);
  return !status || status >= 500 || status === 408 || status === 429;
}
async function executePendingRequest(pending, options = {}) {
  if (!state.user || state.busy) return;
  state.pendingRequest = pending;
  if (!state.pendingUi) {
    state.pendingUi = {
      user: addMessage('user', pending.payload.query),
      assistant: addMessage('assistant', '正在恢复任务...', true),
    };
  }
  const { assistant } = state.pendingUi;
  state.busy = true;
  byId('send-button').disabled = true;
  if (navigator.onLine) {
    assistant.message.classList.remove('pending', 'streaming', 'error');
    assistant.message.classList.add('pending');
    assistant.body.textContent = '正在分析';
  } else {
    assistant.message.classList.remove('pending', 'streaming', 'error');
    assistant.body.textContent = '网络已断开，恢复连接后将自动继续。';
    state.busy = false;
    byId('send-button').disabled = false;
    return;
  }
  try {
    let streamedText = '';
    const result = await streamAgent(
      pending.payload,
      (chunk) => {
        if (!streamedText) {
          assistant.message.classList.remove('pending');
          assistant.message.classList.add('streaming');
          assistant.body.textContent = '';
        }
        streamedText += chunk;
        assistant.body.textContent = streamedText;
        scrollChat();
      },
      () => {
        streamedText = '';
        assistant.message.classList.remove('pending');
        assistant.message.classList.add('streaming');
        assistant.body.textContent = '';
        scrollChat();
      },
    );
    if (result.thread_id) state.threadId = result.thread_id;
    if (result.state?.adapter === 'travel_planning') {
        state.trip = result.state.payload;
      } else if (result.trip_state) {
        state.trip = result.trip_state;
      }
    rememberActiveThread(state.threadId);
    assistant.message.classList.remove('pending', 'streaming');
    assistant.body.textContent =
      result.answer || result.error || '本轮没有生成回答。';
    if (result.status === 'failed') assistant.message.classList.add('error');
    clearPendingRequest();
    if (options.automatic) {
      showMessage('纠错已保存，Agent 已使用审核结果重新回答。');
    }
  } catch (error) {
    assistant.message.classList.remove('pending', 'streaming');
    if (error?.status === 401) {
      pending.status = 'waiting';
      pending.last_error = error.message;
      savePendingRequest(pending);
      assistant.body.textContent = '登录已失效，重新登录后会继续该任务。';
      clearAuthentication();
      showMessage('请重新登录，任务已保留。', true);
    } else if (isRetryableRequestError(error)) {
      pending.status = 'waiting';
      pending.last_error = error.message;
      savePendingRequest(pending);
      assistant.body.textContent =
        `网络或服务暂时不可用，将自动重试。${error.message}`;
      showMessage('任务已保留，将在网络恢复后继续。', true);
    } else {
      clearPendingRequest();
      assistant.message.classList.add('error');
      assistant.body.textContent = `请求失败：${error.message}`;
      showMessage(error.message, true);
    }
  } finally {
    state.busy = false;
    byId('send-button').disabled = false;
    scrollChat();
    byId('composer-attachment').hidden = true;
  }
  await loadThreads();
}
async function sendChat(text, automatic = false) {
  if (!state.user) return;
  const query = text.trim();
  if (!query || state.busy) return;
  if (state.pendingRequest) {
    showMessage('上一项任务尚未完成，请等待自动恢复。', true);
    return;
  }
  resetChatInput();
  const requestId = crypto.randomUUID();
  const pending = {
    request_id: requestId,
    status: 'sending',
    created_at: new Date().toISOString(),
    payload: {
      request_id: requestId,
      user_id: userId(),
      thread_id: state.threadId,
      trip_id: state.trip?.trip_id || null,
      query,
      attachments: state.attachment ? [state.attachment] : [],
    },
  };
  savePendingRequest(pending);
  await executePendingRequest(pending, { automatic });
}
async function restorePendingRequest() {
  const pending = loadPendingRequest();
  if (!pending?.payload?.request_id) return;
  state.threadId = pending.payload.thread_id || state.threadId;
  state.pendingRequest = pending;
  state.pendingUi = null;
  switchView('chat');
  await executePendingRequest(pending, { automatic: false });
}
function retryPendingRequest() {
  if (
    !state.pendingRequest
    || state.busy
    || !navigator.onLine
    || !state.user
  ) return;
  executePendingRequest(state.pendingRequest, { automatic: false });
}
function newChat() {
  if (state.busy) {
    showMessage('请等待当前回答完成后再新建会话。', true);
    return;
  }
  clearPendingRequest();
  state.threadOpenRequestId += 1;
  state.threadId = crypto.randomUUID();
  state.trip = null;
  forgetActiveThread();
  byId('messages').replaceChildren();
  byId('chat-empty').hidden = false;
  renderThreadList();
  switchView('chat');
  byId('chat-input').focus();
}
function setZoom(value) { state.zoom = Math.max(50, Math.min(200, value)); byId('zoom-value').textContent = `${state.zoom}%`; byId('image-frame').style.width = `${state.zoom}%`; }
function switchReviewTab(raw) { byId('fields-tab').classList.toggle('active', !raw); byId('raw-tab').classList.toggle('active', raw); byId('fields-panel').hidden = raw; byId('raw-panel').hidden = !raw; }
function initializeApp() {
  if (state.initialized || !state.user) return;
  state.initialized = true;
  state.historyReadyForUser = null;
  state.threadId = crypto.randomUUID();
  state.trip = null;
  setZoom(100);
  Promise.allSettled([loadCapabilities(), loadAttachments(), loadThreads()]).then(() => {
    restorePendingRequest();
  });
}

document.querySelectorAll('.nav-item').forEach((item) => item.addEventListener('click', () => switchView(item.dataset.view)));
document.querySelectorAll('[data-switch]').forEach((item) => item.addEventListener('click', () => switchView(item.dataset.switch)));
byId('new-chat').addEventListener('click', newChat); byId('refresh-list').addEventListener('click', loadAttachments); byId('clear-attachment').addEventListener('click', () => setActiveAttachment(null));
byId('refresh-threads').addEventListener('click', loadThreads);
byId('refresh-memories').addEventListener('click', loadMemories);
byId('export-memories').addEventListener('click', exportMemories);
byId('chat-upload-shortcut').addEventListener('click', () => switchView('upload')); byId('open-review').addEventListener('click', () => switchView('review'));
byId('chat-form').addEventListener('submit', (event) => { event.preventDefault(); sendChat(byId('chat-input').value); });
byId('chat-input').addEventListener('keydown', (event) => { if (event.key === 'Enter' && !event.shiftKey) { event.preventDefault(); byId('chat-form').requestSubmit(); } });
byId('chat-input').addEventListener('input', (event) => { event.target.style.height = 'auto'; event.target.style.height = `${Math.min(event.target.scrollHeight, 150)}px`; });
byId('choose-file').addEventListener('click', () => byId('file-input').click()); byId('file-input').addEventListener('change', () => pickFile(byId('file-input').files[0])); byId('cancel-file').addEventListener('click', resetFile); byId('upload-button').addEventListener('click', uploadFile);
const dropZone = byId('drop-zone'); ['dragenter', 'dragover'].forEach((name) => dropZone.addEventListener(name, (event) => { event.preventDefault(); dropZone.classList.add('dragging'); })); ['dragleave', 'drop'].forEach((name) => dropZone.addEventListener(name, (event) => { event.preventDefault(); dropZone.classList.remove('dragging'); })); dropZone.addEventListener('drop', (event) => { const file = event.dataTransfer.files[0]; if (!file) return; const transfer = new DataTransfer(); transfer.items.add(file); byId('file-input').files = transfer.files; pickFile(file); });
byId('rerun-button').addEventListener('click', extractCurrent); byId('save-button').addEventListener('click', saveCorrections); byId('fields-tab').addEventListener('click', () => switchReviewTab(false)); byId('raw-tab').addEventListener('click', () => switchReviewTab(true));
byId('zoom-in').addEventListener('click', () => setZoom(state.zoom + 25)); byId('zoom-out').addEventListener('click', () => setZoom(state.zoom - 25)); byId('zoom-reset').addEventListener('click', () => setZoom(100));
byId('document-image').addEventListener('load', () => byId('image-loading').hidden = true); byId('document-image').addEventListener('error', () => { byId('image-loading').textContent = '图片读取失败'; });
byId('auth-form').addEventListener('submit', submitAuthentication);
['auth-name', 'auth-email', 'auth-password'].forEach((id) => {
  byId(id).addEventListener('input', () => showAuthError(''));
});
byId('auth-toggle').addEventListener('click', () => {
  setAuthMode(state.authMode === 'login' ? 'register' : 'login');
});
byId('logout-button').addEventListener('click', logout);
setAuthMode('login');
restoreAuthentication();
setZoom(100);

/* ---------- 侧边栏折叠 ---------- */
const appShell = byId('app-shell');
const SIDEBAR_KEY = 'travelops.sidebar_collapsed';

function setSidebarCollapsed(collapsed) {
  appShell.classList.toggle('collapsed', collapsed);
  byId('sidebar-open').hidden = !collapsed;
  try { localStorage.setItem(SIDEBAR_KEY, collapsed ? '1' : '0'); } catch (_) { /* 隐私模式忽略 */ }
}

byId('sidebar-toggle').addEventListener('click', () => setSidebarCollapsed(true));
byId('sidebar-open').addEventListener('click', () => setSidebarCollapsed(false));

setSidebarCollapsed(
  window.matchMedia('(max-width: 1024px)').matches
    ? true
    : localStorage.getItem(SIDEBAR_KEY) === '1'
);
