let selectedFile = null;
let activeSource = null;
let currentTab = 'file';
let elapsedTimer = null;
let jobStartTime = null;
let modelStatus = {};

const VLM_PRESET_LIST = [
  { key: 'GRANITEDOCLING', label: 'Granite-Docling-258M',  note: '258M · DocTags' },
  { key: 'SMOLDOCLING',    label: 'SmolDocling-256M',      note: '256M · DocTags' },
  { key: 'GRANITE_VISION', label: 'Granite-Vision-3.2-2B', note: '2B · Markdown' },
  { key: 'PHI4',           label: 'Phi-4-multimodal',      note: '14B · Markdown' },
  { key: 'DOLPHIN',        label: 'Dolphin',               note: '~2B · Markdown' },
  { key: 'GOT2',           label: 'GOT-OCR-2.0',           note: '~580M · Markdown' },
];

const PIC_DESC_PRESET_LIST = [
  { key: 'smolvlm',        label: 'SmolVLM-256M',          note: '256M' },
  { key: 'granite_vision', label: 'Granite-Vision-3.3-2B', note: '2B' },
  { key: 'pixtral',        label: 'Pixtral-12B',           note: '12B' },
  { key: 'qwen',           label: 'Qwen2.5-VL-3B',         note: '3B' },
];

async function loadModelStatus() {
  try {
    const r = await fetch('/model-status');
    modelStatus = await r.json();
    renderModelList();
    renderPicDescList();
  } catch (_) {}
}

function _renderRadioList(container, presets, name, statusMap) {
  container.innerHTML = '';
  presets.forEach((p, i) => {
    const item = document.createElement('div');
    item.className = 'model-item';
    const ready = statusMap && statusMap[p.key];
    item.innerHTML = `
      <input type="radio" name="${name}" value="${p.key}" id="${name}-${p.key}" ${i === 0 ? 'checked' : ''}>
      <label for="${name}-${p.key}" class="model-item-label">${p.label} <span class="model-item-note">${p.note}</span></label>
      <span class="badge ${ready ? 'badge-ready' : 'badge-download'}">${ready ? 'ready' : 'download'}</span>
    `;
    container.appendChild(item);
  });
}

function renderModelList() {
  _renderRadioList(
    document.getElementById('vlm-model-list'),
    VLM_PRESET_LIST, 'vlm-model',
    modelStatus.vlm || {}
  );
}

function renderPicDescList() {
  _renderRadioList(
    document.getElementById('pic-desc-model-list'),
    PIC_DESC_PRESET_LIST, 'pic-desc-model',
    modelStatus.pic_desc || {}
  );
}

function updateVlmSection() {
  const pipeline = document.getElementById('pipeline').value;
  document.getElementById('vlm-model-section').style.display = pipeline === 'vlm' ? '' : 'none';
}

function updatePicDescSection() {
  const checked = document.getElementById('do-picture-description').checked;
  document.getElementById('pic-desc-section').style.display = checked ? '' : 'none';
}

function updateChunkSection() {
  const checked = document.getElementById('do-chunk').checked;
  document.getElementById('chunk-tokens-group').style.display = checked ? '' : 'none';
}

let _chunksData = [];
let _lastPageCount = 0;

async function fetchAndRenderChunks(jobId) {
  try {
    const r = await fetch(`/chunks/${jobId}`);
    if (!r.ok) return;
    const data = await r.json();
    _chunksData = data.chunks;
    document.getElementById('chunk-count').textContent = `(${data.count})`;
    const dlBtn = document.getElementById('chunks-download-btn');
    dlBtn.href = `/chunks/${jobId}/download`;
    dlBtn.download = 'chunks.json';
    const list = document.getElementById('chunks-list');
    list.innerHTML = '';
    data.chunks.forEach(c => {
      const card = document.createElement('div');
      card.className = 'chunk-card';
      const headings = c.headings.length ? c.headings.join(' › ') : '';
      card.innerHTML = `
        <div class="chunk-meta">
          <span class="chunk-idx">#${c.index + 1}</span>
          ${c.page != null ? `<span class="chunk-page">p.${c.page}</span>` : ''}
          ${headings ? `<span class="chunk-headings">${headings}</span>` : ''}
        </div>
        <div class="chunk-text">${c.text.replace(/</g, '&lt;')}</div>
      `;
      list.appendChild(card);
    });
    document.getElementById('chunks-section').style.display = '';
  } catch (_) {}
}

function copyChunks() {
  const btn = event.target;
  navigator.clipboard.writeText(JSON.stringify(_chunksData, null, 2))
    .then(() => { btn.textContent = 'Copied!'; setTimeout(() => { btn.textContent = 'Copy JSON'; }, 1500); })
    .catch(() => {});
}

function handleInfo(data) {
  const rows = document.getElementById('job-info-rows');
  rows.innerHTML = '';
  document.getElementById('job-info-section').style.display = '';

  const devRow = document.createElement('div');
  devRow.className = 't-row';
  devRow.innerHTML = `<span class="t-label">Device</span><span class="badge ${data.device === 'cuda' ? 'badge-gpu' : 'badge-cpu'}">${data.device === 'cuda' ? 'GPU' : 'CPU'}</span>`;
  rows.appendChild(devRow);

  if (data.model) {
    const mRow = document.createElement('div');
    mRow.className = 't-row';
    mRow.innerHTML = `<span class="t-label">Model</span><span class="t-val" style="font-size:10px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;max-width:120px">${data.model}</span>`;
    rows.appendChild(mRow);
  }

  if (data.do_picture_description) {
    const pRow = document.createElement('div');
    pRow.className = 't-row';
    const picLabel = data.pic_desc_model || 'on';
    pRow.innerHTML = `<span class="t-label">Pic desc</span><span class="t-val" style="font-size:10px">${picLabel}</span>`;
    rows.appendChild(pRow);
  }

  const pgRow = document.createElement('div');
  pgRow.className = 't-row';
  pgRow.id = 'page-progress-row';
  pgRow.style.display = 'none';
  pgRow.innerHTML = `<span class="t-label">Page</span><span class="t-val" id="page-progress">—</span>`;
  rows.appendChild(pgRow);
}

function updatePageProgress(current, total) {
  const el = document.getElementById('page-progress');
  const row = document.getElementById('page-progress-row');
  if (el && row) {
    el.textContent = `${current} / ${total}`;
    row.style.display = '';
  }
}

// Tab switching
function switchTab(tab) {
  currentTab = tab;
  document.getElementById('input-file').style.display = tab === 'file' ? '' : 'none';
  document.getElementById('input-url').style.display = tab === 'url' ? '' : 'none';
  document.getElementById('tab-file').classList.toggle('active', tab === 'file');
  document.getElementById('tab-url').classList.toggle('active', tab === 'url');
}

// Init
loadModelStatus();

// Drag-and-drop
const dropzone = document.getElementById('dropzone');
const fileInput = document.getElementById('file-input');

dropzone.addEventListener('click', (e) => {
  if (e.target.tagName !== 'LABEL') fileInput.click();
});
dropzone.addEventListener('dragover', e => { e.preventDefault(); dropzone.classList.add('drag-over'); });
dropzone.addEventListener('dragleave', () => dropzone.classList.remove('drag-over'));
dropzone.addEventListener('drop', e => {
  e.preventDefault();
  dropzone.classList.remove('drag-over');
  if (e.dataTransfer.files[0]) setFile(e.dataTransfer.files[0]);
});
fileInput.addEventListener('change', () => { if (fileInput.files[0]) setFile(fileInput.files[0]); });

function setFile(f) {
  selectedFile = f;
  document.getElementById('file-name').textContent = f.name;
}

// Console helpers
function appendLog(text, cls) {
  const el = document.getElementById('log-console');
  const line = document.createElement('span');
  if (cls) line.className = cls;
  line.textContent = text + '\n';
  el.appendChild(line);
  el.scrollTop = el.scrollHeight;
}

function copyOutput() {
  const text = document.getElementById('output-preview').textContent;
  const btn = document.getElementById('copy-btn');
  const done = () => { btn.textContent = 'Copied!'; setTimeout(() => { btn.textContent = 'Copy'; }, 1500); };
  const fail = () => {
    // fallback: select text and execCommand
    const el = document.getElementById('output-preview');
    const sel = window.getSelection();
    const range = document.createRange();
    range.selectNodeContents(el);
    sel.removeAllRanges();
    sel.addRange(range);
    document.execCommand('copy');
    sel.removeAllRanges();
    done();
  };
  if (navigator.clipboard) {
    navigator.clipboard.writeText(text).then(done).catch(fail);
  } else {
    fail();
  }
}

function clearConsole() {
  document.getElementById('log-console').innerHTML = '';
}

// Timing panel helpers
function resetTiming() {
  document.getElementById('timing-rows').innerHTML = '';
  const el = document.getElementById('t-elapsed');
  el.textContent = '0.0s';
  el.className = 't-value';
}

function startElapsedTimer() {
  jobStartTime = Date.now();
  clearInterval(elapsedTimer);
  elapsedTimer = setInterval(() => {
    const secs = ((Date.now() - jobStartTime) / 1000).toFixed(1);
    document.getElementById('t-elapsed').textContent = secs + 's';
  }, 100);
}

function stopElapsedTimer() {
  clearInterval(elapsedTimer);
  elapsedTimer = null;
  document.getElementById('t-elapsed').classList.add('done');
}

function addTimingRow(label, value, cls) {
  const row = document.createElement('div');
  row.className = 't-row' + (cls ? ' ' + cls : '');
  row.innerHTML = `<span class="t-label">${label}</span><span class="t-val">${value}</span>`;
  document.getElementById('timing-rows').appendChild(row);
}

function handleTiming(data) {
  const rows = document.getElementById('timing-rows');
  const s = data.stage;

  if (s === 'pipeline_init') {
    addTimingRow('Pipeline init', data.duration + 's');

  } else if (s === 'conversion_done') {
    _lastPageCount = data.page_count || 0;
    addTimingRow('Pages', data.page_count || '—', 't-section');
    addTimingRow('Conversion', data.duration + 's');

    const t = data.timings || {};

    if (t.layout) {
      addTimingRow('Layout', '', 't-section');
      addTimingRow('  Total', t.layout.total + 's');
      if (t.layout.count > 1) addTimingRow('  Avg/page', t.layout.avg + 's');
    }
    if (t.ocr) {
      addTimingRow('OCR', '', 't-section');
      addTimingRow('  Total', t.ocr.total + 's');
      if (t.ocr.count > 1) addTimingRow('  Avg/page', t.ocr.avg + 's');
    }
    if (t.page_init) {
      addTimingRow('VLM page init', '', 't-section');
      addTimingRow('  Total', t.page_init.total + 's');
      if (t.page_init.count > 1) addTimingRow('  Avg/page', t.page_init.avg + 's');
    }
    if (t.doc_assemble) addTimingRow('Assembly', t.doc_assemble.total + 's');
    if (t.doc_build)    addTimingRow('Build', t.doc_build.total + 's');
    if (t.doc_enrich)   addTimingRow('Enrich', t.doc_enrich.total + 's');

  } else if (s === 'export_done') {
    addTimingRow('Export', data.duration + 's');

  } else if (s === 'chunking_done') {
    addTimingRow('Chunking', data.duration + 's');
    addTimingRow('  Chunks', data.chunk_count);

  } else if (s === 'total') {
    addTimingRow('Total', data.duration + 's', 't-section t-total');
    stopElapsedTimer();
    document.getElementById('t-elapsed').textContent =
      ((Date.now() - jobStartTime) / 1000).toFixed(1) + 's';
  }
}

// Main convert
async function startConvert() {
  if (activeSource) { activeSource.close(); activeSource = null; }
  clearInterval(elapsedTimer);

  const btn = document.getElementById('convert-btn');
  btn.disabled = true;

  clearConsole();
  resetTiming();
  document.getElementById('job-info-section').style.display = 'none';
  document.getElementById('job-info-rows').innerHTML = '';
  document.getElementById('chunks-section').style.display = 'none';
  document.getElementById('chunks-list').innerHTML = '';
  _chunksData = [];
  _lastPageCount = 0;
  document.getElementById('content-row').style.display = 'flex';
  document.getElementById('output-section').style.display = 'none';

  const pipeline = document.getElementById('pipeline').value;
  const ocr = document.getElementById('ocr').checked;
  const format = document.querySelector('input[name="format"]:checked').value;

  const fd = new FormData();
  fd.append('pipeline', pipeline);
  fd.append('ocr', ocr);
  fd.append('format', format);
  fd.append('do_picture_description', document.getElementById('do-picture-description').checked);
  const vlmModelEl = document.querySelector('input[name="vlm-model"]:checked');
  if (vlmModelEl) fd.append('vlm_model', vlmModelEl.value);
  const picDescEl = document.querySelector('input[name="pic-desc-model"]:checked');
  if (picDescEl) fd.append('pic_desc_model', picDescEl.value);
  fd.append('pdf_backend', document.getElementById('pdf-backend').value);
  fd.append('page_from', document.getElementById('page-from').value || 1);
  fd.append('page_to', document.getElementById('page-to').value || 0);
  fd.append('do_chunk', document.getElementById('do-chunk').checked);
  fd.append('chunk_max_tokens', document.getElementById('chunk-max-tokens').value);

  if (currentTab === 'file') {
    if (!selectedFile) { appendLog('ERROR: No file selected.', 'err'); btn.disabled = false; return; }
    fd.append('file', selectedFile);
  } else {
    const url = document.getElementById('url-input').value.trim();
    if (!url) { appendLog('ERROR: No URL entered.', 'err'); btn.disabled = false; return; }
    fd.append('url', url);
  }

  let jobId;
  try {
    const resp = await fetch('/convert', { method: 'POST', body: fd });
    const data = await resp.json();
    if (!resp.ok) {
      appendLog('ERROR: ' + (data.detail || resp.statusText), 'err');
      btn.disabled = false;
      return;
    }
    jobId = data.job_id;
  } catch (e) {
    appendLog('ERROR: ' + e.message, 'err');
    btn.disabled = false;
    return;
  }

  appendLog(`Job ${jobId} started…`);
  startElapsedTimer();

  const es = new EventSource(`/stream/${jobId}`);
  activeSource = es;

  es.addEventListener('log', e => {
    appendLog(e.data);
    const m = e.data.match(/Finished converting pages (\d+)\/(\d+)/);
    if (m) updatePageProgress(parseInt(m[1]), parseInt(m[2]));
  });

  es.addEventListener('timing', e => {
    try { handleTiming(JSON.parse(e.data)); } catch (_) {}
  });

  es.addEventListener('info', e => {
    try { handleInfo(JSON.parse(e.data)); } catch (_) {}
  });

  es.addEventListener('done', async () => {
    es.close(); activeSource = null;
    stopElapsedTimer();
    try {
      const r = await fetch(`/result/${jobId}`);
      const data = await r.json();
      const pageCount = data.page_count || 0;
      const tooLarge = pageCount > 20;
      const preview = document.getElementById('output-preview');
      if (tooLarge) {
        preview.textContent = `Output too large to preview (${pageCount} pages). Use the download button.`;
      } else {
        preview.textContent = data.content;
      }
      const dl = document.getElementById('download-btn');
      dl.href = `/download/${jobId}`;
      dl.download = 'result';
      document.getElementById('output-section').style.display = '';
    } catch (e) {
      appendLog('ERROR: Failed to fetch result: ' + e.message, 'err');
    }
    if (document.getElementById('do-chunk').checked) {
      const pageCount = _lastPageCount;
      if (pageCount > 20) {
        document.getElementById('chunks-section').style.display = '';
        document.getElementById('chunks-list').innerHTML =
          '<div class="chunk-card"><div class="chunk-text">Too many chunks to preview. Use the download button.</div></div>';
        document.getElementById('chunk-count').textContent = '';
        const dlBtn = document.getElementById('chunks-download-btn');
        dlBtn.href = `/chunks/${jobId}/download`;
        dlBtn.download = 'chunks.json';
      } else {
        await fetchAndRenderChunks(jobId);
      }
    }
    btn.disabled = false;
  });

  es.addEventListener('error', e => {
    es.close(); activeSource = null;
    stopElapsedTimer();
    appendLog('ERROR: ' + (e.data || 'Conversion failed.'), 'err');
    btn.disabled = false;
  });
}
