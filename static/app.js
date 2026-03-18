let selectedFiles = [];
let activeSource = null;
let currentTab = 'file';
let elapsedTimer = null;
let jobStartTime = null;
let modelStatus = {};
let isMultiFile = false;
let multiFileReports = {};  // file_index -> report
let selectedFileIndex = 0;  // which file is selected in multi-file view

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

function updateMultiFileOptions() {
  const isMulti = selectedFiles.length > 1;
  // Hide page range and batch size for multi-file
  document.getElementById('pages-group').style.display = isMulti ? 'none' : '';
  document.getElementById('batch-size-group').style.display = isMulti ? 'none' : '';
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
    renderChunksList(data.chunks);
    document.getElementById('chunks-section').style.display = '';
  } catch (_) {}
}

async function fetchAndRenderChunksForFile(jobId, fileIndex) {
  try {
    const r = await fetch(`/chunks/${jobId}/${fileIndex}`);
    if (!r.ok) return;
    const data = await r.json();
    _chunksData = data.chunks;
    document.getElementById('chunk-count').textContent = `(${data.count})`;
    const dlBtn = document.getElementById('chunks-download-btn');
    dlBtn.href = `/chunks/${jobId}/${fileIndex}/download`;
    dlBtn.download = `${data.name || 'chunks'}_chunks.json`;
    renderChunksList(data.chunks);
    document.getElementById('chunks-section').style.display = '';
  } catch (_) {}
}

function renderChunksList(chunks) {
  const list = document.getElementById('chunks-list');
  list.innerHTML = '';
  chunks.forEach(c => {
    const card = document.createElement('div');
    card.className = 'chunk-card';
    const headings = c.headings.length ? c.headings.join(' > ') : '';
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

  if (data.gemini_enrich) {
    const gRow = document.createElement('div');
    gRow.className = 't-row';
    gRow.innerHTML = `<span class="t-label">Gemini</span><span class="badge badge-ready">enabled</span>`;
    rows.appendChild(gRow);
  }

  if (data.multi) {
    const fRow = document.createElement('div');
    fRow.className = 't-row';
    fRow.innerHTML = `<span class="t-label">Files</span><span class="t-val">${data.file_count} (${data.doc_concurrency}x parallel)</span>`;
    rows.appendChild(fRow);
  }

  const pgRow = document.createElement('div');
  pgRow.className = 't-row';
  pgRow.id = 'page-progress-row';
  pgRow.style.display = 'none';
  pgRow.innerHTML = `<span class="t-label">Page</span><span class="t-val" id="page-progress">-</span>`;
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
  if (e.dataTransfer.files.length > 0) setFiles(Array.from(e.dataTransfer.files));
});
fileInput.addEventListener('change', () => {
  if (fileInput.files.length > 0) setFiles(Array.from(fileInput.files));
});

function setFiles(newFiles) {
  // Append to existing or replace
  selectedFiles = newFiles.slice(0, 20);
  if (selectedFiles.length === 1) {
    document.getElementById('file-name').textContent = selectedFiles[0].name;
    document.getElementById('file-list').style.display = 'none';
  } else {
    document.getElementById('file-name').textContent = `${selectedFiles.length} files selected`;
    renderFileList();
  }
  updateMultiFileOptions();
}

function renderFileList() {
  const el = document.getElementById('file-list');
  el.innerHTML = '';
  el.style.display = '';
  selectedFiles.forEach((f, i) => {
    const item = document.createElement('div');
    item.className = 'file-list-item';
    const sizeKb = (f.size / 1024).toFixed(0);
    item.innerHTML = `
      <span class="file-list-name">${f.name}</span>
      <span class="file-list-size">${sizeKb} KB</span>
      <button class="file-list-remove" onclick="removeFile(${i})" title="Remove">&times;</button>
    `;
    el.appendChild(item);
  });
}

function removeFile(index) {
  selectedFiles.splice(index, 1);
  if (selectedFiles.length === 0) {
    document.getElementById('file-name').textContent = '';
    document.getElementById('file-list').style.display = 'none';
  } else if (selectedFiles.length === 1) {
    document.getElementById('file-name').textContent = selectedFiles[0].name;
    document.getElementById('file-list').style.display = 'none';
  } else {
    document.getElementById('file-name').textContent = `${selectedFiles.length} files selected`;
    renderFileList();
  }
  updateMultiFileOptions();
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

function downloadPrompts() {
  if (!currentJobId) return;
  const a = document.createElement('a');
  a.href = `/prompts/${currentJobId}/download`;
  a.download = 'prompts.json';
  a.click();
}

function copyOutput() {
  const text = document.getElementById('output-preview').textContent;
  const btn = document.getElementById('copy-btn');
  const done = () => { btn.textContent = 'Copied!'; setTimeout(() => { btn.textContent = 'Copy'; }, 1500); };
  const fail = () => {
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

function appendGeminiLog(msg) {
  const el = document.getElementById('gemini-console');
  if (!el) return;
  const line = document.createElement('div');
  line.textContent = msg;
  el.appendChild(line);
  el.scrollTop = el.scrollHeight;
}

function updateGeminiSection() {
  const checked = document.getElementById('gemini-enrich').checked;
  const wrap = document.getElementById('gemini-console-wrap');
  if (wrap) wrap.style.display = checked ? '' : 'none';
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
  const s = data.stage;
  const filePrefix = data.file_name ? `[${data.file_name}] ` : '';

  if (s === 'pipeline_init') {
    addTimingRow('Pipeline init', data.duration + 's');

  } else if (s === 'conversion_done') {
    _lastPageCount = data.page_count || 0;
    if (filePrefix) {
      addTimingRow(filePrefix + 'Pages', data.page_count || '-', 't-section');
    } else {
      addTimingRow('Pages', data.page_count || '-', 't-section');
    }
    addTimingRow(filePrefix + 'Conversion', data.duration + 's');

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
    if (t.table_structure) {
      addTimingRow('Table structure', '', 't-section');
      addTimingRow('  Total', t.table_structure.total + 's');
      if (t.table_structure.count > 1) addTimingRow('  Avg', t.table_structure.avg + 's');
    }
    if (t.picture_description) {
      addTimingRow('Pic description', '', 't-section');
      addTimingRow('  Total', t.picture_description.total + 's');
      if (t.picture_description.count > 1) addTimingRow('  Avg', t.picture_description.avg + 's');
    }
    if (t.picture_classification) {
      addTimingRow('Pic classify', t.picture_classification.total + 's');
    }
    if (t.code_formula) {
      addTimingRow('Code/Formula', t.code_formula.total + 's');
    }
    if (t.page_init) {
      addTimingRow('VLM page init', '', 't-section');
      addTimingRow('  Total', t.page_init.total + 's');
      if (t.page_init.count > 1) addTimingRow('  Avg/page', t.page_init.avg + 's');
    }
    if (t.doc_assemble) addTimingRow('Assembly', t.doc_assemble.total + 's');
    if (t.doc_build)    addTimingRow('Build', t.doc_build.total + 's');
    if (t.doc_enrich)   addTimingRow('Enrich', t.doc_enrich.total + 's');

  } else if (s === 'report_done') {
    addTimingRow(filePrefix + 'Report', data.duration + 's');

  } else if (s === 'gemini_enrich_done') {
    if (data.error) {
      addTimingRow('Gemini Enrich', 'ERROR: ' + data.error);
    } else {
      addTimingRow('Gemini Enrich', data.duration + 's');
      addTimingRow('  Pictures', data.pictures);
    }

  } else if (s === 'reorder_done') {
    addTimingRow(filePrefix + 'Reorder', data.duration + 's');

  } else if (s === 'merge_done') {
    addTimingRow('Merge batches', data.duration + 's');

  } else if (s === 'export_done') {
    addTimingRow(filePrefix + 'Export', data.duration + 's');

  } else if (s === 'file_write_done') {
    addTimingRow(filePrefix + 'File write', data.duration + 's');
    if (data.size_kb) addTimingRow('  Size', data.size_kb + ' KB');

  } else if (s === 'chunking_done') {
    addTimingRow(filePrefix + 'Chunking', data.duration + 's');
    addTimingRow('  Chunks', data.chunk_count);

  } else if (s === 'total') {
    addTimingRow('Total', data.duration + 's', 't-section t-total');
    stopElapsedTimer();
    document.getElementById('t-elapsed').textContent =
      ((Date.now() - jobStartTime) / 1000).toFixed(1) + 's';
  }
}

// File status table (multi-file)
function initFileStatusTable(filenames) {
  const body = document.getElementById('file-status-body');
  body.innerHTML = '';
  filenames.forEach((name, i) => {
    const tr = document.createElement('tr');
    tr.id = `file-row-${i}`;
    tr.className = 'file-row';
    tr.onclick = () => selectFileRow(i);
    tr.innerHTML = `
      <td>${i + 1}</td>
      <td class="file-row-name">${name}</td>
      <td><span class="badge badge-pending" id="file-badge-${i}">pending</span></td>
      <td id="file-pages-${i}">-</td>
      <td id="file-actions-${i}">-</td>
    `;
    body.appendChild(tr);
  });
  document.getElementById('file-status-section').style.display = '';
  document.getElementById('file-status-count').textContent = `(${filenames.length})`;
}

function updateFileStatus(data) {
  const i = data.file_index;
  const badge = document.getElementById(`file-badge-${i}`);
  if (!badge) return;

  badge.textContent = data.status;
  badge.className = 'badge badge-' + data.status;

  if (data.page_count != null) {
    const pagesEl = document.getElementById(`file-pages-${i}`);
    if (pagesEl) pagesEl.textContent = data.page_count;
  }

  if (data.status === 'done') {
    const actionsEl = document.getElementById(`file-actions-${i}`);
    if (actionsEl) {
      actionsEl.innerHTML = `<a href="/download/${currentJobId}/${i}" class="btn-download btn-sm" download>Download</a>`;
    }
  } else if (data.status === 'error') {
    const actionsEl = document.getElementById(`file-actions-${i}`);
    if (actionsEl) {
      actionsEl.innerHTML = `<span class="err-text" title="${(data.error || '').replace(/"/g, '&quot;')}">failed</span>`;
    }
  }
}

let currentJobId = null;

function selectFileRow(index) {
  selectedFileIndex = index;
  // Highlight selected row
  document.querySelectorAll('.file-row').forEach(r => r.classList.remove('file-row-selected'));
  const row = document.getElementById(`file-row-${index}`);
  if (row) row.classList.add('file-row-selected');

  // Show report for this file
  if (multiFileReports[index]) {
    handleReport(multiFileReports[index]);
  }

  // Load output preview for this file
  if (currentJobId) {
    loadFileResult(currentJobId, index);
  }
}

async function loadFileResult(jobId, index) {
  try {
    const r = await fetch(`/result/${jobId}/${index}`);
    if (!r.ok) return;
    const data = await r.json();
    const preview = document.getElementById('output-preview');
    preview.textContent = data.content;
    document.getElementById('output-section').style.display = '';
    const dl = document.getElementById('download-btn');
    dl.href = `/download/${jobId}/${index}`;
    dl.download = data.name || 'result';

    // Load chunks for this file
    if (document.getElementById('do-chunk').checked) {
      await fetchAndRenderChunksForFile(jobId, index);
    }
  } catch (_) {}
}

// Report table builder
function handleReport(data) {
  // For multi-file reports, data has file_index and report nested
  let reportData = data;
  if (data.file_index !== undefined) {
    multiFileReports[data.file_index] = data;
    // Only render if this is the selected file
    if (data.file_index !== selectedFileIndex) return;
    reportData = data.report || data;
  }

  const body = document.getElementById('report-body');
  body.innerHTML = '';

  function sectionRow(title) {
    const tr = document.createElement('tr');
    tr.className = 'report-section-row';
    tr.innerHTML = `<td colspan="2">${title}</td>`;
    body.appendChild(tr);
  }

  function row(field, value, indent) {
    const tr = document.createElement('tr');
    if (indent) tr.className = 'report-indent';
    tr.innerHTML = `<td>${field}</td><td>${value}</td>`;
    body.appendChild(tr);
  }

  // Overview
  const ov = reportData.overview || {};
  sectionRow('Document');
  if (ov.filename) row('Filename', ov.filename);
  if (ov.mimetype) row('MIME type', ov.mimetype);
  if (ov.pages != null) row('Pages', ov.pages);
  if (ov.page_dimensions) row('Page dimensions', ov.page_dimensions);
  if (ov.pages_with_image) row('Pages with image', ov.pages_with_image);
  if (reportData.total_elements != null) row('Total elements', reportData.total_elements);

  // Elements by label
  const labels = reportData.elements_by_label || {};
  if (Object.keys(labels).length) {
    sectionRow('Elements by type');
    for (const [lbl, cnt] of Object.entries(labels)) {
      row(lbl, cnt, true);
    }
  }

  // Heading levels
  if (reportData.heading_levels) {
    sectionRow('Heading levels');
    for (const [lv, cnt] of Object.entries(reportData.heading_levels)) {
      row(lv, cnt, true);
    }
  }

  // List items
  if (reportData.list_items) {
    sectionRow('List items');
    if (reportData.list_items.enumerated) row('Enumerated', reportData.list_items.enumerated, true);
    if (reportData.list_items.bulleted) row('Bulleted', reportData.list_items.bulleted, true);
  }

  // Tables detail
  if (reportData.tables_detail) {
    const td = reportData.tables_detail;
    sectionRow('Tables');
    row('Total cells', td.total_cells, true);
    if (td.column_header_cells) row('Column header cells', td.column_header_cells, true);
    if (td.row_header_cells) row('Row header cells', td.row_header_cells, true);
    if (td.merged_cells) row('Merged cells', td.merged_cells, true);
    if (td.fillable_cells) row('Fillable cells', td.fillable_cells, true);
    if (td.with_caption) row('With caption', td.with_caption, true);
    if (td.avg_size) row('Avg size', td.avg_size, true);
  }

  // Pictures detail
  if (reportData.pictures_detail) {
    const pd = reportData.pictures_detail;
    sectionRow('Pictures');
    row('With image data', pd.with_image, true);
    if (pd.with_caption) row('With caption', pd.with_caption, true);
    if (pd.with_description) row('With description', pd.with_description, true);
    if (pd.with_classification) row('With classification', pd.with_classification, true);
    if (pd.classification_labels) {
      for (const [cls, cnt] of Object.entries(pd.classification_labels)) {
        row('  ' + cls, cnt, true);
      }
    }
  }

  // Code languages
  if (reportData.code_languages) {
    sectionRow('Code blocks');
    for (const [lang, cnt] of Object.entries(reportData.code_languages)) {
      row(lang, cnt, true);
    }
  }

  // Text formatting
  if (reportData.text_formatting) {
    sectionRow('Text formatting');
    for (const [fmt, cnt] of Object.entries(reportData.text_formatting)) {
      row(fmt, cnt, true);
    }
  }

  // Structure
  const st = reportData.structure || {};
  if (Object.keys(st).length) {
    sectionRow('Structure');
    if (st.with_bbox) row('With bounding box', st.with_bbox, true);
    if (st.with_parent) row('With parent ref', st.with_parent, true);
    if (st.unique_parents != null) row('Unique parents', st.unique_parents, true);
    if (st.content_layers) {
      for (const [cl, cnt] of Object.entries(st.content_layers)) {
        row('Layer: ' + cl, cnt, true);
      }
    }
    if (st.groups_total) row('Groups', st.groups_total, true);
    if (st.groups) {
      for (const [gl, cnt] of Object.entries(st.groups)) {
        row('  ' + gl, cnt, true);
      }
    }
  }

  // Key-value / forms
  if (reportData.key_value_detail) {
    sectionRow('Key-Value regions');
    row('Regions', reportData.key_value_detail.regions, true);
    row('Cells', reportData.key_value_detail.cells, true);
  }
  if (reportData.form_detail) {
    sectionRow('Forms');
    row('Forms', reportData.form_detail.forms, true);
    row('Cells', reportData.form_detail.cells, true);
  }

  // Pages coverage
  if (reportData.pages_coverage) {
    sectionRow('Pages coverage');
    row('Pages with content', reportData.pages_coverage.pages_with_content, true);
    row('Avg elements/page', reportData.pages_coverage.avg_elements_per_page, true);
  }

  document.getElementById('report-section').style.display = '';
}

function copyReport() {
  const table = document.getElementById('report-table');
  const rows = table.querySelectorAll('tr');
  const lines = [];
  rows.forEach(tr => {
    const cells = tr.querySelectorAll('th, td');
    if (cells.length === 1) {
      lines.push('\n' + cells[0].textContent.trim());
    } else if (cells.length === 2) {
      lines.push(cells[0].textContent.trim() + '\t' + cells[1].textContent.trim());
    }
  });
  const text = lines.join('\n').trim();
  const btn = document.getElementById('report-copy-btn');
  navigator.clipboard.writeText(text)
    .then(() => { btn.textContent = 'Copied!'; setTimeout(() => { btn.textContent = 'Copy'; }, 1500); })
    .catch(() => {});
}

// Main convert
async function startConvert() {
  if (activeSource) { activeSource.close(); activeSource = null; }
  clearInterval(elapsedTimer);

  const btn = document.getElementById('convert-btn');
  btn.disabled = true;

  clearConsole();
  const geminiEl = document.getElementById('gemini-console');
  if (geminiEl) geminiEl.innerHTML = '';
  const promptBtn = document.getElementById('download-prompts-btn');
  if (promptBtn) promptBtn.style.display = 'none';
  resetTiming();
  isMultiFile = false;
  multiFileReports = {};
  selectedFileIndex = 0;
  currentJobId = null;
  document.getElementById('job-info-section').style.display = 'none';
  document.getElementById('job-info-rows').innerHTML = '';
  document.getElementById('chunks-section').style.display = 'none';
  document.getElementById('chunks-list').innerHTML = '';
  document.getElementById('report-section').style.display = 'none';
  document.getElementById('report-body').innerHTML = '';
  document.getElementById('file-status-section').style.display = 'none';
  document.getElementById('file-status-body').innerHTML = '';
  document.getElementById('download-all-btn').style.display = 'none';
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
  fd.append('table_mode', document.getElementById('table-mode').value);
  fd.append('accelerator', document.getElementById('accelerator').value);
  fd.append('do_chunk', document.getElementById('do-chunk').checked);
  fd.append('chunk_max_tokens', document.getElementById('chunk-max-tokens').value);
  fd.append('queue_max_size', document.getElementById('queue-max-size').value);
  fd.append('layout_batch_size', document.getElementById('layout-batch-size').value || 0);
  fd.append('table_batch_size', document.getElementById('table-batch-size').value || 0);
  fd.append('ocr_batch_size', document.getElementById('ocr-batch-size').value || 0);
  fd.append('reorder', document.getElementById('reorder').checked);
  fd.append('free_vram', document.getElementById('free-vram').checked);
  fd.append('gemini_enrich', document.getElementById('gemini-enrich').checked);
  fd.append('doc_concurrency', document.getElementById('doc-concurrency').value);
  fd.append('doc_batch_size_setting', document.getElementById('doc-concurrency').value);

  if (currentTab === 'file') {
    if (selectedFiles.length === 0) { appendLog('ERROR: No file selected.', 'err'); btn.disabled = false; return; }
    if (selectedFiles.length === 1) {
      // Single file: use 'file' field for backward compat
      fd.append('file', selectedFiles[0]);
      fd.append('page_from', document.getElementById('page-from').value || 1);
      fd.append('page_to', document.getElementById('page-to').value || 0);
      fd.append('batch_size', document.getElementById('batch-size').value);
    } else {
      // Multi-file: use 'files' field repeated
      isMultiFile = true;
      selectedFiles.forEach(f => fd.append('files', f));
      initFileStatusTable(selectedFiles.map(f => f.name));
    }
  } else {
    const url = document.getElementById('url-input').value.trim();
    if (!url) { appendLog('ERROR: No URL entered.', 'err'); btn.disabled = false; return; }
    fd.append('url', url);
    fd.append('page_from', document.getElementById('page-from').value || 1);
    fd.append('page_to', document.getElementById('page-to').value || 0);
    fd.append('batch_size', document.getElementById('batch-size').value);
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
    currentJobId = jobId;
  } catch (e) {
    appendLog('ERROR: ' + e.message, 'err');
    btn.disabled = false;
    return;
  }

  appendLog(`Job ${jobId} started...`);
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

  es.addEventListener('report', e => {
    try { handleReport(JSON.parse(e.data)); } catch (_) {}
  });

  es.addEventListener('file_status', e => {
    try { updateFileStatus(JSON.parse(e.data)); } catch (_) {}
  });

  es.addEventListener('gemini_log', e => {
    const msg = e.data;
    if (msg === '__PROMPTS_READY__') {
      const btn = document.getElementById('download-prompts-btn');
      if (btn) btn.style.display = '';
    } else {
      appendGeminiLog(msg);
    }
  });

  es.addEventListener('done', async () => {
    es.close(); activeSource = null;
    stopElapsedTimer();

    if (isMultiFile) {
      // Multi-file done
      const dlAll = document.getElementById('download-all-btn');
      dlAll.href = `/download/${jobId}`;
      dlAll.download = 'results.zip';
      dlAll.style.display = '';

      // Auto-select first done file
      const body = document.getElementById('file-status-body');
      const firstDone = body.querySelector('.badge-done');
      if (firstDone) {
        const row = firstDone.closest('tr');
        const idx = parseInt(row.id.replace('file-row-', ''));
        selectFileRow(idx);
      }
      btn.disabled = false;
      return;
    }

    // Single-file done (unchanged)
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
