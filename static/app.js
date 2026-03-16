let selectedFile = null;
let activeSource = null;
let currentTab = 'file';

// Tab switching
function switchTab(tab) {
  currentTab = tab;
  document.getElementById('input-file').style.display = tab === 'file' ? '' : 'none';
  document.getElementById('input-url').style.display = tab === 'url' ? '' : 'none';
  document.getElementById('tab-file').classList.toggle('active', tab === 'file');
  document.getElementById('tab-url').classList.toggle('active', tab === 'url');
}

// Drag-and-drop
const dropzone = document.getElementById('dropzone');
const fileInput = document.getElementById('file-input');

dropzone.addEventListener('click', () => fileInput.click());
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

function clearConsole() {
  document.getElementById('log-console').innerHTML = '';
}

// Main convert
async function startConvert() {
  if (activeSource) { activeSource.close(); activeSource = null; }

  const btn = document.getElementById('convert-btn');
  btn.disabled = true;

  // Show console, hide output
  clearConsole();
  document.getElementById('console-wrap').style.display = '';
  document.getElementById('output-section').style.display = 'none';

  const pipeline = document.getElementById('pipeline').value;
  const ocr = document.getElementById('ocr').checked;
  const format = document.querySelector('input[name="format"]:checked').value;

  const fd = new FormData();
  fd.append('pipeline', pipeline);
  fd.append('ocr', ocr);
  fd.append('format', format);

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

  const es = new EventSource(`/stream/${jobId}`);
  activeSource = es;

  es.addEventListener('log', e => appendLog(e.data));

  es.addEventListener('done', async () => {
    es.close(); activeSource = null;
    try {
      const r = await fetch(`/result/${jobId}`);
      const data = await r.json();
      document.getElementById('output-preview').textContent = data.content;
      const dl = document.getElementById('download-btn');
      dl.href = `/download/${jobId}`;
      document.getElementById('output-section').style.display = '';
    } catch (e) {
      appendLog('ERROR: Failed to fetch result: ' + e.message, 'err');
    }
    btn.disabled = false;
  });

  es.addEventListener('error', e => {
    es.close(); activeSource = null;
    appendLog('ERROR: ' + (e.data || 'Conversion failed.'), 'err');
    btn.disabled = false;
  });
}
