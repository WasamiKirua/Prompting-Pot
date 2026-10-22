const $ = (id) => document.getElementById(id);
let mode = 'image', targets = [], previewUrl = null, busy = false;
let folderImages = [], existingCaptions = [], downloadUrl = null;
let promptTarget = '', captionTarget = 'Captioning', imageGenerators = [];
const captionImageExtension = /\.(png|jpe?g|webp|bmp)$/i;
function generationLabel() { return mode === 'caption' ? 'Caption my folder ↗' : 'Grow my prompt ↗'; }
function setTheme(theme) {
  document.documentElement.dataset.theme = theme;
  $('theme').textContent = theme === 'dark' ? '☀ Light mode' : '☾ Dark mode';
  $('theme').setAttribute('aria-pressed', String(theme === 'dark'));
  $('theme').setAttribute('aria-label', `Enable ${theme === 'dark' ? 'light' : 'dark'} theme`);
  try { localStorage.setItem('theme', theme); } catch {}
}
let savedTheme;
try { savedTheme = localStorage.getItem('theme'); } catch {}
setTheme(['light', 'dark'].includes(savedTheme) ? savedTheme : (matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light'));
$('theme').onclick = () => setTheme(document.documentElement.dataset.theme === 'dark' ? 'light' : 'dark');
function status(message, error = false) {
  $('status').textContent = message;
  $('status').classList.toggle('error', error);
}
function showProgress(message, processed = null, total = null) {
  $('progress-panel').hidden = false;
  $('progress-label').textContent = message;
  const determinate = processed !== null && total > 0;
  $('progress-bar').classList.toggle('indeterminate', !determinate);
  $('progress-fill').style.width = determinate ? `${100 * processed / total}%` : '35%';
  if (determinate) {
    $('progress-bar').setAttribute('aria-valuemax', String(total));
    $('progress-bar').setAttribute('aria-valuenow', String(processed));
    $('progress-count').textContent = `${processed} / ${total}`;
  } else {
    $('progress-bar').removeAttribute('aria-valuenow');
    $('progress-count').textContent = '';
  }
}
async function readCaptionResult(response) {
  if (!(response.headers.get('Content-Type') || '').includes('application/x-ndjson')) {
    return {archive: await response.blob(), counts: Object.fromEntries(['captioned', 'skipped', 'failed'].map(key => [key, response.headers.get(`X-${key}`)]))};
  }
  const reader = response.body.getReader(), decoder = new TextDecoder();
  let buffer = '', result = null;
  function receive(line) {
    if (!line.trim()) return;
    const event = JSON.parse(line);
    if (event.type === 'stage') showProgress(event.message);
    else if (event.type === 'progress') {
      showProgress(`Processing ${event.filename}`, event.processed, event.total);
      status(`${event.counts.captioned} captioned · ${event.counts.skipped} skipped · ${event.counts.failed} failed`);
    } else if (event.type === 'error') throw new Error(event.message);
    else if (event.type === 'complete') {
      const bytes = Uint8Array.from(atob(event.archive), char => char.charCodeAt(0));
      result = {archive: new Blob([bytes], {type: 'application/zip'}), counts: event.counts};
    }
  }
  try {
    while (true) {
      const {value, done} = await reader.read();
      buffer += done ? decoder.decode() : decoder.decode(value, {stream: true});
      let end;
      while ((end = buffer.indexOf('\n')) !== -1) { receive(buffer.slice(0, end)); buffer = buffer.slice(end + 1); }
      if (done) break;
    }
    if (buffer.trim()) receive(buffer);
    if (!result) throw new Error('The progress connection ended before the ZIP was ready.');
    return result;
  } finally {
    try { await reader.cancel(); } catch {}
    reader.releaseLock();
  }
}
function usesGemmaGenerator() {
  return mode === 'image' && ['krea2', 'zimage', 'flux2-klein', 'anima'].includes($('image-generator').value);
}
function usesTextGenerator() { return mode === 'text' && ['krea2', 'zimage', 'flux2-klein', 'anima', 'qwen-image-2.1', 'moss'].includes($('image-generator').value); }
function usesMossGenerator() { return mode !== 'caption' && $('image-generator').value === 'moss'; }
function isMossCaption() { return $('caption-generator').value === 'moss'; }
function isJsonCaption() { return isIdeogramCaption() || isMossCaption(); }
function usesQwenGenerator() { return (mode === 'image' || mode === 'text') && $('image-generator').value === 'qwen-image-2.1'; }
function usesIdeogramGenerator() { return (mode === 'image' || mode === 'text') && $('image-generator').value === 'ideogram4'; }
function isIdeogramCaption() { return $('caption-generator').value === 'ideogram4'; }
function captionSuffix() { return isJsonCaption() ? '.json' : '.txt'; }
function renderTargetOptions() {
  const gemmaOnly = usesGemmaGenerator(), fiboOnly = usesIdeogramGenerator(), qwenChoice = usesQwenGenerator() || usesTextGenerator() || usesMossGenerator();
  const available = mode === 'caption' ? targets.filter(t => t.captioning && (!isIdeogramCaption() || t.id === 'Captioning')) : gemmaOnly ? targets.filter(t => t.id === 'GemmaUnseen') : fiboOnly ? targets.filter(t => ['Fibo', 'Captioning'].includes(t.id)) : qwenChoice ? targets.filter(t => ['GemmaUnseen', 'Captioning'].includes(t.id)) : mode === 'text' ? targets.filter(t => t.text !== false) : targets;
  const automatic = mode === 'caption' || gemmaOnly || fiboOnly || qwenChoice;
  $('target').replaceChildren();
  if (!automatic || !available.length) $('target').add(new Option(mode === 'caption' ? 'Captioning model not configured' : gemmaOnly ? 'GemmaUnseen model not configured' : fiboOnly ? 'Ideogram model not configured' : qwenChoice ? 'Qwen prompt model not configured' : 'Choose a local model', ''));
  available.forEach(t => $('target').add(new Option(t.name, t.id)));
  $('target').value = mode === 'caption' ? (available.some(t => t.id === captionTarget) ? captionTarget : available.find(t => t.id === 'Captioning')?.id || available[0]?.id || '') : fiboOnly || qwenChoice ? (available.some(t => t.id === promptTarget) ? promptTarget : available.find(t => t.id === (fiboOnly ? 'Fibo' : 'GemmaUnseen'))?.id || available[0]?.id || '') : automatic ? (available[0]?.id || '') : (available.some(t => t.id === promptTarget) ? promptTarget : '');
  if (gemmaOnly || fiboOnly || qwenChoice) promptTarget = $('target').value;
  renderImageGeneratorOptions();
}
function renderImageGeneratorOptions() {
  const target = targets.find(t => t.id === $('target').value);
  const previous = $('image-generator').value;
  const available = imageGenerators;
  $('image-generator').replaceChildren(new Option('Choose an image generator', ''));
  available.forEach(g => $('image-generator').add(new Option(g.name, g.id)));
  $('image-generator').value = mode === 'image' && target?.analysis_only ? (available.some(g => g.id === previous) && target.image_generators?.includes(previous) ? previous : available.some(g => g.id === 'ideogram4') ? 'ideogram4' : '') : (available.some(g => g.id === previous) ? previous : '');
  updateTarget();
}
function updateTarget() {
  const target = targets.find(t => t.id === $('target').value && (mode !== 'caption' || (t.captioning && (!isIdeogramCaption() || t.id === 'Captioning'))) && (!usesGemmaGenerator() || t.id === 'GemmaUnseen') && (!usesIdeogramGenerator() || ['Fibo', 'Captioning'].includes(t.id)) && (!(usesQwenGenerator() || usesTextGenerator() || usesMossGenerator()) || ['GemmaUnseen', 'Captioning'].includes(t.id)));
  $('target').disabled = busy;
  const unsupported = target && (mode === 'text' ? target.text === false : !target.vision);
  const incompatible = mode === 'image' && target?.image_generators && !target.image_generators.includes($('image-generator').value);
  $('changes').disabled = busy || (mode === 'image' && !!target?.analysis_only && !usesMossGenerator());
  $('capability').textContent = mode === 'caption' && isIdeogramCaption() && !targets.some(t => t.id === 'Captioning' && t.captioning)
    ? 'Configure Qwen3.5 Captioning with vision support for Ideogram folder captions.'
    : mode === 'caption' && !targets.some(t => t.captioning)
    ? 'Configure Qwen3.5 Captioning or GemmaUnseen with vision support in config/targets.json to caption images.'
    : usesIdeogramGenerator() && !targets.some(t => ['Fibo', 'Captioning'].includes(t.id)) ? 'Configure Fibo or Qwen3.5 Captioning in config/targets.json to use Ideogram 4.'
    : usesQwenGenerator() && !targets.some(t => ['GemmaUnseen', 'Captioning'].includes(t.id)) ? 'Configure GemmaUnseen or Qwen3.5 Captioning to use Qwen-Image-2.1.'
    : usesGemmaGenerator() && !targets.some(t => t.id === 'GemmaUnseen') ? 'Configure GemmaUnseen in config/targets.json to use this image generator.'
    : mode === 'image' && target?.analysis_only && !usesMossGenerator() ? (target.id === 'Fibo' ? 'Fibo analyzes this image as-is for Ideogram 4 and returns native JSON.' : usesQwenGenerator() ? 'Qwen3.5 describes this image as-is in a standalone Qwen-Image-2.1 paragraph.' : 'Qwen3.5 describes this image as-is using the validated Ideogram 4 JSON schema.') + ' Change instructions are not supported.'
    : unsupported ? 'This model does not support the selected input mode.' : '';
  $('generate').disabled = busy || !target || unsupported || incompatible || (mode !== 'caption' && !$('image-generator').value);
}
function updateFolderSummary() {
  const existing = new Set(existingCaptions.map(name => name.toLowerCase()));
  const skipped = folderImages.filter(file => existing.has(file.name.replace(captionImageExtension, captionSuffix()).toLowerCase())).length;
  $('folder-summary').textContent = `${folderImages.length} top-level images · ${skipped} existing captions to skip. Subfolders are ignored.`;
}
function updateCaptionStyle() {
  const jsonCaption = isJsonCaption();
  $('caption-prose-fields').hidden = jsonCaption;
  $('trigger-word').required = mode === 'caption' && !jsonCaption;
  $('caption-file-help').textContent = jsonCaption
    ? isMossCaption() ? 'MOSS: long/medium/short recaption JSON, watermark boxes and borders. Both models are available. Matching .json captions are skipped.' : 'Ideogram 4: Qwen-only as-is JSON. Matching .json captions are skipped; .txt files do not skip JSON generation.'
    : 'Natural-sentence LoRA captions use .txt files. Matching .txt captions are skipped; .json files do not skip prose generation.';
  $('caption-rules').textContent = $('caption-style').value === 'character'
    ? 'Natural sentences about outfit, pose, lighting and background. Omit the face and fixed identity features; let the trigger carry identity.'
    : 'Natural sentences starting with TRIGGER style depicts, then the actual subject. The trigger labels the aesthetic, never a character. Let the consistent aesthetic across the dataset carry the style.';
  if ($('folder').files.length) updateFolderSummary();
}
$('caption-style').onchange = () => { updateCaptionStyle(); status(''); };
$('caption-generator').onchange = () => { updateCaptionStyle(); renderTargetOptions(); status(''); };
document.querySelectorAll('[data-mode]').forEach(button => {
  button.onclick = () => {
    mode = button.dataset.mode;
    document.querySelectorAll('[data-mode]').forEach(b => b.setAttribute('aria-pressed', String(b === button)));
    $('image-fields').hidden = mode !== 'image';
    $('text-fields').hidden = mode !== 'text';
    $('caption-fields').hidden = mode !== 'caption';
    $('generator-fields').hidden = mode === 'caption';
    updateCaptionStyle();
    $('image').required = mode === 'image';
    $('description').required = mode === 'text';
    $('folder').required = mode === 'caption';
    $('image-generator').required = mode !== 'caption';
    $('target-label').textContent = mode === 'caption' ? 'Captioning model (local, vision-capable)' : 'Prompt-writing model (local)';
    $('generate').textContent = generationLabel();
    status(''); renderTargetOptions();
  };
});
function clearImage() {
  if (previewUrl) URL.revokeObjectURL(previewUrl);
  previewUrl = null; $('image').value = ''; $('preview').removeAttribute('src');
  $('preview').hidden = true; $('upload-hint').hidden = false; $('remove-image').hidden = true;
}
$('remove-image').onclick = clearImage;
$('image').onchange = () => {
  const file = $('image').files[0];
  if (!file) { clearImage(); return; }
  if (!['image/png', 'image/jpeg', 'image/webp'].includes(file.type) || file.size > 10 * 1024 * 1024 || !file.size) {
    clearImage(); status('Choose a PNG, JPEG, or WebP image up to 10 MB.', true); return;
  }
  if (previewUrl) URL.revokeObjectURL(previewUrl);
  previewUrl = URL.createObjectURL(file); $('preview').src = previewUrl;
  $('preview').hidden = false; $('upload-hint').hidden = true; $('remove-image').hidden = false; status('');
};
$('browse-image').onclick = () => { if (!busy) $('image').click(); };
const dropArea = $('image-drop');
['dragenter', 'dragover'].forEach(type => dropArea.addEventListener(type, event => {
  event.preventDefault();
  if (busy) { event.dataTransfer.dropEffect = 'none'; return; }
  event.dataTransfer.dropEffect = 'copy';
  dropArea.classList.add('drag-over');
}));
dropArea.addEventListener('dragleave', event => {
  if (!dropArea.contains(event.relatedTarget)) dropArea.classList.remove('drag-over');
});
dropArea.addEventListener('drop', event => {
  event.preventDefault();
  dropArea.classList.remove('drag-over');
  if (busy) return;
  const files = event.dataTransfer.files;
  if (files.length !== 1) { status('Drop one PNG, JPEG, or WebP image at a time.', true); return; }
  $('image').files = files;
  $('image').onchange();
});
$('preview').onerror = () => { clearImage(); status('This image cannot be previewed. Choose another file.', true); };
$('target').onchange = () => {
  if (mode === 'caption') captionTarget = $('target').value;
  else promptTarget = $('target').value;
  renderImageGeneratorOptions();
  if (usesGemmaGenerator() || usesIdeogramGenerator() || usesQwenGenerator() || usesTextGenerator() || usesMossGenerator()) renderTargetOptions();
};
$('image-generator').onchange = renderTargetOptions;
$('folder').onchange = () => {
  const topLevel = [...$('folder').files].filter(file => file.webkitRelativePath.split('/').length === 2 && !file.name.startsWith('.'));
  folderImages = topLevel.filter(file => captionImageExtension.test(file.name));
  existingCaptions = topLevel.filter(file => /\.(txt|json)$/i.test(file.name)).map(file => file.name);
  updateFolderSummary();
  status('');
};
async function loadTargets() {
  try {
    const [response, generatorResponse] = await Promise.all([fetch('/api/targets'), fetch('/api/image-generators')]);
    if (!response.ok || !generatorResponse.ok) throw new Error('Could not load model choices. Check config/targets.json and reload.');
    targets = await response.json();
    imageGenerators = await generatorResponse.json();
    renderTargetOptions();
    $('target').disabled = false; $('image-generator').disabled = false;
    if (!targets.length) status('Add a model to config/targets.json, then reload.', true);
    updateTarget();
  } catch (error) { status(error.message, true); }
}
$('form').onsubmit = async (event) => {
  event.preventDefault();
  if (busy || !$('form').reportValidity()) return;
  const target = targets.find(t => t.id === $('target').value);
  const description = (mode === 'image' ? (target?.analysis_only && !usesMossGenerator() ? '' : $('changes').value) : $('description').value).trim();
  if (mode === 'text' && !description) { status('Describe the image you want to generate.', true); return; }
  const data = new FormData();
  data.set('target_id', $('target').value);
  if (mode === 'caption') {
    const jsonCaption = isJsonCaption();
    const style = jsonCaption ? $('caption-generator').value : $('caption-style').value;
    const trigger = jsonCaption ? '' : $('trigger-word').value.trim();
    data.set('caption_generator', $('caption-generator').value);
    if (!jsonCaption && !['character', 'style'].includes(style)) { status('Choose Character or Style for prose captioning.', true); return; }
    if (!jsonCaption && (!trigger || /[\r\n]/.test($('trigger-word').value) || $('trigger-word').value.length > 200)) { status('Enter a single-line trigger word up to 200 characters.', true); return; }
    data.set('caption_style', style);
    data.set('trigger_word', trigger);
    if (!folderImages.length || folderImages.length > 250) { status('Choose a folder with 1–250 top-level images.', true); return; }
    if (folderImages.some(file => !file.size || file.size > 25 * 1024 * 1024) || folderImages.reduce((size, file) => size + file.size, 0) > 100 * 1024 * 1024) {
      status('Images must be nonempty, at most 25 MB each and 100 MB total.', true); return;
    }
    const names = folderImages.map(file => file.name.replace(captionImageExtension, captionSuffix()).toLowerCase());
    if (jsonCaption && names.includes('caption-manifest.json')) { status('Rename caption-manifest images to avoid colliding with the ZIP manifest.', true); return; }
    if (new Set(names).size !== names.length) { status('Some images share a caption filename. Rename them before captioning.', true); return; }
    folderImages.forEach(file => data.append('images', file, file.name));
    data.set('existing_captions', JSON.stringify(existingCaptions.filter(name => name.toLowerCase().endsWith(captionSuffix()))));
  } else {
    data.set('mode', mode); data.set('description', description);
    data.set('image_generator', $('image-generator').value);
    if (mode === 'image') data.set('image', $('image').files[0]);
  }
  if (downloadUrl) { URL.revokeObjectURL(downloadUrl); downloadUrl = null; }
  $('download').hidden = true; $('download').removeAttribute('href');
  busy = true;
  $('clear-output').disabled = true;
  document.querySelectorAll('input, select, [data-mode], #changes, #description, #remove-image, #browse-image').forEach(control => { control.disabled = true; });
  updateTarget(); $('generate').textContent = mode === 'caption' ? 'Captioning your folder…' : 'Growing your prompt…';
  $('output-panel')?.setAttribute('aria-busy', 'true');
  showProgress(mode === 'caption' ? 'Uploading your batch…' : 'Waiting for local inference…');
  status(mode === 'caption' ? 'Processing the batch sequentially. Large folders may take a while…' : 'Waiting for your local model…');
  try {
    const response = await fetch(mode === 'caption' ? '/api/caption' : '/api/generate', { method: 'POST', body: data, headers: mode === 'caption' ? {'Accept': 'application/x-ndjson'} : {} });
    if (!response.ok) {
      const result = await response.json();
      throw new Error(typeof result.detail === 'string' ? result.detail : 'Request rejected. Check your inputs and try again.');
    }
    if (mode === 'caption') {
      const result = await readCaptionResult(response);
      downloadUrl = URL.createObjectURL(result.archive);
      $('download').href = downloadUrl; $('download').hidden = false;
      $('download').download = 'captions.zip'; $('download').textContent = 'Download captions.zip ↓';
      $('output').value = `${result.counts.captioned} captioned · ${result.counts.skipped} skipped · ${result.counts.failed} failed\n\nDownload and extract captions.zip into your original image folder.\nSee caption-manifest.json inside the ZIP for per-image results.`;
      $('empty').hidden = true; $('output').hidden = false;
      $('format').hidden = false; $('format').textContent = 'BATCH';
      $('output-title').textContent = 'Your folder captions'; $('copy').textContent = 'Copy summary'; $('copy').disabled = false;
      status('Batch complete. Download your captions below.');
      return;
    }
    const result = await response.json();
    let output = result.output;
    if (typeof output === 'string') { try { output = JSON.parse(output); } catch {} }
    const structured = typeof output !== 'string';
    $('output').value = structured ? JSON.stringify(output, null, 2) : output;
    $('empty').hidden = true; $('output').hidden = false;
    $('format').hidden = false; $('format').textContent = structured ? 'JSON' : 'TEXT';
    $('copy').disabled = !$('output').value; $('copy').textContent = 'Copy prompt';
    $('output-title').textContent = mode === 'text' ? (usesIdeogramGenerator() ? 'Your structured scene prompt' : 'Your fresh prompt') : target?.analysis_only && !usesMossGenerator() ? (target.id === 'Fibo' ? 'Your Fibo scene analysis' : usesQwenGenerator() ? 'Your Qwen-Image-2.1 prompt' : 'Your Ideogram 4 JSON caption') : 'Your fresh prompt';
    if (structured && (usesIdeogramGenerator() || (mode === 'image' && target?.analysis_only && !usesMossGenerator()))) {
      downloadUrl = URL.createObjectURL(new Blob([$('output').value + '\n'], { type: 'application/json' }));
      $('download').href = downloadUrl; $('download').hidden = false;
      $('download').download = target.id === 'Fibo' ? 'fibo-scene.json' : mode === 'text' ? 'ideogram4-prompt.json' : 'ideogram4-caption.json'; $('download').textContent = 'Download scene JSON ↓';
      $('copy').textContent = 'Copy JSON';
    }
    status(mode === 'image' && target?.analysis_only && !usesMossGenerator() ? 'Your scene analysis is ready.' : 'Your prompt is ready.');
  } catch (error) { status(error.message || 'Could not reach the server. Try again.', true); }
  finally {
    busy = false;
    $('progress-panel').hidden = true;
    $('clear-output').disabled = !$('output').value;
    document.querySelectorAll('input, select, [data-mode], #changes, #description, #remove-image, #browse-image').forEach(control => { control.disabled = false; });
    updateTarget(); $('generate').textContent = generationLabel(); $('output-panel')?.removeAttribute('aria-busy');
  }
};
$('clear-output').onclick = () => {
  if (busy) return;
  if (downloadUrl) URL.revokeObjectURL(downloadUrl);
  downloadUrl = null;
  $('download').hidden = true; $('download').removeAttribute('href');
  $('progress-panel').hidden = true;
  $('output').value = ''; $('output').hidden = true; $('empty').hidden = false;
  $('format').hidden = true; $('format').textContent = '';
  $('output-title').textContent = 'Your fresh prompt';
  $('copy').disabled = true; $('copy').textContent = 'Copy prompt';
  $('clear-output').disabled = true;
  status('');
};
$('copy').onclick = async () => {
  try { await navigator.clipboard.writeText($('output').value); status('Output copied.'); }
  catch { $('output').focus(); $('output').select(); status('Clipboard unavailable. The prompt is selected; copy it manually.', true); }
};
loadTargets();
