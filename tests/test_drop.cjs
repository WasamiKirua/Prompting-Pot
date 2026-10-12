const { test } = require('node:test');
const assert = require('node:assert/strict');
const vm = require('node:vm');
const fs = require('node:fs');
const path = require('node:path');

function setup(fetchImpl = () => new Promise(() => {})) {
  const elements = new Map();
  function element(id) {
    if (!elements.has(id)) {
      const classes = new Set();
      elements.set(id, {
        files: [], value: '', hidden: false, events: {}, options: [],
        classList: {
          add: name => classes.add(name), remove: name => classes.delete(name),
          contains: name => classes.has(name), toggle: (name, enabled) => enabled ? classes.add(name) : classes.delete(name),
        },
        setAttribute() {}, removeAttribute() {}, contains: () => false, reportValidity: () => true,
        addEventListener(type, callback) { this.events[type] = callback; },
        click() { this.clicked = true; },
        replaceChildren(...children) { this.options = children; this.value = children[0]?.value || ''; },
        add(option) { this.options.push(option); },
      });
    }
    return elements.get(id);
  }
  const modeButtons = ['image', 'text', 'caption'].map(mode => {
    const button = element(`mode-${mode}`);
    button.dataset = { mode };
    return button;
  });
  const context = vm.createContext({
    document: { getElementById: element, documentElement: { dataset: {} }, querySelectorAll: selector => selector === '[data-mode]' ? modeButtons : [] },
    Option: class { constructor(text, value) { this.text = text; this.value = value; } },
    localStorage: { getItem: () => 'light', setItem() {} },
    URL: { createObjectURL: () => 'blob:preview', revokeObjectURL() {} },
    fetch: fetchImpl, FormData, Blob,
  });
  element('caption-style').value = 'character';
  vm.runInContext(fs.readFileSync(path.join(__dirname, '../src/prompting_pot/static/app.js'), 'utf8'), context);
  return { element, context };
}

function event(files = []) {
  return { prevented: false, preventDefault() { this.prevented = true; }, dataTransfer: { files } };
}

for (const format of ['TEXT', 'JSON', 'BATCH']) {
  test(`Clear resets ${format} output and its download without changing inputs`, () => {
    const { element, context } = setup();
    const revoked = [];
    context.URL.revokeObjectURL = url => revoked.push(url);
    vm.runInContext("downloadUrl = 'blob:result'", context);
    element('output').value = 'generated output'; element('output').hidden = false;
    element('empty').hidden = true;
    element('format').textContent = format; element('format').hidden = false;
    element('download').hidden = false;
    element('changes').value = 'keep my inputs';
    element('target').value = 'GemmaUnseen';
    element('image-generator').value = 'krea2';
    element('clear-output').onclick();
    assert.deepEqual(revoked, ['blob:result']);
    assert.equal(element('output').value, '');
    assert.equal(element('output').hidden, true);
    assert.equal(element('empty').hidden, false);
    assert.equal(element('format').hidden, true);
    assert.equal(element('download').hidden, true);
    assert.equal(element('copy').disabled, true);
    assert.equal(element('clear-output').disabled, true);
    assert.equal(element('output-title').textContent, 'Your fresh prompt');
    assert.equal(element('changes').value, 'keep my inputs');
    assert.equal(element('target').value, 'GemmaUnseen');
    assert.equal(element('image-generator').value, 'krea2');
    assert.equal(vm.runInContext('downloadUrl', context), null);
  });
}

test('Clear does not reset output during active generation', () => {
  const { element, context } = setup();
  vm.runInContext('busy = true', context);
  element('output').value = 'previous output';
  element('clear-output').onclick();
  assert.equal(element('output').value, 'previous output');
});

test('image generation sends local model, image-generator target and changes separately', async () => {
  let request;
  const output = '**Prompt Breakdown:**\n- Lighting\n```text\na flower\n```';
  const { element, context } = setup((url, options) => {
    if (url !== '/api/generate') return new Promise(() => {});
    request = options;
    return Promise.resolve({ ok: true, json: async () => ({ output }) });
  });
  vm.runInContext("targets = [{id: 'GemmaUnseen', vision: true}]", context);
  element('target').value = 'GemmaUnseen';
  element('image-generator').value = 'qwen-image-2.1';
  element('changes').value = '  make the background blue  ';
  element('image').files = [new Blob(['image'], { type: 'image/png' })];
  await element('form').onsubmit(event());
  assert.equal(request.body.get('target_id'), 'GemmaUnseen');
  assert.equal(request.body.get('image_generator'), 'qwen-image-2.1');
  assert.equal(request.body.get('description'), 'make the background blue');
  assert.equal(request.body.get('image').type, 'image/png');
  assert.equal(element('output').value, output);
  assert.equal(element('clear-output').disabled, false);
});

test('caption mode offers both models, defaults to Qwen and remembers Gemma independently', () => {
  const { element, context } = setup();
  vm.runInContext("targets = [{id: 'GemmaUnseen', name: 'Gemma Unseen', vision: true, captioning: true}, {id: 'Captioning', name: 'Qwen3.5 Captioning (NSFW)', vision: true, captioning: true}]", context);
  element('target').value = 'GemmaUnseen';
  element('target').onchange();
  element('mode-caption').onclick();
  assert.deepEqual(element('target').options.map(option => option.value), ['GemmaUnseen', 'Captioning']);
  assert.equal(element('target').value, 'Captioning');
  assert.equal(element('target').disabled, false);
  element('target').value = 'GemmaUnseen'; element('target').onchange();
  element('mode-image').onclick();
  assert.equal(element('target').value, 'GemmaUnseen');
  assert.ok(element('target').options.some(option => option.value === 'GemmaUnseen'));
  element('mode-caption').onclick();
  assert.equal(element('target').value, 'GemmaUnseen');
});

test('caption mode offers Gemma when Qwen is not configured', () => {
  const { element, context } = setup();
  vm.runInContext("targets = [{id: 'GemmaUnseen', name: 'Gemma Unseen', vision: true, captioning: true}]", context);
  element('mode-caption').onclick();
  assert.deepEqual(element('target').options.map(option => option.value), ['GemmaUnseen']);
  assert.equal(element('target').value, 'GemmaUnseen');
  assert.equal(element('target').disabled, false);
  assert.equal(element('generate').disabled, false);
});

test('caption mode disables generation when neither caption model is eligible', () => {
  const { element, context } = setup();
  vm.runInContext("targets = [{id: 'GemmaUnseen', name: 'Gemma Unseen', vision: true, captioning: false}]", context);
  element('mode-caption').onclick();
  assert.equal(element('target').value, '');
  assert.deepEqual(element('target').options.map(option => option.value), ['']);
  assert.equal(element('generate').disabled, true);
  assert.match(element('capability').textContent, /Configure Qwen3.5 Captioning or GemmaUnseen/);
});

for (const generator of ['krea2', 'zimage', 'flux2-klein', 'anima']) {
  test(`${generator} shows and automatically selects only GemmaUnseen`, () => {
    const { element, context } = setup();
    vm.runInContext(`targets = [{id: 'GemmaUnseen', name: 'Gemma Unseen Prompt From Img (NSFW)', vision: true}, {id: 'Captioning', vision: true, captioning: true, analysis_only: true, image_generators: ['ideogram4']}, {id: 'Fibo', vision: true, analysis_only: true, image_generators: ['ideogram4']}]; imageGenerators = [{id: '${generator}', name: '${generator}'}, {id: 'ideogram4', name: 'Ideogram 4'}]`, context);
    element('target').value = 'Captioning';
    element('target').onchange();
    element('image-generator').value = generator;
    element('image-generator').onchange();
    assert.deepEqual(element('target').options.map(option => option.value), ['GemmaUnseen']);
    assert.equal(element('target').value, 'GemmaUnseen');
    assert.equal(element('image-generator').value, generator);
    assert.equal(element('generate').disabled, false);
    element('mode-caption').onclick();
    assert.deepEqual(element('target').options.map(option => option.value), ['Captioning']);
    element('mode-image').onclick();
    assert.deepEqual(element('target').options.map(option => option.value), ['GemmaUnseen']);
  });
}

test('Ideogram remains visible with Gemma and missing Fibo never falls back', () => {
  const { element, context } = setup();
  vm.runInContext("targets = [{id: 'GemmaUnseen', vision: true}]; imageGenerators = [{id: 'krea2', name: 'Krea2'}, {id: 'ideogram4', name: 'Ideogram 4'}]", context);
  element('image-generator').value = 'krea2';
  element('image-generator').onchange();
  assert.ok(element('image-generator').options.some(option => option.value === 'ideogram4'));
  element('image-generator').value = 'ideogram4';
  element('image-generator').onchange();
  assert.equal(element('target').value, '');
  assert.equal(element('image-generator').value, 'ideogram4');
  assert.equal(element('generate').disabled, true);
  assert.match(element('capability').textContent, /Configure Fibo/);
});

test('missing GemmaUnseen disables generation without choosing another model', () => {
  const { element, context } = setup();
  vm.runInContext("targets = [{id: 'Captioning', vision: true}]; imageGenerators = [{id: 'krea2', name: 'Krea2'}]", context);
  element('image-generator').value = 'krea2';
  element('image-generator').onchange();
  assert.equal(element('target').value, '');
  assert.deepEqual(element('target').options.map(option => option.value), ['']);
  assert.equal(element('generate').disabled, true);
  assert.match(element('capability').textContent, /Configure GemmaUnseen/);
});

test('clearing the generator restores model choices and selecting Ideogram restricts models to Fibo and Qwen', () => {
  const { element, context } = setup();
  vm.runInContext("targets = [{id: 'GemmaUnseen', vision: true}, {id: 'Captioning', vision: true}, {id: 'Fibo', vision: true, analysis_only: true, text: false, image_generators: ['ideogram4']}]; imageGenerators = [{id: 'krea2', name: 'Krea2'}, {id: 'ideogram4', name: 'Ideogram 4'}]", context);
  element('image-generator').value = 'krea2';
  element('image-generator').onchange();
  element('image-generator').value = '';
  element('image-generator').onchange();
  assert.ok(element('target').options.some(option => option.value === 'Fibo'));
  element('target').value = 'Fibo';
  element('target').onchange();
  assert.equal(element('image-generator').value, 'ideogram4');
  element('image-generator').onchange();
  assert.equal(element('target').value, 'Fibo');
  assert.deepEqual(element('target').options.map(option => option.value), ['Captioning', 'Fibo']);
});

test('Fibo supports Ideogram image/text workflows but stays absent from caption mode', () => {
  const { element, context } = setup();
  vm.runInContext("targets = [{id: 'Fibo', vision: true, text: true, analysis_only: true, captioning: false, image_generators: ['ideogram4']}, {id: 'GemmaUnseen', vision: true, text: true, image_generators: ['krea2']}, {id: 'Captioning', vision: true, captioning: true}]; imageGenerators = [{id: 'ideogram4', name: 'Ideogram 4'}, {id: 'krea2', name: 'Krea2'}]", context);
  element('target').value = 'Fibo';
  element('target').onchange();
  assert.equal(element('image-generator').value, 'ideogram4');
  assert.deepEqual(element('image-generator').options.map(option => option.value), ['', 'ideogram4', 'krea2']);
  assert.equal(element('changes').disabled, true);
  assert.match(element('capability').textContent, /as-is/);
  element('mode-text').onclick();
  assert.ok(element('target').options.some(option => option.value === 'Fibo'));
  assert.equal(element('generator-fields').hidden, false);
  assert.equal(element('image').required, false);
  element('mode-caption').onclick();
  assert.deepEqual(element('target').options.map(option => option.value), ['Captioning']);
  element('mode-image').onclick();
  assert.equal(element('target').value, 'Fibo');
  element('image-generator').value = 'krea2';
  element('image-generator').onchange();
  assert.equal(element('target').value, 'GemmaUnseen');
  assert.equal(element('changes').disabled, false);
  assert.equal(element('image-generator').value, 'krea2');
  assert.ok(element('image-generator').options.some(option => option.value === 'ideogram4'));
});

test('Ideogram defaults to Fibo and allows Qwen JSON without exposing Gemma', async () => {
  const native = { high_level_description: 'A scene', style_description: {}, compositional_deconstruction: {} };
  let request;
  const { element, context } = setup((url, options) => {
    if (url !== '/api/generate') return new Promise(() => {});
    request = options;
    return Promise.resolve({ ok: true, json: async () => ({ output: JSON.stringify(native) }) });
  });
  vm.runInContext("targets = [{id: 'Captioning', vision: true, captioning: true, analysis_only: true, image_generators: ['ideogram4']}, {id: 'GemmaUnseen', vision: true}, {id: 'Fibo', vision: true, analysis_only: true, image_generators: ['ideogram4']}]; imageGenerators = [{id: 'ideogram4', name: 'Ideogram 4'}, {id: 'krea2', name: 'Krea2'}]", context);
  element('image-generator').value = 'ideogram4';
  element('image-generator').onchange();
  assert.equal(element('target').value, 'Fibo');
  assert.deepEqual(element('target').options.map(option => option.value), ['Captioning', 'Fibo']);
  element('target').value = 'Captioning';
  element('target').onchange();
  assert.equal(element('target').value, 'Captioning');
  assert.equal(element('changes').disabled, true);
  element('changes').value = 'stale changes';
  element('image').files = [new Blob(['image'], {type: 'image/png'})];
  await element('form').onsubmit(event());
  assert.equal(request.body.get('target_id'), 'Captioning');
  assert.equal(request.body.get('description'), '');
  assert.equal(element('download').download, 'ideogram4-caption.json');
  assert.deepEqual(JSON.parse(element('output').value), native);
  element('mode-caption').onclick();
  assert.deepEqual(element('target').options.map(option => option.value), ['Captioning']);
});

test('Fibo sends no stale changes and provides a native JSON download', async () => {
  let request;
  const native = { general: { description: 'A scene' }, objects: [] };
  const { element, context } = setup((url, options) => {
    if (url !== '/api/generate') return new Promise(() => {});
    request = options;
    return Promise.resolve({ ok: true, json: async () => ({ output: native }) });
  });
  vm.runInContext("targets = [{id: 'Fibo', vision: true, analysis_only: true, image_generators: ['ideogram4']}]", context);
  element('target').value = 'Fibo';
  element('image-generator').value = 'ideogram4';
  element('changes').value = 'leftover changes for another model';
  element('image').files = [new Blob(['image'], { type: 'image/png' })];
  await element('form').onsubmit(event());
  assert.equal(request.body.get('description'), '');
  assert.equal(request.body.get('target_id'), 'Fibo');
  assert.deepEqual(JSON.parse(element('output').value), native);
  assert.equal(element('download').download, 'fibo-scene.json');
  assert.equal(element('download').hidden, false);
  assert.equal(element('changes').disabled, true);
  assert.equal(element('format').textContent, 'JSON');
});

test('From words Ideogram offers only Fibo/Qwen and downloads generated JSON', async () => {
  for (const model of ['Fibo', 'Captioning']) {
    const { element, context } = setup(url => {
      if (url !== '/api/generate') return new Promise(() => {});
      return Promise.resolve({ok: true, json: async () => ({output: {general: {description: 'A scene'}, objects: []}})});
    });
    vm.runInContext("targets = [{id: 'GemmaUnseen', text: true}, {id: 'Captioning', text: true, analysis_only: true}, {id: 'Fibo', text: true, analysis_only: true}]; imageGenerators = [{id: 'ideogram4', name: 'Ideogram'}]", context);
    element('mode-text').onclick();
    element('image-generator').value = 'ideogram4'; element('image-generator').onchange();
    assert.deepEqual(element('target').options.map(o => o.value), ['Captioning', 'Fibo']);
    element('target').value = model; element('target').onchange();
    element('description').value = 'A scene.';
    await element('form').onsubmit(event());
    assert.equal(element('format').textContent, 'JSON');
    assert.equal(element('download').hidden, false);
    assert.equal(element('download').download, model === 'Fibo' ? 'fibo-scene.json' : 'ideogram4-prompt.json');
  }
});

test('From words offers both prose models for every non-Ideogram target and sends no image', async () => {
  for (const generator of ['krea2', 'zimage', 'flux2-klein', 'anima', 'qwen-image-2.1']) {
    for (const model of ['GemmaUnseen', 'Captioning']) {
      let request;
      const { element, context } = setup((url, options) => {
        if (url !== '/api/generate') return new Promise(() => {});
        request = options;
        return Promise.resolve({ok: true, json: async () => ({output: 'A red boat.'})});
      });
      vm.runInContext(`targets = [{id: 'GemmaUnseen', vision: true, text: true}, {id: 'Captioning', vision: true, text: true, analysis_only: true}, {id: 'Fibo', text: true, analysis_only: true}]; imageGenerators = [{id: '${generator}', name: '${generator}'}]`, context);
      element('mode-text').onclick();
      element('image-generator').value = generator; element('image-generator').onchange();
      assert.deepEqual(element('target').options.map(o => o.value), ['GemmaUnseen', 'Captioning']);
      element('target').value = model; element('target').onchange();
      element('description').value = 'A red boat.';
      await element('form').onsubmit(event());
      assert.equal(request.body.get('mode'), 'text');
      assert.equal(request.body.get('image_generator'), generator);
      assert.equal(request.body.get('target_id'), model);
      assert.equal(request.body.has('image'), false);
      assert.equal(element('download').hidden, true);
    }
  }
});

test('image generation requires both independent model selections', () => {
  const { element, context } = setup();
  vm.runInContext("targets = [{id: 'GemmaUnseen', vision: true}]; imageGenerators = [{id: 'anima', name: 'Anima'}]", context);
  element('target').value = 'GemmaUnseen';
  element('target').onchange();
  assert.equal(element('generate').disabled, true);
  element('image-generator').value = 'anima';
  element('image-generator').onchange();
  assert.equal(element('generate').disabled, false);
  element('image-generator').value = '';
  element('image-generator').onchange();
  assert.equal(element('generate').disabled, true);
});

test('folder markup exposes only Krea2 Character and Style, without tagged formats', () => {
  const html = fs.readFileSync(path.join(__dirname, '../src/prompting_pot/static/index.html'), 'utf8');
  const options = html.match(/<select id="caption-style"[^>]*>([\s\S]*?)<\/select>/)[1];
  assert.deepEqual([...options.matchAll(/<option value="([^"]+)"/g)].map(match => match[1]), ['character', 'style']);
  assert.match(html, /natural sentences, never comma-separated tags/);
});

test('both Krea2 types require a trigger only in caption mode and explain their rules', () => {
  const { element, context } = setup();
  vm.runInContext("mode = 'caption'", context);
  for (const type of ['character', 'style']) {
    element('caption-style').value = type;
    element('caption-style').onchange();
    assert.equal(element('trigger-word').required, true);
    assert.match(element('caption-rules').textContent, /Natural sentences/);
    assert.match(element('caption-rules').textContent, type === 'character' ? /Omit the face/ : /consistent aesthetic/);
  }
  vm.runInContext("mode = 'image'", context);
  element('caption-style').onchange();
  assert.equal(element('trigger-word').required, false);
});

test('Qwen single-image choices retain the generator and display prose without a JSON download', async () => {
  const { element, context } = setup(url => {
    if (url === '/api/generate') return Promise.resolve({ok: true, json: async () => ({output: 'A descriptive scene paragraph.'})});
    return new Promise(() => {});
  });
  vm.runInContext("targets = [{id: 'GemmaUnseen', vision: true}, {id: 'Captioning', vision: true, captioning: true, analysis_only: true, image_generators: ['ideogram4', 'qwen-image-2.1']}, {id: 'Fibo', vision: true}]; imageGenerators = [{id: 'qwen-image-2.1', name: 'Qwen'}, {id: 'ideogram4', name: 'Ideogram'}]", context);
  element('image-generator').value = 'qwen-image-2.1'; element('image-generator').onchange();
  assert.deepEqual(element('target').options.map(o => o.value), ['GemmaUnseen', 'Captioning']);
  assert.equal(element('target').value, 'GemmaUnseen');
  element('target').value = 'Captioning'; element('target').onchange();
  assert.equal(element('image-generator').value, 'qwen-image-2.1');
  assert.equal(element('changes').disabled, true);
  element('image').files = [new Blob(['image'], {type: 'image/png'})];
  await element('form').onsubmit(event());
  assert.equal(element('output').value, 'A descriptive scene paragraph.');
  assert.equal(element('format').textContent, 'TEXT');
  assert.equal(element('download').hidden, true);
});

test('caption folder selection includes BMP and ignores hidden files and subfolders', () => {
  const { element, context } = setup();
  element('folder').files = [
    { name: 'reference.BMP', webkitRelativePath: 'dataset/reference.BMP' },
    { name: 'reference.txt', webkitRelativePath: 'dataset/reference.txt' },
    { name: '.hidden.png', webkitRelativePath: 'dataset/.hidden.png' },
    { name: 'nested.png', webkitRelativePath: 'dataset/sub/nested.png' },
  ];
  element('folder').onchange();
  assert.equal(vm.runInContext('folderImages.length', context), 1);
  assert.match(element('folder-summary').textContent, /1 top-level images · 1 existing captions/);
});

test('folder upload accepts 150 images and rejects 151 before uploading', async () => {
  for (const count of [150, 151]) {
    let uploaded = false;
    const { element, context } = setup(url => {
      if (url !== '/api/caption') return new Promise(() => {});
      uploaded = true;
      return Promise.resolve({ok: true, blob: async () => new Blob(['zip']), headers: {get: () => '0'}});
    });
    vm.runInContext("mode = 'caption'; targets = [{id: 'Captioning', vision: true, captioning: true}]", context);
    element('target').value = 'Captioning'; element('trigger-word').value = 'TOK';
    element('folder').files = Array.from({length: count}, (_, i) => Object.assign(new Blob(['image']), {name: `${i}.png`, webkitRelativePath: `dataset/${i}.png`}));
    element('folder').onchange();
    await element('form').onsubmit(event());
    assert.equal(uploaded, count === 150);
    if (count === 151) assert.match(element('status').textContent, /1–150/);
  }
});

test('caption requests send style and trigger, and expose the ZIP download', async () => {
  for (const [style, model] of [['character', 'Captioning'], ['style', 'Captioning'], ['character', 'GemmaUnseen'], ['style', 'GemmaUnseen']]) {
    let request;
    const { element, context } = setup((url, options) => {
      if (url !== '/api/caption') return new Promise(() => {});
      request = options;
      return Promise.resolve({ ok: true, blob: async () => new Blob(['zip']), headers: { get: () => '1' } });
    });
    vm.runInContext("mode = 'caption'; targets = [{id: 'Captioning', vision: true, captioning: true}, {id: 'GemmaUnseen', vision: true, captioning: true}]", context);
    element('target').value = model;
    element('caption-style').value = style;
    element('trigger-word').value = '  alex cooper  ';
    element('folder').files = [Object.assign(new Blob(['image'], { type: 'image/bmp' }), { name: 'image.bmp', webkitRelativePath: 'dataset/image.bmp' })];
    element('folder').onchange();
    await element('form').onsubmit(event());
    assert.equal(request.body.get('caption_style'), style);
    assert.equal(request.body.get('trigger_word'), 'alex cooper');
    assert.equal(request.body.get('target_id'), model);
    assert.equal(request.body.get('images').name, 'image.bmp');
    assert.equal(element('download').hidden, false);
  }
});

test('both Krea2 types skip only txt sidecars and ignore JSON files', () => {
  const { element, context } = setup();
  element('folder').files = ['a.png', 'b.png', 'a.txt', 'b.txt', 'a.json'].map(name => ({name, webkitRelativePath: `dataset/${name}`}));
  element('folder').onchange();
  assert.equal(vm.runInContext('existingCaptions.length', context), 2);
  for (const type of ['character', 'style']) {
    element('caption-style').value = type; element('caption-style').onchange();
    assert.match(element('folder-summary').textContent, /2 existing captions/);
  }
});

test('blank Krea2 trigger is rejected before uploading', async () => {
  let uploaded = false;
  const { element, context } = setup((url) => {
    if (url === '/api/caption') uploaded = true;
    return new Promise(() => {});
  });
  vm.runInContext("mode = 'caption'", context);
  element('caption-style').value = 'character';
  element('trigger-word').value = '  ';
  await element('form').onsubmit(event());
  assert.equal(uploaded, false);
  assert.match(element('status').textContent, /trigger word/);
});

test('Browse opens the image picker', () => {
  const { element } = setup();
  element('browse-image').onclick();
  assert.equal(element('image').clicked, true);
});

test('dropping one valid image updates file input and preview', () => {
  const { element } = setup();
  const drop = event([{ name: 'image.png', type: 'image/png', size: 100 }]);
  element('image-drop').events.dragover(drop);
  assert.equal(element('image-drop').classList.contains('drag-over'), true);
  element('image-drop').events.drop(drop);
  assert.equal(drop.prevented, true);
  assert.equal(element('image').files, drop.dataTransfer.files);
  assert.equal(element('preview').src, 'blob:preview');
  assert.equal(element('preview').hidden, false);
  assert.equal(element('image-drop').classList.contains('drag-over'), false);
});

test('invalid and oversized drops use existing validation', () => {
  for (const file of [{ type: 'text/plain', size: 100 }, { type: 'image/png', size: 11 * 1024 * 1024 }]) {
    const { element } = setup();
    element('image-drop').events.drop(event([file]));
    assert.equal(element('preview').hidden, true);
    assert.match(element('status').textContent, /up to 10 MB/);
  }
});

test('multiple files and non-file drops are rejected', () => {
  for (const files of [[], [{}, {}]]) {
    const { element } = setup();
    element('image-drop').events.drop(event(files));
    assert.match(element('status').textContent, /one PNG/);
    assert.equal(element('image').files.length, 0);
  }
});

test('busy generation prevents browsing and replacing the image', () => {
  const { element, context } = setup();
  vm.runInContext('busy = true', context);
  element('browse-image').onclick();
  element('image-drop').events.drop(event([{ type: 'image/png', size: 100 }]));
  assert.equal(element('image').clicked, undefined);
  assert.equal(element('image').files.length, 0);
});
