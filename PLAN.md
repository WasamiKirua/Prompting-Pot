# Implementation plan

## Decisions
- FastAPI + static HTML/CSS/JavaScript; no frontend build pipeline.
- Root `config/targets.json` registry points to dedicated llama-swap YAML files.
- Three exclusive input modes: image with optional changes, description, or folder captioning.
- Caption folders use browser uploads; results download as a ZIP, not direct original-folder writes.
- Caption top-level images only, process sequentially, and skip existing sidecars matching the selected `.txt`/`.json` format.
- All templates live in `src/prompting_pot/prompts.py`; image inference lives in `gemma_unseen.py`, caption inference/batching in `captioning.py`.
- Captioning offers Krea2 Standard, Tagged attributes, Qwen-Image-2.1 text, Qwen-Image-2.1 tagged and Ideogram 4 JSON. Both tagged styles require a trigger word. Caption folders additionally support BMP.
- Ideogram JSON uses schema constraints, backend validation and canonical ordering; no syntax-only result is treated as proof of visual accuracy.
- Local prompt-writing model (registry/YAML/API ID) and target image generator (template ID) are independent selections.
- Automatically format JSON responses; no output-format selector.
- Backend owns llama-swap startup/readiness and cleanup. Image and caption inference are connected; text inference remains stubbed.
- Preserve user edits in llama_swap.py, including IP and BASE_URL; do not modify that file.
- Serialize inference sessions in one application worker; never terminate externally started servers.

## Tiny tasks
- [x] Inspect repository and clarify requirements.
- [x] Choose technology and agree backend boundary.
- [x] Add editable target registry and configuration guidance.
- [x] Define request validation and integration functions.
- [x] Serve frontend and expose target/generation endpoints.
- [x] Build responsive workspace.
- [x] Add persistent light/dark themes with explicit text colors.
- [x] Add image upload and preview.
- [x] Add description mode and target selection.
- [x] Add loading, error, output formatting, and copy behavior.
- [x] Test API contracts and configuration safety (13 tests passing).
- [x] Check frontend syntax and document startup/integration.
- [x] Calculate core text contrast in both themes (all tested pairs exceed 4.5:1).
- [ ] Visually verify in a browser: mobile layout, keyboard navigation, themes, uploads and copy. No browser or Playwright is installed in this environment.
- [x] User: populate GemmaUnseen registry entry and unseen.yaml.
- [x] Wire llama-swap lifecycle and termination into single-generation and batch endpoints.
- [x] Implement image-mode generate_prompt() inference.
- [x] Implement caption inference.
- [ ] Implement text inference and test real model outputs.

## Folder captioning tasks
- [x] Clarify folder picker, skip behavior, scope and inference boundary.
- [x] Add editable task templates and target-specific overrides.
- [x] Pass selected templates into the inference function.
- [x] Validate batch names, collisions, image types and limits.
- [x] Load/reuse llama-swap once per pending batch through the existing stub.
- [x] Create UTF-8 caption sidecars and manifest inside a downloadable ZIP.
- [x] Add third frontend mode, folder summary, busy state and ZIP download.
- [x] Run regression tests (30 passed) and JavaScript syntax checks.
- [x] Validate new backend modules with strict mypy, Ruff and Black.
- [ ] Browser verification of folder picker and ZIP extraction.
- [ ] Evaluate caption/template quality after real inference is wired.

## Reference image drop area
- [x] Replace the upload area with drag-and-drop and a keyboard-accessible Browse button.
- [x] Reuse image preview/type/size validation and prevent replacement during generation.
- [x] Add drag-over feedback using both theme palettes.
- [x] Verify 5 JavaScript behavior tests, syntax checks and 30 backend regression tests.
- [ ] Verify native file dragging and keyboard browsing in a real browser.

## llama-swap lifecycle
- [x] Retain owned process handle, print PID and suppress child terminal output.
- [x] Validate configuration, guard occupied ports, and wait for API readiness.
- [x] Serialize startup, inference and cleanup across requests.
- [x] Gracefully stop owned process groups, with timeout/forced-kill fallback.
- [x] Clean up on startup failures, generation errors and FastAPI shutdown.
- [x] Keep caption batches alive until all pending images finish.
- [x] Preserve externally running instances.
- [x] Verify 49 backend tests and 5 frontend tests; new lifecycle module passes strict mypy, Ruff and Black.
- [ ] Verify startup/readiness/shutdown against the actual installed llama-swap and model YAML.

## GemmaUnseen image-to-prompt tasks
- [x] Inspect krea2_prompt.py and distinguish local model from target image generator.
- [x] Confirm all five targets, preserved output formats and reference inference settings.
- [x] Copy and verify exact system prompts inside this codebase.
- [x] Consolidate all prompt templates in prompts.py and remove the redundant image_prompts.py module.
- [x] Add a separate frontend image-generator dropdown and clarify local-model labels.
- [x] Validate image-generator selection before model startup.
- [x] Send the image and optional changes through the async OpenAI SDK.
- [x] Default empty changes to as-is reconstruction.
- [x] Preserve response text/sections and expose readable inference errors.
- [x] Retain lifecycle cleanup on success and failure.
- [x] Verify 78 backend tests and 7 frontend tests, including separate model/target payloads.
- [x] Verify real OpenAI SDK compatibility against a local fake API (no GPU model launched).
- [x] Verify 100% statement coverage of gemma_unseen.py and strict mypy/Ruff/Black for the new modules.
- [ ] User: test image generation on real GemmaUnseen hardware, including output/context limits.
- [ ] Browser verification of the two dropdowns and output display.

## Qwen captioning integration
- [x] Inspect krea2_captioning.py, registry/YAML IDs and user lifecycle changes.
- [x] Confirm explicit style selector and caption-only BMP support.
- [x] Copy both original templates into prompts.py without changing their text.
- [x] Add Standard/Tagged selector and conditional required trigger field.
- [x] Validate style/trigger before launching any model.
- [x] Call the selected Captioning model with image and original user instructions.
- [x] Retain server generation defaults and 300-second timeout.
- [x] Preserve skip-existing, ZIP sidecars, per-image failures and batch cleanup.
- [x] Verify 98 backend tests and 11 frontend tests, including both styles, trigger validation, BMP and image-mode regressions.
- [x] Verify strict mypy, Ruff, Black and 94% captioning.py statement coverage.
- [x] Verify both caption prompts match the reference exactly and llama_swap.py checksum is unchanged.
- [ ] User: verify real Qwen captions/tags, BMP decoding and model quality.
- [ ] Browser verification of style selection, trigger field and ZIP results.

## Ideogram 4 / Qwen-Image-2.1 caption expansion
- [x] Research Ideogram prompting guide, exact schema/verifier, and llama.cpp structured output.
- [x] Inspect the exact fine-tune's available card; JSON quality is not verified there.
- [x] Confirm expanded style list and .json sidecars.
- [x] Add researched image-caption prompts in prompts.py without changing old templates.
- [x] Add schema-constrained Ideogram requests with an explicit 4096-token budget.
- [x] Validate JSON structure, duplicates, palettes, bbox geometry and truncation.
- [x] Canonically serialize JSON keys and UTF-8 as documented by Ideogram.
- [x] Use selected-format skip detection, refreshed frontend counts and ZIP sidecars.
- [x] Guard JSON sidecar collisions with the reserved batch manifest.
- [x] Document research sources and capability-versus-quality caveats.
- [x] Verify 126 backend tests and 12 frontend tests, including actual SDK schema serialization against a local fake API.
- [x] Verify strict mypy, Ruff and Black; Ideogram validator has 100% statement coverage and captioning.py 94%.
- [x] Verify llama_swap.py and both YAML checksums are unchanged.
- [ ] User: test installed llama.cpp schema support and the exact captioning fine-tune on real images.
- [ ] Browser verification of all five styles and format-specific skip counts.

## Qwen-Image-2.1 tagged captions
- [x] Add a dedicated Qwen-oriented attribute-tag prompt in prompts.py.
- [x] Add a separate frontend option and required trigger handling.
- [x] Validate the trigger for both tagged styles before model startup.
- [x] Preserve .txt outputs, skip-existing, server defaults and batch lifecycle.
- [x] Verify routing/payload, Unicode triggers, invalid triggers and frontend behavior (135 backend and 13 frontend tests pass).
- [x] Verify typing/lint and unchanged lifecycle/YAML checksums.
- [ ] User: evaluate tag quality and exact trigger adherence with real model images.

## From words: generator-specific text and JSON
- [x] Confirm official Fibo expand task and GGUF guidance; verify installed Fibo on two text-only scenes with native validation and owned-process cleanup.
- [x] Expose target-generator selection in From words: Gemma/Qwen for five prose targets; Fibo/Qwen for Ideogram JSON.
- [x] Add text-only requests and generator instructions without fabricated image inputs; preserve image/folder workflows.
- [x] Use schema-constrained Qwen Ideogram JSON and native Fibo expansion JSON; reject empty, invalid and truncated outputs.
- [x] Document Fibo's square expansion default and the distinct native/Ideogram JSON formats.
- [x] Verify live Qwen Krea2 prose + Ideogram JSON and Gemma Krea2 prose; this is limited smoke evidence, not broad quality equivalence.
- [x] Verify 229 backend + 34 frontend tests, 100% text-adapter statement coverage, scoped typing/lint and Chromium text choices/downloads/tab round trips with mocked UI inference.
- [x] Update README and script cache version; preserve YAMLs, registry and lifecycle implementation.

## Folder batch limit: 150 images
- [x] Raise backend/frontend limit to 150; retain 100 MB total and 10 MB per image.
- [x] Update UI, README and cache version; test acceptance at 150 and rejection at 151.

## Folder batch limit: 130 images
- [x] Raise backend/frontend image-count limit to 130; retain 100 MB total and 10 MB per image.
- [x] Update UI/help, README and script cache version; test acceptance at 130 and rejection at 131.

## GemmaUnseen Krea2 folder captions
- [x] Add registered vision-capable GemmaUnseen alongside Qwen3.5; keep Qwen as the default when available.
- [x] Reuse the model-independent Character/Style prose prompts and required trigger validation without duplicating templates.
- [x] Route Gemma through its existing multimodal adapter with reasoning off; leave Qwen server defaults unchanged.
- [x] Enable model selection and remember the caption choice independently of single-image workflows.
- [x] Verify both models/types with BMP, skipping, ZIP sidecars, invalid-prefix handling and vision restrictions: 199 backend + 31 frontend tests pass.
- [x] Check model choice, both Gemma caption types, download and mode round trips in Chromium with mocked inference.
- [x] Update README and script cache version; leave YAMLs, registry and lifecycle source unchanged.
- [ ] Real-model caption quality/performance comparisons remain unverified.

## Krea2-only folder LoRA captions
- [x] Recover the 35 missing tracked files from the latest commit with explicit approval; remove the obsolete checkout after making a private temporary recovery archive.
- [x] Restrict folder captioning to Qwen3.5 and Krea2 Character/Style modes; require a single-line trigger for both.
- [x] Replace comma-tag prompts with natural prose; Character omits faces/fixed identity features, Style describes visible content and lets the shared aesthetic carry style.
- [x] Remove obsolete general/tagged and non-Krea2 folder formats, JSON sidecar branches and extension switching.
- [x] Retain single-image adapters/schema validation; reinstate the previously requested Qwen-Image-2.1 model choice missing from the restored commit.
- [x] Reject captions missing the exact trigger-space prefix per image; retain sequential ZIP/manifest output and skip behavior.
- [x] Update frontend rule hints, fixed caption-model selection, script cache version and README/API documentation.
- [x] Verify 188 backend + 30 frontend tests, 94% combined prompts/captioning coverage, strict typing and scoped Ruff/Black checks; keep runtime lifecycle and model/YAML configuration unchanged.
- [x] Check both LoRA types, trigger controls, TXT-only skipping, ZIP output, mobile layout and single-image choices in Chromium with mocked inference.
- [ ] Broad real-model prose/identity-exclusion adherence remains a manual review requirement; prompt instructions are not semantic validation.

## README technology overview
- [x] Add a compact technology table and GitHub-rendered Mermaid architecture diagram.
- [x] Show local inference, configuration, validation and browser output without implying image generation.

## Pre-publication privacy/security audit
- [x] Inspect Git-publishable source/config/docs/tests/lockfile and the supplied PDF.
- [x] Confirm the repository currently has no commits or staged history to remediate.
- [x] Run checksum-verified Gitleaks v8.30.1 locally with redacted reporting; no credential findings.
- [x] Check personal paths, emails/private IPs and PDF metadata, attachments, fields and URI annotations.
- [x] Document reviewed non-secret placeholders/defaults and that no sensitive substitutions were necessary.
- [x] Strengthen ignore rules for local settings, caches/logs, key material, weights and private input/output/audit artifacts.
- [x] Preserve runtime source, GPU/YAML settings, Git identity and remotes.
- [x] Add staged/history scanning, credential-rotation and Git author privacy guidance to README.md.

## Qwen generation-only prompt refinement
- [x] Review Qwen21_Editguide.pdf against the existing image-generation template.
- [x] Use uploaded images for analysis, with standalone text-to-image output and requested changes expressed as final scene attributes.
- [x] Remove editing modes, reference tags/roles, preservation commands and ratio-follow behavior.
- [x] Add concrete composition, lighting, materials, palette and typography/grounding rules without forced word counts or embellishments.
- [x] Preserve breakdown/prompt/tweak sections and separate aspect-ratio recommendation.
- [x] Leave captioning prompts, inference settings and YAML/lifecycle files untouched.
- [x] Add prompt-contract regression checks and document the source/scope in README.md.
- [ ] Compare actual Qwen-generated images to establish downstream quality improvements.

## Unified user documentation
- [x] Consolidate Docker, local setup, registry/YAML/GPU configuration, workflows, caption formats and technical reference into README.md.
- [x] Add a table of contents, troubleshooting and explicit single-GPU/machine-specific configuration guidance.
- [x] Retain current Gemma exclusion and distinguish native Fibo JSON from Ideogram caption JSON.
- [x] Remove superseded docs/docker.md, docs/caption-formats.md and config/README.md; preserve this developer history separately.
- [x] Remove stale user-guide links and update verification caveats to reflect completed live/browser checks.

## Self-contained Docker / Compose
- [x] Confirm self-contained packaging instead of mounting host binaries.
- [x] Pin the upstream CUDA llama.cpp image and uv image by digest.
- [x] Include checksum-verified llama-swap v262 for amd64/arm64.
- [x] Install locked production Python dependencies and run one worker without reloading.
- [x] Add non-root execution, HTTP health check, init process and graceful stop window.
- [x] Add Compose with 0.0.0.0 publishing, all NVIDIA GPUs and unless-stopped restart behavior.
- [x] Mount models/configuration read-only without changing lifecycle/YAML files.
- [x] Exclude caches, model/evaluation files and local settings from the build context.
- [x] Build the image and test a healthy container on temporary port 18000.
- [x] Verify both GPUs, packaged frontend, real Qwen JSON inference and owned-process cleanup.
- [x] Stop/remove the temporary container/network without affecting the host app.
- [x] Document host prerequisites, path/port overrides, GPU-index portability and update/rollback commands.
- [x] Add daemon-free Compose contract tests; verify 174 backend and 30 frontend tests, Ruff/Black, Compose config and a final image rebuild.
- [ ] Validate other models and different GPU layouts/arm64 when deploying on those machines.

## Clear generated output
- [x] Add Clear beside Copy in the output footer.
- [x] Reset output, placeholder, format badge, title, copy state and result download; revoke its object URL.
- [x] Preserve inputs and model/generator choices.
- [x] Disable Clear for empty output and during generation.
- [x] Cover text, JSON, batch output and busy-state behavior in frontend tests.

## Qwen single-image Ideogram JSON
- [x] Add Qwen3.5 Captioning alongside Fibo for From an image → Ideogram 4.
- [x] Keep Fibo the default and preserve an explicit Qwen selection.
- [x] Reuse existing Qwen caption prompt, JSON Schema request and validation.
- [x] Describe images as-is; disable/reject change instructions.
- [x] Download Qwen JSON as ideogram4-caption.json while preserving Fibo's native schema/download.
- [x] Keep folder captions unchanged and Gemma excluded pending separate evaluation.
- [x] Verify 171 backend and 26 frontend tests, including invalid JSON cleanup.
- [x] Verify live Chromium model selection, caption-mode round trips and JSON rendering/download with mocked inference.
- [x] Preserve lifecycle and all YAML checksums.
- [x] Evaluate GemmaUnseen's Ideogram JSON quality/compatibility separately before enabling it: 3/3 live requests passed the existing schema; complex OCR/palette limitations remain.
- [x] Save live evidence and limitations in docs/evaluations/gemma-ideogram/README.md; verify owned-process cleanup and unchanged YAML/lifecycle hashes.
- [x] User decision after evaluation: keep Gemma excluded from Ideogram; retain Fibo and Qwen only.

## Keep Ideogram selectable
- [x] Keep the full image-generator list visible, including Ideogram 4, with any selected model.
- [x] Selecting Ideogram 4 shows/selects only Fibo; selecting the other five shows/selects only GemmaUnseen.
- [x] Disable generation when Fibo is missing, without a fallback.
- [x] Update the script cache version and regression tests.
- [x] Verify direct Krea2 → Ideogram → Z-Image → Ideogram switching in Chromium.

## Frontend cached-script check
- [x] Add a cache-busting script URL after the user reported stale model choices.
- [x] Verify the running app in Chromium: all five generator selections show only GemmaUnseen, without page errors.
- [x] Recheck 168 backend and 24 frontend tests.

## Frontend prompt-model filtering
- [x] Show and automatically select only GemmaUnseen for Krea2, Z-Image, FLUX.2 Klein, Anima and Qwen-Image-2.1.
- [x] Disable generation if GemmaUnseen is missing, without choosing a fallback.
- [x] Restore model choices when the generator selection is cleared.
- [x] Preserve Qwen-only caption mode and existing Ideogram/Fibo behavior.
- [x] Keep changes frontend-only; no backend, registry, YAML or lifecycle edits.
- [x] Verify all five generator filters, mode transitions, missing-model behavior and Ideogram regression checks.

## Fibo single-image analysis
- [x] Inspect the supplied analyzer and clarify workflow/schema choices.
- [x] Preserve user-provided registry, fibo.yaml, other YAMLs and lifecycle module.
- [x] Vendor the native Pydantic schema and coordinate/color token processing locally.
- [x] Centralize the trained `<caption><full>` task in prompts.py.
- [x] Add dedicated async inference, PNG conversion and reference sampling settings.
- [x] Avoid constrained decoding and retries, matching the supplied server adapter.
- [x] Restrict Fibo to single-image Ideogram 4 analysis in UI and backend.
- [x] Disable change instructions; reject unsupported request combinations before startup.
- [x] Keep Qwen-only folder captions unchanged.
- [x] Pretty-print/copy native JSON and offer fibo-scene.json download.
- [x] Test native validation, nested token conversion, errors, cleanup and UI restrictions: 168 backend and 17 frontend tests pass.
- [x] Verify 100% statement coverage for fibo.py, strict mypy, Ruff, Black and JavaScript syntax.
- [x] Recheck unchanged lifecycle/YAML checksums, including the user-provided fibo.yaml.
- [x] Document native JSON versus Ideogram caption-schema differences.
- [ ] Validate against the actual Fibo model and downstream Ideogram workflow.
- [ ] Verify interactions/downloads in a real browser.

## Caption model restriction
- [x] Mark the registered Captioning vision model as caption-eligible in API metadata.
- [x] Show/select only Qwen3.5 Captioning in caption mode.
- [x] Restore the previous prompt-model selection when leaving caption mode.
- [x] Disable caption generation if the required model is missing; never fall back to Gemma.
- [x] Reject other model IDs in the backend before startup for all five caption styles.
- [x] Verify 141 backend and 15 frontend tests, including wrong-model rejection and missing-model behavior.
- [x] Verify lifecycle/YAML files remain unchanged.

## Integration boundary
GemmaUnseen image inference, Qwen caption inference and llama-swap lifecycle are connected. Text-only inference remains pending. Vision support must match the selected target configuration. Configuration IDs, not arbitrary paths or commands, cross the API boundary. Browser visual verification may require a locally installed browser.
