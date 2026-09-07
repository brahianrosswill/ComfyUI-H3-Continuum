# AGENTS.md

## Scope
MiniMax H3 Continuum V3.8 for ComfyUI. Treat the accepted V3.7 Production engine and the accepted V3.8 GPU Gates as the current execution baseline. V3.8 supports its approved current public surface; pre-V3.8 public Node IDs and saved workflows are distributed through their historical Release/tag and are not V3.8 compatibility requirements. Experimental controls must not change current Production behavior. Still Image Guide remains Experimental / Production HOLD.

Current release-hygiene baseline: O1 and AUDIO-R1 are PASS; the public surface is seven nodes; Sampling Contract is v5; Run Storage is v3; State, Session, Assembly, Driving Audio, and Second Pass/O1 contracts are protected. Release / Registry readiness remains HOLD until package staging is rerun for the latest working tree.

## Non-negotiable rules
- Follow ComfyUI Core behavior for user-facing validation. Continuum-only prompt restrictions, model allowlists, compatibility gates, and policy-based execution stops are prohibited.
- Do not implement a new user-facing hard stop. If execution is structurally impossible or continuing risks corrupting stored data, document the exact broken internal contract and obtain explicit approval before changing the stop.
- Prompt content must never block execution. Empty, malformed, incomplete, or unknown prompt syntax falls back to a Fixed prompt and is passed through as entered. Valid Timeline/List/JSON syntax is still parsed normally.
- Keep warnings diagnostic-only unless execution cannot continue safely. Hard stops are limited to corrupt internal contracts such as invalid latent/layout topology, incompatible Assembly Plan data, broken stored revision schema/hash, unusable required payloads, or decode formats that cannot be assembled.
- Unknown upstream nodes, models, wrappers, or merged models must not be rejected solely because they are unknown.
- Keep legacy `strict_compatibility` inputs loadable for old workflows, but ignore their value. They must never enable blocking behavior.
- Before every change, run `tools\snapshot.ps1` against the authoritative source. Preserve the current V3.8 backend schema and serialized widget order until a separately approved UI/public-surface implementation changes them.
- In public templates, connection examples, screenshots, and documentation, keep ComfyUI Core nodes on their standard display titles. Do not assign custom titles to Core nodes, because users must be able to distinguish Core nodes from Continuum and third-party custom nodes immediately.
- Runtime tests must record the workflow, prompt, media inputs, model/LoRA/sampler/step settings, dimensions, chunk count/duration, elapsed time, and observed result. Inspect embedded workflow metadata when present.
- Do not globally replace `PackedLayout`, `MiniMaxH3.extra_conds`, or other ComfyUI classes.
- Keep SageAttention, Sol-Attn, and Spectrum external; integrate through public ComfyUI wrappers and H3 payload contracts.
- Preserve the identity of `layout.position_ids` when changing MM-RoPE coordinates.
- Never silently resize a continuation State or Session to a different resolution.
- Unknown H3 layout contracts must fail clearly instead of rendering a subtly incorrect join.
- State and Session files remain safetensors + JSON and are written atomically.
- UI nodes call reusable non-UI modules. Legacy nodes and V3.4 must share temporal, state, continuation, and layout logic where their contracts overlap.
- V2 must not decode Video/Audio VAE between H3 chunks.
- Accepted full chunk latents remain on CPU; do not accumulate generated chunks in VRAM.
- Spectrum history must remain run-scoped through its public sampler wrappers; never import Spectrum private runtime functions.
- Do not treat V1-V3.7 public Node IDs as V3.8 compatibility requirements. Removing their V3.8 registration still requires an explicit audited public-surface change; historical implementations may remain internally while V3.8 depends on them. Preserve State/Session data safely unless a separately approved migration replaces it.

## Project files

- `AGENTS.md`: durable rules and prohibitions only.
- `PROJECT_STATE.md`: current paths, version, active direction, and known gaps.
- `WORKLOG.md`: append-only summaries of completed work and evidence.
- Do not duplicate long history across these files. Update `PROJECT_STATE.md` when facts change and append one concise entry to `WORKLOG.md` after completed work.

## Product direction

- Product definition: H3 Continuum is a Production Sampler that generates, reviews, partially regenerates, and resumes long-form video without restarting the entire work.
- Keep one discoverable Main surface plus an in-node `Advanced` disclosure. Do not require users to discover a hidden node property before normal Production controls become usable.
- Evaluate new work against three pillars: `Continuation Quality`, `Non-destructive Production`, and `Open Integration`. Prefer external integration or `Deferred` when work does not clearly strengthen one of them.
- `Main / Production` must be simple by default and expose Production controls when expanded. Do not develop a separate Easy product path. Its primary flow is Continuation Quality, Review/Regenerate, Resume/Branch, and Finalize.
- `Advanced` contains Second Pass, Selective/Temporal Refine, external integration, and diagnostics. Experimental or development-only nodes do not become public merely because their code and tests are retained.
- Do not build an all-in-one node pack. Keep external Upscaler, TensorRT, FBC, Sol-Attn, Spectrum, Director, Prompt Planner, Save/Encoder, and Core or external VAE Decode outside Continuum unless an explicit integration contract is approved.
- Do not grow the Main Sampler with unbounded sockets or serialized widgets. Prefer stable latent/assembly/refine/run contracts and external adapters.
- V3.8 is the current supported product. V3.7 and earlier remain obtainable from historical GitHub Release/tag rather than being carried as public V3.8 Node-ID compatibility.
- A saved workflow that references a Node ID not exported by V3.8 may load as an unknown node. State this explicitly in the V3.8 README, Release Notes, and migration note, and direct users to the corresponding historical Release/tag. Do not restore old registrations solely to hide this intentional support boundary.
- The accepted V3.8 launch surface is seven searchable nodes: Sampler V3.8, Reference Audios, Finalize, Load Image, Load Audio, Load Video, and Second Pass. AUDIO-R1 is the approved one-node modular-input exception; existing six node IDs and schemas remain compatible.
- Public-surface cleanup changes export mappings only. Do not physically delete legacy classes/modules while current V3.8 imports or inherits them, and do not exclude runtime-imported legacy modules from the package archive.
- Main UI disclosure state is presentation-only. It must not enter Sampling or Run Storage identity, alter backend kwargs, or rewrite values of hidden widgets. Connection-specific size controls must be visible when their corresponding input is connected.
- Keep the accepted V3.8 Main facade explicit rather than inference-driven. Its first controls are `Prompt Format`, `Continuity`, `Base Seed`, `Control After Generate`, and `Audio Continuity`, followed by independent `Chunks` and `Seconds per Chunk`, a read-only calculated `Total Length`, an always-visible `Size Source = First Image | Manual`, the applicable Resolution or exact Width/Height controls, Run, Progress, a clear Queue-ready summary, and one Advanced disclosure. Do not show permanent connected/unconnected input badges, infer an Input Mode, derive chunks from a total-duration selector, or automatically change Size Source when an upstream node is bypassed or disconnected. When the canonical backend revision is `review_ready`, replace the setup facade with the amber review status and the three explicit Queue actions; do not infer this state in the frontend.
- Keep Main UI numbers compact: Chunks and pixel dimensions are integers, and Seconds per Chunk shows a decimal only when one is meaningful. Use user-facing `Full Video`, `Review by Chunk`, and `Resume & History` labels while preserving their existing backend values. Review action buttons must remain hidden until the canonical Run Storage revision reports `review_ready`; never infer readiness from frontend widgets alone. Hide empty Render History and keep Advanced settings behind the visible in-node disclosure.
- Every visible V3.8 Main, Advanced, Review, and Render History control, plus every public Sampler input, must have non-empty mouse-over help. Context-sensitive controls must explain the currently selected behavior; `Output Size` must explicitly distinguish First Image for I2VA/FL2VA from Manual Width/Height for T2VA or workflows without a First Image.
- Completed-review UI exception (2026-09-07, user-authorized repair): a canonical backend revision with status `complete` must retain its completion panel, Back to Settings/Return to Review, and non-empty Render History. Show Try this chunk again only when that revision carries a valid review_unit, as already supported by the backend. Hide the no-op Continue/Finish buttons on completion. This extends the earlier review_ready-only presentation rule; never rewrite backend status, infer readiness from widget counts, or invent a retry unit for a full-run completion.
- Keep Core node display names unchanged in every public workflow.
- User-approved distribution exception (2026-09-07): publish the supplied Spectrum graph as `MiniMax_H3_Continuum_V38.json` and the user's matching ZIP, byte-for-byte, including its existing titles, prompts, media names, and saved settings. It is one graph with external Spectrum, rgthree, KJNodes, and ComfyUI-Easy-Use dependencies, not a Core-only template. Document setup differences instead of silently rewriting the workflow. Keep unrelated backup ZIPs out of new commits/packages.
- Hi-Res Fix is not part of the new standard workflow. Preserve its historical implementation until cleanup is approved; use Second Pass as the external latent-processor bridge.
- Keep P1a First Block Cache and P2 boundary-aware Sol-Attn `Deferred`. Keep A5 and A8b `HOLD` under their recorded release contracts. Issue #13 reporter-exact R3 remains the highest Continuation Quality investigation when its required workflow is available.

## H3 completion review handoff

- After every implementation or Gate is complete and its local validation report is ready, send that self-contained completion report to the pinned ChatGPT task titled `H3情報チェック` before starting the next phase.
- The fixed current destination is task ID `6a94b91e-af7c-83e8-be18-ddc504f690c4` (pinned index 3). A second older task has the same title; do not send to the unpinned duplicate.
- Wait for and read the review reply, then bring its conclusions and requested corrections back into the active Continuum task as feedback. Treat the reply as review input under the user's authority; preserve the repository rules and request approval where a proposed change expands scope.
- Do not duplicate a handoff that the user explicitly says they already completed manually. If the destination is unavailable or the reply requires user judgment, stop and report that condition instead of silently skipping the review.

## A8b and A5 HOLD release contracts

- 2026-09-05 explicit user exception ("許可、許可。"): Issue13 R3B may run one isolated Video-only weak tapered context-noise A/B experiment before reporter-exact R3. This waives only the reporter-exact/independence prerequisite for that diagnostic. Keep Production/source runtime unchanged, Audio/masks/depth/Seed/SIGMAS/grouping unchanged, default OFF bit-exact, and pass CPU/Shadow/Replay before GPU. Restore clean exported prefix after the diagnostic perturbation. No combined interventions, A5 work, public controls, or Production promotion are authorized. General A8b and Release HOLD remain; GPU-audit sharing is limited to the fixed H3情報チェック task.

- A8a is the accepted read-only baseline: full pytest `1041`, Manifest `242/242`, every observer record has `execution_applied=false`, and Production/Sampling/Run Storage/State/Session/Assembly parity is preserved.
- Keep A8b execution policy on HOLD until one intervention type is specified and approved. Do not change context, transport, mask, latent, or conditioning in combination. Seed, SIGMAS, and physical grouping remain unchanged unless separately approved.
- Exclude the failed Issue #13 candidates from A8b: effective 13-frame history, 22-frame Video prefix with 75% preservation, Soft Re-anchor, and Production Fixed Prefix/teacher forcing.
- Before A8b GPU Sampling, either complete reporter-exact Issue #13 R3 or prove by static and replay evidence that the proposed intervention is independent of the recursive-feedback path. Then pass an A8a Shadow/Replay Gate with no unnecessary action, Terminal misclassification, or Run Storage replay difference.
- Any approved A8b implementation starts Experimental and Default OFF. Existing Production must remain bit-exact, public schema changes require separate approval, and external Sage/Sol/Spectrum wrappers must remain intact.
- Keep A5 Reference Video Influence Schedule/Fade on HOLD until reporter-exact Issue #13 R3 classifies Reference Video as unrelated, an amplification-only factor with a defined safe range, or a necessary/strong cause candidate. Only the first two outcomes may unlock Experimental A5 work; the last keeps A5 on HOLD.
- A5 OFF must remain bit-exact with Production. Initial A5 may change only Reference Video influence; it must not change Video continuation, Audio, Seed, SIGMAS, Terminal Merge, Assembly, or Run Storage. Do not implement A5 and A8b concurrently.
- A5 promotion requires at least `3x5` plus a long-form continuation GPU Gate covering trajectory, identity, saturation, contrast, sharpness, and highlight drift. Short-form evidence alone is insufficient.

## Validation

```bash
powershell -ExecutionPolicy Bypass -File tools\validate.ps1
```

Runtime installer verification must check native PackedLayout behavior and all registered nodes.

GPU checks must cover Sage only, Sage+Sol, Sage+Spectrum, and Sage+Sol+Spectrum with identical prompts and seeds. V2 must also be compared against the equivalent V1 3×5-second graph.

- Once the user has explicitly approved a GPU run, free-VRAM and process-RSS readings are advisory. Do not refuse to start or stop an otherwise healthy run solely because a numeric free-VRAM/RSS threshold was crossed. Continue unless an actual CUDA/PyTorch/Windows OOM or allocation failure, backend crash/health failure, NaN/Inf, or a structural layout/AV/duration integrity failure makes the run unsafe or invalid.

## Reusable local environment

- Source repository: `D:\Codex\_git_push_work\ComfyUI-H3-Continuum`
- Primary tested runtime: `D:\StabilityMatrix\Data\Packages\ComfyUI_W\custom_nodes\ComfyUI-H3-Continuum`
- Secondary runtime: `D:\StabilityMatrix\Data\Packages\ComfyUI_WAN\custom_nodes\ComfyUI-H3-Continuum`
- User-organized V3.8 test videos and related test artifacts are under `D:\output\video\comfy_video\v38`. Reuse that path when inspecting prior V3.8 outputs; do not move, rename, or delete its contents without explicit approval.
- Restart ComfyUI_W with `D:\StabilityMatrix\Data\Packages\ComfyUI_W\venv\Scripts\python.exe`. Do not launch the base executable reported by `Get-Process.Path`; it omits the venv dependencies even though it is the resolved process image.
- When the source adds or changes runtime imports, do not synchronize only the visibly changed files. Install the complete current `REGISTRY_MANIFEST.sha256` set, or first prove a dependency-complete narrow set; then verify registration through the live ComfyUI `object_info` endpoint. A standalone verifier launched from the source checkout is not sufficient evidence for a mixed installed runtime.
- Reuse the repository's existing pytest configuration and helper scripts instead of rediscovering paths on every task.
- Read this file before each task. Record durable project decisions, paths, commands, failure causes, and validation methods here so later work reuses them instead of restarting from zero.
- GitHub authentication for this repository is already configured. Do not request a new login or token unless an actual authentication command fails.
- Commit and push only when the user explicitly requests publication.
