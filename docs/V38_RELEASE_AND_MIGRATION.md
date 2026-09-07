# V3.8 Release and Migration Policy

## Supported release

V3.8 is the current supported H3 Continuum product. Its package exports seven Node IDs:

```text
H3ContinuumSamplerV38
H3ContinuumReferenceAudios
H3ContinuumAssembleSeamV35
H3EasyLoadImage
H3EasyLoadAudio
H3ContinuumLoadVideo
H3ContinuumSecondPassV35
```

The assembler, loader, and Second Pass IDs are retained while their V3.8 display names are simplified. AUDIO-R1 adds one modular Reference Audios helper and appends one optional Sampler bundle socket. The Resolution / Size Source UX keeps the old Aspect input hidden for compatibility and appends Size Source, Width, and Height; saved fixed-Aspect workflows migrate to an equivalent Manual canvas. Sampling, Run Storage, State/Session, and Assembly contracts are unchanged.

## Saved workflows from older releases

Only the seven IDs listed above are exported by V3.8, including the retained Finalize, loader, and Second Pass IDs. Other public Node IDs from V3.7 and earlier are not exported. ComfyUI can therefore show an unknown node when a saved workflow refers to an ID outside this list. This is an intentional support boundary, not a request to replace the unknown node with a superficially similar V3.8 node.

- Use tag `v3.7.0` for V3.7 workflows.
- Use tag `v3.6.0` for V3.6 workflows.
- V3.4/V3.5 archives will be finalized during Release Preparation. Until then, use the matching historical commit or release asset already associated with that workflow.

Do not mix a historical workflow's private schema with the V3.8 package. Install the matching historical package, open and render the workflow there, or rebuild the graph explicitly with the V3.8 public nodes.

## V3.8 workflow and external integration

The declared Registry payload and GitHub distribution use `examples/workflows/MiniMax_H3_Continuum_V38.json` and `examples/workflows/MiniMax_H3_Continuum_V38.zip`. The ZIP contains exactly the same JSON: one graph, not two variants. This is the user-supplied Spectrum workflow, not the former dependency-free Core template. It requires external Spectrum, rgthree, KJNodes, and ComfyUI-Easy-Use nodes, which Continuum does not bundle or install. Its saved default is the Spectrum path; users may switch the same graph to one task-matched LightX2V Turbo LoRA after disabling Spectrum, but all custom-node types remain in the graph.

The supplied JSON and ZIP are preserved byte-for-byte, including prompt, media filenames, custom titles, and settings. Select local files and disable unused inputs before rendering. A saved title such as `Save 3x5s Video` does not control duration; the Sampler settings do. The output path uses Core Video/Audio Decode, Finalize, Core Create Video, and Core Save Video (with the supplied Easy Use cleanup wrapper before Save). External latent processors and upscalers connect through H3 Continuum Second Pass; the old integrated Hi-Res Fix is not part of the V3.8 standard workflow. The supported processor, decoder, and accelerator boundaries are defined in [V3.8 Open Integration Contract](V38_OPEN_INTEGRATION_CONTRACT.md). Registry validate/pack and publication remain separate gates; a GitHub source commit does not claim they have run.

Tests, runner/probe/diagnostic tools, internal development documents, Labs nodes, and historical workflows remain available in the GitHub source for development and audit purposes but are excluded from the Registry archive. The archive retains `tools/__init__.py` and `tools/p20_file_backed_tensor_poc.py`, which are runtime dependencies of the Production file-backed buffer path, plus `tools/verify_runtime.py` for verification. `MANIFEST.sha256` validates its declared source files; the staged archive instead carries `REGISTRY_MANIFEST.sha256`. These development assets are not part of the supported V3.8 searchable node surface.
