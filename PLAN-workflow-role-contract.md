# Role-tagged workflow contract — implemented

> **Status: implemented 2026-08-13.** Kept as the record of why the change was made and what the
> contract is. The user-facing rules now live in README.md under *Modifying the workflow*; this
> file is the reasoning behind them.
>
> **What changed from the plan.** The workflow was replaced at the same time, which moved more
> than the plan anticipated:
>
> - **Tag syntax is `h3-<role>`**, not `H3:<ROLE>` — matching the tags already on the supplied
>   workflow. The tag is the first whitespace-delimited word of the title, lowercased.
> - **Role names follow the workflow's own tags** (`h3-loadmodel`, `h3-vidcombine`,
>   `h3-reference`) rather than the plan's invented `UNET`/`SAVE`/`MINIMAX`.
> - **`ResolutionSelector`, `ComfyMathExpression` and the duration `PrimitiveFloat` are gone**
>   from the workflow. Width, height and length are now written into `MiniMaxH3ReferenceToVideo`
>   as literals, computed by `mathmirror`. That deletes trap 4 rather than warning about it: there
>   is no server-side arithmetic left to drift from. It also retired `--probe` and
>   `add_math_probes` entirely — there is nothing to cross-check — replaced by `--roles` and a
>   *Check the workflow* button.
> - **The sampler swapped twice, which is what proved the design.** First
>   `MiniMaxH3ProgressiveSampler` → `SamplerCustomAdvanced` + `RandomNoise`, then back again.
>   Rather than tracking whichever is current, the seed and the schedule now follow whichever
>   node carries them: `roles.SEED_ROLES` / `SCHEDULE_ROLES` and `graph_builder.seed_node` /
>   `schedule_node`. `roles.REQUIRED_GROUPS` enforces that at least one seed-carrier exists —
>   neither node is required alone, which no single `required` flag can express. The Schedule
>   field hides itself when no node declares one, because writing an undeclared input makes
>   ComfyUI reject the entire prompt.
> - **`SaveVideo` + `CreateVideo` → `VHS_VideoCombine`**, which is both the output node and the
>   muxer.
> - **`config.MULTIPLE` is no longer overwritten onto a workflow node** — the node it was written
>   to is gone. It is now purely the app-side quantisation constant, which is what it always was
>   in spirit.
> - **New**: `_format_dependent_inputs` in the validator. `VHS_VideoCombine` declares `pix_fmt`,
>   `crf`, `save_metadata` and `trim_to_audio` *inside* its `format` combo spec, one set per
>   format, so `/object_info` never lists them at the top level. Deriving the validated class list
>   from the graph surfaced them as four false "unknown input" errors; the validator now reads the
>   `formats` map.
>
> - **Role names track the workflow's own tags.** The prompt node is `h3-promptinput`, not
>   `h3-prompt` — when the tag in the workflow changes, the role name changes with it rather
>   than the two drifting apart.
>
> Defects in supplied workflows caught by the contract rather than by a wrong-node bug: node 125
> once duplicated `h3-sampler` with node 123, and the prompt node was untagged. Both were named
> at startup with the fix, alongside every other problem in the same message.
>
> Scope held: **one workflow file, no picker UI**, and **strict tagging**.

## Context

HARMON3 drives one ComfyUI workflow (`API/video_minimax_h3_r2v_api.json`) and finds every node
it cares about by hardcoded numeric id — 17 `NODE_*` constants and five injected-id blocks in
[config.py:59-120](harmon3/config.py#L59-L120). The registry is disciplined (one block, a startup
assertion, id→label error mapping), but it means the workflow is effectively frozen: any edit the
user makes in ComfyUI is either invisible to the app or silently undone by it.

The goal is a **template workflow the user may modify under a stated contract**. The app stops
identifying nodes by number and identifies them by *role*, declared inside ComfyUI as a node title
tag (`H3:PROMPT`), which survives API export as `_meta.title`. Node numbering then becomes the
user's business, not the app's.

### What breaks today when a node is inserted

ComfyUI numbers new nodes `max+1`, so inserting a node leaves ids 92–149 alone and the constants
keep pointing at the right nodes. Removal is loud and safe — `load_workflow()` hard-fails at
[config.py:376](harmon3/config.py#L376). Nodes inserted *in-chain* (a LoRA between `UNETLoader`
and the guider) are fine, because the app never walks the model chain.

Four things do break:

1. **Orphan pruning silently deletes new branches.** [`prune_orphans`](harmon3/graph_builder.py#L477)
   keeps only ancestors of node 92. A user-added `PreviewImage`, second `SaveVideo`, or upscale
   side-branch is stripped before submit, unreported. This is the biggest trap.
2. **Injected-id collision.** Loaders are written at fixed 200–208 / 220,230,240 / 260–281 and
   probes at 900–902. Base graph maxes at 149, so ~50 added nodes (or a pasted subgraph carrying
   high ids) silently get overwritten. [config.py:112](harmon3/config.py#L112) claims
   `graph_builder` asserts this at import — it does not; `INJECTED_IDS` is computed at
   [graph_builder.py:149](harmon3/graph_builder.py#L149) and never read. Live latent bug.
3. **Pre-flight goes blind on new classes.** [`USED_NODE_CLASSES`](harmon3/config.py#L309) is a
   static tuple, so an unlisted class degrades to a warning at
   [validator.py:145](harmon3/validator.py#L145) and only the server catches errors.
4. **Math-mirror drift.** [mathmirror.py](harmon3/mathmirror.py) reimplements `ResolutionSelector`
   and the 17k+5 frame formula client-side. Rewiring resolution or frame count leaves the readouts,
   ref-video trimming and `reference_bundle/manifest.json` wrong while the render still succeeds.

### Where values actually come from

The final video is produced entirely server-side (`VAEDecode` → `CreateVideo` → `SaveVideo`); the
app only downloads the finished file ([jobs.py:254-277](harmon3/jobs.py#L254-L277)). PyAV is used
locally only for pose rendering, preview decode and reference probing. So output format, codec and
`filename_prefix` are pure pass-through and already follow the workflow.

Inputs fall into three tiers, and the contract must state which is which:

- **Pass-through** — `build_graph` deepcopies the base and writes exactly nine keys; every other
  input survives verbatim (model filenames, `CreateVideo`/`SaveVideo` settings, `BasicGuider`,
  node 131's expression). Workflow edits take effect immediately.
- **Overwritten by a local constant** — exactly one field:
  `resolution_inputs["multiple"] = config.MULTIPLE` at
  [graph_builder.py:265](harmon3/graph_builder.py#L265). Unexposed in the UI, and the workflow's own
  value is discarded every build.
- **Seeded once, then owned by `settings.json`** — the nine exposed parameters plus prompt and sage.
  [main_window.py:91-92](harmon3/ui/main_window.py#L91-L92) reads the workflow then overlays
  settings, so on any existing install a workflow edit to these is a no-op. They are also sanitised
  against local allow-lists (`SAMPLERS`, `SCHEDULERS`, `REF_IMAGE_SIZES`, `MIN/MAX_STEPS`), so a
  value the app does not list is silently replaced.

**Mirrored constants that desync silently.** `FPS = 24` lives in three places: `config.FPS`, node
130's `fps` widget, and hardcoded inside node 131's expression (`round(a * 24)`). Likewise
`FRAME_MOD`/`FRAME_REM`/`MAX_FRAMES` ([config.py:130-137](harmon3/config.py#L130-L137)) mirror that
same expression. This is trap 4 in its concrete form.

## The contract

The tag is the **first whitespace-delimited token** of `_meta.title` when it starts with `H3:`, so
the rest of the title stays a readable label: `"H3:PROMPT Input Text (Prompt)"`.

Required roles (14, one node each): `SAVE`, `RESOLUTION`, `VAE_VIDEO`, `VAE_AUDIO`, `SAMPLER`,
`SCHEDULER`, `SAMPLER_ADVANCED`, `UNET`, `CLIP`, `CREATE_VIDEO`, `MATH`, `DURATION`, `MINIMAX`,
`PROMPT`.

Optional roles: `PREVIEW`, `SAGE`, `SAGE_SWITCH` (absent ⇒ feature off, as today);
`DECODE_VIDEO`, `DECODE_AUDIO` (progress captions only — these replace the two raw literals
`"122"`/`"121"` at [main_window.py:1376-1377](harmon3/ui/main_window.py#L1376-L1377));
`REF_IMAGE_SEED`, `REF_VIDEO_SEED` (multi-bind, replace `BAKED_IMAGE_NODES`/`BAKED_VIDEO_NODES`;
these are always pruned and only seed first-launch defaults, so demoting them from required is a
genuine improvement).

`H3:KEEP` is the escape hatch: any node so tagged is a pruning root, so a user-added preview or
second output branch survives. This is what makes "add a node" work at all.

**Free to change without touching code:** node numbering, node position/layout, adding untagged
in-chain nodes (LoRA loaders, model patches), adding side branches tagged `H3:KEEP`, and any
pass-through widget — model filenames, `SaveVideo` prefix/format/codec, `CreateVideo` bit_depth.

**Changes only until the first launch:** the eleven settings-owned parameters. Re-seeding these
from a modified workflow needs a "Reset to workflow defaults" action (added below), or the value
is ignored.

**Not free:** removing a required role, double-tagging a role, changing a role node's class outside
its accepted set, changing `ResolutionSelector.multiple` (overwritten with `config.MULTIPLE`), or
changing fps / the node-131 expression without matching `config.FPS` and the frame constants.

## Implementation

### 1. New `harmon3/roles.py` (Qt-free, unit-testable)

- `TAG_PREFIX = "H3:"`; `tag_of(node) -> str | None`.
- `RoleSpec(name, class_types: tuple[str, ...], required: bool, multi: bool, description: str)`
  and a `ROLES` registry — the single source of truth, replacing the `NODE_*` block.
- `NodeRoles`: frozen role→node-id map with attribute access (`roles.PROMPT`, `roles.SAVE`),
  `.optional(name) -> str | None`, `.many(name) -> tuple[str, ...]`, `.ids`, and
  `.describe() -> list[tuple[role, node_id, class_type]]` for the diagnostics readout.
- `resolve(graph) -> NodeRoles`, raising `WorkflowContractError` that reports **every** problem at
  once, not first-fail: unknown tag, duplicate tag, missing required role, class mismatch. The
  two `VAELoader`s stop being ambiguous for free — they differ by tag.
- `injected_bases(graph)` returns image/video/audio/probe bases guaranteed above
  `max(int(id) for id in graph)`, bucketed to preserve today's stride layout. Collision becomes
  structurally impossible rather than asserted. Delete the dead `INJECTED_IDS` and the stale
  comment at [config.py:107-113](harmon3/config.py#L107-L113).

### 2. `harmon3/config.py`

- Delete `NODE_*`, `REQUIRED_NODE_IDS`, `BAKED_*`, and the injected-block constants.
- `MODEL_INPUTS` becomes `(role_name, input_name)` pairs; class_type is read from the resolved node.
- `USED_NODE_CLASSES` becomes `classes_for(graph)` — every class present in the loaded workflow,
  plus `INJECTED_CLASSES = {"LoadImage", "LoadAudio", "TrimAudioDuration", "VHS_LoadVideo",
  "PreviewAny"}`. Fixes trap 3; new nodes are now genuinely pre-validated.
- `load_workflow()` returns a `Workflow(graph, roles)` dataclass, calling `roles.resolve()` in place
  of the `REQUIRED_NODE_IDS` check at [config.py:376](harmon3/config.py#L376). Keep the existing
  API-format guard verbatim.
- `defaults_from_workflow(graph)` → `(graph, roles)`; the baked-ref reads at
  [config.py:402-420](harmon3/config.py#L402-L420) use `roles.many("REF_IMAGE_SEED")`.

### 3. `harmon3/graph_builder.py`

- Thread `roles` as a parameter: `build_graph(base, state, roles)`, `add_math_probes(graph, roles)`,
  `canonicalise(graph, roles)`, `canonical_reference(base, roles)`, `state_from_workflow(base, roles)`.
  Every `config.NODE_X` reference becomes `roles.X`. `_image_node_id`/`_video_node_id`/
  `_audio_node_ids` take the bases from `roles.injected`.
- `prune_orphans(graph, keep)`: `keep` = `roles.SAVE` + every node tagged `H3:KEEP`. Fixes trap 1.
  The already-returned `pruned` list should be surfaced in the UI rather than only logged, so a
  user who forgot the tag finds out.
- `_apply_sage` and `_strip_frontend_inputs` already scan by class/link and need only the role
  substitution.
- `INTENDED_DIFFERENCES` keys off `roles.PROMPT`.
- **New** `mathmirror_warnings(graph, roles) -> list[str]`, covering trap 4 concretely:
  - `MATH`'s expression string differs from the one `mathmirror` implements, or `DURATION` no
    longer feeds it;
  - `CREATE_VIDEO.fps` differs from `config.FPS`;
  - `RESOLUTION.multiple` in the workflow differs from `config.MULTIPLE` (i.e. the app is about to
    silently overwrite the user's value).

  Reuse the existing `--probe` path ([jobs.py:348-372](harmon3/jobs.py#L348-L372)) as the
  authoritative server-side cross-check.
- Stop overwriting `multiple` unconditionally: take it from the workflow, and use `config.MULTIPLE`
  only as the fallback when the role node omits it. `mathmirror` already handles the whole
  8–64 range ([config.py:154-156](harmon3/config.py#L154-L156)), so this costs nothing and removes
  the one genuinely hardcoded field.

### 4. Call sites

- [validator.py:252](harmon3/validator.py#L252) `model_preflight(graph, object_info, roles)`.
- [jobs.py:124](harmon3/jobs.py#L124) uses `config.classes_for(graph)`;
  [jobs.py:283](harmon3/jobs.py#L283) uses `roles.SAVE`; the four `load_workflow()` sites unpack
  `Workflow`.
- [cli.py](harmon3/cli.py) — four `load_workflow()` sites plus `cmd_object_info_dump`; the
  `--diff` canonical comparison at [cli.py:56-85](harmon3/cli.py#L56-L85) passes `roles` through.
- [ui/main_window.py](harmon3/ui/main_window.py) — hold `self.roles` beside `self.workflow`
  ([main_window.py:90](harmon3/ui/main_window.py#L90)); `_stage_name` ([:1370](harmon3/ui/main_window.py#L1370))
  builds its map from roles, dropping the two raw literals; `_on_executed`
  ([:1440](harmon3/ui/main_window.py#L1440)) compares against `roles.SAVE`. `_is_sampler`
  ([:1366](harmon3/ui/main_window.py#L1366)) is already class-driven — leave it.
- [ui/settings_panel.py](harmon3/ui/settings_panel.py) Diagnostics group: a **Workflow contract**
  readout from `NodeRoles.describe()` — role, bound node id, class — plus the `H3:KEEP` list, the
  pruned-node list, and any mathmirror warnings. This is how the user checks their edit before
  queueing.
- [ui/params_panel.py](harmon3/ui/params_panel.py): a **Reset to workflow defaults** action that
  re-runs `state_from_workflow` and discards the settings overlay for the eleven owned parameters.
  Without it, a user who edits steps/sampler/duration in ComfyUI sees no change and has no way to
  find out why.

### 5. Data + docs

- Add `H3:<ROLE>` prefixes to the `_meta.title` of the 21 role-bearing nodes in
  `API/video_minimax_h3_r2v_api.json`. Values and wiring untouched, so `--dry-run --diff` must stay
  clean apart from `INTENDED_DIFFERENCES` (`canonicalise` already drops `_meta`).
- README: a "Modifying the workflow" section stating the contract, the tag syntax, `H3:KEEP`, and
  the three-tier free/seeded/not-free lists above — including the fps-in-three-places rule and the
  fact that exposed parameters are owned by `settings.json` after first launch. NOTES.md: why role
  tags replaced numeric ids, and the four traps.

### 6. Tests

Update `tests/test_graph_builder.py:450` (hardcoded `"126"`) and the probe-id assertions at
`:526-529` to resolve through roles. New `tests/test_roles.py`:

- tag parsing incl. `"H3:PROMPT Input Text (Prompt)"`;
- duplicate tag / missing required role / class mismatch each raise, and the error lists *all*
  problems;
- a renumbered workflow (every id +1000) builds identically after canonicalisation;
- an untagged in-chain node (LoRA between `UNET` and the guider) survives the build;
- an `H3:KEEP`-tagged side branch survives pruning, and the same branch untagged is pruned *and*
  reported;
- a workflow with 60 extra nodes gets non-colliding injected ids;
- `mathmirror_warnings` fires for a changed node-131 expression, a `CreateVideo.fps` other than 24,
  and a `multiple` the app would have overwritten;
- a workflow shipping `multiple: 16` builds with 16, not 32.

## Verification

1. `pytest` — full suite green.
2. `python -m harmon3 --dry-run --diff` — must report no differences beyond
   `INTENDED_DIFFERENCES`. This is the proof the retitled JSON and the role-resolved builder
   produce the byte-identical graph the numeric version did.
3. `python -m harmon3 --check` against a running ComfyUI — model preflight and validation clean.
4. `python -m harmon3 --probe` — server-reported width/height/frames match `mathmirror`.
5. Hand-edit a copy of the workflow: renumber nodes, insert an untagged LoRA loader in the model
   chain, and add an `H3:KEEP`-tagged `PreviewImage` branch. Launch the GUI, confirm the Workflow
   contract readout binds all 14 required roles, then queue a short render and confirm the video
   downloads and the preview branch executed.
6. Negative case: remove the `H3:PROMPT` tag and confirm startup fails with a message naming the
   missing role, not a `KeyError`.

## If only one thing gets done

Traps 1 and 2 are live bugs independent of the role work and are cheap in isolation: give
`prune_orphans` an `H3:KEEP` root set, and compute the injected-id bases from `max(id)+1` instead of
the fixed 200/220/260/900 blocks.
