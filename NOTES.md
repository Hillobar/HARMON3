# HARMON3 — findings and progress

Engineering notes. [README.md](README.md) covers how to use the app; this covers what was
learned building it and what state it is in. Most of what follows cost real time to
discover, so it is written down rather than left in the code.

Last updated after per-reference sizing, projects, and the reference bundle.

---

## Status

**Working and verified end to end against a live ComfyUI 0.31.0.** Pose estimation
verified on real footage on an RTX 5090: 124 frames in 7.6 s, skeleton and audio correct.

| | |
|---|---|
| Source | 40 files, ~14,300 lines |
| Tests | 29 files, ~8,400 lines, **940 passing** |
| Workflow | `API/video_minimax_h3_r2v_api.json` (17 nodes, found by `h3-` role tag) |
| Python | 3.12, own `.venv` |
| Dependencies | PySide6 6.11.1, requests, websocket-client, av |
| Pose extras | rtmlib, onnxruntime-gpu (CUDA 13), numpy, opencv — see `requirements.txt` |

### Features

- Reference loaders — up to 9 images, 3 videos, 3 audio; drag and drop; per-kind
  remembered folders; live `<Picture i>` / `<Video k>` / `<Audio j>` tags with drift
  detection and one-click rewrite.
- **Drag rows by their grip to reorder them**, within a kind. The order is the numbering,
  so this is how a reference is given a particular tag.
- Prompt in six collapsible sections, combined into one string on send.
- Resolution and duration computed client-side and written into the graph as literals,
  with live readouts.
- One **Parameters** panel: resolution, duration, steps, sampler, scheduler, schedule,
  stage upscale, sigma shift, `ref_image_size` and seed, none of it folded away. The
  dimension grid is fixed at 32 rather than exposed. Schedule, stage upscale and sigma
  shift each hide themselves on a workflow with no node that declares them.
- Seed with randomize.
- **References own the files; the result frame owns the mark** — files are added in the
  reference panel, clicking one opens it in the frame (paused), and the in point set there
  is where the section sent begins. A reference is a name plus one mark.
- **Trim marks** — set on the timeline in the result frame, editor-style: a draggable in
  handle, I at the playhead, playback looped inside the section. Stored in frames for
  video, seconds for audio. There is **no out point and no on/off switch**: every video
  and audio reference is cut to the generated length, so *Duration* is the out point and
  the only decision left is where the cut starts.
- **One timeline** — the same track scrubs whatever the frame is showing and carries the
  in point when that is a reference. The marking controls hide themselves for anything
  else.
- **Pose** — a per-video toggle that sends a skeleton of the clip instead of the clip.
  Estimated locally (ViTPose-L over ONNX Runtime, CUDA when available), over just the
  section that will be sent, before the run is queued. Cached; renderable on demand from
  the row's `POSE?` thumbnail; cancellable on the spot.
- **Result transport** — Play/Stop/scrub/volume follow whatever the frame is showing,
  reference clip or finished video.
- **Sage Attention toggle** in Settings, driving the workflow's own switch. Off removes the
  patch node from the submitted graph rather than bypassing it.
- **Live sampler preview** — animated, decoded from the preview node's MP4 stream.
- **ETA** rather than a bare step counter.
- **Pose model, style and cache** in Settings — three ViTPose weights including
  wholebody, two skeleton topologies, and a button that clears every rendered clip.
- **Per-reference size ceilings** — a slider on every image and video row deciding how
  much of it is sent. Images are a share of their own file; clips are a share of the
  canvas the node fits every clip to. Applied by preparing a copy locally, so the row
  always names the user's own file. See *Reference sizing* below.
- **Prompt format helpers** — a row of clickable chips under each section carrying the
  output format's own vocabulary, plus live chips for every `<Subject N>` the prompt
  defines. Behind a *Helpers* toggle; the guide itself opens beside the editor.
- **Projects** — scenes grouped into the finished video they are pieces of, with a
  running order and a running total. Drag a scene onto a project to file it.
- **Frame stepping** — the wheel over the picture or the timeline moves one frame,
  ten with Shift.
- **Export references** (Settings → Diagnostics) — writes exactly what the node is about
  to be given: the files as they will be uploaded, the prompt, and a manifest of what the
  node then resizes each one to.
- **Scenes** — named, described, reusable; own folder, configurable in Settings.
- Run history with replay and verbatim re-queue.
- Settings tab; techno-cyber theme.

### Verification

Five tiers, cheapest first. The first three cost no GPU time.

| Tier | What it proves | State |
|---|---|---|
| 0 `--roles` | Which node plays which part, and what is untagged | 17 of 17 bound |
| 0 `--dry-run --diff` | The built graph vs the shipped workflow | **Only the prompt and the reference node's geometry differ**, both named |
| 1 `--check` | Every node validates against live `/object_info` | 17 nodes OK |
| 2 `--upload-test` | Upload → `/view` round trip | OK |
| 3–4 live runs | Real generations end to end | Many, all successful |

`--probe` is gone. It submitted with `partial_execution_targets` to make the server
evaluate `ResolutionSelector` and the frame-count expression, so their numbers could be
compared against this side's mirrors. Those nodes no longer exist and the app writes the
numbers as literals, so there is nothing left to disagree with.

`--pose CLIP --out X` sits beside these: it runs the estimator with no server involved, so
the model, the threshold and the drawing can be judged in one command. It is also the only
tier that catches a rendering fault, which bug 16 says is worth having.

Live runs additionally confirmed: preview animation, cancel mid-sample, re-queue, the
marked section reaching the model, and the combined prompt arriving at the
`h3-promptinput` node.

---

## Findings: ComfyUI contracts

Verified from source at `D:\SD\Generation\Comfyui_01`, not assumed.

### `MiniMaxH3ReferenceToVideo`

- Autogrow inputs flatten to dotted keys: `ref_images.ref_image_0…8`,
  `ref_videos.ref_video_0…2`, `ref_video_audios.ref_video_audio_0…2`,
  `ref_audios.ref_audio_0…2`.
- **`ref_videos` takes `IMAGE`, not `VIDEO`** — a frame batch. `VHS_LoadVideo` yields one
  directly on slot 0, and its slot 2 is the matching audio.
- A soundtrack pairs to its video **purely by trailing index**
  (`ref_video_audio_ + name.rsplit("_",1)[-1]`). Sparse indices are legal; omit the key
  rather than passing null.
- Expansion order is by numeric index, not JSON key order.
- `length`: min 5, max 3600, step 17, snapped to `n % 17 == 5`. **The largest reachable
  value is 3592 frames = 149.6 s** — nothing else catches an over-long duration, so it
  must be clamped client-side.

### Reference sizing — read from `nodes_minimax_h3.py`, not inferred

This is the part nobody can see from the outside, and it decides what the model actually
looks at. `harmon3/scaling.py` mirrors all of it so the app can say so.

**Images** are scaled by one factor applied to both axes, then each axis is rounded to a
multiple of 32:

```python
scale = min(1.0, sqrt((width*height) / (w*h)))   # ref_image_size = "match"
scale = min(1.0, 2048 / min(w, h))               # ref_image_size = "max"
```

Two consequences, both load-bearing:

- **`min(1.0, ...)` in both branches — a reference is capped, never enlarged.** That is
  the whole basis of the per-image size ceiling: an image made smaller before upload stays
  smaller, because the node will not raise it back. Both modes are the same code path;
  only the formula for `scale` differs, so this is preprocessing rather than a separate
  pipeline.
- **The aspect ratio survives.** The only distortion is the independent 32-grid snap,
  measured at ≤3.6% across a range of tall references and usually under 2%. A reference
  arriving visibly stretched is therefore never the node's sizing.

**Videos never consult `ref_image_size` at all.** Every clip is fitted to `adapt_canvas`:
a 768 short edge capped at 768×1344 of area, rounded to 32 — so 3840×2160 and 1920×1080
both encode at 1344×768. The exception is its own never-upscale guard, `if vw*vh < cw*ch`,
which keeps a clip already smaller than the canvas at its own size. This is why a video's
size slider is a share of the *canvas* rather than of the file: a share of a 4K source
would move 3840→1920 and change nothing.

Costs, for a 1344×768 generation, as `(w//16)*(h//16)` tokens riding every sampling step:

| Reference | `match` | `max` |
|---|---|---|
| 4000×3000 | 1184×864 — 3,996 | 2720×2048 — **21,760** |
| 1024×1024 | 1024×1024 — 4,096 | 1024×1024 — 4,096 |
| any clip ≥1 MP | 1344×768 — 4,032 per latent frame | same; the setting is ignored |

`ref_image_size` now defaults to **`max`**, because each reference carries its own ceiling
and `match` would re-cap them all at the generation's area.

### EXIF orientation — ComfyUI applies it, Qt does not

`LoadImage` runs `ImageOps.exif_transpose` (nodes.py). Qt's `QImage(path)` returns the
stored pixels, and `QImageReader.size()` reports the stored size *even with
`setAutoTransform(True)`* — only `read()` applies the rotation. Everything that opens a
reference image goes through `harmon3/imaging.py` for that reason. See bug 20.

### Prompt tag ordinals

Assigned in `execute()` purely from the order and kind of connected inputs: **all images,
then each video with its soundtrack's `<Audio j>` emitted *before* its `<Video k>`, then
standalone audio.** So two images + one video with soundtrack + one standalone audio gives

```
<Picture 1>  <Picture 2>  <Audio 1>  <Video 1>  <Audio 2>
                          ↑ the soundtrack        ↑ the standalone
```

This is the app's sharpest usability hazard: unticking one soundtrack silently renumbers
every later `<Audio j>`, so a prompt keeps reading correctly while pointing at a different
asset. Hence the live diff and the simultaneous rewrite — a chained `str.replace` would
corrupt overlapping moves (1→2 then 2→3 turns the original 1 into 3).

### Node input names that surprise

| Node | Input | Note |
|---|---|---|
| `LoadVideo` | `file` | not `video` |
| `LoadImage` | `image` | filename relative to the input dir |
| `LoadAudio` | `audio` | accepts video containers too |
| `MiniMaxH3ProgressiveSampler` | `noise_seed` | not `seed`, and a **widget** rather than a NOISE link — the node makes its own noise, so the graph carries no `RandomNoise` |
| `ComfyMathExpression` | `values.a` | **required, and must be a link** — not a literal |

`ComfyMathExpression` outputs are slot 0 = FLOAT, 1 = INT, 2 = BOOL.
`GetVideoComponents` outputs images, audio, fps, bit_depth.

### Server API

- **`/upload/image` is the only upload route.** It takes audio and video too, on the same
  `image` form field. Response `name` is authoritative.
- `LoadImage`, `LoadAudio` and `LoadVideo` each declare a custom validator naming their
  filename argument, which **disables the combo-membership check** (`execution.py`). That
  is why uploading into `input/harmon3/` works even though `os.listdir` is non-recursive
  and such files never appear in the enumerated options.
- `/prompt` accepts a **client-supplied `prompt_id`** (canonical lowercase UUID), which
  makes job correlation race-free.
- `/prompt` accepts **`partial_execution_targets`**; `validate_prompt` only walks the
  ancestors of the listed outputs. This is what makes Tier 2 possible — the server
  computes the resolution arithmetic with **no model loaded**.
- `/view` takes the basename in `filename` and the directory in `subfolder`; do not put
  the subfolder in the filename.
- Some websocket messages are broadcast (`execution_interrupted`), so **every event must
  be filtered by `prompt_id`** or a browser tab on the same `--listen` server drives this
  app's progress bar. With several of our own runs outstanding the filter is "is this one
  of ours", not "is this *the* one" — the server names which, and taking its word for it
  beats assuming.
- **`/interrupt` takes no prompt id.** It stops whatever is executing, whoever queued it.
  Cancelling a run that has not begun is `POST /queue {"delete": [id]}`; only a run the
  server has said is executing may be interrupted, or a shared server loses somebody
  else's job to our Cancel button.
- The queue is a FIFO, so the oldest of *our* outstanding prompts is the one being worked
  on. That is the whole basis of `harmon3/runqueue.py` — no polling of `/queue` needed.
- `SaveVideo` embeds the API prompt in the mp4 metadata unless `--disable-metadata` — a
  free cross-check on what produced a file.

### `ModelPreviewOverrideKJ` (KJNodes)

- Does **not** use ComfyUI's binary preview stream. It emits its own JSON message,
  `kj_preview_override`, addressed to `PromptServer.instance.client_id` — the client that
  submitted the prompt, i.e. this app.
- Payload is base64: fragmented H.264 MP4 when the server's PyAV has NVENC (**it does
  here**), animated WebP otherwise, plain JPEG when `preview_frames == 1`.
- **`suppress_default_preview: true` switches the binary preview stream off entirely.** A
  workflow with this node has only this preview.
- Also reports `step`, `total`, `sigma`, `delta`, `step_ms`, `avg_step_ms` — the averaged
  step time is measured at the sampler, so it beats anything inferred from message arrival.

### `VHS_LoadVideo` (VideoHelperSuite)

- Outputs `(IMAGE, frame_count, AUDIO, VHS_VIDEOINFO)`. One node replaces
  `LoadVideo` → `GetVideoComponents` → `ImageFromBatch` → `TrimAudioDuration`.
- **`skip_first_frames` and `frame_load_cap` limit what is *decoded*,** not what is kept
  afterwards. This is the whole reason to use it — see bug 15.
- **The audio is windowed to the same span, for free.** From the source:
  `start_time = skip_first_frames * target_frame_time` and
  `lazy_get_audio(video, start_time, frame_load_cap * target_frame_time)`. It derives both
  from the source's own frame time, so frames and sound cannot drift apart and the app
  needs no probed frame rate to line them up. It is lazy, so an unused soundtrack costs
  nothing.
- `frame_load_cap: 0` means "no limit", so a sub-frame window must never round to 0.
- **`format` rounds dimensions, not the frame rate.** `target_rate` is never consulted on
  the load path — only `force_rate` resamples. What `format` supplies is `dim`, whose first
  element becomes `downscale_ratio`, and `target_size()` rounds width and height to a
  multiple of it *even when* `custom_width`/`custom_height` are 0. `AnimateDiff`'s dim of 8
  turns a 480×854 clip into 480×856; `None` (no dim, ratio 1) leaves it alone.
- Its COMBO lists only the input *root*, but it declares `VALIDATE_INPUTS(s, video)` and
  resolves through `folder_paths.get_annotated_filepath`, so `harmon3/…` uploads are fine.

### `Switch` and `PathchSageAttentionKJ`

- `Switch` takes `on_true` and `on_false` and returns one — but **both are wired, so both
  branches are ancestors of the output and ComfyUI executes both.** Setting the switch is
  therefore not enough to avoid running a node; the app repoints the switch's consumers at
  the loader instead, and the orphan sweep takes the patch and the switch out.
- `PathchSageAttentionKJ` (the typo is the node's own) patches a cloned model's attention.
  The workflow ships it with `allow_compile: True`, which is not something to run for a
  setting that is switched off.

### Pose estimation (local, in `harmon3/pose.py`)

- **It is not a ComfyUI node any more.** `DWPreprocessor` is long gone from the shipped
  workflow. A ticked *Pose* row is rendered here, by ONNX Runtime, into an ordinary mp4
  that goes through the normal upload path — so `graph_builder` contains no mention of
  pose at all, and a test asserts that.
- **The odd-dimensions trap survived the rewrite.** H.264 with yuv420p refuses odd width
  or height, so a pose *video* silently fails to encode. The old in-graph fix was
  `GetImageSize` → `ComfyMathExpression` (`a - a % 2`) → `ImageScale`; the local one is
  `pose.even()`, applied to the source's dimensions before the encoder is opened.
- **Only the section that will be sent is posed** — from the mark, for the generated
  length. The rendered clip therefore *starts* at the mark, so its row must submit
  `skip_first_frames = 0`. Both halves of that live in `pose.swap_in`, which is the only
  place they are stated together, and it runs on the job snapshot so the row on screen
  keeps naming the user's own file.
- **`draw_skeleton` picks its skeleton from the keypoint count**, and reads a bare 17 with
  `openpose_skeleton=True` as *animal17*. Eighteen is what selects the human OpenPose
  skeleton, which is why `coco17_to_openpose18` exists — `rtmlib` carries that conversion
  inside its `RTMPose`/`RTMO` classes but not its `ViTPose` one.
- **A pose clip carries the source's audio.** Not a nicety: a video whose soundtrack goes
  missing stops emitting its `<Audio j>` tag, which renumbers every standalone audio after
  it and quietly repoints the prompt.
- **`onnxruntime.preload_dlls()` is what makes the CUDA wheels visible.** This venv has no
  torch and no system CUDA toolkit, so without it `get_available_providers()` reports no
  CUDA provider even with `onnxruntime-gpu` installed.
- **`draw_skeleton` returns the drawn image; it does not reliably draw in place.** The
  OpenPose style alpha-blends its limbs, and `cv2.addWeighted` hands back a *new* array —
  so a canvas passed in by reference comes back completely black, and every later circle
  and line lands on a copy nobody keeps. Cost an hour: the pipeline reported 124 frames
  written, 0 held, and every one of them empty. Use the return value.
- **The render goes to a `.part` and is renamed when whole.** Cancelling half way would
  otherwise leave a truncated clip at exactly the name `resolve()` looks for, and the
  next run would send six frames of skeleton without a word about it.
- **Measured on an RTX 5090, ViTPose-L + YOLOX-x, 640×480 source:** 124 frames in 7.6 s,
  about 60 ms a frame steady state, ~100 ms including session setup. A 20-second
  generation (481 frames) is under a minute. The weights are 1.2 GB (ViTPose) plus 351 MB
  (YOLOX); rtmlib keeps the detector in `~/.cache/rtmlib`, not in `models/pose`.

### The `figure` style (`harmon3/posefigure.py`)

- **Front and back are very nearly mirrors in 2D**, which is what made the rendered clips
  hard to read. `Facing` votes on five cues — shoulder order, hip order, ear order, toe
  medial order, mean face confidence — each weighted by keypoint score, EMA-smoothed, and
  **latched** through a dead band so a subject near profile holds its last verdict instead
  of flickering. The face is then drawn *only* when the verdict says frontal: the estimator
  regresses face keypoints rather than detecting them and will place a full set of eyes and
  a mouth on the back of a head.
- **The cues project onto the body's own lateral axis**, not image x, and divide by the
  torso *length*. Shoulder width collapses at profile, so using it as the denominator would
  report profile as a confident verdict either way; and raw image x gets a subject lying
  down backwards. Costs three lines.
- **Only a fresh estimate votes.** `_render_frames` re-draws `last_pose` when the estimate
  fails, and folding those keypoints in again would count one frame's evidence many times
  and talk the estimate into whatever it already believed.
- **`draw_openpose` ignores the `radius` argument.** It hardcodes 3 or 4 pixels and the
  scaling line is commented out in the library, so `draw`'s `radius * 2` has always been
  discarded and joints never scaled with the frame. Links past index 16 are likewise a
  hardcoded 2px line.
- **rtmlib's alpha is a no-op.** `draw_polygons` does `fillConvexPoly(img.copy(), ...)` and
  then `addWeighted(img, 1-a, img, a)` — the filled image blended against *itself*. Nothing
  is ever translucent; the only real effect is the copy. So draw *order* already controls
  occlusion, which is what lets this style paint the trunk over the arms when the subject
  faces away and under them when they do not. `posefigure.bone` reproduces the behaviour
  rather than correcting it, so the two OpenPose styles keep their pixels.
- **`openpose134` has no face links and no foot links.** Its 57 links are the body (0-16)
  and the two hands (17-56). The 68 face points were drawn as loose white dots and the six
  foot keypoints carry `color=[0, 0, 0]`, which `draw_openpose` skips outright — so picking
  the wholebody model bought face dots and hand hairlines and nothing else. Both tables are
  defined in `posefigure`; the feet had never been drawn at all.
- **`openpose134`'s `keypoint_info[i]["id"]` is off by one from 18 to 91.** Index 18
  (`left_big_toe`) carries `id=17`, colliding with `left_ear`, and it corrects itself at 92.
  `draw_openpose` builds its lookup from `id` but only ever indexes the body and the hands
  with it, so the bug has never had anything to break. Key by dict index, never by `id`.
- **A chord across the nape reads as a mouth.** The back-of-head mark started there and had
  to move to an arc over the crown, which reads as a hairline and cannot be mistaken for the
  one thing it must not say.
- `skeleton_for` used to treat "not the default style" as "the torso style" and applied
  `_TORSO_LINKS` unconditionally, so a style with no rtmlib table behind it would have been
  handed a plausible wrong answer. It is keyed on `_SUBSTITUTIONS` now.

### Getting ONNX Runtime onto the GPU (all four of these cost time)

- **`rtmlib` depends on plain `onnxruntime`, and the CPU and GPU packages install over the
  top of each other.** Whichever lands last wins. When the CPU build won,
  `get_available_providers()` returned `['AzureExecutionProvider', 'CPUExecutionProvider']`
  while `onnxruntime_providers_cuda.dll` sat in the same directory — the pybind module is
  the CPU one and does not know CUDA exists. pip cannot express "this dependency, but the
  GPU flavour", so `setup.bat` uninstalls the CPU build and force-reinstalls the GPU one
  afterwards.
- **NVIDIA's CUDA 13 pip names are inconsistent, on purpose.** cuBLAS and the runtime
  dropped the suffix and are versioned 13.x (`nvidia-cublas`); cuDNN kept it and is
  versioned 9.x (`nvidia-cudnn-cu13`). `nvidia-cublas-cu13` *does* exist on PyPI as a
  0.0.1 stub with no wheel, which fails to build and looks like a broken environment.
- **cudart, cuFFT and cuRAND are not optional.** With only cuBLAS and cuDNN the provider
  registers and then fails on its first session with "Failed to load cudart64_13.dll".
  The symptom is a CUDA provider that is listed but never used.
- **`onnxruntime-gpu>=1.27`** is the CUDA 13 line: 1.27 is both the first release built
  against CUDA 13 and the release that drops CUDA 12, so the pin has no overlap to get
  wrong.
- **OpenMMLab's SDK zips are named with a per-model hash.** There is no `yolox_l` build
  published; pairing the `l` name with another model's hash 404s on OpenMMLab *and* on
  rtmlib's HuggingFace mirror, which reads like a network problem and is not one.

### Client-side mirrors

`ResolutionSelector` and the duration expression are reproduced in `mathmirror.py` so the
GUI can show live values. Both use Python's **banker's rounding** — ComfyUI calls the
builtin `round`, and `simpleeval` resolves `round` to the same builtin, so `int(x + 0.5)`
would disagree on exact .5 values. Anchors: `(16:9, 0.4 MP, 32) → 864×480`; `5.0 s → 124
frames`. Tier 2 checks these against the server on demand.

---

## Findings: Qt / PySide6 pitfalls

Every one of these was hit for real.

| Symptom | Cause | Fix |
|---|---|---|
| `OverflowError` on the seed | **`QSpinBox` is 32-bit**; the workflow's seed is 157368968253448 | validated `QLineEdit` |
| Signals silently not delivered | `Signal(dict)` marshals through `QVariantMap`, which cannot hold the uint64 bounds in node schemas | declare payloads as `Signal(object)` |
| **Hard crash (segfault)** | `QBuffer(QByteArray(payload))` holds a pointer to a Python temporary that is then freed | `QBuffer.setData()`, which copies |
| A whole panel inert | `textChanged` carries a `str`; a zero-arg `Signal.emit` **raises** on the extra argument (unlike a plain callable, where PySide truncates it) | wrap in a lambda |
| `QtMultimedia` missing | **PySide6-Essentials 6.11.1 no longer ships it** | the full `PySide6` metapackage |
| Every glyph a tofu box | Qt's stylesheet parser mishandles comma-separated `font-family` fallback lists | `QFont.setFamilies()`; no `font-family` in the stylesheet at all |
| MP4 preview silently blank | `frame.to_ndarray()` needs numpy, which **PyAV does not require** | decode from the frame plane buffer, using QImage's row stride |
| Tests that never elide | `resize()` does not deliver a resize event until a widget has been shown | `WA_DontShowOnScreen` + `show()` |
| Offscreen renders all tofu | the offscreen platform reports **zero font families** | screenshot with the real platform + `WA_DontShowOnScreen` |

Qt stylesheets also have no `text-transform` and no `letter-spacing`, so the uppercase
panel headings are applied to the widgets in `style.stylise()`.

---

## Bugs found and fixed

A review pass plus the live tests turned these up. All are fixed and regression-tested.

**Concurrency and state**

1. The job worker shared **live `RefRow` objects** with the UI. Editing a reference
   mid-upload could pin the row to the wrong file *permanently*, since the stale name was
   persisted. Submissions now take a deep-copied snapshot.
2. A local file's server name was persisted; editing the file in place kept submitting the
   old bytes. Now recomputed from content hash every submit.
3. `upload_cache` was handed to the worker by reference; `json.dump(indent=...)` uses the
   pure-Python encoder and iterates as it writes, so a concurrent insert raised
   `RuntimeError` mid-save — from `closeEvent`, that skipped the thread shutdown.
4. Queue re-enabled while a submit was in flight → double submission, orphaned run.
5. `closeEvent` used `BlockingQueuedConnection`, freezing the window for up to the
   5-minute HTTP timeout.

**Correctness**

6. Per-row server-error highlighting was dead code — the built graph was cleared before
   the guard that needed it. `JobFailure` now carries the labels.
7. The result downloaded twice: `executed` started it and `execution_success` started it
   again before the first finished.
8. A re-queued run recorded the *current editor* state instead of the re-queued run's.
9. `MediaProbe` leaked players and cancelled the wrong one on rapid re-probes.
10. `ElidedLabel`'s constructor bypassed its own `setText`, so it neither elided nor set a
    tooltip — defeating the entire point of the class.
11. With randomize on, every queue rolled a new seed and marked the scene **modified** —
    defeating the point of a randomized scene. The seed is now excluded from the
    comparison when randomize is on.
12. **Undecodable files were uploaded and submitted without being checked.** A zero-byte
    or truncated file went into ComfyUI's input folder and failed several minutes later
    inside `GetVideoComponents` with `moov atom not found` — a server stack trace for
    something knowable locally in microseconds. The probe now returns a verdict as well as
    metadata: empty is refused outright, and a file neither PyAV nor Qt can open is marked
    unreadable, which blocks the row and says so in the result frame. `_upload` refuses an
    empty file as a last guard, for one emptied after the row was checked.
13. The result frame went black whenever it was not playing. `QMediaPlayer.stop()` tears
    the decode pipeline down and takes the video surface with it, and while a player is
    stopped a seek moves the position **without presenting a frame** — so Stop, the end of
    a clip, and any scrub afterwards all left a black rectangle. Every stop is now a pause
    plus a rewind (`player.park()`), which keeps the pipeline up so the frame seeked to is
    the frame shown. Measured, not assumed: `tests/test_player_frames.py` asserts both
    halves against a real decoded clip through a `QVideoSink`.

14. **The app died without a traceback when a reference video was played.** Making the
    frame open *paused* meant re-applying the pause once the media loaded, and that was
    wired as a closure connected to `mediaStatusChanged` which disconnected itself from
    inside its own slot — mutating Qt's connection list while it was emitting through it,
    while `pause()`/`setPosition()` re-entered the same signal. A segfault, not an
    exception, so nothing reached the log. Now: one connection made once for the life of
    the pane, a flag cleared *before* the work, and the pause and seek deferred to a zero
    timer so they never run inside the emission. `VideoPlayer._on_media_status` had the
    same shape and got the same treatment.

    Worth recording that it was reproducible all along — it killed several of my own
    verification scripts earlier and was wrongly written off as a headless-environment
    artefact. A segfault in a test harness is evidence, not noise.

15. **A reference video was decoded in full, into CPU RAM, before anything else ran.**
    Reported as "the app takes far longer to initialise than the same workflow in ComfyUI;
    it looks like it loads into CPU memory before starting". It did. The builder wired
    `LoadVideo` → `GetVideoComponents` → `ImageFromBatch`, and `GetVideoComponents`
    materialises the **entire file** as a float32 IMAGE batch before the slice throws most
    of it away:

    | | frames | as float32 |
    |---|---|---|
    | app, whole 226 s reference at 640×480 | 6,777 | **25.0 GB** |
    | capped at the generated length | 124 | 0.5 GB |

    The shipped workflow never had the problem because its `VHS_LoadVideo` links
    `frame_load_cap` to the same math node that computes the generated length. The app now
    does the same, and uses `skip_first_frames`/`frame_load_cap` for the marks, so only the
    section that will be sent is ever decoded. `frame_load_cap` is never left open: even
    unmarked it is capped, because MiniMax truncates a reference to the generated length
    anyway and decoding more is work whose output is discarded.

    The same node also asked for a 1536px height, which turned that 640×480 clip into
    roughly 2736×1536 — 6.3 GB for 124 frames against 0.5 GB at native size, for a picture
    `adapt_canvas` then scales back down. Both custom sizes are now 0 — first in the app,
    and since 2026-08-11 in the shipped workflow too, so the loader is no longer a
    departure from it at all and came back out of `INTENDED_DIFFERENCES`.

    The assembled prompt is now the *only* deliberate departure from the workflow, listed
    in `graph_builder.INTENDED_DIFFERENCES` with a reason. Tier 0 excuses exactly that and
    nothing else, and a test asserts each entry still describes a real difference, so a
    stale excuse cannot quietly cover a new one — which is exactly what caught the loader
    once the workflow changed.

16. **Every pose frame came out black, and the pipeline said it had worked.** The first
    full render reported "124 frames written, 0 held" — meaning the estimator had found a
    person in every single frame — and produced 124 entirely empty pictures. Nothing in
    the run was wrong except the last step: `rtmlib.draw_skeleton` was being called for
    its side effect on a canvas passed in by reference. Its OpenPose path alpha-blends
    each limb, `cv2.addWeighted` returns a *new* array, and from the first limb onward
    everything is drawn on a copy that is then dropped. The keypoint circles come after
    the limbs, so not even those survived. **Use the return value.** The confidence of the
    progress output is the lesson: "0 held" only proved keypoints existed, not that any of
    them reached a pixel. Worth having a check that actually looks at the output.

17. **The button to render a pose only appeared once a pose had been rendered.** The pose
    thumbnail was shown when `row.pose_path` pointed at a real file, which is exactly the
    condition under which you no longer need to ask for one. The render-on-demand path was
    written, wired, tested at the handler level and reachable from nowhere on screen, so
    the only way to get a first skeleton was to queue a run — the one thing the feature
    existed to let you avoid. Found by the user asking where to press it.

    The handler test passed throughout, because it called `_on_ref_preview(row, "pose",
    …)` directly instead of clicking anything. A test that invokes a handler proves the
    handler works, not that anybody can reach it; this one now emits the widget's own
    `clicked` signal and asserts the widget is visible before the render exists.

18. **The wholebody pose model downloaded its weights and then failed on the first
    frame.** `vitpose-l-wholebody` outputs 133 keypoints; `draw_skeleton` with
    `openpose_skeleton=True` accepts 17, 18, 26 or 134 and raises `NotImplementedError` on
    anything else. Only COCO-17 was being converted. Past the body the two layouts are the
    same list displaced by one, because OpenPose inserts the neck at index 1 — so the fix
    is the existing conversion one size up. Conversion is now table-driven, and a layout
    with no entry fails when it is *chosen* rather than minutes into a render.

19. **The pose renderer cropped rather than resized.** `frame.to_ndarray()[:height,
    :width]` was a no-op while the canvas was always the source's own size, and became a
    silent crop the moment a scaled canvas was allowed — a corner of the clip, sent as the
    clip. Found by reading it while adding the video size ceiling, not by a test.

20. **EXIF orientation, twice.** A photograph stored sideways reached the model upright,
    because ComfyUI rotates it — but the app showed it sideways, reported its dimensions
    transposed, and, worst, *wrote a rescaled copy from the stored pixels*. PNG carries no
    EXIF, so a scaled reference arrived sideways while the same reference unscaled arrived
    upright. Fixed by routing every read through `imaging`.

    Then it came back. The first fix patched the `kind == "image"` branch of `MediaProbe`
    — which PyAV never let us reach, because PyAV was asked first and opens a still
    perfectly happily as a one-frame video, reporting the *stored* dimensions. So the row
    recorded 1000×600 for a picture that was really 600×1000, every size was computed for
    a landscape image, and `write_scaled` squashed the portrait picture into a landscape
    box with `IgnoreAspectRatio`. **Stills are now asked of Qt before PyAV**, and
    `write_scaled` no longer trusts the caller's shape: a request that disagrees with the
    loaded picture by more than a tenth keeps the pixel budget and the picture's own
    proportions, with a warning. The lesson is that the first fix was reasoning about the
    right function and the wrong caller.

21. **Drag and drop was refused for the whole gesture.** `dragMoveEvent` read
    `dropIndicatorPosition()` *before* calling `super()` — but `super()` is the only thing
    that computes it. The first move saw a stale value, refused, and so never ran
    `super()`, so the value stayed stale: a forbidden cursor, permanently. The accept is
    now ours rather than the base class's, which decides from model flags describing a move
    it is never going to perform.

22. **A wheel over `QVideoWidget` never reaches its parent.** Frame stepping was built on
    event propagation and worked everywhere except over the picture — the one place it is
    aimed at. Its native surface swallows the event; the result frame installs a filter on
    the whole subtree instead. Caught by testing each surface separately rather than one.

23. **The `Helpers` fold state inverted itself.** Remembered folds were tracked as a set of
    *expanded* projects with "expand everything if the set is empty" as the first-run
    default, so collapsing the only one emptied the set and re-expanded it. Tracked as
    collapses now, which has no such ambiguity.

---

## Architecture notes

### Layering

`config`, `mathmirror`, `refs`, `graph_builder`, `validator`, `comfy_http`,
`prompt`, `progress` import **no Qt**. That makes the graph contract unit-testable and
CLI-drivable, and it is why Tiers 0–2 exist at all. Only `comfy_ws`, `jobs`, `settings`,
`history`, `preview` and `ui/*` touch Qt.

### Threads

Three dedicated `QThread`s hosting `QObject` workers. No asyncio. No worker touches a
widget; everything crosses back as queued signals.

- **`comfy_ws`** — one persistent socket, reconnecting with backoff. Preview decoding
  happens here (tens of ms against ~1 message/second) so the GUI never stalls.
- **`jobs`** — one thread-confined `requests.Session`. Serialising is correct for a
  single-GPU ComfyUI. Order per submit: upload → build → validate → submit.
- **`pose`** — the local estimator. It gets its own thread rather than sharing the job
  thread for one reason: while `submit_job` is blocked, that thread's event loop is not
  spinning, so a queued cancel cannot run until the work it is cancelling has finished.
  A tens-of-seconds local job with a stop button needs a loop that stays free, so cancel
  here is a plain `threading.Event` checked between frames — the same shape as
  `ComfyWsClient.stop()`.

Queueing is therefore two phases: render any missing skeletons on the pose thread, then
hand the snapshot to the job thread. A failure or a cancel in the first phase means the
run never goes out, because a graph naming a clip that was never rendered would fail on
the server minutes later instead.

### Node-ID allocation

Fixed blocks, not `max(id)+1`, so IDs are stable across runs (ComfyUI's execution cache is
keyed on them) and dry-run dumps stay diffable. Asserted non-colliding at import.

```
200–208   LoadImage per reference image
220/230/240      VHS_LoadVideo per reference video -- it loads, windows and
                 de-muxes in one go, so there is nothing beside it
260/270/280 (+1) LoadAudio / TrimAudioDuration per reference audio -- always both,
                 marked or not
900–902   PreviewAny probes (Tier 2 only)
```

### Deliberate design choices

- **Orphan nodes are pruned and reported.** ComfyUI only executes ancestors of output
  nodes, but this app validates every node, and an unwired node with a missing required
  input would wrongly block the queue. The workflow's `DWPreprocessor` is exactly that: it
  is never executed; it is a leftover of the pose feature that has since been removed.
- **The workflow JSON is never modified.** Every build deep-copies it.
- **The dimension grid is a constant, not a state field.** `ResolutionSelector.multiple`
  was editable and is now fixed at 32, and it was removed from `BuildState` and from
  settings rather than merely hidden — a field nothing can change is one that only drifts,
  and a stale 64 left in an old settings file would have quietly moved every resolution.
  `mathmirror.resolution()` still takes it as a defaulted argument so the mirror can be
  tested across the node's whole declared range.
- **A reference is a file plus its mark; nothing is cut on disk.** The mark reaches the
  model as `VHS_LoadVideo.skip_first_frames` (video) and `TrimAudioDuration.start_index`
  (audio), which are exact and free. Cutting real files was tried and removed: it made a
  row name its own clip, but it put a server round trip, a re-encode and a cache between
  pressing a button and seeing a reference, for no change in what the model receives.
- **The generated length is the out point, so there is no out point to set.**
  `MiniMaxH3ReferenceToVideo` truncates every reference to the generated length. A marked
  end beyond that could only ask for material the model discards; one short of it makes
  the reference run out before the clip does, which is a thing to be *warned* about rather
  than a thing to aim for. So the row stores a start and nothing else, `frame_load_cap`
  stays linked to the workflow's own math node in every build, and the window the editor
  draws is derived — it moves when *Duration* moves, and stops early when the file does.
- **Trimming is unconditional, so there is nothing to toggle.** A switch that could be off
  would mean sending something other than what the model keeps, so `trim_enabled` was
  removed from the row and the *Trim* checkbox and the row's *Trim* button with it. Audio
  therefore gets its `TrimAudioDuration` whether it is marked or not: the same shape of
  graph either way, and the truncation is visible in the workflow rather than implied.
  What is left is `trim_start` and `RefRow.marked`, which means "starts somewhere other
  than its own beginning" — a question about what is worth *saying*, not about what is
  sent.
- **One timeline, not two.** There was a transport slider and a trim track under it,
  showing the same clip a line apart. They are one widget now (`ui/timeline.py`), bound to
  a player and, when there is one, a reference row.
- **The result frame does not narrate.** No caption line, no path line, no
  "sending 48-168f" line. The reference list says which file it is and what will be sent,
  the track draws it, and the tooltip has the words. The one line left under the transport
  stays hidden unless playback actually fails — an error with nowhere else to go.
- **Marking happens where the clip is, not on the row.** The row has no room for a
  timeline and no picture to mark against; the result frame has both. The row shows the
  result — name and section — rather than the controls.
- **The frame opens paused.** Clicking through references is browsing; autoplay makes that
  unpleasant, and marking an in point wants a still picture anyway.
- **Pose extraction was removed, and then came back the other way round.** The first
  version was a second ComfyUI graph, and it grew a cache, a staleness rule, a separate
  workflow and a persistence story to support one checkbox. The second version is a local
  ONNX pass: the same cache and staleness rule, but no second graph, no new node classes,
  and nothing in `graph_builder` that knows about it. What made the difference was doing
  the substitution on the job snapshot — the row keeps naming the user's own file, and the
  skeleton only exists between `swap_in` and the upload.
- **`steps` and `ref_image_size` are render settings, so scenes do not capture them.**
  Same reasoning as resolution: a scene is a shot, and the same shot gets run at draft
  quality and then at final. They live in settings.json and in each run's history record.

- **Nodes are found by role tag, not by number** (`harmon3/roles.py`). Every node the app
  touches carries `h3-<role>` at the start of its ComfyUI title, which survives an API
  export as `_meta.title`. The alternative — inferring the role from `class_type` — cannot
  work here: the workflow has two `VAELoader`s, and picking the wrong one sends the video
  VAE where the audio VAE belongs, which fails deep inside a run rather than at startup.
  So tagging is strict, and an untagged node never binds to anything.

  `resolve()` reports **every** contract problem at once rather than raising on the first.
  A workflow that has drifted usually has several, and fixing them one launch at a time is
  miserable.

  Injected loader ids are derived from `max(existing) + 1` rounded up to a hundred, not
  from the constants they used to be. The old fixed blocks (200/220/260) could be
  overwritten silently by a large enough workflow, and the comment claiming an import-time
  assert guarded that was describing code that was never written.

- **The seed and the schedule follow whichever node carries them.** Two sampler
  arrangements are legal and both have shipped: `SamplerCustomAdvanced` takes its noise by
  link, so the seed belongs on the `RandomNoise` feeding it; `MiniMaxH3ProgressiveSampler`
  makes its own noise from a `noise_seed` widget and carries the staging `schedule` too.
  `graph_builder.seed_node` and `schedule_node` resolve which, from `roles.SEED_ROLES` and
  `SCHEDULE_ROLES`, so swapping one sampler for the other is a workflow edit rather than a
  code change.

  `roles.REQUIRED_GROUPS` is what makes that safe: at least one seed-carrier must exist,
  or there is nowhere to put the seed and the run would silently use whatever the workflow
  shipped with. Neither node is required on its own, which no single `required` flag can
  express.

  The Schedule parameter is hidden when no node declares one. Writing it anyway would be
  an input no node declares, and ComfyUI rejects the whole prompt for that — so the field
  would not merely do nothing, it would break every run.

- **`h3-keep` is the escape hatch that makes the contract usable.** The orphan sweep keeps
  only ancestors of the output node, so before this a user-added `PreviewImage` or second
  save branch was deleted on the way to the server without a word. Tagging a node
  `h3-keep` makes it a root the sweep starts from, so a whole branch behind it survives.
  What is pruned is now reported rather than only logged.

- **The geometry arithmetic moved from the graph into `mathmirror`.** The workflow used to
  carry a `ResolutionSelector` and a `ComfyMathExpression` computing the 17k+5 frame count;
  those nodes are gone and the builder writes `width`, `height` and `length` into the
  reference node as literals. This removes a whole class of silent failure — the two sides
  could previously disagree, which is what `--probe` existed to catch — at the cost of
  making this side solely responsible for the quantisation. Sending a length off the 17k+5
  grid is a hard rejection from the node, so `mathmirror.frames_from_seconds` is now load
  bearing in a way it was not when the server rounded too.

  The frame rate is the one number still duplicated: `config.FPS` here, `frame_rate` on
  the output node there, with nothing linking them. `graph_builder.geometry_warnings`
  exists solely to say so, because a mismatch renders every clip at the wrong length
  without failing anything.

- **`VHS_VideoCombine` writes three files for a run with audio, and two of them are
  litter.** From `nodes.py`: a PNG of the first frame "to keep metadata" (line 388), the
  silent video, and then `{name}-audio.{ext}` muxed from the two. Only the last is
  returned to the UI, and only the last is what this app downloads — the other two just
  accumulate in the output folder.

  Neither has a widget. Both are read from `extra_pnginfo['workflow']['extra']`, which
  ComfyUI fills from the prompt's `extra_data` for any node declaring an `EXTRA_PNGINFO`
  hidden input (`execution.py:216`). So `graph_builder.extra_data_for` sends
  `VHS_MetadataImage: False` and `VHS_KeepIntermediate: False`, and
  `comfy_http.submit` puts them in the request body. Verified live: three files became
  one, with the muxed video unchanged.

  `KeepIntermediate` deletes `output_files[1:-1]`, which is why it removes the silent
  video and not the muxed one — and why it correctly does nothing on a run without audio,
  where that slice is empty.

  Sent only when the graph holds a node that reads it. `extra_pnginfo` is also what core
  nodes embed in saved files' metadata, so populating it unconditionally would write this
  app's private options into every PNG a workflow happened to save.

- **`VHS_VideoCombine` declares `pix_fmt`, `crf`, `save_metadata` and `trim_to_audio`
  inside its `format` combo, not as inputs.** The `formats` map in the combo's settings
  dict lists a widget set per format, and `/object_info` never surfaces them at the top
  level. Once the validated class list was derived from the graph rather than hardcoded,
  these read as four unknown inputs on a workflow that was entirely correct.
  `validator._format_dependent_inputs` reads that map. It accepts every format's widgets
  rather than only the selected format's: the aim is to stop a false error, and which
  format legitimately declares what is the node's own business to enforce.

---

### Bug: a tag rewrite cancelled itself on any cycle

Found by adding reorder, and present since the rewrite was written. `update_tags` folds a
fresh renumbering into one the user has not acted on yet, so two quick edits do not lose
the first one's mapping. It did that by walking the new migration entry by entry and
chaining each onto the pending map — which treats *the entries of a single migration* as
if they were successive edits.

Swap two references and the migration is `{P1: P2, P2: P1}`. Chaining the second entry
onto the first gives `P1 -> P1`, which the `k != v` filter then drops, and the banner never
appears. A three-way cycle collapsed the same way. Removals happened not to trigger it,
because their migrations are monotone shifts whose keys and values do not interlock in the
order they are iterated — so it sat unnoticed until reordering made cycles routine, since
every swap is one.

`prompt_editor._compose` now composes the two maps from a snapshot instead: carry each
pending entry forward through the new migration, then add the new entries the pending map
does not already account for. `remap_prompt` was always correct — it substitutes in a
single pass for exactly this reason — so the fault was only ever in deciding what to hand
it.

---

## Known limits

- **The trim timeline needs a readable frame rate for a video.** It works in milliseconds
  and the mark is stored in frames, so with no rate there is nothing to convert between;
  the track goes quiet and says so. It will measure a rate from the playing file's own
  length and frame count when the probe never supplied one. Sending is *not* blocked by a
  missing rate any more — `VHS_LoadVideo` windows the frames and the soundtrack from the
  same two inputs using the source's own frame time, so nothing has to be converted on
  this side to keep them aligned.
- **A reference restored from history can be marked but not watched.** It names a file on
  the ComfyUI server and nothing on this machine, so clicking it binds the timeline to the
  number without a clip behind it, and marking at the playhead is unavailable.
- **The trim editor's I key needs the timeline to have focus**, being scoped to the
  widget. A window-wide single-letter shortcut would eat that letter in the prompt boxes.
  The *Mark in* button always works.
- **Tier 2 does not reach the MiniMax/SaveVideo branch.** Partial validation only walks
  the probes' ancestors; Tier 1 covers the rest.
- **Reference video is the expensive input** — its latents ride through every sampling
  step on a canvas up to 768×1344. Long reference videos are the usual cause of an OOM.
- **Reference video is assumed to be 24 fps** and is not resampled; the app warns when a
  source differs.
- **`VHS_LoadVideo`'s `format` rounds dimensions, not the frame rate.** An earlier note
  here claimed `format: "AnimateDiff"` resamples to 8 fps because that is its
  `target_rate`. Read from the source: `target_rate` is never consulted on the load path —
  resampling is driven only by `force_rate`. What `format` *does* supply is `dim`, whose
  first element becomes `downscale_ratio`, and `target_size()` rounds width and height to a
  multiple of it **even when `custom_width`/`custom_height` are 0**. So `AnimateDiff`
  silently resizes a 480×854 clip to 480×856, and `format: "None"` (no `dim`, ratio 1) is
  the value that leaves frames untouched.
- **Pose follows one subject.** The largest detection wins, and continuity by IoU keeps it
  from hopping to a passer-by; a reference with two people in it loses one of them.
- **A posed reference is re-rendered whenever the mark or the duration moves**, because
  those change which frames the skeleton covers. Changing the *seed* does not, so
  iterating on a shot is free after the first render.
- **Pose weights are ~1.5 GB and fetched on first use.** With no network and nothing
  cached, ticking Pose fails at that point rather than at startup.
- **A tall reference in a wide generation may still come back stretched.** The pipeline
  is not the cause — the sizing above preserves aspect to within a few per cent, and the
  reference keeps its own `latent_h`/`latent_w` in the payload. What is left is the model
  composing a portrait subject to fill a landscape frame. Matching the aspect ratio, or
  saying the framing in `detailed_description`, are the remedies that cost nothing. A
  letterboxing option was built and reverted once bug 20 turned out to explain the case
  it was built for; it is recoverable from the session notes if it is wanted again.
- **A video's size ceiling only bites below the node's canvas**, ~1344×768. Above that
  every clip is flattened to the same size regardless, which is why that slider measures
  the canvas rather than the file — and why it does nothing at all for a clip already
  smaller than it, which the readout says.
- **A rescaled or posed clip is re-encoded, losing any rotation metadata.** Phone footage
  carries a display matrix that PyAV's `to_ndarray` does not apply; the image path is
  handled (bug 20) and the video path is not checked. Worth doing before trusting a
  portrait clip through a local pass.
- **The reference bundle is a snapshot, not a record.** It is overwritten by the next
  export and read back by nothing.
- **ComfyUI has no authentication** and is started here with `--listen`, so anything on
  the LAN can queue jobs. Not this app's problem to fix, but worth knowing.

---

## Possible next steps

Nothing here is blocking; listed in rough order of value.

- **Judge the whole-body pose variant against the body-only one.** Now selectable and
  working (bug 18), but still uncompared on real footage. Note the hand keypoints scored
  0.12–0.39 on a 250p source against a 0.4 threshold, so they were culled — the comparison
  wants a higher-resolution reference to be worth anything.
- **`pose_kpt_thr` still has no UI.** The model and the style do; the threshold is the one
  left, and it is the one that decides whether wholebody's hands survive at all.
- **The pose cache has a Clear button but no eviction.** Same as `runs/videos`: it grows
  until someone presses it.
- **Check the video rotation matrix**, per the known limit above — a portrait phone clip
  through a pose or size pass is the untested case.
- **One subject only.** The estimator takes the largest box and follows it by IoU; a
  reference with two dancers loses one. Multi-person would mean deciding what the model
  should even receive.
- Expose a workflow picker if a third variant appears (V1 is still in `API/` and builds
  fine).
- Batch queueing — run several scenes back to back.
- Surface `delta` from the preview node as a convergence readout.
