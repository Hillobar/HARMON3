# HARMON3

A PySide6 desktop front-end for the MiniMax H3 reference-to-video workflow in
`API/video_minimax_h3_r2v_api.json`. It drives a running ComfyUI over its HTTP and
WebSocket API — no models are downloaded, ComfyUI supplies everything.

The app exposes only the controls that matter for this model: reference images, videos
and audio; the prompt; and the generation parameters — resolution, duration, steps,
sampler, scheduler, the progressive-sampling schedule, reference sizing and seed.
Everything else (frame rate, codec, output prefix) stays exactly as the workflow defines
it.

The app finds the nodes it drives by a **role tag** at the start of each node's ComfyUI
title — `h3-promptinput`, `h3-loadmodel`, `h3-vidcombine`. Node numbering belongs to
ComfyUI, so the workflow can be renumbered, rewired and extended without touching any
code. See [Modifying the workflow](#modifying-the-workflow).

Sampling goes through `MiniMaxH3ProgressiveSampler` from the `ComfyUI-HillobarNodes` pack,
which must be installed on the server. It runs the early steps on a smaller latent grid and
upscales the estimate between stages, so most of a run is cheap and only the last stage
pays full resolution — that is what the **Schedule** parameter drives. The node carries its
own `noise_seed`, so the seed is written straight to it and the graph needs no separate
`RandomNoise`. Swap it for a plain `SamplerCustomAdvanced` and the app follows: the seed
moves to whatever `h3-noise` node feeds it, and the Schedule field hides itself.

Width, height and length are computed here rather than in the graph and written into
`MiniMaxH3ReferenceToVideo` as literals: dimensions snapped to a multiple of 32, and the
frame count rounded **up** to the next 17k+5 the model accepts. The workflow carries no
node that does this arithmetic, so the app is the only thing keeping those values legal.

Engineering notes — what was learned about the ComfyUI node contracts, the bugs found, and
the design decisions behind all this — are in [NOTES.md](NOTES.md).

## Setup

```
setup.bat        creates .venv with Python 3.12 and installs the dependencies
run.bat          launches the app
run_debug.bat    launches with a console and verbose logging
```

Requires Python 3.12 (`py -3.12`) and a reachable ComfyUI. The default address is
`http://127.0.0.1:8188`; change it with the **Server...** button.

## Using it

**References.** Up to 9 images, 3 videos and 3 audio files, added with *+ Add* or by
dragging them onto the reference panel. That panel is the only place files are added. Each
*+ Add* button reopens in the folder that kind was last taken from, tracked
separately — stills, clips and sound rarely live in the same place. Each row shows the tag
the model uses for it — `<Picture 1>`, `<Video 1>`,
`<Audio 2>` — and clicking a tag inserts it into the prompt at the cursor, spacing it so
it cannot run into the surrounding words. Reference videos have a *use its soundtrack*
checkbox that feeds their audio to the model alongside their frames.

**Reference size.** Every image and video row carries a *Size* slider, from 100% down to
10%, deciding how much of that reference is sent. Reference tokens ride through every
sampling step, so a smaller reference is a faster run at some cost in fidelity — and this
is the only way to size one reference differently from the others, since the model's own
`ref_image_size` applies to all of them at once. It works because the model **never
enlarges a reference**: shrinking one here is a ceiling it will not raise.

The two kinds measure it differently, and the readout beside the slider always says what
will actually be sent:

- An **image** is a share of its own file. `512x512 - 0.26 MP` at 100%,
  `288x512 - 0.15 MP (from 600x1000)` at 50%.
- A **clip** is a share of the canvas the model fits every clip to — about 1344×768. A
  share of the file would be meaningless: a 4K source and a 1080p one are already
  flattened onto the same canvas before anything is encoded, so 50% of the file changes
  nothing. `672x384 per frame - 25% of the tokens (normally 1344x768)`. A clip already
  smaller than the canvas says so and its slider does nothing.

Your files are never touched. A resized copy is prepared before the run — instantly for
a still, as a pass with a progress bar for a clip — and swapped in on the way to the
server, so the row, your scenes and your settings always name the original.

**The result frame is where sections are marked.** Click a reference and it opens there
with the timeline under it. It opens **paused, on its first frame**: clicking through a
list is browsing, and a clip that starts talking the moment it is selected makes that
unpleasant. Play it with the transport when you want to.

**The wheel steps frames.** Hover the picture or the timeline and roll: one frame a notch,
ten with Shift held. It works on the finished result, a reference clip, both clips side by
side and the live preview, and it pauses playback first — stepping through frames and
running at speed are two different things to be doing. The step is a real frame of
whatever is on screen: a bound reference steps in its own rate, anything this app
generated steps at 24 fps. The arrow keys move by the same amount when the track has
focus.

The in point you set there is where the section the model receives begins. There is no out
point: the model truncates every reference to the generated length, so the section runs
from the mark for exactly as long as the clip being made — **the *Duration* parameter is
the out point**, and changing it moves the far edge of every section at once. Nothing is
cut on disk: a reference stays its **file plus its mark**, and the row shows exactly that
— the name and where it starts.

The model assigns those ordinals from the order and kind of the references, in this
order: every image, then each video with its soundtrack's `<Audio j>` emitted immediately
*before* its `<Video k>`, then every standalone audio. So two images, one video with
soundtrack, and one standalone audio produce:

```
<Picture 1>  <Picture 2>  <Audio 1>  <Video 1>  <Audio 2>
                          ^ the video's soundtrack   ^ the standalone audio
```

**Drag a row by the grip on its left to reorder it.** The order *is* the numbering — the
model assigns `<Picture 1>` and the rest purely from the order it receives them in — so
reordering is the only way to choose which reference gets which tag. Rows reorder within
their own kind: an image can only ever be a `<Picture i>`, so there is no slot for it among
the videos. A line shows where the row will land.

This means reordering, removing a reference, or unticking a soundtrack all **renumber the
tags**. The app watches for that and offers a one-click *Rewrite tags in prompt*, which
substitutes every tag simultaneously so overlapping moves cannot cascade — including a
straight swap or a cycle, where each tag's new value is another tag's old one.

**Parameters.** Everything about a run that is not the prompt or a reference, in one panel
with nothing folded away:

| | |
|---|---|
| *Aspect ratio*, *Megapixels* | the pixel dimensions written into the reference node, shown live and snapped to a fixed 32px grid |
| *Length* | the model only accepts frame counts of the form 17k+5 at 24 fps, so the requested length is rounded **up**. The readout shows the resulting frame count and the true clip length. The ceiling is 149.6 s (3592 frames) |
| *Steps* | sampling steps (`BasicScheduler`). Run time is very nearly proportional to it |
| *Sampler* | which solver walks the sigmas (`KSamplerSelect`). Whatever the workflow ships with is what the model was tuned around. The list starts as stock ComfyUI's and becomes whatever the connected server actually offers, so sampler packs show up on their own |
| *Scheduler* | how the sigmas are spaced across those steps (`BasicScheduler`). Same story: stock ComfyUI's set until the server has been read |
| *Schedule* | the staged sampler's progressive-resolution plan, as `scale:end_percent` per stage. Early steps run on a smaller latent grid and the estimate is upscaled between stages, so `0.5:0.55, 1.0:1.0` spends the first 55% of the steps at half grid. The last stage must be `1.0:1.0`; `1.0:1.0` alone is the full-resolution baseline. Free text, checked against the node's own rules as you type — a schedule it would refuse is called out under the field. **Shown only when the workflow's sampler declares one**, since a `SamplerCustomAdvanced` does not |
| *Stage upscale* | how the estimate is resampled when one staging stage hands over to the next at a larger grid. Does nothing at `1.0:1.0`, where nothing is ever upscaled. The node's own fixed set, so unlike Sampler it is never repopulated from the server. Shown alongside *Schedule*, and hidden with it |
| *Sigma shift* | `MiniMaxH3SigmaShift.shift_video`: where the weight of the sigma schedule sits. Higher spends more of the run at high noise, which moves the picture further from the references; lower holds closer to them. Needs an `h3-shift` node, and hides itself without one. `shift_audio` is deliberately not exposed — it stays as the workflow sets it |
| *Reference size* | `ref_image_size`, applied to every reference at once. **match** scales each one down to the generation's own pixel area; **max** uses the reference pipeline's 2048px short edge for the best identity fidelity. Reference tokens ride through every sampling step, so *max* can be several times slower. **Defaults to max**, because each reference now carries its own ceiling on the row and *match* would re-cap them all. Note it does not apply to reference videos at all — those are fitted to a fixed canvas regardless |
| *Seed* | 56-bit, with a dice button and a *New seed for every run* toggle. The field stays readable and copyable either way |

*Check the workflow* confirms the role contract still binds, that every node the workflow
uses exists on the connected server, and that nothing needed would be pruned. It loads no
models and takes about a second.

The dimension grid is not offered: MiniMax H3's canvas is built in multiples of 32, so any
other value only produces dimensions the model rounds anyway.

**Projects.** Scenes are grouped into the finished video they are pieces of. A project is
a top-level row in the *Projects* tab carrying how many scenes it holds and how long the
whole thing runs; its scenes sit under it, numbered in running order. *New project...*
starts an empty one, and dragging a scene onto it files it there — drop it between two
scenes to set its place in the sequence instead. *Move up*, *Move down* and *Remove from
project* are on the right-click menu.

Deleting a project never deletes a scene: its scenes return to *Ungrouped*. A project is a
way of arranging work, not a container that owns it. Membership travels **on the scenes
themselves**, so copying a scene file to someone else brings its place in the sequence
with it; a `projects.json` beside them records only the names, so a project with nothing
in it yet still has a row to drag onto.

**Scenes.** A scene is a named, reusable shot: its **references, prompt, length and seed
settings**, plus a short description of what it is for. Save the current editor with
*Save as scene...*, and the catalogue keeps it for later — double-click to load it back,
*Load and run* to queue it straight away, *Update* to fold in your edits, *Revert* to
throw them away. *Details...* edits the name and description; duplicate and delete are on
the row and the right-click menu.

The name and description belong to the scene rather than the editor, so *Details...*
writes them immediately and they never show up as unsaved changes.

Resolution is deliberately **not** part of a scene. It is a render setting, so the same
scene can be run small while you iterate and full size when you are happy, and changing it
never marks a scene as edited.

While a scene is loaded, the run bar shows `Scene: <name>`, with a `*` once you have
changed anything it owns. Loading a different scene with unsaved changes asks first.

Each scene is its own file in the scenes folder, so one can be copied to someone else or
kept in version control on its own, and a damaged file cannot take the catalogue with it.
If a scene's reference files have been moved or deleted since it was saved, the catalogue
marks it and the affected rows block the queue rather than failing at submit.

**Settings.** The third tab holds the storage locations, the ComfyUI address, and
**Sage Attention** — the switch the workflow carries. On, the model goes through
`PathchSageAttentionKJ`; off, the patch node is left out of the submitted graph entirely
rather than included and bypassed, because `Switch` wires both branches and ComfyUI would
otherwise still run it. It needs the sageattention library where ComfyUI runs, and takes
effect on the next run.

**Pose settings** live there too: which weights the *Pose* toggle uses (ViTPose-L,
ViTPose-B, or the wholebody variant that adds face, hands and feet), how the keypoints are
joined up, and a *Clear pose clips* button with a running count of what is cached. The
style choice applies to whichever model is selected — *OpenPose (standard)* hangs both
hips off the neck, which is the convention pose-conditioned models are trained on, while
*torso from the shoulders* joins each hip to its own shoulder and the two hips to each
other, giving the trunk width. Anatomical, but no longer the trained convention, so it is
worth an A/B rather than an assumption.

**Diagnostics → Export references** writes out exactly what the model is about to be
given, into `reference_bundle/` beside the app: the reference files as they will be
uploaded — after any section cut, skeleton, resize or rotation — named for the tag that
addresses them, the prompt as the node receives it, and a `manifest.json` saying what the
node then resizes each one to and what that costs in tokens. Nothing is uploaded, and
nothing reads the folder afterwards. It is for the question that comes up whenever a
result looks wrong: *what did it actually see?*

The rest of that tab: The
scenes folder defaults to `scenes/` beside the app and can be pointed anywhere — a synced
drive, a project folder, wherever the rest of that job lives. Changing it asks whether to
bring the existing scene files along or leave them behind, and refuses a folder it cannot
write to. Changes are staged and committed with **Apply**, since each one either
reconnects or moves files.

**Prompt.** The prompt is written in six collapsible boxes — `subject_definitions`,
`summary`, `retention_analysis`, `detailed_description`, `overall_soundscape`,
`non_diegetic_music`. Each folds to a single header line carrying a character count, so
all six closed cost about 150 px; *Expand all* / *Collapse all* are at the top.

Under each open section is a row of **format chips**. The output format has a vocabulary
nobody memorises — four label kinds, a fixed set of bracketed task types and relationship
markers, shot lines whose timestamps have a shape — so each chip's caption *is* the thing
it inserts. The row under `retention_analysis` is both the way to insert a marker and the
list of which seven markers exist. Placeholders arrive selected, so typing replaces them;
`skeleton` drops a whole section's starting shape in one undoable step; the task-type
chips merge into the summary's `[...]` prefix rather than pasting beside it.

Every `<Subject N>` your `subject_definitions` defines also appears as a chip in the other
five sections, live as you type. Define it once and click it after that — the model
resolves a label by matching the string, so a mistyped one silently stops referring to
anything. *Helpers* in the header hides the rows once the format is in your fingers, and
*Guide* opens the format spec beside the editor.

They are joined into the one string the model receives, in that order:

```
subject_definitions:
whatever you typed

summary:
whatever you typed
```

An empty box still appears, carrying `N/A`, so the model always sees the same structure.
The combined prompt is printed to the console on every queue — run from `run_debug.bat`
to see it.

Tag insertion goes into whichever box last had the caret; the lint and the tag rewrite
span all six.

**Trimming.** Every video and audio reference is cut to the generated length — there is
nothing to switch on, because `MiniMaxH3ReferenceToVideo` truncates one to that anyway.
The only thing left to decide is *where the cut starts*, and that is marked in the result
frame against the clip itself.

So there is no out point either: one beyond the generated length could only ask for
material that would be discarded, and one short of it would make the reference run out
before the clip does. The *Duration* parameter is the out point, and the drawn section
follows it live.

For video the mark drives the loader rather than a slice after it — `skip_first_frames`,
with `frame_load_cap` linked to the same math node the workflow links it to — so only that
section is ever **decoded**. This matters more than it sounds; see below. Audio gets a
`TrimAudioDuration` from the mark for the same length, marked or not, which makes the
truncation the model would do anyway visible in the workflow.

There is **one timeline**, and it is both the scrub bar and the mark editor. It works the
way a video editor's does:

| | |
|---|---|
| drag the handle | move the in point; the picture follows it |
| drag anywhere else | scrub |
| `I` | mark in at the playhead |
| ← / → | nudge a frame, shift for ten |
| *Play section* | play what will be sent, on a loop |
| *Reset* | start at the beginning of the reference again |

Everything outside the section is drawn cut away, and playback loops inside it so it can be
judged rather than guessed at. The spin box beside the buttons is the same in point as an
exact number: frames for a video, seconds for audio — the unit each slicing node actually
takes. Those controls appear only while a reference is open; a finished video gets the same
track with just a playhead. The far edge is drawn as a plain line rather than a handle,
since nothing here moves it, and it stops early when the reference runs out before the clip
does — which the row also says in words, and the track in its tooltip.

A trimmed video's soundtrack comes off the same loader, already cut to the same span:
`VHS_LoadVideo` derives the audio start and duration from those same two inputs using the
source's own frame time, so picture and sound cannot drift apart and nothing has to be
converted on this side.

**Pose.** Tick *Pose* on a video reference and the model receives a skeleton of that clip
on black instead of the clip: the movement without the person. Useful when a reference is
there for how someone moves rather than for who they are.

The estimation happens **here**, not on the server — a local ONNX pass over the frames
that will actually be sent, run before the job is queued. The result is an ordinary mp4
that goes through the same upload path as any other reference, so nothing in the graph
knows the feature exists. The clip's own audio for the same span is carried across, so a
posed reference keeps its soundtrack and its `<Audio>` tag.

| | |
|---|---|
| model | ViTPose-L by default, 17 keypoints, drawn OpenPose-style. ViTPose-B (faster) and ViTPose-L wholebody (133 keypoints — adds face, hands and feet) are selectable in Settings |
| why that one | it holds the state of the art on OCHuman, the occluded-people benchmark, which is the case that matters for a partly hidden or cropped dancer |
| where it runs | CUDA if `onnxruntime-gpu` can see it, otherwise CPU — the app says which, and says why when it is the second |
| weights | 1.2 GB for the estimator plus 351 MB for its person detector, downloaded once on first use |
| speed | measured on an RTX 5090: 124 frames of 640×480 in 7.6 s, about 60 ms a frame |

Only the marked section is posed, which is the difference between 124 frames and 6,777.
The rendered clip therefore *starts* at the mark, and the row submits `skip_first_frames:
0` to avoid taking a section of a section. Renders are cached against the source's
contents, the mark and the generated length, so moving any of the three re-renders and
touching none of them costs nothing.

The thumbnail beside the source shows the skeleton once there is one; click it to watch
the clip, or click it before there is one to render it there and then. Queueing does the
same thing for anything still missing, as a stage before upload — with *Cancel* wired to
it, which is the one thing in this app that stops on the spot rather than asking the
server nicely.

The model and the skeleton style are in Settings; where it runs and the keypoint
threshold are `pose_runtime` and `pose_kpt_thr` in `settings.json`. `--pose` renders a clip
from the command line, which is a much faster way to judge the estimator than a GUI round
trip.

**Reference video is never decoded in full, and never resized on the way in.**
`frame_load_cap` is always the generated length, marked or not — the same link the
workflow uses, so the server computes it and the two cannot disagree — because MiniMax
truncates a reference to that anyway. Without the cap a
226-second 640×480 reference is 6,777 frames, about **25 GB** of CPU RAM built before the
diffusion model even begins loading; with it, half a gigabyte. `custom_width` and
`custom_height` are both 0, because MiniMax scales every reference to its own canvas
before encoding, so resizing on the way in only changes how much has to be carried to get
there.

The assembled prompt is now the only place the app deliberately builds something other
than what the workflow says — `--dry-run --diff` names it, and anything else showing up
there is a bug rather than a decision.

The transport under the frame always drives whatever the frame is showing — the finished
video, or the reference that has replaced it — so *Play*, the scrub bar and the volume
never act on something hidden behind what you are looking at. *Stop* appears whenever there
is sound or picture to stop and takes it back to the start — the in point, if one is
marked; *Play* pauses in place. Neither appears for a still.

The picture stays up whenever playback is not running: stopped, paused, finished, or being
dragged along the scrub bar, the frame at that position is what you see. Dragging the bar
seeks as you go rather than only on release.

**Live preview.** The workflow includes KJNodes' *Model Preview Override*, so the clip
being sampled animates in the result frame, step by step, decoded through `taeh3_KJ`. On
by default; turn it off with *Show sampler previews* in Settings.

That node does not use ComfyUI's binary preview stream — it sends its own
`kj_preview_override` message addressed to whichever client submitted the prompt, carrying
a fragmented H.264 MP4 per step (or an animated WebP where the server has no NVENC, or a
plain JPEG when `preview_frames` is 1). All three are decoded here. Note that its
`suppress_default_preview` option switches ComfyUI's own preview stream off, so a workflow
with this node has *only* this preview.

**Queueing more than one run.** *Queue* stays available while a run is going, so you can
change the prompt or the parameters and queue the next one straight away — each press
snapshots the editor as it stands, rolls a fresh seed if *New seed for every run* is on,
and hands it to ComfyUI, which works through them in order. *Re-queue* from History joins
the same queue. The only thing that holds a press up is the app's own submission — one at
a time, since references upload serially — and a pose pass, which owns the run bar while
it renders.

*Cancel* takes the **oldest** outstanding run, so pressing it repeatedly clears a batch in
the order it was submitted. It reads `Cancel (3)` when there is more than one. A run the
server has already begun is interrupted; one still waiting is deleted from the queue
without touching whatever is executing — which matters on a shared server, where
`/interrupt` would otherwise take down somebody else's job.

The run bar describes the run being executed and shows `2 queued` between them; the depth
of the queue is otherwise read in History, where a run waiting its turn shows **WAIT**
until the server starts it. A result that lands while others are still queued loads
quietly into the *Results* tab rather than taking the frame away from what you are looking
at; the last one of a batch raises it.

**Progress and ETA.** The progress bar reads `sampling  13/20  48s left`. The pace is
measured from ComfyUI's `progress` messages, ignoring the first interval because it starts
before the model is loaded. When the preview node is present its own averaged step time is
used instead, since that is measured at the sampler rather than inferred from message
arrival.

**Results.** Finished videos are downloaded to `runs/videos/`, played in the app with
audio, and recorded in `runs/runs.jsonl`. Each history entry stores the exact graph that
was submitted, so *Re-queue* resubmits it verbatim and *Load settings* puts its prompt and
parameters back in the editor.

## How it talks to ComfyUI

`MiniMaxH3ReferenceToVideo` accepts far more references than a workflow usually wires. At
submit time the app clones the base JSON, strips whatever reference wiring it shipped
with, and injects the loaders you configured into id blocks chosen to clear every number
already in use, wiring them to the node's flattened autogrow keys:

```
ref_images.ref_image_0 .. _8             IMAGE
ref_videos.ref_video_0 .. _2             IMAGE  (frames, not a VIDEO object)
ref_video_audios.ref_video_audio_0 .. _2 AUDIO  (paired to ref_video_N by index)
ref_audios.ref_audio_0 .. _2             AUDIO
```

**One file per run.** `VHS_VideoCombine` otherwise writes three: a PNG of the first frame
"to keep metadata", the silent video, and then the muxed one. The app submits
`VHS_MetadataImage: false` and `VHS_KeepIntermediate: false` in the prompt's `extra_data`,
which is the only channel the node offers for them — it has no widget for either — so the
PNG is never written and the silent video is deleted once the mux succeeds. The prompt
still travels in the video's own metadata, which is what the node's `save_metadata` widget
controls. Sent only when the graph actually contains a node that reads it.

Reference files are uploaded to `POST /upload/image` (the only upload route ComfyUI has —
it takes audio and video too) into an `input/harmon3/` subfolder, under a name carrying
the file's SHA-256 so re-submitting the same reference never re-uploads it.
The workflow JSON itself is never modified.

Nodes that nothing consumes are left out of the submitted graph and named in the log.
ComfyUI only executes the ancestors of output nodes, so such a node would never run
anyway — but it would still be validated here, and an unwired node with a missing
required input would wrongly block the queue. On the shipped workflow the sweep fires only
when *Sage Attention* is off, which leaves the patch node and its switch feeding nothing.
The roots it sweeps from are the output node and anything tagged `h3-keep`.

## Modifying the workflow

The app finds every node it drives by a role tag: the first word of the node's title in
ComfyUI, starting with `h3-`. The rest of the title is yours — `h3-promptinput Input Text`
tags the node and still reads like a name. Re-export with **Export (API)** as usual.

Run `python -m harmon3 --roles` (or open *Settings → Diagnostics*) to see what bound to
what. Startup refuses a workflow that breaks the contract and lists **every** problem at
once, rather than one per launch.

**Required** — one node each: `h3-promptinput`, `h3-reference`, `h3-loadmodel`,
`h3-loadclip`, `h3-loadvideovae`, `h3-loadaudiovae`, `h3-sampler`, `h3-scheduler`,
`h3-vidcombine`.

**Required as a pair — at least one of** `h3-progressivesampler` or `h3-noise`, because the
seed has to go somewhere. `MiniMaxH3ProgressiveSampler` carries its own `noise_seed` and a
`schedule`; `SamplerCustomAdvanced` takes its noise by link, so that arrangement needs a
`RandomNoise` tagged `h3-noise`. Either works, and the app adapts: the seed goes wherever
it can, and the Schedule parameter appears only when a node declares one.

**Optional** — absent simply means the feature or caption is: `h3-preview`, `h3-sage`,
`h3-switch`, `h3-sampleradvanced`, `h3-guider`, `h3-imagedecode`, `h3-audiodecode`, and
`h3-refimage` / `h3-refvideo` for loaders whose filenames seed the reference list on a
first launch. `h3-shift` is optional too, and drives the *Sigma shift* parameter — without
it that field simply does not appear.

`h3-keep` is not a role. It marks a node as a root the orphan sweep starts from, which is
how a branch of your own — a second output, a preview — survives a build.

**Free to change:** node numbering and layout; adding untagged nodes anywhere in the model
chain (a LoRA loader, a model patch); adding side branches, as long as something in each
is tagged `h3-keep`; and any widget the app does not write — model filenames, the output
node's prefix, format and codec.

**Changes only until the first launch:** aspect ratio, megapixels, duration, seed, steps,
sampler, scheduler, schedule, stage upscale, sigma shift, reference size, prompt and the
Sage switch. These are read from the
workflow to seed `settings.json`, and after that the settings file owns them.

**Not free:** removing a required role, putting one tag on two nodes, or changing a role
node's class to one outside its accepted set — all three are refused at startup with the
reason. And changing the output node's `frame_rate` without changing `config.FPS`: nothing
links the two, so every clip would render at the wrong length without failing. The *Check
the workflow* button and `--roles` both warn about that one.

## Verification

Everything below runs from the command line. The first three cost no GPU time at all.

```
.venv\Scripts\python -m harmon3 --dry-run --diff    the graph, diffed against the workflow
.venv\Scripts\python -m harmon3 --check             validate against the server's schemas
.venv\Scripts\python -m harmon3 --roles             which node plays which role
.venv\Scripts\python -m harmon3 --upload-test FILE  upload and retrieve a file
.venv\Scripts\python -m harmon3 --pose CLIP --out P.mp4 --start 1356 --frames 124
                                                    render one skeleton, no server needed
.venv\Scripts\python -m pytest tests -q             unit tests
```

`--dry-run --diff` on a freshly launched app reports only the two intended differences:
the prompt, assembled from the six sections rather than sent bare, and the reference
node's width/height/length, computed here from aspect ratio, megapixels and duration.
Those match the workflow's own literals only when the literals happen to sit on the same
32px and 17k+5 grids. Every other node must match exactly.

## Appearance

A dark technical theme built around cyan (focus, primary action, computed values) and
magenta (reference tags), defined in one place: `harmon3/ui/style.py`. Change the palette
constants at the top of that file and the whole app follows.

Typography is set through `QFont.setFamilies` rather than the stylesheet, because Qt's
stylesheet parser does not handle a comma-separated font fallback list reliably. Computed
values — dimensions, frame counts, seeds, elapsed time, tags — are monospaced so they read
as data rather than prose.

## Notes

- The workflow names two reference images (`red_superboy_on_city_roof.png`,
  `mecha_dragon_lightning.png`). If they are not in ComfyUI's `input` folder, the app
  flags those rows and blocks the queue rather than letting the submit fail.
- Reference video is the most expensive input: its latents are carried through every
  sampling step. Long reference videos are the usual cause of an out-of-memory failure.
- Reference video is assumed to be 24 fps and is not resampled; the app warns when a
  source's frame rate differs.
- Generated state (`settings.json`, `ui_state.ini`, `runs/`, `scenes/`) lives beside the
  app. Set `HARMON3_HOME` to relocate it.
- ComfyUI has no authentication. If it was started with `--listen` it is reachable from
  the local network; HARMON3 itself only talks to the address you configure.
