"""Build a submittable ComfyUI prompt graph from the app's state.

MiniMaxH3ReferenceToVideo accepts up to 9 reference images, 3 reference videos (each with
an optional soundtrack) and 3 standalone reference audios, addressed through flattened
autogrow keys:

    ref_images.ref_image_0 .. _8            IMAGE
    ref_videos.ref_video_0 .. _2            IMAGE  (a frame batch, not a VIDEO object)
    ref_video_audios.ref_video_audio_0 .. _2 AUDIO (paired to ref_video_N by trailing index)
    ref_audios.ref_audio_0 .. _2            AUDIO

So this module clones the base graph, strips whatever reference wiring it shipped with,
and injects the loader nodes the user actually configured.

Nodes are addressed by role rather than by number -- ``roles.promptinput``, ``roles.reference``
-- so the workflow can be renumbered and extended in ComfyUI without touching this file.
See harmon3/roles.py.

Width, height and length are written as literals, computed by ``mathmirror``: the workflow
carries no ResolutionSelector and no frame-count expression, so this side owns the
quantisation entirely.

No Qt, no network.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field

from . import config, mathmirror, prompt as prompt_mod, roles as roles_mod
from .refs import IMAGE, VIDEO, RefRow, RefSet, TagAssignment, compute_tags

REF_KEY_PREFIXES = ("ref_images.", "ref_videos.", "ref_video_audios.", "ref_audios.")

#: Problems the submission pipeline resolves by itself on the way to the server -- the
#: files get uploaded -- so they must not stop the user pressing Queue.
DEFERRED_PROBLEMS = ("has not been uploaded",)


def is_deferred(problem: str) -> bool:
    return any(marker in problem for marker in DEFERRED_PROBLEMS)


def clamp_steps(steps) -> int:
    """A step count the scheduler will accept."""
    try:
        value = int(steps)
    except (TypeError, ValueError):
        return config.DEFAULT_STEPS
    return min(max(value, config.MIN_STEPS), config.MAX_STEPS)


def clean_ref_image_size(value) -> str:
    """One of the node's two options; anything else falls back to its default."""
    return value if value in config.REF_IMAGE_SIZES else config.DEFAULT_REF_IMAGE_SIZE


def clean_sampler(value) -> str:
    """A sampler name, as chosen. Only an empty one falls back to the workflow's.

    Deliberately not checked against ``config.SAMPLERS``: that list is stock ComfyUI's, and
    a server with sampler packs installed offers more. Substituting a different solver for
    the one that was asked for would change the picture silently, where an unknown name is
    caught by the validator and named in the server's own reply.
    """
    return _name_or(value, config.DEFAULT_SAMPLER)


def clean_scheduler(value) -> str:
    """A scheduler name, as chosen. Only an empty one falls back to the workflow's."""
    return _name_or(value, config.DEFAULT_SCHEDULER)


def clean_upscale_method(value) -> str:
    """One of the node's resampling methods; anything else falls back to its default.

    Checked against the list, unlike the sampler and scheduler names: this combo is the
    node's own fixed set rather than something a plugin pack extends, so an unknown value
    here is a stale settings file rather than a server this build has not heard of.
    """
    return value if value in config.UPSCALE_METHODS else config.DEFAULT_UPSCALE_METHOD


def clamp_shift(value) -> float:
    """A sigma shift the node will accept."""
    try:
        shift = float(value)
    except (TypeError, ValueError):
        return config.DEFAULT_SHIFT_VIDEO
    if shift != shift:  # NaN
        return config.DEFAULT_SHIFT_VIDEO
    return round(min(max(shift, config.MIN_SHIFT), config.MAX_SHIFT), 2)


def _name_or(value, fallback: str) -> str:
    """A trimmed non-empty string, or ``fallback``. Anything not a string is not a name."""
    if not isinstance(value, str):
        return fallback
    return value.strip() or fallback


def clean_schedule(value) -> str:
    """The staging schedule, as typed.

    Only emptiness is corrected here. The node owns the grammar and rejects a bad schedule
    with a message far better than anything guessed at this end, and quietly substituting a
    different schedule for the one that was asked for would be worse than the error --
    ``schedule_error`` is how the panel says so before Queue is pressed.
    """
    text = str(value or "").strip()
    return text or config.DEFAULT_SCHEDULE


def schedule_error(value) -> str:
    """Why the node would refuse this schedule, or "" if it would accept it.

    A local mirror of ``parse_schedule`` in MiniMaxH3ProgressiveSampler, so a run that is
    going to fail says so in the panel rather than several minutes later on the server.
    """
    text = str(value or "").strip()
    if not text:
        return "Schedule is empty."

    stages = []
    for chunk in text.replace(",", " ").split():
        if ":" not in chunk:
            return f"'{chunk}' is not scale:end_percent."
        scale_text, percent_text = chunk.split(":", 1)
        try:
            stages.append((float(scale_text), float(percent_text)))
        except ValueError:
            return f"'{chunk}' is not numeric."

    last_percent = last_scale = 0.0
    for scale, percent in stages:
        if not 0.0 < scale <= 1.0:
            return f"Scale {scale:g} must be over 0 and at most 1."
        if not 0.0 < percent <= 1.0:
            return f"End percent {percent:g} must be over 0 and at most 1."
        if percent < last_percent:
            return "End percents must ascend."
        if scale < last_scale:
            return f"Scale {scale:g} follows {last_scale:g}: stages must not shrink the grid."
        last_percent, last_scale = percent, scale

    if stages[-1] != (1.0, 1.0):
        return "The last stage must be 1.0:1.0, so sampling ends at full resolution."
    return ""


def seed_node(roles) -> str:
    """Where the seed is written: whichever seed-carrying node the workflow holds.

    A SamplerCustomAdvanced takes its noise by link, so the seed belongs on the RandomNoise
    feeding it; the progressive sampler makes its own noise from a widget. The contract
    guarantees one of them exists, so this never returns None.
    """
    return roles_mod.first_bound(roles, roles_mod.SEED_ROLES)


def schedule_node(roles) -> str | None:
    """Where the staging schedule goes, or None if this workflow has no sampler that takes
    one -- in which case the Schedule parameter is not offered at all."""
    return roles_mod.first_bound(roles, roles_mod.SCHEDULE_ROLES)


def upscale_node(roles) -> str | None:
    """Where the stage-to-stage upscale method goes, or None if nothing takes one."""
    return roles_mod.first_bound(roles, roles_mod.UPSCALE_ROLES)


def shift_node(roles) -> str | None:
    """Where the video sigma shift goes, or None if the workflow has no shift node."""
    return roles_mod.first_bound(roles, roles_mod.SHIFT_ROLES)


def _image_node_id(roles, index: int) -> str:
    return str(roles.injected["image"] + index)


def _video_node_id(roles, index: int) -> str:
    """The single VHS_LoadVideo that loads, windows and de-muxes reference video ``index``."""
    return str(roles.injected["video"] + roles.injected["video_stride"] * index)


def _audio_node_ids(roles, index: int) -> tuple[str, str]:
    """(LoadAudio id, TrimAudioDuration id) for reference audio ``index``."""
    base = roles.injected["audio"] + roles.injected["audio_stride"] * index
    return str(base), str(base + 1)


@dataclass
class BuildState:
    """Everything the builder needs. Mirrors the GUI's editable fields."""

    #: The prompt is authored in named sections and joined on the way out; prompt_text is
    #: derived from these rather than stored, so the two can never drift apart.
    prompt_sections: dict = field(default_factory=prompt_mod.empty_sections)
    aspect_ratio: str = config.DEFAULT_ASPECT_RATIO
    megapixels: float = config.DEFAULT_MEGAPIXELS
    duration_seconds: float = config.DEFAULT_DURATION
    #: Written to whichever seed-carrying node the workflow holds -- see seed_node.
    seed: int = config.DEFAULT_SEED
    #: Sampling steps (h3-scheduler).
    steps: int = config.DEFAULT_STEPS
    #: Which solver walks the sigmas (h3-sampler).
    sampler_name: str = config.DEFAULT_SAMPLER
    #: How the sigmas are spaced across those steps (h3-scheduler).
    scheduler: str = config.DEFAULT_SCHEDULER
    #: Progressive-resolution staging, "scale:end_percent" per stage. Only reaches the
    #: graph when the workflow has a sampler that takes one -- see schedule_node.
    schedule: str = config.DEFAULT_SCHEDULE
    #: How the estimate is resampled between staging stages -- see upscale_node.
    upscale_method: str = config.DEFAULT_UPSCALE_METHOD
    #: MiniMaxH3SigmaShift.shift_video -- see shift_node.
    shift_video: float = config.DEFAULT_SHIFT_VIDEO
    #: How reference images are sized before encoding (h3-reference).
    ref_image_size: str = config.DEFAULT_REF_IMAGE_SIZE
    #: Where the output node files the result inside ComfyUI's output folder. Empty leaves
    #: the workflow's own value alone.
    filename_prefix: str = ""
    #: Patch the model with Sage Attention (h3-sage / h3-switch).
    sage_attention: bool = config.DEFAULT_SAGE_ATTENTION
    refs: RefSet = field(default_factory=RefSet)

    @property
    def prompt_text(self) -> str:
        """The single string the model receives."""
        return prompt_mod.combine(self.prompt_sections)

    @property
    def resolution(self) -> tuple[int, int]:
        return mathmirror.resolution(self.aspect_ratio, self.megapixels)

    @property
    def frames(self) -> int:
        return mathmirror.frames_from_seconds(self.duration_seconds)


@dataclass
class BuiltGraph:
    """A ready-to-POST graph plus the metadata the UI needs to explain it."""

    graph: dict
    #: node id -> human label, used to turn a server node_error into "Reference image 3 (x.png)"
    labels: dict[str, str]
    tags: TagAssignment
    width: int
    height: int
    frames: int
    #: Nodes dropped because nothing consumes them, reported so their absence is visible
    #: rather than silent.
    pruned: list = field(default_factory=list)

    def node_label(self, node_id: str) -> str:
        return self.labels.get(node_id, f"node {node_id}")


def validate_state(state: BuildState) -> list[str]:
    """Client-side blocking problems, checked before anything touches the network."""
    problems: list[str] = []

    err = mathmirror.duration_error(state.duration_seconds)
    if err:
        problems.append(err)

    try:
        width, height = state.resolution
    except ValueError as exc:
        problems.append(str(exc))
    else:
        err = mathmirror.resolution_error(width, height)
        if err:
            problems.append(err)

    for kind, rows in (
        ("image", state.refs.images),
        ("video", state.refs.videos),
        ("audio", state.refs.audios),
    ):
        limit = {"image": config.MAX_REF_IMAGES,
                 "video": config.MAX_REF_VIDEOS,
                 "audio": config.MAX_REF_AUDIOS}[kind]
        if len(rows) > limit:
            problems.append(f"Too many reference {kind}s: {len(rows)} (model accepts {limit})")
        for i, row in enumerate(rows, start=1):
            if not row.comfy_name:
                problems.append(
                    f"Reference {kind} {i} ({row.display_name}) has not been uploaded yet"
                )

    return problems


#: role -> the name shown when the server reports an error against that node.
ROLE_LABELS = {
    "promptinput": "Prompt",
    "reference": "MiniMax H3 reference conditioning",
    "sampler": "Sampler",
    "sampleradvanced": "Sampling",
    "progressivesampler": "Staged sampler",
    "scheduler": "Sampling schedule",
    "guider": "Guider",
    "shift": "Sigma shift",
    "noise": "Seed",
    "sage": "Sage Attention",
    "switch": "Sage Attention switch",
    "loadmodel": "Diffusion model",
    "loadclip": "Text encoder",
    "loadvideovae": "Video VAE",
    "loadaudiovae": "Audio VAE",
    "imagedecode": "Video decode",
    "audiodecode": "Audio decode",
    "preview": "Live preview",
    "vidcombine": "Save video",
}


def build_graph(base: dict, state: BuildState, roles) -> BuiltGraph:
    """Clone the base workflow and rewrite it to match ``state``.

    Every reference row must already carry a ``comfy_name`` (the filename as it exists in
    ComfyUI's input directory); uploading is the job runner's responsibility.
    """
    graph = copy.deepcopy(base)
    labels: dict[str, str] = {}

    duration = mathmirror.clamp_duration(state.duration_seconds)
    width, height = mathmirror.resolution(state.aspect_ratio, state.megapixels)
    frames = mathmirror.frames_from_seconds(duration)

    # --- scalar widget values -----------------------------------------------------
    graph[roles.promptinput]["inputs"]["value"] = state.prompt_text

    # Width, height and length are literals: the workflow has no node that computes them,
    # so mathmirror's quantisation is the only thing keeping them on the 32px and 17k+5
    # grids the model requires.
    reference_inputs = graph[roles.reference]["inputs"]
    reference_inputs["width"] = int(width)
    reference_inputs["height"] = int(height)
    reference_inputs["length"] = int(frames)
    reference_inputs["ref_image_size"] = clean_ref_image_size(state.ref_image_size)

    graph[seed_node(roles)]["inputs"]["noise_seed"] = int(state.seed)
    # Only when the workflow has a sampler that declares one: writing a schedule into a
    # graph whose sampler does not take one is an input no node declares, which the server
    # would reject outright.
    staging = schedule_node(roles)
    if staging:
        graph[staging]["inputs"]["schedule"] = clean_schedule(state.schedule)
    upscaler = upscale_node(roles)
    if upscaler:
        graph[upscaler]["inputs"]["upscale_method"] = clean_upscale_method(
            state.upscale_method)
    shift = shift_node(roles)
    if shift:
        graph[shift]["inputs"]["shift_video"] = clamp_shift(state.shift_video)

    # Clamped rather than rejected: the widgets cannot produce an out-of-range value, but
    # a hand-edited settings.json can, and a 400 from /prompt explains it far less well.
    scheduler_inputs = graph[roles.scheduler]["inputs"]
    scheduler_inputs["steps"] = clamp_steps(state.steps)
    scheduler_inputs["scheduler"] = clean_scheduler(state.scheduler)
    graph[roles.sampler]["inputs"]["sampler_name"] = clean_sampler(state.sampler_name)

    # Only when something was asked for. An empty prefix means the workflow's own value
    # stands, which is what a user who edited the output node in ComfyUI expects -- and
    # keeps this out of the tier-0 diff on a workflow nobody has overridden.
    prefix = config.clean_filename_prefix(state.filename_prefix)
    if prefix:
        graph[roles.vidcombine]["inputs"]["filename_prefix"] = prefix

    _strip_frontend_inputs(graph)
    _apply_sage(graph, roles, bool(state.sage_attention))

    for role, text in ROLE_LABELS.items():
        node_id = roles.optional(role)
        if node_id:
            labels[node_id] = text

    # --- strip whatever reference wiring the workflow shipped with -----------------
    for key in [k for k in reference_inputs if k.startswith(REF_KEY_PREFIXES)]:
        del reference_inputs[key]

    # Dropped unconditionally: the baked reference loaders are re-created from the user's
    # own lists (seeded from these same filenames on first launch), so there is one code
    # path rather than a "did the list change?" comparison.
    for node_id in roles.many("refimage") + roles.many("refvideo"):
        graph.pop(node_id, None)

    # --- inject the configured loaders ---------------------------------------------
    _inject_images(graph, roles, reference_inputs, labels, state.refs)
    _inject_videos(graph, roles, reference_inputs, labels, state.refs, frames)
    _inject_audios(graph, roles, reference_inputs, labels, state.refs, frames)

    pruned = prune_orphans(graph, keep=(roles.vidcombine,) + roles.keep)

    return BuiltGraph(
        graph=graph,
        labels=labels,
        tags=compute_tags(state.refs),
        width=width,
        height=height,
        frames=frames,
        pruned=pruned,
    )


#: Options VideoHelperSuite reads out of the prompt's ``extra_data`` rather than from any
#: widget, and what this app wants them set to. Both default to True on the server.
#:
#: VHS_VideoCombine writes three files for a run with audio: a PNG of the first frame
#: "to keep metadata", the silent video, and then the muxed one. Only the last is wanted
#: here -- the app already downloads exactly that one, so the other two are litter in the
#: output folder that nothing ever reads.
#:
#: MetadataImage=False skips the PNG. KeepIntermediate=False deletes the silent video once
#: the mux has succeeded. The prompt itself still travels in the video's own metadata,
#: which is what the node's `save_metadata` widget controls, so nothing is lost with it.
VHS_EXTRA_OPTIONS = {
    "VHS_MetadataImage": False,
    "VHS_KeepIntermediate": False,
}

#: Classes that read VHS_EXTRA_OPTIONS.
VHS_OPTION_READERS = ("VHS_VideoCombine",)


def extra_data_for(graph: dict) -> dict:
    """The ``extra_data`` to submit alongside ``graph``, or {} if it needs none.

    Sent only when the graph actually holds a node that reads it. ``extra_pnginfo`` is
    also what several core nodes embed in their output's metadata, so populating it
    unconditionally would write this app's private options into every saved file.
    """
    classes = {
        node.get("class_type") for node in graph.values() if isinstance(node, dict)
    }
    if not classes.intersection(VHS_OPTION_READERS):
        return {}
    return {"extra_pnginfo": {"workflow": {"extra": dict(VHS_EXTRA_OPTIONS)}}}


def geometry_warnings(graph: dict, roles) -> list[str]:
    """Where the workflow disagrees with the constants this app computes against.

    The frame rate is the one that matters: it lives in ``config.FPS`` here and in the
    output node's own widget there, and nothing links the two. Set them differently and
    every render succeeds at the wrong length, which is exactly the kind of failure that
    goes unnoticed for weeks.
    """
    problems: list[str] = []

    combine = graph.get(roles.vidcombine) or {}
    inputs = combine.get("inputs") or {}
    fps = inputs.get("frame_rate", inputs.get("fps"))
    if isinstance(fps, (int, float)) and float(fps) != float(config.FPS):
        problems.append(
            f"{combine.get('class_type', 'the output node')} writes {fps:g} fps but "
            f"durations here are computed at {config.FPS} fps, so a clip will come out "
            f"{config.FPS / float(fps):.2f}x its intended length. "
            "Match the node's frame rate to config.FPS, or the other way round."
        )
    return problems


def _strip_frontend_inputs(graph: dict) -> None:
    """Drop the keys ComfyUI's frontend adds to an API export that no node declares.

    The server ignores them, but sending them means the app's own validator has to treat
    an unknown input as harmless -- and an unknown input is usually a typo worth hearing
    about.
    """
    for node in graph.values():
        for class_type, key in config.FRONTEND_ONLY_INPUTS:
            if node.get("class_type") == class_type:
                node["inputs"].pop(key, None)


def _apply_sage(graph: dict, roles, enabled: bool) -> None:
    """Point the model chain through the Sage patch, or around it entirely.

    Repointing rather than only setting the flag, because whether the flag is enough
    depends on which switch node the workflow carries, and the builder should not have to
    know:

    * ComfyUI's own ``ComfySwitchNode`` declares both branches ``lazy=True`` and asks for
      only the one it took, so the patch really is skipped.
    * ``Switch`` from ComfyUI-ConditioningKrea2Rebalance, which this workflow used to
      carry, wires both branches eagerly -- the patch node stays an ancestor of an output
      and ComfyUI executes it either way. It is configured with ``allow_compile``, which is
      not something to run for a setting that is switched off.

    Repointing the switch's consumers at the loader is correct for both: it leaves the
    patch node and the switch feeding nothing, and the orphan sweep takes them out, so the
    graph that goes to the server contains only what it is actually going to use.
    """
    switch_id = roles.optional("switch")
    if switch_id is None:
        return
    if enabled:
        graph[switch_id]["inputs"]["switch"] = True
        return

    for node in graph.values():
        for key, value in node.get("inputs", {}).items():
            if isinstance(value, list) and len(value) == 2 and value[0] == switch_id:
                node["inputs"][key] = [roles.loadmodel, 0]


def _inject_images(graph: dict, roles, minimax_inputs: dict, labels: dict,
                   refs: RefSet) -> None:
    for index, row in enumerate(refs.images):
        node_id = _image_node_id(roles, index)
        graph[node_id] = {
            "inputs": {"image": row.graph_name()},
            "class_type": "LoadImage",
            "_meta": {"title": f"Ref Image {index + 1}"},
        }
        # slot 0 is IMAGE; slot 1 is the MASK we do not want.
        minimax_inputs[f"ref_images.ref_image_{index}"] = [node_id, 0]
        labels[node_id] = f"Reference image {index + 1} ({row.display_name})"


def _inject_videos(graph: dict, roles, minimax_inputs: dict, labels: dict, refs: RefSet,
                   frames: int) -> None:
    """One ``VHS_LoadVideo`` per reference video, marks and all.

    It replaces four nodes -- ``LoadVideo`` -> ``GetVideoComponents`` ->
    ``ImageFromBatch`` -> ``TrimAudioDuration`` -- and the reason is not tidiness. That
    chain decoded the **whole file** into a float32 IMAGE batch and then threw most of it
    away: a 226-second 640x480 reference is 6,777 frames, about 25 GB of CPU RAM, built
    before the diffusion model had even begun loading. VHS decodes only the frames asked
    for, so the same reference costs a few hundred megabytes.

    ``frame_load_cap`` is always the generated length, because that is what
    MiniMaxH3ReferenceToVideo truncates a reference to anyway -- decoding more is work
    whose output is discarded. A mark therefore says *where to start* and nothing else:
    the section runs for the length of the clip being generated, so there is no out point
    to set. It used to be a link to the workflow's frame-count expression; that node is
    gone, so it is now the same literal written into the reference node's ``length``.

    The soundtrack comes off slot 2 of the same node, already windowed to the same span:
    VHS derives the audio start and duration from ``skip_first_frames`` and
    ``frame_load_cap`` using the source's own frame time. That removes the frames-by-index
    versus sound-by-time conversion the app used to need a probed frame rate for.
    """
    for index, row in enumerate(refs.videos):
        node_id = _video_node_id(roles, index)
        start = max(0, int(row.trim_start))

        graph[node_id] = {
            "inputs": {
                "video": row.graph_name(),
                # Never resample: the mark is a frame index into the source.
                "force_rate": 0,
                "custom_width": config.REF_VIDEO_WIDTH,
                "custom_height": config.REF_VIDEO_HEIGHT,
                # The generated length, marked or not: there is no out point, because the
                # section runs for the whole clip wherever it starts.
                "frame_load_cap": int(frames),
                "skip_first_frames": start,
                "select_every_nth": 1,
                "format": config.REF_VIDEO_FORMAT,
            },
            "class_type": "VHS_LoadVideo",
            "_meta": {"title": f"Ref Video {index + 1}"},
        }
        minimax_inputs[f"ref_videos.ref_video_{index}"] = [node_id, 0]

        if row.use_soundtrack:
            # The node pairs a soundtrack to its video purely by the trailing index, so
            # this must reuse the same index. Sparse indices are legal.
            minimax_inputs[f"ref_video_audios.ref_video_audio_{index}"] = [node_id, 2]

        parts = [row.display_name, f"{frames} frames"]
        if start:
            parts.insert(1, f"from {start}")
        labels[node_id] = f"Reference video {index + 1} ({', '.join(parts)})"


def _inject_audios(graph: dict, roles, minimax_inputs: dict, labels: dict, refs: RefSet,
                   frames: int) -> None:
    """One ``LoadAudio`` and one ``TrimAudioDuration`` per reference audio.

    Cut the same way a video is, and just as unconditionally: a start, and a duration that
    is the generated length in seconds rather than anything the user sets. The model would
    truncate to that length anyway, so trimming here only makes what it does explicit --
    and it means an unmarked reference and a marked one build the same shape of graph.
    Asking for more than the file holds is harmless: the node hands back what is there.
    """
    duration = mathmirror.true_seconds(frames)
    for index, row in enumerate(refs.audios):
        load_id, trim_id = _audio_node_ids(roles, index)
        graph[load_id] = {
            "inputs": {"audio": row.comfy_name},
            "class_type": "LoadAudio",
            "_meta": {"title": f"Ref Audio {index + 1}"},
        }
        start = max(0.0, float(row.trim_start))
        graph[trim_id] = _trim_audio_node(
            [load_id, 0], start, duration, f"Ref Audio {index + 1} trim")

        minimax_inputs[f"ref_audios.ref_audio_{index}"] = [trim_id, 0]
        label = f"Reference audio {index + 1} ({row.display_name}"
        label += f", from {start:.2f}s" if start else ""
        label += f", {duration:.2f}s)"
        labels[load_id] = label
        labels[trim_id] = f"Reference audio {index + 1} trim"


def _trim_audio_node(audio_link, start: float, length: float, title: str) -> dict:
    return {
        "inputs": {
            "audio": audio_link,
            "start_index": round(float(start), 3),
            "duration": round(float(length), 3),
        },
        "class_type": "TrimAudioDuration",
        "_meta": {"title": title},
    }


def prune_orphans(graph: dict, keep: tuple[str, ...]) -> list[str]:
    """Drop nodes nothing downstream consumes. Returns what was removed.

    ComfyUI only executes the ancestors of output nodes, so such a node would never run
    anyway -- but it would still be validated here, and an unwired node with a missing
    required input would wrongly block the queue. What this mostly fires on is Sage
    Attention being switched off, which leaves the patch node and its switch feeding
    nothing.

    ``keep`` is the set of roots: the output node, plus every node the user tagged
    ``h3-keep``. That tag is what makes a second output branch -- a preview, another
    save -- survive a build instead of being silently swept away.
    """
    removed: list[str] = []
    while True:
        referenced = {
            link[0]
            for node in graph.values()
            for link in node.get("inputs", {}).values()
            if isinstance(link, list) and len(link) == 2 and isinstance(link[0], str)
        }
        orphans = [nid for nid in graph if nid not in referenced and nid not in keep]
        if not orphans:
            return removed
        for node_id in orphans:
            removed.append(f"{node_id} ({graph[node_id].get('class_type', '?')})")
            del graph[node_id]


def canonicalise(graph: dict, roles) -> dict:
    """Normalise a graph so an unmodified build diffs cleanly against the base JSON.

    Four normalisations, all of them semantics-preserving:
      * injected loaders are renamed back onto the baked node IDs they replace, so a diff
        shows real value changes rather than renumbering;
      * ``_meta`` is dropped -- it holds display titles ComfyUI never executes on;
      * integral floats collapse to int, so 5.0 does not read as a change from 5;
      * node and input ordering is sorted.
    """
    rename = {
        _image_node_id(roles, i): baked
        for i, baked in enumerate(roles.many("refimage"))
        if _image_node_id(roles, i) in graph
    }
    rename.update({
        _video_node_id(roles, k): baked
        for k, baked in enumerate(roles.many("refvideo"))
        if _video_node_id(roles, k) in graph
    })

    def remap(node_id: str) -> str:
        return rename.get(node_id, node_id)

    out = {}
    for node_id, node in graph.items():
        inputs = {}
        for key, value in sorted(node.get("inputs", {}).items()):
            if isinstance(value, list) and len(value) == 2 and isinstance(value[0], str):
                value = [remap(value[0]), value[1]]
            elif isinstance(value, float) and value.is_integer():
                value = int(value)
            else:
                value = copy.deepcopy(value)
            inputs[key] = value
        out[remap(node_id)] = {"class_type": node["class_type"], "inputs": inputs}
    return dict(sorted(out.items(), key=lambda kv: int(kv[0]) if kv[0].isdigit() else kv[0]))


#: Roles the app deliberately builds differently from the shipped workflow, and why.
#: Everything else must match, which is what Tier 0 checks.
INTENDED_DIFFERENCES = {
    "promptinput": "the prompt is assembled from the six sections",
    "reference": (
        "width, height and length are computed here from aspect ratio, megapixels and "
        "duration, so they match the workflow's literals only when those literals "
        "happen to sit on the same 32px and 17k+5 grids"
    ),
}


def intended_difference_ids(roles) -> dict:
    """``INTENDED_DIFFERENCES`` keyed by node id for the graph currently loaded."""
    return {
        roles.optional(role): why
        for role, why in INTENDED_DIFFERENCES.items()
        if roles.optional(role)
    }


def canonical_reference(base: dict, roles) -> dict:
    """The shipped workflow in the form a faithful rebuild should match.

    The same strip, the same Sage decision and the same orphan sweep are applied to both
    sides, so nothing the builder drops because ComfyUI would never execute it reads as a
    change the app made. The Sage decision belongs here because it is the *workflow's* --
    a first launch takes `sage_attention` from the switch node's own value, so a workflow
    shipping `switch: false` is one whose faithful rebuild routes around the patch.
    """
    reference = copy.deepcopy(base)
    # The same strip as the build, so a key the frontend wrote and the app drops does not
    # read as a change the app made.
    _strip_frontend_inputs(reference)
    _apply_sage(reference, roles,
                bool(config.defaults_from_workflow(base, roles)["sage_attention"]))
    prune_orphans(reference, keep=(roles.vidcombine,) + roles.keep)
    return canonicalise(reference, roles)


def state_from_workflow(base: dict, roles) -> BuildState:
    """Build the first-launch state: what the shipped workflow encodes.

    Any baked LoadImage filenames become *server-file* rows, so a fresh launch can
    reproduce the original run without uploading anything.
    """
    defaults = config.defaults_from_workflow(base, roles)
    refs = RefSet(
        images=[RefRow(kind=IMAGE, comfy_name=name) for name in defaults["ref_images"]],
        videos=[
            RefRow(kind=VIDEO, comfy_name=name, use_soundtrack=wired)
            for name, wired in zip(defaults["ref_videos"],
                                   defaults["ref_video_soundtracks"])
        ],
    )
    return BuildState(
        # The shipped workflow carries one long description, which is what the
        # detailed_description box is for.
        prompt_sections=prompt_mod.from_legacy(defaults["prompt_text"]),
        aspect_ratio=defaults["aspect_ratio"],
        megapixels=defaults["megapixels"],
        duration_seconds=defaults["duration_seconds"],
        seed=defaults["seed"],
        steps=defaults["steps"],
        sampler_name=defaults["sampler_name"],
        scheduler=defaults["scheduler"],
        schedule=defaults["schedule"],
        upscale_method=defaults["upscale_method"],
        shift_video=defaults["shift_video"],
        ref_image_size=defaults["ref_image_size"],
        sage_attention=defaults["sage_attention"],
        refs=refs,
    )
