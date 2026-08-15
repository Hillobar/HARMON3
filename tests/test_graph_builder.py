"""Tests for the clone / prune / inject / wire pipeline that produces a submittable graph."""

from __future__ import annotations

import copy
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from harmon3 import config, graph_builder, mathmirror, prompt  # noqa: E402
from harmon3 import roles as roles_mod                         # noqa: E402
from harmon3.graph_builder import (                           # noqa: E402
    BuildState,
    build_graph,
    canonicalise,
    state_from_workflow,
    validate_state,
)
from harmon3.refs import AUDIO, IMAGE, VIDEO, RefRow, RefSet   # noqa: E402

WORKFLOW = config.load_workflow()
BASE, ROLES = WORKFLOW.graph, WORKFLOW.roles


def _server_row(kind, name, **kw):
    return RefRow(kind=kind, comfy_name=name, **kw)


def _minimax_inputs(built):
    return built.graph[ROLES.reference]["inputs"]


def _max_refs():
    """A state with every reference slot filled, for the id-collision checks."""
    return BuildState(refs=RefSet(
        images=[_server_row(IMAGE, f"i{i}.png") for i in range(config.MAX_REF_IMAGES)],
        videos=[_server_row(VIDEO, f"v{k}.mp4") for k in range(config.MAX_REF_VIDEOS)],
        audios=[_server_row(AUDIO, f"a{j}.wav") for j in range(config.MAX_REF_AUDIOS)],
    ))


# ---------------------------------------------------------------------------------
# Structural invariants
# ---------------------------------------------------------------------------------

def test_injected_ids_never_collide_with_the_base_graph():
    """Even with every slot filled, no injected loader lands on a workflow node."""
    built = build_graph(BASE, _max_refs(), ROLES)
    injected = set(built.graph) - set(BASE)
    assert injected and not (injected & set(BASE))


def test_injected_ids_clear_a_workflow_numbered_far_higher():
    import copy

    from harmon3 import roles as roles_mod

    # Deep-copied: BASE is module-level and shared with every other test here.
    shifted = {str(int(nid) + 1000): copy.deepcopy(node) for nid, node in BASE.items()}
    for node in shifted.values():
        for key, value in node["inputs"].items():
            if isinstance(value, list) and len(value) == 2 and isinstance(value[0], str):
                node["inputs"][key] = [str(int(value[0]) + 1000), value[1]]
    roles = roles_mod.resolve(shifted)

    built = build_graph(shifted, _max_refs(), roles)
    injected = set(built.graph) - set(shifted)
    assert injected and not (injected & set(shifted))
    assert min(int(nid) for nid in injected) > max(int(nid) for nid in shifted)


def test_base_graph_is_never_mutated():
    before = config.load_workflow().graph
    build_graph(BASE, state_from_workflow(BASE, ROLES), ROLES)
    assert BASE == before


def test_baked_load_image_nodes_are_always_dropped():
    for state in (state_from_workflow(BASE, ROLES), BuildState()):
        built = build_graph(BASE, state, ROLES)
        for node_id in ROLES.many("refimage"):
            assert node_id not in built.graph


def test_no_base_reference_keys_survive():
    """The base wires ref_image_0/1 to nodes 137/139; every ref key must be rebuilt."""
    built = build_graph(BASE, BuildState(), ROLES)
    assert not [k for k in _minimax_inputs(built) if k.startswith(graph_builder.REF_KEY_PREFIXES)]


def test_empty_refset_produces_no_ref_keys_and_no_loaders():
    built = build_graph(BASE, BuildState(), ROLES)
    inputs = _minimax_inputs(built)
    assert not [k for k in inputs if k.startswith(graph_builder.REF_KEY_PREFIXES)]
    # ref_image_size is a plain widget, not an autogrow group, and must survive.
    assert inputs["ref_image_size"] == config.DEFAULT_REF_IMAGE_SIZE
    classes = {n["class_type"] for n in built.graph.values()}
    assert not (classes & {"LoadImage", "VHS_LoadVideo", "LoadAudio"})


def test_scalar_writes_land_where_expected():
    state = BuildState(
        prompt_sections=prompt.from_legacy("hello <Picture 1>"),
        aspect_ratio="9:16 (Portrait Widescreen)",
        megapixels=1.25,
        duration_seconds=7.5,
        seed=42,
        steps=12,
        scheduler="karras",
        ref_image_size="max",
    )
    built = build_graph(BASE, state, ROLES)
    g = built.graph
    # The prompt reaches the node as the combined sections, not the raw text.
    assert g[ROLES.promptinput]["inputs"]["value"] == prompt.combine(state.prompt_sections)
    assert "detailed_description:\nhello <Picture 1>" in g[ROLES.promptinput]["inputs"]["value"]
    # Geometry is written as literals: the workflow has no node that computes it.
    reference = g[ROLES.reference]["inputs"]
    assert (reference["width"], reference["height"]) == mathmirror.resolution(
        "9:16 (Portrait Widescreen)", 1.25)
    assert reference["length"] == mathmirror.frames_from_seconds(7.5)
    assert reference["ref_image_size"] == "max"
    assert g[graph_builder.seed_node(ROLES)]["inputs"]["noise_seed"] == 42
    assert g[ROLES.scheduler]["inputs"]["steps"] == 12
    assert g[ROLES.scheduler]["inputs"]["scheduler"] == "karras"


# ---------------------------------------------------------------------------------
# The geometry grids, which this side owns entirely
# ---------------------------------------------------------------------------------

def test_the_dimension_grid_is_always_32():
    built = build_graph(BASE, BuildState(megapixels=1.25), ROLES)
    reference = built.graph[ROLES.reference]["inputs"]
    assert reference["width"] % 32 == 0 and reference["height"] % 32 == 0
    assert built.width % 32 == 0 and built.height % 32 == 0


@pytest.mark.parametrize("seconds", [0.2, 1.0, 5.0, 5.17, 7.5, 30.0, 120.0])
def test_the_written_length_is_always_on_the_models_grid(seconds):
    """The reference node rejects a length that is not 17k+5, and nothing on the server
    rounds it any more -- this is the only thing standing between the user and a 400."""
    built = build_graph(BASE, BuildState(duration_seconds=seconds), ROLES)
    length = built.graph[ROLES.reference]["inputs"]["length"]
    assert length % config.FRAME_MOD == config.FRAME_REM
    assert length == built.frames


def test_geometry_is_written_as_numbers_not_links():
    """They used to be links to ResolutionSelector and a maths expression. If either ever
    comes back as a link, mathmirror's readouts would silently stop being the truth."""
    reference = build_graph(BASE, BuildState(), ROLES).graph[ROLES.reference]["inputs"]
    for key in ("width", "height", "length"):
        assert isinstance(reference[key], int), key


def test_the_grid_is_not_something_the_state_carries():
    """It used to be editable; a stale value in a settings file must not resurrect it."""
    assert "multiple" not in BuildState.__dataclass_fields__


# ---------------------------------------------------------------------------------
# Steps and reference sizing
# ---------------------------------------------------------------------------------

def test_the_workflows_own_values_are_the_starting_point():
    built = build_graph(BASE, state_from_workflow(BASE, ROLES), ROLES)
    assert built.graph[ROLES.scheduler]["inputs"]["steps"] == 40
    assert built.graph[ROLES.reference]["inputs"]["ref_image_size"] == "match"


@pytest.mark.parametrize("given, expected", [
    (0, config.MIN_STEPS),
    (999_999, config.MAX_STEPS),
    ("nonsense", config.DEFAULT_STEPS),
    (None, config.DEFAULT_STEPS),
])
def test_an_impossible_step_count_is_brought_into_range(given, expected):
    """The spin box cannot produce one, but a hand-edited settings.json can."""
    built = build_graph(BASE, BuildState(steps=given), ROLES)
    assert built.graph[ROLES.scheduler]["inputs"]["steps"] == expected


@pytest.mark.parametrize("given", ["", "huge", None, 7])
def test_an_unknown_reference_size_falls_back_to_the_default(given):
    built = build_graph(BASE, BuildState(ref_image_size=given), ROLES)
    assert (built.graph[ROLES.reference]["inputs"]["ref_image_size"]
            == config.DEFAULT_REF_IMAGE_SIZE)


# ---------------------------------------------------------------------------------
# The seed, and the sampler chain around it
# ---------------------------------------------------------------------------------

def test_the_seed_goes_to_whichever_node_carries_one():
    """Two arrangements are legal: a RandomNoise feeding SamplerCustomAdvanced, or the
    progressive sampler making its own noise from a widget. Which one is present decides
    where the seed goes, and nothing else in the app needs to know which it was."""
    seed_id = graph_builder.seed_node(ROLES)
    assert seed_id, "the contract guarantees a seed target"

    built = build_graph(BASE, BuildState(seed=1234), ROLES)
    assert built.graph[seed_id]["inputs"]["noise_seed"] == 1234

    # Exactly one node ends up carrying it, so a stale second copy cannot disagree.
    carriers = [nid for nid, node in built.graph.items() if "noise_seed" in node["inputs"]]
    assert carriers == [seed_id]


def test_the_progressive_sampler_makes_its_own_noise():
    """It declares noise_seed rather than taking a NOISE link, which is why there is no
    RandomNoise in this workflow for the seed to go to."""
    inputs = build_graph(BASE, state_from_workflow(BASE, ROLES), ROLES).graph[
        ROLES.progressivesampler]["inputs"]
    assert "noise" not in inputs
    assert isinstance(inputs["noise_seed"], int)


# ---------------------------------------------------------------------------------
# The staging schedule, which only exists on a progressive sampler
# ---------------------------------------------------------------------------------

def test_the_schedule_reaches_the_progressive_sampler():
    built = build_graph(BASE, BuildState(schedule="0.5:0.4, 0.75:0.7, 1.0:1.0"), ROLES)
    assert (built.graph[ROLES.progressivesampler]["inputs"]["schedule"]
            == "0.5:0.4, 0.75:0.7, 1.0:1.0")


@pytest.mark.parametrize("given", ["", "   ", None])
def test_an_empty_schedule_falls_back_to_the_default(given):
    built = build_graph(BASE, BuildState(schedule=given), ROLES)
    assert (built.graph[ROLES.progressivesampler]["inputs"]["schedule"]
            == config.DEFAULT_SCHEDULE)


def test_a_schedule_the_node_would_refuse_is_still_sent_as_typed():
    """The node owns the grammar; substituting a different schedule would be worse than
    the error, and the panel warns before Queue anyway."""
    built = build_graph(BASE, BuildState(schedule="2.0:1.0"), ROLES)
    assert built.graph[ROLES.progressivesampler]["inputs"]["schedule"] == "2.0:1.0"


def test_no_schedule_is_written_to_a_sampler_that_does_not_take_one():
    """A SamplerCustomAdvanced workflow must not be sent a 'schedule' key: no node
    declares it, and the server rejects the whole prompt for an unknown input."""
    import copy

    from harmon3 import roles as roles_mod

    graph = copy.deepcopy(BASE)
    staged = graph.pop(ROLES.progressivesampler)
    graph["900"] = {
        "inputs": {"noise": ["901", 0], "guider": staged["inputs"]["guider"],
                   "sampler": staged["inputs"]["sampler"],
                   "sigmas": staged["inputs"]["sigmas"],
                   "latent_image": staged["inputs"]["latent_image"]},
        "class_type": "SamplerCustomAdvanced",
        "_meta": {"title": "h3-sampleradvanced"},
    }
    graph["901"] = {"inputs": {"noise_seed": 0}, "class_type": "RandomNoise",
                    "_meta": {"title": "h3-noise"}}
    for node in graph.values():
        for key, value in node["inputs"].items():
            if isinstance(value, list) and value[0] == ROLES.progressivesampler:
                node["inputs"][key] = ["900", value[1]]

    roles = roles_mod.resolve(graph)
    assert graph_builder.schedule_node(roles) is None
    assert graph_builder.seed_node(roles) == "901"

    built = build_graph(graph, BuildState(seed=5, schedule="0.5:0.55, 1.0:1.0"), roles)
    assert built.graph["901"]["inputs"]["noise_seed"] == 5
    assert not [n for n in built.graph.values() if "schedule" in n["inputs"]]


# ---------------------------------------------------------------------------------
# Stage upscale and sigma shift
# ---------------------------------------------------------------------------------

def test_the_upscale_method_reaches_the_progressive_sampler():
    built = build_graph(BASE, BuildState(upscale_method="bislerp"), ROLES)
    assert (built.graph[ROLES.progressivesampler]["inputs"]["upscale_method"]
            == "bislerp")


@pytest.mark.parametrize("given", ["", None, 7, "lanczos", "BICUBIC"])
def test_an_upscale_method_the_node_does_not_offer_falls_back(given):
    """Checked against the list, unlike sampler names: this combo is the node's own fixed
    set rather than something a plugin pack extends."""
    built = build_graph(BASE, BuildState(upscale_method=given), ROLES)
    assert (built.graph[ROLES.progressivesampler]["inputs"]["upscale_method"]
            == config.DEFAULT_UPSCALE_METHOD)


def test_the_sigma_shift_reaches_the_shift_node():
    built = build_graph(BASE, BuildState(shift_video=7.25), ROLES)
    assert built.graph[ROLES.shift]["inputs"]["shift_video"] == 7.25


def test_the_audio_sigma_shift_is_left_as_the_workflow_sets_it():
    """Only shift_video is exposed; writing both would silently take a decision the
    workflow had already made."""
    before = BASE[ROLES.shift]["inputs"]["shift_audio"]
    built = build_graph(BASE, BuildState(shift_video=9.0), ROLES)
    assert built.graph[ROLES.shift]["inputs"]["shift_audio"] == before


@pytest.mark.parametrize("given, expected", [
    (0, config.MIN_SHIFT),
    (-5, config.MIN_SHIFT),
    (10_000, config.MAX_SHIFT),
    (3.14159, 3.14),                       # the node's step is 0.01
    ("nonsense", config.DEFAULT_SHIFT_VIDEO),
    (None, config.DEFAULT_SHIFT_VIDEO),
    (float("nan"), config.DEFAULT_SHIFT_VIDEO),
])
def test_an_impossible_sigma_shift_is_brought_into_range(given, expected):
    built = build_graph(BASE, BuildState(shift_video=given), ROLES)
    assert built.graph[ROLES.shift]["inputs"]["shift_video"] == expected


def test_neither_is_written_to_a_workflow_that_has_no_node_for_it():
    """Same hazard as the schedule: an input no node declares makes ComfyUI reject the
    whole prompt, so a workflow without these nodes must be sent neither key."""
    import copy

    from harmon3 import roles as roles_mod

    graph = copy.deepcopy(BASE)
    shift_id = ROLES.shift
    source = graph[shift_id]["inputs"]["model"]
    for node in graph.values():
        for key, value in node["inputs"].items():
            if isinstance(value, list) and value[0] == shift_id:
                node["inputs"][key] = source
    del graph[shift_id]
    del graph[ROLES.progressivesampler]["inputs"]["upscale_method"]
    graph[ROLES.progressivesampler]["_meta"]["title"] = "h3-noise"
    graph[ROLES.progressivesampler]["class_type"] = "RandomNoise"

    roles = roles_mod.resolve(graph)
    assert graph_builder.shift_node(roles) is None
    assert graph_builder.upscale_node(roles) is None

    built = build_graph(graph, BuildState(shift_video=9.0, upscale_method="area"), roles)
    assert not [n for n in built.graph.values() if "shift_video" in n["inputs"]]
    assert not [n for n in built.graph.values() if "upscale_method" in n["inputs"]]


def test_the_workflows_own_shift_and_upscale_are_the_starting_point():
    state = state_from_workflow(BASE, ROLES)
    assert state.shift_video == BASE[ROLES.shift]["inputs"]["shift_video"]
    assert state.upscale_method == (
        BASE[ROLES.progressivesampler]["inputs"]["upscale_method"])


@pytest.mark.parametrize("text", [
    "1.0:1.0",
    "0.5:0.55, 1.0:1.0",
    "0.5:0.4 0.75:0.7 1.0:1.0",          # whitespace separates just as well
    "0.5:0.4,0.75:0.7,1.0:1.0",
])
def test_schedules_the_node_accepts_raise_no_warning(text):
    assert graph_builder.schedule_error(text) == ""


@pytest.mark.parametrize("text, fragment", [
    ("", "empty"),
    ("0.5", "scale:end_percent"),
    ("half:0.5, 1.0:1.0", "numeric"),
    ("0.0:0.5, 1.0:1.0", "Scale"),
    ("0.5:1.5, 1.0:1.0", "End percent"),
    ("0.5:0.8, 0.75:0.4, 1.0:1.0", "ascend"),
    ("0.75:0.4, 0.5:0.8, 1.0:1.0", "shrink"),
    ("0.5:0.55", "last stage"),
    ("0.5:0.55, 1.0:0.9", "last stage"),
])
def test_a_schedule_the_node_would_refuse_is_explained(text, fragment):
    """Mirrors the node's own parser, so a doomed run says so before it is queued."""
    assert fragment in graph_builder.schedule_error(text)


@pytest.mark.parametrize("given", ["", "   ", None, 7])
def test_a_missing_sampler_or_scheduler_falls_back_to_the_workflows(given):
    built = build_graph(BASE, BuildState(scheduler=given, sampler_name=given), ROLES)
    assert (built.graph[ROLES.scheduler]["inputs"]["scheduler"]
            == config.DEFAULT_SCHEDULER)
    assert (built.graph[ROLES.sampler]["inputs"]["sampler_name"]
            == config.DEFAULT_SAMPLER)


def test_a_sampler_this_build_does_not_list_is_still_sent():
    """config.SAMPLERS is stock ComfyUI's; a server with packs installed has more, and
    silently swapping the solver would change the picture without saying so."""
    built = build_graph(BASE, BuildState(sampler_name="res_2m_ode", scheduler="beta_57"), ROLES)
    assert built.graph[ROLES.sampler]["inputs"]["sampler_name"] == "res_2m_ode"
    assert built.graph[ROLES.scheduler]["inputs"]["scheduler"] == "beta_57"


def test_unspecified_workflow_values_are_left_alone():
    """Sampler, frame rate, codec and filename_prefix are not exposed."""
    built = build_graph(BASE, state_from_workflow(BASE, ROLES), ROLES)
    g = built.graph
    assert g[ROLES.sampler]["inputs"]["sampler_name"] == "euler"
    combine = g[ROLES.vidcombine]["inputs"]
    assert combine["frame_rate"] == 24
    assert combine["filename_prefix"] == "videos/h3"
    assert combine["format"] == "video/h264-mp4"
    # Model names carry Windows backslashes that must survive verbatim.
    assert g[ROLES.loadmodel]["inputs"]["unet_name"] == (
        "minimax_h3\\minimax_h3_ref2va_pruned_int8_convrot.safetensors"
    )


def test_the_output_nodes_frame_rate_matches_the_one_durations_are_computed_at():
    """Nothing links config.FPS to the node's own widget, so a mismatch would render
    every clip at the wrong length without failing. geometry_warnings is the guard."""
    assert graph_builder.geometry_warnings(BASE, ROLES) == []

    wrong = {nid: dict(node) for nid, node in BASE.items()}
    wrong[ROLES.vidcombine] = {
        **BASE[ROLES.vidcombine],
        "inputs": {**BASE[ROLES.vidcombine]["inputs"], "frame_rate": 30},
    }
    warnings = graph_builder.geometry_warnings(wrong, ROLES)
    assert len(warnings) == 1 and "30 fps" in warnings[0]


def test_duration_is_clamped_to_the_models_ceiling():
    built = build_graph(BASE, BuildState(duration_seconds=10_000), ROLES)
    assert built.graph[ROLES.reference]["inputs"]["length"] <= config.MAX_FRAMES
    assert built.frames <= config.MAX_FRAMES


# ---------------------------------------------------------------------------------
# Reference injection
# ---------------------------------------------------------------------------------

def test_images_wire_contiguously_from_index_zero():
    refs = RefSet(images=[_server_row(IMAGE, f"i{i}.png") for i in range(3)])
    built = build_graph(BASE, BuildState(refs=refs), ROLES)
    inputs = _minimax_inputs(built)
    for i in range(3):
        node_id = str(ROLES.injected['image'] + i)
        assert inputs[f"ref_images.ref_image_{i}"] == [node_id, 0]
        assert built.graph[node_id]["class_type"] == "LoadImage"
        assert built.graph[node_id]["inputs"]["image"] == f"i{i}.png"


def test_video_loads_through_one_vhs_node():
    """Frames and sound off the same node, so nothing can decode more than is wanted."""
    refs = RefSet(videos=[_server_row(VIDEO, "clip.mp4", use_soundtrack=True)])
    built = build_graph(BASE, BuildState(refs=refs), ROLES)
    inputs = _minimax_inputs(built)
    load_id = str(ROLES.injected['video'])

    assert built.graph[load_id]["class_type"] == "VHS_LoadVideo"
    assert built.graph[load_id]["inputs"]["video"] == "clip.mp4"   # VHS calls it "video"
    assert inputs["ref_videos.ref_video_0"] == [load_id, 0]        # slot 0 = images
    assert inputs["ref_video_audios.ref_video_audio_0"] == [load_id, 2]   # slot 2 = audio


def test_a_reference_video_is_not_resized_on_the_way_in():
    """MiniMax scales every reference to its own canvas anyway, so resizing here only
    changes how much has to be carried to get there. The workflow used to ask for a 1536px
    height, which is 6.3 GB of frames instead of 0.5 GB for a picture it then shrinks; it
    now ships both at 0, and this keeps the app from drifting back."""
    built = build_graph(BASE, BuildState(refs=RefSet(videos=[_server_row(VIDEO, "clip.mp4")])), ROLES)
    loader = built.graph[str(ROLES.injected['video'])]["inputs"]
    assert (loader["custom_width"], loader["custom_height"]) == (0, 0)


def test_a_reference_video_is_never_decoded_in_full():
    """The bug this replaced: GetVideoComponents built the whole file as a float32 batch,
    25 GB of CPU RAM for a 226-second reference, before the model began loading."""
    built = build_graph(BASE, BuildState(refs=RefSet(videos=[_server_row(VIDEO, "long.mp4")])), ROLES)
    loader = built.graph[str(ROLES.injected['video'])]["inputs"]

    # The generated length as a literal: the workflow no longer carries a node that
    # computes it, so this side is the only thing deciding how much gets decoded.
    assert loader["frame_load_cap"] == built.frames
    assert "GetVideoComponents" not in {n["class_type"] for n in built.graph.values()}


def test_unchecked_soundtrack_omits_the_key_entirely():
    refs = RefSet(videos=[_server_row(VIDEO, "silent.mp4", use_soundtrack=False)])
    built = build_graph(BASE, BuildState(refs=refs), ROLES)
    inputs = _minimax_inputs(built)
    assert "ref_videos.ref_video_0" in inputs
    # Omitted, not null: the node does `.get(...)` and a null would work, but omission
    # keeps the submitted graph unambiguous.
    assert "ref_video_audios.ref_video_audio_0" not in inputs
    # The loader is still there; only its audio output goes unused.
    assert str(ROLES.injected['video']) in built.graph


def test_sparse_soundtrack_indices_are_legal():
    """Only the second video has audio: ref_video_audio_1 exists with no _0."""
    refs = RefSet(videos=[
        _server_row(VIDEO, "a.mp4", use_soundtrack=False),
        _server_row(VIDEO, "b.mp4", use_soundtrack=True),
    ])
    inputs = _minimax_inputs(build_graph(BASE, BuildState(refs=refs), ROLES))
    assert "ref_video_audios.ref_video_audio_0" not in inputs
    assert inputs["ref_video_audios.ref_video_audio_1"] == [
        str(ROLES.injected['video'] + ROLES.injected['video_stride']), 2]


def test_standalone_audio_wiring():
    """Loaded, then cut to the generated length -- the model truncates it to that anyway,
    so every audio reference gets the same pair whether it is marked or not."""
    refs = RefSet(audios=[_server_row(AUDIO, "roar.wav")])
    built = build_graph(BASE, BuildState(refs=refs), ROLES)
    node_id, trim_id = str(ROLES.injected['audio']), str(ROLES.injected['audio'] + 1)
    assert built.graph[node_id]["class_type"] == "LoadAudio"
    assert built.graph[node_id]["inputs"]["audio"] == "roar.wav"     # key is "audio"
    assert built.graph[trim_id]["inputs"]["audio"] == [node_id, 0]
    assert _minimax_inputs(built)["ref_audios.ref_audio_0"] == [trim_id, 0]


def test_full_house_at_model_maximums():
    refs = RefSet(
        images=[_server_row(IMAGE, f"i{i}.png") for i in range(config.MAX_REF_IMAGES)],
        videos=[_server_row(VIDEO, f"v{k}.mp4") for k in range(config.MAX_REF_VIDEOS)],
        audios=[_server_row(AUDIO, f"a{j}.wav") for j in range(config.MAX_REF_AUDIOS)],
    )
    built = build_graph(BASE, BuildState(refs=refs), ROLES)
    inputs = _minimax_inputs(built)
    assert sum(k.startswith("ref_images.") for k in inputs) == config.MAX_REF_IMAGES
    assert sum(k.startswith("ref_videos.") for k in inputs) == config.MAX_REF_VIDEOS
    assert sum(k.startswith("ref_video_audios.") for k in inputs) == config.MAX_REF_VIDEOS
    assert sum(k.startswith("ref_audios.") for k in inputs) == config.MAX_REF_AUDIOS
    # Every injected node id is unique and clear of the workflow's own numbering.
    injected = set(built.graph) - set(BASE)
    assert len(injected) == len(set(injected)) and not (injected & set(BASE))


def test_every_link_points_at_a_node_that_exists():
    refs = RefSet(
        images=[_server_row(IMAGE, "i.png")],
        videos=[_server_row(VIDEO, "v.mp4")],
        audios=[_server_row(AUDIO, "a.wav")],
    )
    graph = build_graph(BASE, BuildState(refs=refs), ROLES).graph
    for node_id, node in graph.items():
        for key, value in node["inputs"].items():
            if isinstance(value, list) and len(value) == 2 and isinstance(value[0], str):
                assert value[0] in graph, f"{node_id}.{key} -> missing node {value[0]}"


def test_labels_map_node_ids_back_to_human_rows():
    refs = RefSet(images=[_server_row(IMAGE, "hero.png"), _server_row(IMAGE, "dragon.png")])
    built = build_graph(BASE, BuildState(refs=refs), ROLES)
    assert built.node_label(str(ROLES.injected['image'] + 1)) == "Reference image 2 (dragon.png)"
    assert built.node_label(ROLES.loadmodel) == "Diffusion model"


def test_tags_travel_with_the_built_graph():
    refs = RefSet(
        images=[_server_row(IMAGE, "a.png"), _server_row(IMAGE, "b.png")],
        videos=[_server_row(VIDEO, "v.mp4", use_soundtrack=True)],
        audios=[_server_row(AUDIO, "a.wav")],
    )
    built = build_graph(BASE, BuildState(refs=refs), ROLES)
    assert built.tags.order == [
        "<Picture 1>", "<Picture 2>", "<Audio 1>", "<Video 1>", "<Audio 2>",
    ]


# ---------------------------------------------------------------------------------
# Tier 0: an untouched build must be identical to the shipped workflow
# ---------------------------------------------------------------------------------

def test_default_state_reproduces_the_shipped_workflow_bar_the_intended_differences():
    """Untouched, the rebuild is the shipped workflow -- bar one place, and deliberate.

    The prompt is assembled from named sections, so node 138 carries the workflow's text
    wrapped in its section heading rather than bare. Everything else must match exactly,
    and any *new* deviation has to be added to INTENDED_DIFFERENCES with a reason rather
    than quietly slipping in.

    Compared against canonical_reference rather than the raw base: both sides get the
    same orphan sweep, so a node ComfyUI would never execute does not read as a change
    the app made.
    """
    built = canonicalise(build_graph(BASE, state_from_workflow(BASE, ROLES), ROLES).graph, ROLES)
    reference = graph_builder.canonical_reference(BASE, ROLES)

    assert set(built) == set(reference)
    for node_id in reference:
        if node_id not in graph_builder.intended_difference_ids(ROLES):
            assert built[node_id] == reference[node_id], node_id


def test_every_intended_difference_is_a_real_one():
    """A stale entry would quietly excuse a node from the comparison above."""
    built = canonicalise(build_graph(BASE, state_from_workflow(BASE, ROLES), ROLES).graph, ROLES)
    reference = graph_builder.canonical_reference(BASE, ROLES)

    for node_id, reason in graph_builder.intended_difference_ids(ROLES).items():
        assert built.get(node_id) != reference.get(node_id), f"{node_id} no longer differs"
        assert reason


def test_the_shipped_prompt_lands_in_the_detailed_description_section():
    built = build_graph(BASE, state_from_workflow(BASE, ROLES), ROLES)
    original = BASE[ROLES.promptinput]["inputs"]["value"]
    sent = built.graph[ROLES.promptinput]["inputs"]["value"]

    assert sent == prompt.combine(prompt.from_legacy(original))
    assert original in sent
    assert sent.startswith("subject_definitions:\nN/A")


def test_default_state_resolution_and_frames():
    built = build_graph(BASE, state_from_workflow(BASE, ROLES), ROLES)
    # Recovered approximately from the workflow's literal 1344x768: the nearest offered
    # ratio at the nearest megapixel step, re-quantised onto the app's own 32px grid.
    assert (built.width, built.height) == mathmirror.resolution("16:9 (Widescreen)", 1.0)
    assert built.frames == 124  # exact: the workflow's length, over the frame rate


def test_changing_a_value_shows_up_in_the_canonical_diff():
    state = state_from_workflow(BASE, ROLES)
    state.seed = 7
    built = build_graph(BASE, state, ROLES)
    seed_id = graph_builder.seed_node(ROLES)
    assert canonicalise(built.graph, ROLES)[seed_id] != \
        graph_builder.canonical_reference(BASE, ROLES)[seed_id]


# ---------------------------------------------------------------------------------
# The shipped workflow's own shape
# ---------------------------------------------------------------------------------

def test_the_preview_override_survives_and_still_feeds_the_sampler():
    """Node 141 is what makes the live sampler preview possible; it must reach the server."""
    built = build_graph(BASE, state_from_workflow(BASE, ROLES), ROLES)
    assert built.graph[ROLES.preview]["class_type"] == "ModelPreviewOverrideKJ"
    # Fed from the sigma shift, which is downstream of the Sage switch.
    assert built.graph[ROLES.preview]["inputs"]["model"] == [ROLES.shift, 0]
    # Both consumers of the model take it through the override, not around it.
    assert built.graph[ROLES.scheduler]["inputs"]["model"] == [ROLES.preview, 0]
    assert built.graph[ROLES.guider]["inputs"]["model"] == [ROLES.preview, 0]


def test_preview_override_settings_are_left_as_the_workflow_defines_them():
    inputs = build_graph(BASE, state_from_workflow(BASE, ROLES), ROLES).graph[ROLES.preview]["inputs"]
    assert inputs["preview_frames"] == 24
    assert inputs["preview_fps"] == 6
    assert inputs["tiny_vae"] == "taeh3_KJ.safetensors"
    assert inputs["suppress_default_preview"] is True


def test_a_default_build_prunes_only_what_the_workflow_switched_off():
    """The app adds no node it then drops. The one thing a default build does drop is the
    Sage pair, and only when the shipped workflow's own switch says so -- turn it on and
    every node in the workflow feeds the output."""
    state = state_from_workflow(BASE, ROLES)
    dead = set() if state.sage_attention else {ROLES.sage, ROLES.switch}
    built = build_graph(BASE, state, ROLES)
    assert {entry.split()[0] for entry in built.pruned} == dead

    state.sage_attention = True
    assert build_graph(BASE, state, ROLES).pruned == []


def test_turning_sage_off_takes_the_patch_out_of_the_graph():
    """Repointing rather than only setting the flag, because whether the flag alone is
    enough depends on which switch node the workflow carries -- ComfyUI's own evaluates its
    branches lazily, the pack's leaves the patch an ancestor of the output and runs it with
    allow_compile on. Taking it out is correct for both."""
    built = build_graph(BASE, BuildState(sage_attention=False), ROLES)
    classes = {n["class_type"] for n in built.graph.values()}

    assert "PathchSageAttentionKJ" not in classes
    assert ROLES.switch not in built.graph
    # The switch's consumer is repointed at the loader; everything downstream of it is
    # untouched, so the preview still reads from the same place it always did.
    assert built.graph[ROLES.shift]["inputs"]["model"] == [ROLES.loadmodel, 0]
    assert built.graph[ROLES.preview]["inputs"]["model"] == [ROLES.shift, 0]
    assert any("PathchSageAttentionKJ" in entry for entry in built.pruned)


def test_leaving_sage_on_keeps_the_workflows_own_wiring():
    built = build_graph(BASE, BuildState(sage_attention=True), ROLES)
    assert built.graph[ROLES.switch]["inputs"]["switch"] is True
    assert built.graph[ROLES.sage]["inputs"]["model"] == [ROLES.loadmodel, 0]


def test_the_switch_starts_where_the_workflow_left_it():
    """Read from the workflow rather than asserted against a constant: the shipped value is
    the workflow author's to change, and only a first launch consults it anyway."""
    assert (state_from_workflow(BASE, ROLES).sage_attention
            is bool(BASE[ROLES.switch]["inputs"]["switch"]))


@pytest.mark.parametrize("class_type", ["ComfySwitchNode", "Switch"])
def test_either_switch_implementation_is_driven_the_same_way(class_type):
    """ComfyUI's own logic node and the ComfyUI-ConditioningKrea2Rebalance one it replaced
    take the same three inputs, so nothing here needs to know which is present."""
    workflow = copy.deepcopy(BASE)
    workflow[ROLES.switch]["class_type"] = class_type
    roles = roles_mod.resolve(workflow)

    off = build_graph(workflow, BuildState(sage_attention=False), roles)
    assert roles.switch not in off.graph
    assert off.graph[roles.shift]["inputs"]["model"] == [roles.loadmodel, 0]

    on = build_graph(workflow, BuildState(sage_attention=True), roles)
    assert on.graph[roles.switch]["inputs"]["switch"] is True
    assert on.graph[roles.switch]["class_type"] == class_type


def test_a_frontend_only_key_is_not_submitted():
    """ComfyUI's exporter writes preview keys no node declares. Injected here rather than
    relied on in the shipped file, so the strip is tested whatever the workflow ships."""
    import copy

    base = copy.deepcopy(BASE)
    base[ROLES.vidcombine]["inputs"]["videopreview"] = {"params": {}}
    built = build_graph(base, BuildState(), ROLES)
    assert "videopreview" not in built.graph[ROLES.vidcombine]["inputs"]


def test_prune_orphans_reports_what_it_removed():
    graph = {
        "1": {"class_type": "SaveVideo", "inputs": {"video": ["2", 0]}},
        "2": {"class_type": "CreateVideo", "inputs": {}},
        "9": {"class_type": "Dangling", "inputs": {}},
    }
    removed = graph_builder.prune_orphans(graph, keep=("1",))
    assert removed == ["9 (Dangling)"]
    assert set(graph) == {"1", "2"}


def test_prune_orphans_cascades():
    """Removing a dangling node can orphan whatever fed only it."""
    graph = {
        "1": {"class_type": "SaveVideo", "inputs": {}},
        "8": {"class_type": "Feeder", "inputs": {}},
        "9": {"class_type": "Dangling", "inputs": {"x": ["8", 0]}},
    }
    removed = graph_builder.prune_orphans(graph, keep=("1",))
    assert set(graph) == {"1"}
    assert len(removed) == 2


# ---------------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------------

def test_validate_state_accepts_the_defaults():
    assert validate_state(state_from_workflow(BASE, ROLES)) == []


def test_validate_state_rejects_an_unuploaded_row():
    refs = RefSet(images=[RefRow(kind=IMAGE, local_path="C:/refs/x.png")])
    problems = validate_state(BuildState(refs=refs))
    assert any("has not been uploaded" in p for p in problems)


def test_validate_state_rejects_an_unknown_aspect_ratio():
    problems = validate_state(BuildState(aspect_ratio="5:4 (Nope)"))
    assert any("5:4 (Nope)" in p for p in problems)


def test_validate_state_rejects_an_over_long_duration():
    problems = validate_state(BuildState(duration_seconds=500.0))
    assert any("too long" in p.lower() for p in problems)


# ---------------------------------------------------------------------------------
# Submission extras: options a node reads from extra_data rather than a widget
# ---------------------------------------------------------------------------------

def test_vhs_is_told_to_write_only_the_muxed_video():
    """VHS_VideoCombine otherwise writes three files for a run with audio: a PNG of the
    first frame, the silent video, and the muxed one. Only the last is ever read."""
    built = build_graph(BASE, BuildState(), ROLES)
    extra = graph_builder.extra_data_for(built.graph)

    options = extra["extra_pnginfo"]["workflow"]["extra"]
    assert options["VHS_MetadataImage"] is False
    assert options["VHS_KeepIntermediate"] is False


def test_no_extra_data_is_sent_to_a_graph_that_reads_none():
    """extra_pnginfo is what several core nodes embed in their output's metadata, so
    populating it regardless would write this app's private options into saved files."""
    assert graph_builder.extra_data_for(
        {"1": {"class_type": "SaveVideo", "inputs": {}}}) == {}
    assert graph_builder.extra_data_for({}) == {}


def test_the_extra_options_are_a_copy_per_call():
    """They travel into a request body; a caller mutating one must not edit the constant."""
    first = graph_builder.extra_data_for(
        build_graph(BASE, BuildState(), ROLES).graph)
    first["extra_pnginfo"]["workflow"]["extra"]["VHS_MetadataImage"] = "tampered"
    second = graph_builder.extra_data_for(
        build_graph(BASE, BuildState(), ROLES).graph)
    assert second["extra_pnginfo"]["workflow"]["extra"]["VHS_MetadataImage"] is False
    assert graph_builder.VHS_EXTRA_OPTIONS["VHS_MetadataImage"] is False
