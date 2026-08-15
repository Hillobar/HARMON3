"""Constants, paths and base-graph node IDs.

No Qt, no network. Everything here is derived from the workflow JSON in ./API and
from the ComfyUI node definitions it targets (ComfyUI 0.31.0).
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

# --------------------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------------------

APP_DIR = Path(__file__).resolve().parent.parent
HOME = Path(os.environ.get("HARMON3_HOME") or APP_DIR)

WORKFLOW_PATH = APP_DIR / "API" / "video_minimax_h3_r2v_api.json"
#: The output-format spec the prompt sections are written to. Read only by the editor's
#: Guide button, and only when it is clicked: the format helpers are code, so the app is
#: fully usable with this file absent.
GUIDE_PATH = APP_DIR / "API" / "VIDEO_PROMPT_WRITING_GUIDE_ref_en.md"
SETTINGS_PATH = HOME / "settings.json"
UI_STATE_PATH = HOME / "ui_state.ini"
RUNS_DIR = HOME / "runs"
RUNS_JSONL = RUNS_DIR / "runs.jsonl"
VIDEO_CACHE_DIR = RUNS_DIR / "videos"
#: Rendered pose clips, keyed by source content + section + settings. Never evicted, the
#: same as the downloaded results beside them.
POSE_CACHE_DIR = RUNS_DIR / "pose"
#: Reference images rescaled before upload, keyed by content and target size.
SCALE_CACHE_DIR = RUNS_DIR / "scaled"
#: Where "what the model is about to be given" is written, on demand. Beside the app
#: rather than under runs/, because it is something to go and look at rather than
#: something the app keeps.
BUNDLE_DIR = HOME / "reference_bundle"
#: ONNX weights for the pose estimator, downloaded once on first use.
POSE_MODELS_DIR = HOME / "models" / "pose"
#: Where scenes live unless the user points them somewhere else in Settings.
SCENES_DIR = HOME / "scenes"


def resolve_scenes_dir(configured) -> Path:
    """The folder scenes are actually read from and written to."""
    text = str(configured or "").strip()
    return Path(text).expanduser() if text else SCENES_DIR


def describe_scenes_dir(configured) -> str:
    resolved = resolve_scenes_dir(configured)
    return f"{resolved}" + ("" if str(configured or "").strip() else "   (default)")

# --------------------------------------------------------------------------------------
# Node identity
#
# Nodes are found by the role tag in their ComfyUI title (h3-loadmodel, h3-prompt, ...),
# not by number. See harmon3/roles.py, which owns the whole contract.
# --------------------------------------------------------------------------------------

#: Keys ComfyUI's own frontend writes into an API export that no node declares. The server
#: ignores them; this app strips them so the graph it sends is only inputs that mean
#: something, and so the validator's "unknown input" stays a real signal.
FRONTEND_ONLY_INPUTS = {
    ("SaveVideo", "video-preview"),
    ("VHS_VideoCombine", "videopreview"),
}

#: Classes the builder adds to the graph, which are therefore not in the shipped workflow
#: and would otherwise never have their schema fetched.
INJECTED_CLASSES = ("LoadImage", "LoadAudio", "TrimAudioDuration", "VHS_LoadVideo")

# --------------------------------------------------------------------------------------
# Model limits (MiniMaxH3ReferenceToVideo, verified via /object_info)
# --------------------------------------------------------------------------------------

MAX_REF_IMAGES = 9
MAX_REF_VIDEOS = 3
MAX_REF_AUDIOS = 3

FPS = 24
FRAME_MOD = 17
FRAME_REM = 5

MIN_FRAMES = 5
MAX_FRAMES = 3600
#: Largest frame count reachable through node 131's round-up-to (17k+5) expression.
MAX_ALIGNED_FRAMES = MAX_FRAMES - ((MAX_FRAMES - FRAME_REM) % FRAME_MOD)  # 3592

MIN_DURATION = 0.2                                  # -> 5 frames
MAX_DURATION = MAX_ALIGNED_FRAMES / FPS             # 149.666...

MIN_DIMENSION = 32
MAX_DIMENSION = 16384

MIN_MEGAPIXELS = 0.1
MAX_MEGAPIXELS = 16.0
STEP_MEGAPIXELS = 0.1

#: The grid the computed width and height are snapped to. Fixed rather than offered:
#: MiniMax H3's canvas is built in multiples of 32, so any other value only produces
#: dimensions the model then has to round anyway.
#:
#: The workflow no longer carries a ResolutionSelector, so this app is now the only place
#: the quantisation happens: mathmirror computes width and height and the builder writes
#: them into the reference node as literals.
MULTIPLE = 32

#: The range the resolution maths is written against and tested across, even though the
#: app itself only ever sends MULTIPLE.
MIN_MULTIPLE = 8
MAX_MULTIPLE = 128
STEP_MULTIPLE = 4

#: BasicScheduler.steps, on the h3-scheduler node.
MIN_STEPS = 1
MAX_STEPS = 10000
DEFAULT_STEPS = 20

#: KSamplerSelect.sampler_name (h3-sampler) and BasicScheduler.scheduler (h3-scheduler):
#: which solver walks the sigmas, and how those sigmas are spaced.
#:
#: Both lists are what stock ComfyUI ships, and both are only a *starting point*: a server
#: with sampler packs installed offers more, and the real list replaces these as soon as
#: /object_info has been read. So neither is treated as a whitelist -- a name that is not
#: here is still sent, and the validator is what says whether the server knows it.
SAMPLERS = (
    "euler", "euler_cfg_pp", "euler_ancestral", "euler_ancestral_cfg_pp", "heun",
    "heunpp2", "exp_heun_2_x0", "exp_heun_2_x0_sde", "dpm_2", "dpm_2_ancestral", "lms",
    "dpm_fast", "dpm_adaptive", "dpmpp_2s_ancestral", "dpmpp_2s_ancestral_cfg_pp",
    "dpmpp_sde", "dpmpp_sde_gpu", "dpmpp_2m", "dpmpp_2m_cfg_pp", "dpmpp_2m_sde",
    "dpmpp_2m_sde_gpu", "dpmpp_2m_sde_heun", "dpmpp_2m_sde_heun_gpu", "dpmpp_3m_sde",
    "dpmpp_3m_sde_gpu", "ddpm", "lcm", "ipndm", "ipndm_v", "deis", "res_multistep",
    "res_multistep_cfg_pp", "res_multistep_ancestral", "res_multistep_ancestral_cfg_pp",
    "gradient_estimation", "gradient_estimation_cfg_pp", "er_sde", "seeds_2", "seeds_3",
    "sa_solver", "sa_solver_pece", "ddim", "uni_pc", "uni_pc_bh2",
)
#: What the shipped workflow uses, and what MiniMax H3 was tuned around.
DEFAULT_SAMPLER = "res_multistep"

SCHEDULERS = (
    "simple", "sgm_uniform", "karras", "exponential", "ddim_uniform", "beta",
    "normal", "linear_quadratic", "kl_optimal",
)
DEFAULT_SCHEDULER = "simple"

#: MiniMaxH3ProgressiveSampler.schedule: comma-separated "scale:end_percent" stages. The
#: sampler runs the early steps on a smaller latent grid and upscales the x0 estimate
#: between stages, so most of the run is cheap and only the last stage pays full
#: resolution. Free text rather than a fixed list -- the node parses it, and the useful
#: schedules are a continuum rather than a menu.
#:
#: Only reaches the graph when the workflow has a sampler that declares it; see
#: graph_builder.schedule_node.
#:
#: One full-resolution stage by default, which is what the sampler would do without any
#: staging at all. Staging is a speed/quality trade the user should reach for knowingly:
#: a default that already runs the first half of every generation at half grid changes
#: what comes out, and someone who never opens this field would have no way of knowing.
DEFAULT_SCHEDULE = "1.0:1.0"
#: The A/B baseline: one stage, at full resolution, i.e. what SamplerCustomAdvanced does.
BASELINE_SCHEDULE = "1.0:1.0"

#: MiniMaxH3ProgressiveSampler.upscale_method: how the x0 estimate is resampled when a
#: stage hands over to the next one at a larger latent grid. Only meaningful for a
#: multi-stage schedule -- at "1.0:1.0" nothing is ever upscaled.
UPSCALE_METHODS = ("bicubic", "bilinear", "area", "nearest-exact", "bislerp")
DEFAULT_UPSCALE_METHOD = "bicubic"

#: MiniMaxH3SigmaShift.shift_video: where the sigma schedule's weight sits. Higher spends
#: more of the run at high noise, which moves more; the node's own range is 0.01..100.
#: shift_audio is deliberately left alone -- it is the workflow's to set.
MIN_SHIFT = 0.01
MAX_SHIFT = 100.0
STEP_SHIFT = 0.1
DEFAULT_SHIFT_VIDEO = 3.0

#: MiniMaxH3ReferenceToVideo.ref_image_size, in the node's own order. "match" scales each
#: reference down to the generation's pixel area; "max" uses the reference pipeline's
#: 2048px short edge for better identity at a real cost in speed, because reference tokens
#: ride through every sampling step.
REF_IMAGE_SIZES = ("match", "max")
#: "max" by default, because each image now carries its own scale. Under "match"
#: the node caps every reference at the generation's pixel area anyway, which
#: overrides most of what a per-image slider was set to; under "max" the cap is
#: 2048 on the short edge and the slider decides everything below it.
DEFAULT_REF_IMAGE_SIZE = "max"

#: Reference videos are truncated to the generated length and need >= 5 frames.
MIN_REF_VIDEO_FRAMES = 5

#: How reference video is loaded. Both sizes are 0 -- "leave it alone" -- because
#: MiniMaxH3ReferenceToVideo scales every reference to its own canvas before encoding, so
#: resizing on the way in only changes how much is carried to get there. The workflow's
#: own node asked for a 1536px height, which turned a 640x480 clip into roughly 2736x1536:
#: 6.3 GB for 124 frames against 0.5 GB at native size, for a picture the node then scales
#: back down.
#:
#: `format` supplies VHS's downscale ratio and does *not* touch the frame rate. AnimateDiff
#: rounds width and height to a multiple of 8 even with both custom sizes at 0.
REF_VIDEO_WIDTH = 0
REF_VIDEO_HEIGHT = 0
REF_VIDEO_FORMAT = "AnimateDiff"

# --------------------------------------------------------------------------------------
# Pose estimation
#
# A reference with Pose ticked is replaced, on the way to the server, by a skeleton
# rendition of itself. The estimator runs here rather than in the graph: one local ONNX
# pass over just the frames that will be sent.
# --------------------------------------------------------------------------------------

#: model id -> (filename, download url, keypoint layout). ViTPose is the default because
#: it is the strongest of the practical options on occluded and partial bodies -- it holds
#: the state of the art on OCHuman -- and the cost of that only shows up once per run.
#: The RTMPose entries are the same pipeline with cheaper weights.
POSE_MODELS = {
    "vitpose-l": (
        "vitpose-l-coco.onnx",
        "https://huggingface.co/JunkyByte/easy_ViTPose/resolve/main/onnx/coco/vitpose-l-coco.onnx",
        "coco17",
    ),
    "vitpose-b": (
        "vitpose-b-coco.onnx",
        "https://huggingface.co/JunkyByte/easy_ViTPose/resolve/main/onnx/coco/vitpose-b-coco.onnx",
        "coco17",
    ),
    "vitpose-l-wholebody": (
        "vitpose-l-wholebody.onnx",
        "https://huggingface.co/JunkyByte/easy_ViTPose/resolve/main/onnx/wholebody/vitpose-l-wholebody.onnx",
        "coco133",
    ),
}
DEFAULT_POSE_MODEL = "vitpose-l"

#: The person detector every top-down estimator needs in front of it. YOLOX-x, the
#: largest published: it runs once every POSE_DETECT_EVERY frames, so its cost is
#: rounding error next to the estimator's, and a missed person is a blank frame.
#:
#: Handed to rtmlib as a URL rather than a path, because these are zipped OpenMMLab SDK
#: bundles and rtmlib already knows how to unpack them and how to fall back to its
#: HuggingFace mirror. It caches them under ~/.cache/rtmlib, not in POSE_MODELS_DIR.
#: The hash in the filename is per-model -- there is no yolox_l build published, and
#: pairing the "l" name with another model's hash gets a 404 from both hosts.
POSE_DETECTOR = (
    "yolox_x.onnx",
    "https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/"
    "yolox_x_8xb8-300e_humanart-a39d44ed.zip",
)

#: ViTPose's own input size. Not configurable: the ONNX graphs are exported at it.
POSE_INPUT_SIZE = (192, 256)
POSE_DETECTOR_INPUT_SIZE = (640, 640)

#: How often the detector re-runs. Between runs the box is carried forward from the last
#: pose, which is both faster and steadier than detecting every frame on a moving subject.
POSE_DETECT_EVERY = 10

#: Keypoints below this score are not drawn. Low enough to keep a partly occluded limb,
#: high enough that a hallucinated one does not flail around behind the subject.
DEFAULT_POSE_KPT_THR = 0.4

#: "auto" tries CUDA and falls back to CPU; the other two are the same names ONNX Runtime
#: uses, forced.
POSE_RUNTIMES = ("auto", "cuda", "cpu")
DEFAULT_POSE_RUNTIME = "auto"

#: How the keypoints are drawn. A property of the rendering, not of the estimator, so
#: every model offers all of them: the models differ in which points they find, the styles
#: in what is done with them. The first two are substitutions into rtmlib's own table -- see
#: pose.skeleton_for -- and the third is painted from scratch; see harmon3.posefigure.
#: Appended rather than ordered, so a stored setting and a combo index both still mean
#: what they meant.
POSE_STYLES = ("openpose", "openpose-torso", "figure")
DEFAULT_POSE_STYLE = "openpose"

#: Encoder settings for the rendered clip. A skeleton on black is nearly all flat colour,
#: so this is generous for what it has to carry.
POSE_VIDEO_CRF = 18

# --------------------------------------------------------------------------------------
# Server / upload
# --------------------------------------------------------------------------------------

DEFAULT_SERVER_URL = "http://127.0.0.1:8188"
UPLOAD_SUBFOLDER = "harmon3"

def classes_for(graph: dict) -> tuple:
    """Node classes the client-side validator fetches from /object_info.

    Derived from the workflow rather than listed, so a node the user adds is checked
    against the server's own schema like every other. A static list meant an unlisted
    class degraded to "schema not fetched" and only the server caught its errors.
    """
    present = {
        node.get("class_type")
        for node in (graph or {}).values()
        if isinstance(node, dict) and node.get("class_type")
    }
    return tuple(sorted(present.union(INJECTED_CLASSES)))

#: (class_type, input_name) pairs whose COMBO membership is NOT enforced server-side,
#: because the class declares a custom validator naming that argument, which disables
#: execution.py's value_not_in_list check. Uploading into a subfolder relies on this.
COMBO_CHECK_EXEMPT = {
    ("LoadImage", "image"),
    ("LoadAudio", "audio"),
    ("LoadVideo", "file"),
    # VideoHelperSuite declares VALIDATE_INPUTS(s, video) / (s, directory) and resolves
    # through folder_paths.get_annotated_filepath, which handles subfolders. Its COMBO
    # only lists the input *root*, so without these every upload into harmon3/ would read
    # as invalid here while the server accepts it happily.
    ("VHS_LoadVideo", "video"),
    ("VHS_LoadImages", "directory"),
}

#: (role, input_name) pairs whose value must name a file present on the server. Used by
#: the startup model preflight; the class type comes from whatever node holds the role.
MODEL_INPUTS = (
    ("loadmodel", "unet_name"),
    ("loadvideovae", "vae_name"),
    ("loadaudiovae", "vae_name"),
    ("loadclip", "clip_name"),
)

# --------------------------------------------------------------------------------------
# Defaults (mirror the shipped workflow so a fresh launch reproduces it exactly)
# --------------------------------------------------------------------------------------

DEFAULT_ASPECT_RATIO = "16:9 (Widescreen)"
DEFAULT_MEGAPIXELS = 0.4
DEFAULT_DURATION = 5.0
DEFAULT_SEED = 157368968253448

#: Sage Attention: on unless the workflow says otherwise. It is a speed patch applied to
#: the model, and the workflow ships it switched on.
DEFAULT_SAGE_ATTENTION = True

#: Where the output node files a finished video inside ComfyUI's own output folder. Only a
#: fallback: the real default is read from whatever the workflow's output node was saved
#: with, which is what a user editing the workflow would expect to keep.
DEFAULT_FILENAME_PREFIX = "videos/h3"

MAX_SEED = 0xFFFFFFFFFFFFFF  # 2**56-1, matches ComfyUI's noise_seed range


def clean_filename_prefix(prefix: str) -> str:
    """A prefix that can only ever name something inside ComfyUI's output folder.

    ComfyUI resolves this against its own output directory and rejects an escape itself,
    but it does so several minutes into a run. Anchoring it here means a leading slash or
    a ``..`` is simply not sent, rather than costing a generation to find out. Backslashes
    are folded to forward slashes because this is a server-side path and the server may
    well not be Windows.

    An empty result is the signal to leave the workflow's own value alone.
    """
    parts = [p.strip() for p in str(prefix or "").replace("\\", "/").split("/")]
    kept = [p for p in parts if p not in ("", ".", "..")]
    return "/".join(kept)


@dataclass(frozen=True)
class Workflow:
    """The base graph and the roles resolved against it. Always travel together."""

    graph: dict
    roles: "object"


def load_workflow(path: Path | None = None) -> Workflow:
    """Load the base API workflow and bind its role tags.

    Raises a clear error if the JSON is the wrong export format or does not satisfy the
    role contract, rather than letting a KeyError surface later from inside the builder.
    """
    from . import roles as roles_mod

    path = path or WORKFLOW_PATH
    if not path.is_file():
        raise FileNotFoundError(f"Workflow JSON not found: {path}")

    with path.open("r", encoding="utf-8") as fh:
        graph = json.load(fh)

    if not isinstance(graph, dict) or "nodes" in graph:
        raise ValueError(
            f"{path.name} is not a ComfyUI API-format workflow. Re-export it with "
            "'Export (API)' rather than 'Save'."
        )

    try:
        resolved = roles_mod.resolve(graph)
    except roles_mod.WorkflowContractError as exc:
        raise ValueError(
            f"{path.name} does not satisfy the HARMON3 role contract:\n{exc}\n"
            "Tag the nodes in ComfyUI by putting the role at the start of the node's "
            "title, e.g. 'h3-promptinput'. See PLAN-workflow-role-contract.md."
        ) from None

    return Workflow(graph=graph, roles=resolved)


def defaults_from_workflow(graph: dict, roles) -> dict:
    """Read the shipped workflow's widget values as the app's first-launch defaults.

    Width, height and length are literals on the reference node now that the workflow
    carries no ResolutionSelector or frame-count expression, so aspect ratio and
    megapixels are recovered *approximately* -- the nearest offered ratio, and the area
    rounded to the megapixel step. Duration is exact, being frames over the frame rate.
    """
    from . import graph_builder, mathmirror

    ref = graph[roles.reference]["inputs"]
    width = int(ref.get("width") or 0)
    height = int(ref.get("height") or 0)
    length = int(ref.get("length") or 0)

    scheduler_inputs = graph[roles.scheduler]["inputs"]
    switch_id = roles.optional("switch")
    staging = graph_builder.schedule_node(roles)
    upscaler = graph_builder.upscale_node(roles)
    shift = graph_builder.shift_node(roles)

    return {
        "aspect_ratio": mathmirror.nearest_aspect_ratio(width, height),
        "megapixels": mathmirror.megapixels_for(width, height),
        "duration_seconds": (length / FPS) if length else DEFAULT_DURATION,
        "seed": int(graph[graph_builder.seed_node(roles)]["inputs"].get(
            "noise_seed", DEFAULT_SEED)),
        "steps": int(scheduler_inputs.get("steps", DEFAULT_STEPS)),
        "sampler_name": str(
            graph[roles.sampler]["inputs"].get("sampler_name", DEFAULT_SAMPLER)),
        "scheduler": str(scheduler_inputs.get("scheduler", DEFAULT_SCHEDULER)),
        "schedule": str(
            graph[staging]["inputs"].get("schedule", DEFAULT_SCHEDULE) if staging
            else DEFAULT_SCHEDULE),
        "upscale_method": str(
            graph[upscaler]["inputs"].get("upscale_method", DEFAULT_UPSCALE_METHOD)
            if upscaler else DEFAULT_UPSCALE_METHOD),
        "shift_video": float(
            graph[shift]["inputs"].get("shift_video", DEFAULT_SHIFT_VIDEO) if shift
            else DEFAULT_SHIFT_VIDEO),
        "ref_image_size": str(ref.get("ref_image_size", DEFAULT_REF_IMAGE_SIZE)),
        "prompt_text": graph[roles.promptinput]["inputs"].get("value", ""),
        "sage_attention": (
            bool(graph[switch_id]["inputs"].get("switch", True)) if switch_id
            else DEFAULT_SAGE_ATTENTION
        ),
        "ref_images": [
            graph[nid]["inputs"]["image"]
            for nid in roles.many("refimage")
            if "image" in graph[nid].get("inputs", {})
        ],
        # VHS_LoadVideo names its file "video"; the plain LoadVideo uses "file".
        "ref_videos": [
            graph[nid]["inputs"].get("video") or graph[nid]["inputs"].get("file")
            for nid in roles.many("refvideo")
            if graph[nid]["inputs"].get("video") or graph[nid]["inputs"].get("file")
        ],
        # Whether each baked video's soundtrack is actually wired. It defaults on for a
        # reference the user adds, but a first launch has to reproduce the workflow: an
        # unwanted soundtrack emits an extra <Audio j> and renumbers every tag after it.
        "ref_video_soundtracks": [
            f"ref_video_audios.ref_video_audio_{k}" in ref
            for k in range(len(roles.many("refvideo")))
        ],
    }
