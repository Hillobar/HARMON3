"""Which node in the workflow plays which part, resolved from its title.

The app used to find its nodes by hardcoded numeric id, which froze the workflow: any edit
the user made in ComfyUI was either invisible here or silently undone. Instead each node
the app touches carries a role tag in its ComfyUI title -- ``h3-loadmodel``,
``h3-promptinput``, ``h3-vidcombine`` -- which survives an API export as ``_meta.title``.
Numbering then belongs to ComfyUI and the user, and this module is the only place that
knows what the app needs to find.

The tag is the first whitespace-delimited word of the title, lowercased, when it starts
with ``h3-``; the rest of the title stays a readable label, so
``"h3-promptinput Input Text"`` tags the node and still reads like a name. Tagging is
strict: an untagged node never binds to a role, because guessing by class type picks the
wrong one of two VAELoaders.

Some parts can be played by more than one node, and which is present changes how the app
behaves rather than whether it works: a ``SamplerCustomAdvanced`` takes its noise by link
from a ``RandomNoise``, while a ``MiniMaxH3ProgressiveSampler`` makes its own from a
widget and carries a staging schedule too. ``SEED_ROLES``/``SCHEDULE_ROLES`` say where
those values go, and ``REQUIRED_GROUPS`` says that at least one seed-carrier must exist --
a constraint no per-role ``required`` flag can express.

``h3-keep`` is not a role. It marks a node as a pruning root, which is what lets a user add
a branch of their own -- a second output, a preview -- without the orphan sweep removing it.

No Qt, no network.
"""

from __future__ import annotations

from dataclasses import dataclass

TAG_PREFIX = "h3-"

#: Tag that protects a node from the orphan sweep instead of naming a role.
KEEP_TAG = "keep"


class WorkflowContractError(ValueError):
    """The workflow does not satisfy the role contract. Carries every problem at once."""

    def __init__(self, problems: list[str]):
        self.problems = list(problems)
        super().__init__("\n".join(f"  - {p}" for p in self.problems))


@dataclass(frozen=True)
class RoleSpec:
    """One part the app needs some node to play."""

    name: str
    #: Class types acceptable for this role. The first is what the shipped workflow uses.
    class_types: tuple[str, ...]
    required: bool
    #: True if several nodes may share the tag (reference seeds), False if exactly one may.
    multi: bool
    why: str


#: Every role, and why the app cares. Roles the app only labels are optional: losing a
#: caption is not worth refusing to start over.
ROLES: tuple[RoleSpec, ...] = (
    RoleSpec("promptinput", ("PrimitiveStringMultiline",), True, False,
             "the assembled prompt text is written to its 'value'"),
    RoleSpec("reference", ("MiniMaxH3ReferenceToVideo",), True, False,
             "width, height, length and ref_image_size are written here, and every "
             "reference loader is wired into its autogrow inputs"),
    RoleSpec("loadmodel", ("UNETLoader",), True, False,
             "model preflight, and the node Sage Attention is bypassed onto"),
    RoleSpec("loadclip", ("CLIPLoader",), True, False, "model preflight"),
    RoleSpec("loadvideovae", ("VAELoader",), True, False, "model preflight"),
    RoleSpec("loadaudiovae", ("VAELoader",), True, False, "model preflight"),
    RoleSpec("sampler", ("KSamplerSelect",), True, False, "'sampler_name' is written here"),
    RoleSpec("scheduler", ("BasicScheduler",), True, False,
             "'steps' and 'scheduler' are written here"),
    RoleSpec("vidcombine", ("VHS_VideoCombine", "SaveVideo", "SaveWEBM", "SaveAnimatedWEBP"),
             True, False,
             "the output node: the run's result is read from it, and it is the root the "
             "orphan sweep keeps"),

    # Where the seed goes, and which of them is present decides how sampling works. A
    # SamplerCustomAdvanced takes its noise by link from a RandomNoise; the progressive
    # sampler makes its own from a noise_seed widget and carries the staging schedule too.
    # One of the two seed-carriers must exist -- see SEED_ROLES.
    RoleSpec("noise", ("RandomNoise",), False, False,
             "the seed is written to its 'noise_seed'"),
    RoleSpec("progressivesampler", ("MiniMaxH3ProgressiveSampler",), False, False,
             "the seed and the staging schedule are written here, and it is what the "
             "Schedule parameter drives"),
    RoleSpec("sampleradvanced", ("SamplerCustomAdvanced",), False, False,
             "progress captions; it takes its noise by link rather than by widget"),
    RoleSpec("guider", ("BasicGuider",), False, False, "error labels"),
    RoleSpec("shift", ("MiniMaxH3SigmaShift",), False, False, "error labels"),
    RoleSpec("imagedecode", ("VAEDecode",), False, False, "progress captions"),
    RoleSpec("audiodecode", ("VAEDecodeAudio",), False, False, "progress captions"),

    # Optional features. Absent simply means the app runs without them.
    RoleSpec("preview", ("ModelPreviewOverrideKJ",), False, False,
             "sends the live sampler preview; without it the app runs without previews"),
    RoleSpec("sage", ("PathchSageAttentionKJ",), False, False,
             "the Sage Attention patch; without it the Settings toggle does nothing"),
    # ComfyUI's own logic node first; "Switch" is the ComfyUI-ConditioningKrea2Rebalance
    # one the workflow used to carry. Both take the same three inputs, so the builder does
    # not care which is present -- but the core node evaluates its branches lazily and the
    # pack's does not. See graph_builder._apply_sage.
    RoleSpec("switch", ("ComfySwitchNode", "Switch"), False, False,
             "chooses between the patched and unpatched model"),

    # Reference loaders baked into the workflow. Always pruned by the builder; their
    # filenames seed the reference list on a first launch.
    RoleSpec("refimage", ("LoadImage",), False, True, "seeds the reference image list"),
    RoleSpec("refvideo", ("VHS_LoadVideo", "LoadVideo"), False, True,
             "seeds the reference video list"),
)

ROLES_BY_NAME = {spec.name: spec for spec in ROLES}

#: Roles that can carry the seed, best first. Whichever the workflow has is where the seed
#: is written; one of them must be present, which is what REQUIRED_GROUPS enforces.
SEED_ROLES = ("progressivesampler", "noise")

#: Roles that accept a staging schedule. A workflow with none simply has no Schedule
#: parameter -- the panel hides the field rather than writing a key no node declares.
SCHEDULE_ROLES = ("progressivesampler",)

#: Roles that accept an upscale method for the hand-over between staging stages. Listed
#: separately from SCHEDULE_ROLES even though one node currently declares both, because
#: they are two independent facts about a sampler rather than one.
UPSCALE_ROLES = ("progressivesampler",)

#: Roles that accept a sigma shift.
SHIFT_ROLES = ("shift",)

#: Sets of roles where at least one must be bound, with what they are collectively for.
REQUIRED_GROUPS = (
    (SEED_ROLES, "somewhere to write the seed"),
)


def tag_of(node: dict) -> str | None:
    """The role tag a node carries, or None. ``"h3-prompt Input Text"`` -> ``"prompt"``."""
    title = ((node or {}).get("_meta") or {}).get("title")
    if not isinstance(title, str):
        return None
    first = title.strip().split(" ")[0].lower()
    if not first.startswith(TAG_PREFIX):
        return None
    return first[len(TAG_PREFIX):] or None


@dataclass(frozen=True)
class NodeRoles:
    """Resolved role -> node id. Single-bind roles are also attributes: ``roles.promptinput``."""

    single: dict
    multi: dict
    keep: tuple
    #: Node-id blocks the builder injects into, chosen to clear every id already in use.
    injected: dict

    def __getattr__(self, name: str) -> str:
        # Only reached for names not found normally, so the dataclass fields still win.
        # Going through the instance dict rather than self.single matters: during copy or
        # unpickling the fields are not set yet, and self.single would recurse into here.
        single = object.__getattribute__(self, "__dict__").get("single") or {}
        try:
            return single[name]
        except KeyError:
            raise AttributeError(
                f"{name!r} is not a resolved single-bind role "
                f"(have: {', '.join(sorted(single))})"
            ) from None

    def optional(self, name: str) -> str | None:
        """The node id bound to ``name``, or None if the workflow does not have one."""
        return self.single.get(name)

    def many(self, name: str) -> tuple:
        return self.multi.get(name, ())

    def has(self, name: str) -> bool:
        return name in self.single or bool(self.multi.get(name))

    @property
    def ids(self) -> set:
        found = set(self.single.values()) | set(self.keep)
        for group in self.multi.values():
            found.update(group)
        return found

    def describe(self, graph: dict) -> list:
        """(role, node id, class type) for every bound role, for the diagnostics readout."""
        rows = []
        for spec in ROLES:
            if spec.multi:
                bound = self.multi.get(spec.name, ())
            else:
                bound = (self.single[spec.name],) if spec.name in self.single else ()
            for node_id in bound:
                rows.append((spec.name, node_id, (graph.get(node_id) or {}).get("class_type", "?")))
        return rows


def resolve(graph: dict) -> NodeRoles:
    """Bind every role to a node, or raise with the full list of what is wrong.

    Every problem is collected rather than raising on the first, because a workflow that
    has drifted usually has several and fixing them one launch at a time is miserable.
    """
    problems: list[str] = []
    tagged: dict[str, list[str]] = {}
    keep: list[str] = []

    for node_id in sorted(graph, key=_id_sort_key):
        tag = tag_of(graph[node_id])
        if tag is None:
            continue
        if tag == KEEP_TAG:
            keep.append(node_id)
            continue
        if tag not in ROLES_BY_NAME:
            problems.append(
                f"node {node_id} is tagged '{TAG_PREFIX}{tag}', which is not a role "
                f"(known: {', '.join(sorted(ROLES_BY_NAME))})"
            )
            continue
        tagged.setdefault(tag, []).append(node_id)

    single: dict[str, str] = {}
    multi: dict[str, tuple] = {}

    for spec in ROLES:
        found = tagged.get(spec.name, [])

        if not found:
            if spec.required:
                problems.append(
                    f"no node is tagged '{TAG_PREFIX}{spec.name}' -- "
                    f"{spec.why}. Expected a {' or '.join(spec.class_types)}."
                )
            continue

        # Checked before the class, because when a tag is on two nodes the duplication is
        # the story: the class mismatch it usually causes is only a symptom of it.
        if not spec.multi and len(found) > 1:
            problems.append(
                f"'{TAG_PREFIX}{spec.name}' is on {len(found)} nodes ({', '.join(found)}); "
                "exactly one node may carry it"
            )
            continue

        wrong = [
            f"node {nid} ({graph[nid].get('class_type', '?')})"
            for nid in found
            if graph[nid].get("class_type") not in spec.class_types
        ]
        if wrong:
            problems.append(
                f"'{TAG_PREFIX}{spec.name}' must be a {' or '.join(spec.class_types)}, "
                f"but {', '.join(wrong)} is not"
            )
            continue

        if spec.multi:
            multi[spec.name] = tuple(found)
        else:
            single[spec.name] = found[0]

    for group, purpose in REQUIRED_GROUPS:
        if not any(name in single or multi.get(name) for name in group):
            problems.append(
                f"the workflow needs {purpose}, so one of "
                f"{', '.join(TAG_PREFIX + name for name in group)} must be present"
            )

    if problems:
        raise WorkflowContractError(problems)

    return NodeRoles(single=single, multi=multi, keep=tuple(keep),
                     injected=injected_bases(graph))


def first_bound(roles: "NodeRoles", names) -> str | None:
    """The node id of the first of ``names`` this workflow actually carries."""
    for name in names:
        node_id = roles.optional(name)
        if node_id:
            return node_id
    return None


def _id_sort_key(node_id: str):
    """Numeric ids in numeric order, everything else after them alphabetically."""
    return (0, int(node_id), "") if str(node_id).isdigit() else (1, 0, str(node_id))


def injected_bases(graph: dict) -> dict:
    """Starting ids for the loaders the builder injects, clear of everything in ``graph``.

    Fixed blocks rather than one running counter, so the ids a given reference gets stay
    the same from run to run: ComfyUI's per-node execution cache is keyed on them, and a
    dry-run dump stays diffable. The blocks used to be constants (200/220/260) that a
    large enough workflow would collide with silently; deriving them from the graph makes
    that impossible.
    """
    used = [int(nid) for nid in graph if str(nid).isdigit()]
    start = (max(used, default=0) // 100 + 1) * 100
    return {"image": start, "video": start + 20, "video_stride": 10,
            "audio": start + 60, "audio_stride": 10}
