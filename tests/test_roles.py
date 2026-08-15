"""Tests for the role contract: which node plays which part, and what a bad one says.

The point of the contract is that the workflow can be edited in ComfyUI -- renumbered,
extended, rewired -- without the app noticing. These tests are what says so.
"""

from __future__ import annotations

import copy
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from harmon3 import config, graph_builder, roles as roles_mod     # noqa: E402
from harmon3.graph_builder import BuildState, build_graph          # noqa: E402
from harmon3.roles import WorkflowContractError                    # noqa: E402

WORKFLOW = config.load_workflow()
BASE, ROLES = WORKFLOW.graph, WORKFLOW.roles


def _base() -> dict:
    """A private copy: BASE is module-level and every test here mutates its own."""
    return copy.deepcopy(BASE)


def _node_for(graph: dict, role: str) -> dict:
    return graph[ROLES.optional(role)]


def _problems(graph: dict) -> list[str]:
    with pytest.raises(WorkflowContractError) as excinfo:
        roles_mod.resolve(graph)
    return excinfo.value.problems


# ---------------------------------------------------------------------------------
# Tag parsing
# ---------------------------------------------------------------------------------

@pytest.mark.parametrize("title, expected", [
    ("h3-promptinput", "promptinput"),
    ("h3-promptinput Input Text (Prompt)", "promptinput"),
    ("H3-PROMPTINPUT", "promptinput"),    # ComfyUI does not lowercase titles for you
    ("  h3-promptinput  ", "promptinput"),
    ("h3-keep", "keep"),
    ("Input Text (Prompt)", None),
    ("promptinput", None),
    ("nh3-promptinput", None),            # the prefix must start the title
    ("h3-", None),                        # a bare prefix names no role
    ("", None),
])
def test_the_tag_is_the_first_word_of_the_title(title, expected):
    assert roles_mod.tag_of({"_meta": {"title": title}}) == expected


@pytest.mark.parametrize("node", [{}, {"_meta": {}}, {"_meta": {"title": None}}, None])
def test_a_node_with_no_usable_title_carries_no_tag(node):
    assert roles_mod.tag_of(node) is None


# ---------------------------------------------------------------------------------
# The shipped workflow
# ---------------------------------------------------------------------------------

def test_the_shipped_workflow_satisfies_the_contract():
    for spec in roles_mod.ROLES:
        if spec.required:
            assert ROLES.has(spec.name), spec.name


def test_every_bound_role_names_a_node_of_an_accepted_class():
    for role, node_id, class_type in ROLES.describe(BASE):
        assert class_type in roles_mod.ROLES_BY_NAME[role].class_types
        assert node_id in BASE


def test_the_two_vae_loaders_are_told_apart_by_tag_alone():
    """They are the same class; nothing but the tag can distinguish them."""
    assert BASE[ROLES.loadvideovae]["class_type"] == "VAELoader"
    assert BASE[ROLES.loadaudiovae]["class_type"] == "VAELoader"
    assert ROLES.loadvideovae != ROLES.loadaudiovae


def test_every_role_in_the_registry_has_a_reason():
    for spec in roles_mod.ROLES:
        assert spec.why and spec.class_types


# ---------------------------------------------------------------------------------
# Renumbering, which is the whole point
# ---------------------------------------------------------------------------------

def test_a_renumbered_workflow_builds_the_same_graph():
    """Every id shifted by 1000. The built graph must be identical after canonicalising,
    because nothing the app does may depend on what a node is called."""
    shifted = {str(int(nid) + 1000): copy.deepcopy(node) for nid, node in BASE.items()}
    for node in shifted.values():
        for key, value in node["inputs"].items():
            if isinstance(value, list) and len(value) == 2 and isinstance(value[0], str):
                node["inputs"][key] = [str(int(value[0]) + 1000), value[1]]

    roles = roles_mod.resolve(shifted)
    state = BuildState(seed=11, steps=7, duration_seconds=3.0)

    original = graph_builder.canonicalise(build_graph(BASE, state, ROLES).graph, ROLES)
    renumbered = graph_builder.canonicalise(build_graph(shifted, state, roles).graph, roles)

    assert len(original) == len(renumbered)
    by_class = lambda g: sorted(  # noqa: E731
        (n["class_type"], sorted(k for k in n["inputs"])) for n in g.values())
    assert by_class(original) == by_class(renumbered)


# ---------------------------------------------------------------------------------
# Nodes the user adds
# ---------------------------------------------------------------------------------

def test_an_untagged_node_in_the_model_chain_survives_the_build():
    """The case this whole change is for: a LoRA loader dropped between the model loader
    and everything downstream. The app knows nothing about it and must not disturb it."""
    graph = _base()
    graph["500"] = {
        "inputs": {"lora_name": "style.safetensors", "strength_model": 0.8,
                   "model": [ROLES.loadmodel, 0]},
        "class_type": "LoraLoaderModelOnly",
        "_meta": {"title": "my lora"},
    }
    # Rewire the Sage patch to take the model through the LoRA.
    graph[ROLES.sage]["inputs"]["model"] = ["500", 0]
    graph[ROLES.switch]["inputs"]["on_false"] = ["500", 0]

    built = build_graph(graph, BuildState(), roles_mod.resolve(graph))
    assert built.graph["500"]["inputs"]["lora_name"] == "style.safetensors"
    assert "500" not in [entry.split()[0] for entry in built.pruned]


def test_an_untagged_side_branch_is_pruned_and_said_so():
    graph = _base()
    graph["500"] = {
        "inputs": {"images": [ROLES.imagedecode, 0]},
        "class_type": "PreviewImage",
        "_meta": {"title": "my preview"},
    }
    built = build_graph(graph, BuildState(), roles_mod.resolve(graph))

    assert "500" not in built.graph
    assert any(entry.startswith("500 ") for entry in built.pruned)


def test_a_keep_tagged_side_branch_survives_the_orphan_sweep():
    """h3-keep is the escape hatch that makes a second output branch possible."""
    graph = _base()
    graph["500"] = {
        "inputs": {"images": [ROLES.imagedecode, 0]},
        "class_type": "PreviewImage",
        "_meta": {"title": "h3-keep my preview"},
    }
    roles = roles_mod.resolve(graph)
    assert roles.keep == ("500",)

    built = build_graph(graph, BuildState(), roles)
    assert "500" in built.graph
    assert not built.pruned


def test_keep_protects_a_whole_branch_not_just_the_tagged_node():
    graph = _base()
    graph["500"] = {"inputs": {"images": [ROLES.imagedecode, 0], "upscale_method": "nearest-exact"},
                    "class_type": "ImageScaleBy", "_meta": {"title": "upscale"}}
    graph["501"] = {"inputs": {"images": ["500", 0]},
                    "class_type": "PreviewImage", "_meta": {"title": "h3-keep"}}

    built = build_graph(graph, BuildState(), roles_mod.resolve(graph))
    assert "500" in built.graph and "501" in built.graph


# ---------------------------------------------------------------------------------
# What a broken workflow says
# ---------------------------------------------------------------------------------

def test_a_missing_required_role_is_named_along_with_what_it_is_for():
    graph = _base()
    graph[ROLES.promptinput]["_meta"]["title"] = "Input Text"

    problems = _problems(graph)
    assert len(problems) == 1
    assert "h3-promptinput" in problems[0]
    assert "PrimitiveStringMultiline" in problems[0]


def test_a_duplicated_tag_is_refused_and_both_nodes_named():
    """A supplied workflow once had h3-sampler on both the selector and the sampler.
    Guessing which was meant would silently write a parameter into the wrong node."""
    graph = _base()
    graph[ROLES.loadaudiovae]["_meta"]["title"] = "h3-loadvideovae"

    # Two problems, not one: retagging a node also vacates the role it used to hold.
    problems = _problems(graph)
    duplicate = [p for p in problems if "is on 2 nodes" in p]
    assert len(duplicate) == 1
    assert ROLES.loadvideovae in duplicate[0] and ROLES.loadaudiovae in duplicate[0]


def test_a_duplicate_is_reported_as_a_duplicate_not_as_a_class_mismatch():
    """When one tag lands on two nodes of different classes, the duplication is the story
    and the class mismatch is only its symptom."""
    graph = _base()
    graph[ROLES.scheduler]["_meta"]["title"] = "h3-sampler"

    problems = _problems(graph)
    assert any("is on 2 nodes" in p for p in problems)
    assert not any("must be a KSamplerSelect" in p for p in problems)


def test_a_role_on_the_wrong_class_is_refused():
    graph = _base()
    graph[ROLES.promptinput]["class_type"] = "PrimitiveInt"

    problems = _problems(graph)
    assert len(problems) == 1
    assert "PrimitiveStringMultiline" in problems[0] and "PrimitiveInt" in problems[0]


def test_an_unknown_tag_is_refused_with_the_list_of_real_ones():
    graph = _base()
    graph[ROLES.promptinput]["_meta"]["title"] = "h3-promt"          # a plausible typo

    problems = _problems(graph)
    assert any("h3-promt" in p and "prompt" in p for p in problems)


def test_every_problem_is_reported_at_once():
    """Fixing a drifted workflow one launch at a time is miserable."""
    graph = _base()
    graph[ROLES.promptinput]["_meta"]["title"] = "Input Text"
    graph[ROLES.sampler]["_meta"]["title"] = "sampler"
    graph[ROLES.scheduler]["class_type"] = "KSamplerSelect"

    problems = _problems(graph)
    assert len(problems) == 3


def test_load_workflow_explains_a_contract_failure_rather_than_raising_a_key_error(tmp_path):
    import json

    graph = _base()
    graph[ROLES.promptinput]["_meta"]["title"] = "Input Text"
    path = tmp_path / "broken_api.json"
    path.write_text(json.dumps(graph), encoding="utf-8")

    with pytest.raises(ValueError) as excinfo:
        config.load_workflow(path)
    message = str(excinfo.value)
    assert "role contract" in message and "h3-promptinput" in message


def test_a_ui_format_export_is_still_rejected_first(tmp_path):
    import json

    path = tmp_path / "ui_api.json"
    path.write_text(json.dumps({"nodes": [], "links": []}), encoding="utf-8")

    with pytest.raises(ValueError, match="Export \\(API\\)"):
        config.load_workflow(path)


# ---------------------------------------------------------------------------------
# Injected id blocks
# ---------------------------------------------------------------------------------

def test_injected_blocks_clear_every_id_in_use():
    for extra in ([], ["199"], ["250"], ["905"], [str(n) for n in range(200, 260)]):
        graph = _base()
        for nid in extra:
            graph[nid] = {"inputs": {}, "class_type": "PreviewImage", "_meta": {"title": "x"}}
        bases = roles_mod.injected_bases(graph)
        highest = max(int(nid) for nid in graph)
        assert all(base > highest for base in
                   (bases["image"], bases["video"], bases["audio"]))


def test_injected_blocks_do_not_overlap_each_other_at_model_maximums():
    bases = ROLES.injected
    images = {bases["image"] + i for i in range(config.MAX_REF_IMAGES)}
    videos = {bases["video"] + bases["video_stride"] * k for k in range(config.MAX_REF_VIDEOS)}
    audios = set()
    for j in range(config.MAX_REF_AUDIOS):
        start = bases["audio"] + bases["audio_stride"] * j
        audios.update({start, start + 1})

    assert not (images & videos) and not (videos & audios) and not (images & audios)


def test_a_non_numeric_node_id_does_not_break_the_id_arithmetic():
    """ComfyUI numbers nodes, but nothing in the format promises it."""
    graph = _base()
    graph["subgraph:a1"] = {"inputs": {}, "class_type": "PreviewImage",
                            "_meta": {"title": "x"}}
    bases = roles_mod.injected_bases(graph)
    assert bases["image"] > max(int(n) for n in graph if n.isdigit())


# ---------------------------------------------------------------------------------
# Optional roles
# ---------------------------------------------------------------------------------

def test_a_workflow_without_the_sage_nodes_simply_has_no_toggle():
    graph = _base()
    switch_id, sage_id = ROLES.switch, ROLES.sage
    # Wire the switch's consumers straight to the loader, then drop both nodes.
    for node in graph.values():
        for key, value in node["inputs"].items():
            if isinstance(value, list) and value[0] == switch_id:
                node["inputs"][key] = [ROLES.loadmodel, 0]
    del graph[switch_id], graph[sage_id]

    roles = roles_mod.resolve(graph)
    assert roles.optional("switch") is None

    for enabled in (True, False):
        built = build_graph(graph, BuildState(sage_attention=enabled), roles)
        assert "PathchSageAttentionKJ" not in {n["class_type"] for n in built.graph.values()}


def test_a_workflow_without_the_preview_node_still_builds():
    graph = _base()
    preview_id = ROLES.preview
    source = graph[preview_id]["inputs"]["model"]
    for node in graph.values():
        for key, value in node["inputs"].items():
            if isinstance(value, list) and value[0] == preview_id:
                node["inputs"][key] = source
    del graph[preview_id]

    roles = roles_mod.resolve(graph)
    assert roles.optional("preview") is None
    assert build_graph(graph, BuildState(), roles).graph


def test_asking_for_an_unbound_role_as_an_attribute_says_which_are_bound():
    with pytest.raises(AttributeError, match="not a resolved single-bind role"):
        ROLES.refimage          # multi-bind, so never an attribute
    with pytest.raises(AttributeError):
        ROLES.nonsense
