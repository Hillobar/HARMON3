"""Tests for the client-side mirror of ComfyUI's prompt validation.

The object_info fixtures below are trimmed copies of what ComfyUI 0.31.0 actually
serves, including the autogrow shapes that this validator has to expand.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from harmon3 import config, roles as roles_mod, validator      # noqa: E402


def _roles_for(graph: dict) -> roles_mod.NodeRoles:
    """Bind whatever roles a fragment of a graph carries, ignoring the required ones.

    resolve() deliberately refuses an incomplete workflow; these tests hand model_preflight
    one or two nodes on purpose, to check it copes with a role that is simply absent.
    """
    single = {tag: nid for nid, node in graph.items()
              if (tag := roles_mod.tag_of(node)) and not roles_mod.ROLES_BY_NAME[tag].multi}
    return roles_mod.NodeRoles(single=single, multi={}, keep=(),
                               injected=roles_mod.injected_bases(graph))

OBJECT_INFO = {
    "MiniMaxH3ReferenceToVideo": {
        "input": {
            "required": {
                "clip": ["CLIP", {}],
                "vae": ["VAE", {}],
                "audio_vae": ["VAE", {}],
                "prompt": ["STRING", {"multiline": True}],
                "width": ["INT", {"default": 1344, "min": 32, "max": 16384, "step": 32}],
                "height": ["INT", {"default": 768, "min": 32, "max": 16384, "step": 32}],
                "length": ["INT", {"default": 124, "min": 5, "max": 3600, "step": 17}],
                "ref_image_size": ["COMBO", {"options": ["match", "max"], "default": "match"}],
            },
            "optional": {
                "ref_images": ["COMFY_AUTOGROW_V3", {"template": {
                    "input": {"required": {"ref_image": ["IMAGE", {}]}},
                    "prefix": "ref_image_", "min": 0, "max": 9}}],
                "ref_videos": ["COMFY_AUTOGROW_V3", {"template": {
                    "input": {"required": {"ref_video": ["IMAGE", {}]}},
                    "prefix": "ref_video_", "min": 0, "max": 3}}],
                "ref_video_audios": ["COMFY_AUTOGROW_V3", {"template": {
                    "input": {"required": {"ref_video_audio": ["AUDIO", {}]}},
                    "prefix": "ref_video_audio_", "min": 0, "max": 3}}],
                "ref_audios": ["COMFY_AUTOGROW_V3", {"template": {
                    "input": {"required": {"ref_audio": ["AUDIO", {}]}},
                    "prefix": "ref_audio_", "min": 0, "max": 3}}],
            },
        },
        "output": ["CONDITIONING", "LATENT"],
    },
    "ComfyMathExpression": {
        "input": {"required": {
            "expression": ["STRING", {"multiline": True}],
            "values": ["COMFY_AUTOGROW_V3", {"template": {
                "input": {"required": {"value": ["FLOAT,INT,BOOLEAN", {}]}},
                "names": ["a", "b", "c"], "min": 1}}],
        }},
        "output": ["FLOAT", "INT", "BOOLEAN"],
    },
    "LoadImage": {
        "input": {"required": {"image": [["already_there.png"], {"image_upload": True}]}},
        "output": ["IMAGE", "MASK"],
    },
    "LoadAudio": {
        "input": {"required": {
            "audio": ["COMBO", {"options": ["b.wav"], "audio_upload": True}]}},
        "output": ["AUDIO"],
    },
    "LoadVideo": {
        "input": {"required": {
            "file": ["COMBO", {"options": ["a.mp4"], "video_upload": True}]}},
        "output": ["VIDEO"],
    },
    "GetVideoComponents": {
        "input": {"required": {"video": ["VIDEO", {}]}},
        "output": ["IMAGE", "AUDIO", "FLOAT", "INT"],
    },
    "PrimitiveFloat": {
        "input": {"required": {"value": ["FLOAT", {}]}},
        "output": ["FLOAT"],
    },
    "UNETLoader": {
        "input": {"required": {
            "unet_name": [["present.safetensors"], {}],
            "weight_dtype": ["COMBO", {"options": ["default"]}]}},
        "output": ["MODEL"],
    },
    "PreviewAny": {
        "input": {"required": {"source": ["*", {}]}},
        "output": ["STRING"],
    },
}


def _minimal_graph(**overrides):
    graph = {
        "1": {"class_type": "LoadImage", "inputs": {"image": "already_there.png"}},
        "2": {"class_type": "PrimitiveFloat", "inputs": {"value": 5.0}},
        "3": {"class_type": "ComfyMathExpression",
              "inputs": {"expression": "a", "values.a": ["2", 0]}},
    }
    graph.update(overrides)
    return graph


def test_valid_graph_reports_nothing():
    report = validator.validate(_minimal_graph(), OBJECT_INFO)
    assert report.ok, report.errors


def test_autogrow_expands_prefixed_keys():
    """ref_images.ref_image_0..8 must be accepted; _9 must not."""
    graph = _minimal_graph(**{
        "4": {"class_type": "MiniMaxH3ReferenceToVideo", "inputs": {
            "clip": ["9", 0], "vae": ["9", 0], "audio_vae": ["9", 0],
            "prompt": "x", "width": 864, "height": 480, "length": 124,
            "ref_image_size": "match",
            "ref_images.ref_image_0": ["1", 0],
            "ref_images.ref_image_8": ["1", 0],
        }},
    })
    report = validator.validate(graph, OBJECT_INFO)
    # The dangling clip/vae source is a separate, expected complaint.
    assert not [e for e in report.errors if "ref_image" in e], report.errors

    graph["4"]["inputs"]["ref_images.ref_image_9"] = ["1", 0]
    report = validator.validate(graph, OBJECT_INFO)
    assert any("ref_image_9" in e for e in report.errors)


def test_autogrow_min_makes_the_first_slot_required():
    """ComfyMathExpression declares min=1, so values.a is mandatory."""
    graph = _minimal_graph()
    del graph["3"]["inputs"]["values.a"]
    report = validator.validate(graph, OBJECT_INFO)
    assert any("values.a" in e and "missing" in e for e in report.errors)


def test_autogrow_optional_slots_are_not_required():
    report = validator.validate(_minimal_graph(), OBJECT_INFO)
    assert not any("values.b" in e for e in report.errors)


def test_link_to_missing_node_is_caught():
    graph = _minimal_graph()
    graph["3"]["inputs"]["values.a"] = ["999", 0]
    report = validator.validate(graph, OBJECT_INFO)
    assert any("missing node 999" in e for e in report.errors)


def test_out_of_range_output_slot_is_caught():
    graph = _minimal_graph()
    graph["3"]["inputs"]["values.a"] = ["2", 3]   # PrimitiveFloat has one output
    report = validator.validate(graph, OBJECT_INFO)
    assert any("slot 3" in e for e in report.errors)


def test_type_mismatch_is_caught():
    graph = _minimal_graph(**{
        "4": {"class_type": "LoadVideo", "inputs": {"file": "a.mp4"}},
        "5": {"class_type": "GetVideoComponents", "inputs": {"video": ["4", 0]}},
        "6": {"class_type": "PreviewAny", "inputs": {"source": ["5", 0]}},
    })
    assert validator.validate(graph, OBJECT_INFO).ok

    # fps (slot 2, FLOAT) fed where a VIDEO is expected
    graph["5"]["inputs"]["video"] = ["5", 2]
    report = validator.validate(graph, OBJECT_INFO)
    assert any("expects VIDEO" in e for e in report.errors)


def test_wildcard_input_accepts_anything():
    graph = _minimal_graph(**{
        "4": {"class_type": "PreviewAny", "inputs": {"source": ["2", 0]}},
    })
    assert validator.validate(graph, OBJECT_INFO).ok


def test_multi_type_input_accepts_any_member():
    """values.a is FLOAT,INT,BOOLEAN and is fed a FLOAT."""
    assert validator.validate(_minimal_graph(), OBJECT_INFO).ok


def test_a_match_type_is_not_compared_against_a_real_one():
    """ComfySwitchNode declares its branches and its output as COMFY_MATCHTYPE_V3, resolved
    from whatever is connected. /object_info reports the placeholder, so comparing against
    it fails a graph that is entirely correct -- in both directions, since the node both
    accepts and produces one."""
    info = dict(OBJECT_INFO)
    info["ComfySwitchNode"] = {
        "input": {"required": {
            "switch": ["BOOLEAN", {}],
            "on_false": [validator.MATCHTYPE, {"lazy": True}],
            "on_true": [validator.MATCHTYPE, {"lazy": True}],
        }},
        "output": [validator.MATCHTYPE],
    }
    graph = _minimal_graph(**{
        "4": {"class_type": "UNETLoader",
              "inputs": {"unet_name": "present.safetensors", "weight_dtype": "default"}},
        "5": {"class_type": "ComfySwitchNode",
              "inputs": {"switch": False, "on_false": ["4", 0], "on_true": ["4", 0]}},
        "6": {"class_type": "PreviewAny", "inputs": {"source": ["5", 0]}},
    })
    assert validator.validate(graph, info).ok


def test_numeric_bounds_are_enforced():
    graph = _minimal_graph(**{
        "4": {"class_type": "MiniMaxH3ReferenceToVideo", "inputs": {
            "clip": ["4", 0], "vae": ["4", 0], "audio_vae": ["4", 0],
            "prompt": "x", "width": 864, "height": 480, "length": 9999,
            "ref_image_size": "match"}},
    })
    report = validator.validate(graph, OBJECT_INFO)
    assert any("above the maximum 3600" in e for e in report.errors)


def test_combo_membership_is_enforced_for_normal_combos():
    graph = _minimal_graph(**{
        "4": {"class_type": "MiniMaxH3ReferenceToVideo", "inputs": {
            "clip": ["4", 0], "vae": ["4", 0], "audio_vae": ["4", 0],
            "prompt": "x", "width": 864, "height": 480, "length": 124,
            "ref_image_size": "enormous"}},
    })
    report = validator.validate(graph, OBJECT_INFO)
    assert any("'ref_image_size'" in e for e in report.errors)


def test_loader_filenames_bypass_combo_membership():
    """Files uploaded into a subfolder never appear in the server's listing.

    LoadImage/LoadAudio/LoadVideo each declare a custom validator naming their filename
    argument, which disables ComfyUI's membership check, so enforcing it here would
    reject perfectly valid uploads.
    """
    graph = {
        "1": {"class_type": "LoadImage", "inputs": {"image": "harmon3/uploaded_abc.png"}},
        "2": {"class_type": "LoadAudio", "inputs": {"audio": "harmon3/uploaded_def.wav"}},
        "3": {"class_type": "LoadVideo", "inputs": {"file": "harmon3/uploaded_ghi.mp4"}},
    }
    assert validator.validate(graph, OBJECT_INFO).ok


def test_exemption_list_matches_the_classes_we_inject():
    assert config.COMBO_CHECK_EXEMPT == {
        ("LoadImage", "image"), ("LoadAudio", "audio"), ("LoadVideo", "file"),
        ("VHS_LoadVideo", "video"), ("VHS_LoadImages", "directory"),
    }


def test_the_vhs_loaders_accept_an_upload_subfolder():
    """Their COMBO lists only the input root, but VALIDATE_INPUTS resolves subfolders.

    Checked against a live server: without the exemption, every clip uploaded into
    harmon3/ reads as invalid here while ComfyUI accepts it.
    """
    graph = {
        "1": {"class_type": "VHS_LoadVideo", "inputs": {"video": "harmon3/uploaded.mp4"}},
        "2": {"class_type": "VHS_LoadImages", "inputs": {"directory": "harmon3_frames_ab"}},
    }
    info = {
        "VHS_LoadVideo": {"input": {"required": {"video": [["only_root.mp4"], {}]}}},
        "VHS_LoadImages": {"input": {"required": {"directory": [["harmon3"], {}]}}},
    }
    assert validator.validate(graph, info).ok


def test_unknown_class_is_an_error_and_unfetched_class_is_a_warning():
    graph = {"1": {"class_type": "NotARealNode", "inputs": {}}}
    report = validator.validate(graph, {"NotARealNode": None})
    assert any("does not have a node" in e for e in report.errors)

    report = validator.validate(graph, {})
    assert report.ok and report.warnings


def test_unknown_input_name_is_caught():
    graph = _minimal_graph()
    graph["1"]["inputs"]["nonsense"] = 1
    report = validator.validate(graph, OBJECT_INFO)
    assert any("unknown input 'nonsense'" in e for e in report.errors)


def test_errors_are_indexed_by_node_for_row_highlighting():
    graph = _minimal_graph()
    graph["1"]["inputs"]["nonsense"] = 1
    report = validator.validate(graph, OBJECT_INFO, {"1": "Reference image 1 (x.png)"})
    assert "1" in report.by_node
    assert report.by_node["1"][0].startswith("Reference image 1 (x.png)")


def test_model_preflight_flags_a_missing_checkpoint():
    """The class type comes from whatever node holds the role, not from a fixed table."""
    graph = {"7": {"class_type": "UNETLoader",
                   "_meta": {"title": "h3-loadmodel"},
                   "inputs": {"unet_name": "absent.safetensors"}}}
    roles = _roles_for(graph)

    problems = validator.model_preflight(graph, OBJECT_INFO, roles)
    assert problems and "absent.safetensors" in problems[0]

    graph["7"]["inputs"]["unet_name"] = "present.safetensors"
    assert validator.model_preflight(graph, OBJECT_INFO, roles) == []


def test_model_preflight_ignores_a_role_the_workflow_does_not_carry():
    """A workflow with no separate audio VAE must not report it as a missing model."""
    graph = {"7": {"class_type": "UNETLoader",
                   "_meta": {"title": "h3-loadmodel"},
                   "inputs": {"unet_name": "present.safetensors"}}}
    assert validator.model_preflight(graph, OBJECT_INFO, _roles_for(graph)) == []
