"""Headless verification commands.

These run without a QApplication, which is the point: the graph contract can be proven
against the live server before any GUI code is involved.

    python -m harmon3 --dry-run [--diff]   print the built graph / diff it against the base
    python -m harmon3 --check              validate the graph against the server's schemas
    python -m harmon3 --roles              show which node plays which role, and why
    python -m harmon3 --pose CLIP --out X  render one clip's skeleton, no server involved
"""

from __future__ import annotations

import difflib
import json
import sys
import time

from . import comfy_http, config, graph_builder, mathmirror, roles as roles_mod, validator
from .comfy_http import ComfyClient, ComfyError, ComfyUnreachable


def _load_state(workflow, settings_path=None) -> graph_builder.BuildState:
    """The saved state if there is one, otherwise the shipped workflow's own values."""
    state = graph_builder.state_from_workflow(workflow.graph, workflow.roles)

    path = settings_path or config.SETTINGS_PATH
    if path and path.is_file():
        try:
            from .settings import apply_to_state, load_settings
            apply_to_state(state, load_settings(path))
        except Exception as exc:  # settings are advisory here; never block verification
            print(f"(ignoring unreadable {path}: {exc})", file=sys.stderr)
    return state


def cmd_roles() -> int:
    """Print the resolved role contract: what bound to what, and what is missing."""
    workflow = config.load_workflow()
    graph, roles = workflow.graph, workflow.roles

    print(f"{config.WORKFLOW_PATH.name}: {len(graph)} nodes\n")
    width = max(len(spec.name) for spec in roles_mod.ROLES) + 2
    print(f"{'role':<{width}}{'node':<7}{'class':<28}purpose")
    for role, node_id, class_type in roles.describe(graph):
        print(f"{role:<{width}}{node_id:<7}{class_type:<28}"
              f"{roles_mod.ROLES_BY_NAME[role].why}")

    unbound = [spec.name for spec in roles_mod.ROLES if not roles.has(spec.name)]
    if unbound:
        print("\nOptional roles this workflow does not carry: " + ", ".join(unbound))
    if roles.keep:
        print("\nProtected from the orphan sweep (h3-keep): " + ", ".join(roles.keep))

    untagged = [
        nid for nid in sorted(graph, key=lambda n: int(n) if n.isdigit() else 0)
        if roles_mod.tag_of(graph[nid]) is None
    ]
    if untagged:
        print(f"\nUntagged, so passed through untouched: {', '.join(untagged)}")

    for warning in graph_builder.geometry_warnings(graph, roles):
        print(f"\nWARN  {warning}")
    return 0


def cmd_dry_run(show_diff: bool) -> int:
    workflow = config.load_workflow()
    base, roles = workflow.graph, workflow.roles
    state = _load_state(workflow)
    built = graph_builder.build_graph(base, state, roles)

    problems = graph_builder.validate_state(state)
    if problems:
        print("State problems:", file=sys.stderr)
        for problem in problems:
            print(f"  - {problem}", file=sys.stderr)
        print(file=sys.stderr)

    if not show_diff:
        print(comfy_http.pretty(built.graph))
        _print_summary(built)
        return 0

    before = comfy_http.pretty(
        graph_builder.canonical_reference(base, roles)).splitlines(keepends=True)
    after = comfy_http.pretty(
        graph_builder.canonicalise(built.graph, roles)).splitlines(keepends=True)
    diff = list(difflib.unified_diff(
        before, after, fromfile=f"API/{config.WORKFLOW_PATH.name}", tofile="built", n=2,
    ))

    if diff:
        sys.stdout.writelines(diff)

    reference = graph_builder.canonical_reference(base, roles)
    current = graph_builder.canonicalise(built.graph, roles)
    intended = graph_builder.intended_difference_ids(roles)
    changed = [
        node_id for node_id in set(reference) | set(current)
        if reference.get(node_id) != current.get(node_id)
    ]
    elsewhere = [node_id for node_id in changed if node_id not in intended]

    if not diff:
        print("No differences: the built graph is identical to the shipped workflow.")
    elif not elsewhere:
        print("\nOnly the intended differences, and nothing else:")
        for node_id in sorted(changed):
            print(f"  {node_id}  {intended[node_id]}")
    else:
        print(f"\nDiffers from the shipped workflow at: {', '.join(sorted(elsewhere))}")
    _print_summary(built)
    return 0


def _print_summary(built: graph_builder.BuiltGraph) -> None:
    print(f"\n{built.width} x {built.height}, {built.frames} frames "
          f"({mathmirror.true_seconds(built.frames):.2f} s at {config.FPS} fps), "
          f"{len(built.graph)} nodes", file=sys.stderr)
    if built.tags.order:
        print("References: " + "  ".join(built.tags.order), file=sys.stderr)
    if built.pruned:
        print("Not submitted (nothing consumes them): " + ", ".join(built.pruned),
              file=sys.stderr)


def cmd_check(server_url: str) -> int:
    """Tier 1: validate the built graph against the server's own node schemas."""
    workflow = config.load_workflow()
    base, roles = workflow.graph, workflow.roles
    state = _load_state(workflow)
    built = graph_builder.build_graph(base, state, roles)

    client = ComfyClient(server_url)
    try:
        info = client.object_info_many(config.classes_for(base))
    except (ComfyUnreachable, ComfyError) as exc:
        print(f"FAIL  {exc}", file=sys.stderr)
        return 2
    finally:
        client.close()

    report = validator.validate(built.graph, info, built.labels)
    for warning in report.warnings:
        print(f"WARN  {warning}")
    for warning in graph_builder.geometry_warnings(base, roles):
        print(f"WARN  {warning}")

    missing_models = validator.model_preflight(built.graph, info, roles)
    for problem in missing_models:
        print(f"FAIL  {problem}")

    for error in report.errors:
        print(f"FAIL  {error}")

    if report.ok and not missing_models:
        print(f"OK    {len(built.graph)} nodes validate against {client.base_url}")
        return 0
    return 1


def cmd_upload_test(server_url: str, path: str) -> int:
    """Tier 3: prove a file survives the upload -> /view round trip."""
    client = ComfyClient(server_url)
    try:
        digest = comfy_http.sha256_of(path)
        name = comfy_http.content_addressed_name(path, digest)
        result = client.upload(path, name, "application/octet-stream")
        print(f"uploaded as {result.reference!r} (type={result.type})")

        # /view takes the directory from its own subfolder parameter, so the filename
        # must stay bare.
        if client.file_exists(result.name, result.subfolder, "input"):
            print("OK    retrievable via /view")
            return 0
        print("FAIL  uploaded but /view could not retrieve it", file=sys.stderr)
        return 1
    except (ComfyUnreachable, ComfyError) as exc:
        print(f"FAIL  {exc}", file=sys.stderr)
        return 2
    finally:
        client.close()


def cmd_pose(source: str, out: str | None, start: int, frames: int | None) -> int:
    """Render one clip's skeleton, the way the Pose toggle would, and say what it took.

    The GUI is a poor place to judge an estimator: this puts the same code path a keypress
    away, so the model, the threshold and the drawing can be looked at directly.
    """
    from pathlib import Path

    from . import pose, settings as settings_mod

    source_path = Path(source)
    if not source_path.is_file():
        print(f"FAIL  {source_path} does not exist", file=sys.stderr)
        return 2

    config_data = settings_mod.load_settings()
    options = settings_mod.pose_settings(config_data)
    length = frames if frames else mathmirror.frames_from_seconds(
        mathmirror.clamp_duration(config_data.get("duration_seconds", config.DEFAULT_DURATION)))
    destination = Path(out) if out else (
        config.POSE_CACHE_DIR / f"{source_path.stem}_pose_{start}_{length}.mp4")

    device, why = pose.preferred_device(options.runtime)
    print(f"model  {options.model}  (kpt_thr {options.kpt_thr})")
    print(f"device {device}  -  {why}")
    for label, url, weights in pose.missing_models(options):
        print(f"fetch  {label} -> {weights}")
        pose.download(url, weights, on_progress=_download_ticker(label))
        print()

    started = time.monotonic()
    try:
        result = pose.render(source_path, start, length, options, destination,
                             on_frame=_frame_ticker())
    except pose.PoseError as exc:
        print(f"\nFAIL  {exc}", file=sys.stderr)
        return 1
    elapsed = time.monotonic() - started

    print(f"\nwrote  {result.path}")
    print(f"       {result.frames} frames, {result.width}x{result.height} @ "
          f"{result.fps:.3g} fps, audio {'yes' if result.has_audio else 'no'}")
    print(f"       {elapsed:.1f}s total, {elapsed / max(1, result.frames) * 1000:.0f} ms/frame")
    if result.held:
        print(f"       {result.held} frame(s) had no detection and held the previous pose")
    return 0


def _frame_ticker():
    def tick(done: int, total: int) -> None:
        print(f"\r  posing {done}/{total}", end="", flush=True)
    return tick


def _download_ticker(label: str):
    def tick(done: int, total: int) -> None:
        if total:
            print(f"\r  {label} {done >> 20}/{total >> 20} MB", end="", flush=True)
    return tick


def cmd_object_info_dump(server_url: str) -> int:
    client = ComfyClient(server_url)
    try:
        info = client.object_info_many(config.classes_for(config.load_workflow().graph))
    finally:
        client.close()
    print(json.dumps({k: v for k, v in info.items() if v}, indent=2)[:20000])
    return 0
