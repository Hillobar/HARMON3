"""Client-side mirror of ComfyUI's prompt validation.

Catching a malformed graph here rather than at POST time means the error can be shown
against the widget that caused it, before any file is uploaded, and it keeps a typo from
looking like a server problem. This is a pre-flight check, not a replacement: the server
still validates everything it receives.

No Qt, no network (it consumes an already-fetched /object_info snapshot).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from . import config

AUTOGROW = "COMFY_AUTOGROW_V3"
WILDCARD = "*"

#: A V3 node whose input and output types are decided by what is connected to them, rather
#: than declared. `/object_info` reports the placeholder itself, so a static comparison
#: against it is meaningless -- `ComfySwitchNode` reads as "expects COMFY_MATCHTYPE_V3 but
#: UNETLoader produces MODEL" on a graph that is entirely correct. The server resolves the
#: template and enforces that its members agree, which is the check that can actually be
#: made; treating it as a wildcard here defers to that rather than inventing a worse one.
MATCHTYPE = "COMFY_MATCHTYPE_V3"


@dataclass
class ValidationReport:
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    #: node id -> messages, for highlighting the offending row in the UI
    by_node: dict[str, list[str]] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return not self.errors

    def add_error(self, node_id: str, message: str, label: str | None = None) -> None:
        text = f"{label or f'node {node_id}'}: {message}"
        self.errors.append(text)
        self.by_node.setdefault(node_id, []).append(text)

    def add_warning(self, message: str) -> None:
        self.warnings.append(message)


def _expand_autogrow(group_name: str, spec_config: dict) -> tuple[list[str], list[str], str]:
    """Return (all valid flattened keys, required keys, member type) for an autogrow group.

    Two template shapes exist. ``TemplatePrefix`` numbers slots ``{prefix}{i}`` up to
    ``max``; ``TemplateNames`` uses an explicit name list. In both, the first ``min``
    slots are required. Flattened keys are ``{group}.{slot}``.
    """
    template = spec_config.get("template") or {}
    minimum = int(template.get("min", 0))

    if "names" in template:
        slots = list(template["names"])
    else:
        prefix = template.get("prefix", "")
        slots = [f"{prefix}{i}" for i in range(int(template.get("max", 0)))]

    member_inputs = (template.get("input") or {}).get("required") or {}
    member_type = ""
    for spec in member_inputs.values():
        member_type = spec[0] if isinstance(spec, list) and spec else ""
        break

    keys = [f"{group_name}.{slot}" for slot in slots]
    return keys, keys[:minimum], member_type


def _input_specs(schema: dict) -> tuple[dict, dict]:
    """Return (required, optional) input specs with autogrow groups already flattened."""
    raw = schema.get("input") or {}
    required: dict[str, list] = {}
    optional: dict[str, list] = {}

    for bucket, target in (("required", required), ("optional", optional)):
        for name, spec in (raw.get(bucket) or {}).items():
            if isinstance(spec, list) and spec and spec[0] == AUTOGROW:
                spec_config = spec[1] if len(spec) > 1 else {}
                keys, required_keys, member_type = _expand_autogrow(name, spec_config)
                for key in keys:
                    entry = [member_type, {"__autogrow__": True}]
                    if key in required_keys:
                        required[key] = entry
                    else:
                        optional[key] = entry
            else:
                target[name] = spec
                optional.update(_format_dependent_inputs(spec))

    return required, optional


def _format_dependent_inputs(spec) -> dict:
    """Widgets a combo option brings with it, as ``{name: spec}``.

    VideoHelperSuite's VHS_VideoCombine declares ``pix_fmt``, ``crf`` and friends inside
    its ``format`` combo -- one set per format -- rather than as inputs of its own, so
    /object_info never lists them at the top level even though the node accepts them.
    Without this every workflow that picks a real format reads as four unknown inputs.

    Every format's widgets are accepted rather than only the selected format's: the point
    here is to stop a false "unknown input", and which format declares what is the node's
    business to enforce.
    """
    settings = spec[1] if isinstance(spec, list) and len(spec) > 1 else None
    formats = settings.get("formats") if isinstance(settings, dict) else None
    if not isinstance(formats, dict):
        return {}

    found = {}
    for widgets in formats.values():
        for widget in widgets or []:
            # [name, type] or [name, type, settings]; anything else is not a widget.
            if isinstance(widget, list) and len(widget) >= 2 and isinstance(widget[0], str):
                found[widget[0]] = list(widget[1:])
    return found


def _type_of(spec) -> str:
    """The declared type of an input spec, normalised to a string."""
    if not isinstance(spec, list) or not spec:
        return ""
    declared = spec[0]
    # V1 combos declare their options list in place of a type name.
    return "COMBO" if isinstance(declared, list) else str(declared)


def _combo_options(spec) -> list | None:
    if not isinstance(spec, list) or not spec:
        return None
    if isinstance(spec[0], list):
        return list(spec[0])
    if spec[0] == "COMBO":
        opts = (spec[1] if len(spec) > 1 else {}).get("options")
        return list(opts) if isinstance(opts, list) else None
    return None


def combo_options(object_info: dict, class_type: str, input_name: str) -> list:
    """The values a server accepts for one combo input, or [] if it has not said.

    The UI uses this to offer exactly what the connected server has rather than a list
    baked in at build time; the same option lists the validator checks against.
    """
    schema = (object_info or {}).get(class_type)
    if not isinstance(schema, dict):
        return []
    required, optional = _input_specs(schema)
    spec = required.get(input_name, optional.get(input_name))
    return _combo_options(spec) or []


def _types_compatible(produced: str, expected: str) -> bool:
    if not produced or not expected:
        return True
    if WILDCARD in (produced, expected) or MATCHTYPE in (produced, expected):
        return True
    produced_set = {t.strip() for t in produced.split(",")}
    expected_set = {t.strip() for t in expected.split(",")}
    return bool(produced_set & expected_set)


def validate(graph: dict, object_info: dict[str, dict | None],
             labels: dict[str, str] | None = None) -> ValidationReport:
    """Check a built graph against the server's node schemas."""
    labels = labels or {}
    report = ValidationReport()

    for node_id, node in graph.items():
        class_type = node.get("class_type")
        label = labels.get(node_id)
        schema = object_info.get(class_type)

        if class_type not in object_info:
            report.add_warning(f"Could not check {class_type} ({node_id}): schema not fetched")
            continue
        if schema is None:
            report.add_error(
                node_id,
                f"the ComfyUI server does not have a node called {class_type!r} "
                "(a custom node pack may be missing)",
                label,
            )
            continue

        required, optional = _input_specs(schema)
        inputs = node.get("inputs") or {}

        for name in required:
            if name not in inputs:
                report.add_error(node_id, f"required input {name!r} is missing", label)

        for name, value in inputs.items():
            spec = required.get(name) or optional.get(name)
            if spec is None:
                report.add_error(node_id, f"unknown input {name!r} for {class_type}", label)
                continue
            if _is_link(value):
                _check_link(report, graph, object_info, node_id, label, name, value, spec)
            else:
                _check_literal(report, node_id, label, class_type, name, value, spec)

    return report


def _is_link(value) -> bool:
    return (
        isinstance(value, list)
        and len(value) == 2
        and isinstance(value[0], str)
        and isinstance(value[1], int)
    )


def _check_link(report, graph, object_info, node_id, label, name, value, spec) -> None:
    source_id, slot = value

    source = graph.get(source_id)
    if source is None:
        report.add_error(node_id, f"input {name!r} is wired to missing node {source_id}", label)
        return

    source_schema = object_info.get(source.get("class_type"))
    if not source_schema:
        return  # unknown source class already reported against that node

    outputs = source_schema.get("output") or []
    if slot >= len(outputs):
        report.add_error(
            node_id,
            f"input {name!r} reads output slot {slot} of {source['class_type']}, "
            f"which only has {len(outputs)}",
            label,
        )
        return

    produced = outputs[slot]
    produced = ",".join(produced) if isinstance(produced, list) else str(produced)
    expected = _type_of(spec)
    if not _types_compatible(produced, expected):
        report.add_error(
            node_id,
            f"input {name!r} expects {expected} but {source['class_type']} "
            f"slot {slot} produces {produced}",
            label,
        )


def _check_literal(report, node_id, label, class_type, name, value, spec) -> None:
    declared = _type_of(spec)
    options = _combo_options(spec)
    settings = spec[1] if isinstance(spec, list) and len(spec) > 1 else {}

    if options is not None:
        # Some classes declare a custom validator for their filename argument, which
        # disables ComfyUI's membership check server-side. Enforcing it here would
        # wrongly reject files uploaded into a subfolder, which never appear in the
        # non-recursive listing this options array is built from.
        if (class_type, name) in config.COMBO_CHECK_EXEMPT:
            return
        if value not in options:
            preview = ", ".join(str(o) for o in options[:6])
            suffix = ", ..." if len(options) > 6 else ""
            report.add_error(
                node_id,
                f"{name!r} is {value!r}, which the server does not offer "
                f"(available: {preview}{suffix})",
                label,
            )
        return

    if declared in ("INT", "FLOAT") and isinstance(value, (int, float)) and not isinstance(value, bool):
        minimum = settings.get("min")
        maximum = settings.get("max")
        if minimum is not None and value < minimum:
            report.add_error(node_id, f"{name!r} is {value}, below the minimum {minimum}", label)
        if maximum is not None and value > maximum:
            report.add_error(node_id, f"{name!r} is {value}, above the maximum {maximum}", label)


def model_preflight(graph: dict, object_info: dict[str, dict | None], roles) -> list[str]:
    """Confirm the workflow's model files exist on the server.

    Run at startup so a missing checkpoint is a banner rather than a rejection after the
    user has configured a whole job.
    """
    problems: list[str] = []

    for role, input_name in config.MODEL_INPUTS:
        node_id = roles.optional(role)
        node = graph.get(node_id) if node_id else None
        if not node:
            continue
        class_type = node.get("class_type")
        schema = object_info.get(class_type)
        if not schema:
            continue

        value = (node.get("inputs") or {}).get(input_name)
        spec = ((schema.get("input") or {}).get("required") or {}).get(input_name)
        options = _combo_options(spec)
        if options is None or value is None:
            continue

        if value not in options:
            problems.append(f"{class_type}: model file not found on the server - {value}")

    return problems
