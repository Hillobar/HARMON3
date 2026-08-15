"""Blocking HTTP client for the ComfyUI REST API.

Every method blocks, so instances are confined to the job worker thread. No Qt here.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse, urlunparse

import requests

from . import config

log = logging.getLogger(__name__)

DEFAULT_TIMEOUT = 15
UPLOAD_TIMEOUT = 300
DOWNLOAD_TIMEOUT = 300

_SAFE_STEM = re.compile(r"[^A-Za-z0-9._-]+")


class ComfyError(Exception):
    """A request reached the server but it refused."""

    def __init__(self, message: str, *, status: int | None = None,
                 node_errors: dict | None = None, payload: dict | None = None):
        super().__init__(message)
        self.status = status
        self.node_errors = node_errors or {}
        self.payload = payload or {}


class ComfyUnreachable(Exception):
    """The server could not be contacted at all."""


@dataclass
class UploadResult:
    """What ComfyUI reports after storing an uploaded file."""

    name: str
    subfolder: str
    type: str

    @property
    def reference(self) -> str:
        """The value a LoadImage/LoadAudio/LoadVideo widget needs."""
        return f"{self.subfolder}/{self.name}" if self.subfolder else self.name


@dataclass
class OutputRef:
    """A file ComfyUI produced, as reported in an `executed` message or /history."""

    filename: str
    subfolder: str = ""
    type: str = "output"

    @classmethod
    def from_dict(cls, data: dict) -> "OutputRef":
        return cls(
            filename=data["filename"],
            subfolder=data.get("subfolder", "") or "",
            type=data.get("type", "output") or "output",
        )


def sha256_of(path: str | Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def content_addressed_name(path: str | Path, digest: str) -> str:
    """A stable, collision-proof server filename for a local file.

    Embedding the hash makes uploads idempotent when combined with ``overwrite=true``:
    the same bytes always land on the same name, and different bytes never share one. It
    also sidesteps ComfyUI's " (1)" duplicate-rename behaviour, which would otherwise
    silently desynchronise the widget value from the stored file.
    """
    p = Path(path)
    stem = _SAFE_STEM.sub("_", p.stem)[:60].strip("_") or "ref"
    suffix = _SAFE_STEM.sub("", p.suffix)[:12]
    return f"{stem}_{digest[:12]}{suffix}"


def normalise_base_url(url: str) -> str:
    url = (url or "").strip() or config.DEFAULT_SERVER_URL
    if "://" not in url:
        url = "http://" + url
    parsed = urlparse(url)
    return urlunparse((parsed.scheme, parsed.netloc, parsed.path.rstrip("/"), "", "", ""))


def websocket_url(base_url: str, client_id: str) -> str:
    parsed = urlparse(normalise_base_url(base_url))
    scheme = "wss" if parsed.scheme == "https" else "ws"
    path = f"{parsed.path}/ws"
    return urlunparse((scheme, parsed.netloc, path, "", f"clientId={client_id}", ""))


class ComfyClient:
    """Thin, thread-confined wrapper over the ComfyUI HTTP endpoints this app uses."""

    def __init__(self, base_url: str = config.DEFAULT_SERVER_URL):
        self.base_url = normalise_base_url(base_url)
        self.session = requests.Session()
        self._object_info_cache: dict[str, dict] = {}

    def set_base_url(self, base_url: str) -> None:
        new_url = normalise_base_url(base_url)
        if new_url != self.base_url:
            self.base_url = new_url
            self._object_info_cache.clear()

    def close(self) -> None:
        self.session.close()

    # -- plumbing ------------------------------------------------------------------

    def _url(self, path: str) -> str:
        return f"{self.base_url}{path}"

    def _request(self, method: str, path: str, **kwargs) -> requests.Response:
        kwargs.setdefault("timeout", DEFAULT_TIMEOUT)
        try:
            return self.session.request(method, self._url(path), **kwargs)
        except requests.exceptions.RequestException as exc:
            raise ComfyUnreachable(f"{self.base_url} is not responding ({exc.__class__.__name__})") from exc

    def _get_json(self, path: str, **kwargs):
        response = self._request("GET", path, **kwargs)
        if response.status_code != 200:
            raise ComfyError(f"GET {path} returned HTTP {response.status_code}",
                             status=response.status_code)
        return response.json()

    # -- status --------------------------------------------------------------------

    def system_stats(self) -> dict:
        return self._get_json("/system_stats", timeout=5)

    def is_reachable(self) -> bool:
        try:
            self.system_stats()
        except (ComfyUnreachable, ComfyError):
            return False
        return True

    def queue_remaining(self) -> int:
        data = self._get_json("/prompt", timeout=5)
        return int(data.get("exec_info", {}).get("queue_remaining", 0))

    # -- node metadata -------------------------------------------------------------

    def object_info(self, class_type: str, *, refresh: bool = False) -> dict | None:
        """Schema for one node class, cached. None if the server does not have it."""
        if not refresh and class_type in self._object_info_cache:
            return self._object_info_cache[class_type]

        response = self._request("GET", f"/object_info/{class_type}")
        if response.status_code == 404:
            self._object_info_cache[class_type] = None
            return None
        if response.status_code != 200:
            raise ComfyError(f"/object_info/{class_type} returned HTTP {response.status_code}",
                             status=response.status_code)

        # The endpoint answers with {class_type: schema}; an unknown class yields {}.
        payload = response.json() or {}
        schema = payload.get(class_type)
        self._object_info_cache[class_type] = schema
        return schema

    def object_info_many(self, class_types, *, refresh: bool = False) -> dict[str, dict | None]:
        return {name: self.object_info(name, refresh=refresh) for name in class_types}

    # -- uploads -------------------------------------------------------------------

    def upload(self, local_path: str | Path, server_name: str, mime_type: str,
               subfolder: str = config.UPLOAD_SUBFOLDER) -> UploadResult:
        """Store a local file in ComfyUI's input directory.

        /upload/image is the only upload route ComfyUI exposes; it accepts audio and
        video on the same endpoint and form field. Uploading into a subfolder keeps the
        user's own LoadImage dropdown clean, and is safe because LoadImage, LoadAudio and
        LoadVideo all declare a custom validator for their filename argument, which
        disables ComfyUI's combo-membership check for that input.
        """
        with open(local_path, "rb") as fh:
            response = self._request(
                "POST", "/upload/image",
                files={"image": (server_name, fh, mime_type)},
                data={"type": "input", "subfolder": subfolder, "overwrite": "true"},
                timeout=UPLOAD_TIMEOUT,
            )

        if response.status_code != 200:
            raise ComfyError(
                f"Upload of {Path(local_path).name} failed (HTTP {response.status_code}): "
                f"{response.text[:300]}",
                status=response.status_code,
            )

        payload = response.json()
        return UploadResult(
            name=payload["name"],
            subfolder=payload.get("subfolder", "") or "",
            type=payload.get("type", "input") or "input",
        )

    def file_exists(self, filename: str, subfolder: str = "", type_: str = "input") -> bool:
        """Confirm a stored file is really retrievable.

        /view is the only reliable check: /object_info lists the input directory
        non-recursively, so anything in a subfolder never appears there.
        """
        response = self._request(
            "GET", "/view",
            params={"filename": filename, "subfolder": subfolder, "type": type_},
            stream=True,
        )
        try:
            return response.status_code == 200
        finally:
            response.close()

    # -- queueing ------------------------------------------------------------------

    def submit(self, graph: dict, client_id: str, prompt_id: str,
               partial_execution_targets: list[str] | None = None,
               extra_data: dict | None = None) -> dict:
        """POST a graph to /prompt. Raises ComfyError carrying node_errors on rejection.

        ``extra_data`` reaches nodes that declare an ``EXTRA_PNGINFO`` hidden input, which
        is the only channel some of them offer for options they have no widget for.
        """
        body = {"prompt": graph, "client_id": client_id, "prompt_id": prompt_id}
        if partial_execution_targets:
            body["partial_execution_targets"] = partial_execution_targets
        if extra_data:
            body["extra_data"] = extra_data

        response = self._request("POST", "/prompt", json=body, timeout=60)
        if response.status_code == 200:
            return response.json()

        try:
            payload = response.json()
        except ValueError:
            payload = {}

        error = payload.get("error") or {}
        message = error.get("message") or response.text[:300] or f"HTTP {response.status_code}"
        details = error.get("details")
        if details:
            message = f"{message}: {details}"
        raise ComfyError(message, status=response.status_code,
                         node_errors=payload.get("node_errors") or {}, payload=payload)

    def interrupt(self) -> None:
        self._request("POST", "/interrupt", timeout=10)

    def cancel_queued(self, prompt_id: str) -> None:
        """Remove a prompt that is still waiting in the queue.

        /interrupt only stops the job that is actually executing, so a queued job needs
        deleting as well or Cancel silently does nothing.
        """
        self._request("POST", "/queue", json={"delete": [prompt_id]}, timeout=10)

    def history(self, prompt_id: str) -> dict | None:
        data = self._get_json(f"/history/{prompt_id}")
        return (data or {}).get(prompt_id)

    def outputs_from_history(self, prompt_id: str, node_id: str) -> list[OutputRef]:
        entry = self.history(prompt_id)
        if not entry:
            return []
        return outputs_from_node_output((entry.get("outputs") or {}).get(node_id) or {})

    # -- downloads -----------------------------------------------------------------

    def download(self, ref: OutputRef, destination: str | Path,
                 progress=None) -> Path:
        """Stream a produced file to disk. ``progress`` receives (bytes_done, total)."""
        destination = Path(destination)
        destination.parent.mkdir(parents=True, exist_ok=True)

        response = self._request(
            "GET", "/view",
            params={"filename": ref.filename, "subfolder": ref.subfolder, "type": ref.type},
            stream=True, timeout=DOWNLOAD_TIMEOUT,
        )
        if response.status_code != 200:
            raise ComfyError(f"Could not fetch {ref.filename} (HTTP {response.status_code})",
                             status=response.status_code)

        total = int(response.headers.get("Content-Length") or 0)
        done = 0
        temp_path = destination.with_suffix(destination.suffix + ".part")
        try:
            with open(temp_path, "wb") as fh:
                for chunk in response.iter_content(chunk_size=1 << 18):
                    if not chunk:
                        continue
                    fh.write(chunk)
                    done += len(chunk)
                    if progress:
                        progress(done, total)
            temp_path.replace(destination)
        finally:
            response.close()
            temp_path.unlink(missing_ok=True)
        return destination


def outputs_from_node_output(node_output: dict) -> list[OutputRef]:
    """Extract file references from a node's UI output payload.

    SaveVideo reports its result under "images" (a PreviewVideo payload), but other save
    nodes use "videos"/"gifs"/"audio", so every known key is scanned rather than assuming
    one shape.
    """
    refs: list[OutputRef] = []
    for key in ("images", "videos", "gifs", "audio", "files"):
        for entry in node_output.get(key) or []:
            if isinstance(entry, dict) and "filename" in entry:
                refs.append(OutputRef.from_dict(entry))
    return refs


def describe_node_errors(node_errors: dict, labels: dict[str, str] | None = None) -> list[str]:
    """Turn a /prompt rejection into lines a user can act on.

    Node IDs are meaningless to the user, so each is mapped through the builder's label
    map -- "Reference image 3 (dragon.png)" rather than "node 202".
    """
    labels = labels or {}
    lines: list[str] = []

    for node_id, info in (node_errors or {}).items():
        label = labels.get(node_id) or f"{info.get('class_type', 'node')} ({node_id})"
        for error in info.get("errors") or []:
            message = error.get("message", "invalid input")
            details = error.get("details")
            extra = error.get("extra_info") or {}
            input_name = extra.get("input_name")

            if error.get("type") == "value_not_in_list" and input_name and _is_model_input(input_name):
                lines.append(
                    f"{label}: model file not found on the ComfyUI server - {details or message}"
                )
                continue

            text = f"{label}: {message}"
            if details:
                text += f" ({details})"
            lines.append(text)

    return lines or ["The server rejected the workflow but reported no per-node detail."]


def _is_model_input(input_name: str) -> bool:
    return input_name in {"unet_name", "vae_name", "clip_name", "ckpt_name", "lora_name"}


def format_execution_error(data: dict, labels: dict[str, str] | None = None) -> str:
    """One-paragraph summary of an execution_error websocket message."""
    labels = labels or {}
    node_id = str(data.get("node_id", ""))
    label = labels.get(node_id) or data.get("node_type") or f"node {node_id}"
    exception_type = data.get("exception_type", "Error")
    message = data.get("exception_message", "")

    text = f"{label} failed: {exception_type}"
    if message:
        text += f"\n\n{message}"

    if "OutOfMemory" in exception_type or "out of memory" in message.lower():
        text += (
            "\n\nThe GPU ran out of memory. Try lowering megapixels or duration, or using "
            "fewer / shorter reference videos - reference latents are carried through "
            "every sampling step."
        )
    return text


def pretty(graph: dict) -> str:
    return json.dumps(graph, indent=2, ensure_ascii=False)
