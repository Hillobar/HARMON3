"""Tests for what actually goes on the wire to /prompt."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from harmon3 import comfy_http                                  # noqa: E402
from harmon3.comfy_http import ComfyClient, ComfyError          # noqa: E402


class _Response:
    def __init__(self, status_code=200, payload=None, text=""):
        self.status_code = status_code
        self._payload = payload if payload is not None else {}
        self.text = text

    def json(self):
        return self._payload

    def close(self):
        pass


@pytest.fixture
def client(monkeypatch):
    """A client whose requests are captured instead of sent."""
    made = []

    def fake_request(self, method, path, **kwargs):
        made.append((method, path, kwargs))
        return _Response(payload={"prompt_id": "p"})

    monkeypatch.setattr(ComfyClient, "_request", fake_request)
    instance = ComfyClient("http://example.invalid")
    instance.sent = made
    return instance


def _body(client):
    return client.sent[-1][2]["json"]


def test_a_plain_submit_carries_only_the_prompt(client):
    client.submit({"1": {}}, "cid", "pid")
    body = _body(client)
    assert body == {"prompt": {"1": {}}, "client_id": "cid", "prompt_id": "pid"}


def test_extra_data_reaches_the_request_body(client):
    """It is the only channel VideoHelperSuite offers for options it has no widget for."""
    extra = {"extra_pnginfo": {"workflow": {"extra": {"VHS_KeepIntermediate": False}}}}
    client.submit({"1": {}}, "cid", "pid", extra_data=extra)
    assert _body(client)["extra_data"] == extra


@pytest.mark.parametrize("empty", [None, {}])
def test_empty_extra_data_is_left_off_entirely(client, empty):
    """A graph that reads none must not have the key invented for it."""
    client.submit({"1": {}}, "cid", "pid", extra_data=empty)
    assert "extra_data" not in _body(client)


def test_partial_execution_targets_and_extra_data_coexist(client):
    client.submit({"1": {}}, "cid", "pid", partial_execution_targets=["9"],
                  extra_data={"extra_pnginfo": {}})
    body = _body(client)
    assert body["partial_execution_targets"] == ["9"]
    assert body["extra_data"] == {"extra_pnginfo": {}}


# ---------------------------------------------------------------------------------
# Where the server writes. /system_stats reports the server's own argv, which is the
# only place the HTTP API says anything about a filesystem path at all.
# ---------------------------------------------------------------------------------

def _stats(*argv):
    return {"system": {"comfyui_version": "0.33.0", "argv": list(argv)}}


@pytest.mark.parametrize("argv", [
    ("main.py", "--output-directory", r"F:\outputs", "--listen"),
    ("main.py", r"--output-directory=F:\outputs"),
    ("main.py", "--listen", "--output-directory", '"F:\\outputs"'),
])
def test_the_output_directory_is_read_from_the_servers_own_argv(argv):
    assert comfy_http.output_dir_from_stats(_stats(*argv)) == r"F:\outputs"


@pytest.mark.parametrize("stats", [
    _stats("main.py", "--listen"),          # no flag: ComfyUI defaults to <install>/output
    _stats("main.py", "--output-directory"),            # flag with nothing after it
    _stats("main.py", "--output-directory", "outputs"),  # relative to a cwd nobody reports
    _stats(), {}, {"system": {}}, {"system": {"argv": None}},
])
def test_an_unknowable_output_directory_is_not_guessed_at(stats):
    """argv[0] is a bare 'main.py', so there is nothing to join a default onto, and a
    relative path is relative to a working directory the API never mentions."""
    assert comfy_http.output_dir_from_stats(stats) is None


def test_a_rejection_still_carries_its_node_errors(monkeypatch):
    """extra_data must not disturb how a 400 is reported."""
    def fake_request(self, method, path, **kwargs):
        return _Response(400, {"error": {"message": "bad", "details": "why"},
                               "node_errors": {"3": {}}})

    monkeypatch.setattr(ComfyClient, "_request", fake_request)
    with pytest.raises(ComfyError) as excinfo:
        ComfyClient("http://example.invalid").submit(
            {"1": {}}, "cid", "pid", extra_data={"extra_pnginfo": {}})
    assert excinfo.value.node_errors == {"3": {}}
    assert "bad: why" in str(excinfo.value)
