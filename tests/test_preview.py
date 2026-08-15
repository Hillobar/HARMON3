"""Decoding the live sampler preview sent by the Model Preview Override node.

The fixtures are the real formats that node emits: a fragmented H.264 MP4 when the
server has NVENC, an animated WebP when it does not, and a plain JPEG when
preview_frames is 1.
"""

from __future__ import annotations

import base64
import io
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

QtGui = pytest.importorskip("PySide6.QtGui")
av = pytest.importorskip("av")

from harmon3 import preview                                     # noqa: E402


@pytest.fixture(scope="module")
def qapp():
    import os
    from PySide6.QtWidgets import QApplication
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    app = QApplication.instance() or QApplication([])
    yield app


#: A 6-frame animated WebP, produced by the same Pillow call the node's fallback uses.
WEBP_ANIM_B64 = (
    "UklGRioCAABXRUJQVlA4WAoAAAACAAAAHwAAFwAAQU5JTQYAAAAAAAAAAABBTk1GVgAAAAAAAAAAAB8A"
    "ABcAAKYAAAJWUDggPgAAAFADAJ0BKiAAGAA+bTSWR6QjIiEoCACADYllAHYA/AAAgzWYAP70UR//+wd"
    "/Qd/Qd/tV///ILlhdcWHYGgAAQU5NRlIAAAAAAAAAAAAfAAAXAACmAAAAVlA4IDoAAADUAgCdASogABg"
    "APm00lkeCgIAAANiWUAyBKAAEBQUAAP7x3I//Qk/+Js/4mz4RX6+AKfh/2B1AAAAAQU5NRkoAAAAAAAA"
    "AAAAfAAAXAACmAAAAVlA4IDIAAAB0AgCdASogABgAPm00lkeCgIAAANiWUAAEVw5AAP7wm0P//kp/95X"
    "+u7cFOWaKEAAAAEFOTUZMAAAAAAAAAAAAHwAAFwAApgAAAFZQOCA0AAAAtAIAnQEqIAAYAD5tMpVHgoC"
    "AAADYllAHYAAICgoAAP7vKUf/w5U+nb2P/xQCkcMwAAAAAEFOTUZOAAAAAAAAAAAAHwAAFwAApgAAAlZ"
    "QOCA2AAAA0AIAnQEqIAAYAD5tMpVHpCKiISgIAIANiWUAAD2joAD+7Qcv87VsfJj//5Kf/eV/ru2YAAA"
    "AQU5NRkoAAAAAAAAAAAAfAAAXAACmAAAAVlA4IDIAAAB0AgCdASogABgAPm0ylkeCgIAAANiWUAAD2jo"
    "AAP7paRpvX//pNn/9Js//pNnxzAAAAA=="
)


def _fragmented_mp4(frames: int = 8, width: int = 64, height: int = 48, fps: int = 6) -> bytes:
    """Build the same container shape the node produces, minus the NVENC codec.

    movflags matter: the node writes a fragmented MP4 so a browser can decode it
    mid-download, and that is what has to survive a from-memory decode here.
    """
    buf = io.BytesIO()
    container = av.open(
        buf, mode="w", format="mp4",
        options={"movflags": "frag_keyframe+empty_moov+default_base_moof"})
    stream = container.add_stream("libx264", rate=fps)
    stream.width, stream.height, stream.pix_fmt = width, height, "yuv420p"
    stream.options = {"preset": "ultrafast", "crf": "30"}

    for i in range(frames):
        frame = _solid_frame(width, height, ((i * 30) % 256, 80, 160))
        for packet in stream.encode(frame):
            container.mux(packet)
    for packet in stream.encode():
        container.mux(packet)
    container.close()
    return buf.getvalue()


def _solid_frame(width: int, height: int, colour: tuple):
    """Build a frame without numpy, which is deliberately not a dependency."""
    frame = av.VideoFrame(width, height, "rgb24")
    plane = frame.planes[0]
    row = bytes(colour) * width
    padded = row + b"\x00" * (plane.line_size - len(row))
    plane.update(padded * height)
    return frame


def _jpeg(width: int = 40, height: int = 30) -> bytes:
    from PySide6.QtCore import QBuffer, QByteArray
    from PySide6.QtGui import QImage

    image = QImage(width, height, QImage.Format_RGB888)
    image.fill(QtGui.QColor("#2277cc"))
    data = QByteArray()
    buffer = QBuffer(data)
    buffer.open(QBuffer.WriteOnly)
    image.save(buffer, "JPEG", 80)
    buffer.close()
    return bytes(data)


def _message(payload: bytes, mime: str, **extra) -> dict:
    body = {
        "node_id": "141",
        "image": base64.b64encode(payload).decode("ascii"),
        "mime": mime,
        "step": 7,
        "total": 20,
        "sigma": 3.25,
        "fps": 6,
        "avg_step_ms": 1450.0,
        "step_ms": 1400.0,
    }
    body.update(extra)
    return body


# ---------------------------------------------------------------------------------
# MP4 -- the path this server actually uses (its PyAV has NVENC)
# ---------------------------------------------------------------------------------

def test_decodes_a_fragmented_mp4_clip(qapp):
    clip = preview.decode_message(_message(_fragmented_mp4(frames=8), "video/mp4"))

    assert clip is not None
    assert len(clip.frames) == 8
    assert all(not f.isNull() for f in clip.frames)
    assert clip.frames[0].width() == 64 and clip.frames[0].height() == 48


def test_mp4_metadata_travels_with_the_clip(qapp):
    clip = preview.decode_message(_message(_fragmented_mp4(frames=4), "video/mp4"))
    assert (clip.step, clip.total) == (7, 20)
    assert clip.fps == 6
    assert clip.sigma == pytest.approx(3.25)
    assert clip.avg_step_ms == pytest.approx(1450.0)


def test_frames_hold_their_own_pixels(qapp):
    """The decoder copies out of PyAV's buffer; without that the frames alias each other."""
    clip = preview.decode_message(_message(_fragmented_mp4(frames=6), "video/mp4"))
    reds = [f.pixelColor(f.width() // 2, f.height() // 2).red() for f in clip.frames]
    assert len(set(reds)) > 1, "every frame decoded to the same image"


def test_large_frames_are_scaled_down(qapp):
    """A 24-frame clip at the node's 1024px cap is ~42 MB held per step otherwise."""
    payload = _fragmented_mp4(frames=2, width=1280, height=720)
    clip = preview.decode_message(_message(payload, "video/mp4"))
    assert max(clip.frames[0].width(), clip.frames[0].height()) == preview.MAX_PREVIEW_EDGE


def test_small_frames_are_left_alone(qapp):
    clip = preview.decode_message(_message(_fragmented_mp4(width=64, height=48), "video/mp4"))
    assert (clip.frames[0].width(), clip.frames[0].height()) == (64, 48)


# ---------------------------------------------------------------------------------
# WebP and JPEG -- the node's fallbacks
# ---------------------------------------------------------------------------------

def test_decodes_an_animated_webp(qapp):
    clip = preview.decode_message(
        _message(base64.b64decode(WEBP_ANIM_B64), "image/webp"))
    assert clip is not None
    assert len(clip.frames) == 6
    assert clip.frames[0].width() == 32


def test_decodes_a_single_jpeg(qapp):
    """preview_frames = 1 sends a still rather than a clip."""
    clip = preview.decode_message(_message(_jpeg(), "image/jpeg", fps=None))
    assert clip is not None
    assert len(clip.frames) == 1
    assert clip.fps == 12.0          # falls back to a sane default


# ---------------------------------------------------------------------------------
# Messages that carry no picture
# ---------------------------------------------------------------------------------

def test_the_opening_sigma_message_is_not_a_clip(qapp):
    """The node's first message delivers the sigma schedule and no image."""
    assert preview.decode_message(
        {"node_id": "141", "step": 0, "total": 20, "sigmas": [1.0, 0.5]}) is None


@pytest.mark.parametrize("data", [None, {}, [], "nope", {"image": ""}, {"image": None}])
def test_junk_messages_are_ignored(qapp, data):
    assert preview.decode_message(data) is None


def test_undecodable_base64_is_ignored(qapp):
    assert preview.decode_message({"image": "not base64!!", "mime": "video/mp4"}) is None


def test_a_truncated_mp4_is_ignored_rather_than_raising(qapp):
    payload = _fragmented_mp4(frames=4)[:120]
    assert preview.decode_message(_message(payload, "video/mp4")) is None


def test_bytes_that_are_not_an_image_are_ignored(qapp):
    assert preview.decode_message(_message(b"\x00\x01\x02\x03" * 40, "image/webp")) is None


def test_an_empty_clip_is_falsy(qapp):
    assert not preview.PreviewClip()
    assert preview.decode_message(_message(_fragmented_mp4(frames=2), "video/mp4"))
