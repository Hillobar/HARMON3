"""Reading a reference image the way the server will see it.

A photograph from a phone or a camera is very often stored landscape with an EXIF
*Orientation* tag saying "turn this a quarter". ComfyUI's ``LoadImage`` obeys that tag --
``ImageOps.exif_transpose``, nodes.py -- and Qt, by default, does not: ``QImage(path)``
hands back the stored pixels, and even ``QImageReader.size()`` reports the stored size
with ``setAutoTransform(True)`` on. Only ``read()`` applies it.

Left alone, that splits the app from the server in three ways at once: the thumbnail and
the preview show a picture lying on its side, the row reports 4000x3000 for something the
model will receive as 3000x4000, and -- worst -- a rescaled copy is written as PNG, which
carries no EXIF at all, so the *unscaled* reference arrives upright and the scaled one
arrives sideways.

So everything that opens a reference image goes through here.
"""

from __future__ import annotations

import logging
from pathlib import Path

from PySide6.QtGui import QImage, QImageIOHandler, QImageReader, QPixmap

log = logging.getLogger(__name__)

#: Qt reports orientation as three flags -- Mirror, Flip and Rotate90 -- from which the
#: eight EXIF values are composed (180 degrees is Mirror|Flip, 270 is all three). Only the
#: quarter turn swaps width and height, so it is the one bit worth testing: mirroring an
#: image leaves it exactly as wide as it was.
_ROTATE_90 = QImageIOHandler.Transformation.TransformationRotate90


def _reader(path) -> QImageReader:
    reader = QImageReader(str(path))
    reader.setAutoTransform(True)
    return reader


def oriented_size(path) -> tuple[int, int] | None:
    """The dimensions after EXIF orientation, without decoding the whole picture.

    ``QImageReader.size()`` is the *stored* size whatever autoTransform says, so the swap
    is done here from the transformation it reports. Reading the header rather than the
    image keeps this cheap enough to run on every reference as it is dropped.
    """
    reader = _reader(path)
    size = reader.size()
    if not size.isValid():
        return None
    width, height = size.width(), size.height()
    if reader.transformation() & _ROTATE_90:
        width, height = height, width
    return width, height


def load(path) -> QImage:
    """The picture, turned the right way up. A null QImage if it cannot be read."""
    image = _reader(path).read()
    if image.isNull():
        log.info("Could not read %s as an image", Path(path).name)
    return image


def load_pixmap(path) -> QPixmap:
    """The same, for the widgets that draw one. Null if it cannot be read."""
    image = load(path)
    return QPixmap() if image.isNull() else QPixmap.fromImage(image)
