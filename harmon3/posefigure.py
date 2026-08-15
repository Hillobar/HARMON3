"""The ``figure`` pose style: a skeleton drawn to be looked at.

The two OpenPose styles are drawn by ``rtmlib``'s ``draw_openpose``, and three of its
constants are hardcoded past reach: joint radius is 3 or 4 pixels whatever the frame size
(the scaling line is commented out in the library), links past index 16 are a 2px line,
and its alpha blend is ``addWeighted(img, 1-a, img, a)`` -- the filled image against
itself, so nothing is ever translucent. Reaching around all three costs more than not
using it, so this style paints its own.

What it buys, beyond size:

* **It works out which way the subject is facing** and draws a face only when there is one
  to see. A 2D skeleton of someone facing away is very nearly the mirror of one facing the
  camera, and the estimator will happily place eyes and a mouth on the back of a head.
* **The trunk is a filled shape** and the limbs are rimmed in black, so a limb crossing
  the body separates instead of merging with it -- and the trunk is painted over the arms
  when the subject faces away, under them when it does not, which is the whole cue.
* **Right is warm and left is cool.** OpenPose's palette is a continuous hue sweep by link
  index, so it does not separate the sides at all.
* **The feet exist.** ``openpose134`` gives them ``color=[0, 0, 0]`` and ``draw_openpose``
  skips any joint whose colour sums to zero, so the six foot keypoints the wholebody model
  pays for have never been drawn. Neither has the face as anything but loose dots:
  ``openpose134``'s 57 links are the body and the two hands, and there are no face links
  at all. Both tables are defined here.

Nothing in here imports ``pose``; ``pose`` imports this. ``numpy``, ``cv2`` and ``rtmlib``
are imported inside the functions that need them, so the app and the test suite still load
on a machine without the pose dependencies -- and ``Facing`` is deliberately plain Python
arithmetic, so the facing tests run there too.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

#: The one style id this module owns. ``pose.draw`` branches on it before it reaches
#: ``skeleton_for``, because there is no rtmlib table behind this style to substitute into.
STYLE = "figure"


# ------------------------------------------------------------------------------ layout

# OpenPose-18, which every layout here starts with.
NOSE, NECK = 0, 1
R_SHOULDER, R_ELBOW, R_WRIST = 2, 3, 4
L_SHOULDER, L_ELBOW, L_WRIST = 5, 6, 7
R_HIP, R_KNEE, R_ANKLE = 8, 9, 10
L_HIP, L_KNEE, L_ANKLE = 11, 12, 13
R_EYE, L_EYE, R_EAR, L_EAR = 14, 15, 16, 17

#: OpenPose-134 adds the feet at 18-23, the dlib-68 face at 24-91 and 21 points per hand
#: from 92. Keyed by position in the table and never by ``keypoint_info[i]["id"]``: that
#: field is off by one from 18 to 91 in rtmlib's own table -- index 18 carries ``id=17``,
#: colliding with ``left_ear`` -- and corrects itself at 92. ``draw_openpose`` builds its
#: lookup from ``id`` but only ever indexes the body and the hands with it, so the bug has
#: never had anything to break.
L_BIG_TOE, L_SMALL_TOE, L_HEEL = 18, 19, 20
R_BIG_TOE, R_SMALL_TOE, R_HEEL = 21, 22, 23
FACE_FIRST, FACE_STOP = 24, 92
JAW = range(24, 41)
L_HAND_ROOT, R_HAND_ROOT = 92, 113
_HAND_POINTS = 21

#: The dlib-68 groups, shifted by the neck OpenPose inserts at index 1, as (points, closed).
#: The jaw is missing on purpose -- it is what sizes the head, and once the head is a
#: filled disc an outline around its lower half is clutter rather than information.
_FACE_GROUPS = (
    (range(41, 46), False),                      # one brow
    (range(46, 51), False),                      # the other
    (range(51, 55), False),                      # nose bridge
    (range(55, 60), False),                      # nose base
    (range(72, 84), True),                       # outer lip
    (range(84, 92), True),                       # inner lip
)
#: Kept apart from the rest because they are coloured by side rather than by feature.
_EYE_GROUPS = (range(60, 66), range(66, 72))

#: A hand is a fixed 21-point layout, so it is five chains off its root rather than a table.
_FINGERS = ((1, 2, 3, 4), (5, 6, 7, 8), (9, 10, 11, 12), (13, 14, 15, 16), (17, 18, 19, 20))
_MIDDLE_TIP = 12                                 # what a hand's size is measured against
_PALM = (5, 17)                                  # forefinger and pinky knuckles


# ------------------------------------------------------------------------------ palette

# Every colour here is BGR, straight to cv2, reversed nowhere. rtmlib is inconsistent
# about this -- it passes link colours raw and joint colours reversed, so the same triple
# means two different colours a dozen lines apart -- and one convention is worth more than
# matching its numbers.

RIM = (0, 0, 0)

R_UPPER_ARM, R_FOREARM = (40, 110, 255), (55, 175, 255)
R_THIGH, R_SHIN = (60, 60, 235), (75, 120, 255)
R_EXTREMITY = (90, 205, 255)

L_UPPER_ARM, L_FOREARM = (255, 175, 60), (230, 200, 90)
L_THIGH, L_SHIN = (235, 110, 45), (255, 200, 120)
L_EXTREMITY = (235, 225, 150)

TRUNK = (120, 120, 128)
SPINE = (200, 200, 205)
#: Darker than the trunk on purpose, twice over: so the head is not read as the top of the
#: torso, and so the near-white features have something to be legible against.
HEAD = (100, 100, 112)
FACE = (245, 245, 250)

#: (start, end, colour, width) for the four limb segments of each side. Adjacent segments
#: differ so an elbow or a knee is visible even where the rim is not.
_ARMS = ((R_SHOULDER, R_ELBOW, R_UPPER_ARM, "arm"),
         (R_ELBOW, R_WRIST, R_FOREARM, "forearm"),
         (L_SHOULDER, L_ELBOW, L_UPPER_ARM, "arm"),
         (L_ELBOW, L_WRIST, L_FOREARM, "forearm"))
_LEGS = ((R_HIP, R_KNEE, R_THIGH, "thigh"),
         (R_KNEE, R_ANKLE, R_SHIN, "shin"),
         (L_HIP, L_KNEE, L_THIGH, "thigh"),
         (L_KNEE, L_ANKLE, L_SHIN, "shin"))

#: Joint -> the colour it takes, brightened. A joint reads as the end of the segment it
#: terminates rather than as a thing of its own.
_JOINTS = ((R_SHOULDER, R_UPPER_ARM), (R_ELBOW, R_FOREARM), (R_WRIST, R_EXTREMITY),
           (L_SHOULDER, L_UPPER_ARM), (L_ELBOW, L_FOREARM), (L_WRIST, L_EXTREMITY),
           (R_HIP, R_THIGH), (R_KNEE, R_SHIN), (R_ANKLE, R_EXTREMITY),
           (L_HIP, L_THIGH), (L_KNEE, L_SHIN), (L_ANKLE, L_EXTREMITY))


# ------------------------------------------------------------------------- facing

#: Divides the shoulder projection before it is clamped. Biacromial width is about 0.8 of
#: neck-to-pelvis, so the projection is 0.8*cos(azimuth); 0.35 saturates it within about
#: 64 degrees of frontal or of back and lets it decay to zero only near true profile.
#: Confident across most of the range, honestly undecided where the geometry genuinely is.
_GAIN = 0.35
#: EMA rate, scaled per frame by how much of the evidence actually turned up. About an
#: eight-frame time constant, a third of a second at 24fps.
_ALPHA = 0.12
#: Dead band on a latched decision, which is already a Schmitt trigger -- separate enter
#: and exit thresholds would say the same thing twice.
_BAND = 0.20

#: cue -> weight. The shoulders carry it; the rest can shade a verdict the torso has
#: nearly reached, and together they can outvote it only when it is close to profile.
_W_SHOULDERS, _W_HIPS, _W_EARS, _W_TOES, _W_FACE = 1.0, 0.8, 0.6, 0.5, 0.4


@dataclass
class Facing:
    """Which way the subject is turned, smoothed over the clip.

    ``frontal`` is a latch rather than a threshold on ``smoothed``: a subject near profile
    produces a score near zero, and a figure whose face flickered on and off through every
    profile frame would be worse to watch than one that simply held its last verdict.

    One person's worth of state. ``_pick_subject`` can fall through to the biggest box and
    change person without saying so, which leaves this briefly describing the wrong body;
    it corrects itself within about a third of a second, which is cheaper than threading a
    "did I switch" signal through a function three tests already hold still.
    """

    smoothed: float | None = None
    frontal: bool = True

    @property
    def score(self) -> float:
        """+1 facing the camera, -1 facing away, 0 before anything has been seen."""
        return 0.0 if self.smoothed is None else self.smoothed

    def update(self, keypoints, scores, kpt_thr: float) -> None:
        """Fold one frame's keypoints in. Call this only on a *fresh* estimate.

        A held frame re-draws the previous keypoints, and feeding those in again would
        count the same evidence twice and pull the estimate toward whatever it already
        believed -- which is exactly the failure the hold exists to avoid elsewhere.
        """
        frame = _body_frame(keypoints, scores, kpt_thr)
        if frame is None:
            return
        lat, _up, scale = frame

        total = weighted = available = 0.0
        for value, gate, weight in _cues(keypoints, scores, lat, scale, kpt_thr):
            available += weight
            if gate < kpt_thr:
                continue
            total += weight * gate * value
            weighted += weight * gate

        if weighted <= 0.0:
            # Nothing confident this frame. Left where it was rather than nudged toward
            # zero: absence of evidence is not evidence of a profile.
            return

        instant = total / weighted
        confidence = weighted / available

        if self.smoothed is None:
            # Seeded, not eased in from zero. Easing would open every back-facing clip
            # with a third of a second of face drawn on the back of a head.
            self.smoothed = instant
        else:
            self.smoothed += _ALPHA * confidence * (instant - self.smoothed)

        if self.smoothed > _BAND:
            self.frontal = True
        elif self.smoothed < -_BAND:
            self.frontal = False


def _cues(keypoints, scores, lat, scale, kpt_thr):
    """Every reading available for this layout, as (value in -1..1, gate, weight).

    A cue the layout does not carry is not yielded at all, rather than yielded with a zero
    gate: a keypoint that does not exist is not missing evidence, and counting it as such
    would permanently slow the estimate on the 18-point models for no reason.
    """
    count = len(scores)

    # The two that matter. Projected onto the body's own lateral axis rather than read off
    # image x, so a subject lying down or a canted camera is still read correctly.
    yield (_clamp(_dot(_delta(keypoints, L_SHOULDER, R_SHOULDER), lat) / (scale * _GAIN)),
           min(_score(scores, L_SHOULDER), _score(scores, R_SHOULDER)), _W_SHOULDERS)
    yield (_clamp(_dot(_delta(keypoints, L_HIP, R_HIP), lat) / (scale * _GAIN)),
           min(_score(scores, L_HIP), _score(scores, R_HIP)), _W_HIPS)

    # The ears are the cosine of the head's own azimuth, so they normalise by their own
    # span -- and have to abstain when that span collapses, which is where it is noise.
    ear_gap = _delta(keypoints, L_EAR, R_EAR)
    span = math.hypot(*ear_gap)
    ear_gate = min(_score(scores, L_EAR), _score(scores, R_EAR))
    yield (_clamp(_dot(ear_gap, lat) / span) if span else 0.0,
           ear_gate if span >= 0.15 * scale else 0.0, _W_EARS)

    if count <= R_HEEL:
        return

    # The big toe is the medial one, so both feet flip together when the subject turns.
    # Toe-below-heel would say the same thing for someone standing and the wrong thing for
    # someone seated or kneeling. Weak against heavy turnout, where each foot's own axis
    # has rotated far enough to shrink the projection -- hence the low weight.
    toes = 0.5 * (_dot(_delta(keypoints, R_BIG_TOE, R_SMALL_TOE), lat)
                  - _dot(_delta(keypoints, L_BIG_TOE, L_SMALL_TOE), lat))
    yield (_clamp(toes / (0.08 * scale)),
           min(_score(scores, index) for index in
               (R_BIG_TOE, R_SMALL_TOE, L_BIG_TOE, L_SMALL_TOE)), _W_TOES)

    if count < FACE_STOP:
        return

    # How sure the estimator is about the face at all. Gated on the neck rather than on the
    # face points it reads: this cue's evidence *is* the collapse of face confidence, so
    # gating it on face confidence would silence it in the one case it has something to say.
    yield (_clamp((_face_confidence(scores) - 0.45) / 0.25), _score(scores, NECK), _W_FACE)


def _face_confidence(scores) -> float:
    return (sum(float(scores[i]) for i in range(FACE_FIRST, FACE_STOP))
            / (FACE_STOP - FACE_FIRST))


def _body_frame(keypoints, scores, kpt_thr):
    """(lateral axis, up axis, torso length), or None if there is no torso to measure.

    ``lat`` points toward the subject's anatomical left while they face the camera, which
    is what makes the sign of every projection above mean "frontal". The scale is the torso
    *length* and not the shoulder width, because shoulder width collapses at profile and
    dividing by it would report profile as a confident verdict either way.
    """
    neck = _point(keypoints, scores, NECK, kpt_thr)
    shoulders = _midpoint(keypoints, scores, R_SHOULDER, L_SHOULDER, kpt_thr)
    top = neck or shoulders
    bottom = _midpoint(keypoints, scores, R_HIP, L_HIP, kpt_thr) or shoulders

    if top is not None and bottom is not None:
        length = math.hypot(top[0] - bottom[0], top[1] - bottom[1])
        if length >= 1.0:
            up = ((top[0] - bottom[0]) / length, (top[1] - bottom[1]) / length)
            return (-up[1], up[0]), up, length

    if shoulders is None:
        return None
    # Shoulders and nothing below them -- a head-and-shoulders framing. Upright is the
    # only assumption available, and the torso length is estimated from the one width there
    # is, so the shoulder cue still votes instead of the whole estimate going dark.
    width = math.hypot(*_delta(keypoints, L_SHOULDER, R_SHOULDER))
    if width < 1.0:
        return None
    return (1.0, 0.0), (0.0, -1.0), width / 0.8


# ----------------------------------------------------------------------- plain geometry

def _score(scores, index) -> float:
    return float(scores[index])


def _delta(keypoints, a, b):
    return (float(keypoints[a][0]) - float(keypoints[b][0]),
            float(keypoints[a][1]) - float(keypoints[b][1]))


def _dot(u, v) -> float:
    return u[0] * v[0] + u[1] * v[1]


def _clamp(value: float, limit: float = 1.0) -> float:
    return max(-limit, min(limit, value))


def _point(keypoints, scores, index, kpt_thr):
    """One keypoint, or None if it is not confident enough to build on."""
    if float(scores[index]) < kpt_thr:
        return None
    return float(keypoints[index][0]), float(keypoints[index][1])


def _midpoint(keypoints, scores, a, b, kpt_thr):
    first, second = (_point(keypoints, scores, a, kpt_thr),
                     _point(keypoints, scores, b, kpt_thr))
    if first is None or second is None:
        return None
    return (first[0] + second[0]) / 2.0, (first[1] + second[1]) / 2.0


def _confident(scores, kpt_thr, *indices) -> bool:
    return all(float(scores[i]) >= kpt_thr for i in indices)


def _brighten(colour, amount: float = 0.45):
    return tuple(int(round(c + (255 - c) * amount)) for c in colour)


# -------------------------------------------------------------------------- primitives

def bone(canvas, keypoints, scores, start: int, end: int, colour, *, kpt_thr: float,
         line_width: int, alpha: float = 1.0):
    """One limb, as an ellipse along the bone. Returns the canvas to keep.

    The geometry is ``draw_openpose``'s, so a limb drawn here is indistinguishable from one
    drawn by the library, and so are the guards: a limb with an unconfident or off-canvas
    end is skipped rather than drawn to a point clamped at the edge of the frame.

    ``alpha`` below 1.0 draws onto a *copy*, which is the whole of what rtmlib's
    ``draw_polygons`` does with it -- its blend is the filled image weighted against
    itself, so the fill is opaque either way and only the copy is real. Reproduced rather
    than corrected, so the two OpenPose styles keep the pixels they have always produced;
    this style passes 1.0 and saves a full-frame copy per limb.
    """
    import cv2

    height, width = canvas.shape[:2]
    x0, y0 = float(keypoints[start][0]), float(keypoints[start][1])
    x1, y1 = float(keypoints[end][0]), float(keypoints[end][1])
    if scores[start] < kpt_thr or scores[end] < kpt_thr:
        return canvas
    if not (0 < x0 < width and 0 < y0 < height and 0 < x1 < width and 0 < y1 < height):
        return canvas

    target = canvas if alpha >= 1.0 else canvas.copy()
    cv2.fillConvexPoly(target, _bone_polygon((x0, y0), (x1, y1), line_width), colour)
    return target


def _bone_polygon(start, end, half_width: int):
    import cv2

    length = math.hypot(start[0] - end[0], start[1] - end[1])
    angle = math.degrees(math.atan2(start[1] - end[1], start[0] - end[0]))
    return cv2.ellipse2Poly((int((start[0] + end[0]) / 2), int((start[1] + end[1]) / 2)),
                            (int(length / 2), max(1, int(half_width))), int(angle), 0, 360, 1)


def _rimmed_bone(canvas, keypoints, scores, start, end, colour, *, kpt_thr, half, rim):
    """A limb with a black edge, so one crossing another separates instead of merging.

    Rim and fill go down together, per limb, in depth order. A pass of every rim followed
    by a pass of every fill would leave every crossing exactly as unreadable as it is now:
    the later limb's fill lands on the earlier limb's rim and covers it.
    """
    bone(canvas, keypoints, scores, start, end, RIM,
         kpt_thr=kpt_thr, line_width=half + rim)
    return bone(canvas, keypoints, scores, start, end, colour,
                kpt_thr=kpt_thr, line_width=half)


def _filled(canvas, points, colour, rim):
    """A convex shape with the same black edge. ``points`` is a list of (x, y) floats.

    The hull rather than the polygon as given: a torso twisted past profile makes a
    self-intersecting quad, which ``fillConvexPoly`` renders as garbage rather than as the
    triangle it has degenerated into.
    """
    import cv2
    import numpy as np

    hull = cv2.convexHull(np.array([[int(round(x)), int(round(y))] for x, y in points],
                                   dtype=np.int32))
    cv2.polylines(canvas, [hull], True, RIM, thickness=max(1, rim * 2), lineType=cv2.LINE_AA)
    cv2.fillConvexPoly(canvas, hull, colour, lineType=cv2.LINE_AA)
    return canvas


def _disc(canvas, centre, radius: int, colour, rim: int):
    import cv2

    middle = (int(round(centre[0])), int(round(centre[1])))
    cv2.circle(canvas, middle, radius + rim, RIM, -1, lineType=cv2.LINE_AA)
    cv2.circle(canvas, middle, radius, colour, -1, lineType=cv2.LINE_AA)
    return canvas


def _chain(canvas, keypoints, scores, indices, colour, *, kpt_thr, width, closed=False):
    """A polyline through keypoints, drawn only if every one of them is confident.

    All or nothing on purpose: half an eye or three quarters of a lip is read as a
    deformity rather than as missing data.
    """
    import cv2
    import numpy as np

    indices = list(indices)
    if not _confident(scores, kpt_thr, *indices):
        return canvas
    points = np.array([[int(round(float(keypoints[i][0]))),
                        int(round(float(keypoints[i][1])))] for i in indices], dtype=np.int32)
    cv2.polylines(canvas, [points], closed, colour, thickness=max(1, width),
                  lineType=cv2.LINE_AA)
    return canvas


# ------------------------------------------------------------------------------ painting

def paint(canvas, keypoints, scores, *, kpt_thr: float, radius: int, line_width: int,
          facing: Facing | None = None):
    """One subject, drawn to be read. Returns the canvas to keep.

    One person's keypoints, shaped (N, 2); the loop over people stays with the caller, the
    way ``draw_openpose`` is called. Works on OpenPose-18 and OpenPose-134 -- the parts of
    the second layout the first does not have simply do not run.

    ``line_width`` is taken as the half-width of a limb, where the OpenPose styles double
    it first, so a figure here is about half as thick as one of theirs. Deliberate: the
    rims and the filled trunk carry the mass now, and a 24-pixel arm at 1080p swallows its
    own elbow.
    """
    frontal = True if facing is None else facing.frontal
    count = len(scores)
    frame = _body_frame(keypoints, scores, kpt_thr)
    lat, up = ((1.0, 0.0), (0.0, -1.0)) if frame is None else (frame[0], frame[1])

    rim = max(1, round(line_width * 0.45))
    half = {"trunk": max(1, round(line_width * 1.25)),
            "thigh": max(1, round(line_width * 1.25)),
            "arm": max(1, int(line_width)),
            "shin": max(1, int(line_width)),
            "forearm": max(1, round(line_width * 0.85))}

    def trunk():
        _paint_trunk(canvas, keypoints, scores, kpt_thr=kpt_thr, half=half["trunk"], rim=rim)

    def limbs():
        for start, end, colour, width in _LEGS + _ARMS:
            _rimmed_bone(canvas, keypoints, scores, start, end, colour,
                         kpt_thr=kpt_thr, half=half[width], rim=rim)

    # The one cue that does the work: an arm crossed in front of the body is painted over
    # the trunk when the subject faces the camera and buried under it when they do not.
    if frontal:
        trunk()
        limbs()
    else:
        limbs()
        trunk()

    head = _head_circle(keypoints, scores, lat, up, kpt_thr, frame, radius)
    if head is not None:
        centre, head_radius = head
        _neck_to_head(canvas, keypoints, scores, centre, kpt_thr=kpt_thr,
                      half=half["arm"], rim=rim)
        _disc(canvas, centre, head_radius, HEAD, rim)
        if frontal:
            _paint_face(canvas, keypoints, scores, count, centre, head_radius, lat,
                        kpt_thr=kpt_thr, line_width=line_width)
        else:
            _paint_crown(canvas, centre, head_radius, up, rim)

    _paint_hands(canvas, keypoints, scores, count, kpt_thr=kpt_thr, line_width=line_width)
    _paint_feet(canvas, keypoints, scores, count, kpt_thr=kpt_thr, rim=rim, half=half["shin"])
    _paint_joints(canvas, keypoints, scores, count, kpt_thr=kpt_thr, radius=radius)
    return canvas


def _paint_trunk(canvas, keypoints, scores, *, kpt_thr, half, rim):
    """A filled trunk when all four corners are there, struts and a pelvis bar when not.

    The fallback is the ``openpose-torso`` arrangement rather than OpenPose's own, because
    a partly occluded trunk should still read as a trunk with width and not as two long
    struts through the middle of the body.
    """
    corners = [_point(keypoints, scores, index, kpt_thr)
               for index in (R_SHOULDER, L_SHOULDER, L_HIP, R_HIP)]
    if all(corner is not None for corner in corners):
        return _filled(canvas, corners, TRUNK, rim)

    for shoulder, hip in ((R_SHOULDER, R_HIP), (L_SHOULDER, L_HIP)):
        _rimmed_bone(canvas, keypoints, scores, shoulder, hip, TRUNK,
                     kpt_thr=kpt_thr, half=half, rim=rim)
    _rimmed_bone(canvas, keypoints, scores, R_HIP, L_HIP, TRUNK,
                 kpt_thr=kpt_thr, half=half, rim=rim)
    _rimmed_bone(canvas, keypoints, scores, R_SHOULDER, L_SHOULDER, TRUNK,
                 kpt_thr=kpt_thr, half=half, rim=rim)
    return canvas


def _neck_to_head(canvas, keypoints, scores, centre, *, kpt_thr, half, rim):
    """The neck, drawn to wherever the head was worked out to be rather than to the nose.

    Written against a synthetic pair of keypoints so it can go through the same guarded
    primitive as every other limb instead of a second, unguarded path.
    """
    neck = _point(keypoints, scores, NECK, kpt_thr)
    if neck is None:
        return canvas
    return _rimmed_bone(canvas, (neck, centre), (1.0, 1.0), 0, 1, SPINE,
                        kpt_thr=kpt_thr, half=half, rim=rim)


def _head_circle(keypoints, scores, lat, up, kpt_thr, frame, radius):
    """Where the head is and how big, from whichever measurement is available.

    In order of preference, because each is a better measurement than the next: the jaw
    contour, the ears, the eyes, and failing all three a guess hung off the neck. The jaw
    needs lifting because the dlib-68 contour stops at the brow -- there is no skull point
    in it at all, so its centroid sits well below the middle of a head.
    """
    scale = frame[2] if frame is not None else None
    centre = size = None

    if len(scores) >= FACE_STOP and _confident(scores, kpt_thr, *JAW):
        projected = [_dot((float(keypoints[i][0]), float(keypoints[i][1])), lat) for i in JAW]
        width = max(projected) - min(projected)
        middle = (sum(float(keypoints[i][0]) for i in JAW) / len(JAW),
                  sum(float(keypoints[i][1]) for i in JAW) / len(JAW))
        size = 0.62 * width
        centre = (middle[0] + up[0] * 0.25 * width, middle[1] + up[1] * 0.25 * width)

    if centre is None and _confident(scores, kpt_thr, R_EAR, L_EAR):
        centre = _midpoint(keypoints, scores, R_EAR, L_EAR, kpt_thr)
        size = 0.75 * math.hypot(*_delta(keypoints, L_EAR, R_EAR))

    if centre is None and _confident(scores, kpt_thr, R_EYE, L_EYE):
        eyes = _midpoint(keypoints, scores, R_EYE, L_EYE, kpt_thr)
        size = 1.6 * math.hypot(*_delta(keypoints, L_EYE, R_EYE))
        centre = (eyes[0] + up[0] * 0.4 * size, eyes[1] + up[1] * 0.4 * size)

    if centre is None and scale is not None:
        neck = _point(keypoints, scores, NECK, kpt_thr)
        if neck is None:
            return None
        size = 0.16 * scale
        centre = (neck[0] + up[0] * 1.3 * size, neck[1] + up[1] * 1.3 * size)

    if centre is None or size is None:
        return None

    # Floored twice: against the joint size, so a head is never smaller than the dots
    # around it, and against the torso, so a collapsed ear span at profile cannot shrink it.
    size = max(size, 2.5 * radius, 0.10 * scale if scale else 0.0)
    return centre, max(1, int(round(size)))


def _paint_face(canvas, keypoints, scores, count, centre, head_radius, lat, *,
                kpt_thr, line_width):
    """The features, and only while the subject is facing the camera.

    The estimator will place a full set of them on the back of a head without hesitation --
    they are regressed, not detected -- so this is gated on the facing verdict rather than
    on the confidence of the points themselves.
    """
    # Scaled to the head rather than to the frame. A face is small next to a 1080p frame,
    # so a frame-scaled width comes out at the 2 pixels this exists to stop being.
    width = max(1, min(int(line_width), int(round(head_radius / 12.0))))

    if count < FACE_STOP:
        _paint_small_face(canvas, keypoints, scores, head_radius, kpt_thr=kpt_thr)
        return canvas

    if _face_confidence(scores) < kpt_thr:
        return canvas

    for group, closed in _FACE_GROUPS:
        _chain(canvas, keypoints, scores, group, FACE,
               kpt_thr=kpt_thr, width=width, closed=closed)

    eyes = _eye_colours(keypoints, scores, centre, lat, kpt_thr)
    for group, colour in zip(_EYE_GROUPS, eyes):
        _chain(canvas, keypoints, scores, group, colour,
               kpt_thr=kpt_thr, width=width, closed=True)
    return canvas


def _eye_colours(keypoints, scores, centre, lat, kpt_thr):
    """Which of the two eye contours is the subject's right, worked out rather than assumed.

    The 300W convention says the first group is, and rtmlib's ``swap`` table only proves
    the two are a mirror pair, not which is which. Deriving it from where each sits
    relative to the shoulders costs four lines and is also right for a mirrored source.
    """
    warm, cool = R_UPPER_ARM, L_UPPER_ARM
    if not _confident(scores, kpt_thr, R_SHOULDER, L_SHOULDER):
        return warm, cool                       # the convention, as the last resort

    reference = _dot(_delta(keypoints, R_SHOULDER, L_SHOULDER), lat)
    first = list(_EYE_GROUPS[0])
    middle = (sum(float(keypoints[i][0]) for i in first) / len(first),
              sum(float(keypoints[i][1]) for i in first) / len(first))
    offset = _dot((middle[0] - centre[0], middle[1] - centre[1]), lat)
    return (warm, cool) if offset * reference >= 0 else (cool, warm)


def _paint_small_face(canvas, keypoints, scores, head_radius, *, kpt_thr):
    """What an 18-point layout has: two eyes and a nose, as dots on the head disc.

    Not much, but it is the difference between a head that is looking at you and a head
    that is not, which is the entire question this style exists to answer.
    """
    import cv2

    size = max(2, int(round(head_radius / 6.0)))
    for index, colour in ((R_EYE, R_UPPER_ARM), (L_EYE, L_UPPER_ARM), (NOSE, FACE)):
        spot = _point(keypoints, scores, index, kpt_thr)
        if spot is None:
            continue
        cv2.circle(canvas, (int(round(spot[0])), int(round(spot[1]))), size, colour, -1,
                   lineType=cv2.LINE_AA)
    return canvas


def _paint_crown(canvas, centre, head_radius, up, rim):
    """A hairline arc over the top of the head when the subject is facing away.

    So that "turned away" is something drawn rather than an absence the eye has to notice.
    Over the crown rather than across the nape, which is where this started: a straight
    chord low on a blank disc is read as a mouth, which is the one thing it must not say.
    Placed in the body's frame, so it stays on the crown of a head that is tilted.
    """
    import cv2

    inset = max(2, int(round(head_radius * 0.66)))
    cv2.ellipse(canvas, (int(round(centre[0])), int(round(centre[1]))), (inset, inset),
                math.degrees(math.atan2(up[1], up[0])), -58, 58, RIM,
                thickness=max(1, rim), lineType=cv2.LINE_AA)
    return canvas


def _paint_hands(canvas, keypoints, scores, count, *, kpt_thr, line_width):
    """Five chains off each root, at a width scaled to the hand's own size."""
    if count < R_HAND_ROOT + _HAND_POINTS:
        return canvas

    for root, colour in ((L_HAND_ROOT, L_EXTREMITY), (R_HAND_ROOT, R_EXTREMITY)):
        if float(scores[root]) < kpt_thr:
            continue
        span = math.hypot(*_delta(keypoints, root + _MIDDLE_TIP, root))
        width = max(1, min(int(line_width), int(round(span / 8.0))))
        _chain(canvas, keypoints, scores, (root, root + _PALM[0], root + _PALM[1]),
               colour, kpt_thr=kpt_thr, width=width, closed=True)
        for finger in _FINGERS:
            _chain(canvas, keypoints, scores, (root,) + tuple(root + i for i in finger),
                   colour, kpt_thr=kpt_thr, width=width)
    return canvas


def _paint_feet(canvas, keypoints, scores, count, *, kpt_thr, rim, half):
    """The first style to draw these at all: rtmlib gives them a colour of black.

    A wedge from the heel to the two toes, which is the one part of a 2D skeleton that says
    which way the subject is pointing without any inference at all.
    """
    if count <= R_HEEL:
        return canvas

    feet = ((L_HEEL, L_BIG_TOE, L_SMALL_TOE, L_ANKLE, L_EXTREMITY),
            (R_HEEL, R_BIG_TOE, R_SMALL_TOE, R_ANKLE, R_EXTREMITY))
    for heel, big, small, ankle, colour in feet:
        if not _confident(scores, kpt_thr, heel, big, small):
            continue
        corners = [(float(keypoints[i][0]), float(keypoints[i][1])) for i in (heel, big, small)]
        toe = ((corners[1][0] + corners[2][0]) / 2.0, (corners[1][1] + corners[2][1]) / 2.0)
        foot = _point(keypoints, scores, ankle, kpt_thr)
        if foot is not None:
            _rimmed_bone(canvas, (foot, toe), (1.0, 1.0), 0, 1, colour,
                         kpt_thr=kpt_thr, half=half, rim=rim)
        _filled(canvas, corners, colour, rim)
    return canvas


def _paint_joints(canvas, keypoints, scores, count, *, kpt_thr, radius):
    """Every joint, brightened against the limb it ends, and rimmed like everything else.

    Wrists and ankles come out larger on a layout with no hands or feet to attach: a limb
    that stops at a dot the same size as an elbow reads as an amputation.
    """
    import cv2

    terminal = max(1, int(round(radius * 1.4)))
    has_hands, has_feet = count >= R_HAND_ROOT + _HAND_POINTS, count > R_HEEL
    for index, colour in _JOINTS:
        spot = _point(keypoints, scores, index, kpt_thr)
        if spot is None:
            continue
        size = radius
        if index in (R_WRIST, L_WRIST) and not has_hands:
            size = terminal
        elif index in (R_ANKLE, L_ANKLE) and not has_feet:
            size = terminal
        middle = (int(round(spot[0])), int(round(spot[1])))
        cv2.circle(canvas, middle, size + max(1, radius // 2), RIM, -1, lineType=cv2.LINE_AA)
        cv2.circle(canvas, middle, size, _brighten(colour), -1, lineType=cv2.LINE_AA)
    return canvas
