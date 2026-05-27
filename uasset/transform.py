"""Shared UE transform math utilities.

Provides rotation and transform conversions for UE5 coordinate system.
Used by umap.py (level parsing) and optionally by other modules that
need UE-to-render coordinate conversion.
"""
import math

import numpy as np


def rotator_to_matrix(pitch: float, yaw: float, roll: float) -> np.ndarray:
    """Convert UE FRotator (degrees) to a 3×3 rotation matrix.

    Uses the standard UE rotation order: R = Ry(yaw) · Rx(pitch) · Rz(roll).

    Args:
        pitch: Rotation around Y axis in degrees.
        yaw:   Rotation around Z axis in degrees.
        roll:  Rotation around X axis in degrees.

    Returns:
        3×3 numpy rotation matrix.
    """
    p = math.radians(pitch)
    y = math.radians(yaw)
    r = math.radians(roll)

    cp, sp = math.cos(p), math.sin(p)
    cy, sy = math.cos(y), math.sin(y)
    cr, sr = math.cos(r), math.sin(r)

    R = np.array([
        [cy * cp,  cy * sp * sr - sy * cr,  cy * sp * cr + sy * sr],
        [sy * cp,  sy * sp * sr + cy * cr,  sy * sp * cr - cy * sr],
        [-sp,      cp * sr,                 cp * cr],
    ])
    return R


# ---------------------------------------------------------------------------
# UE → render coordinate conversion matrix
# ---------------------------------------------------------------------------

COORD_CONVERT = np.array([
    [0,  1,  0, 0],
    [0,  0,  1, 0],
    [-1, 0,  0, 0],
    [0,  0,  0, 1]
], dtype=float)


def ue_transform_to_matrix(location: tuple, rotation: tuple, scale: tuple) -> np.ndarray:
    """Build a 4×4 transform matrix from UE transform data.

    Applies the UE rotation and scale, then converts from UE coordinate
    space (left-handed Z-up) to the render coordinate space defined by
    :data:`COORD_CONVERT` (right-handed Y-up).

    Args:
        location: (x, y, z) UE world position.
        rotation: (pitch, yaw, roll) UE rotation in degrees.
        scale:    (x, y, z) UE scale factors.

    Returns:
        4×4 numpy transform matrix in render space.
    """
    R = rotator_to_matrix(*rotation)
    s = np.array([scale[0], scale[1], scale[2]])

    T = np.eye(4, dtype=float)
    T[:3, :3] = R * s[np.newaxis, :]
    T[0, 3] = location[0]
    T[1, 3] = location[1]
    T[2, 3] = location[2]

    return COORD_CONVERT @ T
