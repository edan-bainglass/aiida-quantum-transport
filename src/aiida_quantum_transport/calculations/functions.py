from __future__ import annotations

import numpy as np
from aiida import orm
from aiida.engine import calcfunction


@calcfunction
def get_scattering_region(
    device: orm.StructureData,
    x_min: orm.Float | None = None,
    x_max: orm.Float | None = None,
    y_min: orm.Float | None = None,
    y_max: orm.Float | None = None,
) -> orm.ArrayData:
    """docstring"""

    device_positions = device.get_ase().positions

    device_min_z = np.min(device_positions[:, 2])
    device_max_z = np.max(device_positions[:, 2])

    z_average = (device_min_z + device_max_z) / 2

    x, y, z = np.split(device_positions, 3, axis=1)

    x_min = x.min() if x_min is None else x_min
    x_max = x.max() if x_max is None else x_max
    y_min = y.min() if y_min is None else y_min
    y_max = y.max() if y_max is None else y_max

    scattering_region = np.where(
        (z >= z_average)
        & (x >= x_min)
        & (x <= x_max)
        & (y >= y_min)
        & (y <= y_max)
    )[0]

    return orm.ArrayData(scattering_region)
