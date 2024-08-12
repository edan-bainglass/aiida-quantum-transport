from __future__ import annotations

from aiida.engine import CalcJob


class BaseCalculation(CalcJob):
    """docstring"""

    _default_parser_name = ""

    @classmethod
    def define(cls, spec) -> None:
        """docstring"""

        super().define(spec)

        _DEFAULTS = {
            "metadata.options.parser_name": cls._default_parser_name,
            "metadata.options.withmpi": False,
            "metadata.options.max_wallclock_seconds": 3600,
            "metadata.options.resources": lambda: {
                "num_machines": 1,
                "num_mpiprocs_per_machine": 1,
                "num_cores_per_mpiproc": 1,
            },
        }

        for port, default in _DEFAULTS.items():
            spec.inputs.get_port(port).default = default
