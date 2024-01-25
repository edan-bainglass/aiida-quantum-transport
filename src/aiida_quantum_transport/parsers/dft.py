from __future__ import annotations

from pathlib import Path

from aiida import orm
from aiida.engine import ExitCode
from aiida.parsers import Parser


class DFTParser(Parser):
    """docstring"""

    _OUTPUT_FILE_MAPPING = {
        "log": "log.txt",
        "restart": "restart.gpw",
        "hamiltonian": "hs.npy",
    }

    def parse(self, **kwargs) -> ExitCode | None:
        """docstring"""

        for label, filename in self._OUTPUT_FILE_MAPPING.items():
            path = Path(self.node.get_remote_workdir()) / filename
            output_label = f"{label}_file"
            self.out(output_label, orm.SinglefileData(path))

        return None
