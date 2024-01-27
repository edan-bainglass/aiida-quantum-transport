from __future__ import annotations

from pathlib import Path

from aiida import orm
from aiida.engine import ExitCode
from aiida.parsers import Parser


class HybridizationParser(Parser):
    """docstring"""

    _OUTPUT_FILE_LIST = [
        "hybridization.bin",
        "energies.npy",
        "hamiltonian.npy",
        "eigenvalues.npy",
        "matsubara_hybridization.bin",
        "matsubara_energies.npy",
        "occupancies.npy",
    ]

    def parse(self, **kwargs) -> ExitCode | None:
        """docstring"""

        for filename in self._OUTPUT_FILE_LIST:
            path = Path(self.node.get_remote_workdir()) / filename
            prefix = filename.split(".")[0]
            output_label = f"{prefix}_file"
            self.out(output_label, orm.SinglefileData(path))

        return None
