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

        try:
            self.out(
                "remote_results_folder",
                orm.RemoteData(
                    f"{self.node.get_remote_workdir()}/results",
                    computer=self.node.computer,
                ),
            )
            with self.retrieved.as_path() as retrieved_path:
                results_dir = Path(retrieved_path / "results")
                for label, filename in self._OUTPUT_FILE_MAPPING.items():
                    path = results_dir / filename
                    self.out(f"{label}_file", orm.SinglefileData(path))
        except OSError:
            return self.exit_codes.ERROR_ACCESSING_OUTPUT_FILE

        return None
