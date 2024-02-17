from __future__ import annotations

from pathlib import Path

from aiida import orm
from aiida.engine import ExitCode
from aiida.parsers import Parser


class GreensFunctionParametersParser(Parser):
    """docstring"""

    _OUTPUT_FILES = [
        "leads_nao.npy",
        "hamiltonian_ii.pkl",
        "hamiltonian_ij.pkl",
        "self_energies.pkl",
    ]

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
                results_dir = Path(retrieved_path) / "results"
                for filename in self._OUTPUT_FILES:
                    path = results_dir / filename
                    prefix = filename.split(".")[0]
                    self.out(f"{prefix}_file", orm.SinglefileData(path))
        except OSError:
            return self.exit_codes.ERROR_ACCESSING_OUTPUT_FILE

        return None
