from __future__ import annotations

from pathlib import Path

from aiida import orm
from aiida.engine import ExitCode
from aiida.parsers import Parser


class DMFTParser(Parser):
    """docstring"""

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
                root = Path(retrieved_path) / "results"

                path = root / "delta_folder"
                self.out("delta_folder", orm.FolderData(tree=path))

                path = root / "sigma_folder"
                self.out("sigma_folder", orm.FolderData(tree=path))

                if self.node.inputs.adjust_mu:
                    path = root / "mu.txt"
                    self.out("mu_file", orm.SinglefileData(path))
        except OSError:
            return self.exit_codes.ERROR_ACCESSING_OUTPUT_FILE

        return None
