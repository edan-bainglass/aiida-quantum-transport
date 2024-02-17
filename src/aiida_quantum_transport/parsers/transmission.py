from __future__ import annotations

from pathlib import Path

from aiida import orm
from aiida.engine import ExitCode
from aiida.parsers import Parser


class TransmissionParser(Parser):
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
                path = Path(retrieved_path) / "results" / "transmission_folder"
                self.out("transmission_folder", orm.FolderData(tree=path))
        except OSError:
            return self.exit_codes.ERROR_ACCESSING_OUTPUT_FILE

        return None
