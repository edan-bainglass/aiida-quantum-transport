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
            with self.retrieved.as_path() as retrieved_path:
                path = Path(retrieved_path) / "transmission_folder"
                self.out("transmission_folder", orm.FolderData(tree=path))
        except OSError:
            return self.exit_codes.ERROR_ACCESSING_OUTPUT_FILE

        return None
