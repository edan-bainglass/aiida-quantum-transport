from __future__ import annotations

from pathlib import Path

from aiida import orm
from aiida.engine import ExitCode
from aiida.parsers import Parser


class TransmissionParser(Parser):
    """docstring"""

    def parse(self, **kwargs) -> ExitCode | None:
        """docstring"""

        path = Path(self.node.get_remote_workdir()) / "transmission_folder"
        self.out("transmission_folder", orm.FolderData(tree=path))

        return None
