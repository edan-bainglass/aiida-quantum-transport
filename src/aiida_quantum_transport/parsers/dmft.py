from __future__ import annotations

from pathlib import Path

from aiida import orm
from aiida.engine import ExitCode
from aiida.parsers import Parser


class DMFTParser(Parser):
    """docstring"""

    def parse(self, **kwargs) -> ExitCode | None:
        """docstring"""

        path = Path(self.node.get_remote_workdir()) / "delta_folder"
        self.out("delta_folder", orm.FolderData(tree=path))

        path = Path(self.node.get_remote_workdir()) / "sigma_folder"
        self.out("sigma_folder", orm.FolderData(tree=path))

        adjust_mu: orm.Bool = self.node.inputs.adjust_mu

        if adjust_mu.value:
            path = Path(self.node.get_remote_workdir()) / "mu.txt"
            self.out("mu_file", orm.SinglefileData(path))

        return None
