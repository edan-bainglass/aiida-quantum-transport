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
            with self.retrieved.as_path() as retrieved_path:
                root = Path(retrieved_path)

                path = root / "delta_folder"
                self.out("delta_folder", orm.FolderData(tree=path))

                path = root / "sigma_folder"
                self.out("sigma_folder", orm.FolderData(tree=path))

                adjust_mu: orm.Bool = self.node.inputs.adjust_mu

                if adjust_mu.value:
                    path = root / "mu.txt"
                    self.out("mu_file", orm.SinglefileData(path))
        except OSError:
            return self.exit_codes.ERROR_ACCESSING_OUTPUT_FILE

        return None
