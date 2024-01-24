from __future__ import annotations

from pathlib import Path

from aiida import orm
from aiida.engine import ExitCode
from aiida.parsers import Parser


class GpawParser(Parser):
    """docstring"""

    _OUTPUT_FILE_MAPPING = {
        ".txt": "log",
        ".gpw": "restart",
        ".hs.npy": "hamiltonian",
    }

    def parse(self, **kwargs) -> ExitCode | None:
        """docstring"""

        prefix: orm.Str = self.node.inputs.output_filename_prefix

        output_files = filter(
            lambda f: prefix.value in f,
            self.retrieved.base.repository.list_object_names(),
        )

        for filename in output_files:
            path = Path(self.node.get_remote_workdir()) / filename
            extension = filename.strip(prefix.value)
            output_label = f"{self._OUTPUT_FILE_MAPPING[extension]}_file"
            self.out(output_label, orm.SinglefileData(path))

        return None
