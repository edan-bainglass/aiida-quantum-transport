from __future__ import annotations

from pathlib import Path

from aiida import orm
from aiida.engine import ExitCode
from aiida.parsers import Parser


class LocalizationParser(Parser):
    """docstring"""

    _OUTPUT_FILE_MAPPING = {
        "index": "idx_los.npy",
        "hamiltonian": "hs_los.npy",
    }

    def parse(self, **kwargs) -> ExitCode | None:
        """docstring"""

        try:
            with self.retrieved.as_path() as retrieved_path:
                for label, filename in self._OUTPUT_FILE_MAPPING.items():
                    path = Path(retrieved_path) / filename
                    self.out(f"{label}_file", orm.SinglefileData(path))
        except OSError:
            return self.exit_codes.ERROR_ACCESSING_OUTPUT_FILE

        return None
