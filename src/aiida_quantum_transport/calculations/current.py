from __future__ import annotations

import pickle
from pathlib import Path
from typing import TYPE_CHECKING

from aiida import orm
from aiida.common.datastructures import CalcInfo, CodeInfo
from aiida.common.folders import Folder
from aiida.engine import CalcJob

if TYPE_CHECKING:
    from aiida.engine.processes.calcjobs.calcjob import CalcJobProcessSpec


class CurrentCalculation(CalcJob):
    """docstring"""

    _default_parser_name = "quantum_transport.current"

    @classmethod
    def define(cls, spec: CalcJobProcessSpec) -> None:
        """docstring"""

        super().define(spec)

        spec.input(
            "code",
            valid_type=orm.AbstractCode,
            help="The current script",
        )

        spec.input(
            "parameters",
            valid_type=orm.Dict,
            default=lambda: orm.Dict({}),
            help="The parameters used to compute current",
        )

        spec.input(
            "hybridization.energies_file",
            valid_type=orm.SinglefileData,
            help="file containing computed energies",
        )

        spec.input(
            "transmission.transmission_folder",
            valid_type=orm.FolderData,
            help="folder containing transmission files",
        )

        spec.input(
            "metadata.options.parser_name",
            valid_type=str,
            default=cls._default_parser_name,
        )

        spec.output(
            "current_file",
            valid_type=orm.SinglefileData,
            help="The current data file",
        )

        spec.output(
            "derivative_file",
            valid_type=orm.SinglefileData,
            help="The current derivative data file",
        )

        spec.exit_code(
            400,
            "ERROR_ACCESSING_OUTPUT_FILE",
            "an issue occurred while accessing an expected retrieved file",
        )

    def prepare_for_submission(self, folder: Folder) -> CalcInfo:
        """docstring"""

        temp_dir = Path(folder.abspath)
        input_dir = Path("inputs")
        temp_input_dir = temp_dir / input_dir
        temp_input_dir.mkdir()

        parameters_filename = "parameters.pkl"
        with open(temp_input_dir / parameters_filename, "wb") as file:
            parameters: orm.Dict = self.inputs.parameters
            pickle.dump(parameters.get_dict(), file)

        precomputed_input_dir = input_dir / "precomputed"
        (temp_dir / precomputed_input_dir).mkdir()
        energies_filepath = (precomputed_input_dir / "energies.npy").as_posix()
        transmission_folder_path = (
            precomputed_input_dir / "transmission_folder"
        ).as_posix()

        codeinfo = CodeInfo()
        codeinfo.code_uuid = self.inputs.code.uuid
        codeinfo.cmdline_params = [
            "--parameters-filename",
            parameters_filename,
            "--energies-filepath",
            energies_filepath,
            "--transmission-folder-path",
            transmission_folder_path,
        ]

        calcinfo = CalcInfo()
        calcinfo.codes_info = [codeinfo]
        calcinfo.local_copy_list = [
            (
                self.inputs.hybridization.energies_file.uuid,
                self.inputs.hybridization.energies_file.filename,
                energies_filepath,
            ),
            (
                self.inputs.transmission.transmission_folder.uuid,
                ".",
                transmission_folder_path,
            ),
        ]
        calcinfo.retrieve_list = ["results"]

        return calcinfo
