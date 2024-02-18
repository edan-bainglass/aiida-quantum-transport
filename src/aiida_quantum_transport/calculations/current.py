from __future__ import annotations

import pickle
from pathlib import Path
from typing import TYPE_CHECKING

from aiida import orm
from aiida.common.datastructures import CalcInfo, CodeInfo
from aiida.common.folders import Folder

from .base import BaseCalculation

if TYPE_CHECKING:
    from aiida.engine.processes.calcjobs.calcjob import CalcJobProcessSpec


class CurrentCalculation(BaseCalculation):
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
            "temperature",
            valid_type=orm.Float,
            default=lambda: orm.Float(300.0),
            help="The temperature in Kelvin",
        )

        spec.input(
            "parameters",
            valid_type=orm.Dict,
            default=lambda: orm.Dict({}),
            help="The parameters used to compute current",
        )

        spec.input(
            "hybridization.remote_results_folder",
            valid_type=orm.RemoteData,
            help="The results folder of the hybridization calculation",
        )

        spec.input(
            "transmission.remote_results_folder",
            valid_type=orm.RemoteData,
            help="The results folder of the transmission calculation",
        )

        spec.output(
            "remote_results_folder",
            valid_type=orm.RemoteData,
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
            parameters = {
                **self.inputs.parameters.get_dict(),
                "temperature": self.inputs.temperature.value,
            }
            pickle.dump(parameters, file)

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

        hybridization_data: orm.RemoteData = (
            self.inputs.hybridization.remote_results_folder
        )
        transmission_data: orm.RemoteData = (
            self.inputs.transmission.remote_results_folder
        )

        calcinfo = CalcInfo()
        calcinfo.codes_info = [codeinfo]
        calcinfo.local_copy_list = []
        calcinfo.remote_symlink_list = [
            (
                hybridization_data.computer.uuid,
                f"{hybridization_data.get_remote_path()}/energies.npy",
                energies_filepath,
            ),
            (
                transmission_data.computer.uuid,
                f"{transmission_data.get_remote_path()}/transmission_folder",
                transmission_folder_path,
            ),
        ]
        calcinfo.retrieve_list = ["results"]

        return calcinfo
