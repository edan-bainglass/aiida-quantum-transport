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


class TransmissionCalculation(BaseCalculation):
    """docstring"""

    _default_parser_name = "quantum_transport.transmission"

    @classmethod
    def define(cls, spec: CalcJobProcessSpec) -> None:
        """docstring"""

        super().define(spec)

        spec.input(
            "code",
            valid_type=orm.AbstractCode,
            help="The hybridization script",
        )

        spec.input(
            "los.remote_results_folder",
            valid_type=orm.RemoteData,
            help="The results folder of the local orbitals calculation",
        )

        spec.input(
            "greens_function.remote_results_folder",
            valid_type=orm.RemoteData,
            help="The results folder of the greens function parameters calculation",
        )

        spec.input(
            "dmft.remote_results_folder",
            valid_type=orm.RemoteData,
            help="The results folder of the dmft sweep calculation",
        )

        spec.input(
            "greens_function_parameters",
            valid_type=orm.Dict,
            default=lambda: orm.Dict({}),
            help="The parameters used to define the greens function",
        )

        spec.input(
            "energy_grid_parameters",
            valid_type=orm.Dict,
            default=lambda: orm.Dict({}),
            help="The parameters used to define the energy grid",
        )

        spec.output(
            "remote_results_folder",
            valid_type=orm.RemoteData,
        )

        spec.output(
            "transmission_folder",
            valid_type=orm.FolderData,
            help="The transmission folder",
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
                **self.inputs.greens_function_parameters,
                **self.inputs.energy_grid_parameters,
            }
            pickle.dump(parameters, file)

        precomputed_input_dir = input_dir / "precomputed"
        (temp_dir / precomputed_input_dir).mkdir()
        los_indices_filepath = (
            precomputed_input_dir / "los_indices.npy"
        ).as_posix()
        leads_nao_filepath = (
            precomputed_input_dir / "leads_nao.npy"
        ).as_posix()
        hamiltonian_ii_filepath = (
            precomputed_input_dir / "hamiltonian_ii.pkl"
        ).as_posix()
        hamiltonian_ij_filepath = (
            precomputed_input_dir / "hamiltonian_ij.pkl"
        ).as_posix()
        self_energies_filepath = (
            precomputed_input_dir / "self_energies.pkl"
        ).as_posix()
        sigma_folder_path = (precomputed_input_dir / "sigma_folder").as_posix()

        codeinfo = CodeInfo()
        codeinfo.code_uuid = self.inputs.code.uuid
        codeinfo.cmdline_params = [
            "--parameters-filename",
            parameters_filename,
            "--los-indices-filepath",
            los_indices_filepath,
            "--leads-nao-filepath",
            leads_nao_filepath,
            "--hamiltonian-ii-filepath",
            hamiltonian_ii_filepath,
            "--hamiltonian-ij-filepath",
            hamiltonian_ij_filepath,
            "--self-energies-filepath",
            self_energies_filepath,
            "--sigma-folder-path",
            sigma_folder_path,
        ]

        los_data: orm.RemoteData = self.inputs.los.remote_results_folder
        greens_function_data: orm.RemoteData = (
            self.inputs.greens_function.remote_results_folder
        )
        dmft_data: orm.RemoteData = self.inputs.dmft.remote_results_folder

        calcinfo = CalcInfo()
        calcinfo.codes_info = [codeinfo]
        calcinfo.local_copy_list = []
        calcinfo.remote_symlink_list = [
            (
                los_data.computer.uuid,
                f"{los_data.get_remote_path()}/idx_los.npy",
                los_indices_filepath,
            ),
            (
                greens_function_data.computer.uuid,
                f"{greens_function_data.get_remote_path()}/leads_nao.npy",
                leads_nao_filepath,
            ),
            (
                greens_function_data.computer.uuid,
                f"{greens_function_data.get_remote_path()}/hamiltonian_ii.pkl",
                hamiltonian_ii_filepath,
            ),
            (
                greens_function_data.computer.uuid,
                f"{greens_function_data.get_remote_path()}/hamiltonian_ij.pkl",
                hamiltonian_ij_filepath,
            ),
            (
                greens_function_data.computer.uuid,
                f"{greens_function_data.get_remote_path()}/self_energies.pkl",
                self_energies_filepath,
            ),
            (
                dmft_data.computer.uuid,
                f"{dmft_data.get_remote_path()}/sigma_folder",
                sigma_folder_path,
            ),
        ]
        calcinfo.retrieve_list = ["results"]

        return calcinfo
