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


class HybridizationCalculation(BaseCalculation):
    """docstring"""

    _default_parser_name = "quantum_transport.hybridize"

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
            "temperature",
            valid_type=orm.Float,
            default=lambda: orm.Float(300.0),
            help="The temperature in Kelvin",
        )

        spec.input(
            "matsubara_grid_size",
            valid_type=orm.Int,
            default=lambda: orm.Int(3000),
            help="The size of the Matsubara energy grid",
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

        spec.input(
            "parameters",
            valid_type=orm.Dict,
            default=lambda: orm.Dict({}),
            help="The parameters used for orbital hybridization",
        )

        spec.output(
            "remote_results_folder",
            valid_type=orm.RemoteData,
        )

        spec.output(
            "hybridization_file",
            valid_type=orm.SinglefileData,
            help="",  # TODO fill in
        )

        spec.output(
            "energies_file",
            valid_type=orm.SinglefileData,
            help="The energies file",
        )

        spec.output(
            "hamiltonian_file",
            valid_type=orm.SinglefileData,
            help="The hamiltonian file",
        )

        spec.output(
            "eigenvalues_file",
            valid_type=orm.SinglefileData,
            help="The eigenvalues file",
        )

        spec.output(
            "matsubara_hybridization_file",
            valid_type=orm.SinglefileData,
            help="The Matsubara hybridization file",
        )

        spec.output(
            "matsubara_energies_file",
            valid_type=orm.SinglefileData,
            help="The Matsubara energies file",
        )

        spec.output(
            "occupancies_file",
            valid_type=orm.SinglefileData,
            help="The occupancies file",
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
                **self.inputs.parameters,
                "temperature": self.inputs.temperature.value,
                "matsubara_grid_size": self.inputs.matsubara_grid_size.value,
            }
            pickle.dump(parameters, file)

        precomputed_input_dir = input_dir / "precomputed"
        (temp_dir / precomputed_input_dir).mkdir()
        los_indices_filepath = (
            precomputed_input_dir / "los_indices.npy"
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

        codeinfo = CodeInfo()
        codeinfo.code_uuid = self.inputs.code.uuid
        codeinfo.cmdline_params = [
            "--parameters-filename",
            parameters_filename,
            "--los-indices-filepath",
            los_indices_filepath,
            "--hamiltonian-ii-filepath",
            hamiltonian_ii_filepath,
            "--hamiltonian-ij-filepath",
            hamiltonian_ij_filepath,
            "--self-energies-filepath",
            self_energies_filepath,
        ]

        los_data: orm.RemoteData = self.inputs.los.remote_results_folder
        greens_function_data: orm.RemoteData = (
            self.inputs.greens_function.remote_results_folder
        )

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
        ]
        calcinfo.retrieve_list = ["results"]

        return calcinfo
