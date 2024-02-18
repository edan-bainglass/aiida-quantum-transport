from __future__ import annotations

import pickle
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from aiida import orm
from aiida.common.datastructures import CalcInfo, CodeInfo
from aiida.common.folders import Folder

from .base import BaseCalculation

if TYPE_CHECKING:
    from aiida.engine.processes.calcjobs.calcjob import CalcJobProcessSpec


class DMFTCalculation(BaseCalculation):
    """docstring"""

    _default_parser_name = "quantum_transport.dmft"

    @classmethod
    def define(cls, spec: CalcJobProcessSpec) -> None:
        """docstring"""

        super().define(spec)

        spec.input(
            "code",
            valid_type=orm.AbstractCode,
            help="The DMFT script",
        )

        spec.input(
            "device.structure",
            valid_type=orm.StructureData,
            help="The structure of the device",
        )

        spec.input(
            "scattering.region",
            valid_type=orm.ArrayData,
            help="The scattering region",
        )

        spec.input(
            "scattering.active",
            valid_type=orm.Dict,
            help="The active species",
        )

        spec.input(
            "adjust_mu",
            valid_type=orm.Bool,
            default=lambda: orm.Bool(False),
            help="True if the chemical potential is to be adjusted",
        )

        spec.input(
            "parameters",
            valid_type=orm.Dict,
            default=lambda: orm.Dict({}),
            help="DMFT parameters",
        )

        spec.input(
            "sweep.parameters",
            valid_type=orm.Dict,
            default=lambda: orm.Dict({}),
            help="The chemical potential sweep parameters",
        )

        spec.input(
            "hybridization.remote_results_folder",
            valid_type=orm.RemoteData,
            help="The results folder of the hybridization calculation",
        )

        spec.input(
            "hybridization.remote_results_folder",
            valid_type=orm.RemoteData,
            help="The results folder of the hybridization calculation",
        )

        spec.input(
            "mu_file",
            valid_type=orm.SinglefileData,
            required=False,
            help="The converged chemical potential file",
        )

        spec.output(
            "remote_results_folder",
            valid_type=orm.RemoteData,
        )

        spec.output(
            "mu_file",
            valid_type=orm.SinglefileData,
            required=False,
            help="The converged chemical potential file",
        )

        spec.output(
            "delta_folder",
            valid_type=orm.FolderData,
            help="The delta folder",
        )

        spec.output(
            "sigma_folder",
            valid_type=orm.FolderData,
            help="The sigma folder",
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

        device_structure_filename = "device_structure.pkl"
        with open(temp_input_dir / device_structure_filename, "wb") as file:
            device: orm.StructureData = self.inputs.device.structure
            pickle.dump(device.get_ase(), file)

        active_species_filename = "active.pkl"
        with open(temp_input_dir / active_species_filename, "wb") as file:
            active: orm.Dict = self.inputs.scattering.active
            pickle.dump(active.get_dict(), file)

        parameters_filename = "parameters.pkl"
        with open(temp_input_dir / parameters_filename, "wb") as file:
            parameters: orm.Dict = self.inputs.parameters
            pickle.dump(parameters.get_dict(), file)

        sweep_paramters_filename = "sweep_parameters.pkl"
        with open(temp_input_dir / sweep_paramters_filename, "wb") as file:
            sweep_parameters: orm.Dict = self.inputs.sweep.parameters
            pickle.dump(sweep_parameters.get_dict(), file)

        precomputed_input_dir = input_dir / "precomputed"
        temp_precomputed_input_dir = temp_dir / precomputed_input_dir
        temp_precomputed_input_dir.mkdir()
        energies_filepath = (precomputed_input_dir / "energies.npy").as_posix()
        matsubara_energies_filepath = (
            precomputed_input_dir / "matsubara_energies.npy"
        ).as_posix()
        matsubara_hybridization_filepath = (
            precomputed_input_dir / "matsubara_hybridization.bin"
        ).as_posix()
        hamiltonian_filepath = (
            precomputed_input_dir / "hamiltonian.npy"
        ).as_posix()
        occupancies_filepath = (
            precomputed_input_dir / "occupancies.npy"
        ).as_posix()
        scattering_region_filename = "scatt.npy"
        scattering_region_filepath = (
            precomputed_input_dir / scattering_region_filename
        ).as_posix()

        region: orm.ArrayData = self.inputs.scattering.region
        np.save(
            temp_precomputed_input_dir / scattering_region_filename,
            region.get_array("default"),
        )

        codeinfo = CodeInfo()
        codeinfo.code_uuid = self.inputs.code.uuid
        codeinfo.cmdline_params = [
            "--device-structure-filename",
            device_structure_filename,
            "--active-species-filename",
            active_species_filename,
            "--parameters-filename",
            parameters_filename,
            "--sweep-parameters-filename",
            sweep_paramters_filename,
            "--scattering-region-filepath",
            scattering_region_filepath,
            "--energies-filepath",
            energies_filepath,
            "--matsubara-energies-filepath",
            matsubara_energies_filepath,
            "--matsubara-hybridization-filepath",
            matsubara_hybridization_filepath,
            "--hamiltonian-filepath",
            hamiltonian_filepath,
            "--occupancies-filepath",
            occupancies_filepath,
        ]

        hybridization_data: orm.RemoteData = (
            self.inputs.hybridization.remote_results_folder
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
                hybridization_data.computer.uuid,
                f"{hybridization_data.get_remote_path()}/matsubara_energies.npy",
                matsubara_energies_filepath,
            ),
            (
                hybridization_data.computer.uuid,
                f"{hybridization_data.get_remote_path()}/matsubara_hybridization.bin",
                matsubara_hybridization_filepath,
            ),
            (
                hybridization_data.computer.uuid,
                f"{hybridization_data.get_remote_path()}/hamiltonian.npy",
                hamiltonian_filepath,
            ),
            (
                hybridization_data.computer.uuid,
                f"{hybridization_data.get_remote_path()}/occupancies.npy",
                occupancies_filepath,
            ),
        ]
        calcinfo.retrieve_list = ["results"]

        if self.inputs.adjust_mu:
            codeinfo.cmdline_params.append("--adjust-mu")
        else:
            mu_filename: str = self.inputs.mu_file.filename
            mu_filepath = (precomputed_input_dir / mu_filename).as_posix()
            codeinfo.cmdline_params.extend(
                (
                    "--mu-filepath",
                    mu_filepath,
                ),
            )
            calcinfo.local_copy_list.append(
                (
                    self.inputs.mu_file.uuid,
                    self.inputs.mu_file.filename,
                    mu_filepath,
                )
            )

        return calcinfo
