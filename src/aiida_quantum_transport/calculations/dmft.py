from __future__ import annotations

import pickle
from typing import TYPE_CHECKING

import numpy as np
from aiida import orm
from aiida.common.datastructures import CalcInfo, CodeInfo
from aiida.common.folders import Folder
from aiida.engine import CalcJob

if TYPE_CHECKING:
    from aiida.engine.processes.calcjobs.calcjob import CalcJobProcessSpec


class DMFTCalculation(CalcJob):
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
            "mu_file",
            valid_type=orm.SinglefileData,
            required=False,
            help="The converged chemical potential file",
        )

        spec.input(
            "hybridization.energies_file",
            valid_type=orm.SinglefileData,
            help="",  # TODO fill in
        )

        spec.input(
            "hybridization.matsubara_energies_file",
            valid_type=orm.SinglefileData,
            help="",  # TODO fill in
        )

        spec.input(
            "hybridization.matsubara_hybridization_file",
            valid_type=orm.SinglefileData,
            help="",  # TODO fill in
        )

        spec.input(
            "hybridization.hamiltonian_file",
            valid_type=orm.SinglefileData,
            help="",  # TODO fill in
        )

        spec.input(
            "hybridization.occupancies_file",
            valid_type=orm.SinglefileData,
            help="",  # TODO fill in
        )

        spec.input(
            "metadata.options.parser_name",
            valid_type=str,
            default=cls._default_parser_name,
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

        pickled_device_structure_filename = "device_structure.pkl"
        with folder.open(pickled_device_structure_filename, "wb") as file:
            device: orm.StructureData = self.inputs.device.structure
            pickle.dump(device.get_ase(), file)

        pickled_scattering_region_filename = "scatt.npy"
        with folder.open(pickled_scattering_region_filename, "wb") as file:
            region: orm.ArrayData = self.inputs.scattering.region
            np.save(file, region.get_array("default"))

        pickled_active_species_filename = "active.pkl"
        with folder.open(pickled_active_species_filename, "wb") as file:
            active: orm.Dict = self.inputs.scattering.active
            pickle.dump(active.get_dict(), file)

        pickled_parameters_filename = "parameters.pkl"
        with folder.open(pickled_parameters_filename, "wb") as file:
            parameters: orm.Dict = self.inputs.parameters
            pickle.dump(parameters.get_dict(), file)

        pickled_sweep_paramters_filename = "sweep_parameters.pkl"
        with folder.open(pickled_sweep_paramters_filename, "wb") as file:
            sweep_parameters: orm.Dict = self.inputs.sweep.parameters
            pickle.dump(sweep_parameters.get_dict(), file)

        codeinfo = CodeInfo()
        codeinfo.code_uuid = self.inputs.code.uuid
        codeinfo.cmdline_params = [
            "--device-structure-filename",
            pickled_device_structure_filename,
            "--scattering-region-filename",
            pickled_scattering_region_filename,
            "--active-species-filename",
            pickled_active_species_filename,
            "--adjust-mu",
            str(self.inputs.adjust_mu.value),
            "--parameters-filename",
            pickled_parameters_filename,
            "--sweep-parameters-filename",
            pickled_sweep_paramters_filename,
            "--energies-filename",
            self.inputs.hybridization.energies_file.filename,
            "--matsubara-energies-filename",
            self.inputs.hybridization.matsubara_energies_file.filename,
            "--matsubara-hybridization-filename",
            self.inputs.hybridization.matsubara_hybridization_file.filename,
            "--hamiltonian-filename",
            self.inputs.hybridization.hamiltonian_file.filename,
            "--occupancies-filename",
            self.inputs.hybridization.occupancies_file.filename,
        ]

        calcinfo = CalcInfo()
        calcinfo.codes_info = [codeinfo]
        calcinfo.local_copy_list = [
            (
                self.inputs.hybridization.energies_file.uuid,
                self.inputs.hybridization.energies_file.filename,
                self.inputs.hybridization.energies_file.filename,
            ),
            (
                self.inputs.hybridization.matsubara_energies_file.uuid,
                self.inputs.hybridization.matsubara_energies_file.filename,
                self.inputs.hybridization.matsubara_energies_file.filename,
            ),
            (
                self.inputs.hybridization.matsubara_hybridization_file.uuid,
                self.inputs.hybridization.matsubara_hybridization_file.filename,
                self.inputs.hybridization.matsubara_hybridization_file.filename,
            ),
            (
                self.inputs.hybridization.hamiltonian_file.uuid,
                self.inputs.hybridization.hamiltonian_file.filename,
                self.inputs.hybridization.hamiltonian_file.filename,
            ),
            (
                self.inputs.hybridization.occupancies_file.uuid,
                self.inputs.hybridization.occupancies_file.filename,
                self.inputs.hybridization.occupancies_file.filename,
            ),
        ]
        calcinfo.retrieve_list = [
            "delta_folder",
            "sigma_folder",
        ]

        if self.inputs.adjust_mu.value:
            calcinfo.retrieve_list.append("mu.txt")
        else:
            codeinfo.cmdline_params.extend(
                (
                    "--mu-filename",
                    self.inputs.mu_file.filename,
                ),
            )
            calcinfo.local_copy_list.append(
                (
                    self.inputs.mu_file.uuid,
                    self.inputs.mu_file.filename,
                    self.inputs.mu_file.filename,
                )
            )

        return calcinfo
