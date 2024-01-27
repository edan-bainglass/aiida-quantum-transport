from __future__ import annotations

import pickle
from typing import TYPE_CHECKING

from aiida import orm
from aiida.common.datastructures import CalcInfo, CodeInfo
from aiida.common.folders import Folder
from aiida.engine import CalcJob

if TYPE_CHECKING:
    from aiida.engine.processes.calcjobs.calcjob import CalcJobProcessSpec


class HybridizationCalculation(CalcJob):
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
            "leads.structure",
            valid_type=orm.StructureData,
            help="The structure of the leads",
        )

        spec.input(
            "leads.kpoints",
            valid_type=orm.KpointsData,
            help="The kpoints mesh used for the leads",
        )

        spec.input(
            "leads.hamiltonian_file",
            valid_type=orm.SinglefileData,
            help="The file holding the leads hamiltonian",
        )

        spec.input(
            "device.structure",
            valid_type=orm.StructureData,
            help="The structure of the device",
        )

        spec.input(
            "localization.hamiltonian_file",
            valid_type=orm.SinglefileData,
            help="The file holding the localized scattering hamiltonian",
        )

        spec.input(
            "localization.index_file",
            valid_type=orm.SinglefileData,
            help="",  # TODO fill in
        )

        spec.input(
            "basis",
            valid_type=orm.Dict,
            help="",  # TODO fill in
        )

        spec.input(
            "parameters",
            valid_type=orm.Dict,
            required=False,
            default=lambda: orm.Dict({}),
            help="parameters used for orbital hybridization",
        )

        spec.input(
            "metadata.options.parser_name",
            valid_type=str,
            default=cls._default_parser_name,
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

    def prepare_for_submission(self, folder: Folder) -> CalcInfo:
        """docstring"""

        pickled_leads_structure_filename = "leads_structure.pkl"
        with folder.open(pickled_leads_structure_filename, "wb") as file:
            leads: orm.StructureData = self.inputs.leads.structure
            pickle.dump(leads.get_ase(), file)

        pickled_device_structure_filename = "device_structure.pkl"
        with folder.open(pickled_device_structure_filename, "wb") as file:
            device: orm.StructureData = self.inputs.device.structure
            pickle.dump(device.get_ase(), file)

        pickled_leads_kpoints_filename = "leads_kpoints.pkl"
        with folder.open(pickled_leads_kpoints_filename, "wb") as file:
            kpoints: orm.KpointsData = self.inputs.leads.kpoints
            pickle.dump(kpoints.get_kpoints_mesh()[0], file)

        pickled_basis_filename = "basis.pkl"
        with folder.open(pickled_basis_filename, "wb") as file:
            basis: orm.Dict = self.inputs.basis
            pickle.dump(basis.get_dict(), file)

        pickled_parameters_filename = "parameters.pkl"
        with folder.open(pickled_parameters_filename, "wb") as file:
            parameters: orm.Dict = self.inputs.parameters
            pickle.dump(parameters.get_dict(), file)

        codeinfo = CodeInfo()
        codeinfo.code_uuid = self.inputs.code.uuid
        codeinfo.cmdline_params = [
            "--leads-structure-filename",
            pickled_leads_structure_filename,
            "--device-structure-filename",
            pickled_device_structure_filename,
            "--leads-kpoints-filename",
            pickled_leads_kpoints_filename,
            "--leads-hamiltonian-filename",
            f"leads_{self.inputs.leads.hamiltonian_file.filename}",
            "--localized-hamiltonian-filename",
            f"device_{self.inputs.localization.hamiltonian_file.filename}",
            "--localization-index-filename",
            self.inputs.localization.index_file.filename,
            "--basis-filename",
            pickled_basis_filename,
            "--parameters-filename",
            pickled_parameters_filename,
        ]

        calcinfo = CalcInfo()
        calcinfo.codes_info = [codeinfo]
        calcinfo.local_copy_list = [
            (
                self.inputs.leads.hamiltonian_file.uuid,
                self.inputs.leads.hamiltonian_file.filename,
                f"leads_{self.inputs.leads.hamiltonian_file.filename}",
            ),
            (
                self.inputs.localization.hamiltonian_file.uuid,
                self.inputs.localization.hamiltonian_file.filename,
                f"device_{self.inputs.localization.hamiltonian_file.filename}",
            ),
            (
                self.inputs.localization.index_file.uuid,
                self.inputs.localization.index_file.filename,
                self.inputs.localization.index_file.filename,
            ),
        ]
        calcinfo.retrieve_list = [
            "hybridization.bin",
            "energies.npy",
            "hamiltonian.npy",
            "eigenvalues.npy",
            "matsubara_hybridization.bin",
            "matsubara_energies.npy",
            "occupancies.npy",
        ]

        return calcinfo
