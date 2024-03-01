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


class GreensFunctionParametersCalculation(BaseCalculation):
    """docstring"""

    _default_parser_name = "quantum_transport.greens"

    @classmethod
    def define(cls, spec: CalcJobProcessSpec) -> None:
        """docstring"""

        super().define(spec)

        spec.input(
            "code",
            valid_type=orm.AbstractCode,
            help="The greens-function parameters script",
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
            "leads.remote_results_folder",
            valid_type=orm.RemoteData,
            help="The results folder of the leads dft calculation",
        )

        spec.input(
            "device.structure",
            valid_type=orm.StructureData,
            help="The structure of the device",
        )

        spec.input(
            "los.remote_results_folder",
            valid_type=orm.RemoteData,
            help="The results folder of the local orbitals calculation",
        )

        spec.input(
            "basis",
            valid_type=orm.Dict,
            help="",  # TODO fill in
        )

        spec.output(
            "remote_results_folder",
            valid_type=orm.RemoteData,
        )

        spec.output(
            "leads_nao_file",
            valid_type=orm.SinglefileData,
            help="The pickled leads natural orbitals file",
        )

        spec.output(
            "hamiltonian_ii_file",
            valid_type=orm.SinglefileData,
            help="The pickled diagonal hamiltonian elements file",
        )

        spec.output(
            "hamiltonian_ij_file",
            valid_type=orm.SinglefileData,
            help="The pickled off-diagonal hamiltonian elements file",
        )

        spec.output(
            "self_energies_file",
            valid_type=orm.SinglefileData,
            help="The pickled self-energies file",
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

        leads_structure_filename = "leads_structure.pkl"
        with open(temp_input_dir / leads_structure_filename, "wb") as file:
            leads: orm.StructureData = self.inputs.leads.structure
            pickle.dump(leads.get_ase(), file)

        device_structure_filename = "device_structure.pkl"
        with open(temp_input_dir / device_structure_filename, "wb") as file:
            device: orm.StructureData = self.inputs.device.structure
            pickle.dump(device.get_ase(), file)

        leads_kpoints_filename = "leads_kpoints.pkl"
        with open(temp_input_dir / leads_kpoints_filename, "wb") as file:
            kpoints: orm.KpointsData = self.inputs.leads.kpoints
            pickle.dump(kpoints.get_kpoints_mesh()[0], file)

        basis_filename = "basis.pkl"
        with open(temp_input_dir / basis_filename, "wb") as file:
            basis: orm.Dict = self.inputs.basis
            pickle.dump(basis.get_dict(), file)

        precomputed_input_dir = input_dir / "precomputed"
        (temp_dir / precomputed_input_dir).mkdir()
        leads_hamiltonian_filepath = (
            precomputed_input_dir / "hs_leads.npy"
        ).as_posix()
        los_hamiltonian_filepath = (
            precomputed_input_dir / "hs_los.npy"
        ).as_posix()

        codeinfo = CodeInfo()
        codeinfo.code_uuid = self.inputs.code.uuid
        codeinfo.cmdline_params = [
            "--leads-structure-filename",
            leads_structure_filename,
            "--device-structure-filename",
            device_structure_filename,
            "--leads-kpoints-filename",
            leads_kpoints_filename,
            "--basis-filename",
            basis_filename,
            "--leads-hamiltonian-filepath",
            leads_hamiltonian_filepath,
            "--los-hamiltonian-filepath",
            los_hamiltonian_filepath,
        ]

        leads_data: orm.RemoteData = self.inputs.leads.remote_results_folder
        los_data: orm.RemoteData = self.inputs.los.remote_results_folder

        calcinfo = CalcInfo()
        calcinfo.codes_info = [codeinfo]
        calcinfo.local_copy_list = []
        calcinfo.remote_symlink_list = [
            (
                leads_data.computer.uuid,
                f"{leads_data.get_remote_path()}/hs.npy",
                leads_hamiltonian_filepath,
            ),
            (
                los_data.computer.uuid,
                f"{los_data.get_remote_path()}/hs_los.npy",
                los_hamiltonian_filepath,
            ),
        ]
        calcinfo.retrieve_list = ["results"]

        return calcinfo
