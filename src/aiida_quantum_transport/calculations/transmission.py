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


class TransmissionCalculation(CalcJob):
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

        spec.input(
            "parameters",
            valid_type=orm.Dict,
            default=lambda: orm.Dict({}),
            help="The parameters used to compute transmission",
        )

        spec.input(
            "dmft.remote_results_folder",
            valid_type=orm.RemoteData,
            help="The results folder of the dmft sweep calculation",
        )

        spec.input(
            "metadata.options.parser_name",
            valid_type=str,
            default=cls._default_parser_name,
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

        parameters_filename = "parameters.pkl"
        with open(temp_input_dir / parameters_filename, "wb") as file:
            parameters: orm.Dict = self.inputs.parameters
            pickle.dump(parameters.get_dict(), file)

        precomputed_input_dir = input_dir / "precomputed"
        (temp_dir / precomputed_input_dir).mkdir()
        leads_hamiltonian_filepath = (
            precomputed_input_dir / "hs_leads.npy"
        ).as_posix()
        los_hamiltonian_filepath = (
            precomputed_input_dir / "hs_los.npy"
        ).as_posix()
        los_indices_filepath = (
            precomputed_input_dir / "los_indices.npy"
        ).as_posix()
        sigma_folder_path = (precomputed_input_dir / "sigma_folder").as_posix()

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
            "--parameters-filename",
            parameters_filename,
            "--leads-hamiltonian-filepath",
            leads_hamiltonian_filepath,
            "--los-hamiltonian-filepath",
            los_hamiltonian_filepath,
            "--los-index-filepath",
            los_indices_filepath,
            "--sigma-folder-path",
            sigma_folder_path,
        ]

        leads_data: orm.RemoteData = self.inputs.leads.remote_results_folder
        los_data: orm.RemoteData = self.inputs.los.remote_results_folder
        dmft_data: orm.RemoteData = self.inputs.dmft.remote_results_folder

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
            (
                los_data.computer.uuid,
                f"{los_data.get_remote_path()}/idx_los.npy",
                los_indices_filepath,
            ),
            (
                dmft_data.computer.uuid,
                f"{dmft_data.get_remote_path()}/sigma_folder",
                sigma_folder_path,
            ),
        ]
        calcinfo.retrieve_list = ["results"]

        return calcinfo
