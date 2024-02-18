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


class DFTCalculation(BaseCalculation):
    """docstring"""

    _default_parser_name = "quantum_transport.dft"

    @classmethod
    def define(cls, spec: CalcJobProcessSpec) -> None:
        """docstring"""

        super().define(spec)

        spec.input(
            "code",
            valid_type=orm.AbstractCode,
            help="The DFT script",
        )

        spec.input(
            "structure",
            valid_type=orm.StructureData,
            help="The structure of interest",
        )

        spec.input(
            "kpoints",
            valid_type=orm.KpointsData,
            help="The kpoints mesh",
        )

        spec.input(
            "parameters",
            valid_type=orm.Dict,
            help="The input parameters",
        )

        spec.output(
            "remote_results_folder",
            valid_type=orm.RemoteData,
        )

        for file in ("log", "restart", "hamiltonian"):
            spec.output(
                f"{file}_file",
                valid_type=orm.SinglefileData,
                help=f"The {file} file",
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

        atoms_filename = "atoms.pkl"
        with open(temp_input_dir / atoms_filename, "wb") as file:
            structure: orm.StructureData = self.inputs.structure
            pickle.dump(structure.get_ase(), file)

        kpoints_filename = "kpoints.pkl"
        with open(temp_input_dir / kpoints_filename, "wb") as file:
            kpoints: orm.KpointsData = self.inputs.kpoints
            pickle.dump(kpoints.get_kpoints_mesh()[0], file)

        parameters_filename = "parameters.pkl"
        with open(temp_input_dir / parameters_filename, "wb") as file:
            parameters: orm.Dict = self.inputs.parameters
            pickle.dump(parameters.get_dict(), file)

        codeinfo = CodeInfo()
        codeinfo.code_uuid = self.inputs.code.uuid
        codeinfo.cmdline_params = [
            "--structure-filename",
            atoms_filename,
            "--kpoints-filename",
            kpoints_filename,
            "--parameters-filename",
            parameters_filename,
        ]

        calcinfo = CalcInfo()
        calcinfo.codes_info = [codeinfo]
        calcinfo.local_copy_list = []
        calcinfo.retrieve_list = ["results"]

        return calcinfo
