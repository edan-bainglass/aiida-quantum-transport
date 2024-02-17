from __future__ import annotations

import pickle
from typing import TYPE_CHECKING

from aiida import orm
from aiida.common.datastructures import CalcInfo, CodeInfo
from aiida.common.folders import Folder
from aiida.engine import CalcJob

if TYPE_CHECKING:
    from aiida.engine.processes.calcjobs.calcjob import CalcJobProcessSpec


class DFTCalculation(CalcJob):
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

        spec.input(
            "metadata.options.parser_name",
            valid_type=str,
            default=cls._default_parser_name,
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

        pickled_atoms_filename = "atoms.pkl"
        with folder.open(pickled_atoms_filename, "wb") as file:
            structure: orm.StructureData = self.inputs.structure
            pickle.dump(structure.get_ase(), file)

        pickled_kpoints_filename = "kpoints.pkl"
        with folder.open(pickled_kpoints_filename, "wb") as file:
            kpoints: orm.KpointsData = self.inputs.kpoints
            pickle.dump(kpoints.get_kpoints_mesh()[0], file)

        pickled_parameters_filename = "parameters.pkl"
        with folder.open(pickled_parameters_filename, "wb") as file:
            parameters: orm.Dict = self.inputs.parameters
            pickle.dump(parameters.get_dict(), file)

        codeinfo = CodeInfo()
        codeinfo.code_uuid = self.inputs.code.uuid
        codeinfo.cmdline_params = [
            "--structure-filename",
            pickled_atoms_filename,
            "--kpoints-filename",
            pickled_kpoints_filename,
            "--parameters-filename",
            pickled_parameters_filename,
        ]

        calcinfo = CalcInfo()
        calcinfo.codes_info = [codeinfo]
        calcinfo.local_copy_list = []
        calcinfo.retrieve_list = ["log.txt", "restart.gpw", "hs.npy"]

        return calcinfo
