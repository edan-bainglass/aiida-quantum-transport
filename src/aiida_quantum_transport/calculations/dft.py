from __future__ import annotations

import pickle
from typing import TYPE_CHECKING

from aiida import orm
from aiida.common.datastructures import CalcInfo, CodeInfo
from aiida.common.folders import Folder
from aiida.engine import CalcJob

if TYPE_CHECKING:
    from aiida.engine.processes.calcjobs.calcjob import CalcJobProcessSpec


class GpawCalculation(CalcJob):
    """docstring"""

    _default_parser_name = "quantum_transport.gpaw"

    _default_filenames = {
        "log": "gpaw.txt",
        "restart": "gpaw.gpw",
        "hamiltonian": "gpaw.hs.npy",
    }

    @classmethod
    def define(cls, spec: CalcJobProcessSpec) -> None:
        """docstring"""

        super().define(spec)

        spec.input(
            "code",
            valid_type=orm.AbstractCode,
            help="The GPAW script.",
        )

        spec.input(
            "structure",
            valid_type=orm.StructureData,
            help="",  # TODO fill in
        )

        spec.input(
            "kpoints",
            valid_type=orm.KpointsData,
            help="",  # TODO fill in
        )

        spec.input(
            "parameters",
            valid_type=orm.Dict,
            help="",  # TODO fill in
        )

        spec.input(
            "output_filename_prefix",
            valid_type=orm.Str,
            help="A prefix for output files.",
        )

        spec.input(
            "metadata.options.parser_name",
            valid_type=str,
            default=cls._default_parser_name,
        )

        for i, file in enumerate(("log", "restart", "hamiltonian")):
            spec.output(
                f"{file}_file",
                valid_type=orm.SinglefileData,
                help=f"The {file} file",
            )

            spec.exit_code(
                301 + i,
                f"ERROR_MISSING_{file.upper()}_FILE",
                f"Missing {file} file.",
            )

    def prepare_for_submission(self, folder: Folder) -> CalcInfo:
        """docstring"""

        output_filenames = {}
        for key in ("log", "restart", "hamiltonian"):
            prefix: orm.Str = self.inputs.output_filename_prefix
            filename = self._default_filenames[key]
            output_filenames[key] = filename.replace("gpaw", prefix.value)

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
            "--log-filename",
            output_filenames["log"],
            "--restart-filename",
            output_filenames["restart"],
            "--hamiltonian-filename",
            output_filenames["hamiltonian"],
        ]

        calcinfo = CalcInfo()
        calcinfo.codes_info = [codeinfo]
        calcinfo.local_copy_list = []
        calcinfo.retrieve_list = [
            output_filenames["log"],
            output_filenames["restart"],
            output_filenames["hamiltonian"],
        ]

        return calcinfo
