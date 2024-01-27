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


class LocalizationCalculation(CalcJob):
    """docstring"""

    _default_parser_name = "quantum_transport.localize"

    @classmethod
    def define(cls, spec: CalcJobProcessSpec) -> None:
        """docstring"""

        super().define(spec)

        spec.input(
            "code",
            valid_type=orm.AbstractCode,
            help="The LOS script",
        )

        spec.input(
            "restart_file",
            valid_type=orm.SinglefileData,
            help="The dft restart file",
        )

        spec.input(
            "scattering.region",
            valid_type=orm.ArrayData,
            required=False,
            default=lambda: orm.ArrayData([]),
            help="The scattering region",
        )

        spec.input(
            "scattering.active",
            valid_type=orm.Dict,
            help="The active species",
        )

        spec.input(
            "lowdin",
            valid_type=orm.Bool,
            default=lambda: orm.Bool(False),
            help="",  # TODO fill in
        )

        spec.input(
            "metadata.options.parser_name",
            valid_type=str,
            default=cls._default_parser_name,
        )

        spec.output(
            "index_file",
            valid_type=orm.SinglefileData,
            help="The localized orbitals index file",
        )

        spec.output(
            "hamiltonian_file",
            valid_type=orm.SinglefileData,
            help="The transformed hamiltonian file",
        )

    def prepare_for_submission(self, folder: Folder) -> CalcInfo:
        """docstring"""

        pickled_scattering_region_filename = "scatt.npy"
        with folder.open(pickled_scattering_region_filename, "wb") as file:
            region: orm.ArrayData = self.inputs.scattering.region
            np.save(file, region.get_array("default"))

        pickled_active_species_filename = "active.pkl"
        with folder.open(pickled_active_species_filename, "wb") as file:
            active: orm.Dict = self.inputs.scattering.active
            pickle.dump(active.get_dict(), file)

        restart_file: orm.SinglefileData = self.inputs.restart_file

        codeinfo = CodeInfo()
        codeinfo.code_uuid = self.inputs.code.uuid
        codeinfo.cmdline_params = [
            "--restart-filename",
            restart_file.filename,
            "--scattering-region-filename",
            pickled_scattering_region_filename,
            "--active-species-filename",
            pickled_active_species_filename,
            "--lowdin",
            str(self.inputs.lowdin.value),
        ]

        calcinfo = CalcInfo()
        calcinfo.codes_info = [codeinfo]
        calcinfo.local_copy_list = [
            (
                restart_file.uuid,
                restart_file.filename,
                restart_file.filename,
            )
        ]
        calcinfo.retrieve_list = ["idx_los.npy", "hs_los.npy"]

        return calcinfo
