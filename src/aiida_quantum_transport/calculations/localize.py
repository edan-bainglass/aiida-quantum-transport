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


class LocalizationCalculation(BaseCalculation):
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
            "device.remote_results_folder",
            valid_type=orm.RemoteData,
            help="The results folder of the device dft calculation",
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
            "lowdin",
            valid_type=orm.Bool,
            default=lambda: orm.Bool(False),
            help="",  # TODO fill in
        )

        spec.output(
            "remote_results_folder",
            valid_type=orm.RemoteData,
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

        active_species_filename = "active.pkl"
        with open(temp_input_dir / active_species_filename, "wb") as file:
            active: orm.Dict = self.inputs.scattering.active
            pickle.dump(active.get_dict(), file)

        precomputed_input_dir = input_dir / "precomputed"
        temp_precomputed_input_dir = temp_dir / precomputed_input_dir
        temp_precomputed_input_dir.mkdir()
        restart_filename = "restart.gpw"
        restart_filepath = (
            precomputed_input_dir / f"device_{restart_filename}"
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
            "--restart-filepath",
            restart_filepath,
            "--active-species-filename",
            active_species_filename,
            "--scattering-region-filepath",
            scattering_region_filepath,
        ]

        if self.inputs.lowdin:
            codeinfo.cmdline_params.append("--lowdin")

        calcinfo = CalcInfo()
        calcinfo.codes_info = [codeinfo]
        calcinfo.local_copy_list = []
        calcinfo.remote_symlink_list = [
            (
                self.inputs.device.remote_results_folder.computer.uuid,
                (
                    Path(
                        self.inputs.device.remote_results_folder.get_remote_path()
                    )
                    / restart_filename
                ).as_posix(),
                restart_filepath,
            )
        ]
        calcinfo.retrieve_list = ["results"]

        return calcinfo
