#!/usr/bin/env python

from __future__ import annotations

import pickle
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
from ase.atoms import Atoms
from gpaw import GPAW
from gpaw.lcao.tools import get_lcao_hamiltonian
from gpaw.mpi import rank


def run_gpaw(
    structure: Atoms,
    kpoints: np.ndarray,
    parameters: dict,
) -> None:
    """docstring"""

    output_dir = Path("results")
    output_dir.mkdir(exist_ok=True)

    calc = GPAW(kpts=kpoints, txt=output_dir / "log.txt", **parameters)

    structure.set_calculator(calc)
    structure.get_potential_energy()
    calc.write(f"{output_dir}/restart.gpw")

    fermi = calc.get_fermi_level()

    with open(output_dir / "fermi.txt", "w") as file:
        file.write(repr(fermi))

    H_skMM, S_kMM = get_lcao_hamiltonian(calc)

    if rank == 0:
        H_kMM = H_skMM[0]
        H_kMM -= fermi * S_kMM
        np.save(output_dir / "hs.npy", (H_kMM, S_kMM))


if __name__ == "__main__":
    """docstring"""

    parser = ArgumentParser()

    parser.add_argument(
        "-sf",
        "--structure-filename",
        help="name of pickled structure file",
    )

    parser.add_argument(
        "-kf",
        "--kpoints-filename",
        help="name of pickled kpoints file",
    )

    parser.add_argument(
        "-pf",
        "--parameters-filename",
        help="name of pickled parameters file",
    )

    args = parser.parse_args()

    input_dir = Path("inputs")

    with open(input_dir / args.structure_filename, "rb") as file:
        structure = pickle.load(file)

    with open(input_dir / args.kpoints_filename, "rb") as file:
        kpoints = pickle.load(file)

    with open(input_dir / args.parameters_filename, "rb") as file:
        parameters = pickle.load(file)

    run_gpaw(structure, kpoints, parameters)
