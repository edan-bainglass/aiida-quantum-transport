#!/usr/bin/env python

import pickle
from argparse import ArgumentParser

import numpy as np
from ase.atoms import Atoms
from gpaw import GPAW
from gpaw.lcao.tools import get_lcao_hamiltonian
from gpaw.mpi import rank


def run_gpaw(
    structure: Atoms,
    kpoints: np.ndarray,
    parameters: dict,
    write_nao=False,
) -> None:
    """docstring"""

    calc = GPAW(kpts=kpoints, txt="log.txt", **parameters)

    structure.set_calculator(calc)
    structure.get_potential_energy()
    calc.write("restart.gpw")

    fermi = calc.get_fermi_level()

    with open("fermi.txt", "w") as file:
        file.write(repr(fermi))

    H_skMM, S_kMM = get_lcao_hamiltonian(calc)

    if rank == 0:
        H_kMM = H_skMM[0]
        H_kMM -= fermi * S_kMM
        np.save("hs.npy", (H_kMM, S_kMM))

    if write_nao:
        np.save("nao.npy", [setup.nao for setup in calc.wfs.setups])


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

    parser.add_argument(
        "-wn",
        "--write-nao",
        help="if naos should be written to file",
    )

    args = parser.parse_args()

    with open(args.structure_filename, "rb") as file:
        structure = pickle.load(file)

    with open(args.kpoints_filename, "rb") as file:
        kpoints = pickle.load(file)

    with open(args.parameters_filename, "rb") as file:
        parameters = pickle.load(file)

    run_gpaw(structure, kpoints, parameters, args.write_nao)
