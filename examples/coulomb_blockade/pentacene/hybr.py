#!/usr/bin/env python

from __future__ import annotations

import pickle
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
from ase.units import kB
from qtpyt.block_tridiag import greenfunction
from qtpyt.continued_fraction import get_ao_charge
from qtpyt.hybridization import Hybridization
from qtpyt.parallel import comm
from qtpyt.parallel.egrid import GridDesc
from qtpyt.projector import ProjectedGreenFunction
from scipy.linalg import eigvalsh


def hybridize_orbitals(
    los_indices: np.ndarray,
    hs_list_ii,
    hs_list_ij,
    self_energies,
    temperature=300.0,
    solver="dyson",
    eta=1e-4,
    E_min=-3.0,
    E_max=3.0,
    E_step=1e-2,
    matsubara_grid_size=3000,
) -> None:
    """docstring"""

    output_dir = Path("results")
    output_dir.mkdir(exist_ok=True)

    gf = greenfunction.GreenFunction(
        hs_list_ii,
        hs_list_ij,
        self_energies,
        solver=solver,
        eta=eta,
    )

    gfp = ProjectedGreenFunction(gf, los_indices)
    hyb = Hybridization(gfp)

    energies = np.linspace(E_min, E_max, int((E_max - E_min) / E_step) + 1)

    no = len(los_indices)
    gd = GridDesc(energies, no, complex)
    HB = gd.empty_aligned_orbs()
    D = np.empty(gd.energies.size)

    for e, energy in enumerate(gd.energies):
        HB[e] = hyb.retarded(energy)
        D[e] = gfp.get_dos(energy)

    D = gd.gather_energies(D)
    gd.write(HB, f"{output_dir}/hybridization.bin")

    Heff = (hyb.H + hyb.retarded(0.0)).real

    if comm.rank == 0:
        np.save(output_dir / "partial_dos.npy", D.real)
        np.save(output_dir / "energies.npy", energies + 1.0j * eta)
        np.save(output_dir / "hamiltonian.npy", hyb.H)
        np.save(output_dir / "hamiltonian_effective.npy", Heff)
        np.save(output_dir / "eigenvalues.npy", eigvalsh(Heff, gfp.S))

    # Matsubara
    gf.eta = 0.0
    beta = 1 / (kB * temperature)
    matsubara_energies = (
        1.0j * (2 * np.arange(matsubara_grid_size) + 1) * np.pi / beta
    )
    gd = GridDesc(matsubara_energies, no, complex)
    HB = gd.empty_aligned_orbs()

    for e, energy in enumerate(gd.energies):
        HB[e] = hyb.retarded(energy)

    gd.write(HB, f"{output_dir}/matsubara_hybridization.bin")

    if comm.rank == 0:
        np.save(output_dir / "occupancies.npy", get_ao_charge(gfp))
        np.save(output_dir / "matsubara_energies.npy", matsubara_energies)


if __name__ == "__main__":
    """docstring"""

    parser = ArgumentParser()

    parser.add_argument(
        "-pf",
        "--parameters-filename",
        help="name of parameters file",
    )

    parser.add_argument(
        "-loif",
        "--los-indices-filepath",
        help="path to local orbitals index file",
    )

    parser.add_argument(
        "-hiif",
        "--hamiltonian-ii-filepath",
        help="path to pickled diagonal hamiltonian elements hamiltonian file",
    )

    parser.add_argument(
        "-hijf",
        "--hamiltonian-ij-filepath",
        help="path to pickled off-diagonal hamiltonian elements hamiltonian file",
    )

    parser.add_argument(
        "-sef",
        "--self-energies-filepath",
        help="path to pickled self-energies file",
    )

    args = parser.parse_args()

    input_dir = Path("inputs")

    with open(input_dir / args.parameters_filename, "rb") as file:
        parameters = pickle.load(file)

    with open(args.hamiltonian_ii_filepath, "rb") as file:
        hs_list_ii = pickle.load(file)

    with open(args.hamiltonian_ij_filepath, "rb") as file:
        hs_list_ij = pickle.load(file)

    with open(args.self_energies_filepath, "rb") as file:
        self_energies = pickle.load(file)

    los_indices = np.load(args.los_indices_filepath)

    hybridize_orbitals(
        los_indices,
        hs_list_ii,
        hs_list_ij,
        self_energies,
        **parameters,
    )
