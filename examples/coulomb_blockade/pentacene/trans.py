#!/usr/bin/env python

from __future__ import annotations

import pickle
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
from qtpyt.base.selfenergy import DataSelfEnergy as BaseDataSelfEnergy
from qtpyt.block_tridiag import greenfunction
from qtpyt.parallel import comm
from qtpyt.parallel.egrid import GridDesc
from qtpyt.projector import expand

TRANSMISSION_DIRNAME = "transmission_folder"


def compute_transmission(
    los_indices: np.ndarray,
    leads_nao: np.ndarray,
    hs_list_ii,
    hs_list_ij,
    self_energies,
    solver="dyson",
    eta=1e-4,
    E_min=-3.0,
    E_max=3.0,
    E_step=1e-2,
    sigma_folder_path="sigma_folder",
) -> None:
    """docstring"""

    output_dir = Path("results")
    output_dir.mkdir(exist_ok=True)

    transmission_dir = output_dir / TRANSMISSION_DIRNAME
    transmission_dir.mkdir(exist_ok=True)

    gf = greenfunction.GreenFunction(
        hs_list_ii,
        hs_list_ij,
        self_energies,
        solver=solver,
        eta=eta,
    )

    energies = np.linspace(E_min, E_max, int((E_max - E_min) / E_step) + 1)

    i1 = los_indices - leads_nao
    s1 = hs_list_ii[1][1]

    class DataSelfEnergy(BaseDataSelfEnergy):
        """Wrapper"""

        def retarded(self, energy):
            return expand(s1, super().retarded(energy), i1)

    def run(filepath: Path):
        gd = GridDesc(energies, 1, float)
        T = np.empty(gd.energies.size)

        for e, energy in enumerate(gd.energies):
            T[e] = gf.get_transmission(energy)

        T = gd.gather_energies(T)

        if comm.rank == 0:
            np.save(filepath, T.real)

    run(output_dir / "transmission_dft.npy")
    for filepath in Path(sigma_folder_path).glob("dmu_*"):
        dmft_self_energy = DataSelfEnergy(energies, np.load(filepath))
        gf.selfenergies.append((1, dmft_self_energy))
        run(transmission_dir / filepath.name)
        gf.selfenergies.pop()


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
        "-lnf",
        "--leads-nao-filepath",
        help="path to leads nao file",
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

    parser.add_argument(
        "-sfp",
        "--sigma-folder-path",
        help="path to folder containing self-energy files",
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

    leads_nao = np.load(args.leads_nao_filepath)

    compute_transmission(
        los_indices,
        leads_nao,
        hs_list_ii,
        hs_list_ij,
        self_energies,
        **parameters,
        sigma_folder_path=args.sigma_folder_path,
    )
