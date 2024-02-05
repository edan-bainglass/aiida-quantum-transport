#!/usr/bin/env python

from __future__ import annotations

import pickle
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
from ase.atoms import Atoms
from edpyt.dmft import DMFT, Gfimp
from edpyt.nano_dmft import Gfimp as nanoGfimp
from edpyt.nano_dmft import Gfloc
from scipy.interpolate import interp1d

DELTA_DIR = "delta_folder"
SIGMA_DIR = "sigma_folder"


def run_dmft(
    device: Atoms,
    scattering_region: np.ndarray,
    active: list,
    energies: np.ndarray,
    matsubara_energies: np.ndarray,
    matsubara_hybridization: np.ndarray,
    H: np.ndarray,
    occupancies: np.ndarray,
    adjust_mu=False,
    U=4.0,
    number_of_baths=4,
    tolerance=1e-1,
    alpha=0.0,
    mu=0.0,
    dmu_min=0.0,
    dmu_max=0.9,
    dmu_step=1.0,
    inner_max_iter=10,
    outer_max_iter=1000,
) -> None:
    """docstring"""

    device = device[scattering_region]
    mask = np.where(np.isin(device.symbols, active))[0]
    device = device[mask]

    beta = np.pi / (matsubara_energies[0].imag)

    L = occupancies.size

    matsubara_hybridization = matsubara_hybridization.reshape(
        matsubara_energies.size, L, L
    )

    _HybMats = interp1d(
        matsubara_energies.imag,
        matsubara_hybridization,
        axis=0,
        bounds_error=False,
        fill_value=0.0,
    )

    def HybMats(z):
        return _HybMats(z.imag)

    H = H.real
    S = np.eye(L)

    idx_neq = np.arange(L)
    idx_inv = np.arange(L)

    V = np.eye(L) * U
    DC = np.diag(V.diagonal() * (occupancies - 0.5))
    gfloc = Gfloc(H - DC, S, HybMats, idx_neq, idx_inv)

    number_of_impurities = gfloc.idx_neq.size
    gfimp: list[Gfimp] = []

    for i in range(number_of_impurities):
        gfimp.append(
            Gfimp(
                number_of_baths,
                matsubara_energies.size,
                V[i, i],
                beta,
            )
        )

    gfimp = nanoGfimp(gfimp)

    occupancies = occupancies[gfloc.idx_neq]
    dmft = DMFT(
        gfimp,
        gfloc,
        occupancies,
        max_iter=inner_max_iter,
        tol=tolerance,
        adjust_mu=adjust_mu,
        alpha=alpha,
    )

    def Sigma(z):
        """docstring"""
        return np.zeros((number_of_impurities, z.size), complex)

    def _Sigma(z):
        """docstring"""
        return -DC.diagonal()[:, None] - gfloc.mu + gfloc.Sigma(z)[idx_inv]

    def save_sigma(sigma_diag, dmu):
        """docstring"""
        L, ne = sigma_diag.shape
        sigma = np.zeros((ne, L, L), complex)

        def save():
            """docstring"""
            for diag, mat in zip(sigma_diag.T, sigma):
                mat.flat[:: (L + 1)] = diag
            np.save(f"{SIGMA_DIR}/dmu_{dmu:1.4f}.npy", sigma)

        save()

    Path(DELTA_DIR).mkdir()
    Path(SIGMA_DIR).mkdir()

    number_of_steps = int((dmu_max - dmu_min) / dmu_step + 1)

    for dmu in np.linspace(dmu_min, dmu_max, number_of_steps):
        new_mu = mu + dmu
        delta = dmft.initialize(V.diagonal().mean(), Sigma, mu=new_mu)

        dmft.it = 0

        if outer_max_iter < inner_max_iter:
            raise ValueError(
                "absolute maximum iterations must be greater than internal DMFT maximum iterations"
            )

        while dmft.it < outer_max_iter:
            if dmft.it > 0:
                print("Restarting")
            outcome = dmft.solve(delta, verbose=False)
            delta = dmft.delta
            if outcome == "converged":
                print(f"Converged in {dmft.it} steps")
                break
            print(outcome)
            dmft.max_iter += inner_max_iter

        np.save(f"{DELTA_DIR}/dmu_{dmu:1.4f}.npy", dmft.delta)

        if adjust_mu:
            with open("mu.txt", "w") as file:
                file.write(str(gfloc.mu))

        save_sigma(_Sigma(energies), dmu)


if __name__ == "__main__":
    """docstring"""

    parser = ArgumentParser()

    parser.add_argument(
        "-dsf",
        "--device-structure-filename",
        help="name of device structure file",
    )

    parser.add_argument(
        "-srf",
        "--scattering-region-filename",
        help="name of pickled scattering region file",
    )

    parser.add_argument(
        "-af",
        "--active-species-filename",
        help="name of pickled active species file",
    )

    parser.add_argument(
        "-am",
        "--adjust-mu",
        type=bool,
        help="if the chemical potential is to be adjusted",
    )

    parser.add_argument(
        "-pf",
        "--parameters-filename",
        help="name of parameters file",
    )

    parser.add_argument(
        "-spf",
        "--sweep-parameters-filename",
        help="name of chemical potential sweep parameters file",
    )

    parser.add_argument(
        "-ef",
        "--energies-filename",
        help="name of energies file",
    )

    parser.add_argument(
        "-mef",
        "--matsubara-energies-filename",
        help="name of matsubara energies file",
    )

    parser.add_argument(
        "-mhf",
        "--matsubara-hybridization-filename",
        help="name of matsubara hybridization file",
    )

    parser.add_argument(
        "-hf",
        "--hamiltonian-filename",
        help="name of hamiltonian file",
    )

    parser.add_argument(
        "-of",
        "--occupancies-filename",
        help="name of occupancies file",
    )

    parser.add_argument(
        "-mf",
        "--mu-filename",
        required=False,
        help="name of converged mu file",
    )

    args = parser.parse_args()

    with open(args.device_structure_filename, "rb") as file:
        device = pickle.load(file)

    with open(args.scattering_region_filename, "rb") as file:
        region = np.load(file)

    with open(args.active_species_filename, "rb") as file:
        active: dict = pickle.load(file)

    with open(args.parameters_filename, "rb") as file:
        parameters = pickle.load(file)

    with open(args.sweep_parameters_filename, "rb") as file:
        sweep_parameters = pickle.load(file)

    with open(args.energies_filename, "rb") as file:
        energies = np.load(file)

    with open(args.matsubara_energies_filename, "rb") as file:
        matsubara_energies = np.load(file)

    with open(args.matsubara_hybridization_filename, "rb") as file:
        matsubara_hybridization = np.fromfile(file, complex)

    with open(args.hamiltonian_filename, "rb") as file:
        hamiltonian = np.load(file)

    with open(args.occupancies_filename, "rb") as file:
        occupancies = np.load(file)

    if args.mu_filename:
        mu = np.loadtxt(args.mu_filename)
    else:
        mu = 0.0

    run_dmft(
        device,
        region,
        list(active.keys()),
        energies,
        matsubara_energies,
        matsubara_hybridization,
        hamiltonian,
        occupancies,
        mu=mu,
        adjust_mu=args.adjust_mu,
        **parameters,
        **sweep_parameters,
    )
