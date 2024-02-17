#!/usr/bin/env python

from __future__ import annotations

import pickle
from argparse import ArgumentParser, BooleanOptionalAction
from pathlib import Path

import numpy as np
from ase.atoms import Atoms
from edpyt.dmft import DMFT, Gfimp
from edpyt.nano_dmft import Gfimp as nanoGfimp
from edpyt.nano_dmft import Gfloc
from scipy.interpolate import interp1d

DELTA_DIRNAME = "delta_folder"
SIGMA_DIRNAME = "sigma_folder"


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
    inner_max_iter=1000,  # TODO check restart feature
    outer_max_iter=1000,
) -> None:
    """docstring"""

    output_dir = Path("results")
    output_dir.mkdir(exist_ok=True)

    delta_dir = output_dir / DELTA_DIRNAME
    delta_dir.mkdir(exist_ok=True)

    sigma_dir = output_dir / SIGMA_DIRNAME
    sigma_dir.mkdir(exist_ok=True)

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
    gfimp_list: list[Gfimp] = []

    for i in range(number_of_impurities):
        gfimp_list.append(
            Gfimp(
                number_of_baths,
                matsubara_energies.size,
                V[i, i],
                beta,
            )
        )

    gfimp = nanoGfimp(gfimp_list)

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
            np.save(sigma_dir / f"dmu_{dmu:1.4f}.npy", sigma)

        save()

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

        np.save(delta_dir / f"dmu_{dmu:1.4f}.npy", dmft.delta)

        if adjust_mu:
            with open(output_dir / "mu.txt", "w") as file:
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
        "-af",
        "--active-species-filename",
        help="name of pickled active species file",
    )

    parser.add_argument(
        "-am",
        "--adjust-mu",
        action=BooleanOptionalAction,
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
        "-srf",
        "--scattering-region-filepath",
        help="path to scattering region file",
    )

    parser.add_argument(
        "-ef",
        "--energies-filepath",
        help="path to energies file",
    )

    parser.add_argument(
        "-mef",
        "--matsubara-energies-filepath",
        help="path to matsubara energies file",
    )

    parser.add_argument(
        "-mhf",
        "--matsubara-hybridization-filepath",
        help="path to matsubara hybridization file",
    )

    parser.add_argument(
        "-hf",
        "--hamiltonian-filepath",
        help="path to hamiltonian file",
    )

    parser.add_argument(
        "-of",
        "--occupancies-filepath",
        help="path to occupancies file",
    )

    parser.add_argument(
        "-mf",
        "--mu-filepath",
        required=False,
        help="path to converged mu file",
    )

    args = parser.parse_args()

    input_dir = Path("inputs")

    with open(input_dir / args.device_structure_filename, "rb") as file:
        device = pickle.load(file)

    with open(input_dir / args.active_species_filename, "rb") as file:
        active: dict = pickle.load(file)

    with open(input_dir / args.parameters_filename, "rb") as file:
        parameters = pickle.load(file)

    with open(input_dir / args.sweep_parameters_filename, "rb") as file:
        sweep_parameters = pickle.load(file)

    region = np.load(args.scattering_region_filepath)

    energies = np.load(args.energies_filepath)

    matsubara_energies = np.load(args.matsubara_energies_filepath)

    matsubara_hybridization = np.fromfile(
        args.matsubara_hybridization_filepath, complex
    )

    hamiltonian = np.load(args.hamiltonian_filepath)

    occupancies = np.load(args.occupancies_filepath)

    if args.adjust_mu:
        mu = 0.0
    elif not args.mu_filepath:
        raise ValueError("missing mu file")
    else:
        mu = np.loadtxt(args.mu_filepath)

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
        adjust_mu=args.adjust_mu or False,
        **parameters,
        **sweep_parameters,
    )
