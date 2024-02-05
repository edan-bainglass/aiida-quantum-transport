#!/usr/bin/env python

from __future__ import annotations

import pickle
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
from ase.atoms import Atoms
from qtpyt.base.leads import LeadSelfEnergy
from qtpyt.base.selfenergy import DataSelfEnergy as BaseDataSelfEnergy
from qtpyt.basis import Basis
from qtpyt.block_tridiag import graph_partition, greenfunction
from qtpyt.parallel import comm
from qtpyt.parallel.egrid import GridDesc
from qtpyt.projector import expand
from qtpyt.surface.tools import prepare_leads_matrices
from qtpyt.tools import remove_pbc

TRANSMISSION_DIR = "transmission_folder"


def compute_transmission(
    leads: Atoms,
    device: Atoms,
    leads_kpoints_grid: list,
    H_leads: np.ndarray,
    S_leads: np.ndarray,
    H_scattering: np.ndarray,
    S_scattering: np.ndarray,
    localization_index: np.ndarray,
    basis: dict,
    solver="dyson",
    E_step=1e-2,
    E_min=-3.0,
    E_max=3.0,
    eta=1e-4,
    sigma_folder_name="sigma_folder",
) -> None:
    """docstring"""

    basis_leads = Basis.from_dictionary(leads, basis)
    basis_device = Basis.from_dictionary(device, basis)

    H_scattering = H_scattering.astype(complex)
    S_scattering = S_scattering.astype(complex)

    h_pl_ii, s_pl_ii, h_pl_ij, s_pl_ij = map(
        lambda m: m[0],
        prepare_leads_matrices(
            H_leads,
            S_leads,
            leads_kpoints_grid,
            align=(0, H_scattering[0, 0, 0]),
        )[1:],
    )

    remove_pbc(basis_device, H_scattering)
    remove_pbc(basis_device, S_scattering)

    se = [
        LeadSelfEnergy((h_pl_ii, s_pl_ii), (h_pl_ij, s_pl_ij)),
        LeadSelfEnergy((h_pl_ii, s_pl_ii), (h_pl_ij, s_pl_ij), id="right"),
        None,
    ]

    nodes = [
        0,
        basis_leads.nao,
        basis_device.nao - basis_leads.nao,
        basis_device.nao,
    ]

    hs_list_ii, hs_list_ij = graph_partition.tridiagonalize(
        nodes,
        H_scattering[0],
        S_scattering[0],
    )

    energies = np.arange(E_min, E_max + E_step / 2.0, E_step).round(7)

    gf = greenfunction.GreenFunction(
        hs_list_ii,
        hs_list_ij,
        [(0, se[0]), (len(hs_list_ii) - 1, se[1])],
        solver=solver,
        eta=eta,
    )

    i1 = localization_index - nodes[1]
    s1 = hs_list_ii[1][1]

    class DataSelfEnergy(BaseDataSelfEnergy):
        """Wrapper"""

        def retarded(self, energy):
            return expand(s1, super().retarded(energy), i1)

    def load(filename):
        return DataSelfEnergy(energies, np.load(filename))

    def run(filename: str):
        gd = GridDesc(energies, 1, float)
        T = np.empty(gd.energies.size)

        for e, energy in enumerate(gd.energies):
            T[e] = gf.get_transmission(energy)

        T = gd.gather_energies(T)

        if comm.rank == 0:
            np.save(f"{TRANSMISSION_DIR}/{filename}", T.real)

    if comm.rank == 0:
        Path(TRANSMISSION_DIR).mkdir(exist_ok=True)

    for filename in Path(sigma_folder_name).glob("dmu_*"):
        se[2] = load(filename)
        gf.selfenergies.append((1, se[2]))
        run(filename=filename.as_posix())
        gf.selfenergies.pop()


if __name__ == "__main__":
    """docstring"""

    parser = ArgumentParser()

    parser.add_argument(
        "-lsf",
        "--leads-structure-filename",
        help="name of leads structure file",
    )

    parser.add_argument(
        "-dsf",
        "--device-structure-filename",
        help="name of device structure file",
    )

    parser.add_argument(
        "-lkf",
        "--leads-kpoints-filename",
        help="name of leads kpoints file",
    )

    parser.add_argument(
        "-bf",
        "--basis-filename",
        help="name of basis file",
    )

    parser.add_argument(
        "-pf",
        "--parameters-filename",
        help="name of parameters file",
    )

    parser.add_argument(
        "-lhf",
        "--leads-hamiltonian-filename",
        help="name of leads hamiltonian file",
    )

    parser.add_argument(
        "-shf",
        "--localized-hamiltonian-filename",
        help="name of localized hamiltonian file",
    )

    parser.add_argument(
        "-lif",
        "--localization-index-filename",
        help="name of localization index file",
    )

    parser.add_argument(
        "-sf",
        "--sigma-folder-name",
        help="name of folder containing self-energy files",
    )

    args = parser.parse_args()

    with open(args.leads_structure_filename, "rb") as file:
        leads = pickle.load(file)

    with open(args.device_structure_filename, "rb") as file:
        device = pickle.load(file)

    with open(args.leads_kpoints_filename, "rb") as file:
        leads_kpoints = pickle.load(file)

    with open(args.leads_hamiltonian_filename, "rb") as file:
        H_leads, S_leads = np.load(file)

    with open(args.localized_hamiltonian_filename, "rb") as file:
        H_scattering, S_scattering = np.load(file)

    with open(args.localization_index_filename, "rb") as file:
        localization_index = np.load(file)

    with open(args.basis_filename, "rb") as file:
        basis = pickle.load(file)

    with open(args.parameters_filename, "rb") as file:
        parameters = pickle.load(file)

    compute_transmission(
        leads,
        device,
        leads_kpoints,
        H_leads,
        S_leads,
        H_scattering,
        S_scattering,
        localization_index,
        basis,
        **parameters,
        sigma_folder_name=args.sigma_folder_name,
    )
