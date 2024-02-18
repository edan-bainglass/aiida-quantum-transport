#!/usr/bin/env python

from __future__ import annotations

import pickle
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
from ase.atoms import Atoms
from qtpyt.base.leads import LeadSelfEnergy
from qtpyt.basis import Basis
from qtpyt.block_tridiag import graph_partition
from qtpyt.surface.tools import prepare_leads_matrices
from qtpyt.tools import remove_pbc


def compute_gf_parameters(
    leads: Atoms,
    device: Atoms,
    leads_kpoints_grid: list,
    H_leads: np.ndarray,
    S_leads: np.ndarray,
    H_los: np.ndarray,
    S_los: np.ndarray,
    basis: dict,
) -> None:
    """docstring"""

    output_dir = Path("results")
    output_dir.mkdir(exist_ok=True)

    basis_leads = Basis.from_dictionary(leads, basis)
    basis_device = Basis.from_dictionary(device, basis)

    H_los = H_los.astype(complex)
    S_los = S_los.astype(complex)

    h_pl_ii, s_pl_ii, h_pl_ij, s_pl_ij = map(
        lambda m: m[0],
        prepare_leads_matrices(
            H_leads,
            S_leads,
            leads_kpoints_grid,
            align=(0, H_los[0, 0, 0]),
        )[1:],
    )

    remove_pbc(basis_device, H_los)
    remove_pbc(basis_device, S_los)

    se = [
        LeadSelfEnergy((h_pl_ii, s_pl_ii), (h_pl_ij, s_pl_ij)),
        LeadSelfEnergy((h_pl_ii, s_pl_ii), (h_pl_ij, s_pl_ij), id="right"),
    ]

    nodes = [
        0,
        basis_leads.nao,
        basis_device.nao - basis_leads.nao,
        basis_device.nao,
    ]

    hs_list_ii, hs_list_ij = graph_partition.tridiagonalize(
        nodes,
        H_los[0],
        S_los[0],
    )

    self_energies = [(0, se[0]), (len(hs_list_ii) - 1, se[1])]

    np.save(output_dir / "leads_nao.npy", basis_leads.nao)

    with open(output_dir / "hamiltonian_ii.pkl", "wb") as file:
        pickle.dump(hs_list_ii, file)

    with open(output_dir / "hamiltonian_ij.pkl", "wb") as file:
        pickle.dump(hs_list_ij, file)

    with open(output_dir / "self_energies.pkl", "wb") as file:
        pickle.dump(self_energies, file)


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
        "-lhf",
        "--leads-hamiltonian-filepath",
        help="path to hamiltonian file",
    )

    parser.add_argument(
        "-lohf",
        "--los-hamiltonian-filepath",
        help="path to local orbitals hamiltonian file",
    )

    args = parser.parse_args()

    input_dir = Path("inputs")

    with open(input_dir / args.leads_structure_filename, "rb") as file:
        leads = pickle.load(file)

    with open(input_dir / args.device_structure_filename, "rb") as file:
        device = pickle.load(file)

    with open(input_dir / args.leads_kpoints_filename, "rb") as file:
        leads_kpoints = pickle.load(file)

    with open(input_dir / args.basis_filename, "rb") as file:
        basis = pickle.load(file)

    H_leads, S_leads = np.load(args.leads_hamiltonian_filepath)

    H_los, S_los = np.load(args.los_hamiltonian_filepath)

    compute_gf_parameters(
        leads,
        device,
        leads_kpoints,
        H_leads,
        S_leads,
        H_los,
        S_los,
        basis,
    )
