#!/usr/bin/env python

from __future__ import annotations

import pickle
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
from ase.atoms import Atoms
from qtpyt.base.leads import LeadSelfEnergy
from qtpyt.basis import Basis
from qtpyt.block_tridiag import graph_partition, greenfunction
from qtpyt.continued_fraction import get_ao_charge
from qtpyt.hybridization import Hybridization
from qtpyt.parallel import comm
from qtpyt.parallel.egrid import GridDesc
from qtpyt.projector import ProjectedGreenFunction
from qtpyt.surface.tools import prepare_leads_matrices
from qtpyt.tools import remove_pbc
from scipy.linalg import eigvalsh


def hybridize_orbitals(
    leads: Atoms,
    device: Atoms,
    leads_kpoints_grid: list,
    H_leads: np.ndarray,
    S_leads: np.ndarray,
    H_los: np.ndarray,
    S_los: np.ndarray,
    los_indices: np.ndarray,
    basis: dict,
    solver="dyson",
    E_step=1e-2,
    E_min=-3.0,
    E_max=3.0,
    eta=1e-4,
    E_grid_size=3000,
    beta=70.0,
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

    gf = greenfunction.GreenFunction(
        hs_list_ii,
        hs_list_ij,
        [(0, se[0]), (len(hs_list_ii) - 1, se[1])],
        solver=solver,
        eta=eta,
    )

    gfp = ProjectedGreenFunction(gf, los_indices)
    hyb = Hybridization(gfp)

    energies = np.arange(E_min, E_max + E_step / 2.0, E_step).round(7)

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
    assert se[0].eta == 0.0
    assert se[1].eta == 0.0
    energies = 1.0j * (2 * np.arange(E_grid_size) + 1) * np.pi / beta
    gd = GridDesc(energies, no, complex)
    HB = gd.empty_aligned_orbs()

    for e, energy in enumerate(gd.energies):
        HB[e] = hyb.retarded(energy)

    gd.write(HB, f"{output_dir}/matsubara_hybridization.bin")

    if comm.rank == 0:
        np.save(output_dir / "occupancies.npy", get_ao_charge(gfp))
        np.save(output_dir / "matsubara_energies.npy", energies)


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
        "--leads-hamiltonian-filepath",
        help="path to hamiltonian file",
    )

    parser.add_argument(
        "-lohf",
        "--los-hamiltonian-filepath",
        help="path to local orbitals hamiltonian file",
    )

    parser.add_argument(
        "-loif",
        "--los-index-filepath",
        help="path to local orbitals index file",
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

    with open(input_dir / args.parameters_filename, "rb") as file:
        parameters = pickle.load(file)

    H_leads, S_leads = np.load(args.leads_hamiltonian_filepath)

    H_scattering, S_scattering = np.load(args.los_hamiltonian_filepath)

    localization_index = np.load(args.los_index_filepath)

    hybridize_orbitals(
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
    )
