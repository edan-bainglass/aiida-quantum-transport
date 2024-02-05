#!/usr/bin/env python

from __future__ import annotations

import pickle
from argparse import ArgumentParser

import numpy as np
from gpaw import restart
from gpaw.lcao.pwf2 import LCAOwrap
from qtpyt.basis import Basis
from qtpyt.lo.tools import lowdin_rotation, rotate_matrix, subdiagonalize_atoms


def localize_orbitals(
    restart_filename: str,
    scattering_region: np.ndarray,
    active: dict,
    lowdin: bool,
) -> None:
    """docstring"""

    atoms, calc = restart(restart_filename, txt=None)
    lcao = LCAOwrap(calc)

    nao_a = np.array([setup.nao for setup in calc.wfs.setups])
    basis = Basis(atoms, nao_a)

    basis_p = basis[scattering_region]
    index_p = basis_p.get_indices()
    index_c = basis_p.extract().take(active)

    fermi = calc.get_fermi_level()
    H = lcao.get_hamiltonian()
    S = lcao.get_overlap()
    H -= fermi * S

    Usub, _ = subdiagonalize_atoms(basis, H, S, a=scattering_region)

    # Positive projection onto p-z AOs
    for idx_lo in index_p[index_c]:
        if Usub[idx_lo - 1, idx_lo] < 0.0:  # change sign
            Usub[:, idx_lo] *= -1

    H = rotate_matrix(H, Usub)
    S = rotate_matrix(S, Usub)

    if lowdin:
        Ulow = lowdin_rotation(H, S, index_p[index_c])

        H = rotate_matrix(H, Ulow)
        S = rotate_matrix(S, Ulow)

    np.save("idx_los.npy", index_p[index_c])
    np.save("hs_los.npy", (H[None, ...], S[None, ...]))


if __name__ == "__main__":
    """docstring"""

    parser = ArgumentParser()

    parser.add_argument(
        "-rfp",
        "--restart-filename",
        help="name of gpaw restart file",
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
        "-l",
        "--lowdin",
        type=bool,
        help="if lowdin rotation should be used",
    )

    args = parser.parse_args()

    with open(args.scattering_region_filename, "rb") as file:
        region = np.load(file)

    with open(args.active_species_filename, "rb") as file:
        active = pickle.load(file)

    localize_orbitals(args.restart_filename, region, active, args.lowdin)
