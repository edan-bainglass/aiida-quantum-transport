#!/usr/bin/env python

import pickle
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
from ase.units import _e, _hplanck, kB

G0 = 2.0 * _e**2 / _hplanck


def natural_sort(filepath: Path) -> float:
    """docstring"""
    numerical_part = filepath.stem.split("_")[-1]
    return float(numerical_part)


def fermidistribution(energy, kt):
    # fermi level is fixed to zero
    # energy can be a single number or a list
    assert kt >= 0.0, "Negative temperature encountered!"

    if kt == 0:
        if isinstance(energy, float):
            return int(energy / 2.0 <= 0)
        else:
            return (energy / 2.0 <= 0).astype(int)
    else:
        # TODO check for large exponents (return 0 if so)
        return 1.0 / (1.0 + np.exp(energy / kt))


def get_current(bias, E, transmission, T=300, unit="uA"):
    """Get the current in nA.

    Fermi-Dirac distribution = 1 / (1 + e^((E - mu) / kT))
    """

    # TODO refactor redundant computations

    if not isinstance(bias, (int, float)):
        bias = bias[np.newaxis]
        E = E[:, np.newaxis]
        transmission = transmission[:, np.newaxis]

    mu = bias / 2.0
    kT = kB * T

    fl = fermidistribution(E - mu, kT)
    fr = fermidistribution(E + mu, kT)

    return G0 * np.trapz((fl - fr) * transmission, x=E, axis=0) * 1e6  # uA


def numerical_derivative(x, y):
    """docstring"""
    dy_dx = np.diff(y) / np.diff(x)
    dy_dx = np.append(dy_dx, np.nan)

    return dy_dx


def compute_current(
    energies: np.ndarray,
    V_min=-2.5,
    V_max=2.5,
    dV=0.1,
    temperature=300.0,
    transmission_folder_path="transmission_folder",
) -> None:
    """docstring"""

    output_dir = Path("results")
    output_dir.mkdir(exist_ok=True)

    files = []
    for file in sorted(
        Path(transmission_folder_path).iterdir(),
        key=natural_sort,
    ):
        files.append(file)

    bias = np.linspace(V_min, V_max, int((V_max - V_min) / dV) + 1)
    transmissions = np.asarray([np.load(fn) for fn in files])

    energies = energies.real

    current = np.asarray(
        [
            get_current(bias, energies, transmission, temperature)
            for transmission in transmissions
        ]
    )

    derivative = np.asarray([numerical_derivative(bias, i) for i in current])

    np.save(output_dir / "current.npy", current)
    np.save(output_dir / "derivative.npy", derivative)


if __name__ == "__main__":
    """docstring"""

    parser = ArgumentParser()

    parser.add_argument(
        "-pf",
        "--parameters-filename",
        help="name of parameters file",
    )

    parser.add_argument(
        "-ef",
        "--energies-filepath",
        help="path to energies file",
    )

    parser.add_argument(
        "-tfp",
        "--transmission-folder-path",
        help="path to folder containing transmission files",
    )

    args = parser.parse_args()

    input_dir = Path("inputs")

    with open(input_dir / args.parameters_filename, "rb") as file:
        parameters = pickle.load(file)

    energies = np.load(args.energies_filepath)

    compute_current(
        energies,
        transmission_folder_path=args.transmission_folder_path,
        **parameters,
    )
