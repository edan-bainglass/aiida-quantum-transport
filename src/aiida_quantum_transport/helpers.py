""" Helper functions for automatically setting up computer & code.

Helper functions for setting up

 1. An AiiDA localhost computer
 2. A code on localhost

"""
from __future__ import annotations

import shutil
import tempfile

from aiida.common.exceptions import NotExistent
from aiida.orm import Code, Computer, Node

LOCALHOST_NAME = "localhost-test"

executables = {
    "quantum_transport": "diff",
}


def get_path_to_executable(executable: str) -> str:
    """Get path to local executable.

    Parameters
    ----------
    `executable` : `str`
        Name of executable in the `$PATH` variable.

    Returns
    -------
    `str`
        Path to executable.

    Raises
    ------
    `ValueError`
        If executable was not found in `$PATH`.
    """
    path = shutil.which(executable)
    if path is None:
        raise ValueError(f"'{executable}' executable not found in PATH.")
    return path


def get_computer(
    name: str = LOCALHOST_NAME,
    workdir: str | None = None,
) -> Computer:
    """Get AiiDA computer.

    Loads computer 'name' from the database, if exists.
    Sets up local computer 'name', if it isn't found in the DB.

    Parameters
    ----------
    `name` : `str`
        Name of computer to load or set up., `LOCALHOST_NAME` by default.
    `workdir` : `str | None`
        path to work directory, `None` by default.
        Used only when creating a new computer.

    Returns
    -------
    `aiida.orm.computers.Computer`
        The computer Node.
    """

    try:
        computer = Computer.objects.get(label=name)
    except NotExistent:
        if workdir is None:
            workdir = tempfile.mkdtemp()

        computer = Computer(
            label=name,
            description="localhost",
            hostname=name,
            workdir=workdir,
            transport_type="core.local",
            scheduler_type="core.direct",
        )
        computer.store()
        computer.set_minimum_job_poll_interval(0.0)
        computer.configure()

    return computer


def get_code(entry_point: str, computer: Computer) -> Node:
    """Get local code.

    Sets up code for given entry point on given computer.

    Parameters
    ----------
    `entry_point` : `str`
        Entry point of calculation plugin.
    `computer` : `Computer`
        (local) AiiDA computer.

    Returns
    -------
    `aiida.orm.Node`
        The code node.

    Raises
    ------
    `KeyError`
        If entry point is not recognized.
    """

    try:
        executable = executables[entry_point]
    except KeyError as exc:
        raise KeyError(
            f"Entry point '{entry_point}' not recognized. Allowed values: {list(executables.keys())}"
        ) from exc

    codes = Code.objects.find(filters={"label": executable})
    if codes:
        return codes[0]

    path = get_path_to_executable(executable)
    code = Code(
        input_plugin_name=entry_point,
        remote_computer_exec=[computer, path],
    )
    code.label = executable
    return code.store()
