from aiida.cmdline.commands.cmd_data import verdi_data


@verdi_data.group("quantum_transport")
def data_cli():
    """Command line interface for aiida-quantum-transport."""
