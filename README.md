[![Build Status][ci-badge]][ci-link]
[![Coverage Status][cov-badge]][cov-link]
[![Docs status][docs-badge]][docs-link]
[![PyPI version][pypi-badge]][pypi-link]

# aiida-quantum-transport

A plugin for quantum transport calculations based on NEGF methods.

This plugin is the default output of the
[AiiDA plugin cutter](https://github.com/aiidateam/aiida-plugin-cutter),
intended to help developers get started with their AiiDA plugins.

## Repository contents

- [`.github/`](.github/): [Github Actions](https://github.com/features/actions) configuration
  - [`ci.yml`](.github/workflows/ci.yml): runs tests, checks test coverage and builds documentation at every new commit
  - [`publish-on-pypi.yml`](.github/workflows/publish-on-pypi.yml): automatically deploy git tags to PyPI - just generate a [PyPI API token](https://pypi.org/help/#apitoken) for your PyPI account and add it to the `pypi_token` secret of your github repository
- [`src/aiida_quantum_transport/`](src/aiida_quantum_transport/): The main source code of the plugin package
  - [`data/`](src/aiida_quantum_transport/data/): Custom data classes
  - [`calculations.py`](src/aiida_quantum_transport/calculations.py): Quantum Transport AiiDA calculation classes
  - [`cli.py`](src/aiida_quantum_transport/cli.py): Custom `verdi` CLI commands
  - [`helpers.py`](src/aiida_quantum_transport/helpers.py): Helpers for setting up an AiiDA code
  - [`parsers.py`](src/aiida_quantum_transport/parsers.py): Custom parser classes
- [`docs/`](docs/): The documentation of the plugin
- [`tests/`](tests/): Basic regression tests using the [pytest](https://docs.pytest.org/en/latest/) framework (submitting a calculation, ...). Install `pip install -e .[testing]` and run `pytest`.
  - [`conftest.py`](tests/conftest.py): Configuration of fixtures for [pytest](https://docs.pytest.org/en/latest/)
- [`.gitignore`](.gitignore): Telling git which files to ignore
- [`.pre-commit-config.yaml`](.pre-commit-config.yaml): Configuration of [pre-commit hooks](https://pre-commit.com/) that sanitize coding style and check for syntax errors. Enable via `pip install -e .[pre-commit] && pre-commit install`
- [`.readthedocs.yml`](.readthedocs.yml): Configuration of documentation build for [Read the Docs](https://readthedocs.org/)
- [`LICENSE`](LICENSE): License for your plugin
- [`README.md`](README.md): This file
- [`pyproject.toml`](setup.json): Python package metadata for registration on [PyPI](https://pypi.org/) and the [AiiDA plugin registry](https://aiidateam.github.io/aiida-registry/) (including entry points)

## Features

Under development

## Installation

```shell
pip install aiida-quantum-transport
verdi quicksetup
verdi plugin list aiida.calculations
```

## Usage

Here goes a complete example of how to submit a test calculation using this plugin.

A quick demo of how to submit a calculation:

```shell
verdi daemon start     # make sure the daemon is running
cd examples
./example_01.py        # run test calculation
verdi process list -a  # check record of calculation
```

## Development

```shell
git clone https://github.com/edan-bainglass/aiida-quantum-transport .
cd aiida-quantum-transport
pip install --upgrade pip
pip install -e .[pre-commit,testing]  # install extra dependencies
pre-commit install  # install pre-commit hooks
pytest -v  # discover and run all tests
```

See the [developer guide](http://aiida-quantum-transport.readthedocs.io/en/latest/developer_guide/index.html) for more information.

## License

MIT

## Contact

edan.bainglass@gmail.com

[ci-badge]: https://github.com/edan-bainglass/aiida-quantum-transport/workflows/ci/badge.svg?branch=master
[ci-link]: https://github.com/edan-bainglass/aiida-quantum-transport/actions
[cov-badge]: https://coveralls.io/repos/github/edan-bainglass/aiida-quantum-transport/badge.svg?branch=master
[cov-link]: https://coveralls.io/github/edan-bainglass/aiida-quantum-transport?branch=master
[docs-badge]: https://readthedocs.org/projects/aiida-quantum-transport/badge
[docs-link]: http://aiida-quantum-transport.readthedocs.io/
[pypi-badge]: https://badge.fury.io/py/aiida-quantum-transport.svg
[pypi-link]: https://badge.fury.io/py/aiida-quantum-transport
