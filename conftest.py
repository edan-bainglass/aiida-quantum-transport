"""pytest fixtures for simplified testing."""
import pytest

pytest_plugins = ["aiida.manage.tests.pytest_fixtures"]


@pytest.fixture(scope="function", autouse=True)
def clear_database_auto(clear_database):  # pylint: disable=unused-argument
    """Automatically clear database in between tests."""


@pytest.fixture(scope="function")
def quantum_transport_code(aiida_local_code_factory):
    """Get a quantum_transport code."""
    return aiida_local_code_factory(executable="diff", entry_point="quantum_transport")
