from __future__ import annotations

from typing import TYPE_CHECKING

from aiida import orm
from aiida.engine import ToContext, WorkChain

from aiida_quantum_transport.calculations import (
    DFTCalculation,
    HybridizationCalculation,
    LocalizationCalculation,
    get_scattering_region,
)

if TYPE_CHECKING:
    from aiida.engine.processes.workchains.workchain import WorkChainSpec


class CoulombDiamondsWorkChain(WorkChain):
    """A workflow for generating Coulomb Diamonds from transmission data."""

    @classmethod
    def define(cls, spec: WorkChainSpec) -> None:
        """Define the workflow specifications (input, output, outline, etc.).

        Parameters
        ----------
        `spec` : `WorkChainSpec`
            The workflow specification.
        """

        super().define(spec)

        spec.input(
            "dft.code",
            valid_type=orm.AbstractCode,
            help="The DFT script",
        )

        spec.expose_inputs(
            DFTCalculation,
            namespace="dft.leads",
            exclude=["code"],
        )

        spec.expose_inputs(
            DFTCalculation,
            namespace="dft.device",
            exclude=["code"],
        )

        # TODO rethink this one (redefines localization input)
        spec.input(
            "scattering.region",
            valid_type=orm.Dict,
            required=False,
            default=lambda: orm.Dict({}),
            help="The xy-limits defining the scattering region",
        )

        spec.input(
            "scattering.active",
            valid_type=orm.Dict,
            help="The active species",
        )

        spec.expose_inputs(
            LocalizationCalculation,
            namespace="localization",
            include=["code", "lowdin", "metadata"],
        )

        spec.expose_inputs(
            HybridizationCalculation,
            namespace="hybridization",
            include=["code", "basis", "parameters", "metadata"],
        )

        spec.expose_outputs(
            DFTCalculation,
            namespace="dft.leads",
        )

        spec.expose_outputs(
            DFTCalculation,
            namespace="dft.device",
        )

        spec.expose_outputs(
            LocalizationCalculation,
            namespace="localization",
        )

        spec.expose_outputs(
            HybridizationCalculation,
            namespace="hybridization",
        )

        spec.outline(
            cls.run_dft,
            cls.define_scattering_region,
            cls.transform_basis,
            cls.compute_hybridization,
            # cls.run_dmft_adjust_mu,
            # cls.run_dmft_sweep_mu,
            # cls.compute_transmission,
            # cls.compute_current,
            cls.gather_results,
        )

    def run_dft(self):
        """docstring"""

        leads_inputs = {
            "code": self.inputs.dft.code,
            **self.exposed_inputs(DFTCalculation, namespace="dft.leads"),
        }

        device_inputs = {
            "code": self.inputs.dft.code,
            **self.exposed_inputs(DFTCalculation, namespace="dft.device"),
        }

        return ToContext(
            dft_leads=self.submit(DFTCalculation, **leads_inputs),
            dft_device=self.submit(DFTCalculation, **device_inputs),
        )

    def define_scattering_region(self):
        """docstring"""
        self.ctx.scattering_region = get_scattering_region(
            device=self.inputs.dft.device.structure,
            **self.inputs.scattering.region,
        )

    def transform_basis(self):
        """docstring"""
        localization_inputs = {
            "restart_file": self.ctx.dft_device.outputs.restart_file,
            "scattering": {
                "region": self.ctx.scattering_region,
                "active": self.inputs.scattering.active,
            },
            **self.exposed_inputs(
                LocalizationCalculation,
                namespace="localization",
            ),
        }
        return ToContext(
            localization=self.submit(
                LocalizationCalculation,
                **localization_inputs,
            )
        )

    def compute_hybridization(self):
        """docstring"""
        hybridization_inputs = {
            "leads": {
                "structure": self.inputs.dft.leads.structure,
                "kpoints": self.inputs.dft.leads.kpoints,
                "hamiltonian_file": self.ctx.dft_leads.outputs.hamiltonian_file,
            },
            "device": {
                "structure": self.inputs.dft.device.structure,
            },
            "localization": {
                "index_file": self.ctx.localization.outputs.index_file,
                "hamiltonian_file": self.ctx.localization.outputs.hamiltonian_file,
            },
            **self.exposed_inputs(
                HybridizationCalculation,
                namespace="hybridization",
            ),
        }
        return ToContext(
            hybridization=self.submit(
                HybridizationCalculation,
                **hybridization_inputs,
            )
        )

    def run_dmft_adjust_mu(self):
        """docstring"""

    def run_dmft_sweep_mu(self):
        """docstring"""

    def compute_transmission(self):
        """docstring"""

    def compute_current(self):
        """docstring"""

    def gather_results(self):
        """docstring"""

        self.out_many(
            self.exposed_outputs(
                self.ctx.dft_leads,
                DFTCalculation,
                namespace="dft.leads",
            )
        )

        self.out_many(
            self.exposed_outputs(
                self.ctx.dft_device,
                DFTCalculation,
                namespace="dft.device",
            )
        )

        self.out_many(
            self.exposed_outputs(
                self.ctx.localization,
                LocalizationCalculation,
                namespace="localization",
            )
        )

        self.out_many(
            self.exposed_outputs(
                self.ctx.hybridization,
                HybridizationCalculation,
                namespace="hybridization",
            )
        )
