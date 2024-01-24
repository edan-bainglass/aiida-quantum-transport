from __future__ import annotations

from typing import TYPE_CHECKING

from aiida import orm
from aiida.engine import ToContext, WorkChain

if TYPE_CHECKING:
    from aiida.engine.processes.workchains.workchain import WorkChainSpec

from aiida_quantum_transport.calculations import GpawCalculation


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
            help="The GPAW script",
        )

        spec.expose_inputs(
            GpawCalculation,
            namespace="dft.leads",
            exclude=["code"],
        )

        # spec.expose_inputs(
        #     GpawCalculation,
        #     namespace="dft.device",
        #     exclude=["code"],
        # )

        spec.expose_outputs(
            GpawCalculation,
            namespace="dft.leads",
        )

        # spec.expose_outputs(
        #     GpawCalculation,
        #     namespace="dft.device",
        # )

        spec.outline(
            cls.run_dft,
            # cls.define_scattering_region,
            # cls.transform_basis,
            # cls.compute_hybridization,
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
            **self.exposed_inputs(GpawCalculation, namespace="dft.leads"),
        }

        # device_inputs = {
        #     "code": self.inputs.dft.code,
        #     **self.exposed_inputs(GpawCalculation, namespace="dft.device"),
        # }

        return ToContext(
            dft_leads=self.submit(GpawCalculation, **leads_inputs),
            # dft_device=self.submit(GpawCalculation, **device_inputs),
        )

    def define_scattering_region(self):
        """docstring"""

    def transform_basis(self):
        """docstring"""

    def compute_hybridization(self):
        """docstring"""

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
                GpawCalculation,
                namespace="dft.leads",
            )
        )

        # self.out_many(
        #     self.exposed_outputs(
        #         self.ctx.dft_device,
        #         GpawCalculation,
        #         namespace="dft.device",
        #     )
        # )
