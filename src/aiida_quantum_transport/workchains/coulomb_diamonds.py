from __future__ import annotations

from typing import TYPE_CHECKING

from aiida import orm
from aiida.engine import ToContext, WorkChain

from aiida_quantum_transport.calculations import (
    CurrentCalculation,
    DFTCalculation,
    DMFTCalculation,
    GreensFunctionParametersCalculation,
    HybridizationCalculation,
    LocalizationCalculation,
    TransmissionCalculation,
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
            GreensFunctionParametersCalculation,
            namespace="greens_function",
            include=["code", "basis", "metadata"],
        )

        spec.input(
            "greens_function_parameters",
            valid_type=orm.Dict,
            default=lambda: orm.Dict({}),
            help="The parameters used to define the greens function",
        )

        spec.input(
            "energy_grid_parameters",
            valid_type=orm.Dict,
            default=lambda: orm.Dict({}),
            help="The parameters used to define the energy grid",
        )

        spec.expose_inputs(
            HybridizationCalculation,
            namespace="hybridization",
            include=["code", "temperature", "matsubara_grid_size", "metadata"],
        )

        spec.expose_inputs(
            DMFTCalculation,
            namespace="dmft",
            include=["code", "parameters"],
        )

        spec.expose_inputs(
            DMFTCalculation,
            namespace="dmft.converge_mu",
            include=["adjust_mu", "metadata"],
        )

        spec.expose_inputs(
            DMFTCalculation,
            namespace="dmft.sweep_mu",
            include=["metadata"],
        )

        spec.input(
            "dmft.sweep_mu.parameters",
            valid_type=orm.Dict,
            default=lambda: orm.Dict({}),
            help="The chemical potential sweep parameters",
        )

        spec.expose_inputs(
            TransmissionCalculation,
            namespace="transmission",
            include=["code", "metadata"],
        )

        spec.expose_inputs(
            CurrentCalculation,
            namespace="current",
            include=["code", "parameters", "metadata"],
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
            GreensFunctionParametersCalculation,
            namespace="greens_function",
        )

        spec.expose_outputs(
            HybridizationCalculation,
            namespace="hybridization",
        )

        spec.expose_outputs(
            DMFTCalculation,
            namespace="dmft.converge_mu",
        )

        spec.expose_outputs(
            DMFTCalculation,
            namespace="dmft.sweep_mu",
        )

        spec.expose_outputs(
            TransmissionCalculation,
            namespace="transmission",
        )

        spec.expose_outputs(
            CurrentCalculation,
            namespace="current",
        )

        spec.outline(
            cls.run_dft,
            cls.define_scattering_region,
            cls.transform_basis,
            cls.generate_greens_function_parameters,
            cls.compute_hybridization,
            cls.run_dmft_converge_mu,
            cls.run_dmft_sweep_mu,
            cls.compute_transmission,
            cls.compute_current,
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
            "device": {
                "remote_results_folder": self.ctx.dft_device.outputs.remote_results_folder,
            },
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

    def generate_greens_function_parameters(self):
        """docstring"""
        greens_function_inputs = {
            "leads": {
                "structure": self.inputs.dft.leads.structure,
                "kpoints": self.inputs.dft.leads.kpoints,
                "remote_results_folder": self.ctx.dft_leads.outputs.remote_results_folder,
            },
            "device": {
                "structure": self.inputs.dft.device.structure,
            },
            "los": {
                "remote_results_folder": self.ctx.localization.outputs.remote_results_folder,
            },
            **self.exposed_inputs(
                GreensFunctionParametersCalculation,
                namespace="greens_function",
            ),
        }
        return ToContext(
            greens_function=self.submit(
                GreensFunctionParametersCalculation,
                **greens_function_inputs,
            )
        )

    def compute_hybridization(self):
        """docstring"""
        hybridization_inputs = {
            "los": {
                "remote_results_folder": self.ctx.localization.outputs.remote_results_folder,
            },
            "greens_function": {
                "remote_results_folder": self.ctx.greens_function.outputs.remote_results_folder,
            },
            "greens_function_parameters": self.inputs.greens_function_parameters,
            "energy_grid_parameters": self.inputs.energy_grid_parameters,
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

    def run_dmft_converge_mu(self):
        """docstring"""
        dmft_converge_mu_inputs = {
            **self.exposed_inputs(
                DMFTCalculation,
                namespace="dmft",
            ),
            "device": {
                "structure": self.inputs.dft.device.structure,
            },
            "scattering": {
                "region": self.ctx.scattering_region,
                "active": self.inputs.scattering.active,
            },
            "hybridization": {
                "remote_results_folder": self.ctx.hybridization.outputs.remote_results_folder,
            },
            **self.exposed_inputs(
                DMFTCalculation,
                namespace="dmft.converge_mu",
            ),
        }
        return ToContext(
            dmft_converge_mu=self.submit(
                DMFTCalculation,
                **dmft_converge_mu_inputs,
            )
        )

    def run_dmft_sweep_mu(self):
        """docstring"""
        dmft_sweep_mu_inputs = {
            **self.exposed_inputs(
                DMFTCalculation,
                namespace="dmft",
            ),
            "device": {
                "structure": self.inputs.dft.device.structure,
            },
            "scattering": {
                "region": self.ctx.scattering_region,
                "active": self.inputs.scattering.active,
            },
            "hybridization": {
                "remote_results_folder": self.ctx.hybridization.outputs.remote_results_folder,
            },
            "mu_file": self.ctx.dmft_converge_mu.outputs.mu_file,
            "sweep": {
                "parameters": self.inputs.dmft.sweep_mu.parameters,
            },
            **self.exposed_inputs(
                DMFTCalculation,
                namespace="dmft.sweep_mu",
            ),
        }
        return ToContext(
            dmft_sweep_mu=self.submit(
                DMFTCalculation,
                **dmft_sweep_mu_inputs,
            )
        )

    def compute_transmission(self):
        """docstring"""
        transmission_inputs = {
            "los": {
                "remote_results_folder": self.ctx.localization.outputs.remote_results_folder,
            },
            "greens_function": {
                "remote_results_folder": self.ctx.greens_function.outputs.remote_results_folder,
            },
            "dmft": {
                "remote_results_folder": self.ctx.dmft_sweep_mu.outputs.remote_results_folder,
            },
            "greens_function_parameters": self.inputs.greens_function_parameters,
            "energy_grid_parameters": self.inputs.energy_grid_parameters,
            **self.exposed_inputs(
                TransmissionCalculation,
                namespace="transmission",
            ),
        }
        return ToContext(
            transmission=self.submit(
                TransmissionCalculation,
                **transmission_inputs,
            )
        )

    def compute_current(self):
        """docstring"""
        current_inputs = {
            "hybridization": {
                "remote_results_folder": self.ctx.hybridization.outputs.remote_results_folder,
            },
            "transmission": {
                "remote_results_folder": self.ctx.transmission.outputs.remote_results_folder,
            },
            "temperature": self.inputs.hybridization.temperature,
            **self.exposed_inputs(
                CurrentCalculation,
                namespace="current",
            ),
        }
        return ToContext(
            current=self.submit(
                CurrentCalculation,
                **current_inputs,
            )
        )

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
                self.ctx.greens_function,
                GreensFunctionParametersCalculation,
                namespace="greens_function",
            )
        )

        self.out_many(
            self.exposed_outputs(
                self.ctx.hybridization,
                HybridizationCalculation,
                namespace="hybridization",
            )
        )

        self.out_many(
            self.exposed_outputs(
                self.ctx.dmft_converge_mu,
                DMFTCalculation,
                namespace="dmft.converge_mu",
            )
        )

        self.out_many(
            self.exposed_outputs(
                self.ctx.dmft_sweep_mu,
                DMFTCalculation,
                namespace="dmft.sweep_mu",
            )
        )

        self.out_many(
            self.exposed_outputs(
                self.ctx.transmission,
                TransmissionCalculation,
                namespace="transmission",
            )
        )

        self.out_many(
            self.exposed_outputs(
                self.ctx.current,
                CurrentCalculation,
                namespace="current",
            )
        )
