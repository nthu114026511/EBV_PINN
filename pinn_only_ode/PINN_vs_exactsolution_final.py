#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from sympy import Symbol, Number, Function, StrictLessThan, sqrt
from physicsnemo.sym.eq.pde import PDE

import physicsnemo.sym
from physicsnemo.sym.hydra import instantiate_arch, PhysicsNeMoConfig
from physicsnemo.sym.solver import Solver
from physicsnemo.sym.domain import Domain
from physicsnemo.sym.geometry.primitives_1d import Line1D
from physicsnemo.sym.domain.constraint import (
    PointwiseInteriorConstraint, PointwiseBoundaryConstraint,
)
from physicsnemo.sym.domain.validator import PointwiseValidator
from physicsnemo.sym.key import Key

import time
from custom_plotter import CustomValidatorPlotter


@physicsnemo.sym.main(config_path="conf", config_name="config")
def run(cfg: PhysicsNeMoConfig) -> None:
    # ===== Physical params =====
    t_0, t_f = 0.0, 60.0
    time_scale = t_f - t_0

    r_param = 0.10
    K_param = 5.0
    sigma0_param = 0.01
    k12_param = 0.05
    kc_param = 0.15

    B0, R0, E0 = 0.05, 0.0, 0.0
    # ===========================

    # --- PDE on x (x ≡ t_s ∈ [0,1]) ---
    class CustomODE(PDE):
        def __init__(self, r, K, sigma0, k12, kc, time_scale):
            x = Symbol("x")                                # 用 x 當縮放時間
            input_variables = {"x": x}
            B = Function("B")(*input_variables)
            R = Function("R")(*input_variables)
            E = Function("E")(*input_variables)

            r, K, s0, k12, kc, ts = map(Number, (r, K, sigma0, k12, kc, time_scale))
            eps = Number(1e-8)

            # d/dx = d/dt_s；dt = (t_f - t_0) dt_s = ts * dx
            resB = B.diff(x) - ts * r * B * (1 - B/K)
            resR = R.diff(x) - ts * (s0 * B - k12 * R)
            resE = E.diff(x) - ts * (k12 * R - kc * E)

            B_scale = K
            R_scale = K * s0 / k12
            E_scale = K * s0 / kc

            self.equations = {
                "ode_B": resB / sqrt(B**2 + (eps * B_scale)**2),
                "ode_R": resR / sqrt(R**2 + (eps * R_scale)**2 + R_scale**2),
                "ode_E": resE / sqrt(E**2 + (eps * E_scale)**2 + E_scale**2),
            }

    ode = CustomODE(r_param, K_param, sigma0_param, k12_param, kc_param, time_scale)

    # --- NN takes x, outputs B,R,E ---
    FC = instantiate_arch(
        input_keys=[Key("x")],                       # ★ 用 x
        output_keys=[Key("B"), Key("R"), Key("E")],
        cfg=cfg.arch.fully_connected,
    )
    nodes = ode.make_nodes() + [FC.make_node(name="FC")]

    # --- Geometry: x ∈ [0,1] ---
    x = Symbol("x")
    geo = Line1D(0.0, 1.0)                           # ★ 不設定 parameterization

    domain = Domain()

    # --- IC at x=0 ---
    IC = PointwiseBoundaryConstraint(
        nodes=nodes,
        geometry=geo,
        outvar={"B": B0, "R": R0, "E": E0},
        lambda_weighting={"B": 100.0, "R": 100.0, "E": 100.0},
        batch_size=cfg.batch_size.IC,
        parameterization={x: 0.0},                   # ★ 用 x
    )
    domain.add_constraint(IC, "IC")

    # --- Interior residuals ---
    wB, wR, wE = 1.0, 5.0, 5.0
    interior_total = cfg.batch_size.interior
    # early_size = int(0.7 * interior_total)
    # late_size = interior_total - early_size
    #
    # criteria_early = StrictLessThan(x, 0.3)          # ★ 用 x
    # criteria_late = ~criteria_early
    #
    # interior_early = PointwiseInteriorConstraint(
    #     nodes=nodes,
    #     geometry=geo,
    #     outvar={"ode_B": 0, "ode_R": 0, "ode_E": 0},
    #     lambda_weighting={"ode_B": wB, "ode_R": wR, "ode_E": wE},
    #     criteria=criteria_early,
    #     batch_size=early_size,
    # )
    # domain.add_constraint(interior_early, "interior_early")
    #
    # interior_late = PointwiseInteriorConstraint(
    #     nodes=nodes,
    #     geometry=geo,
    #     outvar={"ode_B": 0, "ode_R": 0, "ode_E": 0},
    #     lambda_weighting={"ode_B": wB, "ode_R": wR, "ode_E": wE},
    #     criteria=criteria_late,
    #     batch_size=late_size,
    # )
    # domain.add_constraint(interior_late, "interior_late")

    # 不分前後期：統一使用一個 interior constraint
    interior = PointwiseInteriorConstraint(
        nodes=nodes,
        geometry=geo,
        outvar={"ode_B": 0, "ode_R": 0, "ode_E": 0},
        lambda_weighting={"ode_B": wB, "ode_R": wR, "ode_E": wE},
        batch_size=interior_total,
    )
    domain.add_constraint(interior, "interior")

    # --- Validation data (x ≡ t_s) ---
    total_point = 10000
    X_flat = np.linspace(0, 1.0, total_point)[:, None]   # ★ x

    from scipy.integrate import cumulative_trapezoid

    r, K, sigma0, k12, kc = r_param, K_param, sigma0_param, k12_param, kc_param
    c = (K - B0) / B0
    t_s = X_flat.flatten()
    T = t_0 + t_s * (t_f - t_0)

    def F_alpha(t_arr, alpha, r, K, c):
        result = np.zeros_like(t_arr)
        for i, ti in enumerate(t_arr):
            if ti == 0:
                result[i] = 0.0
            else:
                s_vals = np.linspace(0, ti, 500)
                B_vals = K / (1.0 + c * np.exp(-r * s_vals))
                integrand = np.exp(alpha * s_vals) * B_vals
                result[i] = np.trapz(integrand, s_vals)
        return result

    B_ex = K / (1.0 + c * np.exp(-r * T))
    Fk12 = F_alpha(T, k12, r, K, c)
    R_ex = np.exp(-k12 * T) * (R0 + sigma0 * Fk12)

    if not np.isclose(kc, k12, atol=1e-12):
        Fkc = F_alpha(T, kc, r, K, c)
        E_ex = (np.exp(-kc * T) * E0
                + (k12 / (kc - k12)) * (np.exp(-k12 * T) - np.exp(-kc * T)) * R0
                + sigma0 * (k12 / (kc - k12)) *
                  (np.exp(-k12 * T) * Fk12 - np.exp(-kc * T) * Fkc))
    else:
        k = kc
        Fk = F_alpha(T, k, r, K, c)
        Gk = cumulative_trapezoid(Fk, T, initial=0.0)
        E_ex = np.exp(-k * T) * (E0 + k * (T * R0 + sigma0 * Gk))

    invar_numpy = {"x": X_flat}                        # ★ 用 x
    outvar_numpy = {
        "B": B_ex[:, None],
        "R": R_ex[:, None],
        "E": E_ex[:, None],
    }

    _plotter = CustomValidatorPlotter() if cfg.run_mode == "eval" else None
    validator = PointwiseValidator(
        nodes=nodes,
        invar=invar_numpy,
        true_outvar=outvar_numpy,
        batch_size=10000,
        plotter=_plotter,
    )
    domain.add_validator(validator)

    # --- Solver ---
    slv = Solver(cfg, domain)
    slv.solve()

    # --- Post-training evaluation ---
    if cfg.run_mode == "train":
        print("\n" + "=" * 60)
        print("Training completed! Now generating comparison plots...")
        print("=" * 60 + "\n")

        import torch
        plotter = CustomValidatorPlotter()
        pred_outvar = {}

        with torch.no_grad():
            invar_torch = {k: torch.tensor(v, dtype=torch.float32, device=slv.device)
                           for k, v in invar_numpy.items()}
            pred_dict = FC(invar_torch)
            for key in pred_dict:
                pred_outvar[key] = pred_dict[key].detach().cpu().numpy()

        plotter(invar_numpy, outvar_numpy, pred_outvar)


if __name__ == "__main__":
    start_time = time.time()
    run()
    end_time = time.time()
    print(f"Total training time: {end_time - start_time:.2f} seconds")
