import numpy as np
from sympy import Symbol, Number, Function
from physicsnemo.sym.eq.pde import PDE

import physicsnemo.sym
from physicsnemo.sym.hydra import instantiate_arch, PhysicsNeMoConfig
from physicsnemo.sym.solver import Solver
from physicsnemo.sym.domain import Domain
from physicsnemo.sym.geometry.primitives_1d import Line1D
from physicsnemo.sym.domain.constraint import (
    PointwiseInteriorConstraint, PointwiseConstraint,
)
import numpy as np

from physicsnemo.sym.domain.validator import PointwiseValidator
from physicsnemo.sym.key import Key

import time

# ODE:
#   B' = r * B * (1 - B/K)
#   R' = sigma0 * B - k12 * R
#   E' = k12 * R - kc * E

@physicsnemo.sym.main(config_path="conf", config_name="config")
def run(cfg: PhysicsNeMoConfig) -> None:
    class CustomODE(PDE):
        def __init__(self, r=0.10, K=5.0, sigma0=0.01, k12=0.5, kc=0.15):

            x = Symbol("x")
            input_variables = {"x": x}
            B = Function("B")(*input_variables)
            R = Function("R")(*input_variables)
            E = Function("E")(*input_variables)

            r = Number(r)
            K = Number(K)
            sigma0 = Number(sigma0)
            k12 = Number(k12)
            kc = Number(kc)

            self.equations = {
                "B": B.diff(x) -r*B*(1-B/K),
                "R": R.diff(x) - (sigma0*B - k12*R),
                "E": E.diff(x) - (k12*R-kc*E),
            }
    # geometry & PDE
    geo = Line1D(0.0, 200.0)
    ode = CustomODE(r=0.10, K=5.0, sigma0=0.01, k12=0.5, kc=0.15)

    # network
    FC = instantiate_arch(
        cfg = cfg.arch.full_connected,
        input_keys=[Key("x")],
        output_keys=[Key("B"), Key("R"), Key("E")],
    )
    nodes = ode.make_nodes() + [FC.make_node(name="FC")]

    # domain
    domain = Domain()

    # add constraints to solver
    # interior constraint
    interior = PointwiseInteriorConstraint.from_config(
            nodes=nodes,
            geometry=geo,
            outvar={"B": 0, "R": 0, "E": 0},
            batch_size=cfg.batch_size.interior,
        )
    domain.add_constraint(interior, "ODE_residual")

    # initial condition at x=0
    initial_condition = PointwiseConstraint.from_numpy(
        nodes=nodes,
        invar={"x": np.array([[0.0]])},
        outvar={"B": 0.05, "R": 0.0, "E": 0.0},
        batch_size=cfg.batch_size.initial_condition,
    )
    domain.add_constraint(initial_condition, "IC")

    # make solver
    slv = Solver(cfg, domain)

    # start solver
    slv.solve()

if __name__ == "__main__":
    start_time = time.time()  # record start time
    run()
    end_time = time.time()    # record end time
    elapsed_time = end_time - start_time
    print(f"Total training time: {elapsed_time:.2f} seconds")