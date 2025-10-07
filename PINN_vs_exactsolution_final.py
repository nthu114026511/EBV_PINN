import numpy as np
from sympy import Symbol, Number, Function
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

# ODE:
#   B' = r * B * (1 - B/K)
#   R' = sigma0 * B - k12 * R
#   E' = k12 * R - kc * E

@physicsnemo.sym.main(config_path="conf", config_name="config")
def run(cfg: PhysicsNeMoConfig) -> None:
    class CustomODE(PDE):
        def __init__(self, r=0.10, K=5.0, sigma0=0.01, k12=0.05, kc=0.15):
            # coordinates
            x = Symbol("x")

            # make input variables
            input_variables = {"x": x}

            # make output variables
            B = Function("B")(*input_variables)
            R = Function("R")(*input_variables)
            E = Function("E")(*input_variables)

            r = Number(r)
            K = Number(K)
            sigma0 = Number(sigma0)
            k12 = Number(k12)
            kc = Number(kc)

            self.equations = {
                "ode_B": B.diff(x) - r*B*(1-B/K),
                "ode_R": R.diff(x) - (sigma0*B - k12*R),
                "ode_E": E.diff(x) - (k12*R - kc*E),
            }
    
    # Setup PDE
    ode = CustomODE(r=0.10, K=5.0, sigma0=0.01, k12=0.05, kc=0.15)

    # network
    FC = instantiate_arch(        
        input_keys=[Key("x")],
        output_keys=[Key("B"), Key("R"), Key("E")],
        cfg=cfg.arch.fully_connected,
    )
    nodes = ode.make_nodes() + [FC.make_node(name="FC")]

    # make geometry
    x = Symbol("x")
    x_max = 200.0
    geo = Line1D(0.0, x_max)

    # domain
    domain = Domain()

    # initial condition at x=0
    IC = PointwiseBoundaryConstraint(
        nodes=nodes,
        geometry=geo,
        outvar={"B": 0.05, "R": 0.0, "E": 0.0},
        batch_size=cfg.batch_size.IC,
        parameterization={x: 0.0},  # 指定在邊界 x=0 處
    )
    domain.add_constraint(IC, "IC")

    # add constraints to solver
    # interior constraint
    interior = PointwiseInteriorConstraint(
        nodes=nodes,
        geometry=geo,
        outvar={"ode_B": 0, "ode_R": 0, "ode_E": 0},
        batch_size=cfg.batch_size.interior,
    )
    domain.add_constraint(interior, "interior")

    # add validation data
    total_point = 10000 
    X_flat = np.linspace(0, x_max, total_point)[:, None]  # total_point points in x direction
    
    # Exact analytical solution
    from scipy.integrate import cumulative_trapezoid
    
    # Parameters
    r = 0.10
    K = 5.0
    sigma0 = 0.01
    k12 = 0.05
    kc = 0.15
    
    # Initial conditions
    B0 = 0.05
    R0 = 0.0
    E0 = 0.0
    
    # Calculate constant c from initial condition: B0 = K / (1 + c)
    c = (K - B0) / B0
    
    T = X_flat.flatten()  # time array
    
    # Helper function: F_alpha(t, alpha) = ∫_0^t e^{alpha*s} * B(s) ds
    def F_alpha(t, alpha, r, K, c):
        """
        Compute ∫_0^t exp(alpha*s) * B(s) ds where B(s) = K/(1 + c*exp(-r*s))
        """
        # B(s) = K / (1 + c*exp(-r*s))
        # ∫ exp(alpha*s) * K/(1 + c*exp(-r*s)) ds
        
        # For numerical stability, we compute this integral numerically
        result = np.zeros_like(t)
        for i, ti in enumerate(t):
            if ti == 0:
                result[i] = 0.0
            else:
                s_vals = np.linspace(0, ti, 500)
                B_vals = K / (1.0 + c * np.exp(-r * s_vals))
                integrand = np.exp(alpha * s_vals) * B_vals
                result[i] = np.trapz(integrand, s_vals)
        return result
    
    # ---------- Exact solution (vectorized) ----------
    B_ex = K / (1.0 + c * np.exp(-r * T))                       # logistic
    
    Fk12 = F_alpha(T, k12, r, K, c)                              # ∫_0^t e^{k12 s} B(s) ds
    R_ex = np.exp(-k12 * T) * (R0 + sigma0 * Fk12)
    
    if not np.isclose(kc, k12, atol=1e-12):
        Fkc  = F_alpha(T, kc, r, K, c)
        E_ex = (np.exp(-kc * T) * E0
                + (k12 / (kc - k12)) * (np.exp(-k12 * T) - np.exp(-kc * T)) * R0
                + sigma0 * (k12 / (kc - k12)) * (np.exp(-k12 * T) * Fk12 - np.exp(-kc * T) * Fkc))
    else:
        # kc == k12 = k 的極限式：E(t) = e^{-k t}[ E0 + k ∫_0^t e^{k s} R(s) ds ]
        k = kc
        Fk = F_alpha(T, k, r, K, c)
        # ∫_0^t e^{k s} R(s) ds = t*R0 + sigma0 ∫_0^t F_k(s) ds
        Gk = cumulative_trapezoid(Fk, T, initial=0.0)
        E_ex = np.exp(-k * T) * (E0 + k * (T * R0 + sigma0 * Gk))
    
    B_true = B_ex[:, None]  # shape=(total_point, 1)
    R_true = R_ex[:, None]  # shape=(total_point, 1)
    E_true = E_ex[:, None]  # shape=(total_point, 1)
    
    # Build invar and outvar
    invar_numpy = {   # dict of input variables
        "x": X_flat,  # shape=(total_point, 1)
    }
    outvar_numpy = {  # dict of output variables
        "B": B_true,  # shape=(total_point, 1)
        "R": R_true,  # shape=(total_point, 1)
        "E": E_true,  # shape=(total_point, 1)
    }

    _plotter = None
    if cfg.run_mode == 'eval':
        _plotter = CustomValidatorPlotter()

    validator = PointwiseValidator(
        nodes=nodes, 
        invar=invar_numpy, 
        true_outvar=outvar_numpy, 
        batch_size=10000,
        plotter=_plotter,
    )

    domain.add_validator(validator)

    # make solver
    slv = Solver(cfg, domain)

    # start solver
    slv.solve()
    
    # 訓練完成後，自動執行評估並生成對比圖
    if cfg.run_mode == 'train':
        print("\n" + "="*60)
        print("Training completed! Now generating comparison plots...")
        print("="*60 + "\n")
        
        # 創建繪圖器並執行評估
        plotter = CustomValidatorPlotter()
        
        # 從訓練好的模型獲取預測值
        import torch
        pred_outvar = {}
        with torch.no_grad():
            # 將 numpy 數據轉換為 torch tensor
            invar_torch = {k: torch.tensor(v, dtype=torch.float32, device=slv.device) 
                          for k, v in invar_numpy.items()}
            
            # 使用 FC 網絡進行預測
            pred_dict = FC(invar_torch)
            
            # 轉換回 numpy
            for key in pred_dict:
                pred_outvar[key] = pred_dict[key].detach().cpu().numpy()
        
        # 生成對比圖
        plotter(invar_numpy, outvar_numpy, pred_outvar)

if __name__ == "__main__":
    start_time = time.time()  # record start time
    run()
    end_time = time.time()    # record end time
    elapsed_time = end_time - start_time
    print(f"Total training time: {elapsed_time:.2f} seconds")
