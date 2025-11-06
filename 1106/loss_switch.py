import numpy as np
import pandas as pd
from sympy import Symbol, Number, Function, sqrt, exp, log
from physicsnemo.sym.eq.pde import PDE
from typing import Sequence
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
from plotter import CustomValidatorPlotter
import torch
from physicsnemo.sym.domain.constraint.continuous import PointwiseConstraint
from physicsnemo.sym.dataset.continuous import DictPointwiseDataset
from physicsnemo.sym.loss import PointwiseLossNorm
import os

# ODE (原始):
#   dB/dt = r * B * (1 - B/K)
#   dR/dt = sigma0 * B - k12 * R
#   dE/dt = k12 * R - kc * E
#
# 時間縮放: t_s = (t - t_0) / (t_f - t_0), 其中 t ∈ [t_0, t_f], t_s ∈ [0, 1]
# 變數變換: dt = (t_f - t_0) * dt_s
# 轉換後的 ODE:
#   dB/dt_s = (t_f - t_0) * r * B * (1 - B/K)
#   dR/dt_s = (t_f - t_0) * (sigma0 * B - k12 * R)
#   dE/dt_s = (t_f - t_0) * (k12 * R - kc * E)

@physicsnemo.sym.main(config_path="conf", config_name="config")
def run(cfg: PhysicsNeMoConfig) -> None:
    # ========== 統一物理參數定義 ==========
    # 時間縮放參數
    t_0 = 0.0      # 原始時間起點
    t_f = 60.0     # 原始時間終點
    time_scale = t_f - t_0  # 縮放因子
    
    # ODE 參數
    r_param = 0.10      # growth rate
    K_param = 5.0       # carrying capacity
    sigma0_param = 0.01  # B → R rate
    k12_param = 0.05    # R → E rate
    kc_param = 0.15     # E decay rate

    # 事件與硬約束
    t_js_original = (10.0, 20.0, 40.0)  # 原始時間（單位：天）
    t_js: Sequence[float] = tuple((t - t_0) / time_scale for t in t_js_original)  # 縮放到 [0, 1]
    d_js: Sequence[float] = (1.5, 1.5, 1.5)  # 劑量序列
    alpha: float = 0.30  # SF = exp(-alpha*d - beta*d^2)
    beta: float = 0.03
    sigma1: float = 0.20
    kappa: float = 30.0  # 供平滑階梯 H_k 使用
    mask_eps = 0.05      # ε：事件半寬（同時用於 ODE 凹槽與 data 啟用）

    # 初始條件
    B0 = 0.05
    R0 = 0.0
    E0 = 0.0
    # ========================================
    
    class CustomODE(PDE):
        """
        這裡的 equations 對應 L_ode（連續區域）。
        透過 w(t)=∏(1-exp(-((t-t_j)^2)/(2ε^2))) 在事件附近把 ODE 殘差壓到 0。
        """
        def __init__(self, r, K, sigma0, k12, kc, time_scale,
                     t_js, d_js, alpha, beta, sigma1, kappa, mask_eps):
            x = Symbol("x")  # x ≡ t_s
            input_variables = {"x": x}
            B_bar = Function("B")(*input_variables)
            R_bar = Function("R")(*input_variables)
            E_bar = Function("E")(*input_variables)

            r, K, s0, k12, kc, ts = map(Number, (r, K, sigma0, k12, kc, time_scale))

            # 平滑階躍 H_k
            kappa_ = Number(kappa)
            def H_kappa(tau):
                return 1 / (1 + exp(-kappa_ * tau))

            # 生存率 SF_j = exp(-alpha*d_j - beta*d_j^2)
            alpha_ = Number(alpha)
            beta_ = Number(beta)
            SF_list = [exp(-alpha_ * Number(d) - beta_ * Number(d)**2) for d in d_js]

            # H_k(t - t_j)
            H_list  = [H_kappa(x - Number(tj)) for tj in t_js]
            logSF   = [log(sf) for sf in SF_list]

            # M(t) 與 R 的事件型態（保持 Step 3 的結構）
            M = exp(sum(l * Hj for l, Hj in zip(logSF, H_list)))
            B = M * B_bar
            R = R_bar + Number(sigma1) * sum((1 - sf) * B_bar * Hj for sf, Hj in zip(SF_list, H_list))
            E = E_bar

            eps = Number(1e-8)  # 防除 0

            # --- ODE 殘差（縮放時間 x=t_s）---
            resB = B.diff(x) - ts * r * B * (1 - B/K)
            resR = R.diff(x) - ts * (s0 * B - k12 * R)
            resE = E.diff(x) - ts * (k12 * R - kc * E)

            # --- 事件凹槽遮罩：事件附近 → 0；事件外 → 1 ---
            eps_mask = Number(mask_eps)
            def notch(arg):  # 1 - exp(-Δ^2/(2ε^2))
                return 1 - exp(- (arg**2) / (2 * eps_mask**2))
            w = Number(1.0)
            for tj in t_js:
                w = w * notch(x - Number(tj))
            # 只在連續區域累積殘差
            resB = w * resB
            resR = w * resR
            resE = w * resE

            # 相對殘差正規化
            B_scale = K
            R_scale = K * s0 / k12
            E_scale = K * s0 / kc
            
            self.equations = {
                "ode_B": resB / sqrt(B**2 + (eps * B_scale)**2),
                "ode_R": resR / sqrt(R**2 + (eps * R_scale)**2),
                "ode_E": resE / sqrt(E**2 + (eps * E_scale)**2),
            }

    # === 網路與幾何 ===
    FC = instantiate_arch(        
        input_keys=[Key("x")],
        output_keys=[Key("B"), Key("R"), Key("E")],
        cfg=cfg.arch.fully_connected,
    )
    x = Symbol("x")
    geo = Line1D(0.0, 1.0)

    # --- PDE/Nodes ---
    ode = CustomODE(
        r=r_param, K=K_param, sigma0=sigma0_param, k12=k12_param, kc=kc_param,
        time_scale=time_scale, t_js=t_js, d_js=d_js,
        alpha=alpha, beta=beta, sigma1=sigma1, kappa=kappa, mask_eps=mask_eps
    )
    nodes = ode.make_nodes() + [FC.make_node(name="FC")]

    # --- Domain 與約束 ---
    domain = Domain()

    # 1) 初始條件（強化）
    IC = PointwiseBoundaryConstraint(
        nodes=nodes,
        geometry=geo,
        outvar={"B": B0, "R": R0, "E": E0},
        lambda_weighting={"B": 100.0, "R": 100.0, "E": 100.0},
        batch_size=cfg.batch_size.IC,
        parameterization={x: 0.0},
    )
    domain.add_constraint(IC, "IC")

    # 2) 內點殘差（L_ode，事件外）
    wB, wR, wE = 1.0, 5.0, 5.0
    interior = PointwiseInteriorConstraint(
        nodes=nodes,
        geometry=geo,
        outvar={"ode_B": 0, "ode_R": 0, "ode_E": 0},
        lambda_weighting={"ode_B": wB, "ode_R": wR, "ode_E": wE},
        batch_size=cfg.batch_size.interior,
    )
    domain.add_constraint(interior, "interior")

    # ========== L_data：只在事件點附近啟用的 Data Loss ==========
    csv_path = os.path.join(os.path.dirname(__file__), "evb_training_data_jump.csv")
    df = pd.read_csv(csv_path)
    print(f"[INFO] Loaded CSV from: {csv_path}")
    
    # CSV 欄位：t, B, R, E（t 為原始時間 [0, 60]）
    t_original = df["t"].values
    t_scaled = (t_original - t_0) / (t_f - t_0)  # -> [0, 1]

    B_true = df["B"].values[:, None]
    R_true = df["R"].values[:, None]
    E_true = df["E"].values[:, None]
    X_flat = t_scaled[:, None]

    # ---- 事件權重：w_data(t) = 1 - w_ode(t) ----
    # w_ode(t) 與 PDE 中的 w(t) 同型：事件外≈1、事件近≈0
    def gaussian_notch(ts, tj, eps):
        return 1.0 - np.exp(-((ts - tj) ** 2) / (2.0 * eps ** 2))
    w_ode_np = np.ones_like(t_scaled, dtype=np.float64)
    for tj in t_js:
        w_ode_np *= gaussian_notch(t_scaled, tj, mask_eps)
    w_data_np = 1.0 - w_ode_np  # 事件近≈1，其它≈0

    # （可選）把事件窗外的小尾巴截 0，讓更乾脆
    w_data_np[w_data_np < 1e-3] = 0.0

    # 統計一下事件加權覆蓋率方便 debug
    coverage = float((w_data_np > 0).sum()) / float(w_data_np.size)
    print(f"[INFO] L_data active ratio: {coverage:.2%} of CSV samples")

    invar_numpy = {"x": X_flat}
    outvar_numpy = {"B": B_true, "R": R_true, "E": E_true}

    # 僅在事件附近啟用 data loss（權重隨 w_data_np 而動）
    data_B_weight = 5.0
    data_R_weight = 5.0
    data_E_weight = 5.0
    data_batch_size = 64

    lambda_weighting = {
        "B": (data_B_weight * w_data_np)[:, None],
        "R": (data_R_weight * w_data_np)[:, None],
        "E": (data_E_weight * w_data_np)[:, None],
    }

    dataset = DictPointwiseDataset(
        invar=invar_numpy,
        outvar=outvar_numpy,
        lambda_weighting=lambda_weighting,
    )

    data_constraint = PointwiseConstraint(
        nodes=nodes,
        dataset=dataset,
        batch_size=data_batch_size,
        loss=PointwiseLossNorm(),  # L2
        shuffle=True,
        drop_last=True,  # CUDA graphs 要求
        num_workers=0,
    )
    domain.add_constraint(data_constraint, "data_constraint")

    # ========== 驗證器 ==========
    _plotter = None
    if cfg.run_mode == "eval":
        _plotter = CustomValidatorPlotter()

    validator = PointwiseValidator(
        nodes=nodes,
        invar=invar_numpy,
        true_outvar=outvar_numpy,
        batch_size=len(X_flat),
        plotter=_plotter,
    )
    domain.add_validator(validator)

    # make solver
    slv = Solver(cfg, domain)

    # start solver
    slv.solve()
    
    # 訓練完成後：繪圖
    if cfg.run_mode == "train":
        print("\n" + "=" * 60)
        print("Training completed! Now generating comparison plots...")
        print("=" * 60 + "\n")
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
