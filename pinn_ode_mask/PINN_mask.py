import numpy as np
import pandas as pd
from sympy import Symbol, Number, Function, StrictLessThan, sqrt, exp, log
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
    sigma0_param = 0.30 # B → R rate
    k12_param = 0.20    # R → E rate
    kc_param = 0.15     # E decay rate

    # 事件與硬約束
    t_js_original = (5.0, 12.0, 20.0)  # 原始時間（單位：天）
    t_js: Sequence[float] = tuple((t - t_0) / time_scale for t in t_js_original)  # 轉換為縮放時間 [0, 1]
    d_js: Sequence[float] = (2, 2, 2)  # 劑量序列
    alpha: float = 0.30  # SF = exp(-alpha*d - beta*d^2)
    beta: float = 0.03
    sigma1: float = 0.20
    kappa_default: float = 20.0  # 初值不再直接使用，退火時會被覆蓋
    mask_eps = 0.02  # ε：遮罩半寬
    
    # 初始條件
    B0 = 0.05
    R0 = 0.0
    E0 = 0.0
    # ========================================
    
    class CustomODE(PDE):
        def __init__(self, r, K, sigma0, k12, kc, time_scale,
                     t_js, d_js, alpha, beta, sigma1, kappa, mask_eps):
            x = Symbol("x")
            input_variables = {"x": x}
            B_bar = Function("B")(*input_variables)
            R_bar = Function("R")(*input_variables)
            E_bar = Function("E")(*input_variables)

            r, K, s0, k12, kc, ts = map(Number, (r, K, sigma0, k12, kc, time_scale))

            # 平滑階躍
            kappa_ = Number(kappa)
            def H_kappa(tau):
                return 1 / (1 + exp(-kappa_ * tau))

            # 生存率 SF_j = exp(-alpha*d_j - beta*d_j^2)
            alpha_ = Number(alpha)
            beta_ = Number(beta)
            SF_list = [exp(-alpha_ * Number(d) - beta_ * Number(d)**2) for d in d_js]

            # --- 生存率與階梯 ---
            H_list  = [H_kappa(x - Number(tj)) for tj in t_js]
            logSF   = [log(sf) for sf in SF_list]

            # M_full：包含所有事件
            M_full = exp(sum(l * Hj for l, Hj in zip(logSF, H_list)))

            # M_pre_list[j]：只累積到第 j-1 個事件（jump 前的乘積）
            M_pre_list = []
            prefix = Number(0.0)
            for j in range(len(t_js)):
                M_pre_list.append(exp(prefix))
                prefix = prefix + logSF[j] * H_list[j]

            # 物理解釋：B = M_full * B_bar；R 的跳躍用 B_pre_j = M_pre_list[j] * B_bar
            B = M_full * B_bar
            R = R_bar + Number(sigma1) * sum(
                (1 - sf) * (M_pre_list[j] * B_bar) * H_list[j]
                for j, sf in enumerate(SF_list)
            )
            E = E_bar

            eps = Number(1e-8)  # 防止除 0

            # 殘差（x≡t_s）
            resB = B.diff(x) - ts * r * B * (1 - B/K)
            resR = R.diff(x) - ts * (s0 * B - k12 * R)
            resE = E.diff(x) - ts * (k12 * R - kc * E)

            # Step 3: 事件遮罩 w(t)
            eps_mask = Number(mask_eps)
            def mask_one(arg):
                # m_j(t) = 1 - exp(- ( (t - t_j)^2 / (2 eps^2) ))
                return 1 - exp(- (arg**2) / (2 * eps_mask**2))
            
            w = Number(1.0)
            for tj in t_js:
                w = w * mask_one(x - Number(tj))
            
            # 對殘差施加遮罩
            resB = w * resB
            resR = w * resR
            resE = w * resE

            # 相對殘差正規化尺度
            B_scale = K
            R_scale = K * s0 / k12
            E_scale = K * s0 / kc
            
            self.equations = {
                "ode_B": resB / sqrt(B**2 + (eps * B_scale)**2),
                "ode_R": resR / sqrt(R**2 + (eps * R_scale)**2),
                "ode_E": resE / sqrt(E**2 + (eps * E_scale)**2),
            }

    # === 共用網路與幾何（退火各階段沿用這同一顆 FC 權重） ===
    FC = instantiate_arch(        
        input_keys=[Key("x")],
        output_keys=[Key("B"), Key("R"), Key("E")],
        cfg=cfg.arch.fully_connected,
    )
    x = Symbol("x")
    geo = Line1D(0.0, 1.0)

    # --- Validation data (x ≡ t_s，使用解析對照) ---
    total_point = 10000
    X_flat = np.linspace(0, 1.0, total_point)[:, None]   # x

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

    invar_numpy = {"x": X_flat}
    outvar_numpy = {
        "B": B_ex[:, None],
        "R": R_ex[:, None],
        "E": E_ex[:, None],
    }

    # ===========================================================
    # κ 退火設定：會自動根據 conf.max_steps 分段
    # ===========================================================
    kappa_schedule = [5.0, 10.0, 20.0, 40.0, 80.0]  # 可自行調整
    total_steps = getattr(cfg, "max_steps", getattr(getattr(cfg, "training", {}), "max_steps", 2000))
    epochs_per_stage = max(1, int(total_steps // len(kappa_schedule)))
    print(f"\n[κ-anneal] total={total_steps}, stages={len(kappa_schedule)}, per_stage={epochs_per_stage}\n")

    # === 進行多階段 κ 退火訓練 ===
    for stage_idx, kappa in enumerate(kappa_schedule, start=1):
        print(f"=== κ-Anneal Stage {stage_idx}/{len(kappa_schedule)} | κ = {kappa:.1f} ===")
        
        # --- 重建 PDE/Nodes（帶入當前 κ） ---
        ode = CustomODE(
            r=r_param, K=K_param, sigma0=sigma0_param, k12=k12_param, kc=kc_param,
            time_scale=time_scale, t_js=t_js, d_js=d_js,
            alpha=alpha, beta=beta, sigma1=sigma1, kappa=kappa, mask_eps=mask_eps
        )
        nodes = ode.make_nodes() + [FC.make_node(name="FC")]

        # --- Domain 與約束 ---
        domain = Domain()

        # 1) IC 權重放大
        IC = PointwiseBoundaryConstraint(
            nodes=nodes,
            geometry=geo,
            outvar={"B": B0, "R": R0, "E": E0},
            lambda_weighting={"B": 100.0, "R": 100.0, "E": 100.0},  # IC 強化
            batch_size=cfg.batch_size.IC,
            parameterization={x: 0.0},
        )
        domain.add_constraint(IC, "IC")

        # 2) 內點殘差配重（R/E 比 B 重）
        wB, wR, wE = 1.0, 5.0, 5.0
        interior_total = cfg.batch_size.interior
        interior = PointwiseInteriorConstraint(
            nodes=nodes,
            geometry=geo,
            outvar={"ode_B": 0, "ode_R": 0, "ode_E": 0},
            lambda_weighting={"ode_B": wB, "ode_R": wR, "ode_E": wE},
            batch_size=interior_total,
        )
        domain.add_constraint(interior, "interior")

        # 2c. 事件點附近密集採樣（在 tj ± δ 範圍內）
        delta = 0.05  # 事件點附近的採樣範圍（縮放時間單位）
        event_batch_per_tj = max(1, interior_total // max(1, len(t_js)) // 2)
        from sympy import StrictGreaterThan, StrictLessThan, And
        for i, tj in enumerate(t_js):
            criteria_event = And(
                StrictGreaterThan(x, max(0.0, tj - delta)),
                StrictLessThan(x, min(1.0, tj + delta))
            )
            interior_event = PointwiseInteriorConstraint(
                nodes=nodes,
                geometry=geo,
                outvar={"ode_B": 0, "ode_R": 0, "ode_E": 0},
                lambda_weighting={"ode_B": wB * 2.0, "ode_R": wR * 2.0, "ode_E": wE * 2.0},
                criteria=criteria_event,
                batch_size=event_batch_per_tj,
            )
            domain.add_constraint(interior_event, f"interior_event_{i+1}")

        # --- Validator ---
        _plotter = CustomValidatorPlotter() if cfg.run_mode == "eval" else None
        validator = PointwiseValidator(
            nodes=nodes,
            invar=invar_numpy,
            true_outvar=outvar_numpy,
            batch_size=len(invar_numpy["x"]),
            plotter=_plotter,
        )
        domain.add_validator(validator)

        # --- 每階段設定 epoch（覆蓋 conf 的單段步數） ---
        if hasattr(cfg, "max_steps"):
            cfg.max_steps = epochs_per_stage
        elif hasattr(cfg, "training") and isinstance(cfg.training, dict):
            cfg.training["max_steps"] = epochs_per_stage

        # --- Solver & 訓練（FC 權重沿用） ---
        slv = Solver(cfg, domain)
        slv.solve()

    print("\n=== All κ-Anneal stages finished ===\n")

    # 訓練完成後，自動執行評估並生成對比圖（用最後一段的 FC 權重）
    if cfg.run_mode == 'train':
        print("\n" + "="*60)
        print("Training completed! Now generating comparison plots...")
        print("="*60 + "\n")
        
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
    elapsed_time = end_time - start_time
    print(f"Total training time: {elapsed_time:.2f} seconds")
