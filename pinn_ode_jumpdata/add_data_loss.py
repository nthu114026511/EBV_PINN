import numpy as np
import pandas as pd
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
    t_f = 60.0     # 原始時間終點（根據 CSV 數據範圍）
    time_scale = t_f - t_0  # 縮放因子
    
    # ODE 參數
    r_param = 0.10      # growth rate
    K_param = 5.0       # carrying capacity
    sigma0_param = 0.01 # B → R rate
    k12_param = 0.05    # R → E rate
    kc_param = 0.15     # E decay rate
    
    # 初始條件
    B0 = 0.05
    R0 = 0.0
    E0 = 0.0
    # ========================================
    
    class CustomODE(PDE):
        def __init__(self, r=0.10, K=5.0, sigma0=0.01, k12=0.05, kc=0.15, time_scale=1.0):
            x = Symbol("x")
            input_variables = {"x": x}
            B = Function("B")(*input_variables)
            R = Function("R")(*input_variables)
            E = Function("E")(*input_variables)

            r, K, s0, k12, kc, ts = map(Number, (r, K, sigma0, k12, kc, time_scale))

            eps = Number(1e-8)  # 防止除 0
            # 原始殘差
            resB = B.diff(x) - ts * r * B * (1 - B/K)
            resR = R.diff(x) - ts * (s0 * B - k12 * R)
            resE = E.diff(x) - ts * (k12 * R - kc * E)

            # 相對殘差（使用 sqrt 避免 abs 的梯度問題）
            # 典型尺度：B_scale ~ K, R_scale ~ K*σ₀/k12, E_scale ~ K*σ₀/kc
            B_scale = K
            R_scale = K * s0 / k12
            E_scale = K * s0 / kc
            
            self.equations = {
                "ode_B": resB / sqrt(B**2 + (eps * B_scale)**2),
                "ode_R": resR / sqrt(R**2 + (eps * R_scale)**2 + R_scale**2),
                "ode_E": resE / sqrt(E**2 + (eps * E_scale)**2 + E_scale**2),
            }
    
    # Setup PDE (傳入縮放因子)
    ode = CustomODE(r=r_param, K=K_param, sigma0=sigma0_param, 
                    k12=k12_param, kc=kc_param, time_scale=time_scale)

    # network
    FC = instantiate_arch(        
        input_keys=[Key("x")],
        output_keys=[Key("B"), Key("R"), Key("E")],
        cfg=cfg.arch.fully_connected,
    )
    nodes = ode.make_nodes() + [FC.make_node(name="FC")]

    # make geometry
    x = Symbol("x")
    geo = Line1D(0.0, 1.0)

    # domain
    domain = Domain()

    # 1) IC 權重放大
    IC = PointwiseBoundaryConstraint(
        nodes=nodes,
        geometry=geo,
        outvar={"B": B0, "R": R0, "E": E0},
        lambda_weighting={"B": 100.0, "R": 100.0, "E": 100.0},  # ★IC 強化
        batch_size=cfg.batch_size.IC,
        parameterization={x: 0.0},
    )
    domain.add_constraint(IC, "IC")

    # 2) 內點殘差配重（R/E 比 B 重），並偏重前期區間
    wB, wR, wE = 1.0, 5.0, 5.0

    # 確保 batch_size 整數劃分正確
    interior_total = cfg.batch_size.interior
    interior_early_size = int(0.7 * interior_total)
    interior_late_size = interior_total - interior_early_size  # 確保總和正確

    # 2a. 前期內點（x < 0.3）
    criteria_early = StrictLessThan(x, 0.3)
    interior_early = PointwiseInteriorConstraint(
        nodes=nodes,
        geometry=geo,
        outvar={"ode_B": 0, "ode_R": 0, "ode_E": 0},
        lambda_weighting={"ode_B": wB, "ode_R": wR, "ode_E": wE},
        criteria=criteria_early,                 # ★集中抽樣早期（變化快）
        batch_size=interior_early_size,          # ★大部分點給早期
    )
    domain.add_constraint(interior_early, "interior_early")

    # 2b. 後期內點（x ≥ 0.3）
    criteria_late = ~criteria_early
    interior_late = PointwiseInteriorConstraint(
        nodes=nodes,
        geometry=geo,
        outvar={"ode_B": 0, "ode_R": 0, "ode_E": 0},
        lambda_weighting={"ode_B": wB, "ode_R": wR, "ode_E": wE},
        criteria=criteria_late,
        batch_size=interior_late_size,
    )
    domain.add_constraint(interior_late, "interior_late")

    # ========== 讀取 CSV 數據作為 Data Loss ==========
    print("Loading training data from CSV...")
    csv_path = "evb_training_data.csv"
    df = pd.read_csv(csv_path)
    
    # CSV 欄位：t, B, R, E
    # t 是原始時間 [0, 200]，需要轉換為縮放時間 t_s ∈ [0, 1]
    t_original = df['t'].values
    t_scaled = (t_original - t_0) / (t_f - t_0)  # 轉換為 [0, 1]
    
    # 取得對應的 B, R, E 數值
    B_csv = df['B'].values
    R_csv = df['R'].values
    E_csv = df['E'].values
    
    # 準備數據格式
    X_flat = t_scaled[:, None]  # shape=(n_points, 1)
    B_true = B_csv[:, None]     # shape=(n_points, 1)
    R_true = R_csv[:, None]     # shape=(n_points, 1)
    E_true = E_csv[:, None]     # shape=(n_points, 1)
    
    print(f"Loaded {len(df)} data points from CSV")
    print(f"Time range: t ∈ [{t_original.min():.2f}, {t_original.max():.2f}]")
    print(f"Scaled time range: t_s ∈ [{t_scaled.min():.4f}, {t_scaled.max():.4f}]")
    
    # Build invar and outvar
    invar_numpy = {   # dict of input variables
        "x": X_flat,  # shape=(n_points, 1)
    }
    outvar_numpy = {  # dict of output variables
        "B": B_true,  # shape=(n_points, 1)
        "R": R_true,  # shape=(n_points, 1)
        "E": E_true,  # shape=(n_points, 1)
    }

    _plotter = None
    if cfg.run_mode == 'eval':
        _plotter = CustomValidatorPlotter()

    # 使用 CSV 數據點數量作為 batch_size
    n_data_points = len(X_flat)
    
    validator = PointwiseValidator(
        nodes=nodes, 
        invar=invar_numpy, 
        true_outvar=outvar_numpy, 
        batch_size=min(n_data_points, 1000),  # 使用較小的 batch size 以避免內存問題
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
