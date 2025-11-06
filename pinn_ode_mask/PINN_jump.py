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
    kappa: float = 20.0
    mask_eps = 0.01  # ε：遮罩半寬
    
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

            # 平滑階躍與其導數（供 sympy 使用，實際微分由 diff 完成）
            kappa_ = Number(kappa)
            def H_kappa(tau):
                return 1 / (1 + exp(-kappa_ * tau))

            # 計算生存率 SF_j = exp(-alpha*d_j - beta*d_j^2)
            alpha_ = Number(alpha)
            beta_ = Number(beta)
            SF_list = [exp(-alpha_ * Number(d) - beta_ * Number(d)**2) for d in d_js]

            # M(t) 與非負
            H_list = [H_kappa(x - Number(tj)) for tj in t_js]
            logSF  = [log(sf) for sf in SF_list]
            M = exp(sum(l * Hj for l, Hj in zip(logSF, H_list)))

            B = M * B_bar
            R = R_bar + Number(sigma1) * sum(
                (1 - sf) * B_bar * Hj for sf, Hj in zip(SF_list, H_list)
            )
            E = E_bar

            eps = Number(1e-8)  # 防止除 0
            # 原始殘差 t
            resB = B.diff(x) - ts * r * B * (1 - B/K)
            resR = R.diff(x) - ts * (s0 * B - k12 * R)
            resE = E.diff(x) - ts * (k12 * R - kc * E)

            # Step 3: 事件遮罩 w(t) X
            eps_mask = Number(mask_eps)
            def mask_one(arg):
                # m_j(t) = 1 - exp(- ( (t - t_j)^2 / (2 eps^2) ))
                return 1 - exp(- (arg**2) / (2 * eps_mask**2))
            
            # 對所有事件點累乘遮罩
            w = Number(1.0)
            for tj in t_js:
                w = w * mask_one(x - Number(tj))
            
            # 對殘差施加遮罩
            resB = w * resB
            resR = w * resR
            resE = w * resE

            # 相對殘差（使用 sqrt 避免 abs 的梯度問題）
            # 典型尺度：B_scale ~ K, R_scale ~ K*σ₀/k12, E_scale ~ K*σ₀/kc
            B_scale = K
            R_scale = K * s0 / k12
            E_scale = K * s0 / kc
            
            self.equations = {
                "ode_B": resB / sqrt(B**2 + (eps * B_scale)**2 + B_scale**2),
                "ode_R": resR / sqrt(R**2 + (eps * R_scale)**2 + R_scale**2),
                "ode_E": resE / sqrt(E**2 + (eps * E_scale)**2 + E_scale**2),
            }
    
    # Setup PDE (傳入所有參數，包括事件參數)
    ode = CustomODE(r=r_param, K=K_param, sigma0=sigma0_param, 
                    k12=k12_param, kc=kc_param, time_scale=time_scale,
                    t_js=t_js, d_js=d_js, alpha=alpha, beta=beta,
                    sigma1=sigma1, kappa=kappa, mask_eps=mask_eps)

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
    # 針對每個事件時間點 tj，在其附近區域增加採樣密度
    delta = 0.05  # 事件點附近的採樣範圍（縮放時間單位）
    event_batch_per_tj = max(1, interior_total // len(t_js) // 2)  # 每個事件點分配的採樣數
    
    for i, tj in enumerate(t_js):
        # 定義事件點附近的區域：|x - tj| < delta
        # 使用兩個條件的交集：x > tj - delta AND x < tj + delta
        from sympy import StrictGreaterThan, And
        criteria_event = And(
            StrictGreaterThan(x, max(0.0, tj - delta)),
            StrictLessThan(x, min(1.0, tj + delta))
        )
        
        interior_event = PointwiseInteriorConstraint(
            nodes=nodes,
            geometry=geo,
            outvar={"ode_B": 0, "ode_R": 0, "ode_E": 0},
            lambda_weighting={"ode_B": wB * 2.0, "ode_R": wR * 2.0, "ode_E": wE * 2.0},  # ★事件區域權重加倍
            criteria=criteria_event,
            batch_size=event_batch_per_tj,
        )
        domain.add_constraint(interior_event, f"interior_event_{i+1}")  # 命名為 interior_event_1, interior_event_2, ...



    # ========== 讀取 CSV 數據作為 Data Loss ==========
    # 使用絕對路徑，避免 Hydra 改變工作目錄後找不到檔案
    csv_path = os.path.join(os.path.dirname(__file__), "evb_training_data.csv")
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

    # Build invar and outvar
    invar_numpy = {   # dict of input variables
        "x": X_flat,  # shape=(n_points, 1)
    }
    outvar_numpy = {  # dict of output variables
        "B": B_true,  # shape=(n_points, 1)
        "R": R_true,  # shape=(n_points, 1)
        "E": E_true,  # shape=(n_points, 1)
    }

    # 使用 PointwiseConstraint 將 CSV 數據添加為訓練約束
    # 創建數據集
    #data_B_weight = 1.0
    #data_R_weight = 5.0
    #data_E_weight = 5.0
    #data_batch_size = 64
    
    # dataset = DictPointwiseDataset(
    #     invar=invar_numpy,
    #     outvar=outvar_numpy,
    #     lambda_weighting={
    #         "B": np.full((len(X_flat), 1), data_B_weight),
    #         "R": np.full((len(X_flat), 1), data_R_weight),
    #         "E": np.full((len(X_flat), 1), data_E_weight),
    #     },
    # )

    # # 創建數據約束
    # data_constraint = PointwiseConstraint(
    #     nodes=nodes,
    #     dataset=dataset,
    #     batch_size=data_batch_size,
    #     loss=PointwiseLossNorm(),  # 使用標準點式損失
    #     shuffle=True,  # 打亂數據順序
    #     drop_last=True,  # 使用 CUDA graphs 時必須為 True
    #     num_workers=0,  # 避免多進程問題
    # )
    # domain.add_constraint(data_constraint, "data_constraint")

    # ========== 驗證器 ==========
    _plotter = None
    if cfg.run_mode == 'eval':
        _plotter = CustomValidatorPlotter()

    # 使用所有數據點進行驗證（不使用 batch）
    validator = PointwiseValidator(
        nodes=nodes, 
        invar=invar_numpy, 
        true_outvar=outvar_numpy, 
        batch_size=len(X_flat),  # 驗證時使用所有數據點
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
