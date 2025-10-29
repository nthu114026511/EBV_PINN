#!/usr/bin/env python3
# -*- coding: utf-8 -*-


from typing import Sequence, Dict, Any
from dataclasses import dataclass

from sympy import Symbol, Function, exp, log, sqrt, Number

try:
    from physicsnemo.sym.eq.pde import PDE
    # 為了修復 Variables 未定義的問題，我們需要修補它
    import physicsnemo.sym.eq.pde as pde_module
    if not hasattr(pde_module, 'Variables'):
        pde_module.Variables = dict
except Exception:
    class PDE:
        def __init__(self):
            self.equations = {}

@dataclass
class HardJumpParamsV2:
    # BRE 參數
    r: float = 0.10
    K: float = 5.0
    sigma0: float = 0.01
    k12: float = 0.05
    kc: float = 0.15

    # 時間縮放
    t0: float = 0.0
    tf: float = 60.0
    time_scale: float = None  # None → (tf - t0)

    # 事件與硬約束
    t_js: Sequence[float] = (10.0, 20.0, 40.0)
    d_js: Sequence[float] = (1.5, 1.5, 1.5)  # 劑量序列
    alpha: float = 0.30  # SF = exp(-alpha*d - beta*d^2)
    beta: float = 0.03
    sigma1: float = 0.20
    kappa: float = 80.0

    # 非負性
    nonneg: str = "softplus"  # "softplus" | "square"
    eps: float = 1e-8

    # Step 3: 事件遮罩參數
    use_mask: bool = True
    mask_eps: float = 0.8   # ε：遮罩半寬


def nonneg_fn(z, mode: str = "softplus", eps: float = 1e-8):
    if mode == "square":
        return z**2 + Number(eps)
    return log(1 + exp(z)) + Number(eps)


class HardJumpPDEV2(PDE):
    def __init__(self, params: HardJumpParamsV2):
        super().__init__()

        # 參數
        r_ = Number(params.r)
        K_ = Number(params.K)
        s0_ = Number(params.sigma0)
        k12_ = Number(params.k12)
        kc_ = Number(params.kc)

        t0_ = Number(params.t0)
        tf_ = Number(params.tf)
        ts_ = Number((params.tf - params.t0) if params.time_scale is None else params.time_scale)

        kappa_ = Number(params.kappa)
        eps_   = Number(params.eps)

        # 縮放時間 x ∈ [0,1]，t = t0 + x (tf - t0)
        x = Symbol("x")
        input_variables = {"x": x}
        t = t0_ + x * (tf_ - t0_)

        # 原網路輸出（raw heads）
        B_raw = Function("B")(*input_variables)
        R_raw = Function("R")(*input_variables)
        E_raw = Function("E")(*input_variables)

        # 平滑階躍與其導數（供 sympy 使用，實際微分由 diff 完成）
        def H_kappa(tau):
            return 1 / (1 + exp(-kappa_ * tau))

        # 計算生存率 SF_j = exp(-alpha*d_j - beta*d_j^2)
        alpha_ = Number(params.alpha)
        beta_ = Number(params.beta)
        SF_list = [exp(-alpha_ * Number(d) - beta_ * Number(d)**2) for d in params.d_js]
        
        # M(t) 與非負
        H_list = [H_kappa(t - Number(tj)) for tj in params.t_js]
        logSF  = [log(sf) for sf in SF_list]
        M = exp(sum(l * Hj for l, Hj in zip(logSF, H_list)))

        B_bar = nonneg_fn(B_raw, mode=params.nonneg, eps=float(params.eps))
        R_bar = nonneg_fn(R_raw, mode=params.nonneg, eps=float(params.eps))
        E_pos = nonneg_fn(E_raw, mode=params.nonneg, eps=float(params.eps))

        B_expr = M * B_bar
        R_expr = R_bar + Number(params.sigma1) * sum(
            (1 - sf) * B_expr * Hj for sf, Hj in zip(SF_list, H_list)
        )
        E_expr = E_pos

        # ODE 殘差（d/dx = ts_ d/dt）
        resB = B_expr.diff(x) - ts_ * r_  * B_expr * (1 - B_expr / K_)
        resR = R_expr.diff(x) - ts_ * (s0_ * B_expr - k12_ * R_expr)
        resE = E_expr.diff(x) - ts_ * (k12_ * R_expr - kc_ * E_expr)

        # Step 3: 事件遮罩 w(t)
        w = Number(1.0)
        if params.use_mask and len(params.t_js) > 0:
            eps_mask = Number(params.mask_eps)
            def mask_one(arg):
                # m_j(t) = 1 - exp(- ( (t - t_j)^2 / (2 eps^2) ))
                return 1 - exp(- (arg**2) / (2 * eps_mask**2))
            for tj in params.t_js:
                w = w * mask_one(t - Number(tj))
            resB = w * resB
            resR = w * resR
            resE = w * resE

        # 正規化
        B_scale = K_
        R_scale = K_ * s0_ / (k12_ + eps_)
        E_scale = K_ * s0_ / (kc_ + eps_)

        self.equations = {
            # Step 4.1：遮罩化的 ODE 殘差（事件外權重較高）
            "ode_B": resB / sqrt(B_expr**2 + (eps_ * B_scale)**2),
            "ode_R": resR / sqrt(R_expr**2 + (eps_ * R_scale)**2 + R_scale**2),
            "ode_E": resE / sqrt(E_expr**2 + (eps_ * E_scale)**2 + E_scale**2),
        }

    def get_equations(self) -> Dict[str, Any]:
        return self.equations


def build_hardjump_pde_v2(**kwargs) -> HardJumpPDEV2:
    p = HardJumpParamsV2(**kwargs)
    return HardJumpPDEV2(p)


if __name__ == "__main__":
    ode = build_hardjump_pde_v2()
    print("✅ HardJumpPDEV2 equations (with mask):")
    for k, v in ode.get_equations().items():
        print(f" - {k}: {v}")
