#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RK4 under LQ model — 連續版（無任何 jump）
連續 ODE：
  B' = r*B*(1 - B/K)
  R' = sigma0*B - k12*R
  E' = k12*R - kc*E

輸出：
  - 圖：B, R, E 隨時間
  - CSV：./data/evb_training_data.csv ，欄位 t, B, R, E
"""

from __future__ import annotations
from typing import Dict, Callable, Tuple
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# -------------------------- 參數與初值（沿用你的設定） --------------------------
r      = 0.10     # logistic growth rate
K      = 5.00     # carrying capacity
sigma0 = 0.01     # base transfer rate
sigma1 = 0.20     # 未用於連續方程（保留只是對齊參數表）
k12    = 0.05     # R→E transfer rate
kc     = 0.15     # E clearance rate

B0, R0, E0 = 0.05, 0.00, 0.00

# 全域時間網格
t0, tf = 0.0, 60.0
h = 0.1  # 基本步長

pars: Dict[str, float] = dict(r=r, K=K, sigma0=sigma0, k12=k12, kc=kc)

# -------------------------- ODE 右端 --------------------------
def rhs(t: float, y: np.ndarray, p: Dict[str, float]) -> np.ndarray:
    """y = [B, R, E] → dy/dt"""
    B, R, E = y
    dB = p["r"] * B * (1.0 - B / p["K"])
    dR = p["sigma0"] * B - p["k12"] * R
    dE = p["k12"] * R - p["kc"] * E
    return np.array([dB, dR, dE], dtype=float)

def rk4_step(f: Callable[[float, np.ndarray, Dict[str, float]], np.ndarray],
             t: float, y: np.ndarray, h: float, p: Dict[str, float]) -> np.ndarray:
    """單步 RK4；附非負裁切避免微小負值。"""
    k1 = f(t,           y,               p)
    k2 = f(t + 0.5*h,   y + 0.5*h*k1,    p)
    k3 = f(t + 0.5*h,   y + 0.5*h*k2,    p)
    k4 = f(t + h,       y + h*k3,        p)
    y1 = y + (h/6.0) * (k1 + 2*k2 + 2*k3 + k4)
    return np.maximum(y1, 0.0)

def integrate_rk4(f: Callable[[float, np.ndarray, Dict[str, float]], np.ndarray],
                  tspan: Tuple[float, float],
                  y0: np.ndarray,
                  h: float,
                  p: Dict[str, float]) -> tuple[np.ndarray, np.ndarray]:
    """單段連續 RK4 積分（無事件）。"""
    t0, tf = float(tspan[0]), float(tspan[1])
    T = [t0]
    Y = [y0.copy()]
    t, y = t0, y0.copy()

    while t < tf - 1e-14:
        hstep = min(h, tf - t)  # 收尾縮步對齊 tf
        y = rk4_step(f, t, y, hstep, p)
        t += hstep
        T.append(t)
        Y.append(y.copy())

    return np.array(T), np.vstack(Y)

def main():
    y0 = np.array([B0, R0, E0], dtype=float)

    # 積分
    T, Y = integrate_rk4(rhs, (t0, tf), y0, h, pars)
    B, R, E = Y[:, 0], Y[:, 1], Y[:, 2]

    # 繪圖
    fig, axes = plt.subplots(3, 1, figsize=(9, 8), sharex=True)
    axes[0].plot(T, B, linewidth=1.5)
    axes[0].set_ylabel("B")
    axes[0].set_title("B, R, E (continuous, no jumps)")

    axes[1].plot(T, R, linewidth=1.5)
    axes[1].set_ylabel("R")

    axes[2].plot(T, E, linewidth=1.5)
    axes[2].set_ylabel("E")
    axes[2].set_xlabel("t")

    fig.tight_layout()
    plt.show()

    # 儲存 CSV
    output_dir = os.path.join(os.getcwd(), "data")
    os.makedirs(output_dir, exist_ok=True)
    df = pd.DataFrame({"t": T, "B": B, "R": R, "E": E})
    csv_path = os.path.join(output_dir, "evb_training_data.csv")
    df.to_csv(csv_path, index=False, encoding="utf-8")
    print(f"✓ CSV 數據已儲存至: {csv_path}")

if __name__ == "__main__":
    main()
