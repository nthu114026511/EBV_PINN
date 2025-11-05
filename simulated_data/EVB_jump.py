#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RK4 with fraction (impulse) jumps under LQ model — Python 版本
對應 MATLAB 腳本：
  連續 ODE（跳躍間）:
    B' = r*B*(1 - B/K)
    R' = sigma0*B - k12*R
    E' = k12*R - kc*E
  跳躍於 t = t_j, 劑量 d_j:
    SF = exp(-alpha*d_j - beta*d_j^2)
    B^+ = B^- * SF
    R^+ = R^- + sigma1 * (B^- - B^+)
    E^+ = E^- (連續)

輸出：
  - 圖：B, R, E 隨時間，並以垂直虛線標記每次分次
  - CSV：./data/evb_training_data.csv ，欄位 t, B, R, E
  - 印出每次分次的 SF 與跳躍量（類似 MATLAB 的 struct2table）
"""

from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Callable, Tuple, List, Dict

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# -------------------------- 參數與初值（對齊你的 MATLAB 設定） --------------------------
r      = 0.10     # logistic growth rate
K      = 5.00     # carrying capacity (was 1.0)
sigma0 = 0.01     # base transfer rate (was 0.30)
sigma1 = 0.20     # fraction of killed B converted to R (was 1.00)
k12    = 0.05     # R→E transfer rate (was 0.20)
kc     = 0.15     # E clearance rate

B0, R0, E0 = 0.05, 0.00, 0.00

# LQ 參數與分次時刻/劑量（可自行調整）
alpha = 0.30      # 腫瘤 alpha (1/Gy)
beta  = 0.03      # 腫瘤 beta  (1/Gy^2)

t_frac = np.array([10.0, 20.0, 40.0])   # 跳躍時刻
d_frac = np.array([1.5,  1.5,  1.5 ])   # 對應劑量

# 全域時間
t0, tf = 0.0, 60.0
h = 0.1  # 基本步長；事件前會自動縮步以精確對齊 t_j

# 打包參數
pars: Dict[str, float] = dict(r=r, K=K, sigma0=sigma0, k12=k12, kc=kc)


# -------------------------- ODE 右端（連續段） --------------------------
def rhs(t: float, y: np.ndarray, p: Dict[str, float]) -> np.ndarray:
    """y = [B, R, E]. 返回 dy/dt。"""
    B, R, E = y
    dB = p["r"] * B * (1.0 - B / p["K"])
    dR = p["sigma0"] * B - p["k12"] * R
    dE = p["k12"] * R - p["kc"] * E
    return np.array([dB, dR, dE], dtype=float)


def rk4_step(f: Callable[[float, np.ndarray, Dict[str, float]], np.ndarray],
             t: float, y: np.ndarray, h: float, p: Dict[str, float]) -> np.ndarray:
    """單步 RK4；附帶非負裁切避免數值微幅負值。"""
    k1 = f(t,           y,               p)
    k2 = f(t + 0.5*h,   y + 0.5*h*k1,    p)
    k3 = f(t + 0.5*h,   y + 0.5*h*k2,    p)
    k4 = f(t + h,       y + h*k3,        p)
    y1 = y + (h/6.0) * (k1 + 2*k2 + 2*k3 + k4)
    return np.maximum(y1, 0.0)  # 保證非負（可選）


# -------------------------- 分次事件紀錄 --------------------------
@dataclass
class JumpRecord:
    t: float
    dose: float
    SF: float
    dB_instant: float
    R_gain: float


# -------------------------- 分段 RK4 + 跳躍 --------------------------
def rk4_with_fractions(f: Callable[[float, np.ndarray, Dict[str, float]], np.ndarray],
                       tspan: Tuple[float, float],
                       y0: np.ndarray,
                       h: float,
                       t_frac: np.ndarray,
                       d_frac: np.ndarray,
                       alpha: float,
                       beta: float,
                       sigma1: float,
                       p: Dict[str, float]
                       ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[JumpRecord]]:
    """
    在各跳點之間用 RK4 積分；到跳點時施加 LQ 跳躍。
    回傳：
      T: (N,) 時間序列
      Y: (N,3) 對應的狀態 [B,R,E]
      Tjump: (M,) 實際跳點時間（已排序且落在 (t0,tf)）
      JumpInfo: 每次分次的紀錄（SF、瞬時殺死量等）
    """
    t0, tf = float(tspan[0]), float(tspan[1])

    # 僅保留 (t0, tf) 內的事件並依時間排序
    mask = (t_frac > t0) & (t_frac < tf)
    t_events = t_frac[mask].astype(float)
    d_events = d_frac[mask].astype(float)
    order = np.argsort(t_events)
    t_events = t_events[order]
    d_events = d_events[order]
    n_events = len(t_events)

    # 輸出容器（與 MATLAB 對齊：先放初始點）
    T = [t0]
    Y = [np.asarray(y0, dtype=float)]
    Tjump: List[float] = []
    JumpInfo: List[JumpRecord] = []

    t = t0
    y = np.asarray(y0, dtype=float)

    for j in range(n_events + 1):
        t_end = t_events[j] if j < n_events else tf

        # 積分至 t_end（縮步對齊）
        while t < t_end - 1e-14:
            hstep = min(h, t_end - t)
            y = rk4_step(f, t, y, hstep, p)
            t = t + hstep
            T.append(t)
            Y.append(y.copy())

        # 到達事件：施加跳躍
        if j < n_events:
            Bm, Rm, Em = y  # minus state
            dose = float(d_events[j])
            SF = float(np.exp(-alpha * dose - beta * dose * dose))
            Bp = Bm * SF
            dB_inst = Bm - Bp
            Rp = Rm + sigma1 * dB_inst
            Ep = Em  # 連續

            y = np.array([Bp, Rp, Ep], dtype=float)

            # 與 MATLAB 保持一致：在同一個 t 再追加一次（跳躍後的狀態）
            T.append(t)
            Y.append(y.copy())

            Tjump.append(t)
            JumpInfo.append(JumpRecord(t=t, dose=dose, SF=SF,
                                       dB_instant=float(dB_inst),
                                       R_gain=float(sigma1 * dB_inst)))

    return np.array(T), np.vstack(Y), np.array(Tjump), JumpInfo


def main():
    # 初值
    y0 = np.array([B0, R0, E0], dtype=float)

    # 積分 + 跳躍
    T, Y, Tjump, JumpInfo = rk4_with_fractions(
        f=rhs, tspan=(t0, tf), y0=y0, h=h,
        t_frac=t_frac, d_frac=d_frac,
        alpha=alpha, beta=beta, sigma1=sigma1, p=pars
    )

    B, R, E = Y[:, 0], Y[:, 1], Y[:, 2]

    # 畫圖
    fig, axes = plt.subplots(3, 1, figsize=(9, 8), sharex=True)
    axes[0].plot(T, B, linewidth=1.5)
    axes[0].set_ylabel("B")
    axes[0].set_title("B, R, E with fraction (LQ) jumps")

    axes[1].plot(T, R, linewidth=1.5)
    axes[1].set_ylabel("R")

    axes[2].plot(T, E, linewidth=1.5)
    axes[2].set_ylabel("E")
    axes[2].set_xlabel("t")

    for ax in axes:
        for tj in Tjump:
            ax.axvline(tj, linestyle=":", linewidth=1.0, color="k")

    fig.tight_layout()
    plt.show()

    # 列印每次分次的 SF 與跳躍量（類似 MATLAB 的 struct2table）
    if len(JumpInfo) > 0:
        df_jump = pd.DataFrame([asdict(j) for j in JumpInfo])
        print(df_jump.to_string(index=False))
    else:
        print("（此區間內沒有任何分次事件）")

    # 儲存 CSV
    base_path = os.getcwd()
    output_dir = os.path.join(base_path, "data")
    os.makedirs(output_dir, exist_ok=True)

    df = pd.DataFrame({"t": T, "B": B, "R": R, "E": E})
    csv_path = os.path.join(output_dir, "evb_training_data_jump.csv")
    df.to_csv(csv_path, index=False, encoding="utf-8")
    print(f"✓ CSV 數據已儲存至: {csv_path}")


if __name__ == "__main__":
    main()
