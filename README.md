# PINN 求解三變數 ODE 系統 - 程式說明

> 使用 **Physics-Informed Neural Networks (PINN)** 求解耦合 ODE 系統，並與解析解比較

---
## 📐 核心 ODE 系統

### 原始方程組
```
dB/dt = r·B·(1 - B/K)      # Logistic growth
dR/dt = σ₀·B - k₁₂·R       # R 的產生與消失
dE/dt = k₁₂·R - kc·E       # E 的產生與消失
```

**變數意義**:
- **B(t)**: 腫瘤細胞濃度
- **R(t)**: 放射性示蹤劑濃度  
- **E(t)**: 代謝產物濃度

### 物理參數
```python
r  = 0.10   # 成長率
K  = 5.0    # 承載容量
σ₀ = 0.01   # B→R 轉換率
k₁₂ = 0.05  # R 消失率
kc = 0.15   # E 消失率
```

---

## 📊 訓練配置

```
📊 訓練配置
├─ 神經網路：512節點 × 6層
├─ 總步數：10,000 步
├─ 學習率：指數衰減 (decay_rate=0.95, 每100步)
│
├─ 🎯 Initial Condition (IC)
│   ├─ Batch size: 600
│   └─ Lambda weighting: 100.0 (強化IC)
│
├─ 📈 Interior Early (x < 0.3)
│   ├─ Batch size: 4200 (70%)
│   └─ Lambda: wB=1.0, wR=5.0, wE=5.0
│
└─ 📉 Interior Late (x ≥ 0.3)
    ├─ Batch size: 1800 (30%)
    └─ Lambda: wB=1.0, wR=5.0, wE=5.0
```

**訓練流程**：
```
初始化 → For step=1 to 10,000:
  ├─ 採樣約束點 (IC: 600, Early: 4200, Late: 1800)
  ├─ 前向傳播: x → NN(x) → [B, R, E]
  ├─ 自動微分: 計算 dB/dx, dR/dx, dE/dx
  ├─ 計算損失: Loss = Loss_IC + Loss_Interior
  ├─ 反向傳播 + Adam 更新參數
  ├─ 每 100 步: lr ← lr × 0.95
  └─ 每 1000 步: 記錄結果到 TensorBoard
→ 訓練完成後自動生成對比圖
```

---