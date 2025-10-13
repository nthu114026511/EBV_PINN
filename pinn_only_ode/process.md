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

## 🎯 訓練策略配置

### 網路架構
```
Input (1D) → 6 × Hidden(512) → Output (3D: B, R, E)
```
- **隱藏層**: 6 層，每層 512 個神經元
- **總參數量**: ~1.31M 參數
- **激活函數**: ReLU/Tanh

### 訓練超參數
```yaml
最大步數:     10,000 steps
優化器:       Adam
學習率調度:   指數衰減 (lr × 0.95^(step/100))
  - decay_rate: 0.95
  - decay_steps: 100
損失函數:     Sum (IC Loss + Interior Loss)
記錄頻率:     每 1000 步
```

### 約束條件權重配置
```python
# 1. 初始條件 (IC) - 強化權重
IC:
  - Batch size: 600
  - Lambda weighting: B=100.0, R=100.0, E=100.0

# 2. 內點約束 - 早期/後期分段採樣
Interior Early (x < 0.3):
  - Batch size: 4,200 (70%)
  - Lambda weighting: B=1.0, R=5.0, E=5.0
  
Interior Late (x ≥ 0.3):
  - Batch size: 1,800 (30%)
  - Lambda weighting: B=1.0, R=5.0, E=5.0
```

**設計理念**:
- ✅ **IC 強化** (λ=100): 確保初始條件精確滿足
- ✅ **R/E 重於 B** (5:1): 因 R/E 量級較小，需更高權重
- ✅ **早期密集採樣** (70%): 捕捉快速變化區域
- ✅ **相對殘差**: 避免不同變數量級差異影響訓練

---

### 時間縮放
為改善訓練，將 `t ∈ [0, 200]` 縮放到 `x ∈ [0, 1]`:
```
x = t / 200  →  dx/dt = 1/200
dB/dx = 200 · dB/dt = 200 · r·B·(1 - B/K)
```

### 初始條件
```python
B(0) = 0.05,  R(0) = 0.0,  E(0) = 0.0
```

---

## 🔬 程式實現流程

### 1. CustomODE 類別定義

將 ODE 轉換為 PINN 可訓練的形式：

```python
class CustomODE(PDE):
    def __init__(self, r, K, sigma0, k12, kc, time_scale):
        # 定義符號變數
        x = Symbol("x")  # 縮放後的時間 ∈ [0,1]
        B, R, E = Function("B")(x), Function("R")(x), Function("E")(x)
        
        # 計算原始殘差
        resB = B.diff(x) - time_scale * r * B * (1 - B/K)
        resR = R.diff(x) - time_scale * (sigma0 * B - k12 * R)
        resE = E.diff(x) - time_scale * (k12 * R - kc * E)
        
        # 相對殘差正規化 (避免量級差異)
        B_scale = K
        R_scale = K * sigma0 / k12
        E_scale = K * sigma0 / kc
        
        self.equations = {
            "ode_B": resB / sqrt(B² + ε²·B_scale²),
            "ode_R": resR / sqrt(R² + ε²·R_scale² + R_scale²),
            "ode_E": resE / sqrt(E² + ε²·E_scale² + E_scale²),
        }
```

**關鍵技術**:
- **相對殘差**: 用 `sqrt(變數² + 典型尺度²)` 正規化，避免 `abs()` 梯度問題
- **典型尺度**: 根據穩態分析估計 (B~K, R~K·σ₀/k₁₂, E~K·σ₀/kc)

### 2. 神經網路構建

```python
FC = instantiate_arch(
    input_keys=[Key("x")],              # 輸入: 縮放時間
    output_keys=[Key("B"), Key("R"), Key("E")],  # 輸出: 三個變數
    cfg=cfg.arch.fully_connected,
)
nodes = ode.make_nodes() + [FC.make_node(name="FC")]
```

### 3. 約束條件設定

```python
# (1) 初始條件約束
IC = PointwiseBoundaryConstraint(
    nodes=nodes,
    geometry=Line1D(0.0, 1.0),
    outvar={"B": 0.05, "R": 0.0, "E": 0.0},
    lambda_weighting={"B": 100.0, "R": 100.0, "E": 100.0},
    batch_size=600,
    parameterization={x: 0.0},  # 固定在 x=0
)

# (2) 早期內點約束 (x < 0.3)
interior_early = PointwiseInteriorConstraint(
    nodes=nodes,
    geometry=Line1D(0.0, 1.0),
    outvar={"ode_B": 0, "ode_R": 0, "ode_E": 0},
    lambda_weighting={"ode_B": 1.0, "ode_R": 5.0, "ode_E": 5.0},
    criteria=StrictLessThan(x, 0.3),
    batch_size=4200,
)

# (3) 後期內點約束 (x ≥ 0.3)
interior_late = PointwiseInteriorConstraint(
    ...
    criteria=~StrictLessThan(x, 0.3),
    batch_size=1800,
)
```

### 4. 解析解計算 (用於驗證)

#### B(t) - Logistic 方程解析解
```python
c = (K - B0) / B0
B(t) = K / (1 + c·exp(-r·t))
```

#### R(t) - 線性 ODE 解析解
```python
R(t) = exp(-k12·t) · [R0 + σ₀·∫₀ᵗ exp(k12·s)·B(s) ds]
```

#### E(t) - 線性 ODE 解析解
```python
# 當 kc ≠ k12 時
E(t) = exp(-kc·t)·E0 
     + (k12/(kc-k12))·[exp(-k12·t) - exp(-kc·t)]·R0
     + σ₀·(k12/(kc-k12))·[exp(-k12·t)·F_k12 - exp(-kc·t)·F_kc]
```
其中 `F_α(t) = ∫₀ᵗ exp(α·s)·B(s) ds` 由數值積分計算。

### 5. 驗證與繪圖

```python
validator = PointwiseValidator(
    nodes=nodes, 
    invar={"x": 10000個測試點},
    true_outvar={"B": B_true, "R": R_true, "E": E_true},
    batch_size=10000,
    plotter=CustomValidatorPlotter(),
)
```

**誤差指標**:
- **L² error**: `√(mean((pred - true)²))`
- **L∞ error**: `max(|pred - true|)`

---

##  輸出結果

### 目錄結構
```
outputs/PINN_vs_exactsolution_final/
├── BRE_comparison.png           # 三變數對比圖
├── FC.0.pth                     # 網路權重
├── optim_checkpoint.0.pth       # 優化器狀態
├── events.out.tfevents.*        # TensorBoard 日誌
├── constraints/
│   ├── IC.vtp                   # 初始條件點
│   └── interior.vtp             # 內部約束點
└── validators/
    └── validator.vtp            # 驗證點
```

### 終端輸出範例
```
============================================================
Error Analysis:
============================================================
B - L2 error: 1.23e-03,  L∞ error: 5.67e-03
R - L2 error: 2.34e-04,  L∞ error: 8.90e-04
E - L2 error: 3.45e-04,  L∞ error: 9.01e-04
============================================================

Total training time: 123.45 seconds
```

---

## 🎨 CustomValidatorPlotter

自動生成 3 個子圖比較 PINN 預測與精確解：

```python
fig, axes = plt.subplots(3, 1, figsize=(10, 12))

# 子圖 1: B(x) 比較 (藍色實線 vs 紅色虛線)
# 子圖 2: R(x) 比較 (綠色實線 vs 橙色虛線)  
# 子圖 3: E(x) 比較 (紫色實線 vs 棕色虛線)

# 每個子圖標題顯示 L² 和 L∞ 誤差
```

---

## ⚙️ 運行方式

### 訓練模式 (預設)
```bash
python PINN_vs_exactsolution_final.py
```

### 評估模式
```bash
python PINN_vs_exactsolution_final.py run_mode=eval
```

### 使用 TensorBoard 監控
```bash
tensorboard --logdir outputs/PINN_vs_exactsolution_final
```

---

## 🔧 調優建議

### 若誤差過大
1. **增加訓練步數**: `max_steps: 20000`
2. **增大網路**: `layer_size: 1024`, `nr_layers: 8`
3. **增加採樣點**: `interior: 10000`
4. **調整權重**: IC λ 調高到 200，或 R/E λ 調到 10

### 若訓練過慢
1. **減小網路**: `layer_size: 256`, `nr_layers: 4`
2. **減少採樣**: `interior: 3000`
3. **使用 GPU**: 確保 CUDA 可用

### 實驗參數
- **時間分界點**: 將 `0.3` 改為 `0.2` (更早期密集)
- **權重比例**: 將 `wR, wE = 5.0` 改為 `10.0` (更強調 R/E)
- **IC 權重**: 將 `100.0` 改為 `200.0` (更強調初始條件)

---

## 💡 核心優勢

1. **物理約束內嵌**: Loss 直接包含 ODE residual
2. **自動微分**: 無數值截斷誤差
3. **連續解**: 可在任意點評估，無需網格
4. **GPU 加速**: 大規模並行計算
5. **彈性高**: 易於擴展到更複雜系統

---

## 📚 參考資料

- **PINN 原論文**: Raissi et al., "Physics-informed neural networks..." (2019)
- **Logistic 模型**: Verhulst population model
- **PhysicsNeMo**: NVIDIA Modulus 框架文檔

---

## 🎯 總結

這個 PINN 實現展示了如何：
1. ✅ 將 ODE 系統轉換為神經網路訓練問題
2. ✅ 使用相對殘差處理多尺度問題
3. ✅ 設計分段採樣策略捕捉動態變化
4. ✅ 強化關鍵約束 (IC 和小量級變數)
5. ✅ 與解析解對比驗證精度

**適用場景**: 解析解難求或數值方法不穩定的 ODE/PDE 系統