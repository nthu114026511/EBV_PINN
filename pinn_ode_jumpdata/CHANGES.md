# 修改說明：使用 CSV 數據計算 Data Loss

## 修改內容

### 1. 新增 pandas 套件
```python
import pandas as pd
```

### 2. 修改時間範圍
- 原本：`t_f = 200.0`
- 修改為：`t_f = 60.0`（根據 CSV 數據的實際範圍）

### 3. 替換數據來源
**修改前：**
- 使用程式內部計算的解析解（analytical solution）
- 需要複雜的積分計算 `F_alpha()` 函數
- 產生 10000 個時間點

**修改後：**
- 直接從 `evb_training_data.csv` 讀取數據
- 包含 606 個數據點（0.1 時間間隔）
- 時間範圍：t ∈ [0, 60]

### 4. 數據讀取流程
```python
# 讀取 CSV
df = pd.read_csv("evb_training_data.csv")

# 時間縮放：將原始時間 [0, 60] 轉換為 [0, 1]
t_scaled = (t_original - t_0) / (t_f - t_0)

# 取得 B, R, E 數值
B_csv = df['B'].values
R_csv = df['R'].values
E_csv = df['E'].values
```

### 5. Validator 設定調整
- batch_size：從固定的 10000 改為 `min(n_data_points, 1000)`
- 避免內存問題，同時適應不同數據量

## Loss 計算方式

現在的 Total Loss = Physics Loss + Data Loss，其中：

1. **Physics Loss**（來自 ODE 殘差 + 初始條件）
   - IC loss: λ = 100
   - ODE_B loss: λ = 1
   - ODE_R loss: λ = 5
   - ODE_E loss: λ = 5

2. **Data Loss**（來自 CSV 數據）
   - 使用 `PointwiseValidator` 計算 MSE
   - 比較模型預測和 CSV 中的真實值
   - Loss = mean[(B_pred - B_csv)² + (R_pred - R_csv)² + (E_pred - E_csv)²]

## CSV 數據格式

```
t,B,R,E
0,0.05,0,0
0.1,0.0504771431680392,0.0014921985480682,1.48733474717969e-05
...
60,0.683196946787212,0.850407383519269,0.844109442619396
```

- 共 606 行（含表頭）
- 時間間隔：0.1
- 時間範圍：[0, 60]

## 優點

1. **使用真實/實驗數據**：不再依賴理論解析解
2. **更靈活**：可以輕鬆更換不同的數據集
3. **更符合實際應用**：在真實場景中通常沒有解析解，只有觀測數據
4. **簡化程式碼**：移除了複雜的積分計算部分
