### Step 2 | 把「分次跳躍」嵌入到解的構造式

**理由**：直接在解的構造式保證 式 (1.2) and (1.3)（$\kappa \to \infty$ 時精確），避免以懲罰近似。

**方法**：用平滑階躍 $H_\kappa(\tau) = \sigma(\kappa\tau)$，設

$$M(t) = \exp\left(\sum_j (\log SF_j) H_\kappa(t - t_j)\right), \quad B(t) = M(t) \underline{B}(t), \tag{2.2}$$

$$\Delta B_j = B(t_j^-) - B(t_j^+) = \underline{B}(t_j^-)[1 - SF_j], \tag{2.3}$$

$$R(t) = \underline{R}(t) + \sigma_1 \sum_j \Delta B_j H_\kappa(t - t_j), \tag{2.4}$$

並令 E(t) 連續近似（不嵌入跳階）。依據 可微事件函數與混合流-跳系統（Neural
ODEs with events §1.1；Neural Jump-SDE）提供將跳躍以可微觸發函數嵌入的範式；
界面跳條件與 cPINN/XPINN 的介面強制思路完全相符 [5, 6, 7, 8]。

**檢核點**：隨 $\kappa$ 退火增大，$t_j$ 近旁圖像趨近垂直跳階。

#### 平滑階躍函數 $H_\kappa(\tau) = \sigma(\kappa\tau)$ 的解釋

平滑階躍函數 $H_\kappa(\tau) = \sigma(\kappa\tau)$ 是用來處理模型中「分次跳躍」（piecewise jumps）的關
鍵工具，特別嵌入解的構造式中（見 Step 2）。

##### 定義與數學形式

- **基本形式**：$H_\kappa(\tau)$ 是 sigmoid 激活函數 $\sigma(x) = \frac{1}{1+e^{-x}}$ 的縮放版本，其中 $x = \kappa\tau$。
  - $\tau = t - t_j$：相對於第 j 次事件時間 $t_j$ 的時間偏移（事件觸發後的「延遲」）。
  - $\kappa > 0$：陡峭度參數（steepness parameter），控制階躍的「銳利度」。

- **行為**：
  - 當 $\tau < 0$（事件前），$H_\kappa(\tau) \approx 0$（平滑過渡）。
  - 當 $\tau > 0$（事件後），$H_\kappa(\tau) \approx 1$（平滑過渡）。
  - 轉換點在 $\tau = 0$，轉換寬度約 $\sim 1/\kappa$。

- **極限情況**：當 $\kappa \to \infty$，$H_\kappa(\tau)$ 逼近 Heaviside 階躍函數 $H(\tau)$（理想步階：$H(\tau) =
0$ 若 $\tau < 0$，$H(\tau) = 1$ 若 $\tau > 0$），但保持可微分性，避免梯度中斷。

這是神經網路中常見的「軟階躧」（soft step）逼近，借鑑自可微事件函數（Neural
ODEs with events）與混合流-跳系統（Neural Jump-SDE）[5, 6]。

##### 為何需要平滑階躧？

- **問題**：EBV 模型包含連續 ODE（式 (2.1)）與不連續事件（LQ 放療模型的分次跳
躍，式 (2.2)–(2.4)），如 B 在 $t_j$ 處突然衰減 $SF_j$ 倍（生存分率），R 增加 $\sigma_1 \Delta B_j$。
  - 直接使用理想階躧 $H(\tau)$ 會在 $t_j$ 產生不連續，導致 PINN 訓練時自動微分
（autodiff）梯度爆炸或殘差尖峰。
  - 若用懲罰項（penalty）近似，會引入多損失失衡（multi-loss imbalance）。

- **解法**：將跳躍「內建」到解的構造式中，使用平滑 $H_\kappa$ 保證可微分性，允許端
到端優化。依據 cPINN/XPINN 的介面強制思路 [7, 8]，這能精確滿足跳躍條件
（$\kappa \to \infty$ 時），同時穩定訓練。

##### 在模型中的應用

構造式嵌入（式 (1.2) and (1.3)）：

$$M(t) = \exp\left(\sum_j (\log SF_j)H_\kappa(t - t_j)\right), \tag{2.4}$$

$$B(t) = M(t)\underline{B}(t), \tag{2.5}$$

$$\Delta B_j = B(t_j^-) - B(t_j^+) = \underline{B}(t_j^-)[1 - SF_j], \tag{2.6}$$

$$R(t) = \underline{R}(t) + \sigma_1 \sum_j \Delta B_j H_\kappa(t - t_j), \tag{2.7}$$

其中 $\underline{B}(t)$ 與 $\underline{R}(t)$ 是神經網路輸出，$M$ 提供跳躍疊加，E 保持連續。

**導數計算**（Step 3）：使用鏈式法則確保可微：

$$\dot{H}_\kappa(\tau) = \kappa\sigma(\kappa\tau)[1 - \sigma(\kappa\tau)] = \kappa H_\kappa(\tau)(1 - H_\kappa(\tau)), \tag{2.8}$$

這是 sigmoid 的導數，峰值在轉換點，避免脈衝誤當 ODE 殘差。

**$\kappa$ 退火**（annealing）（Step 7）：訓練初期 $\kappa$ 小（平滑，易優化）；後期逐步增大
（每 500–$10^3$ epochs 乘 1.5），逼近真跳躍，降低優化難度 [9, 10]。

##### 優點與檢核

- **優點**：使不連續系統可微，穩定 PINN 收斂；保留 LQ 物理可解釋性；適用多事
件（30+ 次）。

- **檢核點**：隨 $\kappa$ 增大，$t_j$ 附近圖像趨近垂直跳階（垂直線）。

- **文獻依據**：源自可微事件與界面條件 [5, 6, 7, 8]，在不連續 PINN 中加權平滑策
略 [11]。

此設計是為讓 split 訓練（Stage-1 資料、Stage-2 物理）更穩定，尤其在 LOD 雜訊
下。

---

### Step 3 | 事件遮罩與導數鏈式法則

**理由**：避免把事件脈衝誤當 ODE 殘差。

**方法**：每個 $t_j$ 設平滑遮罩 $w(t) = \prod_j \left(1 - \exp\left(-\frac{(t-t_j)^2}{2\varepsilon^2}\right)\right)$；自動微分時用

$$\dot{B} = \dot{M} \underline{B} + M \dot{\underline{B}},$$

$$\dot{R} = \dot{\underline{R}} + \sigma_1 \sum_j \Delta B_j \dot{H}_\kappa(t - t_j) + \sigma_1 \sum_j \Delta \dot{B_j} H_\kappa(t - t_j),$$

其中 $\dot{H}_\kappa(\tau) = \kappa \sigma(\kappa\tau) [1 - \sigma(\kappa\tau)]$。

依據 對不連續/銳層的 PINN 加權與平滑策略（Discontinuity Computing using PINNs）
以及課程式學習可顯著穩定殘差 [11]。

**檢核點**：關掉遮罩 $w \equiv 1$ 時，$t_j$ 附近殘差尖峰；打開遮罩恢復平穩。

---

### Step 4 | 三塊 Loss（核心）

#### 4.1 ODE 殘差（事件外）

$$L_{\text{ODE}} = \frac{1}{N_c} \sum_i w(t_i) \left[\left(\dot{B} - rB\left(1 - \frac{B}{K}\right)\right)^2 + \left(\dot{R} - (\sigma_0 B - k_{12} R)\right)^2 + \left(\dot{E} - (k_{12} R - k_c E)\right)^2\right], \quad t_i \notin \bigcup_j (t_j \pm \varepsilon) \tag{4.1}$$

$K \gg B$ 時可改第一項為 $(\dot{B} - rB)^2$。