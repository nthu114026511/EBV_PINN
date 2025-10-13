import numpy as np
import matplotlib.pyplot as plt
from physicsnemo.sym.utils.io.plotter import ValidatorPlotter

class CustomValidatorPlotter(ValidatorPlotter):
    def __init__(self):
        super().__init__()
    
    def __call__(self, invar, true_outvar, pred_outvar):
        x_all = invar["x"][:, 0]
        
        # Extract true and predicted values
        B_true_all = true_outvar["B"][:, 0]
        B_pred_all = pred_outvar["B"][:, 0]
        
        R_true_all = true_outvar["R"][:, 0]
        R_pred_all = pred_outvar["R"][:, 0]
        
        E_true_all = true_outvar["E"][:, 0]
        E_pred_all = pred_outvar["E"][:, 0]

        # 計算誤差 - B
        error_B = B_pred_all - B_true_all
        l2_error_B = np.sqrt(np.mean(error_B**2))          # L2 norm
        sup_error_B = np.max(np.abs(error_B))              # L∞ norm

        # 計算誤差 - R
        error_R = R_pred_all - R_true_all
        l2_error_R = np.sqrt(np.mean(error_R**2))          # L2 norm
        sup_error_R = np.max(np.abs(error_R))              # L∞ norm

        # 計算誤差 - E
        error_E = E_pred_all - E_true_all
        l2_error_E = np.sqrt(np.mean(error_E**2))          # L2 norm
        sup_error_E = np.max(np.abs(error_E))              # L∞ norm

        print(f"\n{'='*60}")
        print(f"Error Analysis:")
        print(f"{'='*60}")
        print(f"B - L2 error: {l2_error_B:.4e},  L∞ error: {sup_error_B:.4e}")
        print(f"R - L2 error: {l2_error_R:.4e},  L∞ error: {sup_error_R:.4e}")
        print(f"E - L2 error: {l2_error_E:.4e},  L∞ error: {sup_error_E:.4e}")
        print(f"{'='*60}\n")

        # 繪圖 - 3個子圖
        fig, axes = plt.subplots(3, 1, figsize=(10, 12))
        
        # Plot B
        axes[0].plot(x_all, B_true_all, label="B_true", linewidth=2, color='blue')
        axes[0].plot(x_all, B_pred_all, label="B_pred", linestyle="--", linewidth=2, color='red')
        axes[0].set_xlabel("x", fontsize=12)
        axes[0].set_ylabel("B(x)", fontsize=12)
        axes[0].set_title(
            f"$B(x)$ - $L^2$ error: {l2_error_B:.2e}, $L^\\infty$ error: {sup_error_B:.2e}",
            fontsize=11
        )
        axes[0].legend(fontsize=10, loc='upper right')
        axes[0].grid(True, alpha=0.3)

        # Plot R
        axes[1].plot(x_all, R_true_all, label="R_true", linewidth=2, color='green')
        axes[1].plot(x_all, R_pred_all, label="R_pred", linestyle="--", linewidth=2, color='orange')
        axes[1].set_xlabel("x", fontsize=12)
        axes[1].set_ylabel("R(x)", fontsize=12)
        axes[1].set_title(
            f"$R(x)$ - $L^2$ error: {l2_error_R:.2e}, $L^\\infty$ error: {sup_error_R:.2e}",
            fontsize=11
        )
        axes[1].legend(fontsize=10, loc='upper right')
        axes[1].grid(True, alpha=0.3)

        # Plot E
        axes[2].plot(x_all, E_true_all, label="E_true", linewidth=2, color='purple')
        axes[2].plot(x_all, E_pred_all, label="E_pred", linestyle="--", linewidth=2, color='brown')
        axes[2].set_xlabel("x", fontsize=12)
        axes[2].set_ylabel("E(x)", fontsize=12)
        axes[2].set_title(
            f"$E(x)$ - $L^2$ error: {l2_error_E:.2e}, $L^\\infty$ error: {sup_error_E:.2e}",
            fontsize=11
        )
        axes[2].legend(fontsize=10, loc='upper right')
        axes[2].grid(True, alpha=0.3)

        # 主標題
        fig.suptitle("Comparison of Predicted and True Solutions for 3 ODE System", 
                     fontsize=14, fontweight='bold')

        plt.tight_layout(rect=[0, 0, 1, 0.97])

        # 儲存成圖片
        fig.savefig("./BRE_comparison.png", dpi=300, bbox_inches="tight")
        print(f"Figure saved as 'BRE_comparison.png'")

        return [(fig, "BRE_comparison")]
