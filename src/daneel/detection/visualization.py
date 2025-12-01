from sklearn.metrics import (
    confusion_matrix, classification_report, accuracy_score,
    roc_auc_score, roc_curve, precision_recall_curve, average_precision_score
)
import matplotlib.pyplot as plt
import numpy as np

class Visual:
    def __init__(self):
        pass
        
    def evaluate_with_optimal_threshold(self,model, X_test, y_test):
        print("\n" + "="*70)
        print("THRESHOLD OPTIMIZATION & EVALUATION")
        print("="*70)
            
        proba = model.predict_proba(X_test)[:, 1]
        fpr, tpr, thresholds = roc_curve(y_test, proba)
        j_scores = tpr - fpr
        best_idx = np.argmax(j_scores)
        best_thresh = thresholds[best_idx]
            
        y_pred_default = (proba >= 0.5).astype(int)
        y_pred_best = (proba >= best_thresh).astype(int)
            
        auc = roc_auc_score(y_test, proba)
        acc_default = accuracy_score(y_test, y_pred_default)
        acc_best = accuracy_score(y_test, y_pred_best)
            
        print(f"Optimal threshold: {best_thresh:.4f} (default=0.5)")
        print(f"  At this threshold: TPR={tpr[best_idx]:.4f}, FPR={fpr[best_idx]:.4f}")
        print(f"AUC-ROC: {auc:.4f}")
        print(f"Accuracy @0.5: {acc_default:.4f} ({acc_default*100:.2f}%)")
        print(f"Accuracy @{best_thresh:.4f}: {acc_best:.4f} ({acc_best*100:.2f}%)")
            
        print("\nClassification report (optimal threshold):")
        print(classification_report(y_test, y_pred_best, target_names=['Non-Planet','Planet'], digits=4, zero_division=0))
            
        return y_pred_best, proba, best_thresh, (fpr, tpr, thresholds)

    def plot_confusion_matrix_image(self,y_true, y_pred, threshold, save_path='confusion_matrix_rf.png'):
        cm = confusion_matrix(y_true, y_pred)
        print(cm)
        fig = plt.figure(figsize=(6, 5))
        ax = plt.gca()
        im = ax.imshow(cm, interpolation='nearest')
        ax.figure.colorbar(im, ax=ax)
        ax.set(xticks=np.arange(2),
            yticks=np.arange(2),
            xticklabels=['Non-Planet', 'Planet'],
            yticklabels=['Non-Planet', 'Planet'],
            xlabel='Predicted', ylabel='True',
            title=f'Confusion Matrix (threshold={threshold:.3f})')
        total = cm.sum()
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                count = cm[i, j]
                pct = (count / total * 100) if total > 0 else 0.0
                ax.text(j, i, f"{count}\n({pct:.1f}%)", ha='center', va='center')
        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        print(f"Saved: {save_path}")
        plt.close(fig)

    def plot_roc_curve(self,y_true, proba, save_path='roc_curve_rf.png'):
        fpr, tpr, _ = roc_curve(y_true, proba)
        auc = roc_auc_score(y_true, proba)
        fig = plt.figure(figsize=(6, 5))
        ax = plt.gca()
        ax.plot(fpr, tpr, linewidth=2)
        ax.plot([0,1], [0,1], linestyle='--')
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title(f'ROC Curve (AUC={auc:.4f})')
        ax.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        print(f"Saved: {save_path}")
        plt.close(fig)

    def plot_pr_curve(self,y_true, proba, save_path='pr_curve_rf.png'):
        precision, recall, _ = precision_recall_curve(y_true, proba)
        ap = average_precision_score(y_true, proba)
        fig = plt.figure(figsize=(6, 5))
        ax = plt.gca()
        ax.plot(recall, precision, linewidth=2)
        ax.set_xlabel('Recall')
        ax.set_ylabel('Precision')
        ax.set_title(f'Precision-Recall Curve (AP={ap:.4f})')
        ax.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        print(f"Saved: {save_path}")
        plt.close(fig)

    def plot_probability_histograms(self,y_true, proba, save_path='probability_hist_RF.png'):
        fig = plt.figure(figsize=(6, 5))
        ax = plt.gca()
        ax.hist(proba[y_true==0], bins=30, alpha=0.6, label='Non-Planet', density=True)
        ax.hist(proba[y_true==1], bins=30, alpha=0.6, label='Planet', density=True)
        ax.set_xlabel('Predicted Probability (class=1)')
        ax.set_ylabel('Density')
        ax.set_title('Predicted Probability Distributions by Class')
        ax.legend()
        ax.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        print(f"Saved: {save_path}")
        plt.close(fig)

    def plot_lightcurve_sample(self,idx, X_test_standardized, X_err_test, metadata_test, scaler=None, proba=None, y_true=None, y_pred=None, save_prefix='sample_lightcurve_RF'):
        # Make a single-figure plot for one sample (no subplots)
        x_std = X_test_standardized[idx].reshape(1, -1)
        if scaler is not None:
            x_orig = scaler.inverse_transform(x_std).flatten()
            yerr = X_err_test[idx]
        else:
            x_orig = x_std.flatten()
            yerr = X_err_test[idx]
        
        fig = plt.figure(figsize=(10, 4))
        ax = plt.gca()
        ax.errorbar(np.arange(len(x_orig)), x_orig, yerr=yerr, fmt='o', markersize=2, alpha=0.6)
        ax.axhline(np.median(x_orig), linestyle='--', linewidth=1)
        ax.set_xlabel('Time Bin')
        ax.set_ylabel('Flux')
        
        toi = metadata_test.loc[idx, 'toi_name']
        tic = metadata_test.loc[idx, 'tic']
        disp = metadata_test.loc[idx, 'disp']
        sector = metadata_test.loc[idx, 'sector']
        
        tstr = f'TOI {toi} (TIC {tic}, {disp}) - Sector {sector}'
        if proba is not None and y_true is not None and y_pred is not None:
            pred_str = 'Transit' if y_pred[idx]==1 else 'Non-Transit'
            true_str = 'Transit' if y_true[idx]==1 else 'Non-Transit'
            tstr += f'\nTrue: {true_str} | Pred: {pred_str} (p={proba[idx]:.3f})'
        
        ax.set_title(tstr)
        ax.grid(alpha=0.3)
        plt.tight_layout()
        path = f"{save_prefix}_{idx}.png"
        plt.savefig(path, dpi=300)
        print(f"Saved: {path}")
        plt.close(fig)