from sklearn.metrics import (
    confusion_matrix, classification_report, accuracy_score,
    roc_auc_score, roc_curve, precision_recall_curve, average_precision_score
)
import matplotlib.pyplot as plt
import numpy as np

class Visual:
    def __init__(self):
        pass
        
    def evaluate_with_optimal_threshold(self,model, X_test, y_test,opt_thres = False):
        print("\n" + "="*70)
        print("THRESHOLD OPTIMIZATION & EVALUATION")
        print("="*70)

        if opt_thres:
            y_pred_proba = model.predict(X_test, verbose=0).flatten()
            fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
            
            # Youden's J statistic
            j_scores = tpr - fpr
            optimal_idx = np.argmax(j_scores)
            optimal_threshold = thresholds[optimal_idx]
            
            print(f"\nOptimal threshold: {optimal_threshold:.4f} (default=0.5)")
            print(f"  At this threshold: TPR={tpr[optimal_idx]:.4f}, FPR={fpr[optimal_idx]:.4f}")
            
            # Predictions with optimal vs default thresholds
            y_pred_optimal = (y_pred_proba >= optimal_threshold).astype(int)
            y_pred_default = (y_pred_proba >= 0.5).astype(int)
            
            # Metrics
            acc_optimal = accuracy_score(y_test, y_pred_optimal)
            acc_default = accuracy_score(y_test, y_pred_default)
            auc = roc_auc_score(y_test, y_pred_proba)
            
            print("\nResults:")
            print(f"  AUC-ROC: {auc:.4f}")
            print(f"  Accuracy (default threshold=0.5): {acc_default:.4f} ({acc_default*100:.2f}%)")
            print(f"  Accuracy (optimal threshold={optimal_threshold:.4f}): {acc_optimal:.4f} ({acc_optimal*100:.2f}%)")
            
            print("\nWith optimal threshold:")
            print(classification_report(y_test, y_pred_optimal,
                                        target_names=['Non-Planet', 'Planet'],
                                        digits=4,
                                        zero_division=0))
            
            print("\nPrediction distribution (optimal threshold):")
            print(f"  Predicted 0: {(y_pred_optimal == 0).sum()}")
            print(f"  Predicted 1: {(y_pred_optimal == 1).sum()}")
            print("True distribution:")
            print(f"  True 0: {(y_test == 0).sum()}")
            print(f"  True 1: {(y_test == 1).sum()}")
            
            return y_pred_optimal, y_pred_proba, optimal_threshold
        else:
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

    def plot_all(self, y_test, y_pred, y_pred_proba, history, metadata_test, X_test, threshold,
             X_test_orig, X_err_test, scaler):
        """Create and save confusion matrix and training curves. Optionally plot light curves."""
        print("\n" + "="*70)
        print("VISUALIZATIONS")
        print("="*70)
        
        # Confusion matrix (Matplotlib-only)
        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots(figsize=(8, 6))
        im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
        ax.figure.colorbar(im, ax=ax)
        ax.set(xticks=np.arange(2),
            yticks=np.arange(2),
            xticklabels=['Non-Planet', 'Planet'],
            yticklabels=['Non-Planet', 'Planet'],
            xlabel='Predicted', ylabel='True',
            title=f'Confusion Matrix (threshold={threshold:.3f})')
        
        # Add counts and percentages
        total = cm.sum()
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                count = cm[i, j]
                pct = (count / total * 100) if total > 0 else 0.0
                ax.text(j, i, f"{count}\n({pct:.1f}%)", ha='center', va='center', color='black', fontsize=10)
        
        plt.tight_layout()
        plt.savefig('confusion_matrix_final.png', dpi=300)
        print("Saved: confusion_matrix_final.png")
        plt.close()
        
        # Training history
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        metrics = [('loss', 'Loss'), ('accuracy', 'Accuracy'),
                ('auc', 'AUC'), ('recall', 'Recall')]
        
        for idx, (metric, title) in enumerate(metrics):
            ax = axes[idx // 2, idx % 2]
            if metric in history.history and f'val_{metric}' in history.history:
                ax.plot(history.history[metric], label='Train', linewidth=2)
                ax.plot(history.history[f'val_{metric}'], label='Val', linewidth=2)
                ax.set_xlabel('Epoch')
                ax.set_ylabel(title)
                ax.set_title(f'{title} vs Epoch', fontweight='bold')
                ax.legend()
                ax.grid(alpha=0.3)
        
        plt.suptitle('Training History - Final Model', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig('training_history_final.png', dpi=300)
        print("Saved: training_history_final.png")
        plt.close()
        
        # Optional: light-curve panel
        if X_test_orig is not None and X_err_test is not None and scaler is not None:
            plot_lightcurves_with_predictions(X_test_orig, X_err_test, y_test, y_pred, 
                                            y_pred_proba, metadata_test, scaler, threshold, n_samples=6)
            

def plot_lightcurves_with_predictions(X_test_orig, X_err_test, y_test, y_pred, y_pred_proba, 
                                   metadata_test, scaler, threshold, n_samples=6,
                                    save_path='sample_lightcurves_predictions.png'):
    """Plot light curves with error bars and prediction info; save to file."""
    print("\n" + "="*70)
    print(f"PLOTTING LIGHTCURVES WITH PREDICTIONS (n={n_samples})")
    print("="*70)
        
    n_samples = min(n_samples, len(X_test_orig))
        
    # Select diverse samples: correct/incorrect for both classes
    correct_planet = np.where((y_test == 1) & (y_pred == 1))[0]
    incorrect_planet = np.where((y_test == 1) & (y_pred == 0))[0]
    correct_nonplanet = np.where((y_test == 0) & (y_pred == 0))[0]
    incorrect_nonplanet = np.where((y_test == 0) & (y_pred == 1))[0]
        
    selected_idx = []
    per_category = max(1, n_samples // 4)
        
    for idx_list in [correct_planet, incorrect_planet, correct_nonplanet, incorrect_nonplanet]:
        if len(idx_list) > 0:
            n_select = min(per_category, len(idx_list))
            selected_idx.extend(np.random.choice(idx_list, n_select, replace=False))
        
    while len(selected_idx) < n_samples:
        remaining = list(set(range(len(y_test))) - set(selected_idx))
        if remaining:
            selected_idx.append(np.random.choice(remaining))
        else:
            break
        
    selected_idx = np.array(selected_idx[:n_samples])

    #Figure layout
    n_cols = 2
    n_rows = (n_samples + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 4*n_rows))
    if n_samples == 1:
        axes = np.array([axes])
    axes = axes.flatten()
        
    for plot_i, idx in enumerate(selected_idx):
        ax = axes[plot_i]
            
        # Inverse transform to original scale for plotting
        flux_norm = X_test_orig[idx].flatten()
        flux_err = X_err_test[idx]
        flux_original = scaler.inverse_transform(flux_norm.reshape(1, -1)).flatten()
            
        time_bins = np.arange(len(flux_original))
            
        # Metadata
        toi_name = metadata_test.loc[idx, 'toi_name']
        tic = metadata_test.loc[idx, 'tic']
        disp = metadata_test.loc[idx, 'disp']
        sector = metadata_test.loc[idx, 'sector']
            
        true_label = y_test[idx]
        pred_label = y_pred[idx]
        pred_prob = y_pred_proba[idx]
            
        is_correct = (true_label == pred_label)
        true_str = 'Transit' if true_label == 1 else 'Non-Transit'
        pred_str = 'Transit' if pred_label == 1 else 'Non-Transit'
            
        # Errorbar plot
        ax.errorbar(time_bins, flux_original, yerr=flux_err, fmt='o', markersize=2,
                        ecolor='gray', elinewidth=0.5, capsize=0, alpha=0.6, label='Data')
        
        # Baseline median
        baseline = np.median(flux_original)
        ax.axhline(baseline, linestyle='--', linewidth=1, alpha=0.7, label='Baseline')
            
        ax.set_xlabel('Time Bin', fontsize=10, fontweight='bold')
        ax.set_ylabel('Flux (original scale)', fontsize=10, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right', fontsize=8)
            
        status_symbol = '✓' if is_correct else '✗'
        color = 'green' if is_correct else 'red'
        title = (f'TOI {toi_name} (TIC {tic}, {disp}) - TESS Sector {sector}\n'
                    f'True: {true_str} | Pred: {pred_str} (p={pred_prob:.3f}) {status_symbol}')
        ax.set_title(title, fontsize=10, fontweight='bold', color=color, pad=10)
            
        for spine in ax.spines.values():
            spine.set_edgecolor(color)
            spine.set_linewidth(2.0)
        
    # Hide unused axes
    for j in range(n_samples, len(axes)):
        axes[j].axis('off')
        
    plt.suptitle(f'Sample Light-curve Predictions (Threshold={threshold:.3f})',
                fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.close()
