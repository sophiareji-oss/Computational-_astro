import numpy as np
import joblib

from .data_process import *
from .classifiers import *
from .visualization import *

class ML:
    def __init__(self,CSV_PATH,N_BINS,USE_SCALER,SAMPLES_PER_CLASS,MODEL):
        self.CSV_PATH = CSV_PATH   
        self.N_BINS = N_BINS
        self.USE_SCALER = USE_SCALER                  
        self.SAMPLES_PER_CLASS = SAMPLES_PER_CLASS
        self.MODEL=MODEL

    def main(self):
        data = Data_process(csv_path=self.CSV_PATH, n_bins=self.N_BINS,
                             use_scaler=self.USE_SCALER, samples_per_class=self.SAMPLES_PER_CLASS)

        X_train, X_test, y_train, y_test, metadata_test, X_test_orig, X_err_test, scaler = data.load_data()

        if self.MODEL =='rf':
            rf = RFC(
                n_estimators=500,
                max_depth=None,
                min_samples_leaf=1,
                max_features='sqrt',
                bootstrap=True,
                class_weight=None,     # already balanced via augmentation
                random_state=42,
                n_jobs=-1,
                oob_score=True
            )
            rf_model = rf.build_random_forest()
            rf_trained = rf.train_model(rf_model, X_train, y_train)

            vs = Visual()

            y_pred_opt, proba_test, best_thresh, roc_tuple = vs.evaluate_with_optimal_threshold(rf_trained, X_test, y_test)
            # Visualizations
            vs.plot_confusion_matrix_image(y_test, y_pred_opt, best_thresh)
            vs.plot_roc_curve(y_test, proba_test, save_path='roc_curve_rf.png')
            vs.plot_pr_curve(y_test, proba_test, save_path='pr_curve_rf.png')
            vs.plot_probability_histograms(y_test, proba_test, save_path='probability_hist_RF.png')

            # Optional: save a handful of single-sample light curves with predictions
            # (No subplots; one figure per sample.)
            num_samples = min(4, len(X_test_orig))
            idxs = np.random.choice(len(X_test_orig), num_samples, replace=False)
            for i in idxs:
                vs.plot_lightcurve_sample(
                    i, X_test_orig, X_err_test, metadata_test,
                    scaler=scaler, proba=proba_test, y_true=y_test, y_pred=y_pred_opt,
                    save_prefix='sample_lightcurve_RF'
                )

            joblib.dump(rf, 'tess_rf_model.joblib')
            np.save('rf_optimal_threshold.npy', best_thresh)
            print("Saved: tess_rf_model.joblib, rf_optimal_threshold.npy")

        elif self.MODEL == 'dt':
            dt = DT()
            dt = dt.train_model(X_train,y_train)
            vs = Visual()
            y_pred_opt, proba_test, best_thresh, roc_tuple = vs.evaluate_with_optimal_threshold(dt, X_test, y_test)
            # Visualizations
            vs.plot_confusion_matrix_image(y_test, y_pred_opt, best_thresh,save_path='confusion_matrix_dt.png')
            vs.plot_roc_curve(y_test, proba_test, save_path='roc_curve_dt.png')
            vs.plot_pr_curve(y_test, proba_test, save_path='pr_curve_dt.png')
            vs.plot_probability_histograms(y_test, proba_test, save_path='probability_hist_dt.png')

        elif self.MODEL == 'cnn':
            model = DL.build_simple_cnn(n_bins=self.N_BINS)
            
            history = DL.train_model(model, X_train, y_train, X_test, y_test, epochs=200)

            vs = Visual()
            y_pred, y_pred_proba, threshold = vs.evaluate_with_optimal_threshold(model, X_test, y_test,opt_thres = True)
           
            vs.plot_all(y_test, y_pred, y_pred_proba, history, metadata_test, X_test, threshold,
                    X_test_orig, X_err_test, scaler)

            
            model.save('tess_model_final.keras')
            np.save('optimal_threshold.npy', threshold)

            print("\n" + "="*70)
            print("TRAINING COMPLETE!")
            print("="*70)
            print("\nKey improvements:")
            print("  ✓ Perfectly balanced training data")
            print("  ✓ Focal loss for hard examples")
            print("  ✓ Optimal threshold selection")
            print("  ✓ AUC-focused optimization")
            print("\nFiles:")
            print("  - tess_model_final.keras")
            print("  - best_model_final.keras")
            print("  - optimal_threshold.npy")
            print("  - confusion_matrix_final.png")
            print("  - training_history_final.png")
            print("  - sample_lightcurves_predictions.png")
            print("="*70)


