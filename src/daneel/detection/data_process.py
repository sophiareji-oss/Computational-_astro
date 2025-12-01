import pandas as pd
import numpy as np
from scipy import ndimage, fft
from sklearn.preprocessing import normalize, StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split


RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)



class Data_process():
    def __init__(self,csv_path='tess_data.csv', n_bins=1000, use_scaler=False, samples_per_class=350):
        self.csv_path = csv_path
        self.n_bins = n_bins
        self.use_scaler = use_scaler
        self.samples_per_class = samples_per_class
        
    def load_data(self):
        print("\n" + "="*70)
        print("LOADING DATA")
        print("="*70)
        
        df = pd.read_csv(self.csv_path)
        print(f"Dataset: {df.shape[0]} samples")
        
        flux_cols = [f'flux_{i:04d}' for i in range(self.n_bins)]
        flux_err_cols = [f'flux_err_{i:04d}' for i in range(self.n_bins)]
        X = df[flux_cols].values
        X_err = df[flux_err_cols].values
        y = df['label'].values
        
        metadata_cols = ['toi_name', 'tic', 'label', 'disp', 'period_d', 't0_bjd', 'dur_hr', 'sector']
        metadata = df[metadata_cols]
        
        print("Original distribution:")
        print(f"  Class 0: {(y==0).sum()}, Class 1: {(y==1).sum()}")
        if (y==0).sum() > 0:
            print(f"  Ratio: {(y==1).sum() / (y==0).sum():.2f}:1")
        
        X_train, X_test, y_train, y_test, X_err_train, X_err_test, idx_train, idx_test = train_test_split(
            X, y, X_err, np.arange(len(y)),
            test_size=0.2,
            random_state=RANDOM_STATE,
            stratify=y
        )
        print(f"Initial split - Train: {len(X_train)}, Test: {len(X_test)}")
        
        # Balance training set
        X_train, y_train = create_balanced_dataset(X_train, y_train, samples_per_class=self.samples_per_class)
        
        scaler = None
        if self.use_scaler:
            print("\n" + "="*70)
            print("STANDARDIZATION")
            print("="*70)
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)
            print(f"Train: mean={X_train.mean():.6f}, std={X_train.std():.6f}")
            print(f"Test:  mean={X_test.mean():.6f}, std={X_test.std():.6f}")
        
        metadata_test = metadata.iloc[idx_test].reset_index(drop=True)
        print(f"Final - X_train: {X_train.shape}, X_test: {X_test.shape}")
        print(f"Train dist: 0={(y_train==0).sum()}, 1={(y_train==1).sum()}")
        
        # Return X_test copy (for optional inverse transform plotting if scaler is used)
        return X_train, X_test, y_train, y_test, metadata_test, X_test.copy(), X_err_test, scaler
        


def create_balanced_dataset(X, y, samples_per_class=400):
    print("\n" + "="*70)
    print("CREATING BALANCED DATASET")
    print("="*70)
        
    X0 = X[y == 0]
    X1 = X[y == 1]
    print(f"Original - Class 0: {len(X0)}, Class 1: {len(X1)}")
        
    def augment_to_target(X_orig, n_target):
        if len(X_orig) >= n_target:
            idx = np.random.choice(len(X_orig), n_target, replace=False)
            return X_orig[idx]
            
        X_result = [X_orig]
        while len(np.vstack(X_result)) < n_target:
            n_needed = n_target - len(np.vstack(X_result))
            idx = np.random.choice(len(X_orig), min(len(X_orig), n_needed))
            aug_type = np.random.rand()
            if aug_type < 0.25:
                X_aug = X_orig[idx] + np.random.normal(0, 0.01, (len(idx), X_orig.shape[1]))
            elif aug_type < 0.5:
                scale = 1.0 + np.random.uniform(-0.03, 0.03, (len(idx), 1))
                X_aug = X_orig[idx] * scale
            elif aug_type < 0.75:
                shifts = np.random.randint(-20, 20, len(idx))
                X_aug = np.array([np.roll(X_orig[i], s) for i, s in zip(idx, shifts)])
            else:
                X_aug = X_orig[idx] * (1.0 + np.random.uniform(-0.02, 0.02, (len(idx), 1)))
                X_aug += np.random.normal(0, 0.008, X_aug.shape)
            X_result.append(X_aug)
        X_final = np.vstack(X_result)
        return X_final[:n_target]
        
    X0_bal = augment_to_target(X0, samples_per_class)
    X1_bal = augment_to_target(X1, samples_per_class)
    print(f"Balanced - Class 0: {len(X0_bal)}, Class 1: {len(X1_bal)}")
        
    X_bal = np.vstack([X0_bal, X1_bal])
    y_bal = np.concatenate([np.zeros(samples_per_class), np.ones(samples_per_class)])
      
    idx = np.arange(len(X_bal))
    np.random.shuffle(idx)
    return X_bal[idx], y_bal[idx]    
