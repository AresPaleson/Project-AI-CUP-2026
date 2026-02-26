import json
import pandas as pd
import numpy as np
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier
from shapely import wkb
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import log_loss


def decode_trajectory(hex_str):
    points = list(wkb.loads(hex_str, hex=True).coords)
    return np.array(points)


def get_features(df):
    feats = []
    print(f"Processing {len(df)} rows...")

    for row in df.itertuples(index=False):
        traj = decode_trajectory(row.trajectory)
        times = np.array(json.loads(row.trajectory_time))

        coords = traj[:, :2]
        altitudes = traj[:, 2]
        rcs = traj[:, 3]

        dt = np.diff(times)
        dt_safe = np.where(dt > 0, dt, 1e-9)
        total_duration = times[-1] - times[0] if len(times) > 1 else 0

        # airspeed is a single float per track
        airspeed = float(getattr(row, 'airspeed'))

        # Path characteristics
        if len(coords) > 1:
            def haversine_scalar(lon1, lat1, lon2, lat2):
                R = 6371000
                phi1, phi2 = np.radians(lat1), np.radians(lat2)
                dphi = np.radians(lat2 - lat1)
                dlambda = np.radians(lon2 - lon1)
                a = np.sin(dphi/2)**2 + np.cos(phi1)*np.cos(phi2)*np.sin(dlambda/2)**2
                return 2 * R * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

            lon1, lat1 = coords[:-1, 0], coords[:-1, 1]
            lon2, lat2 = coords[1:, 0], coords[1:, 1]
            R = 6371000
            phi1, phi2 = np.radians(lat1), np.radians(lat2)
            dphi = np.radians(lat2 - lat1)
            dlambda = np.radians(lon2 - lon1)
            a = np.sin(dphi/2)**2 + np.cos(phi1)*np.cos(phi2)*np.sin(dlambda/2)**2
            dist = 2 * R * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

            path_length = np.sum(dist)
            displacement = haversine_scalar(
                coords[0, 0], coords[0, 1], coords[-1, 0], coords[-1, 1]
            )
        else:
            path_length, displacement = 0, 0

        dz = np.diff(altitudes)

        f = {
            'track_id': row.track_id,
            'point_count': len(traj),
            'total_duration': total_duration,

            # Single airspeed value + meaningful derived combos
            'airspeed': airspeed,
            'airspeed_x_duration': airspeed * total_duration,     # proxy for total distance
            'airspeed_x_pointcount': airspeed * len(traj),        # observation density
            'airspeed_per_point': airspeed / (len(traj) + 1e-9),  # speed per radar ping

            # RCS features
            'mean_rcs': np.mean(rcs),
            'std_rcs': np.std(rcs),
            'rcs_cv': np.std(rcs) / (np.mean(rcs) + 1e-9),
            'rcs_p25': np.percentile(rcs, 25),
            'rcs_p75': np.percentile(rcs, 75),
            'rcs_range': np.max(rcs) - np.min(rcs),

            # Altitude features
            'alt_range': getattr(row, 'max_z', 0) - getattr(row, 'min_z', 0),
            'med_alt': np.median(altitudes),
            'mean_alt': np.mean(altitudes),
            'avg_climb_rate': np.mean(dz / dt_safe) if len(dz) > 0 else 0,
            'std_climb_rate': np.std(dz / dt_safe) if len(dz) > 0 else 0,

            # Path features
            'tortuosity': path_length / (displacement + 1e-9),
            'path_length': path_length,
            'displacement': displacement,
        }
        feats.append(f)

    return pd.DataFrame(feats)


def main():
    print("Pipeline started...")

    train = pd.read_csv("data/train.csv")
    test  = pd.read_csv("data/test.csv")

    print("Extracting features...")
    X_train_df = get_features(train)
    X_test_df  = get_features(test)

    X      = X_train_df.drop(columns=['track_id'])
    X_test = X_test_df.drop(columns=['track_id'])

    le = LabelEncoder()
    Y = le.fit_transform(train['bird_group'])
    n_classes = len(le.classes_)

    n_splits     = 5
    skf          = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    model_names  = ['lgb', 'xgb', 'cat']

    oof_all  = {m: np.zeros((len(X),      n_classes)) for m in model_names}
    test_all = {m: np.zeros((len(X_test), n_classes)) for m in model_names}

    for model_name in model_names:
        print(f"\n{'='*40}\nTraining: {model_name.upper()}\n{'='*40}")

        for fold, (train_idx, val_idx) in enumerate(skf.split(X, Y)):
            print(f"  Fold {fold + 1}/{n_splits}...")
            X_tr, y_tr = X.iloc[train_idx], Y[train_idx]
            X_va, y_va = X.iloc[val_idx],   Y[val_idx]

            if model_name == 'lgb':
                model = lgb.LGBMClassifier(
                    objective='multiclass', num_class=n_classes,
                    n_estimators=1000, learning_rate=0.03,
                    class_weight='balanced', subsample=0.8,
                    colsample_bytree=0.8, random_state=42, verbose=-1
                )
                model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)],
                          callbacks=[lgb.early_stopping(50, verbose=False)])

            elif model_name == 'xgb':
                model = xgb.XGBClassifier(
                    objective='multi:softprob', num_class=n_classes,
                    n_estimators=1000, learning_rate=0.03,
                    subsample=0.8, colsample_bytree=0.8,
                    eval_metric='mlogloss', early_stopping_rounds=50,
                    random_state=42, verbosity=0
                )
                model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)

            elif model_name == 'cat':
                model = CatBoostClassifier(
                    iterations=1000, learning_rate=0.03,
                    loss_function='MultiClass', eval_metric='MultiClass',
                    early_stopping_rounds=50, random_seed=42,
                    verbose=0, auto_class_weights='Balanced'
                )
                model.fit(X_tr, y_tr, eval_set=(X_va, y_va))

            oof_all[model_name][val_idx]  = model.predict_proba(X_va)
            test_all[model_name]         += model.predict_proba(X_test) / n_splits

        score = log_loss(Y, oof_all[model_name])
        print(f"  {model_name.upper()} CV Log Loss: {score:.4f}")

    # --- Ensemble: simple average of all 3 models ---
    print("\n--- Ensemble Results ---")
    for m in model_names:
        print(f"  {m.upper()} Log Loss: {log_loss(Y, oof_all[m]):.4f}")

    oof_ensemble   = np.mean([oof_all[m]  for m in model_names], axis=0)
    test_ensemble  = np.mean([test_all[m] for m in model_names], axis=0)
    ensemble_score = log_loss(Y, oof_ensemble)
    print(f"  ENSEMBLE Log Loss: {ensemble_score:.4f}")

    print("\nGenerating submission...")
    result = pd.DataFrame(test_ensemble, columns=le.classes_)
    result.insert(0, 'track_id', X_test_df['track_id'])
    result.to_csv("data/submission.csv", index=False)
    print("Saved to data/submission.csv")


if __name__ == "__main__":
    main()
