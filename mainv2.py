import json
import pandas as pd
import numpy as np
import lightgbm as lgb
from shapely import wkb
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import log_loss

def decode_trajectory(hex_str):
    # Decodes EWKB Hex to a numpy array: [Long, Lat, Alt, RCS]
    points = list(wkb.loads(hex_str, hex=True).coords)
    return np.array(points)

def get_features(df):
    feats = []
    print(f"Processing {len(df)} rows...")

    for row in df.itertuples(index=False):
        # Decode trajectory
        traj = decode_trajectory(row.trajectory)
        times = np.array(json.loads(row.trajectory_time))

        coords = traj[:, :2]
        altitudes = traj[:, 2]
        rcs = traj[:, 3]

        # Time deltas
        dt = np.diff(times)
        dt_safe = np.where(dt > 0, dt, 1e-9)
        total_duration = times[-1] - times[0] if len(times) > 1 else 0

        # Airspeed
        raw_airspeed = getattr(row, 'airspeed')
        speeds = np.array([float(raw_airspeed)])

        # Acceleration derived from airspeed differences over time
        if len(speeds) > 1 and len(dt_safe) >= len(speeds) - 1:
            accel = np.diff(speeds) / dt_safe[:len(speeds) - 1]
        else:
            accel = np.array([0.0])

        # Path characteristics still use coordinates for tortuosity
        if len(coords) > 1:
            from math import radians, sin, cos, sqrt, atan2
            # Simple displacement for tortuosity (great-circle approximation)
            def haversine_scalar(lon1, lat1, lon2, lat2):
                R = 6371000
                phi1, phi2 = np.radians(lat1), np.radians(lat2)
                dphi = np.radians(lat2 - lat1)
                dlambda = np.radians(lon2 - lon1)
                a = np.sin(dphi/2)**2 + np.cos(phi1)*np.cos(phi2)*np.sin(dlambda/2)**2
                return 2 * R * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

            # Segment distances for path_length
            lon1, lat1 = coords[:-1, 0], coords[:-1, 1]
            lon2, lat2 = coords[1:, 0], coords[1:, 1]
            R = 6371000
            phi1, phi2 = np.radians(lat1), np.radians(lat2)
            dphi = np.radians(lat2 - lat1)
            dlambda = np.radians(lon2 - lon1)
            a = np.sin(dphi/2)**2 + np.cos(phi1)*np.cos(phi2)*np.sin(dlambda/2)**2
            dist = 2 * R * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

            path_length = np.sum(dist)
            displacement = haversine_scalar(coords[0, 0], coords[0, 1], coords[-1, 0], coords[-1, 1])
        else:
            path_length, displacement = 0, 0

        dz = np.diff(altitudes)

        # Build feature dictionary
        f = {
            'track_id': row.track_id,
            'point_count': len(traj),
            'total_duration': total_duration,

            # Airspeed features (from dataset column directly)
            'avg_speed': np.mean(speeds),
            'max_speed': np.max(speeds),
            'min_speed': np.min(speeds),
            'std_speed': np.std(speeds),
            'median_speed': np.median(speeds),
            'speed_p25': np.percentile(speeds, 25),
            'speed_p75': np.percentile(speeds, 75),
            'max_accel': np.max(accel) if len(accel) > 0 else 0,
            'min_accel': np.min(accel) if len(accel) > 0 else 0,
            'std_accel': np.std(accel) if len(accel) > 0 else 0,

            # RCS (Radar Cross Section) features
            'mean_rcs': np.mean(rcs),
            'std_rcs': np.std(rcs),
            'rcs_cv': np.std(rcs) / (np.mean(rcs) + 1e-9),
            'rcs_p25': np.percentile(rcs, 25) if len(rcs) > 0 else 0,
            'rcs_p75': np.percentile(rcs, 75) if len(rcs) > 0 else 0,

            # Altitude features
            'alt_range': getattr(row, 'max_z', 0) - getattr(row, 'min_z', 0),
            'med_alt': np.median(altitudes) if len(altitudes) > 0 else 0,
            'avg_climb_rate': np.mean(dz / dt_safe) if len(dz) > 0 else 0,

            # Path characteristics
            'tortuosity': path_length / (displacement + 1e-9),
            'path_length': path_length,
            'displacement': displacement,
        }
        feats.append(f)

    return pd.DataFrame(feats)

def main():
    print("Pipeline started...")

    # 1. Load Data
    train = pd.read_csv("data/train.csv")
    test = pd.read_csv("data/test.csv")

    # Sanity check
    if 'airspeed' not in train.columns:
        raise ValueError("'airspeed' column not found in train.csv. Check column names: " + str(train.columns.tolist()))

    # 2. Feature Engineering
    print("Extracting Train Features...")
    X_train_df = get_features(train)
    print("Extracting Test Features...")
    X_test_df = get_features(test)

    X = X_train_df.drop(columns=['track_id'])
    X_test = X_test_df.drop(columns=['track_id'])

    # Encode target labels
    le = LabelEncoder()
    Y = le.fit_transform(train['bird_group'])

    # 3. Model Training (Stratified K-Fold)
    print("Training LightGBM with Cross-Validation...")

    n_splits = 5
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    oof_preds = np.zeros((len(X), len(le.classes_)))
    test_preds = np.zeros((len(X_test), len(le.classes_)))

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, Y)):
        print(f"\n--- Fold {fold + 1}/{n_splits} ---")
        X_tr, y_tr = X.iloc[train_idx], Y[train_idx]
        X_va, y_va = X.iloc[val_idx], Y[val_idx]

        clf = lgb.LGBMClassifier(
            objective='multiclass',
            num_class=len(le.classes_),
            n_estimators=500,
            learning_rate=0.03,
            class_weight='balanced',
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            verbose=-1
        )

        clf.fit(
            X_tr, y_tr,
            eval_set=[(X_va, y_va)],
            callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)]
        )

        oof_preds[val_idx] = clf.predict_proba(X_va)
        test_preds += clf.predict_proba(X_test) / n_splits

    # Calculate local validation score
    cv_score = log_loss(Y, oof_preds)
    print(f"\nOverall CV Log Loss: {cv_score:.4f}")

    # 4. Generate Submission
    print("Generating submission...")
    result = pd.DataFrame(test_preds, columns=le.classes_)
    result.insert(0, 'track_id', X_test_df['track_id'])
    result.to_csv("data/submission.csv", index=False)
    print("Saved to data/submission.csv")

if __name__ == "__main__":
    main()