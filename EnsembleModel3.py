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

# -------------------------------------------------------
# Utilities
# -------------------------------------------------------

def normalize_probs(p):
    return p / p.sum(axis=1, keepdims=True)


def decode_trajectory(hex_str):
    points = list(wkb.loads(hex_str, hex=True).coords)
    return np.array(points)


def decode_point(hex_str):
    try:
        if hex_str is None or (isinstance(hex_str, float) and np.isnan(hex_str)):
            return None, None
        pt = wkb.loads(hex_str, hex=True)
        return pt.x, pt.y
    except:
        return None, None


def haversine_vec(lon1, lat1, lon2, lat2):
    R = 6371000
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)
    a = np.sin(dphi/2)**2 + np.cos(phi1)*np.cos(phi2)*np.sin(dlambda/2)**2
    return 2 * R * np.arctan2(np.sqrt(a), np.sqrt(1 - a))


# -------------------------------------------------------
# Feature Engineering
# -------------------------------------------------------

def get_features(df, size_enc=None, name="Dataset"):
    print(f"\n>>> Starting Feature Engineering for: {name}")
    feats = []
    df = df.copy()

    df['radar_bird_size'] = df['radar_bird_size'].fillna('Unknown')

    if size_enc is None:
        size_enc = LabelEncoder()
        size_enc.fit(list(set(df['radar_bird_size'].unique()) | {'Unknown'}))

    known = set(size_enc.classes_)
    df['radar_bird_size_enc'] = size_enc.transform(
        df['radar_bird_size'].apply(lambda x: x if x in known else 'Unknown')
    )

    df['ts_start'] = pd.to_datetime(df['timestamp_start_radar_utc'])

    total_rows = len(df)
    for i, row in enumerate(df.itertuples(index=False)):
        # Optional: Print progress every 1000 rows
        if (i + 1) % 1000 == 0:
            print(f"    Processed {i + 1}/{total_rows} rows...", end='\r')

        traj = decode_trajectory(row.trajectory)
        times = np.array(json.loads(row.trajectory_time))

        coords = traj[:, :2]
        altitudes = traj[:, 2]
        rcs = traj[:, 3]

        dt = np.diff(times)
        dt_safe = np.where(dt > 0, dt, 1e-9)

        total_duration = times[-1] - times[0] if len(times) > 1 else 0
        airspeed = float(getattr(row, 'airspeed'))

        # ---------------- PATH ----------------
        if len(coords) > 1:
            lons1, lats1 = coords[:-1, 0], coords[:-1, 1]
            lons2, lats2 = coords[1:, 0], coords[1:, 1]

            seg_dist = haversine_vec(lons1, lats1, lons2, lats2)
            path_length = np.sum(seg_dist)
            displacement = haversine_vec(
                coords[0, 0], coords[0, 1],
                coords[-1, 0], coords[-1, 1]
            )

            inst_speed = seg_dist / dt_safe

            dlon = np.radians(lons2 - lons1)
            dlat = np.radians(lats2 - lats1)
            headings = np.degrees(np.arctan2(dlon, dlat)) % 360

            turn_angles = np.abs(np.diff(headings))
            turn_angles = np.where(turn_angles > 180, 360 - turn_angles, turn_angles)
            straightness = displacement / (path_length + 1e-9)
        else:
            seg_dist = np.array([0])
            inst_speed = np.array([0])
            turn_angles = np.array([0])
            path_length = 0
            displacement = 0
            straightness = 0

        dz = np.diff(altitudes)
        alt_slope = np.polyfit(times, altitudes, 1)[0] if len(altitudes) > 1 else 0
        rcs_slope = np.polyfit(times, rcs, 1)[0] if len(rcs) > 1 else 0

        # ---------------- TIME ----------------
        ts = row.ts_start
        hour = ts.hour + ts.minute / 60.0
        is_night = int(hour < 6 or hour > 20)
        day_of_year = ts.timetuple().tm_yday

        # ---------------- DISTANCE TO RADAR ----------------
        obs_lon, obs_lat = decode_point(getattr(row, 'observer_position', None))
        if obs_lon is not None:
            dist_to_radar = haversine_vec(
                obs_lon, obs_lat,
                coords[:, 0].mean(),
                coords[:, 1].mean()
            )
        else:
            dist_to_radar = 0

        # ---------------- FEATURES ----------------
        f = {
            'track_id': row.track_id,
            'point_count': len(traj),
            'total_duration': total_duration,
            'point_density': len(traj) / (total_duration + 1e-9),
            'airspeed': airspeed,
            'airspeed_x_duration': airspeed * total_duration,
            'radar_bird_size_enc': row.radar_bird_size_enc,
            'n_birds_observed': float(getattr(row, 'n_birds_observed', 1) or 1),
            'hour_of_day': hour,
            'is_night': is_night,
            'day_of_year': day_of_year,
            'mean_rcs': np.mean(rcs),
            'std_rcs': np.std(rcs),
            'rcs_cv': np.std(rcs) / (np.mean(rcs) + 1e-9),
            'rcs_slope': rcs_slope,
            'mean_alt': np.mean(altitudes),
            'alt_std': np.std(altitudes),
            'alt_cv': np.std(altitudes) / (np.mean(altitudes) + 1e-9),
            'alt_slope': alt_slope,
            'avg_climb_rate': np.mean(dz / dt_safe) if len(dz) > 0 else 0,
            'speed_mean': np.mean(inst_speed),
            'speed_std': np.std(inst_speed),
            'speed_cv': np.std(inst_speed) / (np.mean(inst_speed) + 1e-9),
            'mean_turn': np.mean(turn_angles),
            'std_turn': np.std(turn_angles),
            'turn_cv': np.std(turn_angles) / (np.mean(turn_angles) + 1e-9),
            'path_length': path_length,
            'displacement': displacement,
            'straightness': straightness,
            'dist_to_radar': dist_to_radar
        }
        feats.append(f)

    print(f"\n>>> Feature Engineering for {name} complete.")
    return pd.DataFrame(feats), size_enc


# -------------------------------------------------------
# MAIN
# -------------------------------------------------------

def main():
    print("Loading data...")
    train = pd.read_csv("data/train.csv")
    test = pd.read_csv("data/test.csv")

    X_train_df, size_enc = get_features(train, name="TRAIN")
    X_test_df, _ = get_features(test, size_enc=size_enc, name="TEST")

    X = X_train_df.drop(columns=['track_id'])
    X_test = X_test_df.drop(columns=['track_id'])

    le = LabelEncoder()
    Y = le.fit_transform(train['bird_group'])
    n_classes = len(le.classes_)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    oof_cat = np.zeros((len(X), n_classes))
    oof_xgb = np.zeros((len(X), n_classes))
    oof_lgb = np.zeros((len(X), n_classes))

    test_cat = np.zeros((len(X_test), n_classes))
    test_xgb = np.zeros((len(X_test), n_classes))
    test_lgb = np.zeros((len(X_test), n_classes))

    print(f"\nStarting 5-Fold Cross Validation...")
    for fold, (tr_idx, val_idx) in enumerate(skf.split(X, Y)):
        print(f"\n--- FOLD {fold+1} ---")
        
        X_tr, X_va = X.iloc[tr_idx], X.iloc[val_idx]
        y_tr, y_va = Y[tr_idx], Y[val_idx]

        # CATBOOST
        print("  Training CatBoost...")
        cat = CatBoostClassifier(
            iterations=4000,
            learning_rate=0.01,
            depth=6,
            l2_leaf_reg=10,
            loss_function='MultiClass',
            eval_metric='MultiClass',
            random_seed=42,
            verbose=0,
            allow_writing_files=False
        )
        cat.fit(X_tr, y_tr, eval_set=(X_va, y_va))
        oof_cat[val_idx] = cat.predict_proba(X_va)
        test_cat += cat.predict_proba(X_test) / 5
        print(f"    CatBoost LogLoss: {log_loss(y_va, oof_cat[val_idx]):.5f}")

        # XGBOOST
        print("  Training XGBoost...")
        xgb_model = xgb.XGBClassifier(
            n_estimators=4000,
            learning_rate=0.01,
            max_depth=6,
            reg_lambda=3,
            subsample=0.8,
            colsample_bytree=0.8,
            objective='multi:softprob',
            eval_metric='mlogloss',
            early_stopping_rounds=100,
            random_state=42
        )
        xgb_model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
        probs = normalize_probs(xgb_model.predict_proba(X_va))
        oof_xgb[val_idx] = probs
        test_xgb += normalize_probs(xgb_model.predict_proba(X_test)) / 5
        print(f"    XGBoost LogLoss:  {log_loss(y_va, oof_xgb[val_idx]):.5f}")

        # LIGHTGBM
        print("  Training LightGBM...")
        lgb_model = lgb.LGBMClassifier(
            n_estimators=1500,
            learning_rate=0.02,
            max_depth=5,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42
        )
        lgb_model.fit(
            X_tr, y_tr,
            eval_set=[(X_va, y_va)],
            callbacks=[lgb.early_stopping(100, verbose=False)]
        )
        oof_lgb[val_idx] = lgb_model.predict_proba(X_va)
        test_lgb += lgb_model.predict_proba(X_test) / 5
        print(f"    LightGBM LogLoss: {log_loss(y_va, oof_lgb[val_idx]):.5f}")

    # ---------------- ENSEMBLE ----------------
    print("\n" + "="*30)
    print("FINAL VALIDATION SCORES")
    print("="*30)

    oof_ensemble = (0.4 * oof_cat + 0.3 * oof_xgb + 0.2 * oof_lgb)
    # Re-normalize just in case
    oof_ensemble = normalize_probs(oof_ensemble)

    test_ensemble = (0.4 * test_cat + 0.3 * test_xgb + 0.2 * test_lgb)
    test_ensemble = normalize_probs(test_ensemble)

    print(f"CAT LogLoss:      {log_loss(Y, oof_cat):.5f}")
    print(f"XGB LogLoss:      {log_loss(Y, oof_xgb):.5f}")
    print(f"LGB LogLoss:      {log_loss(Y, oof_lgb):.5f}")
    print(f"ENSEMBLE LogLoss: {log_loss(Y, oof_ensemble):.5f}")

    print("\nSaving submission...")
    result = pd.DataFrame(test_ensemble, columns=le.classes_)
    result.insert(0, 'track_id', X_test_df['track_id'])
    result.to_csv("data/submission.csv", index=False)
    print("Done!")


if __name__ == "__main__":
    main()