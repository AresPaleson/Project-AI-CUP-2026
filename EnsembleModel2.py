import json
import pandas as pd
import numpy as np
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier
from shapely import wkb
from sklearn.model_selection import StratifiedKFold, GroupKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import log_loss


def decode_trajectory(hex_str):
    points = list(wkb.loads(hex_str, hex=True).coords)
    return np.array(points)


def decode_point(hex_str):
    """Decode observer_position EWKB hex into (lon, lat). Returns (None, None) if missing."""
    try:
        if hex_str is None or (isinstance(hex_str, float) and np.isnan(hex_str)):
            return None, None
        pt = wkb.loads(hex_str, hex=True)
        return pt.x, pt.y
    except Exception:
        return None, None


def haversine_vec(lon1, lat1, lon2, lat2):
    R = 6371000
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi    = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)
    a = np.sin(dphi/2)**2 + np.cos(phi1)*np.cos(phi2)*np.sin(dlambda/2)**2
    return 2 * R * np.arctan2(np.sqrt(a), np.sqrt(1 - a))


def get_features(df, size_enc=None):
    """
    size_enc: pass a pre-fit LabelEncoder when processing test set.
              If None, a new one is fit on the current df (use for train).
    Returns: (feature DataFrame, fitted size_enc)
    """
    feats = []
    print(f"Processing {len(df)} rows...")

    df = df.copy()

    # ── radar_bird_size encoding ───────────────────────────────────
    df['radar_bird_size'] = df['radar_bird_size'].fillna('Unknown')
    if size_enc is None:
        size_enc = LabelEncoder()
        # Make sure 'Unknown' is in classes even if not in train
        all_sizes = list(df['radar_bird_size'].unique()) + ['Unknown']
        size_enc.fit(list(set(all_sizes)))
    # Safe transform: map anything unseen to 'Unknown'
    known = set(size_enc.classes_)
    df['radar_bird_size_enc'] = size_enc.transform(
        df['radar_bird_size'].apply(lambda x: x if x in known else 'Unknown')
    )

    # ── Parse timestamps ───────────────────────────────────────────
    df['ts_start'] = pd.to_datetime(df['timestamp_start_radar_utc'])

    for row in df.itertuples(index=False):
        traj  = decode_trajectory(row.trajectory)
        times = np.array(json.loads(row.trajectory_time))

        coords    = traj[:, :2]   # lon, lat
        altitudes = traj[:, 2]
        rcs       = traj[:, 3]

        dt           = np.diff(times)
        dt_safe      = np.where(dt > 0, dt, 1e-9)
        total_duration = times[-1] - times[0] if len(times) > 1 else 0

        airspeed = float(getattr(row, 'airspeed'))

        # ── Path features ──────────────────────────────────────────
        if len(coords) > 1:
            lons1, lats1 = coords[:-1, 0], coords[:-1, 1]
            lons2, lats2 = coords[1:,  0], coords[1:,  1]
            seg_dist     = haversine_vec(lons1, lats1, lons2, lats2)
            path_length  = np.sum(seg_dist)
            displacement = haversine_vec(
                coords[0, 0], coords[0, 1], coords[-1, 0], coords[-1, 1]
            )

            # Heading at each step (degrees 0–360)
            dlon = np.radians(lons2 - lons1)
            dlat = np.radians(lats2 - lats1)
            headings = np.degrees(np.arctan2(dlon, dlat)) % 360

            # Turn angles between consecutive headings
            turn_angles = np.abs(np.diff(headings))
            turn_angles = np.where(turn_angles > 180, 360 - turn_angles, turn_angles)

            # Bounding box area (approx m²)
            lat_range_m = haversine_vec(
                coords[:, 0].mean(), coords[:, 1].min(),
                coords[:, 0].mean(), coords[:, 1].max()
            )
            lon_range_m = haversine_vec(
                coords[:, 0].min(), coords[:, 1].mean(),
                coords[:, 0].max(), coords[:, 1].mean()
            )
            bbox_area = lat_range_m * lon_range_m
        else:
            seg_dist    = np.array([0])
            path_length = displacement = bbox_area = 0
            headings    = turn_angles = np.array([0])

        dz = np.diff(altitudes)

        # ── Altitude trend (linear slope) ─────────────────────────
        alt_slope = np.polyfit(times, altitudes, 1)[0] if len(altitudes) > 1 else 0

        # ── RCS trend (linear slope) ───────────────────────────────
        rcs_slope = np.polyfit(times, rcs, 1)[0] if len(rcs) > 1 else 0

        # ── Speed trend: early vs late path length ─────────────────
        n = len(coords)
        if n >= 3 and len(seg_dist) >= 2:
            t           = n // 3
            speed_trend = np.sum(seg_dist[-t:]) - np.sum(seg_dist[:t])
        else:
            speed_trend = 0

        # ── Timestamp features ─────────────────────────────────────
        ts          = row.ts_start
        hour        = ts.hour + ts.minute / 60.0
        is_night    = int(hour < 6 or hour > 20)
        day_of_year = ts.timetuple().tm_yday

        # ── Distance from radar to track centroid ──────────────────
        obs_pos = getattr(row, 'observer_position', None)
        obs_lon, obs_lat = decode_point(obs_pos)
        if obs_lon is not None and len(coords) > 0:
            dist_to_radar = haversine_vec(
                obs_lon, obs_lat, coords[:, 0].mean(), coords[:, 1].mean()
            )
        else:
            dist_to_radar = 0

        # ── Build feature dict ─────────────────────────────────────
        f = {
            'track_id':       row.track_id,
            'point_count':    len(traj),
            'total_duration': total_duration,
            'point_density':  len(traj) / (total_duration + 1e-9),

            # Airspeed
            'airspeed':            airspeed,
            'airspeed_x_duration': airspeed * total_duration,
            'airspeed_per_point':  airspeed / (len(traj) + 1e-9),

            # Radar bird size label
            'radar_bird_size_enc': row.radar_bird_size_enc,

            # Flock size
            'n_birds_observed': float(getattr(row, 'n_birds_observed', 1) or 1),

            # Temporal
            'hour_of_day':  hour,
            'is_night':     is_night,
            'day_of_year':  day_of_year,

            # RCS
            'mean_rcs':  np.mean(rcs),
            'std_rcs':   np.std(rcs),
            'rcs_cv':    np.std(rcs) / (np.mean(rcs) + 1e-9),
            'rcs_p25':   np.percentile(rcs, 25),
            'rcs_p75':   np.percentile(rcs, 75),
            'rcs_range': np.max(rcs) - np.min(rcs),
            'rcs_slope': rcs_slope,

            # Altitude
            'min_z':          getattr(row, 'min_z', 0),
            'max_z':          getattr(row, 'max_z', 0),
            'alt_range':      getattr(row, 'max_z', 0) - getattr(row, 'min_z', 0),
            'med_alt':        np.median(altitudes),
            'mean_alt':       np.mean(altitudes),
            'avg_climb_rate': np.mean(dz / dt_safe) if len(dz) > 0 else 0,
            'std_climb_rate': np.std(dz  / dt_safe) if len(dz) > 0 else 0,
            'alt_slope':      alt_slope,

            # Path
            'tortuosity':   path_length / (displacement + 1e-9),
            'path_length':  path_length,
            'displacement': displacement,
            'bbox_area':    bbox_area,
            'speed_trend':  speed_trend,

            # Turning behaviour
            'mean_turn': np.mean(turn_angles),
            'max_turn':  np.max(turn_angles),
            'std_turn':  np.std(turn_angles),

            # Radar distance
            'dist_to_radar': dist_to_radar,
        }
        feats.append(f)

    return pd.DataFrame(feats), size_enc


def main():
    print("Pipeline started...")

    train = pd.read_csv("data/train.csv")
    test  = pd.read_csv("data/test.csv")

    print("Extracting Train features...")
    X_train_df, size_enc = get_features(train)               # fit encoder on train

    print("Extracting Test features...")
    X_test_df, _ = get_features(test, size_enc=size_enc)     # reuse encoder on test

    X      = X_train_df.drop(columns=['track_id'])
    X_test = X_test_df.drop(columns=['track_id'])

    le = LabelEncoder()
    Y = le.fit_transform(train['bird_group'])
    n_classes = len(le.classes_)

    n_splits    = 10
    skf         = GroupKFold(n_splits=n_splits, shuffle=True, random_state=42)
    model_names = ['cat', 'lgb', 'xgb']

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
                    n_estimators=2000,
                    learning_rate=0.01, 
                    max_depth=4,        # Shallow trees to reduce overfitting
                    subsample=0.8,      # Row sampling for regularization
                    colsample_bytree=0.8, # Feature sampling for regularization
                    random_state=42,
                    verbose=-1
                )
                model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)],
                          callbacks=[lgb.early_stopping(50, verbose=False)])

            elif model_name == 'xgb':
                model = xgb.XGBClassifier(
                    n_estimators=2000, 
                    learning_rate=0.01,
                    max_depth=4,         # Shallow is better for small datasets
                    reg_lambda=2,        # Added protection against overfitting
                    subsample=0.8,      # Row sampling for regularization
                    colsample_bytree=0.8, # Feature sampling for regularization
                    early_stopping_rounds=50,
                    random_state=42
                )
                model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
                def normalize_probs(p):
                    return p / p.sum(axis=1, keepdims=True)

            elif model_name == 'cat':
                model = CatBoostClassifier(
                    iterations=2000,
                    learning_rate=0.02,
                    depth=4,             # Shallow trees to reduce overfitting
                    l2_leaf_reg=5,       # Added regularization to combat overfitting
                    early_stopping_rounds=100, 
                    random_seed=42,
                    verbose=0,
                    allow_writing_files=False
                )
                model.fit(X_tr, y_tr, eval_set=(X_va, y_va))

            probs = model.predict_proba(X_va)
            probs = normalize_probs(probs)
            oof_all[model_name][val_idx] = probs

        score = log_loss(Y, oof_all[model_name])
        print(f"  {model_name.upper()} CV Log Loss: {score:.4f}")

    # ── Ensemble: simple average of all 3 ─────────────────────────
    print("\n--- Ensemble Results ---")
    for m in model_names:
        print(f"  {m.upper()} Log Loss: {log_loss(Y, oof_all[m]):.4f}")

    oof_ensemble  = np.mean([oof_all[m]  for m in model_names], axis=0)
    test_ensemble = np.mean([test_all[m] for m in model_names], axis=0)
    '''
    # Alternative weighted ensemble:
    oof_ensemble  = (oof_all['lgb'] * 0.25 +
                 oof_all['xgb'] * 0.25 +
                 oof_all['cat'] * 0.50)

    test_ensemble = (test_all['lgb'] * 0.25 +
                    test_all['xgb'] * 0.25 +
                    test_all['cat'] * 0.50)
    '''
    print(f"  ENSEMBLE Log Loss: {log_loss(Y, oof_ensemble):.4f}")

    print("\nGenerating submission...")
    result = pd.DataFrame(test_ensemble, columns=le.classes_)
    result.insert(0, 'track_id', X_test_df['track_id'])
    result.to_csv("data/submission.csv", index=False)
    print("Saved to data/submission.csv")


if __name__ == "__main__":
    main()