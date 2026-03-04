import json
import pandas as pd
import numpy as np
from shapely import wkb
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import log_loss
from sklearn.neural_network import MLPClassifier

def haversine_vec(lon1, lat1, lon2, lat2):
    R = 6371000
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)
    a = np.sin(dphi/2)**2 + np.cos(phi1)*np.cos(phi2)*np.sin(dlambda/2)**2
    return 2 * R * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

def decode_trajectory(hex_str):
    points = list(wkb.loads(hex_str, hex=True).coords)
    return np.array(points)

def get_features(df):
    feats = []
    print(f"Processing {len(df)} rows...")
    df["ts"] = pd.to_datetime(df["timestamp_start_radar_utc"])
    if "timestamp_end_radar_utc" in df.columns:
        df["ts_end"] = pd.to_datetime(df["timestamp_end_radar_utc"])
    else:
        df["ts_end"] = df["ts"]

    for row in df.itertuples(index=False):
        traj = decode_trajectory(row.trajectory)
        times = np.array(json.loads(row.trajectory_time))
        coords = traj[:, :2]
        altitudes = traj[:, 2]
        rcs = traj[:, 3]
        dt = np.diff(times)
        dt_safe = np.where(dt > 0, dt, 1e-9)
        total_duration = times[-1] - times[0] if len(times) > 1 else 0
        airspeed = float(row.airspeed)

        if len(coords) > 1:
            lon1, lat1 = coords[:-1, 0], coords[:-1, 1]
            lon2, lat2 = coords[1:, 0], coords[1:, 1]
            seg_dist = haversine_vec(lon1, lat1, lon2, lat2)
            path_length = np.sum(seg_dist)
            displacement = haversine_vec(coords[0, 0], coords[0, 1], coords[-1, 0], coords[-1, 1])
            inst_speed = seg_dist / dt_safe
            dlon = np.radians(lon2 - lon1)
            dlat = np.radians(lat2 - lat1)
            headings = np.degrees(np.arctan2(dlon, dlat)) % 360
            turn_angles = np.abs(np.diff(headings))
            turn_angles = np.where(turn_angles > 180, 360 - turn_angles, turn_angles)
            straightness = displacement / (path_length + 1e-9)
        else:
            path_length = displacement = 0
            inst_speed = np.array([0])
            turn_angles = np.array([0])
            straightness = 0

        dz = np.diff(altitudes)
        hour = row.ts.hour + row.ts.minute / 60.0
        day_of_year = row.ts.timetuple().tm_yday
        is_night = int(hour < 6 or hour > 20)
        meta_duration = (row.ts_end - row.ts).total_seconds() if pd.notna(row.ts_end) else 0
        min_z_meta = float(row.min_z) if hasattr(row, 'min_z') and pd.notna(row.min_z) else 0.0
        max_z_meta = float(row.max_z) if hasattr(row, 'max_z') and pd.notna(row.max_z) else 0.0
        radar_bird_size = str(row.radar_bird_size) if hasattr(row, 'radar_bird_size') and pd.notna(row.radar_bird_size) else "Unknown"

        f = {
            "track_id": row.track_id,
            "point_count": len(traj),
            "total_duration": total_duration,
            "airspeed": airspeed,
            "hour": hour,
            "day_of_year": day_of_year,
            "is_night": is_night,
            "meta_duration": meta_duration,
            "speed_mean": np.mean(inst_speed),
            "speed_std": np.std(inst_speed),
            "mean_turn": np.mean(turn_angles),
            "std_turn": np.std(turn_angles),
            "mean_alt": np.mean(altitudes),
            "alt_std": np.std(altitudes),
            "climb_rate_mean": np.mean(dz / dt_safe) if len(dz) > 0 else 0,
            "min_z_meta": min_z_meta,
            "max_z_meta": max_z_meta,
            "mean_rcs": np.mean(rcs),
            "rcs_std": np.std(rcs),
            "path_length": path_length,
            "displacement": displacement,
            "straightness": straightness,
            "radar_bird_size": radar_bird_size
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

    print("Encoding categorical features...")
    le_size = LabelEncoder()
    all_sizes = pd.concat([X_train_df["radar_bird_size"], X_test_df["radar_bird_size"]])
    le_size.fit(all_sizes)
    X_train_df["radar_bird_size"] = le_size.transform(X_train_df["radar_bird_size"])
    X_test_df["radar_bird_size"]  = le_size.transform(X_test_df["radar_bird_size"])

    X      = X_train_df.drop(columns=["track_id"])
    X_test = X_test_df.drop(columns=["track_id"])
    le = LabelEncoder()
    Y = le.fit_transform(train["bird_group"])
    groups = X_train_df["track_id"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_test_scaled = scaler.transform(X_test)

    skf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)

    oof = np.zeros((len(X), len(le.classes_)))
    test_preds = np.zeros((len(X_test), len(le.classes_)))

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, Y, groups)):
        print(f"\nFold {fold+1}")
        X_tr, y_tr = X.iloc[train_idx], Y[train_idx]
        X_va, y_va = X.iloc[val_idx], Y[val_idx]

        X_tr_scaled = scaler.fit_transform(X_tr)
        X_va_scaled = scaler.transform(X_va)

        model = MLPClassifier(
            hidden_layer_sizes=(128, 64),
            activation="relu",
            solver="adam",
            alpha=1e-4,
            learning_rate_init=1e-3,
            batch_size=128,
            max_iter=1000,
            early_stopping=True,
            n_iter_no_change=20,
            random_state=42
        )

        model.fit(X_tr_scaled, y_tr)
        oof[val_idx] = model.predict_proba(X_va_scaled)
        test_preds += model.predict_proba(X_test_scaled) / skf.n_splits

    print("CV LogLoss:", log_loss(Y, oof))

    result = pd.DataFrame(test_preds, columns=le.classes_)
    result.insert(0, "track_id", X_test_df["track_id"])
    result.to_csv("data/submission_mlp.csv", index=False)
    print("Submission saved as submission_mlp.csv")


if __name__ == "__main__":
    main()