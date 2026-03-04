import json
import pandas as pd
import numpy as np
from shapely import wkb
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import log_loss
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

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
    logger.info(f"Processing {len(df)} rows...")
    df = df.copy()  # Create a copy to avoid warnings
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
            
            # Rate of turn change
            turn_rate = turn_angles / dt_safe[1:] if len(turn_angles) > 1 else np.array([0])
            
            # Energy metrics
            potential_energy = altitudes * 9.81  # m^2/s^2
            kinetic_energy = 0.5 * inst_speed**2
            
            # Trajectory efficiency
            sinuosity = path_length / displacement if displacement > 0 else 0
            
            # Vertical acceleration
            if len(altitudes) > 2:
                dz2 = np.diff(altitudes, n=2)
                vertical_accel = np.mean(dz2 / dt_safe[1:]**2) if len(dz2) > 0 else 0
            else:
                vertical_accel = 0
        else:
            path_length = displacement = 0
            inst_speed = np.array([0])
            turn_angles = np.array([0])
            turn_rate = np.array([0])
            straightness = 0
            potential_energy = altitudes * 9.81
            kinetic_energy = np.array([0])
            sinuosity = 0
            vertical_accel = 0

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
            "mean_turn_rate": np.mean(turn_rate),
            "std_turn_rate": np.std(turn_rate),
            "mean_alt": np.mean(altitudes),
            "alt_std": np.std(altitudes),
            "climb_rate_mean": np.mean(dz / dt_safe) if len(dz) > 0 else 0,
            "vertical_accel": vertical_accel,
            "min_z_meta": min_z_meta,
            "max_z_meta": max_z_meta,
            "mean_rcs": np.mean(rcs),
            "rcs_std": np.std(rcs),
            "path_length": path_length,
            "displacement": displacement,
            "straightness": straightness,
            "sinuosity": sinuosity,
            "mean_potential_energy": np.mean(potential_energy),
            "mean_kinetic_energy": np.mean(kinetic_energy),
            "radar_bird_size": radar_bird_size
        }
        feats.append(f)
    
    df_feats = pd.DataFrame(feats)
    
    # Handle infinite values
    df_feats = df_feats.replace([np.inf, -np.inf], np.nan)
    
    return df_feats

def main():
    logger.info("Pipeline started...")

    # Load data
    train = pd.read_csv("data/train.csv")
    test  = pd.read_csv("data/test.csv")

    # Extract features
    logger.info("Extracting features...")
    X_train_df = get_features(train)
    X_test_df  = get_features(test)

    # Fill NaN values with column means (fixed warning)
    for df in [X_train_df, X_test_df]:
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())

    # Encode categorical features
    logger.info("Encoding categorical features...")
    le_size = LabelEncoder()
    all_sizes = pd.concat([X_train_df["radar_bird_size"], X_test_df["radar_bird_size"]])
    le_size.fit(all_sizes)
    X_train_df["radar_bird_size"] = le_size.transform(X_train_df["radar_bird_size"])
    X_test_df["radar_bird_size"]  = le_size.transform(X_test_df["radar_bird_size"])

    # Prepare data
    X = X_train_df.drop(columns=["track_id"])
    X_test = X_test_df.drop(columns=["track_id"])
    le = LabelEncoder()
    Y = le.fit_transform(train["bird_group"])
    groups = X_train_df["track_id"]

    # Feature importance analysis
    logger.info("Analyzing feature importance...")
    temp_scaler = StandardScaler()
    X_temp_scaled = temp_scaler.fit_transform(X.fillna(0))
    rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X_temp_scaled, Y)
    importance = pd.DataFrame({
        'feature': X.columns,
        'importance': rf.feature_importances_
    }).sort_values('importance', ascending=False)
    logger.info(f"\nTop 10 features:\n{importance.head(10).to_string(index=False)}")

    # Cross-validation setup
    skf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    oof = np.zeros((len(X), len(le.classes_)))
    test_preds = np.zeros((len(X_test), len(le.classes_)))

    # Use best architecture from previous run
    best_arch = (128, 64)
    logger.info(f"\nUsing best architecture: {best_arch}")

    # Train final model with best architecture
    logger.info("\nTraining final model with best architecture...")
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, Y, groups)):
        logger.info(f"Fold {fold+1}")
        X_tr, y_tr = X.iloc[train_idx], Y[train_idx]
        X_va, y_va = X.iloc[val_idx], Y[val_idx]

        # Scale within fold
        scaler = StandardScaler()
        X_tr_scaled = scaler.fit_transform(X_tr)
        X_va_scaled = scaler.transform(X_va)
        X_test_scaled_fold = scaler.transform(X_test)

        model = MLPClassifier(
            hidden_layer_sizes=best_arch,
            activation="relu",
            solver="adam",
            alpha=1e-4,
            learning_rate_init=1e-3,
            batch_size=128,
            max_iter=1000,
            early_stopping=True,
            n_iter_no_change=20,
            random_state=42,
            verbose=False  # Set to True if you want to see training progress
        )

        model.fit(X_tr_scaled, y_tr)
        oof[val_idx] = model.predict_proba(X_va_scaled)
        test_preds += model.predict_proba(X_test_scaled_fold) / skf.n_splits

    final_score = log_loss(Y, oof)
    logger.info(f"\nFinal CV LogLoss: {final_score:.4f}")

    # Create submission
    result = pd.DataFrame(test_preds, columns=le.classes_)
    result.insert(0, "track_id", X_test_df["track_id"])
    result.to_csv("data/submission_mlp.csv", index=False)
    logger.info("Submission saved as submission_mlp.csv")
    
    # Optional: Save feature importance for reference
    importance.to_csv("data/feature_importance.csv", index=False)
    logger.info("Feature importance saved as feature_importance.csv")


if __name__ == "__main__":
    main()