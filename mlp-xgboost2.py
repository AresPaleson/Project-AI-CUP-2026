import json
import pandas as pd
import numpy as np
from shapely import wkb
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import log_loss
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import lightgbm as lgb
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
    df = df.copy()
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
        
        turn_rate = turn_angles / dt_safe[1:] if len(turn_angles) > 1 else np.array([0])
        potential_energy = altitudes * 9.81
        kinetic_energy = 0.5 * inst_speed**2
        sinuosity = path_length / displacement if displacement > 0 else 0
        
        if len(altitudes) > 2:
            dz2 = np.diff(altitudes, n=2)
            vertical_accel = np.mean(dz2 / dt_safe[1:]**2) if len(dz2) > 0 else 0
        else:
            vertical_accel = 0

        dz = np.diff(altitudes)
        hour = row.ts.hour + row.ts.minute / 60.0
        day_of_year = row.ts.timetuple().tm_yday
        meta_duration = (row.ts_end - row.ts).total_seconds() if pd.notna(row.ts_end) else 0
        min_z_meta = float(row.min_z) if hasattr(row, 'min_z') and pd.notna(row.min_z) else 0.0
        max_z_meta = float(row.max_z) if hasattr(row, 'max_z') and pd.notna(row.max_z) else 0.0
        radar_bird_size = str(row.radar_bird_size) if hasattr(row, 'radar_bird_size') and pd.notna(row.radar_bird_size) else "Unknown"

        # --- NEW FEATURES ---
        # Speed percentiles (capture outlier bursts of speed)
        speed_p25 = np.percentile(inst_speed, 25)
        speed_p75 = np.percentile(inst_speed, 75)
        speed_iqr = speed_p75 - speed_p25

        # Altitude range and percentiles
        alt_range = max_z_meta - min_z_meta
        alt_p25 = np.percentile(altitudes, 25)
        alt_p75 = np.percentile(altitudes, 75)

        # RCS percentiles (bird size signal)
        rcs_min = np.min(rcs)
        rcs_max = np.max(rcs)
        rcs_range = rcs_max - rcs_min

        # Circular time features (hour as sin/cos avoids discontinuity at midnight)
        hour_sin = np.sin(2 * np.pi * hour / 24)
        hour_cos = np.cos(2 * np.pi * hour / 24)
        doy_sin = np.sin(2 * np.pi * day_of_year / 365)
        doy_cos = np.cos(2 * np.pi * day_of_year / 365)

        # Net vertical displacement
        net_altitude_change = altitudes[-1] - altitudes[0]

        f = {
            "track_id": row.track_id,
            "point_count": len(traj),
            "total_duration": total_duration,
            "airspeed": airspeed,
            "hour": hour,
            "hour_sin": hour_sin,
            "hour_cos": hour_cos,
            "doy_sin": doy_sin,
            "doy_cos": doy_cos,
            "day_of_year": day_of_year,
            "meta_duration": meta_duration,
            "speed_mean": np.mean(inst_speed),
            "speed_std": np.std(inst_speed),
            "speed_p25": speed_p25,
            "speed_p75": speed_p75,
            "speed_iqr": speed_iqr,
            "mean_turn": np.mean(turn_angles),
            "std_turn": np.std(turn_angles),
            "mean_turn_rate": np.mean(turn_rate),
            "std_turn_rate": np.std(turn_rate),
            "mean_alt": np.mean(altitudes),
            "alt_std": np.std(altitudes),
            "alt_range": alt_range,
            "alt_p25": alt_p25,
            "alt_p75": alt_p75,
            "net_altitude_change": net_altitude_change,
            "climb_rate_mean": np.mean(dz / dt_safe) if len(dz) > 0 else 0,
            "vertical_accel": vertical_accel,
            "min_z_meta": min_z_meta,
            "max_z_meta": max_z_meta,
            "mean_rcs": np.mean(rcs),
            "rcs_std": np.std(rcs),
            "rcs_min": rcs_min,
            "rcs_max": rcs_max,
            "rcs_range": rcs_range,
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
    df_feats = df_feats.replace([np.inf, -np.inf], np.nan)
    return df_feats

def train_mlp(X, Y, groups, X_test, le, skf):
    logger.info("\n" + "="*50)
    logger.info("Training MLP Classifier")
    logger.info("="*50)
    
    oof = np.zeros((len(X), len(le.classes_)))
    test_preds = np.zeros((len(X_test), len(le.classes_)))

    best_arch = (128, 64)
    logger.info(f"Using architecture: {best_arch}")

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, Y, groups)):
        logger.info(f"Fold {fold+1}")
        X_tr, y_tr = X.iloc[train_idx], Y[train_idx]
        X_va, y_va = X.iloc[val_idx], Y[val_idx]

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
            verbose=False
        )

        model.fit(X_tr_scaled, y_tr)
        oof[val_idx] = model.predict_proba(X_va_scaled)
        test_preds += model.predict_proba(X_test_scaled_fold) / skf.n_splits

    final_score = log_loss(Y, oof)
    logger.info(f"MLP CV LogLoss: {final_score:.4f}")
    return test_preds, final_score

def train_xgboost(X, Y, groups, X_test, le, skf):
    logger.info("\n" + "="*50)
    logger.info("Training XGBoost Classifier")
    logger.info("="*50)
    
    oof = np.zeros((len(X), len(le.classes_)))
    test_preds = np.zeros((len(X_test), len(le.classes_)))

    params = {
        'objective': 'multi:softprob',
        'num_class': len(le.classes_),
        'max_depth': 6,
        'learning_rate': 0.1,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'min_child_weight': 1,
        'gamma': 0,
        'reg_alpha': 0.1,
        'reg_lambda': 1,
        'eval_metric': 'mlogloss',
        'seed': 42
    }

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, Y, groups)):
        logger.info(f"Fold {fold+1}")
        X_tr, y_tr = X.iloc[train_idx], Y[train_idx]
        X_va, y_va = X.iloc[val_idx], Y[val_idx]
        
        dtrain = xgb.DMatrix(X_tr, label=y_tr)
        dvalid = xgb.DMatrix(X_va, label=y_va)
        dtest = xgb.DMatrix(X_test)

        model = xgb.train(
            params,
            dtrain,
            num_boost_round=1000,
            evals=[(dvalid, 'valid')],
            early_stopping_rounds=50,
            verbose_eval=False
        )

        oof[val_idx] = model.predict(dvalid)
        test_preds += model.predict(dtest) / skf.n_splits

    final_score = log_loss(Y, oof)
    logger.info(f"XGBoost CV LogLoss: {final_score:.4f}")
    return test_preds, final_score

def train_lightgbm(X, Y, groups, X_test, le, skf):
    """Train LightGBM model and return predictions"""
    logger.info("\n" + "="*50)
    logger.info("Training LightGBM Classifier")
    logger.info("="*50)
    
    oof = np.zeros((len(X), len(le.classes_)))
    test_preds = np.zeros((len(X_test), len(le.classes_)))

    params = {
        'objective': 'multiclass',
        'num_class': len(le.classes_),
        'metric': 'multi_logloss',
        'num_leaves': 63,           # More leaves = more complex trees (XGB uses max_depth=6 ~ 63 leaves)
        'learning_rate': 0.05,      # Lower than XGB — LightGBM converges well with more rounds
        'feature_fraction': 0.8,    # Same as XGB colsample_bytree
        'bagging_fraction': 0.8,    # Same as XGB subsample
        'bagging_freq': 1,          # Required for bagging to activate
        'min_child_samples': 20,    # Regularization: min samples per leaf
        'reg_alpha': 0.1,
        'reg_lambda': 1.0,
        'verbose': -1,
        'seed': 42,
        'n_jobs': -1
    }

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, Y, groups)):
        logger.info(f"Fold {fold+1}")
        X_tr, y_tr = X.iloc[train_idx], Y[train_idx]
        X_va, y_va = X.iloc[val_idx], Y[val_idx]

        dtrain = lgb.Dataset(X_tr, label=y_tr)
        dvalid = lgb.Dataset(X_va, label=y_va, reference=dtrain)

        callbacks = [
            lgb.early_stopping(stopping_rounds=50, verbose=False),
            lgb.log_evaluation(period=-1)  # Suppress per-round output
        ]

        model = lgb.train(
            params,
            dtrain,
            num_boost_round=1000,
            valid_sets=[dvalid],
            callbacks=callbacks
        )

        oof[val_idx] = model.predict(X_va)
        test_preds += model.predict(X_test) / skf.n_splits

    final_score = log_loss(Y, oof)
    logger.info(f"LightGBM CV LogLoss: {final_score:.4f}")
    return test_preds, final_score

def main():
    logger.info("Pipeline started...")

    train = pd.read_csv("data/train.csv")
    test  = pd.read_csv("data/test.csv")

    logger.info("Extracting features...")
    X_train_df = get_features(train)
    X_test_df  = get_features(test)

    for df in [X_train_df, X_test_df]:
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())

    logger.info("Encoding categorical features...")
    le_size = LabelEncoder()
    all_sizes = pd.concat([X_train_df["radar_bird_size"], X_test_df["radar_bird_size"]])
    le_size.fit(all_sizes)
    X_train_df["radar_bird_size"] = le_size.transform(X_train_df["radar_bird_size"])
    X_test_df["radar_bird_size"]  = le_size.transform(X_test_df["radar_bird_size"])

    X = X_train_df.drop(columns=["track_id"])
    X_test = X_test_df.drop(columns=["track_id"])
    le = LabelEncoder()
    Y = le.fit_transform(train["bird_group"])
    groups = X_train_df["track_id"]

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
    importance.to_csv("data/feature_importance.csv", index=False)

    skf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)

    # Train all models
    mlp_preds, mlp_score   = train_mlp(X, Y, groups, X_test, le, skf)
    xgb_preds, xgb_score   = train_xgboost(X, Y, groups, X_test, le, skf)
    lgbm_preds, lgbm_score = train_lightgbm(X, Y, groups, X_test, le, skf)

    logger.info("\n" + "="*50)
    logger.info("Creating submission files")
    logger.info("="*50)

    def save_submission(preds, name, score=None):
        result = pd.DataFrame(preds, columns=le.classes_)
        result.insert(0, "track_id", X_test_df["track_id"])
        result.to_csv(f"data/submission_{name}.csv", index=False)
        score_str = f" (CV LogLoss: {score:.4f})" if score else ""
        logger.info(f"{name} submission saved{score_str}")

    save_submission(mlp_preds,      "mlp",      mlp_score)
    save_submission(xgb_preds,      "xgboost",  xgb_score)
    save_submission(lgbm_preds,     "lightgbm", lgbm_score)

    logger.info("\n" + "="*50)
    logger.info("SUMMARY")
    logger.info("="*50)
    logger.info(f"MLP      CV LogLoss: {mlp_score:.4f}")
    logger.info(f"XGBoost  CV LogLoss: {xgb_score:.4f}")
    logger.info(f"LightGBM CV LogLoss: {lgbm_score:.4f}")

if __name__ == "__main__":
    main()