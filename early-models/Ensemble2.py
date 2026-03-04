import json
import pandas as pd
import numpy as np
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier
from shapely import wkb
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import log_loss


# --------------------------------------------------
# Utilities
# --------------------------------------------------

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


# --------------------------------------------------
# Feature Engineering
# --------------------------------------------------

def get_features(df):
    feats = []
    print(f"Processing {len(df)} rows...")

    df["ts"] = pd.to_datetime(df["timestamp_start_radar_utc"])

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

        # ---------------- PATH ----------------
        if len(coords) > 1:
            lon1, lat1 = coords[:-1, 0], coords[:-1, 1]
            lon2, lat2 = coords[1:, 0], coords[1:, 1]

            seg_dist = haversine_vec(lon1, lat1, lon2, lat2)
            path_length = np.sum(seg_dist)
            displacement = haversine_vec(
                coords[0, 0], coords[0, 1],
                coords[-1, 0], coords[-1, 1]
            )

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

        # ---------------- TIME ----------------
        hour = row.ts.hour + row.ts.minute / 60.0
        day_of_year = row.ts.timetuple().tm_yday
        is_night = int(hour < 6 or hour > 20)

        # ---------------- FEATURES ----------------
        f = {
            "track_id": row.track_id,
            "point_count": len(traj),
            "total_duration": total_duration,
            "airspeed": airspeed,

            # time
            "hour": hour,
            "day_of_year": day_of_year,
            "is_night": is_night,

            # speed
            "speed_mean": np.mean(inst_speed),
            "speed_std": np.std(inst_speed),

            # turns
            "mean_turn": np.mean(turn_angles),
            "std_turn": np.std(turn_angles),

            # altitude
            "mean_alt": np.mean(altitudes),
            "alt_std": np.std(altitudes),
            "climb_rate_mean": np.mean(dz / dt_safe) if len(dz) > 0 else 0,

            # rcs
            "mean_rcs": np.mean(rcs),
            "rcs_std": np.std(rcs),

            # geometry
            "path_length": path_length,
            "displacement": displacement,
            "straightness": straightness,
        }

        feats.append(f)

    return pd.DataFrame(feats)

def analyze_feature_importance(model, X, label_encoder, top_n=15):
    import shap

    print("\n" + "="*50)
    print("FEATURE IMPORTANCE ANALYSIS (LightGBM)")
    print("="*50)

    feature_names = X.columns
    importances = model.feature_importances_

    imp_df = pd.DataFrame({
        "feature": feature_names,
        "importance": importances
    }).sort_values("importance", ascending=False)

    print(f"\nTop {top_n} Features (Gain Importance):")
    print(imp_df.head(top_n))

    print("\nRunning SHAP analysis...")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    # ---------- HANDLE BOTH SHAP FORMATS ----------
    if isinstance(shap_values, list):
        # Old format: list of (n_samples, n_features)
        shap_array = np.stack(shap_values, axis=-1)
    else:
        # New format: already (n_samples, n_features, n_classes)
        shap_array = shap_values

    print("\nSHAP Direction (Mean Impact per Class):")

    n_classes = shap_array.shape[2]

    for class_idx in range(n_classes):
        mean_impact = shap_array[:, :, class_idx].mean(axis=0)

        shap_df = pd.DataFrame({
            "feature": feature_names,
            "mean_shap": mean_impact
        }).sort_values("mean_shap", ascending=False)

        print("\n" + "-"*40)
        print(f"Class: {label_encoder.classes_[class_idx]}")

        print("\nTop Positive Drivers:")
        print(shap_df.head(5))

        print("\nTop Negative Drivers:")
        print(shap_df.tail(5))

    print("\nAnalysis complete.\n")

    return imp_df

# --------------------------------------------------
# MAIN
# --------------------------------------------------

def main():
    print("Pipeline started...")

    train = pd.read_csv("data/train.csv")
    test  = pd.read_csv("data/test.csv")

    print("Extracting features...")
    X_train_df = get_features(train)
    X_test_df  = get_features(test)

    X      = X_train_df.drop(columns=["track_id"])
    X_test = X_test_df.drop(columns=["track_id"])

    le = LabelEncoder()
    Y = le.fit_transform(train["bird_group"])
    n_classes = len(le.classes_)

    groups = X_train_df["track_id"]

    skf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)

    model_names = ["lgb", "xgb", "cat"]

    oof_all  = {m: np.zeros((len(X), n_classes)) for m in model_names}
    test_all = {m: np.zeros((len(X_test), n_classes)) for m in model_names}

    for model_name in model_names:
        print(f"\nTraining: {model_name.upper()}")

        for fold, (train_idx, val_idx) in enumerate(skf.split(X, Y, groups)):
            print(f"  Fold {fold+1}")

            X_tr, y_tr = X.iloc[train_idx], Y[train_idx]
            X_va, y_va = X.iloc[val_idx],   Y[val_idx]

            if model_name == "lgb":
                model = lgb.LGBMClassifier(
                    objective="multiclass",
                    num_class=n_classes,
                    n_estimators=1500,
                    learning_rate=0.02,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    random_state=42
                )
                model.fit(
                    X_tr, y_tr,
                    eval_set=[(X_va, y_va)],
                    callbacks=[lgb.early_stopping(100, verbose=False)]
                )

            elif model_name == "xgb":
                model = xgb.XGBClassifier(
                    objective="multi:softprob",
                    num_class=n_classes,
                    n_estimators=1500,
                    learning_rate=0.02,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    eval_metric="mlogloss",
                    early_stopping_rounds=100,
                    random_state=42
                )
                model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)

            else:
                model = CatBoostClassifier(
                    iterations=1500,
                    learning_rate=0.02,
                    loss_function="MultiClass",
                    eval_metric="MultiClass",
                    early_stopping_rounds=100,
                    random_seed=42,
                    verbose=0,
                    auto_class_weights="Balanced"
                )
                model.fit(X_tr, y_tr, eval_set=(X_va, y_va))

            oof_all[model_name][val_idx] = model.predict_proba(X_va)
            test_all[model_name] += model.predict_proba(X_test) / 5

        print(f"{model_name.upper()} CV LogLoss: {log_loss(Y, oof_all[model_name]):.4f}")
        print(f"{model_name.upper()} CV LogLoss: ...")        
        if model_name == "lgb":
            analyze_feature_importance(model, X, le)
    
    # Ensemble
    oof_ensemble  = np.mean([oof_all[m] for m in model_names], axis=0)
    test_ensemble = np.mean([test_all[m] for m in model_names], axis=0)

    print("Ensemble LogLoss:", log_loss(Y, oof_ensemble))

    result = pd.DataFrame(test_ensemble, columns=le.classes_)
    result.insert(0, "track_id", X_test_df["track_id"])
    result.to_csv("data/submission.csv", index=False)
    print("Submission saved.")


if __name__ == "__main__":
    main()