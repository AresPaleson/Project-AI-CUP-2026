import json
import pandas as pd
import numpy as np
from shapely import wkb
from sklearn.model_selection import StratifiedGroupKFold, RandomizedSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler, label_binarize
from sklearn.metrics import average_precision_score
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.feature_selection import VarianceThreshold
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from scipy.stats import uniform, randint
import logging
import warnings
warnings.filterwarnings('ignore')

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def haversine_vec(lon1, lat1, lon2, lat2):
    """Calculate haversine distance between points"""
    R = 6371000
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)
    a = np.sin(dphi/2)**2 + np.cos(phi1)*np.cos(phi2)*np.sin(dlambda/2)**2
    return 2 * R * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

def decode_trajectory(hex_str):
    """Decode trajectory from hex string"""
    points = list(wkb.loads(hex_str, hex=True).coords)
    return np.array(points)

def get_enhanced_features(df):
    """Extract enhanced features from trajectory data"""
    feats = []
    logger.info(f"Processing {len(df)} rows for feature extraction...")
    df = df.copy()
    df["ts"] = pd.to_datetime(df["timestamp_start_radar_utc"])
    if "timestamp_end_radar_utc" in df.columns:
        df["ts_end"] = pd.to_datetime(df["timestamp_end_radar_utc"])
    else:
        df["ts_end"] = df["ts"]

    for idx, row in enumerate(df.itertuples(index=False)):
        if idx % 1000 == 0 and idx > 0:
            logger.info(f"Processed {idx}/{len(df)} rows...")
            
        try:
            traj = decode_trajectory(row.trajectory)
            times = np.array(json.loads(row.trajectory_time))
            coords = traj[:, :2]
            altitudes = traj[:, 2]
            rcs = traj[:, 3]
            
            dt = np.diff(times)
            dt_safe = np.where(dt > 0, dt, 1e-9)
            total_duration = times[-1] - times[0] if len(times) > 1 else 0
            airspeed = float(row.airspeed) if pd.notna(row.airspeed) else 0

            lon1, lat1 = coords[:-1, 0], coords[:-1, 1]
            lon2, lat2 = coords[1:, 0], coords[1:, 1]

            # Basic trajectory features
            seg_dist = haversine_vec(lon1, lat1, lon2, lat2)
            path_length = np.sum(seg_dist)
            displacement = haversine_vec(coords[0, 0], coords[0, 1], coords[-1, 0], coords[-1, 1])
            inst_speed = seg_dist / dt_safe
            
            # RCS features
            rcs_max = np.max(rcs) if len(rcs) > 0 else 0
            rcs_diff = np.abs(np.diff(rcs))
            mean_rcs_change = np.mean(rcs_diff) if len(rcs_diff) > 0 else 0

            # Heading features
            dlon = np.radians(lon2 - lon1)
            dlat = np.radians(lat2 - lat1)
            headings = np.degrees(np.arctan2(dlon, dlat)) % 360
            turn_angles = np.abs(np.diff(headings))
            turn_angles = np.where(turn_angles > 180, 360 - turn_angles, turn_angles)
            straightness = displacement / (path_length + 1e-9)
            
            # Rate of turn change
            turn_rate = turn_angles / dt_safe[1:] if len(turn_angles) > 1 else np.array([0])
            
            # Energy metrics
            potential_energy = altitudes * 9.81
            kinetic_energy = 0.5 * inst_speed**2
            
            # Trajectory efficiency
            sinuosity = path_length / displacement if displacement > 0 else 0
            
            # Vertical acceleration
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
            radar_bird_size = str(row.radar_bird_size) if hasattr(row, 'radar_bird_size') else "Unknown"

            # ============= ENHANCED FEATURES =============
            
            # 1. Speed percentiles and advanced metrics
            if len(inst_speed) > 0:
                speed_percentiles = np.percentile(inst_speed, [10, 25, 50, 75, 90])
                speed_volatility = np.max(inst_speed) / (np.min(inst_speed) + 1e-9)
                
                # Speed entropy
                if len(inst_speed) > 1:
                    speed_hist, _ = np.histogram(inst_speed, bins=10)
                    speed_hist = speed_hist / (np.sum(speed_hist) + 1e-9)
                    speed_entropy = -np.sum(speed_hist * np.log(speed_hist + 1e-9))
                else:
                    speed_entropy = 0
            else:
                speed_percentiles = [0]*5
                speed_volatility = 0
                speed_entropy = 0
            
            # 2. Heading consistency
            if len(headings) > 1:
                heading_rad = np.radians(headings)
                circular_mean_x = np.mean(np.cos(heading_rad))
                circular_mean_y = np.mean(np.sin(heading_rad))
                heading_consistency = np.sqrt(circular_mean_x**2 + circular_mean_y**2)
                
                # Direction changes
                heading_diffs = np.abs(np.diff(headings))
                heading_diffs = np.where(heading_diffs > 180, 360 - heading_diffs, heading_diffs)
                direction_changes = np.sum(heading_diffs > 30)
                direction_change_rate = direction_changes / (len(headings) + 1e-9)
            else:
                heading_consistency = 1
                direction_changes = 0
                direction_change_rate = 0
            
            # 3. Enhanced altitude features
            if len(altitudes) > 2:
                alt_percentiles = np.percentile(altitudes, [10, 25, 50, 75, 90])
                alt_range_ratio = (np.max(altitudes) - np.min(altitudes)) / (np.mean(altitudes) + 1e-9)
                alt_skewness = np.mean((altitudes - np.mean(altitudes))**3) / (np.std(altitudes)**3 + 1e-9)
                
                # Altitude oscillation
                alt_peaks = np.sum((altitudes[1:-1] > altitudes[:-2]) & (altitudes[1:-1] > altitudes[2:]))
                alt_valleys = np.sum((altitudes[1:-1] < altitudes[:-2]) & (altitudes[1:-1] < altitudes[2:]))
                alt_oscillations = (alt_peaks + alt_valleys) / 2
                alt_oscillation_rate = alt_oscillations / (len(altitudes) + 1e-9)
            else:
                alt_percentiles = [0]*5
                alt_range_ratio = 0
                alt_skewness = 0
                alt_oscillation_rate = 0
            
            # 4. Enhanced RCS features
            if len(rcs) > 2:
                rcs_percentiles = np.percentile(rcs, [10, 25, 50, 75, 90])
                
                # RCS autocorrelation (persistence)
                if len(rcs) > 5:
                    rcs_centered = rcs - np.mean(rcs)
                    rcs_acf = np.correlate(rcs_centered, rcs_centered, mode='full')
                    rcs_acf = rcs_acf[len(rcs_acf)//2:]
                    rcs_acf = rcs_acf / (rcs_acf[0] + 1e-9)
                    rcs_persistence = np.mean(rcs_acf[1:min(5, len(rcs_acf))])
                else:
                    rcs_persistence = 0
                
                # RCS trend
                if len(rcs) > 1:
                    rcs_trend = np.polyfit(np.arange(len(rcs)), rcs, 1)[0]
                else:
                    rcs_trend = 0
            else:
                rcs_percentiles = [0]*5
                rcs_persistence = 0
                rcs_trend = 0
            
            # 5. Temporal cyclical encoding
            hour_sin = np.sin(2 * np.pi * hour / 24)
            hour_cos = np.cos(2 * np.pi * hour / 24)
            day_sin = np.sin(2 * np.pi * day_of_year / 365)
            day_cos = np.cos(2 * np.pi * day_of_year / 365)
            is_weekend = 1 if row.ts.weekday() >= 5 else 0
            
            # 6. Trajectory curvature
            if len(coords) > 3:
                vec1 = coords[1:-1] - coords[:-2]
                vec2 = coords[2:] - coords[1:-1]
                dot_product = np.sum(vec1 * vec2, axis=1)
                norm_product = np.linalg.norm(vec1, axis=1) * np.linalg.norm(vec2, axis=1)
                cos_angles = dot_product / (norm_product + 1e-9)
                cos_angles = np.clip(cos_angles, -1, 1)
                angles = np.arccos(cos_angles)
                mean_curvature = np.mean(angles)
                std_curvature = np.std(angles)
            else:
                mean_curvature = 0
                std_curvature = 0
            
            # 7. Interaction features
            if len(inst_speed) > 1 and len(altitudes) > 1:
                speed_alt_corr = np.corrcoef(inst_speed, altitudes[:-1])[0, 1] if len(inst_speed) == len(altitudes[:-1]) else 0
            else:
                speed_alt_corr = 0
            
            if len(turn_angles) > 0 and len(inst_speed) > 1:
                min_len = min(len(turn_angles), len(inst_speed)-1)
                if min_len > 1:
                    turn_speed_corr = np.corrcoef(turn_angles[:min_len], inst_speed[:min_len])[0, 1]
                else:
                    turn_speed_corr = 0
            else:
                turn_speed_corr = 0

            # Compile all features
            f = {
                "track_id": row.track_id,
                "point_count": len(traj),
                "total_duration": total_duration,
                "airspeed": airspeed,
                "hour": hour,
                "day_of_year": day_of_year,
                "hour_sin": hour_sin,
                "hour_cos": hour_cos,
                "day_sin": day_sin,
                "day_cos": day_cos,
                "is_weekend": is_weekend,
                "meta_duration": meta_duration,
                "speed_mean": np.mean(inst_speed) if len(inst_speed) > 0 else 0,
                "speed_std": np.std(inst_speed) if len(inst_speed) > 0 else 0,
                "speed_p10": speed_percentiles[0],
                "speed_p25": speed_percentiles[1],
                "speed_p50": speed_percentiles[2],
                "speed_p75": speed_percentiles[3],
                "speed_p90": speed_percentiles[4],
                "speed_volatility": speed_volatility,
                "speed_entropy": speed_entropy,
                "mean_turn": np.mean(turn_angles) if len(turn_angles) > 0 else 0,
                "std_turn": np.std(turn_angles) if len(turn_angles) > 0 else 0,
                "mean_turn_rate": np.mean(turn_rate) if len(turn_rate) > 0 else 0,
                "std_turn_rate": np.std(turn_rate) if len(turn_rate) > 0 else 0,
                "heading_consistency": heading_consistency,
                "direction_changes": direction_changes,
                "direction_change_rate": direction_change_rate,
                "mean_alt": np.mean(altitudes),
                "alt_std": np.std(altitudes),
                "alt_p10": alt_percentiles[0],
                "alt_p25": alt_percentiles[1],
                "alt_p50": alt_percentiles[2],
                "alt_p75": alt_percentiles[3],
                "alt_p90": alt_percentiles[4],
                "alt_range_ratio": alt_range_ratio,
                "alt_skewness": alt_skewness,
                "alt_oscillation_rate": alt_oscillation_rate,
                "climb_rate_mean": np.mean(dz / dt_safe) if len(dz) > 0 else 0,
                "vertical_accel": vertical_accel,
                "min_z_meta": min_z_meta,
                "max_z_meta": max_z_meta,
                "mean_rcs": np.mean(rcs),
                "rcs_std": np.std(rcs),
                "rcs_max": rcs_max,
                "rcs_p10": rcs_percentiles[0],
                "rcs_p25": rcs_percentiles[1],
                "rcs_p50": rcs_percentiles[2],
                "rcs_p75": rcs_percentiles[3],
                "rcs_p90": rcs_percentiles[4],
                "rcs_persistence": rcs_persistence,
                "rcs_trend": rcs_trend,
                "mean_rcs_change": mean_rcs_change,
                "path_length": path_length,
                "displacement": displacement,
                "straightness": straightness,
                "sinuosity": sinuosity,
                "mean_potential_energy": np.mean(potential_energy),
                "mean_kinetic_energy": np.mean(kinetic_energy),
                "mean_curvature": mean_curvature,
                "std_curvature": std_curvature,
                "speed_alt_corr": speed_alt_corr,
                "turn_speed_corr": turn_speed_corr,
                "radar_bird_size": radar_bird_size
            }
            feats.append(f)
            
        except Exception as e:
            logger.warning(f"Error processing row {idx}: {e}")
            continue
    
    df_feats = pd.DataFrame(feats)
    df_feats = df_feats.replace([np.inf, -np.inf], np.nan)
    
    logger.info(f"Feature extraction complete. Shape: {df_feats.shape}")
    return df_feats

def select_features(X, y, threshold=0.01):
    """Select important features"""
    logger.info("Starting feature selection...")
    
    # Handle missing values
    X_clean = X.fillna(X.median())
    
    # Remove constant features
    constant_cols = [col for col in X_clean.columns if X_clean[col].nunique() <= 1]
    if constant_cols:
        logger.info(f"Removing constant columns: {constant_cols}")
        X_clean = X_clean.drop(columns=constant_cols)
    
    # Remove highly correlated features
    corr_matrix = X_clean.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    high_corr_cols = [column for column in upper.columns if any(upper[column] > 0.95)]
    if high_corr_cols:
        logger.info(f"Removing highly correlated columns: {high_corr_cols}")
        X_clean = X_clean.drop(columns=high_corr_cols)
    
    # Variance threshold
    selector = VarianceThreshold(threshold=threshold)
    X_high_var = selector.fit_transform(X_clean)
    selected_features = X_clean.columns[selector.get_support()].tolist()
    logger.info(f"Features after variance threshold: {len(selected_features)}")
    
    # Model-based selection
    if len(selected_features) > 0:
        gb = GradientBoostingClassifier(n_estimators=100, random_state=42)
        gb.fit(X_clean[selected_features], y)
        
        # Select features with importance > mean
        importance = gb.feature_importances_
        mean_importance = np.mean(importance)
        selected_features = [selected_features[i] for i in range(len(selected_features)) 
                           if importance[i] > mean_importance]
        logger.info(f"Features after importance threshold: {len(selected_features)}")
    
    return selected_features

def train_ensemble(X_tr, y_tr, X_va, y_va, X_test_scaled, n_classes):
    """Train multiple models and ensemble them"""
    
    models = {
        'mlp': MLPClassifier(
            hidden_layer_sizes=(128, 64, 32),
            activation="relu",
            solver="adam",
            alpha=1e-4,
            learning_rate_init=1e-3,
            batch_size=128,
            max_iter=500,
            early_stopping=True,
            n_iter_no_change=10,
            random_state=42,
            verbose=False
        ),
        'xgb': XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            use_label_encoder=False,
            eval_metric='mlogloss',
            verbosity=0
        ),
        'lgbm': LGBMClassifier(
            n_estimators=200,
            max_depth=8,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            verbose=-1
        ),
        'catboost': CatBoostClassifier(
            iterations=200,
            depth=6,
            learning_rate=0.05,
            random_seed=42,
            verbose=0,
            early_stopping_rounds=20
        ),
        'rf': RandomForestClassifier(
            n_estimators=200,
            max_depth=12,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        )
    }
    
    predictions = []
    model_scores = {}
    
    for name, model in models.items():
        try:
            logger.info(f"Training {name}...")
            model.fit(X_tr, y_tr)
            pred = model.predict_proba(X_va)
            predictions.append(pred)
            
            # Calculate validation score
            y_va_bin = label_binarize(y_va, classes=range(n_classes))
            score = average_precision_score(y_va_bin, pred, average="weighted")
            model_scores[name] = score
            logger.info(f"{name} CV score: {score:.4f}")
            
        except Exception as e:
            logger.warning(f"Error training {name}: {e}")
            continue
    
    if not predictions:
        logger.error("No models trained successfully!")
        return None, {}
    
    # Simple average ensemble
    ensemble_pred = np.mean(predictions, axis=0)
    
    # Weighted average based on validation scores
    if model_scores:
        weights = np.array([model_scores[name] for name in models.keys() if name in model_scores])
        weights = weights / np.sum(weights)
        weighted_ensemble = np.average(predictions, axis=0, weights=weights)
        return weighted_ensemble, model_scores
    
    return ensemble_pred, model_scores

def optimize_hyperparameters(X, y, groups, n_trials=10):
    """Simple hyperparameter optimization"""
    logger.info("Starting hyperparameter optimization...")
    
    # Simplified parameter grids
    mlp_params = {
        'hidden_layer_sizes': [(64,), (128,), (64, 32), (128, 64)],
        'alpha': [1e-5, 1e-4, 1e-3],
        'learning_rate_init': [1e-4, 1e-3],
        'batch_size': [64, 128]
    }
    
    best_params = {}
    
    # Simple random search for MLP
    try:
        cv = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=42)
        mlp_search = RandomizedSearchCV(
            MLPClassifier(max_iter=300, early_stopping=True, random_state=42),
            mlp_params,
            n_iter=n_trials,
            cv=cv,
            scoring='average_precision_weighted',
            random_state=42,
            n_jobs=-1
        )
        
        # Note: This will ignore groups - for proper group handling you'd need custom CV
        mlp_search.fit(X, y)
        best_params['mlp'] = mlp_search.best_params_
        logger.info(f"Best MLP params: {mlp_search.best_params_}")
        
    except Exception as e:
        logger.warning(f"Hyperparameter optimization failed: {e}")
        best_params['mlp'] = {
            'hidden_layer_sizes': (128, 64),
            'alpha': 1e-4,
            'learning_rate_init': 1e-3,
            'batch_size': 128
        }
    
    return best_params

def main():
    """Main execution function"""
    logger.info("="*50)
    logger.info("ENHANCED BIRD CLASSIFICATION PIPELINE")
    logger.info("="*50)
    
    try:
        # Load data
        logger.info("Loading data...")
        train = pd.read_csv("data/train.csv")
        test = pd.read_csv("data/test.csv")
        logger.info(f"Train shape: {train.shape}, Test shape: {test.shape}")
        
        # Extract enhanced features
        logger.info("Extracting enhanced features...")
        X_train_df = get_enhanced_features(train)
        X_test_df = get_enhanced_features(test)
        
        # Encode categorical features
        logger.info("Encoding categorical features...")
        le_size = LabelEncoder()
        all_sizes = pd.concat([X_train_df["radar_bird_size"], X_test_df["radar_bird_size"]])
        le_size.fit(all_sizes)
        X_train_df["radar_bird_size"] = le_size.transform(X_train_df["radar_bird_size"])
        X_test_df["radar_bird_size"] = le_size.transform(X_test_df["radar_bird_size"])
        
        # Prepare data
        feature_cols = [col for col in X_train_df.columns if col not in ['track_id']]
        X = X_train_df[feature_cols]
        X_test = X_test_df[feature_cols]
        
        le = LabelEncoder()
        Y = le.fit_transform(train["bird_group"])
        groups = X_train_df["track_id"]
        
        logger.info(f"Number of classes: {len(le.classes_)}")
        logger.info(f"Class distribution: {np.bincount(Y)}")
        
        # Handle missing values
        X = X.fillna(X.median())
        X_test = X_test.fillna(X_test.median())
        
        # Feature selection
        logger.info("Selecting important features...")
        selected_features = select_features(X, Y)
        
        if not selected_features:
            logger.warning("Feature selection returned no features, using all features")
            selected_features = feature_cols
        
        X_selected = X[selected_features]
        X_test_selected = X_test[selected_features]
        logger.info(f"Final feature count: {len(selected_features)}")
        logger.info(f"Selected features: {selected_features[:10]}...")
        
        # Hyperparameter optimization (optional)
        # best_params = optimize_hyperparameters(X_selected, Y, groups, n_trials=5)
        
        # Cross-validation setup
        skf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
        oof = np.zeros((len(X_selected), len(le.classes_)))
        test_preds = np.zeros((len(X_test_selected), len(le.classes_)))
        
        # Store feature importance
        feature_importance = pd.DataFrame(index=selected_features)
        
        logger.info("\nStarting cross-validation...")
        for fold, (train_idx, val_idx) in enumerate(skf.split(X_selected, Y, groups)):
            logger.info(f"\n{'='*40}")
            logger.info(f"FOLD {fold+1}/5")
            logger.info(f"{'='*40}")
            
            X_tr, y_tr = X_selected.iloc[train_idx], Y[train_idx]
            X_va, y_va = X_selected.iloc[val_idx], Y[val_idx]
            
            # Scale features
            scaler = StandardScaler()
            X_tr_scaled = scaler.fit_transform(X_tr)
            X_va_scaled = scaler.transform(X_va)
            X_test_scaled_fold = scaler.transform(X_test_selected)
            
            # Train ensemble
            fold_pred, model_scores = train_ensemble(
                X_tr_scaled, y_tr, 
                X_va_scaled, y_va,
                X_test_scaled_fold, 
                len(le.classes_)
            )
            
            if fold_pred is not None:
                oof[val_idx] = fold_pred
                test_preds += fold_pred / skf.n_splits
                
                # Calculate fold score
                y_va_bin = label_binarize(y_va, classes=range(len(le.classes_)))
                fold_score = average_precision_score(y_va_bin, fold_pred, average="weighted")
                logger.info(f"Fold {fold+1} score: {fold_score:.4f}")
        
        # Final evaluation
        logger.info("\n" + "="*50)
        logger.info("FINAL EVALUATION")
        logger.info("="*50)
        
        Y_bin = label_binarize(Y, classes=range(len(le.classes_)))
        final_score = average_precision_score(Y_bin, oof, average="weighted")
        logger.info(f"\nFinal CV Average Precision (Weighted): {final_score:.4f}")
        
        # Per-class scores
        per_class_scores = average_precision_score(Y_bin, oof, average=None)
        for i, class_name in enumerate(le.classes_):
            logger.info(f"Class {class_name}: {per_class_scores[i]:.4f}")
        
        # Create submission
        logger.info("\nCreating submission file...")
        result = pd.DataFrame(test_preds, columns=le.classes_)
        result.insert(0, "track_id", X_test_df["track_id"])
        
        output_file = "data/submission_enhanced.csv"
        result.to_csv(output_file, index=False)
        logger.info(f"Submission saved as {output_file}")
        
        # Save feature importance
        if hasattr(gb, 'feature_importances_'):
            importance_df = pd.DataFrame({
                'feature': selected_features,
                'importance': gb.feature_importances_
            }).sort_values('importance', ascending=False)
            importance_df.to_csv("data/feature_importance_enhanced.csv", index=False)
            logger.info("Feature importance saved")
        
        logger.info("\n" + "="*50)
        logger.info("PIPELINE COMPLETED SUCCESSFULLY")
        logger.info("="*50)
        
        return final_score
        
    except Exception as e:
        logger.error(f"Pipeline failed: {e}", exc_info=True)
        raise

def fast_inference_mode():
    """Simplified mode for quick inference"""
    logger.info("Running in fast inference mode...")
    
    # Load data
    train = pd.read_csv("data/train.csv")
    test = pd.read_csv("data/test.csv")
    
    # Use original features (faster)
    X_train_df = get_enhanced_features(train)  # Still uses enhanced but you can revert to original
    X_test_df = get_enhanced_features(test)
    
    # Quick processing
    le_size = LabelEncoder()
    all_sizes = pd.concat([X_train_df["radar_bird_size"], X_test_df["radar_bird_size"]])
    le_size.fit(all_sizes)
    X_train_df["radar_bird_size"] = le_size.transform(X_train_df["radar_bird_size"])
    X_test_df["radar_bird_size"] = le_size.transform(X_test_df["radar_bird_size"])
    
    # Use only top features for speed
    top_features = ['point_count', 'speed_mean', 'speed_std', 'mean_alt', 'alt_std', 
                   'mean_rcs', 'path_length', 'straightness', 'hour_sin', 'hour_cos']
    
    X = X_train_df[top_features].fillna(0)
    X_test = X_test_df[top_features].fillna(0)
    
    le = LabelEncoder()
    Y = le.fit_transform(train["bird_group"])
    
    # Single model for speed
    model = XGBClassifier(n_estimators=100, max_depth=6, random_state=42)
    model.fit(X, Y)
    
    # Predict
    test_preds = model.predict_proba(X_test)
    
    # Create submission
    result = pd.DataFrame(test_preds, columns=le.classes_)
    result.insert(0, "track_id", X_test_df["track_id"])
    result.to_csv("data/submission_fast.csv", index=False)
    logger.info("Fast inference complete!")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Bird classification pipeline')
    parser.add_argument('--mode', type=str, default='full', 
                        choices=['full', 'fast'],
                        help='Run mode: full (enhanced) or fast (quick inference)')
    
    args = parser.parse_args()
    
    if args.mode == 'fast':
        fast_inference_mode()
    else:
        main()