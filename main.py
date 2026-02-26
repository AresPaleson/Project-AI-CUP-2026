from math import dist

import pandas as pd
import numpy as np
import lightgbm as lgb
from shapely import coords, wkb
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder


# --- SECTION 1: SMART DECODING ---
def decode_trajectory(hex_str):
    if pd.isna(hex_str):
        return np.zeros((1, 4)) # Return dummy data if empty
    # Decodes EWKB Hex to a numpy array: [Long, Lat, Alt, RCS]
    points = list(wkb.loads(hex_str, hex=True).coords)
    return np.array(points)

# --- SECTION 2: FEATURE ENGINEERING ---
def get_features(df):
    feats = []
    print(f"Processing {len(df)} rows...")
    for i, row in df.iterrows():
        traj = decode_trajectory(row['trajectory'])
        times = np.array(eval(row['trajectory_time'])) # eval to convert string list to numpy array
        
        # Long, Lat
        coords = traj[:, :2] 
        dist = np.sqrt(np.sum(np.diff(coords, axis=0)**2, axis=1)) # Calculate Euclidean distance between consecutive points
        dt = np.diff(times)
        speeds = dist / np.where(dt > 0, dt, 1e-9) 
        
        # Vertical Velocity
        altitudes = traj[:, 2]
        dz = np.diff(altitudes)
        # Flapping
        rcs = traj[:, 3]
        # Tortuosity (Wiggliness)

        start_pos = coords[0]
        end_pos = coords[-1]
        displacement = np.sqrt(np.sum((end_pos - start_pos)**2))
        path_length = np.sum(dist)       

        f = {
            'track_id': row['track_id'],
            'avg_speed': np.mean(speeds) if len(speeds) > 0 else 0,
            'max_speed': np.max(speeds) if len(speeds) > 0 else 0,
            'mean_rcs': np.mean(rcs),
            'std_rcs': np.std(rcs),
            'rcs_cv': np.std(rcs) / np.mean(rcs) + 1e-9 if np.mean(rcs) > 0 else 0,
            'alt_range': row.get('max_z', 0) - row.get('min_z', 0),
            'point_count': len(traj),
            'avg_climb_rate': np.mean(dz / np.where(dt > 0, dt, 1e-9)),
            'tortuosity': path_length / (displacement + 1e-9)
        }
        feats.append(f)
    #print(traj[1])
    return pd.DataFrame(feats)

# --- SECTION 3: THE MAIN PIPELINE ---
def main():
    print("Pipeline...")

    # 1. Load Data
    train = pd.read_csv("data/train.csv")
    test = pd.read_csv("data/test.csv")

    # 2. Feature Engineering
    X_train_df = get_features(train)
    X_test_df = get_features(test)
    X = X_train_df.drop(columns=['track_id'])
    X_test = X_test_df.drop(columns=['track_id'])
    
    # Encode target labels (Species names -> 0-8)
    le = LabelEncoder()
    Y = le.fit_transform(train['bird_group'])

# 3. Model Training
    print("Training LightGBM...")
    clf = lgb.LGBMClassifier(
        objective='multiclass',
        num_class=len(le.classes_),
        n_estimators=200,     
        learning_rate=0.05,   
        class_weight='balanced', 
        random_state=42,
        verbose=-1
    )
    clf.fit(X, Y)

    print("Generating submission...")
    probs = clf.predict_proba(X_test)
    
    result = pd.DataFrame(probs, columns=le.classes_)
    result.insert(0, 'track_id', X_test_df['track_id'])
    result.to_csv("data/submission.csv", index=False)
    print("Saved to data/submission.csv")

if __name__ == "__main__":
    main()