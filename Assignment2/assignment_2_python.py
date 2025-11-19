import json, numpy as np, pandas as pd, matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, PredefinedSplit, GridSearchCV
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score, make_scorer

# function to load calibration parameters from file
def load_calib(path):
    j = json.load(open(path, "r"))
    acc_off = np.array(j.get("acc_offset", [0,0,0]), float)
    acc_s   = np.array(j.get("acc_scale",  [1,1,1]), float)
    gyr_b   = np.array(j.get("gyr_bias",   [0,0,0]), float)
    mag_hi  = np.array(j.get("mag_hard_iron", [0,0,0]), float)
    mag_si  = np.array(j.get("mag_soft_iron",  np.eye(3).tolist()), float)
    return acc_off, acc_s, gyr_b, mag_hi, mag_si

# function to correct sensor data using calibration parameters
def apply_calib(d, acc_off, acc_s, gyr_b, mag_hi, mag_si):
    out = d.copy()
    A = out[["acc_x[g]","acc_y[g]","acc_z[g]"]].to_numpy(float)
    G = out[["gyro_x[dps]","gyro_y[dps]","gyro_z[dps]"]].to_numpy(float)
    M = out[["mag_x[uT]","mag_y[uT]","mag_z[uT]"]].to_numpy(float)

    out[["acc_x[g]","acc_y[g]","acc_z[g]"]] = (A - acc_off) * acc_s
    out[["gyro_x[dps]","gyro_y[dps]","gyro_z[dps]"]] = (G - (gyr_b)/1000.0)
    out[["mag_x[uT]","mag_y[uT]","mag_z[uT]"]] = (M - (mag_hi*0.1)) @ mag_si.T
    return out

# function to perform rolling average on sensor data for smoothing the data
def rolling_mean(dfcols, w):
    if (w and w > 1):
        return dfcols.rolling(w, min_periods = 1, center = True). mean()
    else:
        return dfcols

# function to calculate Euler angles from given IMU sensor data
def compute_euler(df_sensor):

    acc = rolling_mean(df_sensor[["acc_x[g]","acc_y[g]","acc_z[g]"]], ROLLING_WINDOW).to_numpy()
    mag = rolling_mean(df_sensor[["mag_x[uT]","mag_y[uT]","mag_z[uT]"]], ROLLING_WINDOW).to_numpy()

    ax, ay, az = acc[:,0], acc[:,1], acc[:,2]

    # Roll/Pitch from accelerometer (deg)
    roll  = np.degrees(np.arctan2(ay, az))
    pitch = np.degrees(np.arctan2(-ax, np.sqrt(ay**2 + az**2)))

    # Tilt compensated yaw from magnetometer (deg) using roll/pitch above
    cr, sr = np.cos(np.radians(roll)),  np.sin(np.radians(roll))
    cp, sp = np.cos(np.radians(pitch)), np.sin(np.radians(pitch))
    mx, my, mz = mag[:,0], mag[:,1], mag[:,2]
    m_xc = mx*cp + my*sr*sp + mz*cr*sp
    m_yc = my*cr - mz*sr
    yaw  = np.degrees(np.arctan2(m_yc, m_xc))  

    return pd.DataFrame({"roll_deg": roll, "pitch_deg": pitch, "yaw_deg": yaw})

def angle_diff_deg(pred, true):
    """Minimal signed difference between angles in degrees, in [-180, 180]."""
    diff = pred - true
    diff = (diff + 180.0) % 360.0 - 180.0
    return diff

# function for metrics calculation
def compute_metrics(name, Y_true, Y_pred, idx_subset):

    yt = Y_true[idx_subset]   # (N_sub, 3)
    yp = Y_pred[idx_subset]   # (N_sub, 3)

    # diff array same shape
    diff = np.zeros_like(yp)

    # roll, pitch: normal subtraction
    diff[:, 0] = yp[:, 0]- yt[:, 0]
    diff[:, 1] = yp[:, 1]- yt[:, 1]

    # yaw: wrapped difference
    diff[:, 2] = angle_diff_deg(yp[:, 2], yt[:, 2])

    # RMSE / MAE from diff
    rmse = np.sqrt(np.mean(diff**2, axis=0))
    mae  = np.mean(np.abs(diff), axis=0)

    yaw_pred_aligned = yt[:, 2] + diff[:, 2]
    Y_true_all = yt
    Y_pred_all = np.column_stack([yp[:, 0], yp[:, 1], yaw_pred_aligned])

    r2 = r2_score(Y_true_all, Y_pred_all, multioutput='raw_values')

    print(f"\n{name} metrics vs Calibrated Euler")
    print(f"RMSE (roll,pitch,yaw): {rmse.round(3)}")
    print(f"MAE  (roll,pitch,yaw): {mae.round(3)}")
    print(f"R²   (roll,pitch,yaw): {r2.round(3)}")

# data path
csv_path   = "Data_Log_3.csv"
calib_json = "Calibration_coeff.json"
ROLLING_WINDOW = 5
RANDOM_STATE   = 22

input_features = ["acc_x[g]","acc_y[g]","acc_z[g]","gyro_x[dps]","gyro_y[dps]","gyro_z[dps]","mag_x[uT]","mag_y[uT]","mag_z[uT]"]
targets  = ["roll_deg","pitch_deg","yaw_deg"] 

# load dataset and calibration parameters
df = pd.read_csv(csv_path)
acc_off, acc_s, gyr_b, mag_hi, mag_si = load_calib(calib_json)

# raw and calibrated IMU
df_raw = df.copy()
df_cal = apply_calib(df, acc_off, acc_s, gyr_b, mag_hi, mag_si)

# Euler angles from raw and from calibrated IMU
Y_raw_df = compute_euler(df_raw)  # labels for training
Y_cal_df = compute_euler(df_cal)  # labels for testing

yaw_raw = Y_raw_df["yaw_deg"].to_numpy()
yaw_cal = Y_cal_df["yaw_deg"].to_numpy()

# unwrap raw yaw
yaw_raw_unwrapped = np.unwrap(np.deg2rad(yaw_raw))
yaw_raw_unwrapped_deg = np.rad2deg(yaw_raw_unwrapped)
Y_raw_df["yaw_deg"] = yaw_raw_unwrapped_deg

# building datasets (converting to numpy array)
Y_raw_all = Y_raw_df[targets].to_numpy(float)   # raw Euler (train labels)
Y_cal_all = Y_cal_df[targets].to_numpy(float)   # calibrated Euler (true labels)
X_raw = df_raw[input_features].to_numpy(float)

N = len(X_raw)
indices = np.arange(N)

# create train/val/test splits on indices 
idx_tr, idx_tmp = train_test_split(indices, test_size = 0.4, random_state = RANDOM_STATE, shuffle = True)
idx_val, idx_te = train_test_split(idx_tmp, test_size = 0.5, random_state = RANDOM_STATE+1, shuffle = True)

# split train/validation/test dataset
X_tr = X_raw[idx_tr]
X_val = X_raw[idx_val]
X_te = X_raw[idx_te]

Y_tr_raw  = Y_raw_all[idx_tr]   # raw Euler labels for training
Y_val_raw = Y_raw_all[idx_val]  # raw Euler labels for validation

# Calibrated true labels (for metrics)
Y_trv_cal = Y_cal_all[np.concatenate([idx_tr, idx_val])]
Y_te_cal  = Y_cal_all[idx_te]

# model definitions for hyperparameter tuning
def neg_mean_rmse(y_true, y_pred):
    rmse = np.sqrt(np.mean((y_true - y_pred)**2, axis=0))
    return -float(np.mean(rmse))

rmse_scorer = make_scorer(neg_mean_rmse, greater_is_better=True)

X_tune = np.vstack([X_tr, X_val])
Y_tune = np.vstack([Y_tr_raw, Y_val_raw])
test_fold = np.concatenate([
    np.full(len(X_tr), -1),
    np.zeros(len(X_val), dtype=int)
])
ps = PredefinedSplit(test_fold)

lin = LinearRegression()

poly_lr = Pipeline([
    ("poly",   PolynomialFeatures(include_bias=False)),
    ("scaler", StandardScaler()),
    ("lin",    LinearRegression())
])

poly_cv = GridSearchCV(
    poly_lr,
    {"poly__degree":[2, 3]},
    cv=ps,
    scoring=rmse_scorer,
    refit=True
)
poly_cv.fit(X_tune, Y_tune)
best_poly = poly_cv.best_estimator_
print("Best Polynomial degree:", poly_cv.best_params_["poly__degree"])

tree = DecisionTreeRegressor(random_state=RANDOM_STATE)
tree_cv = GridSearchCV(
    tree,
    {"max_depth":[5,10,15,None], "min_samples_leaf":[1,5,15]},
    cv=ps,
    scoring=rmse_scorer,
    refit=True
)
tree_cv.fit(X_tune, Y_tune)
best_tree = tree_cv.best_estimator_
print("Best Tree params:", tree_cv.best_params_)

# train final model with both training and validation set using best hyperparameters
X_trv     = np.vstack([X_tr, X_val])
Y_trv_raw = np.vstack([Y_tr_raw, Y_val_raw])

lin.fit(X_trv, Y_trv_raw)
best_poly.fit(X_trv, Y_trv_raw)
best_tree.fit(X_trv, Y_trv_raw)

# prediction on full dataset
pred_lin_all  = lin.predict(X_raw)   
pred_poly_all = best_poly.predict(X_raw)
pred_tree_all = best_tree.predict(X_raw)

# Train + Validation indices combined
idx_trv = np.concatenate([idx_tr, idx_val])

# Train+Validation data metrics (how well raw-trained model approximates calibrated Euler angles on training data)
compute_metrics("Linear regression (Train+Val)", Y_cal_all, pred_lin_all,  idx_trv)
compute_metrics("Polynomial Regression (Train+Val)", Y_cal_all, pred_poly_all, idx_trv)
compute_metrics("Decision Tree (Train+Val)", Y_cal_all, pred_tree_all, idx_trv)

# Test Metrics
compute_metrics("Linear regression (Test)",  Y_cal_all, pred_lin_all,  idx_te)
compute_metrics("Polynomial regression (Test)", Y_cal_all, pred_poly_all, idx_te)
compute_metrics("Decision Tree (Test)", Y_cal_all, pred_tree_all, idx_te)

# comparison plots for roll, pitch and yaw
idx_full = np.arange(N)
angle_names = ["roll_deg", "pitch_deg", "yaw_deg"]   

for angle_idx, angle_name in enumerate(angle_names):
    true_angle = Y_cal_all[:, angle_idx]  # calibrated angle

    # Linear regression plot
    plt.figure(figsize=(12,4))
    plt.plot(idx_full, true_angle, label=f"{angle_name} True (calibrated)", color="k", linewidth=3.0)
    plt.plot(idx_full, pred_lin_all[:, angle_idx], '--', label="Linear Pred")
    plt.xlabel("sample index")
    plt.ylabel("deg")
    plt.title(f"{angle_name.upper()}: True (calibrated) vs Linear regression prediction (full dataset)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Polynomial regression plot
    plt.figure(figsize=(12,4))
    plt.plot(idx_full, true_angle, label=f"{angle_name} True (calibrated)", color="k", linewidth=3.0)
    plt.plot(idx_full, pred_poly_all[:, angle_idx], '--', label="Poly-Linear Pred")
    plt.xlabel("sample index")
    plt.ylabel("deg")
    plt.title(f"{angle_name.upper()}: True (calibrated) vs Polynomial regression prediction (full dataset)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Decision Tree plot
    plt.figure(figsize=(12,4))
    plt.plot(idx_full, true_angle, label=f"{angle_name} True (calibrated)", color="k", linewidth=3.0)
    plt.plot(idx_full, pred_tree_all[:, angle_idx], '--', label="Decision Tree Pred")
    plt.xlabel("sample index")
    plt.ylabel("deg")
    plt.title(f"{angle_name.upper()}: True (calibrated) vs Decision Tree prediction (full dataset)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

