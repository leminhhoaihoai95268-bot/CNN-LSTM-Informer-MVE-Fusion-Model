# -*- coding: utf-8 -*-

import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import gc
import math
import random
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.layers import Input, Dense, LSTM, Conv1D, Dropout, Flatten
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt



DATA_PATH = r'外层混凝土2.csv'
FEATURE_COLUMNS = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']
TARGET_COLUMN = 'I'

# IMPORTANT: the true I-stage values are NEVER used as model inputs.
# X contains only A-H; y contains only I.
if TARGET_COLUMN in FEATURE_COLUMNS:
    raise ValueError('Target leakage detected: TARGET_COLUMN must not appear in FEATURE_COLUMNS.')

INPUT_DIMS = len(FEATURE_COLUMNS)
TIME_STEPS = 8
LSTM_UNITS = 64
BATCH_SIZE = 16

# Three-stage MVE epochs
EPOCHS_MEAN = 300
EPOCHS_VAR = 200
EPOCHS_JOINT = 200

LEARNING_RATE_MEAN = 1e-3
LEARNING_RATE_VAR = 1e-3
LEARNING_RATE_JOINT = 3e-4

SEEDS = [42, 43, 44, 45, 46]
N_FOLDS = 5

# Within each non-test fold of each CV run:
# first 90% -> proper training; last 10% -> shared validation/conformal calibration.
PROPER_TRAIN_RATIO = 0.90
VAL_CAL_RATIO = 0.10

ALPHA = 0.05               # nominal miscoverage, i.e. 95% interval
EPS = 1e-6

# No early stopping or validation-based model selection is used.
# All specified epochs are always run.

OUTPUT_DIR = Path("MVE_5fold_5seed_3stage_conformal_NO_TARGET_LEAKAGE_results")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)



def set_global_seed(seed: int):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    try:
        tf.config.experimental.enable_op_determinism()
    except Exception:
        pass



class InformerBlock(layers.Layer):
    def __init__(self, d_model, num_heads, dropout=0.05, **kwargs):
        super().__init__(**kwargs)
        self.mha = layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=d_model // num_heads
        )
        self.ffn = tf.keras.Sequential([
            Dense(d_model * 4, activation='relu'),
            Dense(d_model)
        ])
        self.ln1 = layers.LayerNormalization()
        self.ln2 = layers.LayerNormalization()
        self.dropout = Dropout(dropout)

    def call(self, x, training=None):
        attn_output = self.mha(x, x)
        out1 = self.ln1(x + self.dropout(attn_output, training=training))
        ffn_output = self.ffn(out1)
        return self.ln2(out1 + self.dropout(ffn_output, training=training))


def create_mean_model(time_steps, input_dims):
    """Two-layer CNN-LSTM-Informer mean branch."""
    inputs = Input(shape=(time_steps, input_dims), name="mean_input")

    # 1st 1D convolution layer: kernel size K=7, filters F=64
    x = Conv1D(
        filters=64,
        kernel_size=7,
        padding='same',
        name="cnn_conv1_k7_f64"
    )(inputs)

    # 2nd 1D dilated convolution layer: K=3, F=32, dilation rate d=2
    x = Conv1D(
        filters=32,
        kernel_size=3,
        dilation_rate=2,
        padding='same',
        name="cnn_dilated_conv2_k3_f32_d2"
    )(x)

    x = LSTM(LSTM_UNITS, return_sequences=True, name="damage_lstm")(x)
    x = InformerBlock(d_model=64, num_heads=4, name="informer_block")(x)
    x = Flatten(name="mean_flatten")(x)
    mu = Dense(1, activation='linear', name='mean_output')(x)
    return Model(inputs=inputs, outputs=mu, name="mean_model")


def create_variance_model(time_steps, input_dims):
    """Variance branch outputs log variance, preserving the existing architecture."""
    inputs = Input(shape=(time_steps, input_dims), name="var_input")
    x = Flatten(name="var_flatten")(inputs)
    x = Dense(64, activation='relu', name="var_dense64")(x)
    x = Dense(32, activation='relu', name="var_dense32")(x)
    log_var = Dense(1, activation='linear', name="log_variance")(x)
    return Model(inputs=inputs, outputs=log_var, name="variance_model")


def gaussian_nll_from_outputs(y_true, y_pred):
    """Gaussian NLL for concatenated [mu, log_var]."""
    y_true = tf.cast(y_true, tf.float32)
    mu = y_pred[:, 0:1]
    log_var = tf.clip_by_value(y_pred[:, 1:2], -10.0, 6.0)
    precision = tf.exp(-log_var)
    return tf.reduce_mean(0.5 * (log_var + tf.square(y_true - mu) * precision))


def variance_nll_from_residual(y_residual, log_var_pred):
    """Stage-2 NLL with Stage-1 residual as target."""
    y_residual = tf.cast(y_residual, tf.float32)
    log_var = tf.clip_by_value(log_var_pred, -10.0, 6.0)
    precision = tf.exp(-log_var)
    return tf.reduce_mean(0.5 * (log_var + tf.square(y_residual) * precision))


def build_joint_model(mean_model, variance_model):
    inputs = Input(shape=(TIME_STEPS, INPUT_DIMS), name="joint_input")
    mu = mean_model(inputs)
    log_var = variance_model(inputs)
    outputs = layers.Concatenate(name="mve_output")([mu, log_var])
    return Model(inputs=inputs, outputs=outputs, name="joint_mve_model")



def split_into_five_contiguous_folds(n_rows, n_folds=5):
    all_indices = np.arange(n_rows)
    return [np.asarray(x, dtype=int) for x in np.array_split(all_indices, n_folds)]


def split_train_valcal_inside_fold(fold_idx, proper_train_ratio=0.90):
    
    n = len(fold_idx)
    split_pos = int(np.floor(n * proper_train_ratio))

    
    split_pos = max(TIME_STEPS + 1, split_pos)
    split_pos = min(n - 1, split_pos)

    proper_idx = fold_idx[:split_pos]
    valcal_idx = fold_idx[split_pos:]
    return proper_idx, valcal_idx, split_pos

def make_windows_by_target_positions(
    block_scaled,
    block_global_idx,
    target_positions,
    target_scaler,
    raw_target,
    look_back,
):
    
    X, y, global_targets = [], [], []

    for pos in target_positions:
        if pos < look_back or pos >= len(block_scaled):
            continue
        X.append(block_scaled[pos - look_back:pos, :])
        global_t = int(block_global_idx[pos])
        global_targets.append(global_t)
        y_scaled = target_scaler.transform(
            np.asarray(raw_target[global_t]).reshape(1, 1)
        )[0, 0]
        y.append(y_scaled)

    if not X:
        return (
            np.empty((0, look_back, block_scaled.shape[1]), dtype=np.float32),
            np.empty((0,), dtype=np.float32),
            np.empty((0,), dtype=int),
        )

    return (
        np.asarray(X, dtype=np.float32),
        np.asarray(y, dtype=np.float32),
        np.asarray(global_targets, dtype=int),
    )


def prepare_fold_data(raw_features, raw_target, fold_indices, test_fold_id):
    
    train_fold_ids = [i for i in range(N_FOLDS) if i != test_fold_id]

    proper_train_rows_parts = []
    split_positions = {}

    for fid in train_fold_ids:
        proper_idx, valcal_idx, split_pos = split_train_valcal_inside_fold(
            fold_indices[fid], PROPER_TRAIN_RATIO
        )
        if len(valcal_idx) == 0:
            raise ValueError(
                f"Fold {fid + 1} is too short to reserve a shared validation/calibration segment."
            )
        proper_train_rows_parts.append(proper_idx)
        split_positions[fid] = split_pos

    proper_train_rows = np.concatenate(proper_train_rows_parts)

    
    feature_scaler = MinMaxScaler(feature_range=(0, 1))
    feature_scaler.fit(raw_features[proper_train_rows, :])

    target_scaler = MinMaxScaler(feature_range=(0, 1))
    target_scaler.fit(np.asarray(raw_target[proper_train_rows]).reshape(-1, 1))

    X_train_parts, y_train_parts, train_target_parts = [], [], []
    X_valcal_parts, y_valcal_parts, valcal_target_parts = [], [], []

    for fid in train_fold_ids:
        idx = fold_indices[fid]
        split_pos = split_positions[fid]
        scaled_block = feature_scaler.transform(raw_features[idx, :])

        # Proper-training targets: TIME_STEPS ... split_pos-1.
        train_target_positions = np.arange(TIME_STEPS, split_pos, dtype=int)
        Xp, yp, gp = make_windows_by_target_positions(
            scaled_block,
            idx,
            train_target_positions,
            target_scaler,
            raw_target,
            TIME_STEPS,
        )

        
        valcal_target_positions = np.arange(split_pos, len(idx), dtype=int)
        Xvc, yvc, gvc = make_windows_by_target_positions(
            scaled_block,
            idx,
            valcal_target_positions,
            target_scaler,
            raw_target,
            TIME_STEPS,
        )

        if len(Xp) > 0:
            X_train_parts.append(Xp)
            y_train_parts.append(yp)
            train_target_parts.append(gp)
        if len(Xvc) > 0:
            X_valcal_parts.append(Xvc)
            y_valcal_parts.append(yvc)
            valcal_target_parts.append(gvc)

    if not X_train_parts or not X_valcal_parts:
        raise ValueError("Insufficient proper-training or shared validation/calibration windows.")

    X_train = np.concatenate(X_train_parts, axis=0)
    y_train = np.concatenate(y_train_parts, axis=0)
    train_global_targets = np.concatenate(train_target_parts, axis=0)

    X_valcal = np.concatenate(X_valcal_parts, axis=0)
    y_valcal = np.concatenate(y_valcal_parts, axis=0)
    valcal_global_targets = np.concatenate(valcal_target_parts, axis=0)

   
    y_val = y_valcal
    val_global_targets = valcal_global_targets
    X_cal = X_valcal
    y_cal = y_valcal
    cal_global_targets = valcal_global_targets

  
    test_idx = fold_indices[test_fold_id]
    test_scaled_block = feature_scaler.transform(raw_features[test_idx, :])
    test_target_positions = np.arange(TIME_STEPS, len(test_idx), dtype=int)
    X_test, y_test, test_global_targets = make_windows_by_target_positions(
        test_scaled_block,
        test_idx,
        test_target_positions,
        target_scaler,
        raw_target,
        TIME_STEPS,
    )

    return {
        "X_train": X_train,
        "y_train": y_train,
        "X_val": X_val,
        "y_val": y_val,
        "X_cal": X_cal,
        "y_cal": y_cal,
        "X_test": X_test,
        "y_test": y_test,
        "train_global_targets": train_global_targets,
        "val_global_targets": val_global_targets,
        "cal_global_targets": cal_global_targets,
        "test_global_targets": test_global_targets,
        "feature_scaler": feature_scaler,
        "target_scaler": target_scaler,
        "test_raw_indices": test_idx,
    }

def inverse_target(target_scaler, x):
    x = np.asarray(x).reshape(-1, 1)
    return target_scaler.inverse_transform(x).reshape(-1)



def conformal_multiplier(y_true_scaled, mu_scaled, std_scaled, alpha=0.05):
   
    y_true_scaled = np.asarray(y_true_scaled).reshape(-1)
    mu_scaled = np.asarray(mu_scaled).reshape(-1)
    std_scaled = np.asarray(std_scaled).reshape(-1)

    scores = np.abs(y_true_scaled - mu_scaled) / np.maximum(std_scaled, EPS)
    scores = scores[np.isfinite(scores)]

    if len(scores) == 0:
        raise ValueError("No valid calibration scores were produced.")

    n = len(scores)
    k = int(math.ceil((n + 1) * (1.0 - alpha)))
    k = min(max(k, 1), n)
    q_hat = float(np.sort(scores)[k - 1])

    return q_hat, scores



def point_metrics(y_true, y_pred):
    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
        "R2": r2_score(y_true, y_pred),
    }


def interval_metrics(y_true, lower, upper, nominal_coverage=0.95, eta=30.0):
    
    y_true = np.asarray(y_true).reshape(-1)
    lower = np.asarray(lower).reshape(-1)
    upper = np.asarray(upper).reshape(-1)

    
    picr = np.mean((y_true >= lower) & (y_true <= upper))

  
    piaw = np.mean(upper - lower)

  
    data_range = np.max(y_true) - np.min(y_true)

    if data_range > 0:
        npiaw = piaw / data_range
        cwc = (1.0 - npiaw) * np.exp(
            -eta * (picr - nominal_coverage) ** 2
        )
    else:
        npiaw = np.nan
        cwc = np.nan

    return {
        "PICR": picr,
        "PIAW": piaw,
        "NPIAW": npiaw,
        "CWC": cwc,
    }



def save_three_stage_loss(
    history_mean, val_history_mean,
    history_var, val_history_var,
    history_joint, val_history_joint,
    fold_id, seed
):
    
    fig = plt.figure(figsize=(10, 6))

    x1 = np.arange(1, len(history_mean) + 1)
    x2 = np.arange(len(history_mean) + 1, len(history_mean) + len(history_var) + 1)
    x3 = np.arange(
        len(history_mean) + len(history_var) + 1,
        len(history_mean) + len(history_var) + len(history_joint) + 1,
    )

    plt.plot(x1, history_mean, label="Stage 1 Train MSE")
    plt.plot(x1, val_history_mean, linestyle="--", label="Stage 1 Validation MSE")
    plt.plot(x2, history_var, label="Stage 2 Train Variance NLL")
    plt.plot(x2, val_history_var, linestyle="--", label="Stage 2 Validation Variance NLL")
    plt.plot(x3, history_joint, label="Stage 3 Train Joint MVE NLL")
    plt.plot(x3, val_history_joint, linestyle="--", label="Stage 3 Validation Joint MVE NLL")
    plt.axvline(len(history_mean), linestyle="--", linewidth=1)
    plt.axvline(len(history_mean) + len(history_var), linestyle="--", linewidth=1)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"Three-stage train/validation loss - Fold {fold_id + 1}, Seed {seed}")
    plt.legend()
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / f"loss_train_validation_fold{fold_id+1}_seed{seed}.png", dpi=300)
    plt.close(fig)

    stage = (
        ["mean"] * len(history_mean) +
        ["variance"] * len(history_var) +
        ["joint"] * len(history_joint)
    )
    stage_epoch = (
        list(range(1, len(history_mean) + 1)) +
        list(range(1, len(history_var) + 1)) +
        list(range(1, len(history_joint) + 1))
    )
    train_loss = history_mean + history_var + history_joint
    val_loss = val_history_mean + val_history_var + val_history_joint

    pd.DataFrame({
        "stage": stage,
        "stage_epoch": stage_epoch,
        "train_loss": train_loss,
        "validation_loss": val_loss,
    }).to_csv(
        OUTPUT_DIR / f"loss_train_validation_fold{fold_id+1}_seed{seed}.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # A second, validation-only file is convenient for plotting in Origin/Excel.
    pd.DataFrame({
        "stage": stage,
        "stage_epoch": stage_epoch,
        "validation_loss": val_loss,
    }).to_csv(
        OUTPUT_DIR / f"validation_loss_only_fold{fold_id+1}_seed{seed}.csv",
        index=False,
        encoding="utf-8-sig",
    )


def save_prediction_plot(
    x_index,
    y_true,
    y_pred,
    cal_lower,
    cal_upper,
    fold_id,
    seed,
    q_hat,
):
    order = np.argsort(x_index)
    x = np.asarray(x_index)[order]
    yt = np.asarray(y_true)[order]
    yp = np.asarray(y_pred)[order]
    clo = np.asarray(cal_lower)[order]
    cup = np.asarray(cal_upper)[order]

    fig = plt.figure(figsize=(12, 6))
    plt.plot(x, yt, label="Observed")
    plt.plot(x, yp, label="Predicted mean")
    plt.fill_between(x, clo, cup, alpha=0.2, label=f"Conformal 95% PI (q={q_hat:.3f})")
    plt.xlabel("Original data index")
    plt.ylabel("Target value")
    plt.title(f"Conformal-calibrated test interval - Fold {fold_id + 1}, Seed {seed}")
    plt.legend()
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / f"prediction_fold{fold_id+1}_seed{seed}.png", dpi=300)
    plt.close(fig)

def train_one_run(raw_features, raw_target, fold_indices, fold_id, seed):
    print("\n" + "=" * 90)
    print(f"Fold {fold_id + 1}/{N_FOLDS} | Seed {seed}")
    print("=" * 90)

    tf.keras.backend.clear_session()
    gc.collect()
    set_global_seed(seed)

    fold_data = prepare_fold_data(raw_features, raw_target, fold_indices, fold_id)
    X_train = fold_data["X_train"]
    y_train = fold_data["y_train"]
    X_val = fold_data["X_val"]
    y_val = fold_data["y_val"]
    X_cal = fold_data["X_cal"]
    y_cal = fold_data["y_cal"]
    X_test = fold_data["X_test"]
    y_test = fold_data["y_test"]
    target_scaler = fold_data["target_scaler"]

    if min(len(X_train), len(X_val), len(X_cal), len(X_test)) == 0:
        raise ValueError(
            f"Fold {fold_id + 1} has insufficient train/calibration/test samples."
        )

    print(
        f"Samples | proper train={len(X_train)}, shared VAL/CAL={len(X_val)}, test={len(X_test)}"
    )

    # -------------------- Stage 1: mean learning --------------------
    mean_model = create_mean_model(TIME_STEPS, INPUT_DIMS)
    mean_model.compile(
        optimizer=tf.keras.optimizers.Adam(LEARNING_RATE_MEAN),
        loss="mse",
    )
    print("Stage 1/3: mean learning...")
    h1 = mean_model.fit(
        X_train,
        y_train,
        epochs=EPOCHS_MEAN,
        batch_size=BATCH_SIZE,
        shuffle=True,
        verbose=0,
        validation_data=(X_val, y_val),
    )

    train_mu_stage1 = mean_model.predict(
        X_train, batch_size=BATCH_SIZE, verbose=0
    ).reshape(-1)
    train_residual = y_train - train_mu_stage1
    val_mu_stage1 = mean_model.predict(
        X_val, batch_size=BATCH_SIZE, verbose=0
    ).reshape(-1)
    val_residual = y_val - val_mu_stage1

    # -------------------- Stage 2: variance learning --------------------
    variance_model = create_variance_model(TIME_STEPS, INPUT_DIMS)

    # Initialize final variance bias around Stage-1 residual variance.
    residual_var = max(float(np.var(train_residual)), EPS)
    final_var_layer = variance_model.get_layer("log_variance")
    w, b = final_var_layer.get_weights()
    w[:] = 0.0
    b[:] = np.log(residual_var)
    final_var_layer.set_weights([w, b])

    variance_model.compile(
        optimizer=tf.keras.optimizers.Adam(LEARNING_RATE_VAR),
        loss=variance_nll_from_residual,
    )
    print("Stage 2/3: variance learning...")
    h2 = variance_model.fit(
        X_train,
        train_residual.reshape(-1, 1),
        epochs=EPOCHS_VAR,
        batch_size=BATCH_SIZE,
        shuffle=True,
        verbose=0,
        validation_data=(X_val, val_residual.reshape(-1, 1)),
    )

    # -------------------- Stage 3: joint MVE optimization --------------------
    mean_model.trainable = True
    variance_model.trainable = True
    joint_model = build_joint_model(mean_model, variance_model)
    joint_model.compile(
        optimizer=tf.keras.optimizers.Adam(LEARNING_RATE_JOINT),
        loss=gaussian_nll_from_outputs,
    )
    print("Stage 3/3: joint MVE optimization...")
    h3 = joint_model.fit(
        X_train,
        y_train.reshape(-1, 1),
        epochs=EPOCHS_JOINT,
        batch_size=BATCH_SIZE,
        shuffle=True,
        verbose=0,
        validation_data=(X_val, y_val.reshape(-1, 1)),
    )

    # -------------------- Inference on shared VAL/CAL and test --------------------
    # No validation metric other than per-epoch val_loss is computed for model selection.
    # The shared held-out segment is evaluated here only because it is also the conformal
    # calibration set required to estimate q_hat.
    valcal_out = joint_model.predict(X_val, batch_size=BATCH_SIZE, verbose=0)
    test_out = joint_model.predict(X_test, batch_size=BATCH_SIZE, verbose=0)

    cal_mu = valcal_out[:, 0]
    test_mu = test_out[:, 0]

    cal_log_var = np.clip(valcal_out[:, 1], -10.0, 6.0)
    test_log_var = np.clip(test_out[:, 1], -10.0, 6.0)

    cal_std = np.sqrt(np.exp(cal_log_var))
    test_std = np.sqrt(np.exp(test_log_var))

    # -------------------- Split conformal calibration --------------------
    q_hat, cal_scores = conformal_multiplier(
        y_cal,
        cal_mu,
        cal_std,
        alpha=ALPHA,
    )

    print(
        f"Shared VAL/CAL | N={len(cal_scores)}, q_hat(95%)={q_hat:.6f}"
    )

    # Only the final conformal-calibrated test interval is generated.
    test_conf_lower = test_mu - q_hat * test_std
    test_conf_upper = test_mu + q_hat * test_std

    # -------------------- Inverse transform --------------------
    y_test_org = inverse_target(target_scaler, y_test)
    test_mu_org = inverse_target(target_scaler, test_mu)
    test_conf_lower_org = inverse_target(target_scaler, test_conf_lower)
    test_conf_upper_org = inverse_target(target_scaler, test_conf_upper)

    # -------------------- Final test metrics --------------------
    test_pm = point_metrics(y_test_org, test_mu_org)
    test_conf_im = interval_metrics(
        y_test_org, test_conf_lower_org, test_conf_upper_org
    )

    print(
        f"Test point | RMSE={test_pm['RMSE']:.6f}, MAE={test_pm['MAE']:.6f}, "
        f"R2={test_pm['R2']:.6f}"
    )
    print(
        f"Test conformal PI | PICR={test_conf_im['PICR']:.4f}, "
        f"PIAW={test_conf_im['PIAW']:.6f}, "
        f"NPIAW={test_conf_im['NPIAW']:.6f}, CWC={test_conf_im['CWC']:.6f}"
    )

    # -------------------- Save run-level outputs --------------------
    save_three_stage_loss(
        list(map(float, h1.history["loss"])),
        list(map(float, h1.history["val_loss"])),
        list(map(float, h2.history["loss"])),
        list(map(float, h2.history["val_loss"])),
        list(map(float, h3.history["loss"])),
        list(map(float, h3.history["val_loss"])),
        fold_id,
        seed,
    )

    save_prediction_plot(
        fold_data["test_global_targets"],
        y_test_org,
        test_mu_org,
        test_conf_lower_org,
        test_conf_upper_org,
        fold_id,
        seed,
        q_hat,
    )


    pd.DataFrame({
        "Fold": fold_id + 1,
        "Seed": seed,
        "Original_Index": fold_data["test_global_targets"],
        "True": y_test_org,
        "Predicted_Mean": test_mu_org,
        "Conformal_Lower_95": test_conf_lower_org,
        "Conformal_Upper_95": test_conf_upper_org,
        "Conformal_Q": q_hat,
    }).to_csv(
        OUTPUT_DIR / f"test_predictions_fold{fold_id+1}_seed{seed}.csv",
        index=False,
        encoding="utf-8-sig",
    )

    row = {
        "Fold": fold_id + 1,
        "Seed": seed,
        "Proper_Train_N": len(X_train),
        "VAL_CAL_N": len(X_val),
        "Test_N": len(X_test),
        "Conformal_Q": q_hat,

        "Test_MAE": test_pm["MAE"],
        "Test_RMSE": test_pm["RMSE"],
        "Test_R2": test_pm["R2"],
        "Test_PICR": test_conf_im["PICR"],
        "Test_PIAW": test_conf_im["PIAW"],
        "Test_NPIAW": test_conf_im["NPIAW"],
        "Test_CWC": test_conf_im["CWC"],

        "Stage1_Final_MSE": float(h1.history["loss"][-1]),
        "Stage1_Final_Val_MSE": float(h1.history["val_loss"][-1]),
        "Stage2_Final_NLL": float(h2.history["loss"][-1]),
        "Stage2_Final_Val_NLL": float(h2.history["val_loss"][-1]),
        "Stage3_Final_NLL": float(h3.history["loss"][-1]),
        "Stage3_Final_Val_NLL": float(h3.history["val_loss"][-1]),
    }

    del mean_model, variance_model, joint_model
    tf.keras.backend.clear_session()
    gc.collect()
    return row


# ============================================================
# 9. Summary helpers
# ============================================================
def mean_std_text(series, digits=6):
    return f"{series.mean():.{digits}f} ± {series.std(ddof=1):.{digits}f}"


def save_summaries(results_df):
    results_df.to_csv(
        OUTPUT_DIR / "all_25_runs_metrics.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # Only point-prediction metrics and final conformal-calibrated interval metrics.
    metric_cols = [
        "Test_MAE",
        "Test_RMSE",
        "Test_R2",
        "Test_PICR",
        "Test_PIAW",
        "Test_NPIAW",
        "Test_CWC",
        "Conformal_Q",
    ]

    # Each fold summarized across five seeds: random-seed stability within the same test region.
    fold_rows = []
    for fold_id, g in results_df.groupby("Fold"):
        row = {"Fold": int(fold_id)}
        for c in metric_cols:
            row[c + "_Mean"] = g[c].mean()
            row[c + "_Std"] = g[c].std(ddof=1)
            row[c + "_Mean±Std"] = mean_std_text(g[c])
        fold_rows.append(row)
    fold_summary = pd.DataFrame(fold_rows)
    fold_summary.to_csv(
        OUTPUT_DIR / "summary_by_fold_across_5seeds.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # Each seed summarized across five test folds: variation across contiguous regions.
    seed_rows = []
    for seed, g in results_df.groupby("Seed"):
        row = {"Seed": int(seed)}
        for c in metric_cols:
            row[c + "_Mean"] = g[c].mean()
            row[c + "_Std"] = g[c].std(ddof=1)
            row[c + "_Mean±Std"] = mean_std_text(g[c])
        seed_rows.append(row)
    seed_summary = pd.DataFrame(seed_rows)
    seed_summary.to_csv(
        OUTPUT_DIR / "summary_by_seed_across_5folds.csv",
        index=False,
        encoding="utf-8-sig",
    )

    overall = {}
    for c in metric_cols:
        overall[c + "_Mean"] = results_df[c].mean()
        overall[c + "_Std"] = results_df[c].std(ddof=1)
        overall[c + "_Mean±Std"] = mean_std_text(results_df[c])
    overall_df = pd.DataFrame([overall])
    overall_df.to_csv(
        OUTPUT_DIR / "overall_25run_mean_std.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # Paper-friendly compact table: only final calibrated interval results.
    paper_table = pd.DataFrame({
        "Metric": [
            "RMSE",
            "MAE",
            "R2",
            "PICR_Conformal",
            "PIAW_Conformal",
            "NPIAW_Conformal",
            "CWC_Conformal",
            "Conformal_Q",
        ],
        "25-run Mean±Std": [
            mean_std_text(results_df["Test_RMSE"]),
            mean_std_text(results_df["Test_MAE"]),
            mean_std_text(results_df["Test_R2"]),
            mean_std_text(results_df["Test_PICR"]),
            mean_std_text(results_df["Test_PIAW"]),
            mean_std_text(results_df["Test_NPIAW"]),
            mean_std_text(results_df["Test_CWC"]),
            mean_std_text(results_df["Conformal_Q"]),
        ],
    })
    paper_table.to_csv(
        OUTPUT_DIR / "paper_summary_table.csv",
        index=False,
        encoding="utf-8-sig",
    )

    return fold_summary, seed_summary, overall_df


# ============================================================
# 10. Main
# ============================================================
def main():
    print("Loading data:", DATA_PATH)
    data = pd.read_csv(DATA_PATH)

    required_columns = FEATURE_COLUMNS + [TARGET_COLUMN]
    missing = [c for c in required_columns if c not in data.columns]
    if missing:
        raise KeyError(f"Missing required columns: {missing}")

    # Strict separation between predictors and target prevents I-stage target leakage.
    raw_features = data.loc[:, FEATURE_COLUMNS].astype(np.float32).values
    raw_target = data.loc[:, TARGET_COLUMN].astype(np.float32).values

    if np.isnan(raw_features).any() or np.isnan(raw_target).any():
        raise ValueError("NaN detected in selected input/target columns. Please clean data before training.")

    if raw_features.shape[1] != 8:
        raise ValueError(f"Expected 8 input features A-H, got {raw_features.shape[1]}.")

    fold_indices = split_into_five_contiguous_folds(len(raw_features), N_FOLDS)

    fold_definition_rows = []
    print("\nFive contiguous folds (raw rows):")
    for i, idx in enumerate(fold_indices):
        print(
            f"Fold {i+1}: rows {idx[0]}-{idx[-1]} | N={len(idx)} | "
            f"ratio={len(idx)/len(raw_features):.2%}"
        )
        fold_definition_rows.append({
            "Fold": i + 1,
            "Start_Row": int(idx[0]),
            "End_Row": int(idx[-1]),
            "Raw_N": int(len(idx)),
            "Raw_Ratio": len(idx) / len(raw_features),
        })

    pd.DataFrame(fold_definition_rows).to_csv(
        OUTPUT_DIR / "five_fold_definition.csv",
        index=False,
        encoding="utf-8-sig",
    )

    config_df = pd.DataFrame([{
        "N_Folds": N_FOLDS,
        "Seeds": ",".join(map(str, SEEDS)),
        "Time_Steps": TIME_STEPS,
        "Input_Features": ",".join(FEATURE_COLUMNS),
        "Target_Column": TARGET_COLUMN,
        "Target_In_Input": False,
        "Proper_Train_Ratio_Within_NonTest_Fold": PROPER_TRAIN_RATIO,
        "Shared_Validation_Calibration_Ratio": VAL_CAL_RATIO,
        "Validation_and_Calibration_Overlap": True,
        "Nominal_Coverage": 1.0 - ALPHA,
        "CWC_Eta": 30.0,
        "CWC_Definition": "(1-NPIAW)*exp[-eta*(PICR-mu)^2]",
        "Epochs_Mean": EPOCHS_MEAN,
        "Epochs_Variance": EPOCHS_VAR,
        "Epochs_Joint": EPOCHS_JOINT,
        "LR_Mean": LEARNING_RATE_MEAN,
        "LR_Variance": LEARNING_RATE_VAR,
        "LR_Joint": LEARNING_RATE_JOINT,
    }])
    config_df.to_csv(
        OUTPUT_DIR / "experiment_configuration.csv",
        index=False,
        encoding="utf-8-sig",
    )

    all_results = []
    for fold_id in range(N_FOLDS):
        for seed in SEEDS:
            result = train_one_run(raw_features, raw_target, fold_indices, fold_id, seed)
            all_results.append(result)

            # Incremental checkpoint: completed runs survive interruption.
            pd.DataFrame(all_results).to_csv(
                OUTPUT_DIR / "all_25_runs_metrics_partial.csv",
                index=False,
                encoding="utf-8-sig",
            )

    results_df = pd.DataFrame(all_results)
    save_summaries(results_df)

    print("\n" + "=" * 90)
    print("ALL EXPERIMENTS FINISHED")
    print("=" * 90)

    print("\nOverall 25-run point-prediction metrics:")
    for c in ["Test_RMSE", "Test_MAE", "Test_R2"]:
        print(f"{c}: {mean_std_text(results_df[c])}")


    print("\nOverall 25-run conformal-calibrated interval metrics:")
    for c in ["Test_PICR", "Test_PIAW", "Test_NPIAW", "Test_CWC", "Conformal_Q"]:
        print(f"{c}: {mean_std_text(results_df[c])}")

    print(f"\nAll outputs saved to: {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
