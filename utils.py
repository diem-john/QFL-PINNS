import pandas as pd
import numpy as np
import torch
from torch import Tensor
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from haversine import haversine, Unit
from tqdm import tqdm


# --- Configuration and Constants ---
class Config:
    SEQUENCE_SIZE = 48
    TARGET_WINDOW = 24
    TYPHOON_STEPS_IN = 12
    PROXIMITY_THRESHOLD_KM = 600

    WEATHER_FEATURES = ['air_pressure', 'temperature', 'relative_humidity', 'wind_speed',
                        'wind_direction', 'gust_max', 'gust_max_dir', 'precipitation', 'solar_rad']
    TARGET_FEATURE = ['wind_speed']
    TYPHOON_FEATURES_CORE = ['lat', 'lng', 'wind', 'long50']
    TYPHOON_FEATURES_EXO_PRED = ['typhoon_lat', 'typhoon_lng', 'wind', 'long50', 'distance_km']

    PINN_ALL_FEATURES = WEATHER_FEATURES + TYPHOON_FEATURES_EXO_PRED

    # DEVICE removed: now determined by CLI argument
    SEQ2SEQ_BATCH_SIZE = 64
    PINN_BATCH_SIZE = 16
    PINN_EPOCHS = 10
    SEQ2SEQ_EPOCHS = 10


# ... (Utility Functions remain the same, using the imported 'tqdm')
def calculate_distance(row: pd.Series) -> float:
    if 'lat' in row and 'lng' in row:
        typhoon_loc = (row['lat'], row['lng'])
    elif 'typhoon_lat' in row and 'typhoon_lng' in row:
        typhoon_loc = (row['typhoon_lat'], row['typhoon_lng'])
    else:
        return np.nan

    if 'station_latitude' in row and 'station_longitude' in row:
        station_loc = (row['station_latitude'], row['station_longitude'])
    elif 'latitude' in row and 'longitude' in row:
        station_loc = (row['latitude'], row['longitude'])
    else:
        return np.nan

    if pd.isnull(typhoon_loc[0]) or pd.isnull(station_loc[0]):
        return np.nan

    try:
        return haversine(typhoon_loc, station_loc, unit=Unit.KILOMETERS)
    except ValueError:
        corrected_longitude = typhoon_loc[1]
        if corrected_longitude > 180:
            corrected_longitude = 180
        elif corrected_longitude < -180:
            corrected_longitude = -180
        corrected_typhoon_loc = (typhoon_loc[0], corrected_longitude)
        return haversine(corrected_typhoon_loc, station_loc, unit=Unit.KILOMETERS)


def create_sequences_typhoon(data_values: np.ndarray, n_steps_in: int, n_steps_out: int) -> tuple[
    np.ndarray, np.ndarray]:
    """Prepares sequences for Seq2Seq training (input-output sequence pairs)."""
    X, y = [], []
    for i in range(len(data_values)):
        end_ix = i + n_steps_in
        out_end_ix = end_ix + n_steps_out
        if out_end_ix > len(data_values):
            break
        seq_x = data_values[i:end_ix]
        seq_y = data_values[end_ix:out_end_ix]
        X.append(seq_x)
        y.append(seq_y)
    return np.array(X), np.array(y)


def create_sequence_weather(sequence_length: int, target_window: int, scaled_data_df: pd.DataFrame) -> tuple[
    Tensor, Tensor]:
    """Prepares sequences for PINN training (input features and target wind speed)."""
    target_column_name = 'wind_speed'
    target_column_index = scaled_data_df.columns.get_loc(target_column_name)
    feature_indices = [scaled_data_df.columns.get_loc(col) for col in scaled_data_df.columns if col != 'time']

    x = []
    y = []

    for i in tqdm(range(len(scaled_data_df) - sequence_length - target_window + 1), desc="Creating Weather Sequences"):
        sequence = scaled_data_df.iloc[i:i + sequence_length, feature_indices].values
        target = scaled_data_df.iloc[i + sequence_length:i + sequence_length + target_window,
                 target_column_index].values
        x.append(sequence)
        y.append(target)

    return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)


def physics_loss_fn(u: Tensor, v: Tensor, p: Tensor, x: Tensor) -> Tensor:
    """Placeholder for the Physics-Informed Loss Function."""
    return torch.tensor(0.0, dtype=torch.float32).to(u.device)


def calculate_metrics(y_true_scaled: np.ndarray, y_pred_scaled: np.ndarray, scaler_target: MinMaxScaler) -> dict:
    """Calculates R2, RMSE, MSE, and MAE after inverse transforming scaled predictions."""
    y_true_flat_scaled = y_true_scaled.flatten().reshape(-1, 1)
    y_pred_flat_scaled = y_pred_scaled.flatten().reshape(-1, 1)

    y_true = scaler_target.inverse_transform(y_true_flat_scaled)
    y_pred = scaler_target.inverse_transform(y_pred_flat_scaled)

    r2 = r2_score(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)

    return {
        "R2": float(r2),
        "RMSE": float(rmse),
        "MSE": float(mse),
        "MAE": float(mae)
    }