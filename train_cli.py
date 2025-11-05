# train_cli.py

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from torch.optim.lr_scheduler import ReduceLROnPlateau
import os
import sys
import argparse
import datetime
from tqdm import tqdm  # <- ADDED: TQDM import

# Import components from other files
from models import Seq2Seq, HybridModelPINN
from utils import Config, calculate_distance, create_sequences_typhoon, \
    create_sequence_weather, physics_loss_fn, calculate_metrics


class E2EPipeline:
    def __init__(self, station_name: str, config: Config):
        self.station_name = station_name
        self.config = config
        self.device = config.DEVICE
        self.seq2seq_config = {
            "input_size": len(config.TYPHOON_FEATURES_CORE),
            "hidden_size": 128,
            "num_layers": 2,
            "output_size": len(config.TYPHOON_FEATURES_CORE),
            "num_heads": 4
        }
        self.pinn_config = {
            "input_size": len(config.PINN_ALL_FEATURES),
            "output_size": config.TARGET_WINDOW,
            "sequence_length": config.SEQUENCE_SIZE,
            "hidden_dense": 512
        }

        # <- UPDATED: Ensure all necessary directories exist
        self._setup_directories(station_name)

    def _setup_directories(self, station_name: str):
        """Creates model and metadata directories if they do not exist."""
        print("Setting up model directories...")
        # Directory for typhoon model
        os.makedirs('models/typhoon', exist_ok=True)
        # Directory for station-specific PINN models
        os.makedirs('models/weather_sta', exist_ok=True)
        # Directory for station-specific metadata (metrics)
        os.makedirs('models/weather_sta/metadata', exist_ok=True)
        print("Directories verified/created.")

    def _load_typhoon_data(self) -> tuple[pd.DataFrame, MinMaxScaler, pd.DataFrame]:
        # ... (function body remains the same)
        print("   -> Loading and preparing historical typhoon data...")
        typhoon_data_hist_raw = pd.read_csv('data/typhoon_data.csv', parse_dates=['Date'], infer_datetime_format=True)
        typhoon_data_hist_raw.rename(columns={'Date': 'time'}, inplace=True)

        typhoon_data_hist_filtered = typhoon_data_hist_raw[(typhoon_data_hist_raw['time'].dt.year >= 2015)].copy()
        typhoon_data_hist_filtered = typhoon_data_hist_filtered[self.config.TYPHOON_FEATURES_CORE + ['time']].copy()
        typhoon_data_hist_filtered.sort_values(by='time', inplace=True)
        typhoon_data_hist_filtered.dropna(inplace=True)

        scaler_typhoon = MinMaxScaler(feature_range=(0, 1))
        typhoon_data_scaled_for_training = typhoon_data_hist_filtered.copy()
        typhoon_data_scaled_for_training[self.config.TYPHOON_FEATURES_CORE] = \
            scaler_typhoon.fit_transform(typhoon_data_hist_filtered[self.config.TYPHOON_FEATURES_CORE])

        return typhoon_data_scaled_for_training, scaler_typhoon, typhoon_data_hist_raw

    def _train_and_forecast_typhoon_model(self, typhoon_df: pd.DataFrame, scaler_typhoon: MinMaxScaler) -> pd.DataFrame:
        """Trains the Seq2Seq model and generates a long-term forecast dataset."""
        print(f"\n--- Typhoon Model Training ({Config.TYPHOON_STEPS_IN} input to {Config.TARGET_WINDOW} output) ---")

        # 1. Prepare sequences (same as before)
        X_typhoon, y_typhoon = create_sequences_typhoon(
            typhoon_df[self.config.TYPHOON_FEATURES_CORE].values,
            self.config.TYPHOON_STEPS_IN,
            self.config.TARGET_WINDOW
        )

        # 2. Train/Validation/Test Split (same as before)
        split_idx = int(0.7 * len(X_typhoon))
        X_temp = X_typhoon[split_idx:]
        val_split_idx = int(0.5 * len(X_temp))
        X_test_tensor = torch.from_numpy(X_temp[val_split_idx:]).float()

        X_train_tensor = torch.from_numpy(X_typhoon[:split_idx]).float()
        y_train_tensor = torch.from_numpy(y_typhoon[:split_idx]).float()
        X_valid_tensor = torch.from_numpy(X_temp[:val_split_idx]).float()
        y_valid_tensor = torch.from_numpy(y_temp[:val_split_idx]).float()

        train_loader = DataLoader(TensorDataset(X_train_tensor, y_train_tensor),
                                  batch_size=self.config.SEQ2SEQ_BATCH_SIZE, shuffle=True)
        valid_loader = DataLoader(TensorDataset(X_valid_tensor, y_valid_tensor),
                                  batch_size=self.config.SEQ2SEQ_BATCH_SIZE, shuffle=False)
        test_loader = DataLoader(TensorDataset(X_test_tensor), batch_size=self.config.SEQ2SEQ_BATCH_SIZE, shuffle=False)

        # 3. Model Training
        typhoon_model = Seq2Seq(**self.seq2seq_config).to(self.device)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(typhoon_model.parameters(), lr=0.001)

        min_val_loss = float('inf')
        early_stop_count = 0

        # <- ADDED: TQDM progress bar for epochs
        for epoch in tqdm(range(self.config.SEQ2SEQ_EPOCHS), desc="Typhoon Model Training"):
            typhoon_model.train()
            # <- ADDED: TQDM progress bar for training batches
            for batch_x, batch_y in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{self.config.SEQ2SEQ_EPOCHS} (Train)",
                                         leave=False):
                optimizer.zero_grad()
                output = typhoon_model(batch_x.to(self.device), self.config.TARGET_WINDOW)
                loss = criterion(output, batch_y.to(self.device))
                loss.backward()
                optimizer.step()

            typhoon_model.eval()
            val_losses = []
            with torch.no_grad():
                # <- ADDED: TQDM progress bar for validation batches
                for batch in tqdm(valid_loader, desc=f"Epoch {epoch + 1}/{self.config.SEQ2SEQ_EPOCHS} (Validation)",
                                  leave=False):
                    x_batch, y_batch = batch
                    outputs = typhoon_model(x_batch.to(self.device), self.config.TARGET_WINDOW)
                    loss = criterion(outputs, y_batch.to(self.device))
                    val_losses.append(loss.item())

            val_loss = np.mean(val_losses)

            if val_loss < min_val_loss:
                min_val_loss = val_loss
                early_stop_count = 0
            else:
                early_stop_count += 1

            if early_stop_count >= 10:
                print(f"[LOG] Early stopping at epoch {epoch + 1}")
                break

        # 4. Save the model (same as before)
        model_path = 'models/typhoon/typhoon_model.pth'
        torch.save(typhoon_model.state_dict(), model_path)
        print(f"   -> Saved model state to '{model_path}'")

        # 5. Generate forecast for all test sequences (same as before)
        typhoon_model.eval()
        all_predictions = []
        with torch.no_grad():
            for batch_x_test in test_loader:
                output = typhoon_model(batch_x_test[0].to(self.device), self.config.TARGET_WINDOW).cpu().numpy()
                all_predictions.append(output)

        raw_predictions = np.concatenate(all_predictions, axis=0)
        predicted_physical_output = scaler_typhoon.inverse_transform(
            raw_predictions.reshape(-1, len(self.config.TYPHOON_FEATURES_CORE))
        ).reshape(raw_predictions.shape)

        # 6. Create the full exogenous data (same as before)
        full_exogenous_df_list = []
        test_sequences_start_indices = np.arange(split_idx,
                                                 len(typhoon_df) - self.config.TYPHOON_STEPS_IN - self.config.TARGET_WINDOW + 1)

        for i in range(raw_predictions.shape[0]):
            current_sequence_start_index = test_sequences_start_indices[i]
            df_input = typhoon_df.iloc[
                       current_sequence_start_index: current_sequence_start_index + self.config.TYPHOON_STEPS_IN].reset_index(
                drop=True)
            input_end_time = typhoon_df['time'].iloc[current_sequence_start_index + self.config.TYPHOON_STEPS_IN - 1]
            forecast_times = pd.date_range(start=input_end_time + pd.Timedelta(hours=1),
                                           periods=self.config.TARGET_WINDOW, freq='H')
            df_pred = pd.DataFrame(predicted_physical_output[i], columns=self.config.TYPHOON_FEATURES_CORE)
            df_pred['time'] = forecast_times
            df_full_seq = pd.concat([df_input, df_pred], ignore_index=True)
            df_full_seq['sequence_id'] = i
            full_exogenous_df_list.append(df_full_seq)

        print("✅ Typhoon Model Training & Forecasting Complete.")
        return pd.concat(full_exogenous_df_list, ignore_index=True)

    def _load_and_preprocess_station_data(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        # ... (function body remains the same)
        # Load and preprocess data (omitted for brevity, assume the logic is unchanged)

        station_coords = pd.read_csv('data/weather_station_coords.csv')
        station_coords.rename(columns={'lat': 'station_latitude',
                                       'long': 'station_longitude',
                                       'location': 'station'}, inplace=True)

        weather_df = pd.read_csv(f'data/indiv_weather_station/{self.station_name}.csv').fillna(0)
        time_col = [col for col in weather_df.columns if 'time' in col.lower()][0]
        weather_df['time'] = pd.to_datetime(weather_df[time_col], format='mixed', dayfirst=True)
        weather_df.columns = [f'time_{self.station_name}', 'air_pressure', 'temperature',
                              'relative_humidity', 'wind_speed', 'wind_direction',
                              'gust_max', 'gust_max_dir', 'precipitation',
                              'solar_rad', 'time']

        weather_df.drop(columns=[f'time_{self.station_name}'], inplace=True)
        weather_df = weather_df[(weather_df['time'].dt.year >= 2015)].copy()
        weather_df['station'] = self.station_name

        weather_df = pd.merge(weather_df,
                              station_coords[['station', 'station_latitude', 'station_longitude']],
                              on='station', how='left')

        return weather_df.tail(10000).copy(), station_coords

    def _prepare_pinn_exo_data(self, weather_df: pd.DataFrame, typhoon_data_hist_raw: pd.DataFrame) -> tuple[
        pd.DataFrame, int]:
        # ... (function body remains the same)
        # Merge, calculate distance, apply filter (omitted for brevity, assume the logic is unchanged)
        weather_exo_merge = pd.merge_asof(
            weather_df.sort_values('time'),
            typhoon_data_hist_raw[['time'] + self.config.TYPHOON_FEATURES_CORE].sort_values('time'),
            on='time',
            direction='nearest'
        )

        weather_exo_merge.rename(columns={'lat': 'typhoon_lat', 'lng': 'typhoon_lng'}, inplace=True)
        weather_exo_merge['distance_km'] = weather_exo_merge.apply(calculate_distance, axis=1)
        weather_exo_merge['is_affected'] = (
                    weather_exo_merge['distance_km'] <= self.config.PROXIMITY_THRESHOLD_KM).astype(int)
        affected_count = weather_exo_merge['is_affected'].sum()

        final_pinn_df = weather_exo_merge[
            self.config.WEATHER_FEATURES + self.config.TYPHOON_FEATURES_EXO_PRED + ['time']].copy()
        final_pinn_df.loc[weather_exo_merge['is_affected'] == 0, self.config.TYPHOON_FEATURES_EXO_PRED] = 0

        return final_pinn_df, affected_count

    def _train_pinn_model(self, final_pinn_df: pd.DataFrame, affected_count: int):
        """Scales data, creates sequences, trains the Hybrid PINN model, and saves metrics."""
        print(f"\n--- Hybrid PINN Model Training for {self.station_name} ---")

        # Scaling and Sequence Creation (uses TQDM via utils.py)
        scaler_features = MinMaxScaler()
        scaler_target = MinMaxScaler()

        final_pinn_df[self.config.PINN_ALL_FEATURES] = scaler_features.fit_transform(
            final_pinn_df[self.config.PINN_ALL_FEATURES])
        final_pinn_df[self.config.TARGET_FEATURE] = scaler_target.fit_transform(
            final_pinn_df[self.config.TARGET_FEATURE])

        X_pinn, y_pinn = create_sequence_weather(self.config.SEQUENCE_SIZE, self.config.TARGET_WINDOW, final_pinn_df)

        # Data Splitting (same as before)
        X_train, X_temp, y_train, y_temp = train_test_split(X_pinn, y_pinn, test_size=0.2, random_state=42)
        X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

        train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=self.config.PINN_BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=self.config.PINN_BATCH_SIZE, shuffle=False)

        # Model Training Setup (same as before)
        model_pinn = HybridModelPINN(**self.pinn_config).to(self.device)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model_pinn.parameters(), lr=0.001)
        scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=3)

        min_val_loss = float('inf')
        early_stop_count = 0

        # Training Loop
        # <- ADDED: TQDM progress bar for epochs
        for epoch in tqdm(range(self.config.PINN_EPOCHS), desc="PINN Model Training"):
            model_pinn.train()
            train_loss = []
            # <- ADDED: TQDM progress bar for training batches
            for x_batch, y_batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{self.config.PINN_EPOCHS} (Train)",
                                         leave=False):
                x_batch, y_batch = x_batch.to(self.device), y_batch.to(self.device)
                optimizer.zero_grad()

                outputs, u_pred, v_pred, p_pred = model_pinn(x_batch)

                data_loss = criterion(outputs, y_batch)
                physics_loss = physics_loss_fn(u_pred, v_pred, p_pred, x_batch)
                loss = 0.73 * data_loss + 0.27 * physics_loss

                train_loss.append(loss.item())
                loss.backward()
                optimizer.step()

            # Validation
            model_pinn.eval()
            val_losses = []
            with torch.no_grad():
                # <- ADDED: TQDM progress bar for validation batches
                for x_batch, y_batch in tqdm(val_loader,
                                             desc=f"Epoch {epoch + 1}/{self.config.PINN_EPOCHS} (Validation)",
                                             leave=False):
                    x_batch, y_batch = x_batch.to(self.device), y_batch.to(self.device)
                    outputs, _, _, _ = model_pinn(x_batch)
                    loss = criterion(outputs, y_batch)
                    val_losses.append(loss.item())

            val_loss = np.mean(val_losses)

            scheduler.step(val_loss)

            if val_loss < min_val_loss:
                min_val_loss = val_loss
                early_stop_count = 0
            else:
                early_stop_count += 1

            if early_stop_count >= 10:
                print(f"[LOG] Early stopping at epoch {epoch + 1}")
                break

        # Save the model and metrics (same as before)
        model_path = f'models/weather_sta/{self.station_name.lower()}_pinn_model.pth'
        torch.save(model_pinn.state_dict(), model_path)
        print(f"   -> Saved model state to '{model_path}'")

        model_pinn.eval()
        X_test_tensor = X_test.to(self.device)
        with torch.no_grad():
            y_pred_tensor, _, _, _ = model_p