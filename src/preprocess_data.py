import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.preprocessing import StandardScaler
from config import continuous_batter, continuous_pitcher, categorical_batter, categorical_pitcher

def normalize_features(df, continuous_features):
    scaler = StandardScaler()
    df[continuous_features] = scaler.fit_transform(df[continuous_features])
    return df, scaler

def one_hot_encode_features(df, categorical_features):
    dummies = pd.get_dummies(df[categorical_features], drop_first=True, dtype=float)
    df = df.join(dummies)
    return df

def compute_metrics(preds, targets):
    preds = preds.detach().cpu().numpy()
    targets = targets.detach().cpu().numpy()
    mae = np.mean(np.abs(preds - targets))
    rmse = np.sqrt(np.mean((preds - targets)**2))
    ss_res = np.sum((targets - preds)**2)
    ss_tot = np.sum((targets - np.mean(targets))**2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return mae, rmse, r2

def assign_game_index(df):
    # If your dataset contains a unique GAME-ID column, use it. Otherwise, use TEAM and DATE.
    if 'GAME-ID' in df.columns:
        unique_games = df[['TEAM', 'GAME-ID', 'DATE']].drop_duplicates().sort_values(by=['TEAM', 'DATE'])
        unique_games['game_index'] = unique_games.groupby('TEAM').cumcount()
        # Merge back the game_index onto the original DataFrame
        df = df.merge(unique_games[['TEAM', 'GAME-ID', 'game_index']], on=['TEAM', 'GAME-ID'], how='left')
    else:
        unique_games = df[['TEAM', 'DATE']].drop_duplicates().sort_values(by=['TEAM', 'DATE'])
        unique_games['game_index'] = unique_games.groupby('TEAM').cumcount()
        df = df.merge(unique_games[['TEAM', 'DATE', 'game_index']], on=['TEAM', 'DATE'], how='left')
    return df

def clean_and_split(df, inference=False):
    df.columns = df.columns.str.strip().str.replace('"', '')

    # Flag pitchers and set targets
    df['is_pitcher'] = df['STARTING\nPITCHER'].notnull()
    df['target_SO'] = df.apply(lambda row: row["SO.1"] if row['is_pitcher'] else row["SO"], axis=1)
    df['target_H']  = df.apply(lambda row: row["H.1"] if row['is_pitcher'] else row["H"], axis=1)

    # Offset targets per player so current game features predict next game outcomes
    if not inference:
        df.sort_values(["PLAYER-ID", "DATE"], inplace=True)
        df["next_target_SO"] = df.groupby("PLAYER-ID")["target_SO"].shift(-1)
        df["next_target_H"]  = df.groupby("PLAYER-ID")["target_H"].shift(-1)
        df = df.dropna(subset=["next_target_SO", "next_target_H"])

    # -----------------------------
    # 2. Split into Pitchers and Batters
    # -----------------------------
    pitchers_df = df[df['is_pitcher']].copy()
    batters_df = df[~df['is_pitcher']].copy()

    return pitchers_df, batters_df



def normalize_and_index(pitchers_df, batters_df, pitcher_scaler=None, batter_scaler=None):
    pitchers_df['DATE'] = pd.to_datetime(pitchers_df['DATE'])
    pitchers_df = assign_game_index(pitchers_df)

    batters_df['DATE'] = pd.to_datetime(batters_df['DATE'])
    batters_df = assign_game_index(batters_df)
    
    # normalize numerical features
    if pitcher_scaler is None:
        pitchers_df, pitcher_scaler = normalize_features(pitchers_df, continuous_pitcher)
    else:
        pitchers_df[continuous_pitcher] = pitcher_scaler.transform(pitchers_df[continuous_pitcher].values)
    
    if batter_scaler is None:
        batters_df, batter_scaler = normalize_features(batters_df, continuous_batter)
    else:
        batters_df[continuous_batter] = batter_scaler.transform(batters_df[continuous_batter].values)

    # one-hot encode categorical features
    batters_df = one_hot_encode_features(batters_df, categorical_batter)
    pitchers_df = one_hot_encode_features(pitchers_df, categorical_pitcher)

    # save sklearn scalers
    torch.save(batter_scaler, "batter_scaler.pt")
    torch.save(pitcher_scaler, "pitcher_scaler.pt")

    return pitchers_df, batters_df

def align_opps(pitchers_df, batters_df):
    # Create a subset of the pitcher DataFrame with only the relevant columns.
    # We assume 'TEAM' in pitchers_df is the pitcher's team and that each row is a starting pitcher.
    # Rename the 'PLAYER-ID' column to something like 'opp_starting_pitcher' to clarify its role.
    pitcher_lookup = pitchers_df[['GAME-ID', 'TEAM', 'DATE', 'PLAYER-ID']].copy()
    pitcher_lookup = pitcher_lookup.rename(columns={
        'TEAM': 'opp_team', 
        'PLAYER-ID': 'opp_starting_pitcher'
    })

    # Merge batter_df with pitcher_lookup based on the batter's opponent and the game date.
    # This will add a new column 'opp_starting_pitcher' to batters_df.
    batters_df = batters_df.merge(
        pitcher_lookup[['opp_team', 'DATE', 'opp_starting_pitcher']], 
        left_on=['OPPONENT', 'DATE'], 
        right_on=['opp_team', 'DATE'], 
        how='left'
    )

    # Optionally, drop the extra 'opp_team' column if you don't need it.
    batters_df.drop(columns=['opp_team'], inplace=True)
    batters_df['opp_starting_pitcher'].fillna(0, inplace=True)
    batters_df['opp_starting_pitcher'] = batters_df['opp_starting_pitcher'].astype(int)
    return pitchers_df, batters_df


# pipeline these functions
def preprocess(df, pitcher_scaler=None, batter_scaler=None, inference=False):
    pitchers_df, batters_df = clean_and_split(df, inference=inference)
    pitchers_df, batters_df = normalize_and_index(pitchers_df, batters_df, pitcher_scaler, batter_scaler)
    pitchers_df, batters_df = align_opps(pitchers_df, batters_df)
    return pitchers_df, batters_df
