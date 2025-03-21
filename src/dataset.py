import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np

class MLB_Batter_Dataset(Dataset):
    def __init__(self, batter_df, pitcher_df, sequence_length=3, batter_features=None, pitcher_features=None):
        """
        Args:
            batter_df: DataFrame with batter stats.
            pitcher_df: DataFrame with pitcher stats.
            sequence_length: Number of past games to use as history.
            batter_features: List of feature names for batter sequence.
            pitcher_features: List of feature names for pitcher sequence.
        """
        self.sequence_length = sequence_length
        self.batter_features = batter_features
        self.pitcher_features = pitcher_features

        # Sort the batter and pitcher data by date
        self.batter_df = batter_df.sort_values(by='DATE')
        self.pitcher_df = pitcher_df.sort_values(by='DATE')
        
        # Group by player IDs for fast lookup
        self.batter_groups = self.batter_df.groupby('PLAYER-ID')
        self.pitcher_groups = self.pitcher_df.groupby('PLAYER-ID')
        
        self.samples = []
        for idx, row in self.batter_df.iterrows():
            batter_id = row['PLAYER-ID']
            game_date = row['DATE']
            # Extract batter's historical games (only those before the current game)
            batter_history = self.batter_groups.get_group(batter_id)
            batter_history = batter_history[batter_history['DATE'] < game_date]
            if len(batter_history) < sequence_length:
                continue
            batter_seq = batter_history.tail(sequence_length)
            
            # Extract opposing starting pitcher information using the "STARTING\nPITCHER" column
            opp_pitcher_id = row['opp_starting_pitcher']
            if pd.isnull(opp_pitcher_id):
                continue
            try:
                opp_pitcher_history = self.pitcher_groups.get_group(opp_pitcher_id)
            except KeyError:
                continue
            opp_pitcher_history = opp_pitcher_history[opp_pitcher_history['DATE'] < game_date]
            if len(opp_pitcher_history) < sequence_length:
                continue
            opp_seq = opp_pitcher_history.tail(sequence_length)
            
            # Optionally, you can also include the opposing pitcher ID in the sample if needed
            self.samples.append({
                'batter_seq': batter_seq[self.batter_features].values.astype(np.float32),
                'opp_pitcher_seq': opp_seq[self.pitcher_features].values.astype(np.float32),
                'opp_pitcher_id': opp_pitcher_id,
                'target': np.array(row['next_target_H'], dtype=np.float32)
            })
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        batter_seq_tensor = torch.tensor(sample['batter_seq'])
        opp_seq_tensor = torch.tensor(sample['opp_pitcher_seq'])
        target_tensor = torch.tensor(sample['target'])
        # If needed, you could also return the opposing pitcher ID
        return batter_seq_tensor, opp_seq_tensor, target_tensor


class MLB_Pitcher_Dataset(Dataset):
    def __init__(self, pitcher_df, batter_df, sequence_length=3, pitcher_features=None, batter_features=None):
        """
        pitcher_df: DataFrame containing pitcher stats.
        batter_df: DataFrame containing batter stats.
        sequence_length: Number of past games to use as history.
        pitcher_features: List of feature names for pitcher sequence.
        batter_features: List of feature names for aggregated opponent batter sequence.
        """
        self.sequence_length = sequence_length
        self.pitcher_features = pitcher_features
        self.batter_features = batter_features

        # Sort data by date
        self.pitcher_df = pitcher_df.sort_values(by='DATE')
        self.batter_df = batter_df.sort_values(by='DATE')
        
        # Group by pitcher id and by opponent team for batters
        self.pitcher_groups = self.pitcher_df.groupby('PLAYER-ID')
        self.batter_groups_by_team = batter_df.groupby('TEAM')
        
        self.samples = []
        for idx, row in self.pitcher_df.iterrows():
            pitcher_id = row['PLAYER-ID']
            game_date = row['DATE']
            # Pitcher's history
            pitcher_history = self.pitcher_groups.get_group(pitcher_id)
            pitcher_history = pitcher_history[pitcher_history['DATE'] < game_date]
            if len(pitcher_history) < sequence_length:
                continue
            pitcher_seq = pitcher_history.tail(sequence_length)
            
            # For the opposing batters, use the 'OPPONENT' team column.
            opp_team = row['OPPONENT']
            if pd.isnull(opp_team):
                continue
            try:
                team_batter_history = self.batter_groups_by_team.get_group(opp_team)
            except KeyError:
                continue
            # Consider only games before the current game date
            team_batter_history = team_batter_history[team_batter_history['DATE'] < game_date]
            if team_batter_history.empty:
                continue
            # Aggregate batters’ stats by date (mean) on only the batter_features columns
            agg_history = team_batter_history.groupby('DATE')[self.batter_features].mean().reset_index()
            if len(agg_history) < sequence_length:
                continue
            opp_seq = agg_history.tail(sequence_length)
            
            # Target: pitcher's strikeouts (column 'SO.1')
            target = row['next_target_SO']
            
            self.samples.append({
                'pitcher_seq': pitcher_seq[self.pitcher_features].values.astype(np.float32),
                'opp_batter_seq': opp_seq[self.batter_features].values.astype(np.float32),
                'target': np.array(target, dtype=np.float32)
            })
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        pitcher_seq_tensor = torch.tensor(sample['pitcher_seq'])
        opp_seq_tensor = torch.tensor(sample['opp_batter_seq'])
        target_tensor = torch.tensor(sample['target'])
        return pitcher_seq_tensor, opp_seq_tensor, target_tensor

class PretrainingDataset(Dataset):
    def __init__(self, df, player_id_col, date_col, useful_stats_cols, sequence_length=3):
        """
        Builds a dataset for pretraining an autoencoder from sequences.
        Each sample is a sequence of the last `sequence_length` games for a given player.
        """
        self.sequence_length = sequence_length
        self.useful_stats_cols = useful_stats_cols
        
        # Sort the data by player and date
        self.df = df.sort_values(by=[player_id_col, date_col])
        self.player_id_col = player_id_col
        self.date_col = date_col
        
        self.samples = []
        grouped = self.df.groupby(player_id_col)
        for player, group in grouped:
            group = group.sort_values(by=date_col)
            if len(group) >= sequence_length:
                for i in range(sequence_length, len(group)):
                    seq = group.iloc[i-sequence_length:i]
                    self.samples.append(seq[self.useful_stats_cols].values.astype(np.float32))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        # For pretraining, input and target are the same (reconstruction objective)
        sequence = torch.tensor(self.samples[idx])
        return sequence
