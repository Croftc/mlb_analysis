batter_features = [
 'game_index',
 'AB',
 'R',
 'H',
 'RBI',
 'BB',
 'SO',
 'VENUE_Road',
 'HAND_R']

pitcher_features = [
 'game_index',
 'IP',
 'R.1',
 'H.1',
 'ER',
 'BB.1',
 'SO.1',
 'ERA',
 'VENUE_Road',
 'HAND.1_R']

batter_features_preprocess = ['game_index', 'DATE', 'PLAYER', 'TEAM',
                    'VENUE', 'OPPONENT','AB', 'R', 'H', 'RBI',
                    'BB', 'SO', 'next_target_H', "PLAYER-ID",
                    'next_target_H', "PLAYER-ID", 'VENUE_Road',
                    'opp_starting_pitcher', 'HAND_R']
continuous_batter = ['AB', 'R', 'H', 'RBI', 'BB', 'SO']

pitcher_features_preprocess = ['game_index', 'DATE', 'PLAYER', 'TEAM',
                    'VENUE', 'OPPONENT', 'IP', 'H.1', 'R.1',
                    'ER', 'ERA', 'BB.1', 'SO.1', 'next_target_SO',
                    "PLAYER-ID", 'VENUE_Road', 'HAND.1_R']
continuous_pitcher = ['IP', 'H.1', 'R.1', 'ER', 'ERA', 'BB.1', 'SO.1']

categorical_batter = ['VENUE', 'HAND']
categorical_pitcher = ['VENUE','HAND.1']

model_config = {
    'seq_len': 5,
    'pitcher_input_dim':len(pitcher_features),
    'opp_batter_input_dim':len(batter_features),
    'model_dim':64, 
    'n_heads':4, 
    'num_layers':2, 
    'sequence_length':5
}