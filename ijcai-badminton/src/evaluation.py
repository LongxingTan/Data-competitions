import pandas as pd
import numpy as np
import torch
import math
from tqdm import tqdm


class StrokeEvaluator:
    def __init__(self, path):
        self.type_list = ['short service', 'net shot', 'lob', 'clear', 'drop', 'push/rush', 'smash', 'defensive shot', 'drive', 'long service']
        self.prediction = pd.read_csv(f"../../input/dataset/prediction.csv")    

        print(self.prediction['landing_y'].sum())
        hist = pd.read_csv('../../input/dataset/val_offline_given.csv')
        for i, row in self.prediction.iterrows():
            rally_id = int(row['rally_id'])
            ball_round = row['ball_round'] % 2  # 0或1
            if ball_round == 0:
                ball_round = 2

            landing_y = row['landing_y']
            flag = hist.loc[(hist['rally_id'] == rally_id) & (hist['ball_round'] == ball_round), 'landing_y'].values[0]

            if landing_y * flag < 0:
                #print(rally_id, row['ball_round'], landing_y, flag)
                #row['landing_y'] = -1 * landing_y
                self.prediction.loc[i, 'landing_y'] = np.clip(-1 * landing_y * 0.8, -1, 1)
 
        print(self.prediction['landing_y'].sum())
        print(self.prediction)
        self.prediction['landing_x'] = self.prediction['landing_x'].apply(lambda x: np.clip(float(x), -1.4, 1.4))
        self.prediction['landing_y'] = self.prediction['landing_y'].apply(lambda x: np.clip(float(x), -1.5, 1.5))
        #print(self.prediction)
        self.ground_truth = pd.read_csv(f"../../input/dataset/val_offline_true.csv")

        self.l1_loss = torch.nn.L1Loss(reduction='mean')
        self.ce_loss = torch.nn.NLLLoss(reduction='mean')   # we use NLL since we need to check if we need to softmax the probs of each shot

        self.check_and_convert_type_prob_with_softmax()
        self.compute_metrics()

    def softmax(self, prob_list):
        return np.exp(prob_list) / sum(np.exp(prob_list))
    
    def check_and_convert_type_prob_with_softmax(self):
        # normalized prob of types if the sum of prob is not 1
        converted_type_probs = []
        for n, row in self.prediction.iterrows():
            # round to 5 decimals to prevent minor computation error
            if round(self.prediction.iloc[n][self.type_list].sum(), 5) != 1:
                converted_type_probs.append(self.softmax(row[self.type_list].values))
            else:
                converted_type_probs.append(row[self.type_list].values)
        self.prediction.loc[:, self.type_list] = converted_type_probs

    def compute_metrics(self):
        """
        for each rally
            for each sample
                compute metrics
                check if metrics is the best
        """

        total_score, total_ce_score, total_mae_score = 0, 0, 0
        rally_ids, rally_score, rally_ce_score, rally_mae_score = [], [], [], []
        group = self.prediction[['rally_id', 'sample_id', 'ball_round', 'landing_x', 'landing_y', 'short service', 'net shot', 'lob', 'clear', 'drop', 'push/rush', 'smash', 'defensive shot', 'drive', 'long service']].groupby('rally_id').apply(lambda r: (r['sample_id'].values, r['ball_round'].values, r['landing_x'].values, r['landing_y'].values, r['short service'].values, r['net shot'].values, r['lob'].values, r['clear'].values, r['drop'].values, r['push/rush'].values, r['smash'].values, r['defensive shot'].values, r['drive'].values, r['long service'].values))
        ground_truth = self.ground_truth[['rally_id', 'ball_round', 'landing_x', 'landing_y', 'type']].groupby('rally_id').apply(lambda r: (r['ball_round'].values, r['landing_x'].values, r['landing_y'].values, r['type'].values))

        for i, rally_id in tqdm(enumerate(ground_truth.index), total=len(ground_truth)):
            best_sample_score, best_ce_score, best_mae_score = 1e6, 1e6, 1e6
            sample_id, ball_round, landing_x, landing_y, short_service, net_shot, lob, clear, drop, push_rush, smash, defensive_shot, drive, long_service = group[rally_id]
            
            true_ball_round, true_landing_x, true_landing_y, true_types = ground_truth[rally_id]
            converted_true_types = []
            for true_type in true_types:
                converted_true_types.append(self.type_list.index(true_type))
            converted_true_types = torch.tensor(converted_true_types)
            ground_truth_len = len(true_ball_round)
            
            for sample in range(6):      # total 6 samples
                start_index = sample * ground_truth_len
                # compute area score
                area_score = self.compute_mae_metric(landing_x[start_index:start_index+ground_truth_len], landing_y[start_index:start_index+ground_truth_len], true_landing_x, true_landing_y)

                # compute type score
                prediction_type = []
                for shot_index in range(start_index, start_index+ground_truth_len):
                    prediction_type.append([short_service[shot_index], net_shot[shot_index], lob[shot_index], clear[shot_index], drop[shot_index], push_rush[shot_index], smash[shot_index], defensive_shot[shot_index], drive[shot_index], long_service[shot_index]])
                prediction_type = torch.tensor(prediction_type)
                type_score = self.ce_loss(torch.log(prediction_type), converted_true_types).item()  # need to perform log operation
                if math.isinf(type_score):
                    type_score = 1e3        # modify type_score to 1000 if the prediction prob is uniform, which causes inf

                # check if the current score better than the previous best score
                if area_score + type_score < best_sample_score:
                    best_sample_score = area_score + type_score
                    best_ce_score = type_score
                    best_mae_score = area_score

            rally_ids.append(rally_id), rally_score.append(best_sample_score), rally_ce_score.append(best_ce_score), rally_mae_score.append(best_mae_score)
            total_score += best_sample_score
            total_ce_score += best_ce_score
            total_mae_score += best_mae_score

        rally_ids.append('total'), rally_score.append(round(total_score/len(group.index), 5)), rally_ce_score.append(round(total_ce_score/len(group.index), 5)), rally_mae_score.append(round(total_mae_score/len(group.index), 5))
        record_df = pd.DataFrame({'type': rally_ce_score, 'area': rally_mae_score, 'overall': rally_score})
        record_df.index = rally_ids
        record_df.to_csv("record_score.csv")
    
    def compute_mae_metric(self, landing_x, landing_y, true_landing_x, true_landing_y):
        prediction_area = torch.tensor([[x, y] for x, y in zip(landing_x, landing_y)])
        true_area = torch.tensor([[x, y] for x, y in zip(true_landing_x, true_landing_y)])
        area_score = self.l1_loss(prediction_area, true_area)
        return area_score.item()


if __name__ == "__main__":
    path = "../data/"
    stroke_evaluator = StrokeEvaluator(path=path)
    print("Evaluation Done")
