import numpy as np
import pandas as pd

input_dir = "../../input/dataset/"

pred = pd.read_csv(input_dir + 'prediction.csv')

hist = pd.read_csv(input_dir + 'test_given.csv')
print(pred)
print(hist)


for i, row in pred.iterrows():
    rally_id = int(row['rally_id'])
    ball_round = row['ball_round'] % 2  # 0或1
    if ball_round == 0:
        ball_round = 2

    landing_y = row['landing_y']
    flag = hist.loc[(hist['rally_id'] == rally_id) & (hist['ball_round'] == ball_round), 'landing_y'].values[0]

    if landing_y * flag < 0:
        #print(rally_id, row['ball_round'], landing_y, flag)
        #row['landing_y'] = -1 * landing_y
        pred.loc[i, 'landing_y'] = np.clip(-1 * landing_y * 0.8, -1, 1)

   
pred['landing_x'] = pred['landing_x'].apply(lambda x: np.clip(float(x), -1.4, 1.4))
pred['landing_y'] = pred['landing_y'].apply(lambda x: np.clip(float(x), -1.5, 1.5))

pred.to_csv(input_dir + 'prediction_sub.csv', index=False)
