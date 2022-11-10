"""
rule的集大成者: 0.70*

change log: 训练集中的取消订单也可以以一个微笑的权重加进来
"""

import os
import numpy as np
import pandas as pd

# Read the data
base_dir = '../../data'
order = pd.read_csv(os.path.join(base_dir, 'user_data/order_agg_by_hotel.csv'))

# choose the recent date as train example
date_columns = list(order.iloc[:, 1:].columns)
date_columns = pd.DataFrame({'Date': [pd.to_datetime(i) for i in date_columns]})
selected_columns = date_columns.loc[date_columns['Date'] >= '20210601', 'Date'].tolist()
last_sep = pd.date_range('2020-09-01', '2020-09-30')
selected_columns += last_sep
selected_columns = [pd.to_datetime(i).strftime('%Y-%m-%d') for i in selected_columns]
print(selected_columns)

# use recent data
order_new = order[selected_columns]
order_new[order_new > 1] = 2
order = pd.concat([order[['HOTELID']], order_new], axis=1)

# calculate the history order rate as the probability of Sep
hotel_ratio = order.groupby(['HOTELID']).sum()
hotel_ratio['ratio'] = 1 - hotel_ratio.sum(axis=1) / hotel_ratio.shape[1]  # 有人住的标记为0
hotel_ratio = hotel_ratio.reset_index()

# cancel order with lower ratio, even it's cancelled, it still means this hotel is more popular
cancel_order = pd.read_csv(os.path.join(base_dir, 'user_data/order_agg_by_hotel_51_cancel.csv'), dtype={'HOTELID': str})
cancel_hotel_ratio = cancel_order.groupby(['HOTELID']).sum()
cancel_hotel_ratio['cancel_ratio'] = cancel_hotel_ratio.sum(axis=1) / cancel_hotel_ratio.shape[1]
cancel_hotel_ratio = cancel_hotel_ratio.reset_index()[['HOTELID', 'cancel_ratio']]

hotel_ratio = hotel_ratio.merge(cancel_hotel_ratio, on=['HOTELID'], how='left')
hotel_ratio['cancel_ratio'] = hotel_ratio['cancel_ratio'].fillna(0)
hotel_ratio['ratio'] = hotel_ratio['ratio'] + 0.35 * hotel_ratio['cancel_ratio']

# read the submit data, just keep the sequence is the same as evaluation
submit = pd.read_csv(os.path.join(base_dir, '复赛/testb/submit_example_2.csv'))
submit = submit.merge(hotel_ratio[['HOTELID', 'ratio']], how='left', on='HOTELID')
submit.drop(columns=['ROOM_EMPTY'], inplace=True)
submit.rename(columns={'ratio': 'ROOM_EMPTY'}, inplace=True)


# Post-process
# The order information has include some Sep information before Aug
hotel_agg = pd.read_csv(os.path.join(base_dir, 'user_data/order_agg_by_hotel.csv'), dtype={'HOTELID': str})
cancel_test = pd.read_csv(os.path.join(base_dir, 'user_data/order_agg_by_hotel_51_cancel.csv'),  dtype={'HOTELID': str})

for i, example in submit.iterrows():
    if cancel_test.loc[cancel_test['HOTELID'] == example['HOTELID'], example['DATE']].values > 0:
        submit.loc[i, 'ROOM_EMPTY'] -= 0.11
    if hotel_agg.loc[hotel_agg['HOTELID'] == example['HOTELID'], example['DATE']].values > 0:
        submit.loc[i, 'ROOM_EMPTY'] = 0

# The Festival influence
submit.loc[submit['DATE'].isin(['2021-09-19']), 'ROOM_EMPTY'] -= 0.2034
submit.loc[submit['DATE'].isin(['2021-09-20']), 'ROOM_EMPTY'] -= 0.2055
submit.loc[submit['DATE'].isin(['2021-09-21']), 'ROOM_EMPTY'] += 0.09

# The week before festival influence
submit.loc[submit['DATE'].isin(['2021-09-11']), 'ROOM_EMPTY'] -= 0.0155
submit.loc[submit['DATE'].isin(['2021-09-12']), 'ROOM_EMPTY'] -= 0.0155

# The Saturday or Sunday influence
submit.loc[submit['DATE'].isin(['2021-09-04']), 'ROOM_EMPTY'] -= 0.0336
submit.loc[submit['DATE'].isin(['2021-09-11']), 'ROOM_EMPTY'] -= 0.0336

# The clean job of post-process, in case the probability is greater than 1 or less than 0
submit.loc[submit['ROOM_EMPTY'] < 0, 'ROOM_EMPTY'] = 0
submit.loc[submit['ROOM_EMPTY'] > 1, 'ROOM_EMPTY'] = 1

# save the result
submit.to_csv(os.path.join(base_dir, 'submit/03031.csv'), index=False)
