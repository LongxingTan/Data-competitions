"""
只选取2021年5月6号之后的数据
"""
import os
import json
import numpy as np
import pandas as pd
import multiprocessing
from pathlib import Path


def get_selected_hotelid(base_dir):
    submit_example = pd.read_csv(os.path.join(base_dir, 'testb/submit_example_2.csv'))
    selected_hotel_id = submit_example['HOTELID'].unique().tolist()
    print(len(selected_hotel_id))

    with open(os.path.join(Path(base_dir).parent, 'user_data/selected_hotel.json'), 'w') as f:
        json.dump(selected_hotel_id, f)
    return selected_hotel_id


def preprea_hotel_meta(base_dir, selected_hotel_id=None):
    hotel_info = pd.read_csv(os.path.join(base_dir, 'trainb/网约房注册民宿.csv'))

    if selected_hotel_id is not None:
        hotel_info = hotel_info.loc[hotel_info['CODE'].isin(selected_hotel_id)]

    hotel_info = hotel_info.loc[:, (hotel_info != hotel_info.iloc[0]).any()]  # drop single value columns
    print(hotel_info)
    hotel_info.to_csv(os.path.join(Path(base_dir).parent, 'user_data/hotel_info.csv'), index=False,
                      encoding='utf-8-sig')
    return hotel_info


def prepare_order_info(base_dir, selected_hotel_id=None, cancal=False):
    order = pd.read_csv(os.path.join(base_dir, 'trainb/网约平台旅客订单信息.csv'))
    order_stage1 = pd.read_csv(os.path.join(Path(base_dir).parent, '初赛/traina/网约平台旅客订单信息.csv'))
    order = pd.concat([order, order_stage1], axis=0, ignore_index=True)

    if cancal:
        order = order.loc[order['STATUS'] == 0]
    else:
        order = order.loc[order['STATUS'] > 0]  # 删除撤销订单记录
    order = order.drop_duplicates(subset=['ORDER_ID', 'PRE_IN_TIME', 'PRE_OUT_TIME'], keep='first')

    if selected_hotel_id is not None:
        order = order.loc[order['HOTELID'].isin(selected_hotel_id)]

    order['key'] = order['ORDER_ID'].astype(str)
    order['PRE_IN_TIME'] = order['PRE_IN_TIME'].apply(lambda x: str(x)[:8])
    order['PRE_OUT_TIME'] = order['PRE_OUT_TIME'].apply(lambda x: str(x)[:8])
    order['PRE_OUT_TIME'] = order['PRE_OUT_TIME'].apply(
        lambda x: x.replace('1596', '2020').replace('1597', '2020'))  # 错误数据
    order['PRE_IN_TIME'] = pd.to_datetime(order['PRE_IN_TIME']).dt.strftime('%Y%m%d')
    order['PRE_OUT_TIME'] = pd.to_datetime(order['PRE_OUT_TIME']).dt.strftime('%Y%m%d')

    date_range = pd.date_range(min(order['PRE_IN_TIME']), max(order['PRE_OUT_TIME']))
    order_date = pd.DataFrame(np.zeros([len(order), len(date_range)]), index=order['key'], columns=date_range)

    for i, d in order[['key', 'PRE_IN_TIME', 'PRE_OUT_TIME']].iterrows():
        start = pd.to_datetime(d['PRE_IN_TIME'])
        end = pd.to_datetime(d['PRE_OUT_TIME']) - pd.DateOffset(days=1)
        order_date.loc[d['key'], start:end] = 1

    order_date = order_date[pd.date_range('2021-05-06', max(order['PRE_OUT_TIME']))]  # Update date range from 20210506
    order_date.columns = [pd.to_datetime(i).strftime('%Y-%m-%d') for i in order_date.columns]

    order = order.set_index('key')
    order = pd.concat([order, order_date], axis=1)
    if cancal:
        order.to_csv(os.path.join(Path(base_dir).parent, 'user_data/order_info_51_cancel.csv'), index=False)
    else:
        order.to_csv(os.path.join(Path(base_dir).parent, 'user_data/order_info_51.csv'), index=False)

    order_by_hotel = order.groupby('HOTELID')[order_date.columns].agg('sum').reset_index()
    if cancal:
        order_by_hotel.to_csv(os.path.join(Path(base_dir).parent, 'user_data/order_agg_by_hotel_51_cancel.csv'),
                              index=False)
    else:
        order_by_hotel.to_csv(os.path.join(Path(base_dir).parent, 'user_data/order_agg_by_hotel_51.csv'), index=False)
    print(order_by_hotel)
    return


if __name__ == '__main__':
    base_dir = '../../data/复赛'
    selected_hotel_id = get_selected_hotelid(base_dir)

    preprea_hotel_meta(base_dir, selected_hotel_id=selected_hotel_id)
    prepare_order_info(base_dir, selected_hotel_id=selected_hotel_id)
    prepare_order_info(base_dir, selected_hotel_id=selected_hotel_id, cancal=True)
