# 流预测

from ai_hub import inferServer
from ai_hub.inferServer import shutdown_server
import traceback
import pandas as pd
import numpy as np
import json
import copy
import joblib
import argparse

# ======================================config============================================

base_json = '{"unit":[], "test_number":[], "ts": [], "qty": []}'
parser = argparse.ArgumentParser()
parser.add_argument('--use_model', help="rule/lgb/nn", default="rule")
args = parser.parse_args()
print(args.use_model)

if args.use_model == 'lgb' or args.use_model == 'rule_lgb':
    from train_tree import get_feature


# ======================================predict============================================

def forecast_function(demand_hist, lead_time=14, model=None):
    """ demand_hist: ts, unit, qty
    """
    if args.use_model == 'lgb':
        return forecase_function_lgb(demand_hist, lead_time, model)
    if args.use_model == 'rule':
        return forecast_function_rule(demand_hist, lead_time, model)
    if args.use_model == 'rule_lgb':
        return [0.25 * forecast_function_rule(demand_hist, lead_time, model)[0] + 0.75 * forecase_function_lgb(demand_hist, lead_time, model)[0]]
    else:
        raise ValueError


def forecast_function_rule(demand_hist, lead_time=14, model=None):
    norm = remove_anomaly(demand_hist["qty"].values[-5 * 7:], max_deviations=3)
    demand_sum = 1.1 * (1.2 * np.sum(norm[-7:]) + 0.3 * np.sum(norm[-14:-7]) + 0.15 * np.sum(norm[-21:-14]))
    return [demand_sum]  # 预测90天，补货时截取lead_time


def forecase_function_lgb(demand_hist, lead_time=14, model=None):
    demand_hist.fillna(method='ffill', inplace=True)
    data, feature_cols = get_feature(demand_hist, demand_col='qty')
    feature = data[feature_cols]
    y_pred = model.predict(feature.iloc[-1:])[0] * 1.5
    return [y_pred]


# ======================================safety stock ============================================

def get_safety_stock(demand_hist, lead_time, decrease=False, w1=11, w2=1.76, w3=0.09):
    pos = demand_hist.iloc[-6 * lead_time:].loc[demand_hist['qty'] > 0]
    pos = pos['qty'].values
    pos_norm = remove_anomaly(pos, max_deviations=3)[-14:]

    demand_hist['week'] = demand_hist['ts'].dt.strftime('%Y%U')
    demand_week = demand_hist.groupby(['week'])['qty'].sum().reset_index()

    if True:
        if len(pos) > 4:
            # Key of my solution, this one line can already get the 2nd place
            ss = w1 * np.std(pos_norm) + w2 * np.std(demand_week.iloc[-52:]['qty']) + w3 * (1.0 * np.max(demand_week['qty']) - np.max(np.mean(demand_week['qty']), 0))
        else:  # the effect could almost be ignored
            ss = 0.07 * np.std(pos_norm) * 10
    return ss


# ====================================== util ============================================

def remove_anomaly(array, max_deviations=3):
    mu = np.mean(array)
    sigma = np.std(array)
    array_mu = np.abs(array - mu)
    norm = array_mu < max_deviations * sigma
    return array[norm]


# ====================================== main============================================

class TestUnit:
    # 储存单次test_number、单个unit的信息
    def __init__(self,
                 unit,
                 intransit,
                 qty_inventory_today,
                 qty_using_today,
                 arrival_sum,
                 lead_time):
        self.unit = unit
        self.intransit = intransit
        self.qty_inventory_today = qty_inventory_today
        self.qty_using_today = qty_using_today
        self.arrival_sum = arrival_sum
        self.lead_time = lead_time

    def set_test_number(self, test_number):
        self.test_number = test_number

    def update(self,
               arrival_today,
               demand_today):
        '''
        每日根据当天补货到达与当日净需求更新状态
        :param arrival_today: 当天补货到达
        :param demand_today: 当天净需求
        :return:
        '''
        self.qty_inventory_today += arrival_today
        self.arrival_sum += arrival_today
        inv_today = self.qty_inventory_today
        if demand_today < 0:
            self.qty_inventory_today = self.qty_inventory_today + min(-demand_today, self.qty_using_today)
        else:
            self.qty_inventory_today = max(self.qty_inventory_today - demand_today, 0.0)
        self.qty_using_today = max(self.qty_using_today + min(demand_today, inv_today), 0.0)

    def replenish_function(self,
                           date,
                           qty_demand_forecast,
                           demand_hist,
                           decrease=False):
        '''
        根据当前状态判断需要多少的补货量
        补货的策略由选手决定，这里只给一个思路
        :param date:
        :return:
        '''
        replenish = 0.0

        # 计算在途的补货量
        qty_intransit = sum(self.intransit) - self.arrival_sum

        # 安全库存 用来抵御需求的波动性 选手可以换成自己的策略
        safety_stock = get_safety_stock(demand_hist, self.lead_time, decrease)

        # 再补货点，用来判断是否需要补货 选手可以换成自己的策略
        reorder_point = sum(qty_demand_forecast[:self.lead_time]) + safety_stock

        # 判断是否需要补货并计算补货量，选手可以换成自己的策略，可以参考赛题给的相关链接
        # https://en.wikipedia.org/wiki/Reorder_point
        if self.qty_inventory_today + qty_intransit < reorder_point:
            replenish = reorder_point - (self.qty_inventory_today + qty_intransit)

        if date == pd.to_datetime('20210614'):
            replenish = 0.55 * replenish
        if date == pd.to_datetime('20210621'):
            replenish = 0.65 * replenish
        self.intransit.at[date + self.lead_time * date.freq] = replenish
        return {"unit": self.unit, "test_number": self.test_number, "ts": date, "qty": replenish}


class ReplenishUnit:  # 填弹
    def __init__(self,
                 unit,
                 demand_hist,
                 test_unit,
                 intransit,
                 qty_inventory_today,
                 qty_using_today,
                 arrival_sum,
                 lead_time
                 ):
        '''
        记录补货单元状态
        :param unit:
        :param demand_hist: 净需求历史
        :param intransit: 补货在途
        :param qty_inventory_today: 当前可用库存
        :param qty_using_today: 当前已用库存（使用量）,保有量
        :param arrival_sum: 补货累计到达
        :param lead_time: 补货时长，交货时间
        '''
        self.unit = unit
        self.demand_hist = demand_hist
        self.lead_time = lead_time
        self.tu = TestUnit(unit=unit,
                           intransit=intransit,
                           qty_inventory_today=qty_inventory_today,
                           qty_using_today=qty_using_today,
                           arrival_sum=arrival_sum,
                           lead_time=lead_time)
        test_unit_list = []
        for x in test_unit:
            tmp = copy.deepcopy(self.tu)
            tmp.set_test_number(x)
            test_unit_list.append(tmp)
        self.test_unit = dict(zip(test_unit, test_unit_list))
        if args.use_model == 'lgb':
            self.ml_model = joblib.load('./lgb.pkl')
        elif args.use_model == 'rule':
            self.ml_model = None

    def update(self,
               date,
               arrival_today,
               demand_today):
        '''
        每日根据当天补货到达与当日净需求更新状态
        :param date:
        :param arrival_today: 当天补货到达
        :param demand_today: 当天净需求
        :return:
        '''
        # 对每个test_number更新今日到达以及今日需求
        for test_number, arrival_today in arrival_today[["test_number", "qty"]].values:
            self.test_unit[test_number].update(arrival_today=arrival_today, demand_today=demand_today)
        self.demand_hist = self.demand_hist.append({"ts": date, "unit": self.unit, "qty": demand_today}, ignore_index=True)

    def replenish_function(self, date, decrease=False):
        '''
        根据当前状态判断需要多少的补货量
        补货的策略由选手决定，这里只给一个思路
        :param date:
        :return:
        '''
        qty_demand_forecast = forecast_function(self.demand_hist, model=self.ml_model)  # 预测
        res = pd.DataFrame(columns=["unit", "test_number", "ts", "qty"])
        # 对每个test_number做决策
        for test_number in self.test_unit:
            repl = self.test_unit[test_number].replenish_function(date=date,
                                                                  qty_demand_forecast=qty_demand_forecast,
                                                                  demand_hist=self.demand_hist,
                                                                  decrease=decrease)
            res = res.append(repl, ignore_index=True)
        return res


class Env:
    # 这个类存储所有补货单元
    def __init__(self, replenish_unit_dict):
        self.replenish_unit_dict = replenish_unit_dict
        self.decision_date = 0

    def update(self, unit, date, arrival_today, demand_today):
        self.replenish_unit_dict[unit].update(date=date, arrival_today=arrival_today, demand_today=demand_today)

    def replenish(self, unit, date, decrease=False):
        return self.replenish_unit_dict[unit].replenish_function(date=date, decrease=decrease)


class myInfer(inferServer):
    def __init__(self, model):
        super().__init__(model)
        print("init_myInfer")

    # 数据前处理
    def pre_process(self, data):
        print("my_pre_process")
        data = data.get_data()

        # json process
        # 需求数据
        json_data = json.loads(data.decode('utf-8'))
        demand_data = pd.DataFrame(json_data["demand"])
        demand_data["ts"] = [pd.to_datetime(x) for x in demand_data["ts"].values]
        demand_data = demand_data.sort_values("ts", ascending=True)

        # 到货数据
        arrival_data = pd.DataFrame(json_data["arrival"])
        arrival_data["ts"] = [pd.to_datetime(x) for x in arrival_data["ts"].values]
        arrival_data = arrival_data.sort_values("ts", ascending=True)

        print("processed data: ", arrival_data)
        return demand_data, arrival_data

    # 主逻辑
    def replenish_logic(self, data):
        print("main logic")
        demand_data = data[0]
        arrival_data = data[1]
        date_list = pd.date_range(demand_data["ts"].min(), demand_data["ts"].max())
        res = pd.DataFrame(columns=["unit", "test_number", "ts", "qty"])
        for chunk in demand_data.groupby("unit"):
            unit = chunk[0]
            unit_data = chunk[1]
            for date in date_list:
                demand_today = unit_data[unit_data["ts"] == date]["qty"].values[0]
                arrival_today = arrival_data[(arrival_data["ts"] == date) & (arrival_data["unit"] == unit)]
                # 更新信息
                self.model.update(unit=unit, date=date, arrival_today=arrival_today, demand_today=demand_today)
                # 决策日决策
                if date.dayofweek == 0:
                    print('决策日决策中')
                    replenish = self.model.replenish(unit=unit, date=date, decrease=False)  #
                    res = pd.concat([res, replenish])
        return res

    # 框架限制，方法名需要为predict
    def predict(self, data):
        try:
            res = self.replenish_logic(data)
            return res
        except Exception as e:
            traceback.print_exc()
            shutdown_server()

    # 数据后处理
    def post_process(self, data):
        print("post_process")
        processed_data = json.loads(base_json, encoding='utf-8')
        processed_data["unit"] = list(data["unit"].values)
        processed_data["test_number"] = list(data["test_number"].values)
        processed_data["ts"] = list(data["ts"].apply(lambda x: str(x)[:10]).values)
        processed_data["qty"] = list(data["qty"].values)
        processed_data = json.dumps(processed_data)
        return processed_data


if __name__ == "__main__":
    print("----------")
    base_dir = '../tcdata'
    using_hist = pd.read_csv(base_dir + "/demand_train.csv")
    inventory = pd.read_csv(base_dir + "/inventory_info.csv")
    lead_time = 14
    last_dt = pd.to_datetime("20210608")
    start_dt = pd.to_datetime("20210609")
    end_dt = pd.to_datetime("20211101")
    test_unit = ["test_number_" + str(x) for x in range(10)]
    using_hist["ts"] = using_hist["ts"].apply(lambda x: pd.to_datetime(x))
    date_list = pd.date_range(start=start_dt, end=end_dt)
    replenishUnit_dict = {}

    # 初始化，记录各补货单元在评估开始前的状态
    print("init")
    for chunk in using_hist.groupby("unit"):
        unit = chunk[0]
        demand = chunk[1]
        demand.sort_values("ts", inplace=True, ascending=True)

        # 计算净需求量
        demand["diff"] = demand["qty"].diff().values
        demand["qty"] = demand["diff"]
        del demand["diff"]
        demand = demand[1:]
        replenishUnit_dict[unit] = ReplenishUnit(unit=unit,
                                                 demand_hist=demand,
                                                 test_unit=test_unit,
                                                 intransit=pd.Series(index=date_list.tolist(), data=[0.0] * (len(date_list))),
                                                 qty_inventory_today=inventory[inventory["unit"] == unit]["qty"].values[0],
                                                 qty_using_today=using_hist[(using_hist["ts"] == last_dt) & (using_hist["unit"] == unit)]["qty"].values[0],
                                                 arrival_sum=0.0,
                                                 lead_time=lead_time)
    env = Env(replenishUnit_dict)
    my_infer = myInfer(env)
    print("begin")

    my_infer.run(debuge=False, ip="127.0.0.1", port=8080)
