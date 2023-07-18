# 划分各道具工作时间
import numpy as np
import pandas as pd
from scipy import stats


def find_t2_stage(load, speed):
    """T2刀工作时间
    思路： 设置一个主轴负载window，计算window内最大值。如果该window最大值小于13，则已经进入T4刀了。window移动时的stride可以设置为一半windo
    如果想要更精确的，找到以上window后，再根据主轴速度进行选择。 转速的前期最低点基本就是T2结束吧
    :return:
    """
    front_min_speed = np.argmin(speed[: int(len(speed) * 0.3)])  # 前30% 转速最小筛选
    front_speed_mode = stats.mode(speed[:front_min_speed])[0][0]
    start = np.min(np.arange(len(speed))[speed == front_speed_mode])
    return start, front_min_speed


def find_t11_stage(load, speed):
    # T11刀工作时间
    t2_start, t2_end = find_t2_stage(load, speed)
    censored_speed = speed[t2_end : int(len(speed) * 0.4)]  # 取T2结束到前40%
    t11_start = (
        np.min(np.arange(len(censored_speed))[censored_speed > 0.1]) + t2_end
    )  # 前期如果先到过0，后面的也不需要
    t11_end = np.max(np.arange(len(censored_speed))[censored_speed > 0.5]) + t2_end
    return t11_start, t11_end


def find_t8_stage(load, speed):
    """# T8
    思路： T8刀转速在一定范围，主轴负载较高。先根据转速找到T11的起点，往回找到负载较大的。确定起始点
    :return:
    """
    t11_start, t11_end = find_t11_stage(load, speed)
    print(t11_start, t11_end)
    t8_end = t11_start - 50
    t8_speed_mode = stats.mode(speed[t8_end - 1000 : t8_end])[0][0]

    censored_speed = speed[t8_end - (t11_end - t11_start) : t8_end]
    t8_start = (
        np.min(np.arange(len(censored_speed))[censored_speed == t8_speed_mode])
        + t8_end
        - (t11_end - t11_start)
    )
    return t8_start, t8_end


def find_t9_stage(load, speed):
    """# T9
    思路： 仍然根据转速找到T11刀的最后，然后根据负载找到T9刀的结束。由于转速都是恒定，所以可以相对容易
    :return:
    """
    t11_start, t11_end = find_t11_stage(load, speed)
    t9_start = t11_end + 50
    censored_speed = speed[t9_start + 900 :]
    t9_end = (
        np.min(np.arange(len(censored_speed))[censored_speed > -111]) + t9_start + 900
    )
    return t9_start, t9_end


if __name__ == "__main__":
    pass
