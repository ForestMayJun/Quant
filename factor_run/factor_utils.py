import pandas as pd
import numpy as np

window = 30
decay_rate = 0.95
# weight = np.array([np.power(decay_rate, window - i) for i in range(window)])
weight = np.array([range(1, 31, 1)])
weight_sum = np.sum(weight)
def skewness_power(x, weight=weight, weight_sum=weight_sum):
    y = (x - x.mean()) / (x.std() + 1e-10)
    y = np.power(y, 3)

    return np.dot(weight, y.values) / weight_sum


def rolling(data, f, window=30):
    '''
    pd.series -> pd.series,
    每个元素是rolling前window窗口 -> 
    '''
    res = pd.Series(index=data.index) # 储存数据 iloc[i]为i-window - i时间段的f的值，f:pd.series -> float
    for i in range(window, len(data)):
        res.iloc[i] = f(data.iloc[i-window:i])
    
    return res.fillna(0)

