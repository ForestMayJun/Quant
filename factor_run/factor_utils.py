'''因子计算工具'''
import pandas as pd
import numpy as np
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from forintern.DataDaily import DataDaily

def log_return(df:pd.DataFrame):
    '''接受价格df对象,返回对数收益率'''
    df = df.fillna(method='ffill')
    ratio = df.pct_change() + 1
    log_ratio = np.log(ratio).fillna(0)

    return log_ratio

if __name__ == '__main__':
    s1 = pd.Series([2, np.nan, 6, 3, np.nan])
    print(log_return(s1))
    # print(np.log(3))