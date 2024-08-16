import numpy as np
import pandas as pd
from tqdm import tqdm
from datetime import datetime, time, timedelta
import sys
import os

sys.path.append('/mnt/datadisk2/aglv/aglv/lab_aglv')
# from forintern.DataDaily import DataDaily
# from forintern.DataMinute_vwap import DataMinute

tqdm.pandas()

def time_list_t0(init='09:30', end='15:00', hour_gap=0, min_gap=10):
    '''
    生成一段均匀间隔的时点,Parameters:
    init:初始时间
    end:结束时间
    hour_gap, min_gap:间隔时间的小时分钟数
    '''
    init_time = datetime.strptime(init, '%H:%M')
    end_time = datetime.strptime(end, '%H:%M')

    noon_begin = datetime.strptime('11:30', '%H:%M')
    noon_end = datetime.strptime('13:00', '%H:%M')

    time_delta = timedelta(hours=hour_gap, minutes=min_gap)

    n_time_number = (end_time - init_time)/time_delta
    time_list = []
    for i in range(int(n_time_number)+1):
        t = init_time + i*time_delta
        if t <= noon_begin or t > noon_end:
            time_list.append(t.strftime('%H:%M'))

    return time_list

def vwap_corr_stock(df, min_gap=10, type='Close'):
    # tl = time_list_t0(min_gap=min_gap)
    # n_tl = len(tl)
    df = df.sort_values('EndTime')
    n_time = int(len(df)/min_gap)
    v_wap = []
    for i in range(n_time):
        v_wap.append(
            np.dot(df[type][min_gap*i:min_gap*(i+1)], df['LastVolume'][min_gap*i:min_gap*(i+1)]) / np.sum(df['LastVolume'][min_gap*i:min_gap*(i+1)]) if np.sum(df['LastVolume'][min_gap*i:min_gap*(i+1)]) !=0 else np.nan
        )
    v_wap = pd.Series(v_wap)
    v_wap = (v_wap - v_wap.mean()) / (v_wap.std() + 1e-10)

    return v_wap.corr(v_wap.shift(-1))

def vwap_corr_df(df):
    s_list = pd.Series(df.index.unique(), index=df.index.unique())
    return s_list.progress_apply(lambda x:vwap_corr_stock(df.loc[x]))



def t0_corr(data_minute):
    
    tl = time_list_t0(init='09:40', min_gap=15)

    


if __name__ == '__main__':
    # data_minute = DataMinute()
    pd.set_option('display.max_rows', 150)
    root = '/mnt/datadisk/pjluo_shared/minute_h5_data/20230724_new.h5'
    df = pd.read_hdf(root)
    print(vwap_corr_df(df))
    # tl = time_list_t0(init='09:40', min_gap=10)
    # print(tl)
    # print(pd.__version__)