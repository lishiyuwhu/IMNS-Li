#! /usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 17.11.24 13:19
# @Author  : Shiyu Li
# @Software: PyCharm Community Edition

import pandas as pd

def task_run_csv_merge(task_info_filename, run_info_filename):
    # project-name_task_info_only.csv project-name_task_run.csv
    # csv_read('whu-imns-mnist-test_task_info_only.csv', 'whu-imns-mnist-test_task_run.csv')

    # 以'task_id'为主键, 合并两个csv
    dtask = pd.read_csv(task_info_filename)
    drun = pd.read_csv(run_info_filename)
    data = pd.merge(dtask, drun, on=['task_id'], how='left')
    data = data[['task_id', 'info', 'title', 'url_b', 'user_id', 'user_ip']]

    #print(data)
    data.to_csv(r'task_run_info.csv')#, encoding='gbk')

if __name__ == '__main__':
    iiii=0