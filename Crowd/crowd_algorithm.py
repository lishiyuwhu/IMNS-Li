#! /usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 17.11.24 14:23
# @Author  : Shiyu Li
# @Software: PyCharm Community Edition

import csv
import numpy as np
import pandas as pd


info_max = 0
#
# def equal_probability_vote1(task_run_info):
#     # 相同task_id的项, 即同一个任务, info取出现次数最高的一个, 次数相同取靠前的, 存入done_data_set.csv
#     # 会丢掉最后一行的数据...等等再用pandas重写吧
#     with open(task_run_info) as f, open('done_data_set.csv', 'w') as w :
#         reader = csv.DictReader(f)
#         fieldnames = ['title', 'info', 'url_b', 'task_id','', 'Unnamed: 0', 'user_id', 'user_ip']
#         writer = csv.DictWriter(w, fieldnames=fieldnames)
#         writer.writeheader()
#         dtask_id = 'NUll'
#         infolist = []
#         i=0
#         for row in reader:
#             if  row['task_id'] == dtask_id:
#                 dtask_id = row['task_id']
#                 infolist.append(int(row['info']))
#                 lastrow = row
#             else:
#                 if i==0:
#                     dtask_id = row['task_id']
#                     infolist.append(int(row['info']))
#                     lastrow = row
#                     i+=1
#                     continue
#                 infoarray = np.array(infolist)
#                 count = np.bincount(infoarray)
#                 info_value = np.argmax(count)
#                 lastrow['info'] = info_value
#                 print(lastrow)
#                 writer.writerow(lastrow)
#                 dtask_id = row['task_id']
#                 infolist = []
#                 infolist.append(int(row['info']))
#                 lastrow = row

def conv_sparse(num):
    #类别是从0开始的
    global  info_max
    s_list =[0 for n in range(info_max + 1)]
    s_list[num]=1
    s_list = np.array(s_list)
    return s_list


def equal_probability_vote(task_run_info):
    # 返回task_id和info的对应关系, 写入taskid_info.csv
    global info_max
    data = pd.read_csv(task_run_info)
    #data = data.tail(5)
    # 添加sparse info:  2->[0,0,1,0,0,...]
    info_max = data['info'].max(axis=0)
    data['sparse_info'] = data['info'].map(conv_sparse)

    datacount = data['task_id'].value_counts()

    tasklist = data['task_id'].drop_duplicates()
    tasklist = tasklist.tolist() 
    data = data.set_index('task_id')

    df_output = pd.DataFrame(columns = ['info']) #创建一个空的dataframe

    for task in tasklist:
        task_frame = data.loc[task,['task_id', 'info', 'sparse_info']]

        if  datacount[int(task)] != 1:# 要判断下是否只有一个, 否则.mean()会出问题
            task_info = task_frame['sparse_info'].mean()
            task_info = np.argmax(task_info)
            task_incert = task_frame.head(1)
            task_incert['info'] = str(task_info)
        else:
            task_incert = task_frame

        df_output = df_output.append(task_incert)
    df_output = df_output['info']

    df_output.to_csv('taskid_info.csv', index='task_id')
    return df_output


if __name__ == '__main__':
    print(equal_probability_vote('task_run_info.csv'))