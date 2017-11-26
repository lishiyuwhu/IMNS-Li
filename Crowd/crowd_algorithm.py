#! /usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 17.11.24 14:23
# @Author  : Shiyu Li
# @Software: PyCharm Community Edition

import csv
import numpy as np

def equal_probability_vote(task_run_info):
    # 相同task_id的项, 即同一个任务, info取出现次数最高的一个, 次数相同取靠前的, 存入done_data_set.csv
    # 会丢掉最后一行的数据...等等再用pandas重写吧
    with open(task_run_info) as f, open('done_data_set.csv', 'w') as w :
        reader = csv.DictReader(f)
        fieldnames = ['title', 'info', 'url_b', 'task_id','', 'Unnamed: 0', 'user_id', 'user_ip']
        writer = csv.DictWriter(w, fieldnames=fieldnames)
        writer.writeheader()
        dtask_id = 'NUll'
        infolist = []
        i=0
        for row in reader:
            if  row['task_id'] == dtask_id:
                dtask_id = row['task_id']
                infolist.append(int(row['info']))
                lastrow = row
            else:
                if i==0:
                    dtask_id = row['task_id']
                    infolist.append(int(row['info']))
                    lastrow = row
                    i+=1
                    continue
                infoarray = np.array(infolist)
                count = np.bincount(infoarray)
                info_value = np.argmax(count)
                lastrow['info'] = info_value
                print(lastrow)
                writer.writerow(lastrow)
                dtask_id = row['task_id']
                infolist = []
                infolist.append(int(row['info']))
                lastrow = row


if __name__ == '__main__':
    equal_probability_vote('task_run_info.csv')