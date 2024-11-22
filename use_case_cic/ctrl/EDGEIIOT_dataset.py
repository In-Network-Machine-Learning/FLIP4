########################################################################
# THIS FILE IS PART OF Planter PROJECT
# Copyright (c) Changgang Zheng and Computing Infrastructure Lab
# Departement of Engineering Science, University of Oxford
# All rights reserved.
# E-mail: changgang.zheng@eng.ox.ac.uk or changgangzheng@qq.com
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at :
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#########################################################################
import copy

import numpy as np
import pandas as pd
import os
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import socket
import struct
import gc

def ip2int(addr):
    return struct.unpack("!I", socket.inet_aton(addr))[0]

def ip2long(ip):
    """
    Convert an IP string to long
    """
    packedIP = socket.inet_aton(ip)
    return struct.unpack("!L", packedIP)[0]

# Convert IP to bin
def ip2bin(ip):
    ip1 = '.'.join([bin(int(x)+256)[3:] for x in ip.split('.')])
    return ip1

# Convert IP to hex
def ip2hex(ip):
    ip1 = '-'.join([hex(int(x)+256)[3:] for x in ip.split('.')])
    return ip1

def bin2dec(ip):
    return int(ip,2)

def load_data(num_features, data_dir):


    label_index = 'Attack_label'
    # normal_label = 'BENIGN'

    file_dir = data_dir+'/EDGEIIOT/'

    files = []

    for dirpath, dirnames, filenames in os.walk(file_dir):
        for file in filenames:
          # print(file.split("-")[0] == 'Friday')
          # if file.split(".")[1] == 'pcap_ISCX' and file.split("-")[0] == 'Friday':
          files.append(os.path.join(dirpath, file))

    for i,file in enumerate (files):
        print('read data from file: '+file+' ...')
        dataset = pd.read_csv(file)
        if  i==0:
            data = pd.DataFrame(dataset)
            # break
        else:
            new_data = pd.DataFrame(dataset)
            data  = pd.concat([data,new_data],axis=0)
    # for key in range(len(data[label_index].values)):
    #     if data[label_index].values[key]=='BENIGN':
    #         data[label_index].values[key] = 0
    #     else:
    #         data[label_index].values[key] = 1


        # percent = np.int(np.ceil(50*key/len(data[label_index].values)))
        # if key%10==0:
        #     print('\rProcessing the raw Data ['+percent*'#'+(50-percent)*'-'+'] '+str(int(np.round(100*key/len(data[label_index].values))))+"%",end="")



    data['tcp.flags'] = data['tcp.flags'].str.replace('0x000000', '')
    print(data['tcp.flags'].value_counts())
    #Replace values with NaN, inf, -inf
    # data.replace([np.inf, -np.inf], np.nan)
    data = data.fillna('0')
    data[['srcip_part_1', 'srcip_part_2', 'srcip_part_3', 'srcip_part_4']] = data['ip.src_host'].apply(ip2bin).str.split('.',expand=True)
    data[['dstip_part_1', 'dstip_part_2', 'dstip_part_3', 'dstip_part_4']] = data['ip.dst_host'].apply(ip2bin).str.split('.',expand=True)
    data = data.fillna('0')
    data['srcip_part_1'] = data['srcip_part_1'].apply(bin2dec)
    data['srcip_part_2'] = data['srcip_part_2'].apply(bin2dec)
    data['srcip_part_3'] = data['srcip_part_3'].apply(bin2dec)
    data['srcip_part_4'] = data['srcip_part_4'].apply(bin2dec)
    data['dstip_part_1'] = data['dstip_part_1'].apply(bin2dec)
    data['dstip_part_2'] = data['dstip_part_2'].apply(bin2dec)
    data['dstip_part_3'] = data['dstip_part_3'].apply(bin2dec)
    data['dstip_part_4'] = data['dstip_part_4'].apply(bin2dec)

    data['ip.src_host'] = data['ip.src_host'].apply(ip2long)
    data['ip.dst_host'] = data['ip.dst_host'].apply(ip2long)
    print('data.head: ', data.head())

    print('')
    # data.replace([np.inf, -np.inf], np.nan)
    # #Remove rows containing NaN
    # data.dropna(how="any", inplace = True)
    # data = data[data.replace([np.inf, -np.inf], np.nan).notnull().all(axis=1)]
    # # data = data[data.replace([np.inf, -np.inf], np.nan).notnull().all(axis=1)]
    # data.describe()
    data.info()
    print(data[label_index].value_counts())

    #used_features = [' Source IP', ' Source Port', ' Destination IP',
    #       ' Destination Port', ' Protocol', 'srcip_part_1', 'srcip_part_2', 'srcip_part_3', 'srcip_part_4', 'dstip_part_1', 'dstip_part_2', 'dstip_part_3', 'dstip_part_4'][:num_features]
    #used_features = [' Source IP', ' Source Port', ' Destination IP',
    #       ' Destination Port', ' Protocol'][:num_features]
    used_features = ['dstip_part_4','srcip_part_1', 'tcp.dstport','tcp.srcport', 'tcp.flags'][:num_features]
    select_features = ['dstip_part_4','srcip_part_1', 'tcp.dstport','tcp.srcport', 'tcp.flags', label_index]
    print('select_features: ', select_features)

    if os.path.exists(data_dir+'/EDGEIIOT/logging/logging_misclass.csv'):
        update_data = pd.read_csv(file_dir+'logging/logging_misclass.csv')
        print(update_data.head())
        data1 = data[select_features]
        data2 = update_data[select_features]
        new_data = pd.concat([data1, data2],axis=0)
        new_data = new_data[new_data.replace([np.inf, -np.inf], np.nan).notnull().all(axis=1)]
        print(new_data.info())
        print(new_data[label_index].value_counts())
        # update_data = new_data.to_csv(file_dir+'logging/debugt.csv')
    else:
        new_data = data


    # used_features = ['ip.dst_host','ip.src_host', 'tcp.dstport','tcp.srcport', 'tcp.flags'][:num_features]
    # select_features = ['ip.dst_host','ip.src_host', 'tcp.dstport','tcp.srcport', 'tcp.flags', label_index]
    # used_features = ['tcp.ack','tcp.dstport','tcp.srcport', 'tcp.seq', 'tcp.flags'][:num_features]
    # select_features = ['tcp.ack','tcp.dstport','tcp.srcport', 'tcp.seq', 'tcp.flags', label_index]

    # X = pd.concat([data[used_features],update_data[used_features]],axis=0)
    # y = pd.concat([data[label_index],update_data[label_index]],axis=0)
    # X = copy.deepcopy(X.astype("int"))
    # y = copy.deepcopy(y.astype("int"))

    X = copy.deepcopy(new_data[used_features].astype("int"))
    y = copy.deepcopy(new_data[label_index].astype("int"))
    # print(X['tcp.flags'].value_counts())
    # print(y[label_index].value_counts())
    # df = pd.concat([X, y], axis = 1)
    # df.to_csv('training_data.csv')
    del new_data
    gc.collect()
    # encoder = LabelEncoder()
    # y = encoder.fit_transform(y)


    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42, shuffle=True)

    print('dataset is loaded')

    return X_train, np.array(y_train), X_test, np.array(y_test), used_features
