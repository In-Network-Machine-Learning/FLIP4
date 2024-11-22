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

import argparse

# to slice the dataset, run $ python3 CICIDS_dataset.py --num_features 5 --dataset /home/p4/Data/CICIDS/ --chunk_size 30000 --num_splits 5

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

def load_data(num_features, data_dir, chunk_size, NUMBER_OF_SPLITS):


    label_index = ' Label'
    # normal_label = 'BENIGN'

    # file_dir = data_dir+'/CICIDS/Data'
    file_dir = data_dir

    files = []

    for dirpath, dirnames, filenames in os.walk(file_dir):
        for file in filenames:
            if os.path.splitext(file)[1] == '.csv':
                files.append(os.path.join(dirpath, file))
    print('DEBUG files: ', files)
    for i,file in enumerate (files):

        print('read data from file: '+file+' ...')
        dataset = pd.read_csv(file)
        if  i==0:
            data = pd.DataFrame(dataset)
            # break
        else:
            new_data = pd.DataFrame(dataset)
            data  = pd.concat([data,new_data],axis=0)

    data[['srcip_part_1', 'srcip_part_2', 'srcip_part_3', 'srcip_part_4']] = data[' Source IP'].apply(ip2bin).str.split('.',expand=True)
    data[['dstip_part_1', 'dstip_part_2', 'dstip_part_3', 'dstip_part_4']] = data[' Destination IP'].apply(ip2bin).str.split('.',expand=True)

    data['srcip_part_1'] = data['srcip_part_1'].apply(bin2dec)
    data['srcip_part_2'] = data['srcip_part_2'].apply(bin2dec)
    data['srcip_part_3'] = data['srcip_part_3'].apply(bin2dec)
    data['srcip_part_4'] = data['srcip_part_4'].apply(bin2dec)
    data['dstip_part_1'] = data['dstip_part_1'].apply(bin2dec)
    data['dstip_part_2'] = data['dstip_part_2'].apply(bin2dec)
    data['dstip_part_3'] = data['dstip_part_3'].apply(bin2dec)
    data['dstip_part_4'] = data['dstip_part_4'].apply(bin2dec)

    data[' Source IP'] = data[' Source IP'].apply(ip2long)
    data[' Destination IP'] = data[' Destination IP'].apply(ip2long)
    print('data.head: ', data.head())


    # for key in range(len(data[label_index].values)):
    #     if data[label_index].values[key]=='BENIGN':
    #         data[label_index].values[key] = 0
    #     else:
    #         data[label_index].values[key] = 1
    #     data[' Source IP'].values[key] = ip2int(data[' Source IP'].values[key])
    #     data[' Destination IP'].values[key] = ip2int(data[' Destination IP'].values[key])

    # data.iloc[0:20000].to_csv("cic_20000_split1_org.csv",index=False)
    # data.iloc[20001:40000].to_csv("cic_20000_split2_org.csv",index=False)
    # data.iloc[40001:60000].to_csv("cic_20000_split3_org.csv",index=False)
    # data.iloc[60001:80000].to_csv("cic_20000_split4_org.csv",index=False)


    # data.iloc[0:30000].to_csv("cic_30000_split1_org_bot.csv",index=False)
    # data.iloc[30001:60000].to_csv("cic_30000_split2_org_bot.csv",index=False)
    # data.iloc[60001:90000].to_csv("cic_30000_split3_org_bot.csv",index=False)
    # # data.iloc[0:30000].to_csv("cic_30000_split1_org_bot.csv",index=False)
    # # data.iloc[30001:60000].to_csv("cic_30000_split2_org_bot.csv",index=False)
    # # data.iloc[60001:90000].to_csv("cic_30000_split3_org_bot.csv",index=False)



    print('')
    data.replace([np.inf, -np.inf], np.nan)
    #Remove rows containing NaN
    data.dropna(how="any", inplace = True)
    data = data[data.replace([np.inf, -np.inf], np.nan).notnull().all(axis=1)]
    # data = data[data.replace([np.inf, -np.inf], np.nan).notnull().all(axis=1)]
    data.describe()
    data.info()
    print(data[label_index].value_counts())

   


    # used_features = ['headerLen', 'srcIP', 'dstIP', 'IPProtocol', 'srcPort', 'dstPort', 'FIN_flag',
    #         'SYN_flag', 'RST_flag', 'PSH_flag', 'ACK_flag', 'URG_flag'][:num_features]

    # used_features = [' Source IP', ' Source Port', ' Destination IP',
    #        ' Destination Port', ' Protocol',' Fwd Packet Length Max', ' Fwd Packet Length Min',
    #        ' Fwd Packet Length Mean', ' SYN Flag Count', ' PSH Flag Count', ][:num_features]
    # used_features = [' Protocol',' Source Port', ' Destination Port', ' SYN Flag Count', ' ACK Flag Count'][:num_features]
    used_features = [' Flow Duration', ' Flow IAT Mean', ' Flow IAT Max', ' Flow IAT Min', ' Min Packet Length']


    # X = copy.deepcopy(data[used_features].astype("int"))
    X = copy.deepcopy(data[used_features])
    # y = copy.deepcopy(data[label_index].astype("int"))
    y = copy.deepcopy(data[label_index])
    
    #data = data[used_features]
     # iterate number of splits times and write each chunk to a separate CSV file
    for i in range(NUMBER_OF_SPLITS):
        start_index = i * chunk_size
        end_index = (i + 1) * chunk_size
        chunk = data.iloc[start_index:end_index]
        chunk.to_csv(f"./CIC/cic_" + str(chunk_size) + "_split_"+str(i+1)+"_flow_dos.csv", index=False)

    del data
    gc.collect()
    # encoder = LabelEncoder()
    # y = encoder.fit_transform(y)


    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42, shuffle=True)

    print('dataset is loaded')

    return X_train, np.array(y_train), X_test, np.array(y_test), used_features


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Dataset preprocessing.')
    parser.add_argument('--num_features', dest='num_features' , type=int, help='NUmber of features')
    parser.add_argument('--dataset', dest='data_dir' , type=str, help='dataset dir')
    parser.add_argument('--chunk_size', dest='chunk_size' , type=int, help='chunk size')
    parser.add_argument('--num_splits', dest='NUMBER_OF_SPLITS' , type=int, help='NUMBER OF SPLITS')

    args = parser.parse_args()
    load_data(args.num_features, args.data_dir, args.chunk_size, args.NUMBER_OF_SPLITS)
