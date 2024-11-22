import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification
from collections import Counter
from pmlb import fetch_data
import random

import os

import socket
import struct
import gc
import copy

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

dirname = os.path.split(os.path.split(os.path.dirname(__file__))[0])[0]
data_path = dirname + "/data/"

class DataLoader():
    def __init__(self, seeds=None):
        self.seeds = [random.randint(0, 2**30)] if seeds is None else seeds

    def _load_synthetic(self, data_dict, seed, test_size, verbose=True, num_samples=100000, num_features=500, informative=5, redundant=2, repeated=2, weights=None):
        name = "synthetic_" + str(num_samples) + "_" + str(num_features) + "_" + str(informative) + "_" + str(redundant) + "_" + str(repeated) + "_" + str(weights)
        X,y = make_classification(round(num_samples/test_size), num_features, n_informative=informative, n_redundant=redundant, n_repeated=repeated, weights=weights, shuffle=True)

        if verbose:
            self.print_data_info(name, X, y)

        data = train_test_split(X, y, test_size=test_size, random_state=seed, stratify=y)
        data_dict[name+"_"+str(seed)] = data

    def print_data_info(self, name, X, y):
        print(name, "shape:", X.shape)
        print(pd.DataFrame(X).corrwith(pd.Series(y)))
        print((pd.DataFrame(X).corrwith(pd.Series(y)).abs() >= 0.5).sum()/X.shape[1])
        print(np.array(list(Counter(y).values()))/len(y), "\n")


    # CICIDS
    def _load_cic(self, data_path, data_dict, seed, test_size, remove_missing, verbose):
        # data = pd.read_csv("/home/p4/FL-PCAP-Analysis/local_approach/FL_working/fede_p4/cic_20000_split2_org.csv")
        # data = pd.read_csv("/home/p4/FL-PCAP-Analysis/local_approach/FL_working/fede_p4/cic_20000_split2_org.csv")
        data = pd.read_csv(data_path)
        label_index = ' Label'
        num_features = 5
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

        for key in range(len(data[label_index].values)):
            if data[label_index].values[key]=='BENIGN':
                data[label_index].values[key] = 0
            else:
                data[label_index].values[key] = 1
        #Replace values with NaN, inf, -inf
        data.replace([np.inf, -np.inf], np.nan)
        #Remove rows containing NaN
        data.dropna(how="any", inplace = True)
        data = data[data.replace([np.inf, -np.inf], np.nan).notnull().all(axis=1)]
        data.describe()
        data.info()
        used_features = [' Source Port', ' Destination Port', ' Protocol', 'srcip_part_1', 'dstip_part_4'][:num_features]
        # used_features = [' Source Port', ' Destination Port', ' Protocol', ' Source IP', ' Destination IP'][:num_features]


        X = copy.deepcopy(data[used_features].astype("int"))
        y = copy.deepcopy(data[label_index].astype("int"))
        del data


        cic_data = train_test_split(X, y, test_size=test_size, stratify=y, random_state=seed, shuffle=True)
        cic_counter = Counter(y)
        print("[Data Loader] CICIDS Class Balance:", cic_counter[1] / sum(cic_counter.values()), "\n")

        if verbose:
                self.print_data_info("CIC", X, y)

        data_dict["CIC"+"_"+str(seed)] = cic_data

    def load_datasets(self, data_path, data_list, remove_missing=False, return_dict=False, verbose=False):
        test_size = 0.3
        data_dict = {}
        new_names = []
        for dataset_name in data_list:
            for seed in self.seeds:
                new_names.append(f"{dataset_name}_{str(seed)}")

                if "synthetic" in dataset_name:
                    n = int(dataset_name.split("n=")[1].split("_")[0])
                    m = int(dataset_name.split("m=")[1].split("_")[0])
                    informative = int(dataset_name.split("informative=")[1].split("_")[0])
                    self._load_synthetic(data_dict, seed, test_size, num_samples=n, num_features=m, informative=informative)
                # CIC
                if dataset_name == "CIC":
                    self._load_cic(data_path, data_dict, seed, test_size, remove_missing, verbose)

        if return_dict:
            return data_dict
        else:
            return [data_dict[dataset] for dataset in new_names]
