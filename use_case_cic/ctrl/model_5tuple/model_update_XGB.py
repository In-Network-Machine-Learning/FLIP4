# THIS FILE IS PART OF Planter PROJECT
# Planter.py - The core part of the Planter library
#
# THIS PROGRAM IS FREE SOFTWARE TOOL, WHICH MAPS MACHINE LEARNING ALGORITHMS TO DATA PLANE, IS LICENSED UNDER Apache-2.0
# YOU SHOULD HAVE RECEIVED A COPY OF THE LICENSE, IF NOT, PLEASE CONTACT THE FOLLOWING E-MAIL ADDRESSES
#
# Copyright (c) 2020-2021 Changgang Zheng
# Copyright (c) Computing Infrastructure Lab, Departement of Engineering Science, University of Oxford
# E-mail: changgang.zheng@eng.ox.ac.uk or changgangzheng@qq.com
# Modified by Mingyuan Zang

import numpy as np
import pandas as pd
from pandas import Series,DataFrame
from pandas import plotting
import os

# %matplotlib inline
import matplotlib.pyplot as plt
plt.style.use('seaborn')

import seaborn as sns
sns.set_style("whitegrid")

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn import metrics
from sklearn.tree import _tree
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from IPython.display import Image
import pydotplus
from sklearn.metrics import classification_report
import xgboost as xgb
import copy
from ..Range_to_TCAM_Top_Down import *
import math
import time
import re
import json

from sklearn.metrics import *
from ..EDGEIIOT_dataset import load_data

def map(value):
    value = value
    return value

def get_path(model, conditions, path, num, leaf_info, tree_index):
    if 'children' in model.keys():
        conditions_yes = copy.deepcopy(conditions)
        conditions_no = copy.deepcopy(conditions)
        if conditions_yes[model["split"]][1] > map(model["split_condition"])-1:
            conditions_yes[model["split"]][1] = map(model["split_condition"])-1
        if conditions_no[model["split"]][0] < map(model["split_condition"]) :
            conditions_no[model["split"]][0] = map(model["split_condition"])
        for child_model in model["children"]:
            if child_model["nodeid"]==model["yes"]:
                path, num, leaf_info = get_path(child_model, conditions_yes, path, num, leaf_info, tree_index)
            if child_model["nodeid"]==model["no"]:
                path, num, leaf_info = get_path(child_model, conditions_no, path, num, leaf_info, tree_index)
    else:
        # print(path, conditions)
        path['path '+str(num)] = conditions
        path['path '+str(num)]['leaf'] = model["leaf"]
        # leaf_info['tree '+str(tree_index)] += [model["leaf"]]
        leaf_info['tree ' + str(tree_index)] += [round(model["leaf"], 1)]
        if model["leaf"] > leaf_info['max value']: leaf_info['max value'] = model["leaf"]
        elif model["leaf"] < leaf_info['min value']: leaf_info['min value'] = model["leaf"]
        num += 1
    return path, num, leaf_info


def find_feature_split(model, tree_index, num_features, feature_names):
    count_layer = 0
    count_route = 0
    count_list = 0
    layer = {}
    route = {}
    layer[count_layer] = {}
    layer[count_layer][count_list] = {}
    layer[count_layer][count_list]["lst"] = [0]
    layer[count_layer][count_list]["tab"] = model
    feature_split = {}
    num_features = len(feature_names)

    for i in range(num_features):
        feature_split["feature " + str(i)] = []
    while True:
        if len(layer[count_layer].keys()) == 0:
            break
        layer[count_layer + 1] = {}
        count_list = 0
        for list_id in layer[count_layer]:
            feature_split["feature " + str(feature_names.index(layer[count_layer][list_id]["tab"]["split"]))] += [
                layer[count_layer][list_id]["tab"]["split_condition"]]
            # (optional add -1)The -1 means the feature splits is for <= =, so each split is largest value in each range

            for i, children in enumerate(layer[count_layer][list_id]["tab"]["children"]):
                if "children" not in children.keys():
                    route[count_route] = layer[count_layer][list_id]["lst"] + [children["nodeid"]]
                    count_route += 1
                else:
                    layer[count_layer + 1][count_list] = {}
                    layer[count_layer + 1][count_list]["lst"] = layer[count_layer][list_id]["lst"] + [
                        children["nodeid"]]
                    layer[count_layer + 1][count_list]["tab"] = children
                    count_list += 1
        count_layer += 1
    for f in range(num_features):
        feature_split['feature ' + str(f)] = sorted(list(set(feature_split['feature ' + str(f)])))
    return feature_split

def generate_feature_tables(split, num_faetures,feature_max, table):
    for i in range(num_faetures):
        table["feature "+str(i)] = {}
        count_code = 0
        nife = sorted(split["feature "+str(i)])
        for j in range(feature_max[i]+1):
            if nife !=[] :
                if len(nife) > count_code:
                    if j == nife[count_code]:
                        count_code+=1
            table["feature " + str(i)][j] = count_code
    return table

def path_to_path_to_leaf(path, num_features, table, leaf_code_list):
    path_to_leaf ={}
    for p in path:
        path_to_leaf[p] = {}
        # path_to_leaf[p]['leaf'] = path[p]['leaf']
        path_to_leaf[p]['leaf'] = leaf_code_list.index(round(path[p]['leaf'], 1))
        # path_to_leaf[p]['leaf'] = leaf_code_list.index(path[p]['leaf'])
        for f in range(num_features):
            ini = table['feature '+str(f)][path[p]['f'+str(f)][0]]
            end = table['feature '+str(f)][path[p]['f'+str(f)][1]]
            path_to_leaf[p]['feature '+str(f)] = np.arange(ini,end+1).tolist()
    return path_to_leaf


def find_path_for_leaf_nodes(model, feature_split, feature_max, num_features, table, leaf_info, tree_index):
    conditions = {}
    for i in range(num_features):
        conditions["f" + str(i)] = [0, feature_max[i]]
        feature_split["feature " + str(i)] += [feature_max[i]]

    path = {}
    path, _, leaf_info = get_path(model, conditions, path, 0, leaf_info, tree_index)
    leaf_info['tree '+str(tree_index)] = sorted(list(set(leaf_info['tree '+str(tree_index)])))
    path_to_leaf = path_to_path_to_leaf(path, num_features, table, leaf_info['tree '+str(tree_index)] )
    return path_to_leaf, leaf_info

def generate_code_table_for_path(table, leaf_path, code_dict, feature_num, num_features, count):
    if feature_num == num_features:
        table['code to vote'][count] = {}
        for f in range(num_features):
            table['code to vote'][count]['f'+str(f)+' code'] = code_dict['feature ' + str(f)]
        table['code to vote'][count]['leaf'] = leaf_path['leaf']
        count += 1
        return table, count
    else:
        for value in leaf_path['feature '+str(feature_num)]:
            code_dict['feature ' + str(feature_num)] = value
            feature_num += 1
            table, count = generate_code_table_for_path(table, leaf_path, code_dict, feature_num, num_features, count)
            feature_num -= 1
    return table, count

def generate_code_table(table, path_to_leaf, num_features):
    table['code to vote'] = {}
    count = 0
    for p in path_to_leaf:
        table, count = generate_code_table_for_path(table, path_to_leaf[p], {}, 0, num_features, count)
    return table

def generate_table(model,tree_index, g_table, num_features, feature_names, feature_max, leaf_info):

    feature_split = find_feature_split(model, tree_index, num_features, feature_names)
    g_table[tree_index] = {}
    g_table[tree_index] = generate_feature_tables(feature_split, num_features, feature_max, g_table[tree_index])
    leaf_info['tree '+str(tree_index)] = []
    path_to_leaf, leaf_info = find_path_for_leaf_nodes(model, feature_split, feature_max, num_features, g_table[tree_index], leaf_info, tree_index)

    code_width_for_feature = np.zeros(num_features)
    for i in range(num_features):
        code_width_for_feature[i] = np.ceil(math.log(
            g_table[tree_index]['feature ' + str(i)][np.max(list(g_table[tree_index]['feature ' + str(i)].keys()))] + 1, 2))
    g_table[tree_index] = generate_code_table(g_table[tree_index], path_to_leaf, num_features)
    print('\rThe table for Tree: {} is generated'.format(tree_index), end="")
    return g_table, leaf_info


def ten_to_bin(num,count):
    num = bin(int(num)).lstrip('0b')

    if len(num) != count:
        cont = count - len(num)
        num = cont * '0' + num
    return num

def MaxMin_Norm_with_range(x, min , max, ranges = 10):
    """[0,1] normaliaztion"""
    x = (x - min) / (max - min)
    return np.floor(ranges*x)

def create_tables_command(fname, dir_table, num_trees, num_classes, num_features, code_width_tree_feature):
    # num_features = config['data config']['number of features']
    # num_classes = config['model config']['number of classes']
    # num_trees = config['model config']['number of trees']
    Ternary_Table = json.load(open(dir_table, 'r'))
    with open(fname, 'w') as file:
        for f in range(num_features):
            for idx in Ternary_Table['feature ' + str(f)]:
                priority = int(idx)
                key = Ternary_Table['feature ' + str(f)][idx][1]
                mask = Ternary_Table['feature ' + str(f)][idx][0]
                codes = ''
                for t in range(num_trees):
                    c_tree = Ternary_Table['feature ' + str(f)][idx][2][t]
                    c_len = config['p4 config']['width of code'][t][f]
                    codes = ten_to_bin(int(c_tree), int(c_len)) + codes
                label = int(codes, 2)
                file.write("table_add SwitchIngress.lookup_feature" + str(f)+" extract_feature" + str(f)+
                           " "+str(key)+"&&&"+str(mask)+" => "+str(label)+" "+str(priority)+"\n")
            file.write("\n")


        for t in range(num_trees):
            for idx in Ternary_Table['tree ' + str(t)]:
                file.write("table_add SwitchIngress.lookup_leaf_id" + str(t) + " read_prob" + str(t) + " ")
                for f in range(num_features):
                    file.write(str(Ternary_Table['tree ' + str(t)][idx]['f' + str(f) + ' code']) + " ")
                file.write("=> 0 " + str(Ternary_Table['tree ' + str(t)][idx]['leaf']) + "\n")
            file.write("\n")

        for idx in Ternary_Table['decision']:
            file.write("table_add SwitchIngress.decision read_lable ")
            for t in range(num_trees):
                file.write(str(Ternary_Table['decision'][idx]['t' + str(t) + ' vote'])+" ")
            file.write("=> "+str(Ternary_Table['decision'][idx]['class'])+"\n")


def run_model(train_X, train_y, test_X, test_y, used_features, dir_exact_table, dir_ternary_table, tree_table_file):
    # config_file = 'src/configs/Planter_config.json'
    #
    # Planter_config = json.load(open(config_file, 'r'))
    num_features = len(used_features)
    num_trees = int(input('- Number of trees? (defalt = 6) ') or '6')
    num_depth = int(input('- Number of depth? (defalt = 4) ') or '4')
    max_leaf_nodes = int(input('- Number of leaf nodes? (defalt = 1000) ') or '1000')
    num_classes = int(np.max(train_y) + 1)
    num_boost_rounds = int(num_trees/num_classes)
    test_mode = str(input('- Use the testing mode or not? (default = y) ') or 'y')
    if test_mode == 'y':
        if train_X.shape[0]>2000:
            train_X = train_X[:2000]
            train_y = train_y[:2000]
        if test_X.shape[0]>600:
            test_X = test_X[:600]
            test_y = test_y[:600]

    feature_names = []
    for i, f in enumerate(used_features):
        train_X.rename(columns={f: "f"+str(i)}, inplace=True)
        test_X.rename(columns={f: "f" + str(i)}, inplace=True)
        feature_names+=["f"+str(i)]

    feature_max = []
    for i in feature_names:
        t_t = [test_X[[i]].max()[0], train_X[[i]].max()[0]]
        feature_max += [np.max(t_t)+1]

    # # =================== train model timer ===================
    # Planter_config['timer log']['train model'] = {}
    # Planter_config['timer log']['train model']['start'] = time.time()
    # # =================== train model timer ===================

    # XGBoost
    data_train = xgb.DMatrix(train_X, label=train_y)
    data_test = xgb.DMatrix(test_X, label=test_y)
    watchlist = [(data_test, 'eval'), (data_train, 'train')]
    param = {'max_depth': num_depth, 'eta': 1, 'silent': 0, 'objective': 'multi:softmax', 'num_class': num_classes}
    bst = xgb.train(param, data_train, num_boost_round=num_boost_rounds, evals=watchlist)

    # param = {'max_depth': 8, 'num_class': 2}
    # bst = xgb.train(param, data_train, num_boost_round=200, evals=watchlist)
    bst.dump_model("temp_tree.txt")
    sklearn_y_predict = bst.predict(data_test)

    result = classification_report(test_y, sklearn_y_predict)
    # exit()
    result = classification_report(test_y, sklearn_y_predict, digits=4)
    print('\n', result)

    # # =================== train model timer ===================
    # Planter_config['timer log']['train model']['end'] = time.time()
    # # =================== train model timer ===================
    #
    # # =================== convert model timer ===================
    # Planter_config['timer log']['convert model'] = {}
    # Planter_config['timer log']['convert model']['start'] = time.time()
    # # =================== convert model timer ===================

    # log_file = 'src/logs/log.json'
    # if os.path.exists(log_file):
    #     log_dict = json.load(open(log_file, 'r'))
    # else:
    #     log_dict = {}
    #
    # if ("num_feature: " + str(num_features)) not in log_dict:
    #     log_dict["num_feature: " + str(num_features)] = {}
    # if ("num_tree: " + str(num_trees)) not in log_dict["num_feature: " + str(num_features)]:
    #     log_dict["num_feature: " + str(num_features)]["num_tree: " + str(num_trees)] = {}
    # if ("num_depth: " + str(num_depth)) not in log_dict["num_feature: " + str(num_features)][
    #     "num_tree: " + str(num_trees)]:
    #     log_dict["num_feature: " + str(num_features)]["num_tree: " + str(num_trees)][
    #         "num_depth: " + str(num_depth)] = {}
    # log_dict["num_feature: " + str(num_features)]["num_tree: " + str(num_trees)]["num_depth: " + str(num_depth)][
    #     "classification_report"] = result
    # log_dict["num_feature: " + str(num_features)]["num_tree: " + str(num_trees)]["num_depth: " + str(num_depth)][
    #     "max number of leaf nodes"] = max_leaf_nodes
    # json.dump(log_dict, open(log_file, 'w'), indent=4)
    # print('Classification results are downloaded to log as', log_file)
    #
    #
    the_model= bst.get_dump(fmap="", with_stats=False, dump_format="json")
    xgb_model = {}
    for i, m in enumerate(the_model):
        xgb_model[i] = json.loads(m)

    g_table = {}
    # feature_names = test_X.columns.T.tolist()
    leaf_info ={}
    leaf_info['max value'] = 0
    leaf_info['min value'] = 0
    for idx in xgb_model:
        estimator = xgb_model[idx]
        g_table, leaf_info = generate_table(estimator, idx, g_table, num_features, feature_names, feature_max, leaf_info)

    def votes_to_class(tree_num, vote_list, num_trees, num_classes, g_table, num, leaf_info):
        print('tree_num ', tree_num)
        print('num_trees ', num_trees)
        if tree_num  == num_trees:
            vote = np.zeros(num_classes).tolist()
            for t in range(num_trees):
                vote[t%num_classes] += leaf_info["tree "+str(t)][vote_list[t]]
            # if vote.index(np.max(vote))== 0:
            # if True :
            g_table['votes to class'][num] = {}
            for t in range(len(vote_list)):
                g_table['votes to class'][num]['t'+str(t)+' vote'] = vote_list[t]
            g_table['votes to class'][num]['class'] = vote.index(np.max(vote))
            num += 1
            return g_table, num
        else:
            print('range(len(leaf_info["tree "+str(tree_num)]))', range(len(leaf_info["tree "+str(tree_num)])))
            for value in range(len(leaf_info["tree "+str(tree_num)])):
                vote_list[tree_num] = value
                tree_num += 1
                g_table, num = votes_to_class(tree_num, vote_list, num_trees, num_classes, g_table, num, leaf_info)
                tree_num -= 1
        return g_table, num


    ranges = 10
    g_table['votes to class'] = {}
    print("\nGenerating vote to class table...",end="")
    g_table, _ = votes_to_class(0, np.zeros(num_trees).tolist(), num_trees, num_classes, g_table, 0, leaf_info)
    print('Done')

    feature_width = []
    for maxs in feature_max:
        feature_width += [int(np.ceil(math.log(maxs, 2)) + 1)]


    code_width_tree_feature = np.zeros((num_trees,num_features))
    for i in range(num_features):
        for tree in range(num_trees):
            # code_width_tree_feature[tree, i] = np.ceil(math.log(g_table[tree]['feature ' + str(i)][feature_max[i]],2))
            code_width_tree_feature[tree, i] = np.ceil(math.log(g_table[tree]['feature ' + str(i)][np.max(list(g_table[tree]['feature ' + str(i)].keys()))]+1,2)+1)
            # print(code_width_tree_feature[tree, i] , g_table[tree]['feature ' + str(i)][feature_max[i]])
            # print('stop')


    Ternary_Table = {}
    Ternary_Table['decision'] = g_table['votes to class']

    for tree in range(num_trees):
        Ternary_Table['tree ' + str(tree)] = g_table[tree]['code to vote']

    for i in range(num_features):
        Ternary_Table['feature '+str(i)] = {}
        for value in range(feature_max[i]):
            Ternary_Table['feature ' + str(i)][value] = []
            for tree in range(num_trees):
                Ternary_Table['feature ' + str(i)][value] += [g_table[tree]["feature "+str(i)][value]]
    Exact_Table = copy.deepcopy(Ternary_Table)
    for i in range(num_features):
        if i!=0:
            print('')
        print('Begine transfer: Feature table ' + str(i))
        Ternary_Table['feature '+str(i)]= Table_to_TCAM(Ternary_Table['feature '+str(i)], feature_width[i])
        # print("\n")

    # ===================== prepare default vote =========================
    print("\nPreparing default vote...", end="")
    collect_votes = []
    for t in range(num_trees):
        for idx in Exact_Table['tree ' + str(t)]:
            collect_votes += [int(Exact_Table['tree ' + str(t)][idx]['leaf'])]
    default_vote = max(collect_votes, key=collect_votes.count)

    code_table_size = 0
    for t in range(num_trees):
        Ternary_Table['tree ' + str(t)] = {}
        for idx in Exact_Table['tree ' + str(t)]:
            if int(Exact_Table['tree ' + str(t)][idx]['leaf']) != default_vote:
                Ternary_Table['tree ' + str(t)][code_table_size] = Exact_Table['tree ' + str(t)][idx]
                code_table_size += 1
        Exact_Table['tree ' + str(t)] = copy.deepcopy(Ternary_Table['tree ' + str(t)])
    print('Done')
    # ===================== prepare default class =========================
    print("Preparing default class...", end="")
    collect_class = np.zeros(num_classes).tolist()
    for idx in Exact_Table['decision']:
        collect_class[Exact_Table['decision'][idx]['class']] += 1
    default_class = collect_class.index(max(collect_class))

    code_table_size = 0
    Ternary_Table['decision'] = {}
    for idx in Exact_Table['decision']:
        if Exact_Table['decision'][idx]['class'] != default_class:
            Ternary_Table['decision'][code_table_size] = Exact_Table['decision'][idx]
            code_table_size += 1
    Exact_Table['decision'] = copy.deepcopy(Ternary_Table['decision'])
    print('Done')

    # # =================== convert model timer ===================
    # Planter_config['timer log']['convert model']['end'] = time.time()
    # # =================== convert model timer ===================

    json.dump(Ternary_Table, open(dir_ternary_table, 'w'), indent=4)
    print(dir_ternary_table, '.txt is generated')
    json.dump(Exact_Table, open(dir_exact_table, 'w'), indent=4)
    print(dir_exact_table, '.txt is generated')

    create_tables_command(tree_table_file, dir_ternary_table, num_trees, num_classes, num_features, code_width_tree_feature)
    print(dir_ternary_table, '.txt is generated')

    return sklearn_y_predict.tolist()
    # json.dump(Exact_Table, open('Tables/Exact_Table.json', 'w'), indent=4)
    # print('Exact_Table is generated')
    #
    # Planter_config['p4 config'] = {}
    # Planter_config['p4 config']["model"] = "RF"
    # Planter_config['p4 config']["number of features"] = num_features
    # Planter_config['p4 config']["number of classes"] =  num_classes
    # Planter_config['p4 config']["number of trees"] =  num_trees
    # Planter_config['p4 config']['table name'] = 'Ternary_Table.json'
    # Planter_config['p4 config']["decision table size"] = len(Ternary_Table['decision'].keys())
    # Planter_config['p4 config']["code table size"] = []
    # for tree in range(num_trees):
    #     Planter_config['p4 config']["code table size"] += [len(Ternary_Table['tree '+str(tree)].keys())]
    # Planter_config['p4 config']["default vote"] = default_vote
    # Planter_config['p4 config']["default label"] = default_class
    # Planter_config['p4 config']["width of feature"] =  feature_width
    # Planter_config['p4 config']["width of code"] = code_width_tree_feature
    # Planter_config['p4 config']["used columns"] = []
    # for i in range(num_features):
    #     Planter_config['p4 config']["used columns"] += [len(Ternary_Table['feature '+str(i)].keys())]
    # Planter_config['p4 config']["width of probability"] = 7
    # Planter_config['p4 config']["width of result"] =  8
    # Planter_config['p4 config']["standard headers"] = [ "ethernet", "Planter", "arp", "ipv4", "tcp", "udp", "vlan_tag" ]
    # Planter_config['test config'] = {}
    # Planter_config['test config']['type of test'] = 'classification'
    #
    # json.dump(Planter_config , open(Planter_config['directory config']['work']+'/src/configs/Planter_config.json', 'w'), indent=4, cls=NpEncoder)
    # print(Planter_config['directory config']['work']+'/src/configs/Planter_config.json is generated')
    #
    # # main()
    # return sklearn_y_predict.tolist()


# if __name__ == '__main__':
def train_model(data_dir, tree_table_file, dir_exact_table, dir_ternary_table, num_features):
    # data_dir = '/home/p4/Data'
    # fname = 'DTtree_table.txt'
    # num_features = 5
    train_X, train_y, test_X, test_y, used_features = load_data(num_features, data_dir)
    y_predict = run_model(train_X, train_y, test_X, test_y, used_features, dir_exact_table, dir_ternary_table, tree_table_file)
