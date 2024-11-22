# THIS FILE IS PART OF Planter PROJECT
# Planter.py - The core part of the Planter library
#
# THIS PROGRAM IS FREE SOFTWARE TOOL, WHICH MAPS MACHINE LEARNING ALGORITHMS TO DATA PLANE, IS LICENSED UNDER Apache-2.0
# YOU SHOULD HAVE RECEIVED A COPY OF THE LICENSE, IF NOT, PLEASE CONTACT THE FOLLOWING E-MAIL ADDRESSES
#
# Copyright (c) 2019 Jong-Hyouk Lee and Kamal Singh
# If you want to use this type of model,
# please cite their work 'SwitchTree: In-network Computing and Traffic Analyses with Random Forests'
# Copyright (c) 2020-2021 Changgang Zheng
# Copyright (c) Computing Infrastructure Lab, Departement of Engineering Science, University of Oxford
# E-mail: changgang.zheng@eng.ox.ac.uk or changgangzheng@qq.com
# Modified by Mingyuan Zang



from sklearn.preprocessing import LabelEncoder
import time
# from create_files import *
import math
import re
import json
from sklearn.metrics import *
import copy
import os
import numpy as np

from sklearn import tree
from sklearn.tree import export_text
from sklearn.tree import _tree
from sklearn.tree import DecisionTreeClassifier
from EDGEIIOT_dataset import load_data
import pandas as pd


def get_lineage(tree, feature_names, file):
    left = tree.tree_.children_left
    right = tree.tree_.children_right
    threshold = tree.tree_.threshold
    features = [feature_names[i] for i in tree.tree_.feature]
    value = tree.tree_.value
    le = '<='
    g = '>'
    # get ids of child nodes
    idx = np.argwhere(left == -1)[:, 0]
    # traverse the tree and get the node information
    def recurse(left, right, child, lineage=None):
        if lineage is None:
            lineage = [child]
        if child in left:
            parent = np.where(left == child)[0].item()
            split = 'l'
        else:
            parent = np.where(right == child)[0].item()
            split = 'r'
        lineage.append((parent, split, threshold[parent], features[parent]))
        if parent == 0:
            lineage.reverse()
            return lineage
        else:
            return recurse(left, right, parent, lineage)
    for j, child in enumerate(idx):
        clause = ' when '
        for node in recurse(left, right, child):
            if len(str(node)) < 3:
                continue
            i = node
            if not isinstance(i, tuple):
                continue
            if i[1] == 'l':
                sign = le
            else:
                sign = g
            clause = clause + i[3] + sign + str(i[2]) + ' and '
        # wirte the node information into text file
        a = list(value[node][0])
        ind = a.index(np.max(a))
        clause = clause[:-4] + ' then ' + str(ind)
        file.write(clause)
        file.write(";\n")


def print_tree(tree, feature_names):
    tree_ = tree.tree_
    feature_name = [
        feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!"
        for i in tree_.feature
    ]
    # print('feature_name:', feature_name)
    print("def tree({}):".format(", ".join(feature_names)))
    share = {}
    def recurse(node, depth, share):
        indent = "  " * depth
        if tree_.feature[node] != _tree.TREE_UNDEFINED:
            name = feature_name[node]
            share[name] = {}
            threshold = tree_.threshold[node]
            print("{}if {} <= {}:".format(indent, name, threshold))
            recurse(tree_.children_left[node], depth + 1, share)
            print("{}else:  # if {} > {}".format(indent, name, threshold))
            recurse(tree_.children_right[node], depth + 1, share)
        else:
            print("{}return {}".format(indent, tree_.value[node]))
    recurse(0, 1, share)




def ten_to_bin(num, count):
    num = bin(int(num)).lstrip('0b')
    if len(num) != count:
        cont = count - len(num)
        num = cont * '0' + num
    return num




def find_feature_split(model, tree_index, num_features):
    feature_names = []
    feature_split = {}
    for l in range(num_features):
        feature_split["feature "+str(l)] = []
        feature_names += ["f" + chr(ord('A') + l)]
    threshold = model.tree_.threshold
    features = [feature_names[i] for i in model.tree_.feature]
    for i, fe in enumerate(features):
        for l in range(num_features):
            if l == 0:
                if fe == feature_names[l]:
                    feature_split["feature "+str(l)].append(threshold[i])
                    continue
            if fe == feature_names[l]:
                if threshold[i] != -2.0:
                    feature_split["feature "+str(l)].append(threshold[i])
                continue
    for l in range(num_features):
        feature_split["feature "+str(l)] = [int(np.floor(i)) for i in feature_split["feature "+str(l)]]
        feature_split["feature "+str(l)].sort()
    tree = open('./tree/tree'+str(tree_index)+'.txt', "w+")
    for l in range(num_features):
        tree.write(str(feature_names[l]) + " = ")
        tree.write(str(feature_split["feature "+str(l)]))
        tree.write(";\n")
    # print_tree(model, feature_names)
    get_lineage(model, feature_names, tree)
    tree.close()
    action = [0, 1]
    textfile = './tree/tree'+str(tree_index)+'.txt'
    for f in range(num_features):
        feature_split['feature ' + str(f)] = sorted(list(set(feature_split['feature ' + str(f)])))
    return textfile, feature_split

def generate_feature_tables(split, num_faetures,feature_max, table):
    for i in range(num_faetures):
        print('i ', i)
        table["feature "+str(i)] = {}
        count_code = 0
        nife = sorted(split["feature "+str(i)])
        for j in range(feature_max[i]+1):
            if nife !=[] :
                if len(nife) > count_code:
                    if j-1 == nife[count_code]:
                        count_code+=1
            table["feature " + str(i)][j] = count_code
    return table


def find_classification(textfile, feature_split, num_features):
    fea = []
    sign = []
    num = []
    f = open(textfile, 'r')
    feature_n = {}
    text = r"("
    for l in range(num_features):
        feature_n[l] = []
        if l==0:
            text += "f"+chr(ord('A')+l)
        else:
            text += "|f" + chr(ord('A')+l)
    text += ")"
    for line in f:
        n = re.findall(r"when", line)
        if n:
            fea.append(re.findall(text, line))
            sign.append(re.findall(r"(<=|>)", line))
            num.append(re.findall(r"\d+\.?\d*", line))
    f.close()
    classfication = []
    featuren = {}
    for i in range(len(fea)):
        for l in range(num_features):
            featuren[l] = [k for k in range(len(feature_split["feature "+str(l)]) + 1)]
        for j, feature in enumerate(fea[i]):
            for l in range(num_features):
                if feature == "f"+chr(ord('A')+l):
                    sig = sign[i][j]
                    thres = int(float(num[i][j]))
                    id = feature_split["feature "+str(l)].index(thres)
                    if sig == '<=':
                        while id < len(feature_split["feature "+str(l)]):
                            if id + 1 in featuren[l]:
                                featuren[l].remove(id + 1)
                            id = id + 1
                    else:
                        while id >= 0:
                            if id in featuren[l]:
                                featuren[l].remove(id)
                            id = id - 1
                    continue
        for l in range(num_features):
            feature_n[l].append(featuren[l])
        a = len(num[i])
        classfication.append(num[i][a - 1])

    return feature_n, classfication


def find_path_for_leaf_nodes(feature_n, classfication, num_features):
    path_to_leaf = {}
    for i in range(len(classfication)):
        path_to_leaf["path "+str(i)] = {}
        path_to_leaf["path " + str(i)]["leaf"] = classfication[i]
        for j in range(num_features):
            path_to_leaf["path " + str(i)]["feature "+str(j)] = feature_n[j][i]
    return path_to_leaf




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

def generate_table(model, tree_index, num_features, g_table, feature_max):
    textfile, feature_split = find_feature_split(model, tree_index, num_features)
    print('textfile ', textfile)
    # print(tree_index, 'th tree ', len(feature_split['feature 0']), len(feature_split['feature 1']), len(feature_split['feature 2']))
    g_table[tree_index] = {}
    g_table[tree_index] = generate_feature_tables(feature_split, num_features, feature_max, g_table[tree_index])
    print('generate_feature_tables ')
    # print(g_table[tree_index]['feature 0'][feature_max[0]],g_table[tree_index]['feature 1'][feature_max[1]] ,g_table[tree_index]['feature 2'][feature_max[2]]  )
    feature_n, classfication = find_classification(textfile, feature_split , num_features)
    print('find_classification ')
    path_to_leaf = find_path_for_leaf_nodes(feature_n, classfication, num_features)
    print('find_path_for_leaf_nodes ')
    code_width_for_feature = np.zeros(num_features)
    for i in range(num_features):
        code_width_for_feature[i] = int(np.ceil(math.log(g_table[tree_index]['feature ' + str(i)][np.max(list(g_table[tree_index]['feature ' + str(i)].keys()))]+1,2))) or 1
    print('code_width_for_feature ')
    g_table[tree_index] = generate_code_table(g_table[tree_index], path_to_leaf, num_features)
    # tables = tree_to_code(action, feature_n, classfication, feature_split, num_features)
    print('\rThe table for Tree: {} is generated'.format(tree_index), end="")
    return g_table

def run_model(train_X, train_y, test_X, test_y, used_features, fname):
    # config_file = 'src/configs/Planter_config.json'

    # Planter_config = json.load(open(config_file, 'r'))

    num_trees = 1
    num_depth = int(input('- Number of depth? (defalt = 4) ') or '4')
    max_leaf_nodes = int(input('- Number of leaf nodes? (defalt = 1000) ') or '1000')
    test_mode = str(input('- Use the testing mode or not? (default = y) ') or 'y')
    if test_mode == 'y':
        if train_X.shape[0]>2000:
            train_X = train_X[:2000]
            train_y = train_y[:2000]
        if test_X.shape[0]>600:
            test_X = test_X[:600]
            test_y = test_y[:600]
    num_classes = int(np.max(train_y) + 1)

    # num_features = Planter_config['data config']['number of features']
    # num_classes = Planter_config['model config']['number of classes']
    # num_depth = Planter_config['model config']['number of depth']
    # num_trees = Planter_config['model config']['number of trees']
    # max_leaf_nodes = Planter_config['model config']['max number of leaf nodes']


    feature_names = []
    for i, f in enumerate(used_features):
        train_X.rename(columns={f: "f" + str(i)}, inplace=True)
        test_X.rename(columns={f: "f" + str(i)}, inplace=True)
        feature_names += ["f" + str(i)]

    feature_max = []
    for i in feature_names:
        t_t = [test_X[[i]].max()[0], train_X[[i]].max()[0]]
        feature_max += [int(np.max(t_t)+1)]

    # # =================== train model timer ===================
    # Planter_config['timer log']['train model'] = {}
    # Planter_config['timer log']['train model']['start'] = time.time()
    # # =================== train model timer ===================

    # Decision Tree
    model = DecisionTreeClassifier(max_depth=num_depth, max_leaf_nodes=max_leaf_nodes)
    model.fit(train_X, train_y)
    sklearn_y_predict = model.predict(test_X)

    result = classification_report(test_y, sklearn_y_predict, digits= 4)
    print('\n',result)

    g_table = {}
    g_table = generate_table(model, 0,  num_features ,g_table, feature_max)

    feature_width = []
    for max_f in feature_max:
        feature_width += [int(np.ceil(math.log(max_f, 2)) + 1)]

    code_width_tree_feature = np.zeros( num_features)
    for i in range(num_features):
        code_width_tree_feature[i] = int(np.ceil(math.log(
                g_table[0]['feature ' + str(i)][np.max(list(g_table[0]['feature ' + str(i)].keys()))] + 1,
                2) + 1)) or 1


    Exact_Table = {}


    Exact_Table['code to vote'] = g_table[0]['code to vote']
    print('Exact_Table[code to vote] = g_table[0]')
    for f in range(num_features):
        Exact_Table['feature ' + str(f)] = {}
        for value in range(feature_max[f]):
            Exact_Table['feature ' + str(f)][value] = g_table[0]["feature " + str(f)][value]
    print('for f in range(num_features):')
    # Ternary_Table = copy.deepcopy(Exact_Table)
    # for f in range(num_features):
    #     print('')
    #     print('Begine transfer: Feature table ' + str(f))
    #     Ternary_Table['feature ' + str(f)] = Table_to_TCAM(Ternary_Table['feature ' + str(f)], feature_width[f])
        # print("\n")

    # prepare default
    collect_votes = []
    # Ternary_Table['code to vote'] = {}
    # for idx in Exact_Table['code to vote']:
    #     collect_votes += [int(Exact_Table['code to vote'][idx]['leaf'])]
    # code_table_size = 0
    # default_label = max(collect_votes , key = collect_votes.count)
    for idx in Exact_Table['code to vote']:
        if int(Exact_Table['code to vote'][idx]['leaf']) != default_label:
            Ternary_Table['code to vote'][code_table_size] = Exact_Table['code to vote'][idx]
            code_table_size += 1
    print('for idx in Exact_Table[')
    Exact_Table['code to vote'] = copy.deepcopy(Ternary_Table['code to vote'])
    print('Exact_Table[code to vote')

    # =================== convert model timer ===================
    # Planter_config['timer log']['convert model']['end'] = time.time()
    # =================== convert model timer ===================

    # json.dump(Ternary_Table, open('Ternary_Table.json', 'w'), indent=4)
    # print('\nTernary_Table is generated')
    json.dump(Exact_Table, open('Exact_Table.json', 'w'), indent=4)
    print('Exact_Table is generated')
    # # =================== convert model timer ===================
    # Planter_config['timer log']['convert model']['end'] = time.time()
    # # =================== convert model timer ===================

    # json.dump(Exact_Table, open('Tables/Exact_Table.json', 'w'), indent=4)
    # print('Depth_Based_Table.txt and Exact_Table.json is generated')

    return sklearn_y_predict.tolist()


if __name__ == '__main__':
# def train_model(data_dir, fname, num_features):
    data_dir = '/home/p4/Data'
    fname = 'DTtree_table.txt'
    num_features = 5
    train_X, train_y, test_X, test_y, used_features = load_data(num_features, data_dir)
    y_predict = run_model(train_X, train_y, test_X, test_y, used_features, fname)
