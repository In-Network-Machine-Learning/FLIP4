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



from sklearn.preprocessing import LabelEncoder
from sklearn.tree import _tree
from sklearn.ensemble import RandomForestClassifier
# from ctrl import EDGEIIOT_dataset
from ..EDGEIIOT_dataset import load_data
import pandas as pd

import time
# from create_files import *
import math
import re
import json
from sklearn.metrics import *
import copy
import os
import numpy as np
from ..Range_to_TCAM_Top_Down import *

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
    tree = open('./tree'+str(tree_index)+'.txt', "w+")
    for l in range(num_features):
        tree.write(str(feature_names[l]) + " = ")
        tree.write(str(feature_split["feature "+str(l)]))
        tree.write(";\n")
    # print_tree(model, feature_names)
    get_lineage(model, feature_names, tree)
    tree.close()
    action = [0, 1]
    textfile = './tree'+str(tree_index)+'.txt'
    for f in range(num_features):
        feature_split['feature ' + str(f)] = sorted(list(set(feature_split['feature ' + str(f)])))
    return textfile, feature_split

def generate_feature_tables(split, num_faetures,feature_max, table):
    for i in range(num_faetures):
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
    # print(tree_index, 'th tree ', len(feature_split['feature 0']), len(feature_split['feature 1']), len(feature_split['feature 2']))
    g_table[tree_index] = {}
    g_table[tree_index] = generate_feature_tables(feature_split, num_features, feature_max, g_table[tree_index])
    # print(g_table[tree_index]['feature 0'][feature_max[0]],g_table[tree_index]['feature 1'][feature_max[1]] ,g_table[tree_index]['feature 2'][feature_max[2]]  )
    feature_n, classfication = find_classification(textfile, feature_split , num_features)
    path_to_leaf = find_path_for_leaf_nodes(feature_n, classfication, num_features)
    code_width_for_feature = np.zeros(num_features)
    for i in range(num_features):
        code_width_for_feature[i] = int(np.ceil(math.log(g_table[tree_index]['feature ' + str(i)][np.max(list(g_table[tree_index]['feature ' + str(i)].keys()))]+1,2))) or 1
    g_table[tree_index] = generate_code_table(g_table[tree_index], path_to_leaf, num_features)
    # tables = tree_to_code(action, feature_n, classfication, feature_split, num_features)
    print('\rThe table for Tree: {} is generated'.format(tree_index), end="")
    return g_table

def votes_to_class(tree_num, vote_list, num_trees, num_classes, g_table, num):
    if tree_num  == num_trees:
        vote = np.zeros(num_classes).tolist()
        for i in range(num_trees):
            vote[vote_list[i]] += 1
        g_table['votes to class'][num] = {}
        for t in range(len(vote_list)):
            g_table['votes to class'][num]['t'+str(t)+' vote'] = vote_list[t]
        g_table['votes to class'][num]['class'] = vote.index(np.max(vote))
        num += 1
        return g_table, num
    else:
        for value in range(num_classes):
            vote_list[tree_num] = value
            tree_num += 1
            g_table, num = votes_to_class(tree_num, vote_list, num_trees, num_classes, g_table, num)
            tree_num -= 1
    return g_table, num

def create_tables_command(fname, dir_exact_table, num_trees, num_classes, num_features, code_width_tree_feature):
    # num_features = config['data config']['number of features']
    # num_classes = config['model config']['number of classes']
    # num_trees = config['model config']['number of trees']
    Table = json.load(open(dir_exact_table, 'r'))
    with open(fname, 'w') as file:
        for f in range(num_features):
            print('f: ', f)
            for idx in Table['feature ' + str(f)]:
                priority = int(idx)
                key = Table['feature ' + str(f)][idx][1]
                mask = Table['feature ' + str(f)][idx][0]
                codes = ''
                for t in range(num_trees):
                    print('t: ', t)
                    c_tree = Table['feature ' + str(f)][idx][2][t]
                    # c_len = config['p4 config']['width of code'][t][f]
                    c_len = code_width_tree_feature[t][f]
                    codes = ten_to_bin(int(c_tree), int(c_len)) + codes
                label = int(codes, 2)
                file.write("table_add SwitchIngress.lookup_feature" + str(f)+" extract_feature" + str(f)+
                           " "+str(key)+"&&&"+str(mask)+" => "+str(label)+" "+str(priority)+"\n")

            file.write("\n")


        for t in range(num_trees):
            for idx in Table['tree ' + str(t)]:
                file.write("table_add SwitchIngress.lookup_leaf_id" + str(t) + " read_prob" + str(t) + " ")
                for f in range(num_features):
                    file.write(str(Table['tree ' + str(t)][idx]['f' + str(f) + ' code']) + " ")
                file.write("=> 0 " + str(Table['tree ' + str(t)][idx]['leaf']) + "\n")

            file.write("\n")

        for idx in Table['decision']:
            file.write("table_add SwitchIngress.decision read_lable ")
            for t in range(num_trees):
                file.write(str(Table['decision'][idx]['t' + str(t) + ' vote'])+" ")
            file.write("=> "+str(Table['decision'][idx]['class'])+"\n")


def run_model(train_X, train_y, test_X, test_y, used_features, dir_exact_table, dir_ternary_table, tree_table_file):
    # config_file = 'src/configs/Planter_config.json'
    #
    # Planter_config = json.load(open(config_file, 'r'))

    num_features = len(used_features)
    num_depth = int(input('- Number of depth? (defalt = 5) ') or '5')
    num_trees = int(input('- Number of trees? (defalt = 5) ') or '5')
    max_leaf_nodes = int(input('- Number of leaf nodes? (defalt = 1000) ') or '1000')
    test_mode = str(input('- Use the testing mode or not? (default = y) ') or 'y')
    if test_mode == 'y':
        if train_X.shape[0]>2000:
            train_X = train_X[:2000]
            train_y = train_y[:2000]
        if test_X.shape[0]>600:
            test_X = test_X[:600]
            test_y = test_y[:600]
    # num_classes = int(np.max(train_y) + 1)
    num_classes = int(np.max(train_y) + 1)

    feature_names = []
    for i, f in enumerate(used_features):
        train_X.rename(columns={f: "f" + str(i)}, inplace=True)
        test_X.rename(columns={f: "f" + str(i)}, inplace=True)
        feature_names += ["f" + str(i)]

    feature_max = []
    for i in feature_names:
        t_t = [test_X[[i]].max()[0], train_X[[i]].max()[0]]
        feature_max += [np.max(t_t)+1]

    # # =================== train model timer ===================
    # Planter_config['timer log']['train model'] = {}
    # Planter_config['timer log']['train model']['start'] = time.time()
    # # =================== train model timer ===================

    # Random Forest

    rfc = RandomForestClassifier(n_estimators=num_trees, max_depth=num_depth, max_leaf_nodes=max_leaf_nodes)
    rfc.fit(train_X, train_y)

    sklearn_y_predict = rfc.predict(test_X)

    result = classification_report(test_y, sklearn_y_predict, digits= 4)
    print('\n',result)
    #
    # # =================== train model timer ===================
    # Planter_config['timer log']['train model']['end'] = time.time()
    # # =================== train model timer ===================
    #
    # # =================== convert model timer ===================
    # Planter_config['timer log']['convert model'] = {}
    # Planter_config['timer log']['convert model']['start'] = time.time()
    # # =================== convert model timer ===================

    # # exit()
    # log_file = 'src/logs/log.json'
    # if os.path.exists(log_file):
    #     log_dict = json.load(open(log_file, 'r'))
    # else:
    #     log_dict = {}
    #
    # if ( "num_feature: "+str(num_features)) not in log_dict:
    #     log_dict["num_feature: "+str(num_features)] = {}
    # if ( "num_tree: "+str(num_trees)) not in log_dict["num_feature: "+str(num_features)]:
    #     log_dict["num_feature: "+str(num_features)]["num_tree: "+str(num_trees)] = {}
    # if ( "num_depth: "+str(num_depth)) not in log_dict["num_feature: "+str(num_features)]["num_tree: "+str(num_trees)]:
    #     log_dict["num_feature: "+str(num_features)]["num_tree: "+str(num_trees)]["num_depth: "+ str(num_depth)]= {}
    # log_dict["num_feature: " + str(num_features)][ "num_tree: " + str(num_trees)]["num_depth: " + str(num_depth)]["classification_report"] = result
    # log_dict["num_feature: " + str(num_features)][ "num_tree: " + str(num_trees)]["num_depth: " + str(num_depth)]["max number of leaf nodes"] =max_leaf_nodes
    # json.dump(log_dict, open(log_file, 'w'), indent=4)
    # print ('Classification results are downloaded to log as', log_file)
    #
    # # num_features = len(train_X.keys())
    # # num_classes = np.max(train_y)+1
    g_table = {}
    for idx, estimator in enumerate(rfc.estimators_):
        g_table = generate_table(estimator, idx,  num_features ,g_table, feature_max)

    print("\nGenerating vote to class table...", end="")
    g_table['votes to class'] = {}
    g_table, _ = votes_to_class(0, np.zeros(num_trees).tolist(), num_trees, num_classes, g_table, 0)
    print('Done')


    feature_width = []
    for max_f in feature_max:
        feature_width += [int(np.ceil(math.log(max_f, 2)) + 1)]


    code_width_tree_feature = np.zeros((num_trees,num_features))
    for i in range(num_features):
        for tree in range(num_trees):
            # code_width_tree_feature[tree, i] = np.ceil(math.log(g_table[tree]['feature ' + str(i)][feature_max[i]],2))
            code_width_tree_feature[tree, i] = int(np.ceil(math.log(g_table[tree]['feature ' + str(i)][np.max(list(g_table[tree]['feature ' + str(i)].keys()))]+1,2)+1)) or 1
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
        print('Begine transfer: Feature table ' +str (i))
        Ternary_Table['feature '+str(i)]= Table_to_TCAM(Ternary_Table['feature '+str(i)], feature_width[i])
        # print("\n")

    # ===================== prepare default vote =========================
    collect_votes = []
    for t in range(num_trees):
        for idx in Exact_Table['tree '+str(t)]:
            collect_votes += [int(Exact_Table['tree '+str(t)][idx]['leaf'])]
    default_vote = max(collect_votes, key=collect_votes.count)

    code_table_size = 0
    for t in range(num_trees):
        Ternary_Table['tree '+str(t)] = {}
        for idx in Exact_Table['tree '+str(t)]:
            if int(Exact_Table['tree '+str(t)][idx]['leaf']) != default_vote:
                Ternary_Table['tree '+str(t)][code_table_size] = Exact_Table['tree '+str(t)][idx]
                code_table_size += 1
        Exact_Table['tree '+str(t)] = copy.deepcopy(Ternary_Table['tree '+str(t)])

    # ===================== prepare default class =========================

    collect_class = []
    for idx in Exact_Table['decision']:
        collect_class += [ Exact_Table['decision'][idx]['class']]
    default_class = max(collect_class, key=collect_class.count)

    code_table_size = 0
    Ternary_Table['decision'] = {}
    for idx in Exact_Table['decision']:
        if Exact_Table['decision'][idx]['class'] != default_class:
            Ternary_Table['decision'][code_table_size] = Exact_Table['decision'][idx]
            code_table_size += 1
    Exact_Table['decision'] = copy.deepcopy(Ternary_Table['decision'])

    # # =================== convert model timer ===================
    # Planter_config['timer log']['convert model']['end'] = time.time()
    # # =================== convert model timer ===================

    # table_name = 'Ternary_Table.json'
    json.dump(Ternary_Table, open(dir_ternary_table, 'w'), indent=4)
    print('\nTernary_Table is generated')
    # json.dump(Exact_Table, open(dir_exact_table, 'w'), indent=4)
    # print('Depth_Based_Table.txt and Exact_Table.json is generated')
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
