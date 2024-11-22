from federated_gbdt.models.gbdt.private_gbdt import PrivateGBDT
from FL.data_loader import DataLoader
from sklearn.metrics import roc_auc_score
import numpy as np
import re
import json
import math
import copy
import time
import random
from sklearn import metrics
from sklearn.metrics import accuracy_score, classification_report
from ctrl.Range_to_TCAM_Top_Down import *
import pandas as pd

import argparse
import socket, pickle
from time import sleep
import time

from FL.supported_models import Supported_models
from FL.fed_transfer import Fed_Avg_Client

dataloader = DataLoader()

# hardcoded for simple demo
feature_names = [' Source Port', ' Destination Port', ' Protocol', 'srcip_part_1', 'dstip_part_4']


def preorderTraversal(root):
    value_list = []
    node_list = []
    threshold_list = []
    preorderTraversalUtil(root, value_list, node_list, threshold_list)
    return value_list, node_list, threshold_list

def preorderTraversalLeftBranch(root, answer, threshold_list, feature_list, feature_names):
    if root is None:
        answer.append(-1)
        return answer
    if root.threshold == None:
        return -1

    answer.append(root.node_id)
    threshold_list.append(root.threshold)
    feature_list.append(feature_names[root.feature_i])
    preorderTraversalLeftBranch(root.true_branch, answer, threshold_list, feature_list, feature_names)
    return

def preorderTraversalRightBranch(root, answer, threshold_list, feature_list, feature_names):
    if root is None:
        return answer.append(-1)
    if root.threshold == None:
        return -1
    answer.append(root.node_id)
    threshold_list.append(root.threshold)

    feature_list.append(feature_names[root.feature_i])
    preorderTraversalRightBranch(root.false_branch, answer, threshold_list, feature_list, feature_names)
    return

def preorderTraversalUtil(root, value_list, node_list, threshold_list):
    if root is None:
        return value_list.append(np.array([random.randint(0,1000)]))
    if root.value == None:
        root.value = np.array([random.randint(0,1000)])
    if root.threshold == None:
        return -1
    value_list.append(root.value)
    node_list.append(root.node_id)
    preorderTraversalUtil(root.true_branch, value_list, node_list, threshold_list)
    preorderTraversalUtil(root.false_branch, value_list, node_list, threshold_list)
    return


# generate table
def get_lineage(tree, feature_names, file, tree_index):
    left = []
    right = []
    threshold = []
    node_id = []
    features = []
    value = []
    for i in range(tree.num_trees):
        value, node_id, threshold = preorderTraversal(tree.trees[i])
        feature_list = []
        preorderTraversalLeftBranch(tree.trees[i], left, threshold, feature_list, feature_names)
        preorderTraversalRightBranch(tree.trees[i], right, threshold, feature_list, feature_names)
    features = feature_list
    le = '<='
    g = '>'
    left = np.array(left)
    right = np.array(right)
    # Function to  print level order traversal of tree
    def printLevelOrder(root):
        level_list = []
        node_list = []
        left_list = []
        right_list = []
        value_list = []
        side = 'c'
        h = height(root)
        cnt = 0
        for i in range(1, h+1):
            cnt = printCurrentLevel(root, i, cnt, level_list, node_list, side, left_list, right_list, value_list)
        if len(left_list) < cnt:
            for i in range(0, cnt - len(left_list)):
                left_list.append(-1)
        if len(right_list) < cnt:
            for i in range(0, cnt - len(right_list)):
                    right_list.append(-1)
        return level_list, node_list, left_list, right_list, value_list

    # Print nodes at a current level
    def printCurrentLevel(root, level, cnt, level_list, node_list, side, left_list, right_list, value_list):
        if root is None:
            return cnt
        if level == 1:
            value_list.append(root.value)
            current_level = int(root.node_id.split('_')[0])
            level_list.append(current_level)
            cnt = cnt + 1
            node_list.append(root.node_id)
            root.node_id = cnt - 1
            if side == 'r':
                right_list.append(root.node_id)
            if side == 'l':
                left_list.append(root.node_id)
            return cnt
        elif level > 1:
            cnt = printCurrentLevel(root.true_branch, level-1, cnt, level_list, node_list, 'l', left_list, right_list, value_list)
            cnt = printCurrentLevel(root.false_branch, level-1, cnt, level_list, node_list, 'r', left_list, right_list, value_list)
            return cnt
    """ Compute the height of a tree--the number of nodes
        along the longest path from the root node down to
        the farthest leaf node
    """
    def height(node):
        if node is None:
            return 0
        else:
            # Compute the height of each subtree
            lheight = height(node.true_branch)
            rheight = height(node.false_branch)

            # Use the larger one
            if lheight > rheight:
                return lheight+1
            else:
                return rheight+1

    level_list, node_list, left_list, right_list, value_list  = printLevelOrder(tree.trees[i])
    idx = np.argwhere(np.array(left_list) == -1)[:, 0]
    def recurse(left, right, child, lineage=None):
          if lineage is None:
               lineage = [child]
          if child in left:
              parent = np.where(np.array(left) == child)[0].item()
              split = 'l'
          else:
              parent = np.where(np.array(right) == child)[0].item()
              split = 'r'
          lineage.append((parent, split, threshold[parent], features[parent]))

          if parent == 0:
               lineage.reverse()
               return lineage
          else:
               return recurse(left, right, parent, lineage)
    for child in idx:
         child = child - 1
    for j, child in enumerate(idx):
        clause = ' when '
        for node in recurse(left_list, right_list, child):
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
        a = [value_list[node]]
        ind = int(np.max(a))
        clause = clause[:-4] + ' then ' + str(ind)
        file.write(clause)
        file.write(";\n")


def print_tree(tree, feature_names):
    tree_ = tree.tree_
    feature_name = [
        feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!"
        for i in tree_.feature
    ]
    share = {}
    def recurse(node, depth, share):
        indent = "  " * depth
        if tree_.feature[node] != _tree.TREE_UNDEFINED:
            name = feature_name[node]
            share[name] = {}
            threshold = model.trees[node].threshold
            recurse(tree_.children_left[node], depth + 1, share)
            recurse(tree_.children_right[node], depth + 1, share)
    recurse(0, 1, share)

def ten_to_bin(num, count):
    num = bin(int(num)).lstrip('0b')
    if len(num) != count:
        cont = count - len(num)
        num = cont * '0' + num
    return num

def find_feature_split(model, tree_index, num_features, threshold):
    feature_names = []
    feature_split = {}
    for l in range(num_features):
        feature_split["feature "+str(l)] = []
        feature_names += ["f" + chr(ord('A') + l)]
    feature_list = []
    left = []
    right = []
    for i in range(model.num_trees):
        preorderTraversalLeftBranch(model.trees[i], left, threshold, feature_list, feature_names)
        preorderTraversalRightBranch(model.trees[i], right, threshold, feature_list, feature_names)
    features = feature_list
    for j, fe in enumerate(features):
        for l in range(num_features):
            if l == 0:
                if fe == feature_names[l]:
                    feature_split["feature "+str(l)].append(threshold[j])
                    continue
            if fe == feature_names[l]:
                if threshold[j] != -2.0:
                    feature_split["feature "+str(l)].append(threshold[j])
                continue
    for l in range(num_features):
        feature_split["feature "+str(l)] = [int(np.floor(i)) for i in feature_split["feature "+str(l)]]
        feature_split["feature "+str(l)].sort()
    tree = open('tree'+str(tree_index)+'.txt', "w+")
    for l in range(num_features):
        tree.write(str(feature_names[l]) + " = ")
        tree.write(str(feature_split["feature "+str(l)]))
        tree.write(";\n")

    get_lineage(model, feature_names, tree, tree_index)
    tree.close()
    action = [0, 1]
    textfile = 'tree'+str(tree_index)+'.txt'
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

def generate_table(model, tree_index, num_features, g_table, feature_max, threshold):
    textfile, feature_split = find_feature_split(model, tree_index, num_features, threshold)
    g_table[tree_index] = {}
    g_table[tree_index] = generate_feature_tables(feature_split, num_features, feature_max, g_table[tree_index])
    feature_n, classfication = find_classification(textfile, feature_split , num_features)
    path_to_leaf = find_path_for_leaf_nodes(feature_n, classfication, num_features)
    code_width_for_feature = np.zeros(num_features)
    for i in range(num_features):
        code_width_for_feature[i] = int(np.ceil(math.log(g_table[tree_index]['feature ' + str(i)][np.max(list(g_table[tree_index]['feature ' + str(i)].keys()))]+1,2))) or 1

    g_table[tree_index] = generate_code_table(g_table[tree_index], path_to_leaf, num_features)
    print('\r[INFO] The table for Tree: {} is generated'.format(tree_index), end="")
    return g_table



def create_tables_Commend(fname, num_features, num_classes):
    Ternary_Table = json.load(open('Ternary_Table.json', 'r'))
    with open(fname, 'w') as file:
        for f in range(num_features):
            for idx in Ternary_Table['feature ' + str(f)]:
                priority = int(idx)
                key = Ternary_Table['feature ' + str(f)][idx][1]
                mask = Ternary_Table['feature ' + str(f)][idx][0]
                label = Ternary_Table['feature ' + str(f)][idx][2]
                file.write("table_add SwitchIngress.lookup_feature" + str(f)+" extract_feature" + str(f)+
                           " "+str(key)+"&&&"+str(mask)+" => "+str(label)+" "+str(priority)+"\n")
            file.write("\n")
        for idx in Ternary_Table['code to vote']:
            key_value = int(idx)
            Entry = {}
            Entry["table"] = "SwitchIngress.decision"
            Entry["match"] = {}
            file.write("table_add SwitchIngress.decision read_lable ")
            for f in range(num_features):
                file.write(str(Ternary_Table['code to vote'][idx]['f' + str(f) + ' code'])+" ")
            file.write("=> "+str(Ternary_Table['code to vote'][idx]['leaf'])+"\n")


def test_tables(client_name, sklearn_test_y, test_X, test_y, num_features, num_classes, num_depth, max_leaf_nodes):
    Exact_Table = json.load(open(client_name + '_Exact_Table.json', 'r'))
    Ternary_Table = json.load(open(client_name + '_Ternary_Table.json', 'r'))

    num_features = num_features
    num_classes = num_classes
    num_depth = num_depth
    max_leaf_nodes = max_leaf_nodes
    same = 0
    correct = 0
    error = 0
    switch_test_y = []
    for i in range(np.shape(test_X.values)[0]):
        code_list = np.zeros(num_features)
        ternary_code_list = np.zeros(num_features)
        input_feature_value = test_X.values[i]

        for f in range(num_features):
            match_or_not = False

            # matcg ternary
            TCAM_table = Ternary_Table['feature ' + str(f)]
            keys = list(TCAM_table.keys())

            for count in keys:
                if int(input_feature_value[f]) & TCAM_table[count][0] == TCAM_table[count][0] & TCAM_table[count][1]:
                    ternary_code_list[f] = TCAM_table[count][2]
                    match_or_not = True
                    break

            if not match_or_not:
                print('[WARN] feature table not matched')
            key, value = list(Exact_Table.items())[0]
            code_list[f] = Exact_Table['feature ' + str(f)][str(int(input_feature_value[f]))]
            if not match_or_not:
                print('[WARN] feature table not matched')
            if str(code_list) != str(ternary_code_list):
                print('[WARN] error in exact to ternary match', code_list, ternary_code_list)


        for key in Exact_Table['code to vote']:
            match_or_not = False
            all_True = True
            for code_f in range(num_features):
                if not Exact_Table['code to vote'][key]['f' + str(code_f) + ' code'] == code_list[code_f]:
                    all_True = False
                    break
            if all_True:
                switch_prediction  = int(Exact_Table['code to vote'][key]['leaf'])
                match_or_not = True
                break
        if not match_or_not:
            switch_prediction =  0

        switch_test_y += [switch_prediction]
        if switch_prediction == test_y[i]:
            correct += 1

        if switch_prediction == sklearn_test_y[i]:
            same += 1
        else:
            error += 1
            # print('\nerror to sklearn with feature: ',input_feature_value,' and votes: ', vote_list,' switch prediction:', switch_prediction, ' sklearn: ',sklearn_test_y[i])
        if i % 100 == 0 and i != 0:
            print(
                '\rswitch_prediction: {}, test_y: {}, with acc: {:.3}, with acc to sklearn: {:.3}, with error: {:.3}, M/A format macro f1: {:.3}, macro f1: {:.3}'.format(
                    switch_prediction, test_y[i], correct / (i + 1), same / (i + 1), error / (i + 1),
                    accuracy_score(switch_test_y[:i], test_y[:i]), accuracy_score(sklearn_test_y[:i], test_y[:i])),
                end="")

    print('\nThe accuracy of the match action format of Decision Tree is', correct / np.shape(test_X.values)[0])
    result = classification_report(switch_test_y, test_y, digits=4)
    print('\n', result)



class Client:
    def __init__(self, name, server_address, server_port, model_name):
        self.name = name
        self.server_address = server_address
        self.server_port = server_port
        self.model_name = model_name
        self.model = None
        self.feature_names = None
        self.token = None
        print(f'[INFO] Creating {self.name}.')

    def train_model(self, epochs, X_train, X_test, y_train, y_test):
        self.model = PrivateGBDT(num_trees=2, epsilon=1, max_depth=5, weight_update_method="rf", training_method='rf', min_samples_split=2)
        self.model = self.model.fit(X_train, y_train)
        y_pred = self.model.predict(X_test)
        y_pred_prob = self.model.predict_proba(X_test)[:, 1]
        for tree_i in range(0, len(self.model.trees)):
            value, node_id, threshold = preorderTraversal(self.model.trees[tree_i])
        roc3=roc_auc_score(np.array(y_test), y_pred_prob)
        return self.model, value, node_id, threshold, y_pred

    def send_data_to_server(self, data):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.connect((self.server_address, self.server_port))

            # Create an instance of ProcessData() to send to server.
            # Pickle the object and send it to the server
            data_string = pickle.dumps((self.token,data))
            s.send(data_string)
            print("[INFO] Data Sent to Server")

    def wait_for_data(self):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.connect((self.server_address, self.server_port))
            data = b""
            while True:
                packet = s.recv(4096)
                if not packet:
                    break
                data += packet
            d = pickle.loads(data)
            return d

    def load_global_model(self, model):
        self.model = model

    def fed_avg_prepare_data(self, epochs, X_train, X_test, y_train, y_test):
        dataset_size = X_train.shape[0]
        print('[INFO] dataset_size to be sent: ', dataset_size)
        fed = Fed_Avg_Client(
            self.name, dataset_size, self.model
        )
        return fed, X_train, X_test, y_train, y_test

    def login(self):
        HOST = self.server_address
        PORT = self.server_port
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.connect((HOST, PORT))
            response = s.recv(2048)
            # Input UserName
            name = input(response.decode())
            s.send(str.encode(name))
            response = s.recv(2048)
            # Input Password
            password = input(response.decode())
            s.send(str.encode(password))
            ''' Response : Status of Connection :
                1 : Registeration successful
                2 : Connection Successful
                3 : Login Failed
            '''
            # Receive response
            response = s.recv(2048)
            response = response.decode()
            self.token = response
