########################################################################
# Copyright (c) Mingyuan Zang, Tomasz Koziak, Networks Technology and Service Platforms Group, Technical University of Denmark
# This work was partly done with Changgang Zheng at Computing Infrastructure Group, University of Oxford
# All rights reserved.
# E-mail: mingyuanzang@outlook.com
# Licensed under the Apache License, Version 2.0 (the 'License')
########################################################################
import argparse
import os
import sys
import subprocess
from time import sleep
from multiprocessing import Process

import grpc

from io import StringIO
from numbers import Integral

import numpy as np
import pandas as pd
import datetime
import pickle
from joblib import dump, load
import sklearn
from sklearn import tree
from sklearn.tree import export_text
from sklearn.tree import _tree
from sklearn.tree import DecisionTreeClassifier


# Import P4Runtime lib from parent utils dir
# Probably there's a better way of doing this.
sys.path.append(
    os.path.join(os.path.dirname(os.path.abspath(__file__)),
                 '../utils/'))
import p4runtime_lib.bmv2
import p4runtime_lib.helper
from p4runtime_lib.switch import ShutdownAllSwitchConnections

from client_mapper import *

data_dir = './Dataset/CIC/'
num_features = 5


def writeFwdRules(p4info_helper, ingress_sw, port,
                     dst_eth_addr, dst_ip_addr):

    table_entry = p4info_helper.buildTableEntry(
        table_name="SwitchIngress.fwd_tb",
        match_fields={
            "hdr.ipv4.dstAddr": (dst_ip_addr, 32)
        },
        action_name="SwitchIngress.ipv4_forward",
        action_params={
            "dstAddr": dst_eth_addr,
            "port": port
        })
    ingress_sw.WriteTableEntry(table_entry)
    print("Installed FWD rule on %s" % ingress_sw.name)

def writeMalwareRules(p4info_helper, ingress_sw, ip_addr, malware_flag):
    table_entry = p4info_helper.buildTableEntry(
        table_name="SwitchIngress.malware",
        match_fields={
            "hdr.ipv4.srcAddr": (ip_addr, 32)
        },
        action_name="SwitchIngress.SetMalware",
        action_params={
            "malware_flag": malware_flag
        })
    ingress_sw.WriteTableEntry(table_entry)
    print("Installed malware rules rule on %s" % ingress_sw.name)

def writeMalwareInverseRules(p4info_helper, ingress_sw, ip_addr, malware_flag):
    table_entry = p4info_helper.buildTableEntry(
        table_name="SwitchIngress.malware_inverse",
        match_fields={
            "hdr.ipv4.dstAddr": (ip_addr, 32)
        },
        action_name="SwitchIngress.SetMalware",
        action_params={
            "malware_flag": malware_flag
        })
    ingress_sw.WriteTableEntry(table_entry)
    print("Installed malware rules rule on %s" % ingress_sw.name)

def resetCounter(p4info_helper, sw, counter_name):
    response = sw.ResetCounters(p4info_helper.get_counters_id(counter_name), dry_run=False)


def printCounter(p4info_helper, sw, counter_name, index):
    """
    Reads the specified counter at the specified index from the switch. In our
    program, the index is the tunnel ID. If the index is 0, it will return all
    values from the counter.

    :param p4info_helper: the P4Info helper
    :param sw:  the switch connection
    :param counter_name: the name of the counter from the P4 program
    :param index: the counter index (in our case, the tunnel ID)
    """
    for response in sw.ReadCounters(p4info_helper.get_counters_id(counter_name), index):
        for entity in response.entities:
            counter = entity.counter_entry
            print("%s %s %d: %d packets (%d bytes)" % (
                sw.name, counter_name, index,
                counter.data.packet_count, counter.data.byte_count
            ))
def readCounter(p4info_helper, sw, counter_name, index):
    for response in sw.ReadCounters(p4info_helper.get_counters_id(counter_name), index):
        for entity in response.entities:
            counter = entity.counter_entry
            return counter.data.packet_count

def SendDigestEntry(p4info_helper, sw, digest_name=None):
    digest_entry = p4info_helper.buildDigestEntry(digest_name=digest_name)
    sw.WriteDigestEntry(digest_entry)
    print("Sent DigestEntry via P4Runtime.")



# print controller packet_in:
# https://github.com/p4lang/p4runtime-shell/issues/26
def receivePacketFromDataPlane():
    send_pkt.sendPacket('send_to_cpu')
    rep = sh.client.get_stream_packet('packet',timeout=2)
    if rep is not None:
        print('ingress port is',int.from_bytes(rep.packet.metadata[0].value,'big'))

def printGrpcError(e):
    print("gRPC Error:", e.details())
    status_code = e.code()
    print("(%s)" % status_code.name)
    traceback = sys.exc_info()[2]
    print("[%s:%d]" % (traceback.tb_frame.f_code.co_filename, traceback.tb_lineno))

def computeConfusionMetrix(TP, TN, FP, FN):
    precision = recall = f1 = FPR = ACC = 0
    # Sensitivity, hit rate, recall, or true positive rate
    if TP != 0:
        recall = TP/(TP+FN)
        # Precision or positive predictive value
        precision = TP/(TP+FP)
        # Overall accuracy
        ACC = (TP+TN)/(TP+FP+FN+TN)
        # F1 score
        f1 = 2 * (precision * recall) / (precision + recall)
    # Specificity or true negative rate
    if TN != 0:
        TNR = TN/(TN+FP)
        # Negative predictive value
        NPV = TN/(TN+FN)
    if FP != 0:
        # Fall out or false positive rate
        FPR = FP/(FP+TN)
        # False discovery rate
        FDR = FP/(TP+FP)
    if FN != 0:
        # False negative rate
        FNR = FN/(TP+FN)
    return precision, recall, f1, FPR, ACC

# format MAC address
def prettify(mac_string):
    return ':'.join('%02x' % ord(b) for b in mac_string)

def bytes_to_int(bytes):
    result = 0
    for b in bytes:
        result = result * 256 + int(b)
    return result

def bytes_to_hex(bytes):
    # result = binascii.hexlify(bytes).decode("ascii")
    result = bytes.hex()
    return result

def hex_to_ip(ip_hex):
    rcv_ip_int_list = []
    for i in range(0, len(ip_hex)):
        rcv_ip_int_list.append(str(int(str(ip_hex[i]), 16)))
    rcv_ip_formatted  = '.'.join(rcv_ip_int_list)
    return rcv_ip_formatted

def bytes_to_ip(ip_bytes):
    result = '.'.join(f'{c}' for c in ip_bytes)
    return result

def read_acc(p4info_helper, sw):
    print('\n----- Reading counters -----')
    print('\n----- s1 detection counters -----')
    printCounter(p4info_helper, sw, "SwitchIngress.counter_true_attack", 0)
    printCounter(p4info_helper, sw, "SwitchIngress.counter_false_attack", 0)
    printCounter(p4info_helper, sw, "SwitchIngress.counter_false_benign", 0)
    printCounter(p4info_helper, sw, "SwitchIngress.counter_true_benign", 0)

    TP = int(readCounter(p4info_helper, sw, "SwitchIngress.counter_true_attack", 0))
    FN = int(readCounter(p4info_helper, sw, "SwitchIngress.counter_false_attack", 0))
    FP = int(readCounter(p4info_helper, sw, "SwitchIngress.counter_false_benign", 0))
    TN = int(readCounter(p4info_helper, sw, "SwitchIngress.counter_true_benign", 0))
    print("TP: ", TP)
    print("FN: ", FN)
    print("FP: ", FP)
    print("TN: ", TN)

    precision, recall, f1, FPR, ACC = computeConfusionMetrix(TP, TN, FP, FN)
    print("precision: ", precision)
    print("recall: ", recall)
    print("f1: ", f1)
    print("FPR: ", FPR)
    print("ACC: ", ACC)
    return TP, FN, FP, TN, precision, recall, f1, FPR, ACC



def process_member(member, conversion_functions):
    if member.WhichOneof('data') == 'bitstring':
        bitstring = member.bitstring
        for key, func in conversion_functions.items():
            if key in member.context:  # Assuming 'context' holds additional info about the expected conversion
                return func(bitstring)
    return None

def read_digests_refactor(p4info_helper, sw):
    print('\n----- Reading digest -----')
    digests = sw.DigestList()
    # print('digests: ', digests)
    if digests.WhichOneof('update')=='digest':
        # print("Received DigestList message")
        digest = digests.digest
        digest_name = p4info_helper.get_digests_name(digest.digest_id)
        print("===============================")
        print ("Digest name: ", digest_name)
        print("List ID: ", digest.digest_id)
        if digest_name == "int_cpu_digest_t":
            conversion_functions = {
                'ip': bytes_to_ip,
                'int': bytes_to_int,
                'hex': bytes_to_hex
            }
            features = {}  # hold feature names and their values
            for members in digest.data:
                if members.WhichOneof('data') == 'struct':
                    for i, member in enumerate(members.struct.members):
                        result = process_member(member, conversion_functions)
                        if result is not None:
                            feature_name = f"meta_feature{i}"
                            features[feature_name] = result

            # Assign features to variables based on their names
            src_IP = features.get('meta_feature0', '')
            dst_IP = features.get('meta_feature1', '')
            meta_features = [features.get(f'meta_feature_{i}', None) for i in range(2, 9)]
            print("get int_cpu_digest digest src_IP:%s" % src_IP)
            print("get int_cpu_digest digest dst_IP:%s" % dst_IP)
            print("get int_cpu_digest digest meta_feature0:%s" % meta_feature0)
            print("get int_cpu_digest digest meta_feature1:%s" % meta_feature1)
            print("get int_cpu_digest digest meta_feature2:%s" % meta_feature2)
            print("get int_cpu_digest digest meta_feature3:%s" % meta_feature3)
            print("get int_cpu_digest digest meta_feature4:%s" % meta_feature4)
            print("get int_cpu_digest digest meta_malware:%s" % meta_malware)
            print("get int_cpu_digest digest meta_class:%s" % meta_class)
            print("===============================")
            cur_time = datetime.datetime.now()
            digest_one_pkt_stats = [cur_time, src_IP, dst_IP, meta_feature0, meta_feature1, meta_feature2, meta_feature3, meta_feature4, meta_malware, meta_class]
            digest_one_pkt_stats_df = pd.DataFrame([digest_one_pkt_stats], columns=None)
            return digest_one_pkt_stats_df


def generate_FL_model_map(client, client_name, data_path):
    X_train, X_test, y_train, y_test = dataloader.load_datasets(data_path, ["CIC"], return_dict=False)[0]
    feature_names = [' Source Port', ' Destination Port', ' Protocol', 'srcip_part_1', 'dstip_part_4']
    g_table = {}
    feature_max = []
    num_features = 5
    num_classes = 2
    num_depth = 3
    max_leaf_nodes = 1000
    print(f'[INFO] Starting FL model mapping...')
    trained_model, value, node_id, threshold, y_pred = client.train_model(epochs = 5, X_train = X_train, X_test = X_test, y_train = y_train, y_test = y_test)
    client.load_global_model(trained_model)
    print(f'[INFO] Staring round: {round}')
    print(f'[INFO] Training model...')
    start_time = time.time()
    data, X_train, X_test, y_train, y_test = client.fed_avg_prepare_data(epochs=5, X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test)
    execution_time = time.time() - start_time
    print(f'[INFO] Execution time: {execution_time} seconds')
    print(f'[INFO] Sending Data...')
    client.send_data_to_server(data)
    sleep(5)
    print(f'[INFO] Wait for server...')
    global_model = client.wait_for_data()
    client.load_global_model(global_model)

    for i in feature_names:
        t_t = [X_test[[i]].max()[0], X_train[[i]].max()[0]]
        feature_max += [int(np.max(t_t)+1)]

    g_table = generate_table(trained_model, 0,  num_features ,g_table, feature_max, threshold)
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

    for f in range(num_features):
        Exact_Table['feature ' + str(f)] = {}
        for value in range(feature_max[f]):
            Exact_Table['feature ' + str(f)][value] = g_table[0]["feature " + str(f)][value]
    Ternary_Table = copy.deepcopy(Exact_Table)
    for f in range(num_features):
        Ternary_Table['feature ' + str(f)] = Table_to_TCAM(Ternary_Table['feature ' + str(f)], feature_width[f])

    # prepare default
    collect_votes = []
    Ternary_Table['code to vote'] = {}
    for idx in Exact_Table['code to vote']:
        collect_votes += [int(Exact_Table['code to vote'][idx]['leaf'])]
    code_table_size = 0
    default_label = max(collect_votes , key = collect_votes.count)
    for idx in Exact_Table['code to vote']:
        if int(Exact_Table['code to vote'][idx]['leaf']) != default_label:
            Ternary_Table['code to vote'][code_table_size] = Exact_Table['code to vote'][idx]
            code_table_size += 1
    Exact_Table['code to vote'] = copy.deepcopy(Ternary_Table['code to vote'])

    json.dump(Ternary_Table, open(client_name + '_Ternary_Table.json', 'w'), indent=4)
    json.dump(Exact_Table, open(client_name + '_Exact_Table.json', 'w'), indent=4)

    commend_file = client_name + "-commands.txt"
    create_tables_Commend(commend_file, num_features, num_classes)
    test_tables(client_name, y_pred, X_test, np.array(y_test), num_features, num_classes, num_depth, max_leaf_nodes)


def run_cpu_tasks_in_parallel(tasks):
    running_tasks = [Process(target=task) for task in tasks]
    for running_task in running_tasks:
        running_task.start()
    for running_task in running_tasks:
        running_task.join()



def main(p4info_file_path, bmv2_file_path):
    # hardcoded for simple demo
    log_cols = ['cur_time', 'src_IP', 'dst_IP', 'dstip_part_4','srcip_part_1', 'tcp.dstport','tcp.srcport', 'tcp.flags', 'Attack_label', 'prediction']
    log_df = pd.DataFrame()
    # Instantiate a P4Runtime helper from the p4info file
    p4info_helper = p4runtime_lib.helper.P4InfoHelper(p4info_file_path)
    try:
        # Create a switch connection object for s1 and s2;
        # this is backed by a P4Runtime gRPC connection.
        # Also, dump all P4Runtime messages sent to switch to given txt files.
        s1 = p4runtime_lib.bmv2.Bmv2SwitchConnection(
            name='s1',
            address='127.0.0.1:50051',
            device_id=0,
            proto_dump_file='s1-p4runtime-requests.txt')
        s2 = p4runtime_lib.bmv2.Bmv2SwitchConnection(
            name='s2',
            address='127.0.0.1:50052',
            device_id=1,
            proto_dump_file='s1-p4runtime-requests.txt')

        # Send master arbitration update message to establish this controller as
        # master (required by P4Runtime before performing any other write operation)
        s1.MasterArbitrationUpdate()
        s2.MasterArbitrationUpdate()


        data_path1 = data_dir + "cic_20000_split2_org.csv"
        data_path2 = data_dir + "cic_20000_split3_org.csv"

        run_cpu_tasks_in_parallel([
            lambda: generate_FL_model_map(Client('s1', '127.0.0.1', 5001, Supported_models.gradient_boosting_classifier), 's1', data_path1),
            lambda: generate_FL_model_map(Client('s2', '127.0.0.1', 5001, Supported_models.gradient_boosting_classifier), 's2', data_path2),
        ])



        # Install the P4 program on the switches
        print("[INFO] SetForwardingPipelineConfig...")
        print(bmv2_file_path)
        s1.SetForwardingPipelineConfig(p4info=p4info_helper.p4info,
                                       bmv2_json_file_path=bmv2_file_path)
        print("[INFO] Installed P4 Program using SetForwardingPipelineConfig on s1")
        s2.SetForwardingPipelineConfig(p4info=p4info_helper.p4info,
                                       bmv2_json_file_path=bmv2_file_path)
        print("[INFO] Installed P4 Program using SetForwardingPipelineConfig on s2")

        # only used for debugging
        malware_rule = "table_add SwitchIngress.malware SetMalware 192.168.0.128/32 => 2"
        malware_inverse_rule = "table_add SwitchIngress.malware_inverse SetMalware 192.168.0.128/32 => 2"
        with open('./s1-commands.txt', 'a') as f:
            f.write(malware_rule)
            f.write('\n')
            f.write(malware_inverse_rule)
            f.write('\n')
        print('\n----- ----- -----')

        SendDigestEntry(p4info_helper, sw=s1, digest_name="int_cpu_digest_t")
        print('\n----- digest sent -----')

        count = 0
        TP = 0
        TN = 0
        FP = 0
        FN = 0
        update_flag = 0
        # Print the tunnel counters every 2 seconds
        while True:
            sleep(2)
            count = count + 1
            if count % 5 == 0:
                update_flag = 1
            # read classification accuracy
            TP, FN, FP, TN, precision, recall, f1, FPR, ACC = read_acc(p4info_helper, s1)
            # read digest
            digest_one_pkt_stats_df = read_digests_refactor(p4info_helper, s1)
            print("precision: ", precision)
            print("recall: ", recall)
            print("f1: ", f1)
            print("FPR: ", FPR)
            print("ACC: ", ACC)
            log_df = log_df.append(digest_one_pkt_stats_df)
            print('new records saved')


    except KeyboardInterrupt:
        print(" Shutting down.")
    except grpc.RpcError as e:
        printGrpcError(e)

    ShutdownAllSwitchConnections()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='P4Runtime Controller')
    parser.add_argument('--p4info', help='p4info proto in text format from p4c',
                        type=str, action="store", required=False,
                        default='./P4/DT_anomaly_detection_digest_CICIDS_flow.p4.p4info.txt')
    parser.add_argument('--bmv2-json', help='BMv2 JSON file from p4c',
                        type=str, action="store", required=False,
                        default='./P4/DT_anomaly_detection_digest_CICIDS_flow.json')
    args = parser.parse_args()

    if not os.path.exists(args.p4info):
        parser.print_help()
        print("\np4info file not found: %s\nHave you run 'make'?" % args.p4info)
        parser.exit(1)
    if not os.path.exists(args.bmv2_json):
        parser.print_help()
        print("\nBMv2 JSON file not found: %s\nHave you run 'make'?" % args.bmv2_json)
        parser.exit(1)
    main(args.p4info, args.bmv2_json)
