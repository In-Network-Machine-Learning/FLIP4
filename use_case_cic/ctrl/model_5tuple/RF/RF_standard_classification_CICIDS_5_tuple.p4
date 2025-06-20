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
# This file was autogenerated

/*
 * Planter
 *
 * This program implements a simple protocol. It can be carried over Ethernet
 * (Ethertype 0x1234).
 *
 * The Protocol header looks like this:
 *
 *        0                1                  2              3
 * +----------------+----------------+----------------+---------------+
 * |      P         |       4        |     Version    |     Type      |
 * +----------------+----------------+----------------+---------------+
 * |                              feature0                            |
 * +----------------+----------------+----------------+---------------+
 * |                              feature1                            |
 * +----------------+----------------+----------------+---------------+
 * |                              feature2                            |
 * +----------------+----------------+----------------+---------------+
 * |                              feature3                            |
 * +----------------+----------------+----------------+---------------+
 * |                              feature4                            |
 * +----------------+----------------+----------------+---------------+
 * |                              Result                              |
 * +----------------+----------------+----------------+---------------+
 *
 * P is an ASCII Letter 'P' (0x50)
 * 4 is an ASCII Letter '4' (0x34)
 * Version is currently 1 (0x01)
 * Type is currently 1 (0x01)
 *
 * The device receives a packet, do the classification, fills in the
 * result and sends the packet back out of the same port it came in on, while
 * swapping the source and destination addresses.
 *
 * If an unknown operation is specified or the header is not valid, the packet
 * is dropped
 */

#include <core.p4>
#include <v1model.p4>

/*************************************************************************
*********************** headers and metadata******************************
*************************************************************************/

const bit<16> ETHERTYPE_Planter = 0x1234;
const bit<8>  Planter_P     = 0x50;   // 'P'
const bit<8>  Planter_4     = 0x34;   // '4'
const bit<8>  Planter_VER   = 0x01;   // v0.1

header ethernet_h {
    bit<48> dstAddr;
    bit<48> srcAddr;
    bit<16> etherType;
}

header Planter_h{
    bit<8> p;
    bit<8> four;
    bit<8> ver;
    bit<8> typ;
    bit<32> feature0;
    bit<32> feature1;
    bit<32> feature2;
    bit<32> feature3;
    bit<32> feature4;
    bit<32> result;
}

struct header_t {
    ethernet_h   ethernet;
    Planter_h    Planter;
}

struct metadata_t {
    bit<6> code_f0;
    bit<6> code_f1;
    bit<5> code_f2;
    bit<10> code_f3;
    bit<11> code_f4;
    bit<7> sum_prob;
    bit<4> tree_0_vote;
    bit<4> tree_1_vote;
    bit<4> tree_2_vote;
    bit<4> tree_3_vote;
    bit<4> tree_4_vote;
    bit<7> tree_0_prob;
    bit<7> tree_1_prob;
    bit<7> tree_2_prob;
    bit<7> tree_3_prob;
    bit<7> tree_4_prob;
    bit<32>  DstAddr;
}

/*************************************************************************
*********************** Ingress Parser ***********************************
*************************************************************************/

parser SwitchParser(
    packet_in pkt,
    out header_t hdr,
    inout metadata_t meta,
    inout standard_metadata_t ig_intr_md) {

    state start {
        transition parse_ethernet;
    }

    state parse_ethernet {
        pkt.extract(hdr.ethernet);
        transition select(hdr.ethernet.etherType) {
        ETHERTYPE_Planter : check_planter_version;
        default           : accept;
        }
    }

    state check_planter_version {
        transition select(pkt.lookahead<Planter_h>().p,
                          pkt.lookahead<Planter_h>().four,
                          pkt.lookahead<Planter_h>().ver) {
        (Planter_P, Planter_4, Planter_VER) : parse_planter;
        default                             : accept;
        }
    }

    state parse_planter {
        pkt.extract(hdr.Planter);
        transition accept;
    }
}

/*************************************************************************
*********************** Egress Deparser *********************************
**************************************************************************/

control SwitchDeparser(
    packet_out pkt,
    in header_t hdr) {
    apply {
        pkt.emit(hdr);
    }
}

/*************************************************************************
********************** Checksum Verification *****************************
*************************************************************************/

control SwitchVerifyChecksum(inout header_t hdr,
                       inout metadata_t meta) {
    apply {}
}
/*************************************************************************
********************** Checksum Computation ******************************
*************************************************************************/

control SwitchComputeChecksum(inout header_t hdr,
                        inout metadata_t meta) {
    apply {}
}
/*************************************************************************
*********************** Ingress Processing********************************
**************************************************************************/

control SwitchIngress(
    inout header_t hdr,
    inout metadata_t meta,
    inout standard_metadata_t ig_intr_md) {

    action send(bit<9> port) {
        ig_intr_md.egress_spec = port;
    }

    action extract_feature0(out bit<6> meta_code, bit<6> tree){
        meta_code = tree;
    }

    action extract_feature1(out bit<6> meta_code, bit<6> tree){
        meta_code = tree;
    }

    action extract_feature2(out bit<5> meta_code, bit<5> tree){
        meta_code = tree;
    }

    action extract_feature3(out bit<10> meta_code, bit<10> tree){
        meta_code = tree;
    }

    action extract_feature4(out bit<11> meta_code, bit<11> tree){
        meta_code = tree;
    }

    table lookup_feature0 {
        key = { hdr.Planter.feature0:ternary; }
        actions = {
            extract_feature0(meta.code_f0);
            NoAction;
            }
        size = 24;
        default_action = NoAction;
    }

    table lookup_feature1 {
        key = { hdr.Planter.feature1:ternary; }
        actions = {
            extract_feature1(meta.code_f1);
            NoAction;
            }
        size = 20;
        default_action = NoAction;
    }

    table lookup_feature2 {
        key = { hdr.Planter.feature2:ternary; }
        actions = {
            extract_feature2(meta.code_f2);
            NoAction;
            }
        size = 6;
        default_action = NoAction;
    }

    table lookup_feature3 {
        key = { hdr.Planter.feature3:ternary; }
        actions = {
            extract_feature3(meta.code_f3);
            NoAction;
            }
        size = 15;
        default_action = NoAction;
    }

    table lookup_feature4 {
        key = { hdr.Planter.feature4:ternary; }
        actions = {
            extract_feature4(meta.code_f4);
            NoAction;
            }
        size = 16;
        default_action = NoAction;
    }


    action read_prob0(bit<7> prob, bit<4> vote){
        meta.tree_0_prob = prob;
        meta.tree_0_vote = vote;
    }
    action write_default_class0() {
        meta.tree_0_vote = 0;
    }


    action read_prob1(bit<7> prob, bit<4> vote){
        meta.tree_1_prob = prob;
        meta.tree_1_vote = vote;
    }
    action write_default_class1() {
        meta.tree_1_vote = 0;
    }


    action read_prob2(bit<7> prob, bit<4> vote){
        meta.tree_2_prob = prob;
        meta.tree_2_vote = vote;
    }
    action write_default_class2() {
        meta.tree_2_vote = 0;
    }


    action read_prob3(bit<7> prob, bit<4> vote){
        meta.tree_3_prob = prob;
        meta.tree_3_vote = vote;
    }
    action write_default_class3() {
        meta.tree_3_vote = 0;
    }


    action read_prob4(bit<7> prob, bit<4> vote){
        meta.tree_4_prob = prob;
        meta.tree_4_vote = vote;
    }
    action write_default_class4() {
        meta.tree_4_vote = 0;
    }

    table lookup_leaf_id0 {
        key = { meta.code_f0[0:0]:exact;
                meta.code_f1[0:0]:exact;
                meta.code_f2[0:0]:exact;
                meta.code_f3[1:0]:exact;
                meta.code_f4[1:0]:exact;
                }
        actions={
            read_prob0;
            write_default_class0;
        }
        size = 1;
        default_action = write_default_class0;
    }

    table lookup_leaf_id1 {
        key = { meta.code_f0[2:1]:exact;
                meta.code_f1[1:1]:exact;
                meta.code_f2[1:1]:exact;
                meta.code_f3[3:2]:exact;
                meta.code_f4[4:2]:exact;
                }
        actions={
            read_prob1;
            write_default_class1;
        }
        size = 1;
        default_action = write_default_class1;
    }

    table lookup_leaf_id2 {
        key = { meta.code_f0[3:3]:exact;
                meta.code_f1[2:2]:exact;
                meta.code_f2[2:2]:exact;
                meta.code_f3[5:4]:exact;
                meta.code_f4[6:5]:exact;
                }
        actions={
            read_prob2;
            write_default_class2;
        }
        size = 1;
        default_action = write_default_class2;
    }

    table lookup_leaf_id3 {
        key = { meta.code_f0[4:4]:exact;
                meta.code_f1[3:3]:exact;
                meta.code_f2[3:3]:exact;
                meta.code_f3[7:6]:exact;
                meta.code_f4[8:7]:exact;
                }
        actions={
            read_prob3;
            write_default_class3;
        }
        size = 1;
        default_action = write_default_class3;
    }

    table lookup_leaf_id4 {
        key = { meta.code_f0[5:5]:exact;
                meta.code_f1[5:4]:exact;
                meta.code_f2[4:4]:exact;
                meta.code_f3[9:8]:exact;
                meta.code_f4[10:9]:exact;
                }
        actions={
            read_prob4;
            write_default_class4;
        }
        size = 1;
        default_action = write_default_class4;
    }

    action read_lable(bit<32> label){
        hdr.Planter.result = label;
    }

    action write_default_decision() {
        hdr.Planter.result = 0;
    }

    table decision {
        key = { meta.tree_0_vote:exact;
                meta.tree_1_vote:exact;
                meta.tree_2_vote:exact;
                meta.tree_3_vote:exact;
                meta.tree_4_vote:exact;
                }
        actions={
            read_lable;
            write_default_decision;
        }
        size = 16;
        default_action = write_default_decision;
    }

    apply{
        lookup_feature0.apply();
        lookup_feature1.apply();
        lookup_feature2.apply();
        lookup_feature3.apply();
        lookup_feature4.apply();
        lookup_leaf_id0.apply();
        lookup_leaf_id1.apply();
        lookup_leaf_id2.apply();
        lookup_leaf_id3.apply();
        lookup_leaf_id4.apply();
        decision.apply();
        bit<48> tmp;
        /* Swap the MAC addresses */
        tmp = hdr.ethernet.dstAddr;
        hdr.ethernet.dstAddr = hdr.ethernet.srcAddr;
        hdr.ethernet.srcAddr = tmp;
        send(ig_intr_md.ingress_port);
    }
}
/*************************************************************************
*********************** egress Processing********************************
**************************************************************************/

control SwitchEgress(inout header_t hdr,
    inout metadata_t meta,
    inout standard_metadata_t eg_intr_md) {
    apply {
    }
}
/*************************************************************************
***********************  S W I T C H  ************************************
*************************************************************************/

V1Switch(
    SwitchParser(),
    SwitchVerifyChecksum(),
    SwitchIngress(),
    SwitchEgress(),
    SwitchComputeChecksum(),
    SwitchDeparser()
) main;