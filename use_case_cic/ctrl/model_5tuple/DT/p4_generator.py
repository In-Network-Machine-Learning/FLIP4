# THIS FILE IS PART OF Planter PROJECT
# Planter.py - The core part of the Planter library
#
# THIS PROGRAM IS FREE SOFTWARE TOOL, WHICH MAPS MACHINE LEARNING ALGORITHMS TO DATA PLANE, IS LICENSED UNDER Apache-2.0
# YOU SHOULD HAVE RECEIVED A COPY OF THE LICENSE, IF NOT, PLEASE CONTACT THE FOLLOWING E-MAIL ADDRESSES
#
# Copyright (c) 2020-2021 Changgang Zheng
# Copyright (c) Computing Infrastructure Lab, Departement of Engineering Science, University of Oxford
# E-mail: changgang.zheng@eng.ox.ac.uk or changgangzheng@qq.com

import sys, getopt
import os
import json
import binascii
import numpy as np
from dedicated_p4 import *
from common_p4 import *
# from src.functions.add_license import *





def create_headers(fname):
    with open(fname, 'a') as headers:
        headers.write("/*************************************************************************\n"
                     "*********************** headers and metadata *****************************\n"
                     "*************************************************************************/\n\n")
    common_basic_headers(fname)
    common_headers(fname)

    with open(fname, 'a') as headers:
        headers.write("struct metadata_t {\n")
    separate_metadata(fname)
    try:
        common_metadata(fname)
    except Exception as e:
        pass
    with open(fname, 'a') as headers:
        headers.write("}\n\n")


###################################################
# Create a parser file to be used
# input: parser file name, configuration
# output: none
###################################################

# This code currently does not support skipping columns. This is to be added once the basic functionality is tested
def create_parser(fname):

    with open(fname, 'a') as parser:

        parser.write("/*************************************************************************\n"
                     "*********************** Ingress Parser ***********************************\n"
                     "*************************************************************************/\n\n")



        parser.write(
            "parser SwitchParser(\n"
            "    packet_in pkt,\n"
            "    out header_t hdr,\n"
            "    inout metadata_t meta,\n"
            "    inout standard_metadata_t ig_intr_md) {\n\n")

        parser.write("    state start {\n"
                     "        transition parse_ethernet;\n"
                     "    }\n\n")

    common_parser(fname, config)

    with open(fname, 'a') as parser:
        parser.write("}\n\n")

        parser.write("/*************************************************************************\n"
                     "*********************** Egress Deparser *********************************\n"
                     "**************************************************************************/\n\n")

        parser.write("control SwitchDeparser(\n"
                     "    packet_out pkt,\n"
                     "    in header_t hdr) {\n"
                     "    apply {\n"
                     "        pkt.emit(hdr);\n"
                     "    }\n"
                     "}\n\n")




        parser.write("/*************************************************************************\n"
                     "********************** Checksum Verification *****************************\n"
                     "*************************************************************************/\n\n")
        parser.write(
            "control SwitchVerifyChecksum(inout header_t hdr,\n"
            "                       inout metadata_t meta) {\n"
            "    apply {}\n"
            "}\n")


        parser.write("/*************************************************************************\n"
                     "********************** Checksum Computation ******************************\n"
                     "*************************************************************************/\n\n")
        parser.write(
            "control SwitchComputeChecksum(inout header_t hdr,\n"
            "                        inout metadata_t meta) {\n"
            "    apply {}\n"
            "}\n")



###################################################
# Create an ingress control file
# input: ingress_control file name, configuration
# output: none
###################################################

# This code currently does not support skipping columns. This is to be added once the basic functionality is tested
def create_ingress_control(fname):

    with open(fname, 'a') as ingress:

        ingress.write("/*************************************************************************\n"
                      "*********************** Ingress Processing********************************\n"
                      "**************************************************************************/\n\n")

        ingress.write("control SwitchIngress(\n    inout header_t hdr,\n"
                      "    inout metadata_t meta,\n"
                      "    inout standard_metadata_t ig_intr_md) {\n\n")


        ingress.write("    action send(bit<9> port) {\n"
                      "        ig_intr_md.egress_spec = port;\n"
                      "    }\n\n")

        ingress.write("    action drop() {\n"
                      "        mark_to_drop(ig_intr_md);\n"
                      "    }\n\n")

    # =================== Tables and actions =================
    try:
        common_tables(fname)
    except Exception as e:
        pass
    separate_tables(fname)


    with open(fname, 'a') as ingress:
        ingress.write("    apply{\n")

    # =================== Logics =================
    try:
        common_feature_extraction(fname)
    except Exception as e:
        pass
    separate_logics(fname)
    common_logics(fnameg)

    with open(fname, 'a') as ingress:
        ingress.write("    }\n" )
        ingress.write("}\n")


###################################################
# Create an egress control file
# input: egress_control file name, configuration
# output: none
###################################################

# This code currently does not support skipping columns. This is to be added once the basic functionality is tested
def create_egress_control(fname):

    with open(fname, 'a') as egress:
        egress.write("/*************************************************************************\n"
                     "*********************** egress Processing********************************\n"
                     "**************************************************************************/\n\n"
                     "control SwitchEgress(inout header_t hdr,\n"
                     "    inout metadata_t meta,\n"
                     "    inout standard_metadata_t eg_intr_md) {\n")

        # The following implements static routing between two ports, change as necessary
        egress.write("    apply {\n"
                     )

        egress.write("    }\n}\n")


###################################################
# Create main function in p4 code
# input: table scipt file name, tables data json file name, configuration
# output: none
###################################################
def create_main(fname):
    with open(fname, 'a') as main:
        main.write("/*************************************************************************\n"
                     "***********************  S W I T C H  ************************************\n"
                     "*************************************************************************/\n\n"
                     "V1Switch(\n"
                     "    SwitchParser(),\n"
                     "    SwitchVerifyChecksum(),\n"
                     "    SwitchIngress(),\n"
                     "    SwitchEgress(),\n"
                     "    SwitchComputeChecksum(),\n"
                     "    SwitchDeparser()\n"
                     ") main;")


###################################################
# Create includes in code
# input: table scipt file name, tables data json file name, configuration
# output: none
###################################################

def create_include(fname):
    with open(fname, 'a') as main:
        main.write("#include <core.p4>\n")
        main.write("#include <v1model.p4>\n\n")

###################################################
# Load the configuration from the config file
# input: config file name
# output: structure of config parameters
###################################################





##################################################
# Main function
# Parse input, set file name and call functions
##################################################

def main():
    ##################################################
    # print('Generate p4 files')
    file_name = 'DT_switchtree_bmv2_iiotedge'
    p4_file = './P4/' + file_name+'.p4'
    # tables_json = Planter_config['p4 config']['table name']

    ##################################################
    print('Generating p4 files and load data file...',end=" ")
    # add_license(p4_file)

    # add_model_intro(p4_file)

    # create include file
    create_include(p4_file)
    # create headers file
    create_headers(p4_file)
    # create ingress parser
    create_parser(p4_file)
    # create ingress control
    create_ingress_control(p4_file)
    # create egress control
    create_egress_control(p4_file)
    # create main function
    create_main(p4_file)
    # print('The p4 file is generated')

    ##################################################
    # load_data_file = Planter_config['directory config']['work'] + '/Tables/load_table.py'
    # # create load tables script
    # # add_license(load_data_file)
    # create_load_tables(load_data_file, tables_json, config, Planter_config, file_name)
    # print('The load_table file is generated')
    print("Done")


if __name__ == "__main__":
    main()
