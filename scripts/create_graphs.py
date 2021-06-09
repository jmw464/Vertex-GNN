#!/usr/bin/env python

import dgl
import torch as th
import torch.nn as nn
import os,sys,math,glob,ROOT
import numpy as np
import h5py
import argparse
from ROOT import gROOT, TFile, TH1D, TLorentzVector, TCanvas
import matplotlib.pyplot as plt
import time

#############################################SCRIPT PARAMS#################################################

#input data parameters
nnfeatures = 10 #number of features per node
nefeatures = 1 #number of features per edge -- NOT CURRENTLY USED

#matching parameters
incl_btoc = 1 #toggle whether to combine tracks from b hadrons and all c hadrons in B->C SV's separate them based on their direct HF ancestors
incl_secondary = 0 #toggle whether to include tracks from secondary processes in SV's based on their HF ancestors

###########################################################################################################


#create the edge list for a complete graph with n nodes
def create_edge_list(n):
    senders = np.zeros(n*(n-1),np.long)
    receivers = np.zeros(n*(n-1),np.long)

    counter = 0
    for i in range(n):
        for j in range(i+1,n):
            senders[counter] = i
            receivers[counter] = j
            senders[counter+1] = j
            receivers[counter+1] = i
            counter += 2

    return senders, receivers


def main(argv):
    gROOT.SetBatch(True)

    #parse command line arguments
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("-d", "--data_dir", type=str, required=True, dest="data_dir", help="name of directory where data is stored")
    parser.add_argument("-s", "--dataset", type=str, required=True, dest="data_name", help="name of dataset to create (without hdf5 extension)")
    args = parser.parse_args()

    data_path = args.data_dir
    data_name = args.data_name

    #file names
    infile_name = data_path+data_name+".hdf5"
    outfile_name = data_path+data_name+".bin"

    start_time = time.time()
    infile = h5py.File(infile_name, "r")
    g_list = []

    total_jets = len(infile['info']['event'])
    track_offset = 0 #tracks are stored in continuous chunk -> need to offset indices for each jet
    total_ones = total_edges = 0
    ngraphs = 0
    event_index = previous_event = -1
    for ientry in range(total_jets):
        
        #read in event/jet information
        current_event = infile['info']['event'][ientry]
        if current_event != previous_event:
            event_index += 1
            previous_event = current_event
        current_jet = infile['info']['jet'][ientry]
        ntracks =  infile['info']['ntracks'][ientry]
        primary_vertex = np.array([infile['efeatures']['event_vx'][event_index], infile['efeatures']['event_vy'][event_index], infile['efeatures']['event_vz'][event_index]])
        nedges = ntracks*(ntracks-1)

        #initialize empty arrays
        node_features = np.zeros((ntracks,nnfeatures))
        node_info = np.zeros((ntracks, 3)) #store event info - file (set in combine_graphs.py), event, jet
        edge_features = np.zeros((nedges,nefeatures))
        ancestors = np.zeros((ntracks,1))
        second_ancestors = np.zeros((ntracks,1))
        flavors = np.zeros((ntracks,1))
        truth_labels = np.zeros((nedges,1))
        
        #read in features
        for j in range(ntracks):
            track_pt  = infile['tfeatures']['pt'][track_offset+j]/1000 #convert to GeV
            track_eta = infile['tfeatures']['eta'][track_offset+j]
            track_theta = infile['tfeatures']['theta'][track_offset+j]
            track_phi = infile['tfeatures']['phi'][track_offset+j]
            track_d0 = infile['tfeatures']['d0'][track_offset+j]
            track_z0 = infile['tfeatures']['z0'][track_offset+j]
            track_q = infile['tfeatures']['q'][track_offset+j]

            jet_pt = infile['jfeatures']['pt'][ientry]/1000 #convert to GeV
            jet_eta = infile['jfeatures']['eta'][ientry]
            jet_phi = infile['jfeatures']['phi'][ientry]

            ancestors[j] = infile['labels']['ancestor'][track_offset+j]
            second_ancestors[j] = infile['labels']['second_ancestor'][track_offset+j]
            flavors[j] = infile['labels']['flavor'][track_offset+j]
            node_features[j] = [track_pt, track_eta, track_theta, track_phi, track_d0, track_z0, track_q, jet_pt, jet_eta, jet_phi]
            node_info[j] = [0, current_event, current_jet]

        track_offset += ntracks

        #calculate edge features and truth labels
        counter = 0
        for j in range(ntracks):

            for k in range(j+1, ntracks):
                
                #edge features - NOT CURRENTLY USED
                delta_pt = abs(node_features[j][0] - node_features[k][0])
                edge_features[counter:counter+2] = [delta_pt]

                #truth labels - vertices have to share the same HF ancestor
                if ancestors[k] == ancestors[j] and ancestors[k] > 0 and flavors[j] != 0 and flavors[k] != 0: #matching direct ancestors for non secondaries (B to B or C to C)
                    truth_labels[counter:counter+2] = 1
                elif second_ancestors[k] == second_ancestors[j] and second_ancestors[k] > 0: #matching second ancestors (B->C to B->C)
                    truth_labels[counter:counter+2] = incl_btoc
                elif (second_ancestors[k] == ancestors[j] and ancestors[j] > 0) or (second_ancestors[j] == ancestors[k] and ancestors[k] > 0): #matching second ancestor and direct ancestor (B to B->C)
                    truth_labels[counter:counter+2] = incl_btoc
                elif ancestors[k] == ancestors[j] and ancestors[k] < 0: #matching direct ancestors for secondaries (B to S, C to S or S to S)
                    truth_labels[counter:counter+2] = incl_secondary
                else:
                    truth_labels[counter:counter+2] = 0

                ####print(ancestors[j], flavors[j], ancestors[k], flavors[k], truth_labels[counter])

                counter += 2
       
        total_ones += np.sum(truth_labels)
        total_edges += np.size(truth_labels)

        #create graph objects and append them to the list
        if ntracks > 1:
            g = dgl.graph((create_edge_list(ntracks)))
            g.ndata['features'] = th.from_numpy(node_features)
            g.ndata['info'] = th.from_numpy(node_info)
            g.edata['labels'] = th.from_numpy(truth_labels)
            g_list.append(g)
            ngraphs += 1

        #output progress
        sys.stdout.write("\rProcessed {}% of jets".format(round(ngraphs*100/total_jets)))
        sys.stdout.flush()

    print("\nPercentage of \"True\" labels: {}%".format(total_ones*100/total_edges))

    #save graphs to file
    dgl.save_graphs(outfile_name, g_list)

    p_time = time.time()-start_time
    print("Finished creating graphs. Time elapsed: {}s.".format(p_time))


if __name__ == '__main__':
    main(sys.argv)
