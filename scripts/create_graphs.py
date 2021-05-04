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

#output data parameters
valp = 0.2 #fraction of data used for validation
testp = 0.1 #fraction of data used for testing

#script options
reco_mode = 'b' #pv, sv or b (reconstruct only primary vertices, only secondary vertices or both)

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
    paramfile_name = data_path+data_name+"_params"
    train_outfile_name = data_path+data_name+"_train.bin"
    val_outfile_name = data_path+data_name+"_val.bin"
    test_outfile_name = data_path+data_name+"_test.bin"

    start_time = time.time()
    infile = h5py.File(infile_name, "r")
    g_list = []

    total_jets = len(infile['info']['event'])
    track_offset = 0 #tracks are stored in continuous chunk -> need to offset indices for each jet
    total_ones = 0
    total_edges = 0
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
        node_info = np.zeros((ntracks, 2)) #store event info
        edge_features = np.zeros((nedges,nefeatures))
        vertex_positions = np.zeros((ntracks,3))
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

            track_vx = infile['labels']['track_vx'][track_offset+j]
            track_vy = infile['labels']['track_vy'][track_offset+j]
            track_vz = infile['labels']['track_vz'][track_offset+j]
            node_features[j] = [track_pt, track_eta, track_theta, track_phi, track_d0, track_z0, track_q, jet_pt, jet_eta, jet_phi]
            node_info[j] = [current_event, current_jet]
            vertex_positions[j] = [track_vx, track_vy, track_vz]
        
        track_offset += ntracks

        #calculate edge features and truth labels
        counter = 0
        for j in range(ntracks):

            #set PV condition on truth labels
            pv_distance = np.linalg.norm(vertex_positions[j]-primary_vertex)
            if reco_mode == 'pv':
                pv_condition = (pv_distance < 1e-4)
            elif reco_mode == 'sv':
                pv_condition = (pv_distance > 1e-4)
            else:
                pv_condition = True

            for k in range(j+1, ntracks):
                
                #edge features
                delta_pt = abs(node_features[j][0] - node_features[k][0])
                edge_features[counter:counter+2] = [delta_pt]
                   
                #truth labels - vertices have to be close, real (not -999) and fulfill PV condition
                distance = np.linalg.norm(vertex_positions[j]-vertex_positions[k])
                if (distance < 1e-4) and pv_condition and vertex_positions[j][0] > -900.:
                    truth_labels[counter:counter+2] = 1
                else:
                    truth_labels[counter:counter+2] = 0
                    
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

    #calculate size of testing, training and validation set
    test_len = int(round(testp*ngraphs))
    val_len = int(round(valp*ngraphs))
    train_len = int(ngraphs - (test_len + val_len))

    #split g_list
    test_list = g_list[:test_len]
    val_list = g_list[test_len:test_len+val_len]
    train_list = g_list[test_len+val_len:]

    #save graphs to file
    dgl.save_graphs(test_outfile_name, test_list)
    dgl.save_graphs(val_outfile_name, val_list)
    dgl.save_graphs(train_outfile_name, train_list)
    infile.close()

    #store important values in paramfile
    paramfile = open(paramfile_name, "w")
    paramfile.write(str(test_len)+'\n')
    paramfile.write(str(val_len)+'\n')
    paramfile.write(str(train_len)+'\n')
    truth_frac = total_ones/total_edges
    paramfile.write(str(truth_frac)+'\n')
    paramfile.close()

    p_time = time.time()-start_time
    print("\nFinished creating graphs. Time elapsed: {}s.".format(p_time))


if __name__ == '__main__':
    main(sys.argv)
