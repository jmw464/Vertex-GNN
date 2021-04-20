#!/usr/bin/env python

import dgl
import torch as th
import torch.nn as nn
import os,sys,math,glob,ROOT
import numpy as np
import h5py
from ROOT import gROOT, TFile, TH1D, TLorentzVector, TCanvas
import matplotlib.pyplot as plt
import time

#############################################SCRIPT PARAMS#################################################

#input data parameters
infile_name = "/global/homes/j/jmw464/ATLAS/Vertex-GNN/data/Btag_07_19_cut.hdf5"
nnfeatures = 10 #number of features per node
nefeatures = 1 #number of features per edge -- NOT CURRENTLY USED

#output data parameters
outfile_path = "/global/homes/j/jmw464/ATLAS/Vertex-GNN/data/"
outfile_name = "Btag_07_19_cut"
max_entries = 100000
valp = 0.2 #fraction of data used for validation
testp = 0.1 #fraction of data used for testing

#script options
reco_mode = 'b' #pv, sv or b (reconstruct only primary vertices, only secondary vertices or both)

###########################################################################################################

#file names
paramfile_name = outfile_path+outfile_name+"_params"
train_outfile_name = outfile_path+outfile_name+"_train.bin"
val_outfile_name = outfile_path+outfile_name+"_val.bin"
test_outfile_name = outfile_path+outfile_name+"_test.bin"


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
    
    print("Processing input data.")

    start_time = time.time()
    infile = h5py.File(infile_name, "r")
    g_list = []

    ievent = -1
    total_jets = len(infile['info']['event'])
    current_event = -1
    track_offset = 0 #tracks are stored in continuous chunk -> need to offset indices for each jet
    total_ones = 0
    total_edges = 0
    ngraphs = 0
    for ientry in range(total_jets):
        
        #read in current event number and update number of processed events - events are not necessarily sequential
        if current_event != infile['info']['event'][ientry]:
            current_event = infile['info']['event'][ientry]
            ievent += 1
            
            if ievent >= max_entries:
                break
            
            print("Processing event {}".format(ievent))
       
        #read in event/jet information
        current_jet = infile['info']['jet'][ientry]
        ntracks =  infile['info']['ntracks'][ientry]
        primary_vertex = np.array([infile['efeatures']['event_vx'][current_event], infile['efeatures']['event_vy'][current_event], infile['efeatures']['event_vz'][current_event]])
        nedges = ntracks*(ntracks-1)

        #initialize empty arrays
        node_features = np.zeros((ntracks,nnfeatures))
        edge_features = np.zeros((nedges,nefeatures))
        vertex_positions = np.zeros((ntracks,3))
        truth_labels = np.zeros((nedges,1))
        print("event %d, jet %d with %d tracks"%(ievent, current_jet, ntracks))
        
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
            node_features[j] = [track_pt, track_eta, track_theta, track_phi, track_d0, track_z0, track_q, jet_pt, jet_eta, jet_phi]#, track_vx, track_vy, track_vz]
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
            g.edata['labels'] = th.from_numpy(truth_labels)
            g_list.append(g)
            ngraphs += 1

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
    print("Finished processing input data. Time elapsed: {}s.".format(p_time))


if __name__ == '__main__':
    main(sys.argv)
