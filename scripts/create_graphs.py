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

import options


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

    #input data parameters
    nnfeatures_base = options.nnfeatures_base
    nnfeatures_errors = options.nnfeatures_errors
    nnfeatures_corrs = options.nnfeatures_corrs
    nnfeatures_hits = options.nnfeatures_hits
    nefeatures = options.nefeatures
    incl_errors = options.incl_errors
    incl_corr = options.incl_corr
    incl_hits = options.incl_hits
    incl_btoc = options.incl_btoc

    #file names
    infile_name = data_path+data_name+".hdf5"
    outfile_name = data_path+data_name+".bin"

    start_time = time.time()
    infile = h5py.File(infile_name, "r")
    g_list = []

    total_jets = len(infile['info']['event'])
    track_offset = 0 #tracks are stored in continuous chunk -> need to offset indices for each jet
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
        pv_x = infile['efeatures']['event_vx'][event_index]
        pv_y = infile['efeatures']['event_vx'][event_index]
        pv_z = infile['efeatures']['event_vx'][event_index]
        nedges = ntracks*(ntracks-1)

        jet_flavor = infile['info']['jet_flavor'][ientry]
        #make flavor label definitions consistent
        if jet_flavor == 4: #c-jet
            jet_flavor = 2
        elif jet_flavor == 5: #b-jet
            jet_flavor = 1
        elif jet_flavor == 15: #tau-jets
            jet_flavor = 0

        #initialize empty arrays
        node_features_base = np.zeros((ntracks,nnfeatures_base))
        if incl_errors: node_features_errors = np.zeros((ntracks,nnfeatures_errors))
        if incl_hits: node_features_hits = np.zeros((ntracks,nnfeatures_hits))
        if incl_corr: node_features_corrs = np.zeros((ntracks,nnfeatures_corrs))
        #edge_features = np.zeros((nedges,nefeatures)) 
        jet_info = np.zeros((ntracks, 4)) #store jet info - jet truth label (0 = light, 1 = b, 2 = c), jet pv coordinates
        track_info = np.zeros((ntracks,8))

        ancestors = np.zeros((ntracks,1))
        second_ancestors = np.zeros((ntracks,1))
        flavors = np.zeros((ntracks,1))
        bin_labels = np.zeros((nedges,1))
        mult_labels = np.zeros((nedges,1))
        flavor_labels = np.zeros((nedges,1))
        reco_labels = np.zeros((ntracks,2)) #use of track in SV0, SV1
        passed_cuts = np.zeros((ntracks,1))
        
        #read in features
        for j in range(ntracks):
            track_pt  = infile['tfeatures']['pt'][track_offset+j]/1000 #convert to GeV
            track_eta = infile['tfeatures']['eta'][track_offset+j] #not used
            track_theta = infile['tfeatures']['theta'][track_offset+j]
            track_phi = infile['tfeatures']['phi'][track_offset+j]
            track_d0 = infile['tfeatures']['d0'][track_offset+j]
            track_z0 = infile['tfeatures']['z0'][track_offset+j]
            track_q = infile['tfeatures']['q'][track_offset+j]
            if incl_errors:
                track_cov_d0d0 = math.sqrt(infile['tfeatures']['cov_d0d0'][track_offset+j])
                track_cov_z0z0 = math.sqrt(infile['tfeatures']['cov_z0z0'][track_offset+j])
                track_cov_phiphi = math.sqrt(infile['tfeatures']['cov_phiphi'][track_offset+j])
                track_cov_thetatheta = math.sqrt(infile['tfeatures']['cov_thetatheta'][track_offset+j])
                track_cov_qoverpqoverp = math.sqrt(abs(infile['tfeatures']['cov_qoverpqoverp'][track_offset+j]))
            if incl_corr:
                track_cov_d0z0 = infile['tfeatures']['cov_d0z0'][track_offset+j]
                track_cov_d0phi = infile['tfeatures']['cov_d0phi'][track_offset+j]
                track_cov_d0theta = infile['tfeatures']['cov_d0theta'][track_offset+j]
                track_cov_d0qoverp = infile['tfeatures']['cov_d0qoverp'][track_offset+j]
                track_cov_z0phi = infile['tfeatures']['cov_z0phi'][track_offset+j]
                track_cov_z0theta = infile['tfeatures']['cov_z0theta'][track_offset+j]
                track_cov_z0qoverp = infile['tfeatures']['cov_z0qoverp'][track_offset+j]
                track_cov_phitheta = infile['tfeatures']['cov_phitheta'][track_offset+j]
                track_cov_phiqoverp = infile['tfeatures']['cov_phiqoverp'][track_offset+j]
                track_cov_thetaqoverp = infile['tfeatures']['cov_thetaqoverp'][track_offset+j]
            if incl_hits:
                track_nPixHits = infile['tfeatures']['nPixHits'][track_offset+j]
                track_nSCTHits = infile['tfeatures']['nSCTHits'][track_offset+j]
                track_nBLHits = infile['tfeatures']['nBLHits'][track_offset+j]
                track_nPixHoles = infile['tfeatures']['nPixHoles'][track_offset+j]
                track_nSCTHoles = infile['tfeatures']['nSCTHoles'][track_offset+j]
                track_nPixShared = infile['tfeatures']['nPixShared'][track_offset+j]
                track_nSCTShared = infile['tfeatures']['nSCTShared'][track_offset+j]
                track_nBLShared = infile['tfeatures']['nBLShared'][track_offset+j]
                track_nPixSplit = infile['tfeatures']['nPixSplit'][track_offset+j]
                track_nBLSplit = infile['tfeatures']['nBLSplit'][track_offset+j]

            track_algo = infile['labels']['algo'][track_offset+j]

            jet_pt = infile['jfeatures']['pt'][ientry]/1000 #convert to GeV
            jet_eta = infile['jfeatures']['eta'][ientry]
            jet_phi = infile['jfeatures']['phi'][ientry]

            ancestors[j] = infile['labels']['ancestor'][track_offset+j]
            ancestors_pdgid = infile['labels']['ancestor_pdgid'][track_offset+j]
            second_ancestors[j] = infile['labels']['second_ancestor'][track_offset+j]
            second_ancestors_pdgid = infile['labels']['second_ancestor_pdgid'][track_offset+j]
            flavors[j] = infile['labels']['flavor'][track_offset+j]
            track_svx = infile['labels']['track_svx'][track_offset+j]
            track_svy = infile['labels']['track_svy'][track_offset+j]
            track_svz = infile['labels']['track_svz'][track_offset+j]
            track_info[j] = [ancestors[j], ancestors_pdgid, second_ancestors[j], second_ancestors_pdgid, flavors[j], track_svx, track_svy, track_svz]

            passed_cuts[j] = infile['labels']['passed_cuts'][track_offset+j]

            node_features_base[j] = [track_q/track_pt, track_theta, track_phi, track_d0, track_z0, jet_pt, jet_eta, jet_phi]
            if incl_errors:
                node_features_errors[j] = [track_cov_qoverpqoverp, track_cov_thetatheta, track_cov_phiphi, track_cov_d0d0, track_cov_z0z0]
            if incl_corr:
                node_features_corrs[j] = [track_cov_thetaqoverp, track_cov_phiqoverp, track_cov_d0qoverp, track_cov_z0qoverp, track_cov_phitheta, track_cov_d0theta, track_cov_z0theta, track_cov_d0phi, track_cov_z0phi, track_cov_d0z0]
            if incl_hits:
                node_features_hits[j] = [track_nPixHits, track_nSCTHits, track_nBLHits, track_nPixHoles, track_nSCTHoles, track_nPixShared, track_nSCTShared, track_nBLShared, track_nPixSplit, track_nBLSplit]

            jet_info[j] = [jet_flavor, pv_x, pv_y, pv_z]
            reco_labels[j] = [(track_algo & 1 << 2)/4, (track_algo & 1 << 3)/8]

        track_offset += ntracks

        #calculate edge features and truth labels
        counter = 0
        for j in range(ntracks):
            for k in range(j+1, ntracks):
                
                #set edge features
                #delta_pt = abs(node_features_base[j][0] - node_features_base[k][0])
                #edge_features[counter:counter+2] = [delta_pt]

                #truth labels - vertices have to share the same HF ancestor
                if ancestors[k] == ancestors[j] and ancestors[k] > 0 and flavors[j] == 1 and flavors[k] == 1: #matching direct ancestors for non secondaries (B to B)
                    bin_labels[counter:counter+2] = 1
                    mult_labels[counter:counter+2] = 1
                    flavor_labels[counter:counter+2] = 1
                elif ancestors[k] == ancestors[j] and ancestors[k] > 0 and flavors[j] == 2 and flavors[k] == 2: #matching direct ancestors for non secondaries (prompt C to prompt C)
                    bin_labels[counter:counter+2] = 1
                    mult_labels[counter:counter+2] = 2
                    flavor_labels[counter:counter+2] = 2
                elif ancestors[k] == ancestors[j] and ancestors[k] > 0 and flavors[j] == 3 and flavors[k] == 3: #matching direct ancestors for non secondaries (B->C to B->C for same C)
                    bin_labels[counter:counter+2] = 1
                    mult_labels[counter:counter+2] = 1
                    flavor_labels[counter:counter+2] = 3
                elif second_ancestors[k] == second_ancestors[j] and second_ancestors[k] > 0: #matching second ancestors (B->C to B->C for different C)
                    bin_labels[counter:counter+2] = incl_btoc
                    mult_labels[counter:counter+2] = incl_btoc
                    flavor_labels[counter:counter+2] = 3*incl_btoc
                elif (second_ancestors[k] == ancestors[j] and ancestors[j] > 0) or (second_ancestors[j] == ancestors[k] and ancestors[k] > 0): #matching second ancestor and direct ancestor (B to B->C)
                    bin_labels[counter:counter+2] = incl_btoc
                    mult_labels[counter:counter+2] = incl_btoc
                    flavor_labels[counter:counter+2] = 3*incl_btoc
                elif ancestors[k] == ancestors[j] and ancestors[k] > 0 and flavors[j] == 0 and flavors[k] == 0: #matching direct ancestors for secondaries (S to S)
                    flavor_labels[counter:counter+2] = 4
                elif ancestors[k] == ancestors[j] and ancestors[k] < 0: #matching fake vertices with no HF ancestors
                    flavor_labels[counter:counter+2] = 4

                counter += 2

        #create graph objects and append them to the list
        if ntracks > 1:
            g = dgl.graph((create_edge_list(ntracks)))
            g.ndata['features_base'] = th.from_numpy(node_features_base)
            if incl_errors: g.ndata['features_errors'] = th.from_numpy(node_features_errors)
            if incl_hits: g.ndata['features_hits'] = th.from_numpy(node_features_hits)
            if incl_corr: g.ndata['features_corr'] = th.from_numpy(node_features_corrs)
            g.ndata['graph_info'] = th.from_numpy(jet_info)
            g.ndata['node_info'] = th.from_numpy(track_info)
            g.ndata['reco_labels'] = th.from_numpy(reco_labels)
            g.ndata['passed_cuts'] = th.from_numpy(passed_cuts)
            g.edata['bin_labels'] = th.from_numpy(bin_labels)
            g.edata['mult_labels'] = th.from_numpy(mult_labels)
            g.edata['flavor_labels'] = th.from_numpy(flavor_labels)
            g_list.append(g)
            ngraphs += 1

        #output progress
        sys.stdout.write("\rProcessed {}% of jets".format(round(ngraphs*100/total_jets)))
        sys.stdout.flush()

    #save graphs to file
    dgl.save_graphs(outfile_name, g_list)

    p_time = time.time()-start_time
    print("\nFinished creating graphs. Time elapsed: {}s.".format(p_time))


if __name__ == '__main__':
    main(sys.argv)
