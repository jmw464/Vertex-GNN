#!/usr/bin/env python

import os,sys,math,glob,ROOT
import numpy as np
import h5py
from ROOT import gROOT, TFile, TH1D, TLorentzVector, TCanvas, TTree
np.set_printoptions(threshold=sys.maxsize)

max_entries = 95000

def check_event(entry):
    return True

def main(argv):
    gROOT.SetBatch(True)
    
    ntuple = TFile('/global/homes/t/toyamaza/workdir/ctag/data/ntuples/v10/output_WpHbb_0.root')
    tree = ntuple.Get("bTag_AntiKt4EMPFlowJets_BTagging201903")
    
    outfile = h5py.File("/global/homes/j/jmw464/ATLAS/cuts.hdf5", "w")
    
    #tot_events = 0
    #tot_jets = 0
    #tot_tracks = 0
    #for ientry,entry in enumerate(tree):
        #njets = entry.njets
        #tot_events += 1
        #tot_jets += njets
        #for i in range(njets):
            #tot_tracks += entry.jet_trk_pt[i].size()

    info = dict()
    info['event'] = []
    info['jet'] = []
    info['ntracks'] = []

    jfeatures = dict()
    jfeatures['pt'] = []
    jfeatures['eta'] = []
    jfeatures['phi'] = []

    tfeatures = dict()
    tfeatures['pt'] = []
    tfeatures['eta'] = []
    tfeatures['theta'] = []
    tfeatures['phi'] = []
    tfeatures['d0'] = []
    tfeatures['z0'] = []
    tfeatures['q'] = []

    labels = dict()
    labels['track_vx'] = []
    labels['track_vy'] = []
    labels['track_vz'] = []

    total_events = 0
    post_cut_events = 0
    for ientry,entry in enumerate(tree):

        if check_event(entry):
            njets = entry.njets
            for i in range(njets):
                ntracks =  entry.jet_trk_pt[i].size()
                #print("event %d, jet %d with %d tracks"%(ientry, i, ntracks))

                jfeatures['pt'].append(entry.jet_pt[i])
                jfeatures['eta'].append(entry.jet_eta[i])
                jfeatures['phi'].append(entry.jet_phi[i])

                track_vx = np.zeros(ntracks)
                track_vy = np.zeros(ntracks)
                track_vz = np.zeros(ntracks)

                #read in features
                for j in range(ntracks):
                    tfeatures['pt'].append(entry.jet_trk_pt[i][j])
                    tfeatures['eta'].append(entry.jet_trk_eta[i][j])
                    tfeatures['theta'].append(entry.jet_trk_theta[i][j])
                    tfeatures['phi'].append(entry.jet_trk_phi[i][j])
                    tfeatures['d0'].append(entry.jet_trk_d0[i][j])
                    tfeatures['z0'].append(entry.jet_trk_z0[i][j])
                    tfeatures['q'].append(entry.jet_trk_charge[i][j])

                    labels['track_vx'].append(entry.jet_trk_vtx_X[i][j])
                    labels['track_vy'].append(entry.jet_trk_vtx_Y[i][j])
                    labels['track_vz'].append(entry.jet_trk_vtx_Z[i][j])
       
                info['event'].append(ientry)
                info['jet'].append(i)
                info['ntracks'].append(ntracks)
                
            post_cut_events += 1

        total_events += 1
        if post_cut_events >= max_entries:
            break

    grp_info = outfile.create_group("info")
    grp_tfeatures = outfile.create_group("tfeatures")
    grp_jfeatures = outfile.create_group("jfeatures")
    grp_labels = outfile.create_group("labels")

    for k in info.keys():
        grp_info.create_dataset(k, data = info[k])
    for k in tfeatures.keys():
        tfeatures[k] = np.asarray(tfeatures[k], dtype=np.double)
        grp_tfeatures.create_dataset(k, data = tfeatures[k])
    for k in jfeatures.keys():
        jfeatures[k] = np.asarray(jfeatures[k], dtype=np.double)
        grp_jfeatures.create_dataset(k, data = jfeatures[k])
    for k in labels.keys():
        grp_labels.create_dataset(k, data = labels[k])

    outfile.close()


if __name__ == '__main__':
    main(sys.argv)
