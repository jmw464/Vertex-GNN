#!/usr/bin/env python

import os,sys,math,glob,ROOT
import numpy as np
import h5py
from ROOT import gROOT, TFile, TH1D, TLorentzVector, TCanvas, TTree
np.set_printoptions(threshold=sys.maxsize)

max_entries = 10000000 #max number of jets
remove_pv = True
jetpt_cut = 25000
jeteta_cut = -10000
trackpt_cut = -10000

#implement cuts for a given event
def check_jet(entry, jet):
    if entry.jet_pt[jet] > jetpt_cut and entry.jet_eta[jet] > jeteta_cut:
        return True
    else:
        return False


def main(argv):
    gROOT.SetBatch(True)
    
    ntuple = TFile("/global/homes/j/jmw464/ATLAS/Vertex-GNN/data/raw/user.jmwagner.24900045.Akt4EMPf_BTagging201903._000007.root")
    #ntuple = TFile('/global/homes/t/toyamaza/workdir/ctag/data/ntuples/v10/output_WpHbb_0.root')
    tree = ntuple.Get("bTag_AntiKt4EMPFlowJets_BTagging201903;19")
    
    outfile = h5py.File("/global/homes/j/jmw464/ATLAS/Vertex-GNN/data/Btag_07_19_cut.hdf5", "w")
    
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

    efeatures = dict()
    efeatures['event_vx'] = []
    efeatures['event_vy'] = []
    efeatures['event_vz'] = []

    labels = dict()
    labels['track_vx'] = []
    labels['track_vy'] = []
    labels['track_vz'] = []

    total_cut = 0
    total_tracks = 0
    total_events = 0
    post_cut_events = 0
    for ientry,entry in enumerate(tree):

        njets = entry.njets

        efeatures['event_vx'].append(entry.truth_PVx)
        efeatures['event_vy'].append(entry.truth_PVy)
        efeatures['event_vz'].append(entry.truth_PVz)

        for i in range(njets):
            
            ntracks = entry.jet_trk_pt[i].size()
            total_tracks += ntracks
            total_cut += ntracks

            if check_jet(entry, i):

                print("event %d, jet %d with %d tracks"%(ientry, i, ntracks))

                jfeatures['pt'].append(entry.jet_pt[i])
                jfeatures['eta'].append(entry.jet_eta[i])
                jfeatures['phi'].append(entry.jet_phi[i])

                #read in track features
                cut_tracks = 0
                for j in range(ntracks):
                    pv_dist = math.sqrt((entry.truth_PVx-entry.jet_trk_vtx_X[i][j])**2 + (entry.truth_PVy-entry.jet_trk_vtx_Y[i][j])**2 + (entry.truth_PVz-entry.jet_trk_vtx_Z[i][j])**2)
                    pv_criterion = entry.jet_trk_isPV_reco[i][j]#(pv_dist < 1e-4 or entry.jet_trk_vtx_X[i][j] < -990)
                    if (remove_pv and pv_criterion) or entry.jet_trk_pt[i][j] < trackpt_cut:
                        cut_tracks += 1
                    else:
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
                info['ntracks'].append(ntracks-cut_tracks)
                
                total_cut -= (ntracks-cut_tracks)

            post_cut_events += 1

        total_events += 1
        if post_cut_events >= max_entries:
            break

    print("Cut {}% of {} tracks".format(total_cut*100./total_tracks, total_tracks))

    grp_info = outfile.create_group("info")
    grp_tfeatures = outfile.create_group("tfeatures")
    grp_jfeatures = outfile.create_group("jfeatures")
    grp_efeatures = outfile.create_group("efeatures")
    grp_labels = outfile.create_group("labels")

    for k in info.keys():
        grp_info.create_dataset(k, data = info[k])
    for k in tfeatures.keys():
        tfeatures[k] = np.asarray(tfeatures[k], dtype=np.double)
        grp_tfeatures.create_dataset(k, data = tfeatures[k])
    for k in jfeatures.keys():
        jfeatures[k] = np.asarray(jfeatures[k], dtype=np.double)
        grp_jfeatures.create_dataset(k, data = jfeatures[k])
    for k in efeatures.keys():
        efeatures[k] = np.asarray(efeatures[k], dtype=np.double)
        grp_efeatures.create_dataset(k, data = efeatures[k])
    for k in labels.keys():
        grp_labels.create_dataset(k, data = labels[k])

    outfile.close()


if __name__ == '__main__':
    main(sys.argv)
