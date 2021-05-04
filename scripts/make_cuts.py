#!/usr/bin/env python

import os,sys,math,glob,ROOT
import numpy as np
import h5py
import argparse
from ROOT import gROOT, TFile, TH1D, TLorentzVector, TCanvas, TTree

np.set_printoptions(threshold=sys.maxsize)


#############################################SCRIPT PARAMS#################################################

remove_pv = True
jetpt_cut = 20000 #20 GeV
jeteta_cut = 2.5 #edge of detector
trackpt_cut = 600 #600 MeV
tracketa_cut = 2.5 #edge of detector
trackz0_cut = 25

###########################################################################################################


#implement cuts on jet level
def check_jet(entry, jet):
    if entry.jet_pt[jet] > jetpt_cut and abs(entry.jet_eta[jet]) < jeteta_cut:
        return True
    else:
        return False


#implement cuts on track level
def check_track(entry, jet, track):
    if entry.jet_trk_pt[jet][track] > trackpt_cut and abs(entry.jet_trk_eta[jet][track]) < tracketa_cut and abs(entry.jet_trk_z0[jet][track]) < trackz0_cut:
        return True
    else:
        return False


def main(argv):
    gROOT.SetBatch(True)
    
    #parse command line arguments
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("-n", "--ntuple", type=str, required=True, dest="ntuple", help="path of ntuple to be processed")
    parser.add_argument("-t", "--tree", type=str, default="bTag_AntiKt4EMPFlowJets_BTagging201903", dest="tree", help="name of tree in ntuple")
    parser.add_argument("-e", "--entries", type=int, default=1000000, dest="max_entries", help="maximum number of entries to be processed")
    parser.add_argument("-d", "--data_dir", type=str, required=True, dest="outfile_dir", help="name of directory to save hdf5 file in")
    parser.add_argument("-s", "--dataset", type=str, required=True, dest="outfile_name", help="name of dataset to create (without hdf5 extension)")
    args = parser.parse_args()

    max_entries = args.max_entries

    ntuple = TFile(args.ntuple)
    tree = ntuple.Get(args.tree)
   
    outfile = h5py.File(args.outfile_dir+args.outfile_name+".hdf5", "w")
    
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

    total_cut_tracks = 0
    total_tracks = 0
    total_jets = 0
    total_rem_jets = 0

    for ientry,entry in enumerate(tree):

        njets = entry.njets
        rem_jets = 0
        for i in range(njets):
            
            ntracks = entry.jet_trk_pt[i].size()
            cut_tracks = 0
            t_pt = []
            t_eta = []
            t_theta = []
            t_phi = []
            t_d0 = []
            t_z0 = []
            t_q = []
            t_vx = []
            t_vy = []
            t_vz = []
            if check_jet(entry, i):
                
                #read in track features
                for j in range(ntracks):
                    pv_criterion = entry.jet_trk_isPV_reco[i][j]
                    if not (remove_pv and pv_criterion) and check_track(entry, i, j):
                        t_pt.append(entry.jet_trk_pt[i][j])
                        t_eta.append(entry.jet_trk_eta[i][j])
                        t_theta.append(entry.jet_trk_theta[i][j])
                        t_phi.append(entry.jet_trk_phi[i][j])
                        t_d0.append(entry.jet_trk_d0[i][j])
                        t_z0.append(entry.jet_trk_z0[i][j])
                        t_q.append(entry.jet_trk_charge[i][j])
                        
                        t_vx.append(entry.jet_trk_vtx_X[i][j])
                        t_vy.append(entry.jet_trk_vtx_Y[i][j])
                        t_vz.append(entry.jet_trk_vtx_Z[i][j])
                    else:
                        cut_tracks += 1
                
                #only write events that have more than one track
                if ntracks-cut_tracks > 1:
                    jfeatures['pt'].append(entry.jet_pt[i])
                    jfeatures['eta'].append(entry.jet_eta[i])
                    jfeatures['phi'].append(entry.jet_phi[i])

                    tfeatures['pt'].extend(t_pt)
                    tfeatures['eta'].extend(t_eta)
                    tfeatures['theta'].extend(t_theta)
                    tfeatures['phi'].extend(t_phi)
                    tfeatures['d0'].extend(t_d0)
                    tfeatures['z0'].extend(t_z0)
                    tfeatures['q'].extend(t_q)

                    labels['track_vx'].extend(t_vx)
                    labels['track_vy'].extend(t_vy)
                    labels['track_vz'].extend(t_vz)
             
                    t_vx_np = np.array(t_vx)
                    #t_vx_np = t_vx_np[t_vx_np > -990]
                    _, counts = np.unique(t_vx_np, return_counts=True)
                    #print(t_vx_np)

                    info['event'].append(ientry)
                    info['jet'].append(i)
                    info['ntracks'].append(ntracks-cut_tracks)

                    total_cut_tracks += cut_tracks
                    total_tracks += ntracks
                    rem_jets += 1
           
            total_jets += 1

        if rem_jets > 0:
            efeatures['event_vx'].append(entry.truth_PVx)
            efeatures['event_vy'].append(entry.truth_PVy)
            efeatures['event_vz'].append(entry.truth_PVz)

        total_rem_jets += rem_jets

        #output progress
        sys.stdout.write("\rProcessed {} jets".format(total_rem_jets))
        sys.stdout.flush()

        if total_rem_jets >= max_entries:
            break

    sys.stdout.write("\rFinished processing. Total jets used from sample: {}".format(total_rem_jets))
    sys.stdout.flush()
    print("\nCut {} jets and {}% of {} tracks in remaining jets".format(total_jets-total_rem_jets, total_cut_tracks*100./total_tracks, total_tracks))

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
