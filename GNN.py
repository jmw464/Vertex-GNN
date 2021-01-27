#!/usr/bin/env python

#GNN for secondary vertex reconstructions

import os,sys,math,glob,ROOT
import numpy as np
from ROOT import gROOT, TFile, TH1D, TLorentzVector, TCanvas

ngfeatures = 0 #number of features for graph
nnfeatures = 4 #number of features per node
nefeatures = 1 #number of features per edge


#create the edge list for a complete graph with n nodes
def create_edge_list(n):
    senders = np.zeros(n*(n-1),np.int8)
    receivers = np.zeros(n*(n-1),np.int8)

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
    
    ntuple = TFile('/global/homes/t/toyamaza/workdir/ctag/data/ntuples/v9/output_ttbarAllHad_nominal.root')
    tree = ntuple.Get("bTag_AntiKt4EMPFlowJets_BTagging201903")

    hist_track_pt = TH1D("track_pt","track_pt",20,0,10) # N_bins, xmin, xmax
    hist_tpj = TH1D("tracks_per_jet","tracks_per_jet",10,0,30) # N_bins, xmin, xmax    

    for ientry,entry in enumerate(tree):
        njets = entry.njets

        for i in range(njets):
            ntracks =  entry.jet_trk_pt[i].size()
            nedges = ntracks*(ntracks-1)
            node_features = np.zeros((ntracks,nnfeatures))
            edge_features = np.zeros((nedges,nefeatures))
            hist_tpj.Fill(ntracks)
            #print("event %d, jet %d with %d tracks"%(ientry, i, ntracks))
        
            #read in features
            for j in range(ntracks):
                track_pt  = entry.jet_trk_pt[i][j]
                track_eta = entry.jet_trk_eta[i][j]
                track_theta = entry.jet_trk_theta[i][j]
                track_phi = entry.jet_trk_phi[i][j]
                node_features[j] = [track_pt, track_eta, track_theta, track_phi]
                hist_track_pt.Fill( track_pt * 0.001 ) # Convert from MeV to GeV

            #calculate edge features
            counter = 0
            for j in range(ntracks):
                for k in range(j+1, ntracks):
                    delta_pt = abs(node_features[j][0] - node_features[k][0])
                    edge_features[counter] = [delta_pt]
                    edge_features[counter+1] = [delta_pt]
                    counter += 2

        if ientry > 10:
            break
        
    canv1 = TCanvas("c1","c1", 800, 600)

    hist_track_pt.GetXaxis().SetTitle("Track p_{T} [GeV]")
    hist_track_pt.GetYaxis().SetTitle("Entries")    
    hist_track_pt.Draw()

    canv2 = TCanvas("c2","c2", 800, 600)

    hist_tpj.GetXaxis().SetTitle("Number of tracks")
    hist_tpj.GetYaxis().SetTitle("Entries")
    hist_tpj.Draw()

    canv1.SaveAs("track_pt.png")
    canv2.SaveAs("tpj.png")


if __name__ == '__main__':
    main(sys.argv)
