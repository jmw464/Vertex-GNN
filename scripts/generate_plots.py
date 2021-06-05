#!/usr/bin/env python

import os,sys,math,ROOT,glob
import numpy as np
import argparse
from ROOT import TFile, TH1D, TH1I, gROOT, TCanvas, gPad, TLegend
import time

from truth import *

#############################################SCRIPT PARAMS#################################################

remove_pv = True

jet_pt_cut = 20000 #20 GeV
jet_eta_cut = 2.5 #edge of detector
track_pt_cut = 650 #600 MeV
track_eta_cut = 2.5 #edge of detector
track_z0_cut = 20

###########################################################################################################

#plot list of histograms with specified labels
def plot_hist(canv, histlist, labellist, norm, filename, options, overflow):
    gPad.SetLogy()
    legend = TLegend(0.78,0.95-0.1*len(histlist),0.98,0.95)
    colorlist = [4, 8, 2, 6, 1]
    if options: same = "SAMES "
    else: same = "SAMES"
    for i in range(len(histlist)):
        entries = histlist[i].GetEntries()
        mean = histlist[i].GetMean()
        histlist[i].SetLineColorAlpha(colorlist[i],0.65)
        histlist[i].SetLineWidth(3)
        nbins = histlist[i].GetNbinsX()
        if overflow: histlist[i].SetBinContent(nbins, histlist[i].GetBinContent(nbins) + histlist[i].GetBinContent(nbins+1))
        if entries and norm: histlist[i].Scale(1./histlist[i].Integral(), "width")
        if i == 0: histlist[i].Draw(options)
        else: histlist[i].Draw(same+options)
        legend.AddEntry(histlist[i], "#splitline{"+labellist[i]+"}{#splitline{%d entries}{mean=%.2f}}"%(entries, mean), "l")
    legend.SetTextSize(0.025)
    legend.Draw("SAME")
    canv.SaveAs(filename)
    gPad.Clear()
    canv.Clear()


def main(argv):
    gROOT.SetBatch(True)

    #parse command line arguments
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("-e", "--entries", type=int, default=1000000, dest="maxentries", help="maximum number of events to be processed")
    parser.add_argument("-r", "--runnumber", type=str, default=0, dest="runnumber", help="unique identifier for current run")
    parser.add_argument("-d", "--dataset", type=str, required=True, dest="infile_name", help="name of dataset to train on (without hdf5 extension)")
    parser.add_argument("-n", "--ntuple", type=str, required=True, dest="ntuple", help="path of ntuple to be processed")
    parser.add_argument("-t", "--tree", type=str, default="bTag_AntiKt4EMPFlowJets_BTagging201903", dest="tree", help="name of tree in ntuple")
    parser.add_argument("-o", "--output_dir", type=str, required=True, dest="output_dir", help="name of directory where plots should be stored")
    parser.add_argument("-b", "--bad_jets", type=int, default=1, dest="only_bad_events", help="indicate whether to plot from list of bad jets (True) or from full dataset (False)")
    args = parser.parse_args()

    maxentries = args.maxentries #processing only stops after every jet in an event is read in -> first event that passes maxentries jets is last one
    runnumber = args.runnumber
    dataname = args.infile_name
    savepath = args.output_dir
    only_bad_events = args.only_bad_events
  
    ntuple = TFile(args.ntuple)
    tree = ntuple.Get(args.tree)

    #association criteria are being the direct descendant (max one level removed) of a HF hadron that spawned a track
    hist_no_char_b = TH1D("no_char_b", "Number of charged particle children per vertex;Number of particles;Normalized entries", 10, 0, 10)
    hist_no_char_c = TH1D("no_char_c", "Number of charged particle children per vertex;Number of particles;Normalized entries", 10, 0, 10)
    hist_no_char_btoc = TH1D("no_char_btoc", "Number of charged particle children per vertex;Number of particles;Normalized entries", 10, 0, 10)
    
    hist_fl_len_b = TH1D("fl_len_b", "Distance between HF hadron decay vertex and track vertex;Distance [mm];Entries", 20, 0, 1)
    hist_fl_len_c = TH1D("fl_len_c", "Distance between HF hadron decay vertex and track vertex;Distance [mm];Entries", 20, 0, 1)
    hist_fl_len_btoc = TH1D("fl_len_btoc", "Distance between HF hadron decay vertex and track vertex;Distance [mm];Entries", 20, 0, 1)

    hist_trk_vtx_dist_b = TH1D("trk_vtx_dist_b", "Distance between track PVs within SV;Distance [mm];Entries", 20, 0, 1)
    hist_trk_vtx_dist_c = TH1D("trk_vtx_dist_c", "Distance between track PVs within SV;Distance [mm];Entries", 20, 0, 1)
    hist_trk_vtx_dist_btoc = TH1D("trk_vtx_dist_btoc", "Distance between track PVs within SV;Distance [mm];Entries", 20, 0, 1)

    hist_no_trk_acc = TH1D("no_trk_acc", "Number of tracks per jet;Number of tracks;Entries", 15, 0, 30)
    hist_no_trk_rej = TH1D("no_trk_rej", "Number of tracks per jet;Number of tracks;Entries", 15, 0, 30)

    hist_trk_pt_b = TH1D("trk_pt_b", "Track pT;pT [MeV];Normalized entries", 20, 0, 30000)
    hist_trk_pt_c = TH1D("trk_pt_c", "Track pT;pT [MeV];Normalized entries", 20, 0, 30000)
    hist_trk_pt_o = TH1D("trk_pt_o", "Track pT;pT [MeV];Normalized entries", 20, 0, 30000)
    hist_trk_pt_nm = TH1D("trk_pt_nm", "Track pT;pT [MeV];Normalized entries", 20, 0, 30000)
    hist_trk_pt_btoc = TH1D("trk_pt_btoc", "Track pT;pT [MeV];Normalized entries", 20, 0, 30000)

    hist_trk_eta_b = TH1D("trk_eta_b", "Track eta;Eta;Normalized entries", 20, -5, 5)
    hist_trk_eta_c = TH1D("trk_eta_c", "Track eta;Eta;Normalized entries", 20, -5, 5)
    hist_trk_eta_o = TH1D("trk_eta_o", "Track eta;Eta;Normalized entries", 20, -5, 5)
    hist_trk_eta_nm = TH1D("trk_eta_nm", "Track eta;Eta;Normalized entries", 20, -5, 5)
    hist_trk_eta_btoc = TH1D("trk_eta_btoc", "Track eta;Eta;Normalized entries", 20, -5, 5)

    hist_trk_phi_b = TH1D("trk_phi_b", "Track phi;Phi;Normalized entries", 20, -math.pi, math.pi)
    hist_trk_phi_c = TH1D("trk_phi_c", "Track phi;Phi;Normalized entries", 20, -math.pi, math.pi)
    hist_trk_phi_o = TH1D("trk_phi_o", "Track phi;Phi;Normalized entries", 20, -math.pi, math.pi)
    hist_trk_phi_nm = TH1D("trk_phi_nm", "Track phi;Phi;Normalized entries", 20, -math.pi, math.pi)
    hist_trk_phi_btoc = TH1D("trk_phi_btoc", "Track phi;Phi;Normalized entries", 20, -math.pi, math.pi)

    hist_trk_z0_b = TH1D("trk_z0_b", "Track z0;z0 [cm];Normalized entries", 20, -25, 25)
    hist_trk_z0_c = TH1D("trk_z0_c", "Track z0;z0 [cm];Normalized entries", 20, -25, 25)
    hist_trk_z0_o = TH1D("trk_z0_o", "Track z0;z0 [cm];Normalized entries", 20, -25, 25)
    hist_trk_z0_nm = TH1D("trk_z0_nm", "Track z0;z0 [cm];Normalized entries", 20, -25, 25)
    hist_trk_z0_btoc = TH1D("trk_z0_btoc", "Track z0;z0 [cm];Normalized entries", 20, -25, 25)

    #association criteria are being the most recent HF antecedant of a track particle (not necessarily only one level removed)
    hist_no_trk_b = TH1D("no_trk_b", "Number of associated tracks per vertex;Number of tracks;Normalized entries", 10, 0, 10)
    hist_no_trk_c = TH1D("no_trk_c", "Number of associated tracks per vertex;Number of tracks;Normalized entries", 10, 0, 10)
    hist_no_trk_btoc = TH1D("no_trk_btoc", "Number of associated tracks per vertex;Number of tracks;Normalized entries", 10, 0, 10)

    hist_no_trk_jet_b = TH1D("no_trk_jet_b", "Number of associated tracks per jet;Number of tracks;Normalized entries", 10, 0, 10)
    hist_no_trk_jet_c = TH1D("no_trk_jet_c", "Number of associated tracks per jet;Number of tracks;Normalized entries", 10, 0, 10)
    hist_no_trk_jet_btoc = TH1D("no_trk_jet_btoc", "Number of associated tracks per jet;Number of tracks;Normalized entries", 10, 0, 10)
    hist_no_trk_jet_o = TH1D("no_trk_jet_o", "Number of associated tracks per jet;Number of tracks;Normalized entries", 10, 0, 10)
    hist_no_trk_jet_nm = TH1D("no_trk_jet_nm", "Number of associated tracks per jet;Number of tracks;Normalized entries", 10, 0, 10)

    hist_frac_trk_b = TH1D("frac_trk_b", "Fraction of tracks per jet;Track fraction;Entries", 10, 0, 1)
    hist_frac_trk_c = TH1D("frac_trk_c", "Fraction of tracks per jet;Track fraction;Entries", 10, 0, 1)
    hist_frac_trk_o = TH1D("frac_trk_o", "Fraction of tracks per jet;Track fraction;Entries", 10, 0, 1)
    hist_frac_trk_nm = TH1D("frac_trk_nm", "Fraction of tracks per jet;Track fraction;Entries", 10, 0, 1)
    hist_frac_trk_btoc = TH1D("frac_trk_btoc", "Fraction of tracks per jet;Track fraction;Entries", 10, 0, 1)

    start_time = time.time()

    #read in dictionary of bad events
    registerfilename = savepath+runnumber+"/"+dataname+"_"+runnumber+"_register"
    bad_events = {}
    if only_bad_events and os.path.isfile(registerfilename):
        registerfile = open(registerfilename, "r")
        contents = registerfile.readlines()
        for line in contents:
            values = line.strip().split()
            event = int(float(values[0]))
            jet = int(float(values[1]))
            if event in bad_events.keys():
                bad_events[event] = np.append(bad_events[event], jet)
            else:
                bad_events[event] = np.array([jet])
        base_filename = savepath+runnumber+"/"+dataname+"_"+runnumber
    else:
        base_filename = savepath+dataname
        only_bad_events = False #default to plotting from full file if register of bad events isn't available

    processed_jets = 0
    for ientry,entry in enumerate(tree):

        #only process events in the register if only_bad_events is true
        if not only_bad_events or ientry in bad_events.keys():
            
            njets = entry.njets

            #check event to see if it can be skipped
            event_pass = False
            for i in range(njets):
                if (not only_bad_events or i in bad_events[ientry]) and check_jet(entry, i, jet_pt_cut, jet_eta_cut):
                    rem_trk = 0
                    nTrack =  entry.jet_trk_pt[i].size()
                    for j in range(nTrack):
                        if check_track(entry, i, j, track_pt_cut, track_eta_cut, track_z0_cut) and (not remove_pv or not entry.jet_trk_isPV_reco[i][j]):
                            rem_trk += 1
                    if rem_trk > 1:
                        event_pass = True
            if not event_pass:
                continue

            particle_dict = build_particle_dict(entry)

            primary_vertex = np.array([entry.truth_PVx, entry.truth_PVy, entry.truth_PVz])

            #build track dictionary for each jet
            for i in range( njets ):

                #only process jets on the register if only_bad_events is true and make sure they pass cuts
                if (not only_bad_events or i in bad_events[ientry]) and check_jet(entry, i, jet_pt_cut, jet_eta_cut):
                    nTrack =  entry.jet_trk_pt[i].size()

                    #count tracks that pass cuts to know which jets to skip entirely
                    rem_trk = 0
                    for j in range(nTrack):
                        if (not remove_pv or not entry.jet_trk_isPV_reco[i][j]) and check_track(entry, i, j, track_pt_cut, track_eta_cut, track_z0_cut):
                            rem_trk += 1
                    if rem_trk <= 1:
                        continue

                    processed_jets += 1
                    
                    track_dict, jet_cut_trk = build_track_dict(entry, i, particle_dict, remove_pv, track_pt_cut, track_eta_cut, track_z0_cut)
                    
                    hist_no_trk_acc.Fill(len(track_dict))
                    hist_no_trk_rej.Fill(len(track_dict)+jet_cut_trk)

                    jet_trk_btoc = jet_trk_ch = jet_trk_bh = jet_trk_o = jet_trk_nm = 0
                    bh_parent_list = np.array([]) #list that stores all direct bH parent particles in jet
                    ch_parent_list = np.array([])
                    btoc_parent_list = np.array([])
                    
                    #classify tracks
                    for ti in track_dict:
                        t_class = classify_track(ti, particle_dict, track_dict)
                        track_dict[ti].classification = t_class

                    #fill histograms
                    for ti in track_dict:
 
                        t_class = track_dict[ti].classification
                        parent = track_dict[ti].hf_ancestor
                        if t_class == 'nm':
                            jet_trk_nm += 1
                            hist_trk_pt_nm.Fill(track_dict[ti].pt)
                            hist_trk_eta_nm.Fill(track_dict[ti].eta)
                            hist_trk_phi_nm.Fill(track_dict[ti].phi)
                            hist_trk_z0_nm.Fill(track_dict[ti].z0)
                        elif t_class == 'b':
                            jet_trk_bh += 1
                            hist_trk_pt_b.Fill(track_dict[ti].pt)
                            hist_trk_eta_b.Fill(track_dict[ti].eta)
                            hist_trk_phi_b.Fill(track_dict[ti].phi)
                            hist_trk_z0_b.Fill(track_dict[ti].z0)
                            bh_parent_list = np.append(bh_parent_list, track_dict[ti].hf_ancestor)
                            hist_fl_len_b.Fill(np.linalg.norm(particle_dict[parent].dv-track_dict[ti].vertex))
                        elif t_class == 'c':
                            jet_trk_ch += 1
                            hist_trk_pt_c.Fill(track_dict[ti].pt)
                            hist_trk_eta_c.Fill(track_dict[ti].eta)
                            hist_trk_phi_c.Fill(track_dict[ti].phi)
                            hist_trk_z0_c.Fill(track_dict[ti].z0)
                            ch_parent_list = np.append(ch_parent_list, track_dict[ti].hf_ancestor)
                            hist_fl_len_c.Fill(np.linalg.norm(particle_dict[parent].dv-track_dict[ti].vertex))
                        elif t_class == 'btoc':
                            jet_trk_btoc += 1
                            hist_trk_pt_btoc.Fill(track_dict[ti].pt)
                            hist_trk_eta_btoc.Fill(track_dict[ti].eta)
                            hist_trk_phi_btoc.Fill(track_dict[ti].phi)
                            hist_trk_z0_btoc.Fill(track_dict[ti].z0)
                            btoc_parent_list = np.append(btoc_parent_list, track_dict[ti].hf_ancestor)
                            hist_fl_len_btoc.Fill(np.linalg.norm(particle_dict[parent].dv-track_dict[ti].vertex))
                        else:
                            jet_trk_o += 1
                            hist_trk_pt_o.Fill(track_dict[ti].pt)
                            hist_trk_eta_o.Fill(track_dict[ti].eta)
                            hist_trk_phi_o.Fill(track_dict[ti].phi)
                            hist_trk_z0_o.Fill(track_dict[ti].z0)

                    #calculate distances between tracks in a particular SV
                    for ti in track_dict:
                        anc_i = track_dict[ti].hf_ancestor
                        for tj in track_dict:
                            anc_j = track_dict[tj].hf_ancestor
                            if ti != tj and anc_i == anc_j:
                                if track_dict[tj].classification == 'b' and track_dict[ti].classification == 'b':
                                    hist_trk_vtx_dist_b.Fill(np.linalg.norm(track_dict[ti].vertex-track_dict[tj].vertex))
                                elif track_dict[tj].classification == 'c' and track_dict[ti].classification == 'c':
                                    hist_trk_vtx_dist_c.Fill(np.linalg.norm(track_dict[ti].vertex-track_dict[tj].vertex))
                                elif track_dict[tj].classification == 'btoc' and track_dict[ti].classification == 'btoc':
                                    hist_trk_vtx_dist_btoc.Fill(np.linalg.norm(track_dict[ti].vertex-track_dict[tj].vertex))

                    #fill histograms
                    if len(track_dict) != 0:
                        hist_no_trk_jet_b.Fill(jet_trk_bh)
                        hist_no_trk_jet_c.Fill(jet_trk_ch)
                        hist_no_trk_jet_o.Fill(jet_trk_o)
                        hist_no_trk_jet_btoc.Fill(jet_trk_btoc)
                        hist_no_trk_jet_nm.Fill(jet_trk_nm)
                        hist_frac_trk_b.Fill(jet_trk_bh/len(track_dict))
                        hist_frac_trk_c.Fill(jet_trk_ch/len(track_dict))
                        hist_frac_trk_o.Fill(jet_trk_o/len(track_dict))
                        hist_frac_trk_btoc.Fill(jet_trk_btoc/len(track_dict))
                        hist_frac_trk_nm.Fill(jet_trk_nm/len(track_dict))

                        bh_parent_list, bh_unique = np.unique(bh_parent_list, return_counts=True)
                        for incidence in bh_unique:
                            hist_no_trk_b.Fill(incidence)
                        for parent in bh_parent_list:
                            bh_children = particle_dict[parent].children
                            bh_charged = 0
                            for barcode in bh_children:
                                if particle_dict[int(barcode)].charged:
                                    bh_charged += 1
                            hist_no_char_b.Fill(bh_charged)

                        ch_parent_list, ch_unique = np.unique(ch_parent_list, return_counts=True)
                        for incidence in ch_unique:
                            hist_no_trk_c.Fill(incidence)
                        for parent in ch_parent_list:
                            ch_children = particle_dict[parent].children
                            ch_charged = 0
                            for barcode in ch_children:
                                if particle_dict[int(barcode)].charged:
                                    ch_charged += 1
                            hist_no_char_c.Fill(ch_charged)
               
                        btoc_parent_list, btoc_unique = np.unique(btoc_parent_list, return_counts=True)
                        for incidence in btoc_unique:
                            hist_no_trk_btoc.Fill(incidence)
                        for parent in btoc_parent_list:
                            btoc_children = particle_dict[parent].children
                            btoc_charged = 0
                            for barcode in btoc_children:
                                if particle_dict[int(barcode)].charged:
                                    btoc_charged += 1
                            hist_no_char_btoc.Fill(btoc_charged)

                    #output progress
                    sys.stdout.write("\rProcessed {} jets. Time elapsed: {:.1f}s".format(processed_jets, time.time()-start_time))
                    sys.stdout.flush()

        if processed_jets>=maxentries: break

    sys.stdout.write("\rFinished processing. Total jets used from sample: {}".format(processed_jets))
    sys.stdout.flush()
    print("\nCreating plots")

    canv1 = TCanvas("c1", "c1", 800, 600)

    plot_hist(canv1, [hist_frac_trk_b, hist_frac_trk_c, hist_frac_trk_btoc], ["bH", "prompt cH", "bH->cH"], False, base_filename+"_frac_trk.png", "", True)
    plot_hist(canv1, [hist_no_char_b, hist_no_char_c, hist_no_char_btoc], ["bH", "prompt cH", "bH->cH"], True, base_filename+"_no_char.png", "", True)
    plot_hist(canv1, [hist_no_trk_b, hist_no_trk_c, hist_no_trk_btoc], ["bH", "prompt cH", "bH->cH"], True, base_filename+"_no_trk.png", "", True)
    plot_hist(canv1, [hist_no_trk_jet_b, hist_no_trk_jet_c, hist_no_trk_jet_btoc, hist_no_trk_jet_o, hist_no_trk_jet_nm], ["bH", "prompt cH", "bH->cH", "other", "no match"], True, base_filename+"_no_trk_jet.png", "", True)
    plot_hist(canv1, [hist_fl_len_b, hist_fl_len_c, hist_fl_len_btoc], ["bH", "prompt cH", "bH->cH"], True, base_filename+"_fl_len.png", "", True)
    plot_hist(canv1, [hist_trk_vtx_dist_b, hist_trk_vtx_dist_c, hist_trk_vtx_dist_btoc], ["bH", "prompt cH", "bH->cH"], True, base_filename+"_trk_vtx_dist.png", "", True)
    plot_hist(canv1, [hist_trk_pt_b, hist_trk_pt_c, hist_trk_pt_btoc, hist_trk_pt_o, hist_trk_pt_nm], ["bH", "prompt cH", "bH->cH", "other", "no match"], True, base_filename+"_trk_pt.png", "HIST", True)
    plot_hist(canv1, [hist_trk_eta_b, hist_trk_eta_c, hist_trk_eta_btoc, hist_trk_eta_o, hist_trk_eta_nm], ["bH", "prompt cH", "bH->cH", "other", "no match"], True, base_filename+"_trk_eta.png", "HIST", True)
    plot_hist(canv1, [hist_trk_phi_b, hist_trk_phi_c, hist_trk_phi_btoc, hist_trk_phi_o, hist_trk_phi_nm], ["bH", "prompt cH", "bH->cH", "other", "no match"], True, base_filename+"_trk_phi.png", "HIST", True)
    plot_hist(canv1, [hist_trk_z0_b, hist_trk_z0_c, hist_trk_z0_btoc, hist_trk_z0_o, hist_trk_z0_nm], ["bH", "prompt cH", "bH->cH", "other", "no match"], True, base_filename+"_trk_z0.png", "HIST", True)
    plot_hist(canv1, [hist_no_trk_acc, hist_no_trk_rej], ["after cuts", "before cuts"], False, base_filename+"_no_trk_cuts.png", "", True)


if __name__ == '__main__':
    main(sys.argv)
