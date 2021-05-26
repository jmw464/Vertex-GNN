#!/usr/bin/env python

import os,sys,math,ROOT,glob
import numpy as np
import argparse
from ROOT import TFile, TH1D, TH1I, gROOT, TCanvas, gPad, TLegend
import time

#############################################SCRIPT PARAMS#################################################

remove_pv = True

jetpt_cut = 20000 #20 GeV
jeteta_cut = 2.5 #edge of detector
trackpt_cut = 650 #600 MeV
tracketa_cut = 2.5 #edge of detector
trackz0_cut = 20

#set thresholds for track classification
threshold_dist = 5 #maximum distance between HF vertex and track vertex
bc_threshold_dist = 5 #maximum distance between b decay vertex and c production vertex for bH->cH tracks

###########################################################################################################


class truth_particle():
    def __init__(self, barcode, pdgid, pv, dv, charged, p, parents, children):
        self.barcode = barcode
        self.pdgid = pdgid
        self.pv = pv
        self.dv = dv
        self.charged = charged
        self.p = p
        self.parents = parents
        self.children = children


class truth_track():
    def __init__(self, barcode, pdgid, vertex, pt, eta, phi):
        self.barcode = barcode
        self.pdgid = pdgid
        self.vertex = vertex
        self.pt = pt
        self.eta = eta
        self.phi = phi
        self.hf_ancestor = -1 #overall HF ancestor (always corresponds to b hadron for bH->cH tracks)
        self.btoc_ancestor = -1 #direct charm ancestor for bH->cH tracks from charm hadrons
        self.classification = ""


def id_particle(pdgid):
    wd_cm = [411, 421, 431] #weakly decaying charm mesons
    wd_cb = [4122, 4132, 4232, 4212, 4332] #weakly decaying charm baryons
    wd_bm = [511, 521, 531, 541] #weakly decaying bottom mesons
    wd_bb = [5122, 5132, 5232, 5112, 5212, 5222, 5332] #weakly decaying bottom baryons

    if abs(pdgid) in wd_cm or abs(pdgid) in wd_cb:
        return 'ch'
    elif abs(pdgid) in wd_bm or abs(pdgid) in wd_bb:
        return 'bh'
    else:
        return 'other'


#function to retrieve most recent independent (non-interrelated) HF relatives (descendants or antecedents) of a given particle
def get_hf_relatives(particle, particle_dict, particle_list, mode, skip): #skip is used when examining the HF ancestors of a HF hadron (so the function can enter the recursive part)
    barcode = particle.barcode
    pdgid = particle.pdgid
    label = id_particle(pdgid)
    
    if not skip and (label == 'ch' or label == 'bh'):
        particle_list = np.append(particle_list, barcode)
        return particle_list #stop searching branch if HF relative is found
    
    if mode == 'a':
        for parent in particle.parents:
            particle_list = get_hf_relatives(particle_dict[parent], particle_dict, particle_list, mode, 0)
    elif mode == 'd':
        for child in particle.children:
            particle_list = get_hf_relatives(particle_dict[child], particle_dict, particle_list, mode, 0)

    return particle_list


def classify_track(index, particle_dict, track_dict):
    barcode = track_dict[index].barcode
    hf_barcode = track_dict[index].hf_ancestor
    track_vertex = track_dict[index].vertex
    if hf_barcode != -1: primary_distance = np.linalg.norm(track_vertex-particle_dict[hf_barcode].dv)

    min_distance = second_ancestor = -1
    if barcode < -990:
        return 'nm' #no truth particle match
    elif hf_barcode != -1 and id_particle(particle_dict[hf_barcode].pdgid) == 'ch' and primary_distance <= threshold_dist:
        ch_barcodes = get_hf_relatives(particle_dict[hf_barcode], particle_dict, np.array([]), 'a', 1)
        for ch_barcode in ch_barcodes:
            distance = np.linalg.norm(particle_dict[hf_barcode].pv - particle_dict[ch_barcode].dv)
            if (min_distance == -1 or distance < min_distance) and id_particle(particle_dict[ch_barcode].pdgid) == 'bh':
                min_distance = distance
                second_ancestor = ch_barcode
        if second_ancestor != -1 and min_distance <= bc_threshold_dist:
            track_dict[index].btoc_ancestor = track_dict[index].hf_ancestor
            track_dict[index].hf_ancestor = second_ancestor
            return 'btoc'
        else:
            return 'c'
    elif hf_barcode != -1 and id_particle(particle_dict[hf_barcode].pdgid) == 'bh' and primary_distance <= threshold_dist:
        #bh_barcodes = get_hf_relatives(particle_dict[hf_barcode], particle_dict, np.array([]), 'd', 1)
        #for bh_barcode in bh_barcodes:
        #    distance = np.linalg.norm(particle_dict[hf_barcode].dv - particle_dict[bh_barcode].pv)
        #    if (min_distance == -1 or distance < min_distance) and id_particle(particle_dict[bh_barcode].pdgid) == 'ch':
        #        min_distance = distance
        #        second_ancestor = bh_barcode
        #if second_ancestor != -1 and min_distance <= bc_threshold_dist:
        #    return 'btoc'
        #else:
        return 'b'
    else:
        return 'o'


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
    
    hist_fl_len_b = TH1D("fl_len_b", "Distance between HF hadron decay vertex and track vertex;Distance [cm];Entries", 20, 0, 50)
    hist_fl_len_c = TH1D("fl_len_c", "Distance between HF hadron decay vertex and track vertex;Distance [cm];Entries", 20, 0, 50)
    hist_fl_len_btoc = TH1D("fl_len_btoc", "Distance between HF hadron decay vertex and track vertex;Distance [cm];Entries", 20, 0, 50)

    hist_trk_vtx_dist_b = TH1D("trk_vtx_dist_b", "Distance between track PVs within SV;Distance [cm];Entries", 20, 0, 50)
    hist_trk_vtx_dist_c = TH1D("trk_vtx_dist_c", "Distance between track PVs within SV;Distance [cm];Entries", 20, 0, 50)
    hist_trk_vtx_dist_btoc = TH1D("trk_vtx_dist_btoc", "Distance between track PVs within SV;Distance [cm];Entries", 20, 0, 50)

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

    hist_trk_pv_d0_rej = TH1D("trk_pv_d0_rej", "d0 of tracks associated with reconstructed PV;Distance [cm];Normalized entries", 100, -100, 100)
    hist_trk_pv_d0_acc = TH1D("trk_pv_d0_acc", "d0 of tracks not associated with reconstructed PV;Distance [cm];Normalized entries", 100, -100, 100)
    hist_trk_nm_d0_rej = TH1D("trk_nm_d0_rej", "d0 of tracks associated with reconstructed PV;Distance [cm];Normalized entries", 100, -100, 100)
    hist_trk_nm_d0_acc = TH1D("trk_nm_d0_acc", "d0 of tracks not associated with reconstructed PV;Distance [cm];Normalized entries", 100, -100, 100)
    hist_trk_o_d0_rej = TH1D("trk_o_d0_rej", "d0 of tracks associated with reconstructed PV;Distance [cm];Normalized entries", 100, -100, 100)
    hist_trk_o_d0_acc = TH1D("trk_o_d0_acc", "d0 of tracks not associated with reconstructed PV;Distance [cm];Normalized entries", 100, -100, 100)

    hist_trk_pv_z0_rej = TH1D("trk_pv_z0_rej", "z0 of tracks associated with reconstructed PV;Distance [cm];Normalized entries", 100, -100, 100)
    hist_trk_pv_z0_acc = TH1D("trk_pv_z0_acc", "z0 of tracks not associated with reconstructed PV;Distance [cm];Normalized entries", 100, -100, 100)
    hist_trk_nm_z0_rej = TH1D("trk_nm_z0_rej", "z0 of tracks associated with reconstructed PV;Distance [cm];Normalized entries", 100, -100, 100)
    hist_trk_nm_z0_acc = TH1D("trk_nm_z0_acc", "z0 of tracks not associated with reconstructed PV;Distance [cm];Normalized entries", 100, -100, 100)
    hist_trk_o_z0_rej = TH1D("trk_o_z0_rej", "z0 of tracks associated with reconstructed PV;Distance [cm];Normalized entries", 100, -100, 100)
    hist_trk_o_z0_acc = TH1D("trk_o_z0_acc", "z0 of tracks not associated with reconstructed PV;Distance [cm];Normalized entries", 100, -100, 100)

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
            particle_dict = {}

            #check event to see if it can be skipped
            event_pass = False
            for i in range(njets):
                if (not only_bad_events or i in bad_events[ientry]) and check_jet(entry,i):
                    rem_trk = 0
                    nTrack =  entry.jet_trk_pt[i].size()
                    for j in range(nTrack):
                        if check_track(entry, i, j) and (not remove_pv or not entry.jet_trk_isPV_reco[i][j]):
                            rem_trk += 1
                    if rem_trk > 1:
                        event_pass = True
            if not event_pass:
                continue

            #build particle dictionary
            nTruth = entry.truth_pdgId.size()
            for i in range ( nTruth ):
                truth_pdgId = entry.truth_pdgId[i]
                truth_barcode = entry.truth_barcode[i]
                truth_status = entry.truth_status[i]
                truth_charged = entry.truth_isCharged[i]

                truth_pv = np.array([entry.truth_pvtx_x[i], entry.truth_pvtx_y[i], entry.truth_pvtx_z[i]])
                truth_dv = np.array([entry.truth_dvtx_x[i], entry.truth_dvtx_y[i], entry.truth_dvtx_z[i]])
                truth_p = np.array([entry.truth_px[i], entry.truth_py[i], entry.truth_pz[i]])

                truth_nParent = entry.truth_parent_pdgId[i].size()
                truth_nChild = entry.truth_child_pdgId[i].size()

                child_pdgId_array = np.zeros(truth_nChild)
                child_barcode_array = np.zeros(truth_nChild)
                for j in range (truth_nChild):
                    child_pdgId = entry.truth_child_pdgId[i][j]
                    child_pdgId_array[j] = child_pdgId
                    child_barcode = entry.truth_child_barcode[i][j]
                    child_barcode_array[j] = child_barcode

                parent_pdgId_array = np.zeros(truth_nParent)
                parent_barcode_array = np.zeros(truth_nParent)
                for j in range (truth_nParent):
                    parent_pdgId = entry.truth_parent_pdgId[i][j]
                    parent_pdgId_array[j] = parent_pdgId
                    parent_barcode = entry.truth_parent_barcode[i][j]
                    parent_barcode_array[j] = parent_barcode

                particle_dict[truth_barcode] = truth_particle(truth_barcode, truth_pdgId, truth_pv, truth_dv, truth_charged, truth_p, parent_barcode_array, child_barcode_array)

            primary_vertex = np.array([entry.truth_PVx, entry.truth_PVy, entry.truth_PVz])

            #build track dictionary for each jet
            for i in range( njets ):

                #only process jets on the register if only_bad_events is true and make sure they pass cuts
                if (not only_bad_events or i in bad_events[ientry]) and check_jet(entry,i):
                    track_dict = {}
                    jet_pt  = entry.jet_pt[i]
                    jet_eta = entry.jet_eta[i]
                    jet_phi = entry.jet_phi[i]
                    jet_m   = entry.jet_m[i]
                    jet_label = entry.jet_LabDr_HadF[i]

                    nTrack =  entry.jet_trk_pt[i].size()
                    jet_cut_trk = 0

                    #count tracks that pass cuts to know which jets to skip entirely
                    rem_trk = 0
                    for j in range(nTrack):
                        if (not remove_pv or not entry.jet_trk_isPV_reco[i][j]) and check_track(entry, i, j):
                            rem_trk += 1
                    if rem_trk <= 1:
                        continue

                    processed_jets += 1
                    
                    for j in range(nTrack):
                        trk_vertex = np.array([entry.jet_trk_vtx_X[i][j], entry.jet_trk_vtx_Y[i][j], entry.jet_trk_vtx_Z[i][j]])

                        if check_track(entry, i, j):
                            if not remove_pv or not entry.jet_trk_isPV_reco[i][j]:
                                trk_pt = entry.jet_trk_pt[i][j]
                                trk_eta = entry.jet_trk_eta[i][j]
                                trk_phi = entry.jet_trk_phi[i][j]
                                trk_pdgId = entry.jet_trk_pdg_id[i][j]
                                trk_barcode = entry.jet_trk_barcode[i][j]
                                
                                track_dict[j] = truth_track(trk_barcode, trk_pdgId, trk_vertex, trk_pt, trk_eta, trk_phi)

                                if trk_barcode < 0:
                                    hist_trk_nm_d0_acc.Fill(entry.jet_trk_d0[i][j])
                                    hist_trk_nm_z0_acc.Fill(entry.jet_trk_z0[i][j])
                                elif np.linalg.norm(trk_vertex-primary_vertex) < 1e-4:
                                    hist_trk_pv_d0_acc.Fill(entry.jet_trk_d0[i][j])
                                    hist_trk_pv_z0_acc.Fill(entry.jet_trk_z0[i][j])
                                else:
                                    hist_trk_o_d0_acc.Fill(entry.jet_trk_d0[i][j])
                                    hist_trk_o_z0_acc.Fill(entry.jet_trk_z0[i][j])

                            else:
                                jet_cut_trk += 1

                                if trk_vertex[0] < -990:
                                    hist_trk_nm_d0_rej.Fill(entry.jet_trk_d0[i][j])
                                    hist_trk_nm_z0_rej.Fill(entry.jet_trk_z0[i][j])
                                elif np.linalg.norm(trk_vertex-primary_vertex) < 1e-4:
                                    hist_trk_pv_d0_rej.Fill(entry.jet_trk_d0[i][j])
                                    hist_trk_pv_z0_rej.Fill(entry.jet_trk_z0[i][j])
                                else:
                                    hist_trk_o_d0_rej.Fill(entry.jet_trk_d0[i][j])
                                    hist_trk_o_z0_rej.Fill(entry.jet_trk_z0[i][j])
                        
                        else:
                            jet_cut_trk += 1
                            
                    hist_no_trk_acc.Fill(len(track_dict))
                    hist_no_trk_rej.Fill(len(track_dict)+jet_cut_trk)

                    jet_npart_trk = 0 #count tracks not associated with any particles
                    for ti in track_dict:
                        track = track_dict[ti]

                        #don't process tracks that don't have associated truth particles
                        if track.pdgid == -999:
                            jet_npart_trk += 1
                            continue

                        #get direct HF ancestors of track particle
                        t_barcode = track.barcode
                        track_particle = particle_dict[t_barcode]
                        ancestors = np.array([])
                        ancestors = get_hf_relatives(track_particle, particle_dict, ancestors, 'a', 0)
                        ancestors = np.unique(ancestors) #only keep unique barcodes

                        #check which ancestor will be marked as primary (by checking distance between HF DV and track PV)
                        min_distance = direct_ancestor = -1
                        for ancestor in ancestors:
                            distance = np.linalg.norm(track_particle.pv - particle_dict[ancestor].dv)
                            if min_distance == -1 or distance < min_distance:
                                min_distance = distance
                                direct_ancestor = ancestor
                        track.hf_ancestor = direct_ancestor

                    jet_trk_btoc = jet_trk_ch = jet_trk_bh = jet_trk_o = jet_trk_nm = 0
                    bh_parent_list = np.array([]) #list that stores all direct bH parent particles in jet
                    ch_parent_list = np.array([])
                    btoc_parent_list = np.array([])
                    
                    #make first attempt at track classification (still need to correct some bH labels to bH->cH after this point)
                    for ti in track_dict:
                        t_class = classify_track(ti, particle_dict, track_dict)
                        track_dict[ti].classification = t_class
                        if t_class == 'btoc':
                            btoc_parent_list = np.append(btoc_parent_list, track_dict[ti].hf_ancestor)

                    #fix classifications and fill histograms
                    for ti in track_dict:
                        if track_dict[ti].classification == 'b' and track_dict[ti].hf_ancestor in btoc_parent_list:
                            track_dict[ti].classification = 'btoc'
                            btoc_parent_list = np.append(btoc_parent_list, track_dict[ti].hf_ancestor)
                        t_class = track_dict[ti].classification
              
                        parent = track_dict[ti].hf_ancestor
                        if t_class == 'nm':
                            jet_trk_nm += 1
                            hist_trk_pt_nm.Fill(track_dict[ti].pt)
                            hist_trk_eta_nm.Fill(track_dict[ti].eta)
                            hist_trk_phi_nm.Fill(track_dict[ti].phi)
                        elif t_class == 'b':
                            jet_trk_bh += 1
                            hist_trk_pt_b.Fill(track_dict[ti].pt)
                            hist_trk_eta_b.Fill(track_dict[ti].eta)
                            hist_trk_phi_b.Fill(track_dict[ti].phi)
                            bh_parent_list = np.append(bh_parent_list, track_dict[ti].hf_ancestor)
                            hist_fl_len_b.Fill(np.linalg.norm(particle_dict[parent].dv-track_dict[ti].vertex))
                        elif t_class == 'c':
                            jet_trk_ch += 1
                            hist_trk_pt_c.Fill(track_dict[ti].pt)
                            hist_trk_eta_c.Fill(track_dict[ti].eta)
                            hist_trk_phi_c.Fill(track_dict[ti].phi)
                            ch_parent_list = np.append(ch_parent_list, track_dict[ti].hf_ancestor)
                            hist_fl_len_c.Fill(np.linalg.norm(particle_dict[parent].dv-track_dict[ti].vertex))
                        elif t_class == 'btoc':
                            jet_trk_btoc += 1
                            hist_trk_pt_btoc.Fill(track_dict[ti].pt)
                            hist_trk_eta_btoc.Fill(track_dict[ti].eta)
                            hist_trk_phi_btoc.Fill(track_dict[ti].phi)
                            cH_parent = track_dict[ti].btoc_ancestor
                            hist_fl_len_btoc.Fill(np.linalg.norm(particle_dict[parent].dv-track_dict[ti].vertex))
                        else:
                            jet_trk_o += 1
                            hist_trk_pt_o.Fill(track_dict[ti].pt)
                            hist_trk_eta_o.Fill(track_dict[ti].eta)
                            hist_trk_phi_o.Fill(track_dict[ti].phi)

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
    plot_hist(canv1, [hist_no_trk_acc, hist_no_trk_rej], ["after cuts", "before cuts"], False, base_filename+"_no_trk_cuts.png", "", True)
    plot_hist(canv1, [hist_trk_pv_d0_acc, hist_trk_o_d0_acc, hist_trk_nm_d0_acc], ["truth PV associated", "not truth PV associated", "no match"], True, base_filename+"_trk_d0_acc.png", "HIST", False)
    plot_hist(canv1, [hist_trk_pv_z0_acc, hist_trk_o_z0_acc, hist_trk_nm_z0_acc], ["truth PV associated", "not truth PV associated", "no match"], True, base_filename+"_trk_z0_acc.png", "HIST", False)
    plot_hist(canv1, [hist_trk_pv_d0_rej, hist_trk_o_d0_rej, hist_trk_nm_d0_rej], ["truth PV associated", "not truth PV associated", "no match"], True, base_filename+"_trk_d0_rej.png", "HIST", False)
    plot_hist(canv1, [hist_trk_pv_z0_rej, hist_trk_o_z0_rej, hist_trk_nm_z0_rej], ["truth PV associated", "not truth PV associated", "no match"], True, base_filename+"_trk_z0_rej.png", "HIST", False)


if __name__ == '__main__':
    main(sys.argv)
