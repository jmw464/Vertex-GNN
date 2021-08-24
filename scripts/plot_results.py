import dgl
import torch as th
import os,sys,math,glob,ROOT
from ROOT import TH1D, TCanvas
import numpy as np
import argparse

import options
from plot_functions import *

#set ATLAS style for plots
gROOT.LoadMacro("/global/homes/j/jmw464/ATLAS/Vertex-GNN/scripts/include/AtlasStyle.C")
gROOT.LoadMacro("/global/homes/j/jmw464/ATLAS/Vertex-GNN/scripts/include/AtlasLabels.C")
from ROOT import SetAtlasStyle


def main(argv):
    gROOT.SetBatch(True)

    #parse command line arguments
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("-r", "--runnumber", type=str, default=0, dest="runnumber", help="unique identifier for current run")
    parser.add_argument("-d", "--data_dir", type=str, required=True, dest="data_dir", help="name of directory where data is stored")
    parser.add_argument("-o", "--output_dir", type=str, required=True, dest="output_dir", help="name of directory where GNN output is stored")
    parser.add_argument("-s", "--dataset", type=str, required=True, dest="infile_name", help="name of dataset to train on (without hdf5 extension)")
    args = parser.parse_args()

    runnumber = args.runnumber
    infile_name = args.infile_name
    infile_path = args.data_dir
    outfile_path = args.output_dir

    #import options from option file
    batch_size = options.batch_size
    atlasstyle = options.use_atlas_style
    track_pt_bound = options.track_pt_bound
    track_d0_bound = options.track_d0_bound
    track_z0_bound = options.track_z0_bound
    jet_pt_bound = options.jet_pt_bound
    jet_eta_bound = options.jet_eta_bound

    if atlasstyle: SetAtlasStyle()

    graphfile_name = outfile_path+runnumber+"/"+infile_name+"_"+runnumber+"_results.bin"
    paramfile_name = infile_path+infile_name+"_params"
    outfile_name = outfile_path+runnumber+"/"+infile_name+"_"+runnumber
    normfile_name = infile_path+infile_name+"_norm"

    #calculate number of features in graphs
    sample_graph = (dgl.load_graphs(graphfile_name, [0]))[0][0]
    incl_errors = incl_corr = incl_hits = False
    nnfeatures_base = sample_graph.ndata['features_base'].size()[1]
    nnfeatures = nnfeatures_base
    if 'features_errors' in sample_graph.ndata.keys():
        nnfeatures_errors = sample_graph.ndata['features_errors'].size()[1]
        incl_errors = True
        nnfeatures += nnfeatures_errors
    if 'features_hits' in sample_graph.ndata.keys():
        nnfeatures_hits = sample_graph.ndata['features_hits'].size()[1]
        incl_hits = True
        nnfeatures += nnfeatures_hits
    if 'features_corr' in sample_graph.ndata.keys():
        nnfeatures_corr = sample_graph.ndata['features_corr'].size()[1]
        incl_corr = True
        nnfeatures += nnfeatures_corr
	
    #read in length of test file
    if os.path.isfile(paramfile_name):
        paramfile = open(paramfile_name, "r")
        test_len = int(paramfile.readline())
    else:
        print("ERROR: Specified parameter file not found")
        return 1
    batches = int(math.ceil(test_len/batch_size))

    #read in normalization constants for features
    if os.path.isfile(normfile_name):
        normfile = open(normfile_name, "r")
        mean_features = np.zeros(nnfeatures)
        std_features = np.zeros(nnfeatures) 
        
        counter = 0
        for line in normfile:
            if int(counter%2) == 0:
                mean_features[int(counter/2)] = float(line)
            else:
                std_features[int((counter-1)/2)] = float(line)
            counter += 1

    hist_trk_pt_b_tot = TH1D("trk_pt_b_tot", "Normalized track pT distribution in test data (total - bad jets);pT [GeV];Difference in normalized entries", 20, track_pt_bound[0], track_pt_bound[1])
    hist_trk_pt_c_tot = TH1D("trk_pt_c_tot", "Normalized track pT distribution in test data (total - bad jets);pT [GeV];Difference in normalized entries", 20, track_pt_bound[0], track_pt_bound[1])
    hist_trk_pt_btoc_tot = TH1D("trk_pt_btoc_tot", "Normalized track pT distribution in test data (total - bad jets);pT [GeV];Difference in normalized entries", 20, track_pt_bound[0], track_pt_bound[1])
    hist_trk_pt_b_bad = TH1D("trk_pt_b_bad", "Normalized track pT distribution in test data (total - bad jets);pT [GeV];Difference in normalized entries", 20, track_pt_bound[0], track_pt_bound[1])
    hist_trk_pt_c_bad = TH1D("trk_pt_c_bad", "Normalized track pT distribution in test data (total - bad jets);pT [GeV];Difference in normalized entries", 20, track_pt_bound[0], track_pt_bound[1])
    hist_trk_pt_btoc_bad = TH1D("trk_pt_btoc_bad", "Normalized track pT distribution in test data (total - bad jets);pT [GeV];Difference in normalized entries", 20, track_pt_bound[0], track_pt_bound[1])
    
    hist_trk_theta_b_tot = TH1D("trk_theta_b_tot", "Normalized track #theta distribution in test data (total - bad jets);#theta;Difference in normalized entries", 20, 0, math.pi)
    hist_trk_theta_c_tot = TH1D("trk_theta_c_tot", "Normalized track #theta distribution in test data (total - bad jets);#theta;Difference in normalized entries", 20, 0, math.pi)
    hist_trk_theta_btoc_tot = TH1D("trk_theta_btoc_tot", "Normalized track #theta distribution in test data (total - bad jets);#theta;Difference in normalized entries", 20, 0, math.pi)
    hist_trk_theta_b_bad = TH1D("trk_theta_b_bad", "Normalized track #theta distribution in test data (total - bad jets);#theta;Difference in normalized entries", 20, 0, math.pi)
    hist_trk_theta_c_bad = TH1D("trk_theta_c_bad", "Normalized track #theta distribution in test data (total - bad jets);#theta;Difference in normalized entries", 20, 0, math.pi)
    hist_trk_theta_btoc_bad = TH1D("trk_theta_btoc_bad", "Normalized track #theta distribution in test data (total - bad jets);#theta;Difference in normalized entries", 20, 0, math.pi) 

    hist_trk_phi_b_tot = TH1D("trk_phi_b_tot", "Normalized track #phi distribution in test data (total - bad jets);#phi;Difference in normalized entries", 20, -math.pi, math.pi)
    hist_trk_phi_c_tot = TH1D("trk_phi_c_tot", "Normalized track #phi distribution in test data (total - bad jets);#phi;Difference in normalized entries", 20, -math.pi, math.pi)
    hist_trk_phi_btoc_tot = TH1D("trk_phi_btoc_tot", "Normalized track #phi distribution in test data (total - bad jets);#phi;Difference in normalized entries", 20, -math.pi, math.pi)
    hist_trk_phi_b_bad = TH1D("trk_phi_b_bad", "Normalized track #phi distribution in test data (total - bad jets);#phi;Difference in normalized entries", 20, -math.pi, math.pi)
    hist_trk_phi_c_bad = TH1D("trk_phi_c_bad", "Normalized track #phi distribution in test data (total - bad jets);#phi;Difference in normalized entries", 20, -math.pi, math.pi)
    hist_trk_phi_btoc_bad = TH1D("trk_phi_btoc_bad", "Normalized track #phi distribution in test data (total - bad jets);#phi;Difference in normalized entries", 20, -math.pi, math.pi)

    hist_trk_z0_b_tot = TH1D("trk_z0_b_tot", "Normalized track z0 distribution in test data (total - bad jets);z0 [cm];Difference in normalized entries", 20, track_z0_bound[0], track_z0_bound[1])
    hist_trk_z0_c_tot = TH1D("trk_z0_c_tot", "Normalized track z0 distribution in test data (total - bad jets);z0 [cm];Difference in normalized entries", 20, track_z0_bound[0], track_z0_bound[1])
    hist_trk_z0_btoc_tot = TH1D("trk_z0_btoc_tot", "Normalized track z0 distribution in test data (total - bad jets);z0 [cm];Difference in normalized entries", 20, track_z0_bound[0], track_z0_bound[1])
    hist_trk_z0_b_bad = TH1D("trk_z0_b_bad", "Normalized track z0 distribution in test data (total - bad jets);z0 [cm];Difference in normalized entries", 20, track_z0_bound[0], track_z0_bound[1])
    hist_trk_z0_c_bad = TH1D("trk_z0_c_bad", "Normalized track z0 distribution in test data (total - bad jets);z0 [cm];Difference in normalized entries", 20, track_z0_bound[0], track_z0_bound[1])
    hist_trk_z0_btoc_bad = TH1D("trk_z0_btoc_bad", "Normalized track z0 distribution in test data (total - bad jets);z0 [cm];Difference in normalized entries", 20, track_z0_bound[0], track_z0_bound[1])

    hist_trk_d0_b_tot = TH1D("trk_d0_b_tot", "Normalized track d0 distribution in test data (total - bad jets);d0 [cm];Difference in normalized entries", 20, track_d0_bound[0], track_d0_bound[1])
    hist_trk_d0_c_tot = TH1D("trk_d0_c_tot", "Normalized track d0 distribution in test data (total - bad jets);d0 [cm];Difference in normalized entries", 20, track_d0_bound[0], track_d0_bound[1])
    hist_trk_d0_btoc_tot = TH1D("trk_d0_btoc_tot", "Normalized track d0 distribution in test data (total - bad jets);d0 [cm];Difference in normalized entries", 20, track_d0_bound[0], track_d0_bound[1])
    hist_trk_d0_b_bad = TH1D("trk_d0_b_bad", "Normalized track d0 distribution in test data (total - bad jets);d0 [cm];Difference in normalized entries", 20, track_d0_bound[0], track_d0_bound[1])
    hist_trk_d0_c_bad = TH1D("trk_d0_c_bad", "Normalized track d0 distribution in test data (total - bad jets);d0 [cm];Difference in normalized entries", 20, track_d0_bound[0], track_d0_bound[1])
    hist_trk_d0_btoc_bad = TH1D("trk_d0_btoc_bad", "Normalized track d0 distribution in test data (total - bad jets);d0 [cm];Difference in normalized entries", 20, track_d0_bound[0], track_d0_bound[1])

    bin_edges = np.linspace(-0.5,6.5,7)
    hist_no_trk_jet_b_tot = TH1D("no_trk_jet_b_tot", "Number of associated tracks per jet in test data;Number of tracks;Normalized entries", 6, bin_edges)
    hist_no_trk_jet_c_tot = TH1D("no_trk_jet_c_tot", "Number of associated tracks per jet in test data;Number of tracks;Normalized entries", 6, bin_edges)
    hist_no_trk_jet_btoc_tot = TH1D("no_trk_jet_btoc_tot", "Number of associated tracks per jet in test data;Number of tracks;Normalized entries", 6, bin_edges)
    hist_no_trk_jet_o_tot = TH1D("no_trk_jet_o_tot", "Number of associated tracks per jet in test data;Number of tracks;Normalized entries", 6, bin_edges)
    hist_no_trk_jet_nm_tot = TH1D("no_trk_jet_nm_tot", "Number of associated tracks per jet in test data;Number of tracks;Normalized entries", 6, bin_edges)
    hist_no_trk_jet_b_bad = TH1D("no_trk_jet_b_bad", "Number of associated tracks per jet among badly reconstructed jets;Number of tracks;Normalized entries", 6, bin_edges)
    hist_no_trk_jet_c_bad = TH1D("no_trk_jet_c_bad", "Number of associated tracks per jet among badly reconstructed jets;Number of tracks;Normalized entries", 6, bin_edges)
    hist_no_trk_jet_btoc_bad = TH1D("no_trk_jet_btoc_bad", "Number of associated tracks per jet among badly reconstructed jets;Number of tracks;Normalized entries", 6, bin_edges)
    hist_no_trk_jet_o_bad = TH1D("no_trk_jet_o_bad", "Number of associated tracks per jet among badly reconstructed jets;Number of tracks;Normalized entries", 6, bin_edges)
    hist_no_trk_jet_nm_bad = TH1D("no_trk_jet_nm_bad", "Number of associated tracks per jet among badly reconstructed jets;Number of tracks;Normalized entries", 6, bin_edges)

    hist_frac_trk_b_tot = TH1D("frac_trk_b_tot", "Fraction of tracks per jet in test data;Track fraction;Entries", 10, 0, 1)
    hist_frac_trk_c_tot = TH1D("frac_trk_c_tot", "Fraction of tracks per jet in test data;Track fraction;Entries", 10, 0, 1)
    hist_frac_trk_o_tot = TH1D("frac_trk_o_tot", "Fraction of tracks per jet; in test dataTrack fraction;Entries", 10, 0, 1)
    hist_frac_trk_nm_tot = TH1D("frac_trk_nm_tot", "Fraction of tracks per jet in test data;Track fraction;Entries", 10, 0, 1)
    hist_frac_trk_btoc_tot = TH1D("frac_trk_btoc_tot", "Fraction of tracks per jet in test data;Track fraction;Entries", 10, 0, 1)
    hist_frac_trk_b_bad = TH1D("frac_trk_b_bad", "Fraction of tracks per jet among badly reconstructed jets;Track fraction;Entries", 10, 0, 1)
    hist_frac_trk_c_bad = TH1D("frac_trk_c_bad", "Fraction of tracks per jet among badly reconstructed jets;Track fraction;Entries", 10, 0, 1)
    hist_frac_trk_o_bad = TH1D("frac_trk_o_bad", "Fraction of tracks per jet among badly reconstructed jets;Track fraction;Entries", 10, 0, 1)
    hist_frac_trk_nm_bad = TH1D("frac_trk_nm_bad", "Fraction of tracks per jet among badly reconstructed jets;Track fraction;Entries", 10, 0, 1)
    hist_frac_trk_btoc_bad = TH1D("frac_trk_btoc_bad", "Fraction of tracks per jet among badly reconstructed jets;Track fraction;Entries", 10, 0, 1)

    for ibatch in range(batches):
        
        #calculate batch indices
        istart = ibatch*batch_size
        if ibatch == (batches-1) and test_len%batch_size != 0:
            iend = istart + (test_len%batch_size)
        else:
            iend = (ibatch+1)*batch_size

        #load batch from file
        batch = dgl.batch(dgl.load_graphs(graphfile_name, list(range(istart, iend)))[0])
        g_list = dgl.unbatch(batch)

        for g in g_list:
            features = g.ndata['features_base'].numpy()
            bad_jet = g.ndata['bad'].numpy()[0,0]
            categories = g.ndata['node_info'].numpy()

            ntrk = len(features[:,0])
            b_trk = c_trk = btoc_trk = nm_trk = o_trk = 0
            for i in range(ntrk):
                if categories[i,2] == 0:
                    nm_trk += 1
                elif categories[i,2] == 1:
                    b_trk += 1
                    hist_trk_pt_b_tot.Fill(abs(1/(features[i,0]*std_features[0]+mean_features[0])))
                    hist_trk_theta_b_tot.Fill(features[i,1]*std_features[1]+mean_features[1])
                    hist_trk_phi_b_tot.Fill(features[i,2]*std_features[2]+mean_features[2])
                    hist_trk_d0_b_tot.Fill(features[i,3]*std_features[3]+mean_features[3])
                    hist_trk_z0_b_tot.Fill(features[i,4]*std_features[4]+mean_features[4])
                    if bad_jet:
                        hist_trk_pt_b_bad.Fill(abs(1/(features[i,0]*std_features[0]+mean_features[0])))
                        hist_trk_theta_b_bad.Fill(features[i,1]*std_features[1]+mean_features[1])
                        hist_trk_phi_b_bad.Fill(features[i,2]*std_features[2]+mean_features[2])
                        hist_trk_d0_b_bad.Fill(features[i,3]*std_features[3]+mean_features[3])
                        hist_trk_z0_b_bad.Fill(features[i,4]*std_features[4]+mean_features[4])
                elif categories[i,2] == 2:
                    c_trk += 1
                    hist_trk_pt_c_tot.Fill(abs(1/(features[i,0]*std_features[0]+mean_features[0])))
                    hist_trk_theta_c_tot.Fill(features[i,1]*std_features[1]+mean_features[1])
                    hist_trk_phi_c_tot.Fill(features[i,2]*std_features[2]+mean_features[2])
                    hist_trk_d0_c_tot.Fill(features[i,3]*std_features[3]+mean_features[3])
                    hist_trk_z0_c_tot.Fill(features[i,4]*std_features[4]+mean_features[4])
                    if bad_jet:
                        hist_trk_pt_c_bad.Fill(abs(1/(features[i,0]*std_features[0]+mean_features[0])))
                        hist_trk_theta_c_bad.Fill(features[i,1]*std_features[1]+mean_features[1])
                        hist_trk_phi_c_bad.Fill(features[i,2]*std_features[2]+mean_features[2])
                        hist_trk_d0_c_bad.Fill(features[i,3]*std_features[3]+mean_features[3])
                        hist_trk_z0_c_bad.Fill(features[i,4]*std_features[4]+mean_features[4])
                elif categories[i,2] == 3:
                    btoc_trk += 1
                    hist_trk_pt_btoc_tot.Fill(abs(1/(features[i,0]*std_features[0]+mean_features[0])))
                    hist_trk_theta_btoc_tot.Fill(features[i,1]*std_features[1]+mean_features[1])
                    hist_trk_phi_btoc_tot.Fill(features[i,2]*std_features[2]+mean_features[2])
                    hist_trk_d0_btoc_tot.Fill(features[i,3]*std_features[3]+mean_features[3])
                    hist_trk_z0_btoc_tot.Fill(features[i,4]*std_features[4]+mean_features[4])
                    if bad_jet:
                        hist_trk_pt_btoc_bad.Fill(abs(1/(features[i,0]*std_features[0]+mean_features[0])))
                        hist_trk_theta_btoc_bad.Fill(features[i,1]*std_features[1]+mean_features[1])
                        hist_trk_phi_btoc_bad.Fill(features[i,2]*std_features[2]+mean_features[2])
                        hist_trk_d0_btoc_bad.Fill(features[i,3]*std_features[3]+mean_features[3])
                        hist_trk_z0_btoc_bad.Fill(features[i,4]*std_features[4]+mean_features[4])
                elif categories[i,2] == 4:
                    o_trk += 1

            hist_no_trk_jet_b_tot.Fill(b_trk)
            hist_no_trk_jet_c_tot.Fill(c_trk)
            hist_no_trk_jet_btoc_tot.Fill(btoc_trk)
            hist_no_trk_jet_nm_tot.Fill(nm_trk)
            hist_no_trk_jet_o_tot.Fill(o_trk)
            hist_frac_trk_b_tot.Fill(b_trk/ntrk)
            hist_frac_trk_c_tot.Fill(c_trk/ntrk)
            hist_frac_trk_btoc_tot.Fill(btoc_trk/ntrk)
            hist_frac_trk_nm_tot.Fill(nm_trk/ntrk)
            hist_frac_trk_o_tot.Fill(o_trk/ntrk)
            if bad_jet:
                hist_no_trk_jet_b_bad.Fill(b_trk)
                hist_no_trk_jet_c_bad.Fill(c_trk)
                hist_no_trk_jet_btoc_bad.Fill(btoc_trk)
                hist_no_trk_jet_nm_bad.Fill(nm_trk)
                hist_no_trk_jet_o_bad.Fill(o_trk) 
                hist_frac_trk_b_bad.Fill(b_trk/ntrk)
                hist_frac_trk_c_bad.Fill(c_trk/ntrk)
                hist_frac_trk_btoc_bad.Fill(btoc_trk/ntrk)
                hist_frac_trk_nm_bad.Fill(nm_trk/ntrk)
                hist_frac_trk_o_bad.Fill(o_trk/ntrk)

    base_filename = outfile_path+runnumber+"/"+infile_name+"_"+runnumber

    plot_hist_diff([hist_trk_pt_b_tot, hist_trk_pt_c_tot, hist_trk_pt_btoc_tot], [hist_trk_pt_b_bad, hist_trk_pt_c_bad, hist_trk_pt_btoc_bad], ["bH", "prompt cH", "bH->cH"], False, True, base_filename+"_trk_pt_comp.png", "HIST")
    plot_hist_diff([hist_trk_theta_b_tot, hist_trk_theta_c_tot, hist_trk_theta_btoc_tot], [hist_trk_theta_b_bad, hist_trk_theta_c_bad, hist_trk_theta_btoc_bad], ["bH", "prompt cH", "bH->cH"], False, True, base_filename+"_trk_theta_comp.png", "HIST")
    plot_hist_diff([hist_trk_phi_b_tot, hist_trk_phi_c_tot, hist_trk_phi_btoc_tot], [hist_trk_phi_b_bad, hist_trk_phi_c_bad, hist_trk_phi_btoc_bad], ["bH", "prompt cH", "bH->cH"], False, True, base_filename+"_trk_phi_comp.png", "HIST")
    plot_hist_diff([hist_trk_d0_b_tot, hist_trk_d0_c_tot, hist_trk_d0_btoc_tot], [hist_trk_d0_b_bad, hist_trk_d0_c_bad, hist_trk_d0_btoc_bad], ["bH", "prompt cH", "bH->cH"], False, True, base_filename+"_trk_d0_comp.png", "HIST")
    plot_hist_diff([hist_trk_z0_b_tot, hist_trk_z0_c_tot, hist_trk_z0_btoc_tot], [hist_trk_z0_b_bad, hist_trk_z0_c_bad, hist_trk_z0_btoc_bad], ["bH", "prompt cH", "bH->cH"], False, True, base_filename+"_trk_z0_comp.png", "HIST")
    plot_hist([hist_no_trk_jet_b_tot, hist_no_trk_jet_c_tot, hist_no_trk_jet_btoc_tot, hist_no_trk_jet_o_tot, hist_no_trk_jet_nm_tot], ["bH", "prompt cH", "bH->cH", "other", "no match"], True, False, True, base_filename+"_no_trk_tot.png", "HIST")
    plot_hist([hist_frac_trk_b_tot, hist_frac_trk_c_tot, hist_frac_trk_btoc_tot, hist_frac_trk_o_tot, hist_frac_trk_nm_tot], ["bH", "prompt cH", "bH->cH", "other", "no match"], True, False, True, base_filename+"_frac_trk_tot.png", "HIST")
    plot_hist([hist_no_trk_jet_b_bad, hist_no_trk_jet_c_bad, hist_no_trk_jet_btoc_bad, hist_no_trk_jet_o_bad, hist_no_trk_jet_nm_bad], ["bH", "prompt cH", "bH->cH", "other", "no match"], True, False, True, base_filename+"_no_trk_bad.png", "HIST")
    plot_hist([hist_frac_trk_b_bad, hist_frac_trk_c_bad, hist_frac_trk_btoc_bad, hist_frac_trk_o_bad, hist_frac_trk_nm_bad], ["bH", "prompt cH", "bH->cH", "other", "no match"], True, False, True, base_filename+"_frac_trk_bad.png", "HIST") 


if __name__ == '__main__':
    main(sys.argv)
