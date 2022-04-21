#!/usr/bin/env python


#########################
# CHANGEABLE PARAMETERS #
#########################


#cuts on data - PROCESSING
jet_pt_cut = 20 #minimum required jet pT value (default: >20 GeV)
jet_eta_cut = 2.5 #maximum allowed |jet eta| value (default <2.5)
track_pt_cut = 0.5 #0.65 #minimum required track pT value
track_eta_cut = 2.5 #2.5 #maximum allowed |track eta| value (default: <2.5)
track_z0_cut = 25. #20 #maximum allowed |track z0| value
vweight_pileup_cut = 1.0 #maximum allowed vertex weight value (pileup vertex association)
vweight_pv_cut = 1.0 #maximum allowed vertex weight value (primary vertex association)

#input data parameters - PROCESSING, GNN
incl_errors = True #include diagonal covariance matrix GNN features
incl_corr = True #include off-diagonal covariance matrix GNN features
incl_hits = True #include GNN features related to low-level hit information
incl_vweight = True #include vertex weight as GNN feature

#neural network options - GNN
use_gpu = True #toggle whether to use GPU for GNN training if available
load_checkpoint = False #toggle whether to load previous neural network checkpoint (continue training from previous point)
valp = 0.2 #fraction of data reserved for validation
testp = 0.1 #fraction of data reserved for testing

#neural network hyperparameters - GNN
batch_size = 1000 #number of graphs contained in a single batch for training, testing and validation
learning_rate = 0.001 #learning rate used for neural network training
dropout = 0.1

#neural network evaluation criteria - GNN
mult_threshold = [0.5, 0.5, 0.5] #threshold for recall of jets to be marked as bad in multi class classification (none, b, c, b->c, other)
bin_threshold = [0.5, 0.7] #threshold for recall of jets to be marked as bad in binary classiciation (False, True)
score_threshold = 0.6 #threshold score for two edges to be associated to a reconstructed secondary vertex

#neural network model parameters - GNN
attention_heads = [2, 2] #number of attention heads in GAT layers
nodemlp_sizes = [64, 64] #number of nodes in NodeMLP hidden layers
gat_sizes = [256, 512] #layer sizes in GAT hidden layers (if attention_heads > 1, output sizes need to be multiplied by attention_heads for each layer)
edgemlp_sizes = [1024, 256, 64] #excluding output features layer sizes in EdgeMLP hidden layers
reweight = True #toggle whether positive labels in loss are reweighted to make positives and negatives equally important
#ADD REWEIGHTING FACTOR FOR BINARY AND MULTI-CLASS
use_lr_scheduler = False #toggle whether to use learning rate schedule during training

#truth definitions - PROCESSING, PLOTTING
vertex_threshold = 0 #maximum distance for non HF tracks to be marked as part of the same vertex ("other" category)
connect_btoc = True #toggle whether to combine tracks from b hadrons and all c hadrons in B->C SV's or separate them based on their direct HF ancestors

#plotting options - PLOTTING
cut_string = "p_{T} > 20 GeV, |#eta| < 2.5"
plot_roc = True #plotting ROC curve significantly increases computation time for compare_performance.py
track_pt_bound = [track_pt_cut,10] #lower bound needs to be > 0
track_pt_err_bound = [10,1000] #lower bound needs to be > 0
track_theta_err_bound = 0.01
track_phi_err_bound = 0.01
track_d0_bound = 25
track_d0_err_bound = 1
track_z0_bound = track_z0_cut
track_z0_err_bound = 0.5
ntrk_bound = 20 #upper bound on number of tracks in plots
lxy_bound = 50
jet_pt_bound = [jet_pt_cut,200] #boundary jet pT [GeV] for plots
jet_eta_bound = 2.5 #boundary jet eta for plots
