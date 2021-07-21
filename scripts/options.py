#changeable parameters are values that can be changed here freely without having to edit the scripts
#constant parameters are values that are more deeply integrated and require changes to be made to the scripts themselves

#######################
# CONSTANT PARAMETERS #
#######################

#input data parameters - PROCESSING
nnfeatures_base = 8 #number of core GNN features (track parameters)
nnfeatures_errors = 5 #number of diagonal covariance matrix GNN features
nnfeatures_corrs = 10 #number of off-diagonal covariance matrix GNN features
nnfeatures_hits = 10 #number of GNN features related to low-level hit information
nefeatures = 0 #number of features per edge

#########################
# CHANGEABLE PARAMETERS #
#########################

#cuts on data - PROCESSING
jet_pt_cut = 20000 #minimum required jet pT value (default: >20 GeV)
jet_eta_cut = 2.5 #maximum allowed |jet eta| value (default <2.5)
track_pt_cut = 650 #minimum required track pT value
track_eta_cut = 2.5 #maximum allowed |track eta| value (default: <2.5)
track_z0_cut = 20 #maximum allowed |track z0| value
remove_pv = True #toggle to remove all tracks that are already associated with primary vertices

#input data parameters - PROCESSING, GNN
incl_errors = True #include diagonal covariance matrix GNN features
incl_corr = False #include off-diagonal covariance matrix GNN features
incl_hits = False #include GNN features related to low-level hit information

#neural network options
load_checkpoint = False #toggle whether to load previous neural network checkpoint (continue training from previous point)
valp = 0.2 #fraction of data reserved for validation
testp = 0.1 #fraction of data reserved for testing

#neural network hyperparameters - GNN
batch_size = 10000 #number of graphs contained in a single batch for training, testing and validation
learning_rate = 0.001 #learning rate used for neural network training

#neural network evaluation criteria - GNN
mult_threshold = [0.5, 0.1, 0.1, 0.7, 0.1] #threshold for recall of jets to be marked as bad in multi class classification (none, b, c, b->c, other)
bin_threshold = [0.5, 0.7] #threshold for recall of jets to be marked as bad in binary classiciation (False, True)
score_threshold = 0.6 #threshold score for two edges to be associated to a reconstructed secondary vertex

#neural network model parameters - GNN
attention_heads = 2 #number of attention heads in GAT layers
nodemlp_sizes = [20, 40] #number of nodes in NodeMLP hidden layers
gat_sizes = [128, 256] #layer sizes in GAT hidden layers
edgemlp_sizes = [256, 128, 64, 32] #excluding output features #layer sizes in EdgeMLP hidden layers
reweight = True #toggle whether positive labels in loss are reweighted to make positives and negatives equally important
#ADD REWEIGHTING FACTOR FOR BINARY AND MULTI-CLASS
use_lr_scheduler = False #toggle whether to use learning rate schedule during training

#truth definitions - PROCESSING, PLOTTING
vertex_threshold = 0 #maximum distance for non HF tracks to be marked as part of the same vertex ("other" category)
incl_btoc = True #toggle whether to combine tracks from b hadrons and all c hadrons in B->C SV's or separate them based on their direct HF ancestors

#plotting options - PLOTTING
jet_pt_bound = [25,200] #boundary jet pT for plots
jet_eta_bound = [-2,2] #boundary jet eta for plots