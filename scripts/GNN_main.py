#!/usr/bin/env python

######################################### GNN_main.py #########################################
# PURPOSE: trains GNN for secondary vertex reconstructions
# EDIT TO: implement new GNN tools, update graph structure (if modified in create_graphs)
# ------------------------------------------Summary-------------------------------------------
# This script serves as the center of this framework and runs the GNN training. It can be run
# in both binary classification mode (edges are "connected" and "not connected") or multi-label
# classification mode (edges are "connected via b SV", "connected via c SV" and
# "not connected"). It will output a results file containing the same graphs as the testing
# dataset, but with GNN predictions included. Many of the pre-implemented features can be
# activated/deactivated in the options script (i.e. LR schedules, label reweighting, etc.).
# There is no need to modify this script unless the graph structure itself is modified in the
# processing scripts or to implement new GNN tools or modify existing ones (such as the LR
# schedule).
###############################################################################################


import dgl
import torch as th
import torch.nn as nn

import os,sys,math,glob,time
import numpy as np
import argparse
import ROOT
from ROOT import gROOT, TFile, TH1D
import matplotlib as mpl
import matplotlib.pyplot as plt

from GNN_model import *
from GNN_eval import *
from plot_functions import *
import options

#th.set_printoptions(edgeitems=10000)
#np.set_printoptions(threshold=sys.maxsize)
mpl.use('Agg')


def main(argv):
    gROOT.SetBatch(True)

    #parse command line arguments
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("-r", "--runnumber", type=str, default=0, dest="runnumber", help="unique identifier for current run")
    parser.add_argument("-e", "--epochs", type=int, default=20, dest="nepochs", help="number of epochs for training")
    parser.add_argument("-d", "--data_dir", type=str, required=True, dest="data_dir", help="name of directory where data is stored")
    parser.add_argument("-o", "--output_dir", type=str, required=True, dest="output_dir", help="name of directory where GNN output is stored")
    parser.add_argument("-s", "--dataset", type=str, required=True, dest="infile_name", help="name of dataset to train on (without hdf5 extension)")
    parser.add_argument("-n", "--normed", type=int, default=1, dest="use_normed", help="choose whether to use normalized features or not")
    parser.add_argument("-m", "--multiclass", type=int, default=0, dest="multi_class", help="choose whether to perform binary of multi-class classification")
    args = parser.parse_args()

    runnumber = args.runnumber
    nepochs = args.nepochs
    infile_name = args.infile_name
    infile_path = args.data_dir
    outfile_path = args.output_dir
    use_normed = args.use_normed
    multi_class = args.multi_class

    #import options from option file
    use_gpu = options.learning_rate
    learning_rate = options.learning_rate
    batch_size = options.batch_size
    dropout = options.dropout
    attention_heads = options.attention_heads
    nodemlp_sizes = options.nodemlp_sizes
    gat_sizes = options.gat_sizes
    edgemlp_sizes = options.edgemlp_sizes
    reweight = options.reweight #reweight positive labels in loss to make positives and negatives equally important
    load_checkpoint = options.load_checkpoint
    use_lr_scheduler = options.use_lr_scheduler

    #---------------------------------------------------DATA-IMPORT-------------------------------------------------

    start_time = time.time()
    print("Importing input data.", flush=True)
    
    #set relevant filenames
    if use_normed:
        ext = ".normed.pruned"
    else:
        ext = ".pruned"
    paramfile_name = infile_path+infile_name+"_params"
    train_infile_name = infile_path+infile_name+"_train"+ext+".bin"
    val_infile_name = infile_path+infile_name+"_val"+ext+".bin"
    test_infile_name = infile_path+infile_name+"_test"+ext+".bin"
    checkpointfile_name = outfile_path+runnumber+"/"+infile_name+"_"+runnumber+"_model.pt"

    #calculate number of features in graphs
    sample_graph = dgl.load_graphs(train_infile_name, [0])[0][0]
    incl_errors = incl_corr = incl_hits = incl_vweight = False
    nnfeatures_base = sample_graph.ndata['features_base'].size()[1]
    in_features = nnfeatures_base
    if 'features_vweight' in sample_graph.ndata.keys():
        nnfeatures_vweight = sample_graph.ndata['features_vweight'].size()[1]
        incl_vweight = True
        in_features += nnfeatures_vweight
    if 'features_errors' in sample_graph.ndata.keys():
        nnfeatures_errors = sample_graph.ndata['features_errors'].size()[1]
        incl_errors = True
        in_features += nnfeatures_errors
    if 'features_hits' in sample_graph.ndata.keys():
        nnfeatures_hits = sample_graph.ndata['features_hits'].size()[1]
        incl_hits = True
        in_features += nnfeatures_hits
    if 'features_corr' in sample_graph.ndata.keys():
        nnfeatures_corr = sample_graph.ndata['features_corr'].size()[1]
        incl_corr = True
        in_features += nnfeatures_corr
    
    #read in values from parameter file
    if os.path.isfile(paramfile_name):
        paramfile = open(paramfile_name, "r")
        train_len = int(float(paramfile.readline()))
        val_len = int(float(paramfile.readline()))
        test_len = int(float(paramfile.readline()))
        truth_frac = float(paramfile.readline())
        b_frac = float(paramfile.readline())
        c_frac = float(paramfile.readline())
    else:
        print("ERROR: Specified parameter file not found", flush=True)
        return 1

    p_time = time.time()-start_time
    print("Finished importing input data. Time elapsed: {}s.\n".format(p_time), flush=True)

    #reweight positive labels automatically if desired
    if reweight:
        pos_weight = th.tensor([0.5*(1-truth_frac)/truth_frac])
        mult_weights = th.tensor([1./(1-b_frac-c_frac), 1./b_frac, 1./c_frac])
        print("Setting positive weight to {}".format(pos_weight), flush=True)
    else:
        pos_weight = th.tensor([1])
        mult_weights = th.tensor([1., 1., 1.])

    #calculate number of testing, training and validation batches
    test_batches = int(math.ceil(test_len/batch_size))
    val_batches = int(math.ceil(val_len/batch_size))
    train_batches = int(math.ceil(train_len/batch_size))

    #set up loss
    if not multi_class:
        loss = nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction='sum')
        outfeats = 1
        cm = np.zeros((2,2),dtype=int)
        activation = nn.Sigmoid()
        labeltype = 'bin_labels'
    else:
        loss = nn.CrossEntropyLoss(weight=mult_weights).double()
        outfeats = 3
        cm = np.zeros((3,3),dtype=int)
        activation = nn.Softmax(dim=1)
        labeltype = 'mult_labels'

    model = EdgePredModel(nodemlp_sizes, gat_sizes, edgemlp_sizes, in_features, outfeats, attention_heads, dropout).double()
    
    if th.cuda.is_available() and use_gpu:
        device = th.device('cuda')
        print("Found {} GPUs".format(th.cuda.device_count()))
        #if th.cuda.device_count() > 1: model = th.nn.DataParallel(model)
    else:
        device = th.device('cpu')
    model.to(device)
    loss.to(device)
    device = th.device('cuda' if th.cuda.is_available() else 'cpu') #automatically run on GPU if available

    opt = th.optim.Adam(model.parameters(), lr=learning_rate)
    if use_lr_scheduler: scheduler = th.optim.lr_scheduler.OneCycleLR(opt,0.1, epochs=nepochs, steps_per_epoch=train_batches) #th.optim.lr_scheduler.ReduceLROnPlateau(opt,patience=5)
        
    train_loss_array = np.zeros(nepochs)
    val_loss_array = np.zeros(nepochs)

    #print model parameters
    print("Model built. Parameters:", flush=True)
    for name, param in model.named_parameters():
        print(name, param.size(), param.requires_grad, flush=True)
    print("", flush=True)

    #load existing checkpoint
    if load_checkpoint and os.path.exists(checkpointfile_name):
        checkpoint = th.load(checkpointfile_name)
        model.load_state_dict(checkpoint['model_state_dict'])
        opt.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']+1
        print("Loading previous model. Starting from epoch {}.".format(start_epoch), flush=True)
    else:
        start_epoch = 1

    #initialize multiprocessing on GPUs
    #if th.cuda.is_available():
    #    th.distributed.init_process_group(backend='nccl')
    #    model = th.nn.parallel.DistributedDataParallel(model)

    #----------------------------------------------------TRAINING---------------------------------------------------

    #main training loop
    t_time = time.time()-start_time
    print("Beginning training. Running on {}. Time elapsed: {}s.\n".format(device, t_time), flush=True)
    for epoch in range(start_epoch,nepochs+1):
        print("Epoch: {}".format(epoch), flush=True)
        
        #training
        total_labels = 0
        model.train()
        for ibatch in range(train_batches):
            
            #load batch from file
            istart = ibatch*batch_size
            if ibatch == (train_batches-1) and train_len%batch_size != 0:
                iend = istart+(train_len%batch_size)
            else:
                iend = (ibatch+1)*batch_size
            batch = dgl.batch(dgl.load_graphs(train_infile_name, list(range(istart, iend)))[0])

            #construct feature matrix
            features = batch.ndata['features_base']
            if incl_vweight: features = th.cat((features, batch.ndata['features_vweight']),dim=1)
            if incl_errors: features = th.cat((features, batch.ndata['features_errors']),dim=1)
            if incl_hits: features = th.cat((features, batch.ndata['features_hits']),dim=1)
            if incl_corr: features = th.cat((features, batch.ndata['features_corr']),dim=1)

            #process batch
            batch = batch.to(device) #transfer batch to relevant device
            features = features.to(device)
            pred = model(batch, features)
            target = batch.edata[labeltype]
            if multi_class: target = target[:,0].long()
            pred_lt = loss(pred, target)

            opt.zero_grad()
            pred_lt.backward()
            opt.step()

            #evaluate loss
            batch_labels = batch.edata['bin_labels'].size()[0]
            total_labels += batch_labels
            print("Training loss: {}".format(pred_lt.item()/batch_labels), flush=True)
            train_loss_array[epoch-1] += pred_lt.item()

            if use_lr_scheduler: scheduler.step()
        
        #normalize loss
        train_loss_array[epoch-1] = train_loss_array[epoch-1]/total_labels

        #save checkpoint
        th.save({'epoch': epoch, 'model_state_dict': model.state_dict(), 'optimizer_state_dict': opt.state_dict()}, checkpointfile_name)

        #validation
        total_labels = 0
        model.eval()
        for ibatch in range(val_batches):

            #load batch from file
            istart = ibatch*batch_size
            if ibatch == (val_batches-1) and val_len%batch_size != 0:
                iend = istart + (val_len%batch_size)
            else:
                iend = (ibatch+1)*batch_size
            val_batch = dgl.batch(dgl.load_graphs(val_infile_name, list(range(istart, iend)))[0])

            #construct feature matrix
            val_features = val_batch.ndata['features_base']
            if incl_vweight: val_features = th.cat((val_features, val_batch.ndata['features_vweight']),dim=1)
            if incl_errors: val_features = th.cat((val_features, val_batch.ndata['features_errors']),dim=1)
            if incl_hits: val_features = th.cat((val_features, val_batch.ndata['features_hits']),dim=1)
            if incl_corr: val_features = th.cat((val_features, val_batch.ndata['features_corr']),dim=1)

            #process batch
            val_batch = val_batch.to(device)
            val_features = val_features.to(device)
            pred = model(val_batch, val_features)
            target = val_batch.edata[labeltype]
            if multi_class: target = target[:,0].long()
            pred_lv = loss(pred, target)

            #evaluate loss
            batch_labels = val_batch.edata['bin_labels'].size()[0]
            total_labels += batch_labels
            print("Validation loss: {}".format(pred_lv.item()/batch_labels), flush=True)
            val_loss_array[epoch-1] += pred_lv.item()

        #normalize loss
        val_loss_array[epoch-1] = val_loss_array[epoch-1]/total_labels

        #print validation results
        e_time = time.time()-start_time
        print('Time elapsed: {}s.\n'.format(e_time), flush=True)

    print("Training finished. Evaluating model.\n", flush=True)

    #---------------------------------------------------EVALUATION-------------------------------------------------- 

    overall_g_list = []

    #testing
    with th.no_grad():
        model.eval()
        for ibatch in range(test_batches):

            #load batch from file
            istart = ibatch*batch_size
            if ibatch == (test_batches-1) and test_len%batch_size != 0:
                iend = istart + (test_len%batch_size)
            else:
                iend = (ibatch+1)*batch_size
            test_batch = dgl.batch(dgl.load_graphs(test_infile_name, list(range(istart, iend)))[0])

            #construct feature matrix
            test_features = test_batch.ndata['features_base']
            if incl_vweight: test_features = th.cat((test_features, test_batch.ndata['features_vweight']),dim=1)
            if incl_errors: test_features = th.cat((test_features, test_batch.ndata['features_errors']),dim=1)
            if incl_hits: test_features = th.cat((test_features, test_batch.ndata['features_hits']),dim=1)
            if incl_corr: test_features = th.cat((test_features, test_batch.ndata['features_corr']),dim=1)

            #process batch
            test_batch = test_batch.to(device)
            test_features = test_features.to(device)
            edge_labels = test_batch.edata[labeltype]
        
            #evaluate results
            pred = activation(model(test_batch, test_features).float()).cpu().detach().numpy()
            true = test_batch.edata[labeltype].cpu().numpy().astype(int) 
        
            test_batch.edata['pred'] = activation(test_batch.edata['pred'])

            g_test_list = dgl.unbatch(test_batch)
            overall_g_list.extend(g_test_list)

            if not multi_class:
                cm += evaluate_confusion_bin(true, pred.round().astype(int))
            else:
                cm += evaluate_confusion_mult(true, pred.round().astype(int))

    #print test results
    print_output(multi_class, cm)

    #save results to file
    outfile_name = outfile_path+runnumber+"/"+infile_name+"_"+runnumber
    dgl.save_graphs(outfile_name+"_results.bin", overall_g_list)

    #plot loss
    plt.ioff()
    plt.plot(range(nepochs), train_loss_array, label="Training")
    plt.plot(range(nepochs), val_loss_array, label="Validation")
    plt.legend()
    plt.xlabel("Epoch")
    plt.savefig(outfile_name+"_lossplot.png")


if __name__ == '__main__':
    main(sys.argv)
