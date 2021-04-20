#!/usr/bin/env python

#GNN for secondary vertex reconstructions

from GNN_model import *

import matplotlib as mpl
mpl.use('Agg')

import dgl
import torch as th
import torch.nn as nn
import os,sys,math,glob,time
import numpy as np
import ROOT
from ROOT import gROOT, TFile, TH1D, TLorentzVector, TCanvas
import matplotlib.pyplot as plt

#th.set_printoptions(edgeitems=10000)
#np.set_printoptions(threshold=sys.maxsize)

#############################################SCRIPT PARAMS#################################################

#input data parameters
infile_path = "/global/homes/j/jmw464/ATLAS/Vertex-GNN/data/"
infile_name = "Btag_07_19_cut"

#training parameters
nepochs = 30
learning_rate = 0.001
batch_size = 10000 #number of jets in a single training batch

#model parameters
attention_heads = 2 #number of attention heads in GAT layer -> these are averaged over
gnn_hidfeats = 256 #number of hidden features in GAT layer output
mlp_hidfeats = 512 #number of hidden features in MLP layers (actual number is twice this since two node feature sets are concatenated)

#script options
reweight = True #reweight positive labels in loss to make positives and negatives equally important
use_normed = True #use normed version of graphs (if available)
load_checkpoint = False
checkpoint_path = "/global/homes/j/jmw464/ATLAS/Vertex-GNN/output/model.pt"

###########################################################################################################

#open relevant files
if use_normed:
    ext = ".normed"
else:
    ext = ""
paramfile_name = infile_path+infile_name+"_params"
train_infile_name = infile_path+infile_name+"_train"+ext+".bin"
val_infile_name = infile_path+infile_name+"_val"+ext+".bin"
test_infile_name = infile_path+infile_name+"_test"+ext+".bin"


#evaluate tp, tn, fp, fn for GNN results
def evaluate_results(true, pred):

    tp = np.sum((true == 1) & (pred == 1))
    tn = np.sum((true == 0) & (pred == 0))
    fp = np.sum((true == 0) & (pred == 1))
    fn = np.sum((true == 1) & (pred == 0))
    
    return tp, tn, fp, fn


def main(argv):
    gROOT.SetBatch(True)
    
    print("Importing input data.")

    start_time = time.time()
    nnfeatures = (dgl.load_graphs(train_infile_name, [0]))[0][0].ndata['features'].size()[1]
    
    #read in values from parameter file
    truth_frac = 0
    if os.path.isfile(paramfile_name):
        paramfile = open(paramfile_name, "r")
        test_len = int(paramfile.readline())
        val_len = int(paramfile.readline())
        train_len = int(paramfile.readline())
        truth_frac = float(paramfile.readline())
    else:
        print("ERROR: Specified parameter file not found")
        return 1

    p_time = time.time()-start_time
    print("Finished importing input data. Time elapsed: {}s.\n".format(p_time))

    #reweight positive labels automatically if desired
    if reweight:
        pos_weight = th.tensor([(1-truth_frac)/truth_frac])
        print("Setting positive weight to {}".format(pos_weight))
    else:
        pos_weight = th.tensor([1])

    #calculate number of testing, training and validation batches
    test_batches = int(math.ceil(test_len/batch_size))
    val_batches = int(math.ceil(val_len/batch_size))
    train_batches = int(math.ceil(train_len/batch_size))

    device = th.device('cuda' if th.cuda.is_available() else 'cpu') #automatically run on GPU if available
    
    model = EdgePredModel(nnfeatures, gnn_hidfeats, mlp_hidfeats, attention_heads).double().to(device)
    opt = th.optim.Adam(model.parameters(), lr=learning_rate)
    loss = nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction='sum').to(device)
    sig = nn.Sigmoid() #since loss already has sigmoid function built in, we need to pass model outputs through a separate sigmoid function for evaluation

    train_loss_array = np.zeros(nepochs)
    val_loss_array = np.zeros(nepochs)

    #print model parameters
    print("Model built. Parameters:")
    for name, param in model.named_parameters():
        print(name, param.size(), param.requires_grad)
    print("")

    #load existing checkpoint
    if load_checkpoint and os.path.exists(checkpoint_path):
        checkpoint = th.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        opt.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']+1
        print("Loading previous model. Starting from epoch {}.".format(start_epoch))
    else:
        start_epoch = 1

    #main training loop
    t_time = time.time()-start_time
    print("Beginning training. Running on {}. Time elapsed: {}s.\n".format(device, t_time))
    for epoch in range(start_epoch,nepochs+1):
        print("Epoch: {}".format(epoch))
        
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
            
            #process batch
            batch = batch.to(device) #transfer batch to relevant device
            pred = model(batch, batch.ndata['features'])
            pred_lt = loss(pred, batch.edata['labels'])
            opt.zero_grad()
            pred_lt.backward()
            opt.step()

            #evaluate loss
            batch_labels = batch.edata['labels'].size()[0]
            total_labels += batch_labels
            print("Training loss: {}".format(pred_lt.item()/batch_labels))
            train_loss_array[epoch-1] += pred_lt.item()

        #normalize loss
        train_loss_array[epoch-1] = train_loss_array[epoch-1]/total_labels

        #save checkpoint
        th.save({'epoch': epoch, 'model_state_dict': model.state_dict(), 'optimizer_state_dict': opt.state_dict()}, checkpoint_path)

        #validation
        total_labels = 0
        tp = tn = fp = fn = 0
        model.eval()
        for ibatch in range(val_batches):

            #load batch from file
            istart = ibatch*batch_size
            if ibatch == (val_batches-1) and val_len%batch_size != 0:
                iend = istart + (val_len%batch_size)
            else:
                iend = (ibatch+1)*batch_size
            val_batch = dgl.batch(dgl.load_graphs(val_infile_name, list(range(istart, iend)))[0])

            #process batch
            val_batch = val_batch.to(device)
            pred = model(val_batch, val_batch.ndata['features'])
            pred_lv = loss(pred, val_batch.edata['labels'])

            #evaluate results
            pred = sig(pred.float()).cpu().detach().numpy().flatten().round().astype(int)
            true = val_batch.edata['labels'].cpu().numpy().flatten().astype(int)
            tpi, tni, fpi, fni = evaluate_results(true, pred)
            tp += tpi
            tn += tni
            fp += fpi
            fn += fni

            #evaluate loss
            batch_labels = val_batch.edata['labels'].size()[0]
            total_labels += batch_labels
            print("Validation loss: {}".format(pred_lv.item()/batch_labels))
            val_loss_array[epoch-1] += pred_lv.item()

        #normalize loss
        val_loss_array[epoch-1] = val_loss_array[epoch-1]/total_labels

        #print validation results
        e_time = time.time()-start_time
        print('Validation results: {} TP, {} FP, {} TN, {} FN. Time elapsed: {}s.\n'.format(tp, fp, tn, fn, e_time))

    print("Training finished. Evaluating model.\n")

    #testing
    tp = tn = fp = fn = 0
    model.eval()
    for ibatch in range(test_batches):
        
        #load batch from file
        istart = ibatch*batch_size
        if ibatch == (test_batches-1) and test_len%batch_size != 0:
            iend = istart + (test_len%batch_size)
        else:
            iend = (ibatch+1)*batch_size
        test_batch = dgl.batch(dgl.load_graphs(test_infile_name, list(range(istart, iend)))[0])

        #process batch
        test_batch = test_batch.to(device)
        node_features = test_batch.ndata['features']
        edge_labels = test_batch.edata['labels']
        
        #evaluate results
        pred = sig(model(test_batch, test_batch.ndata['features']).float()).cpu().detach().numpy().flatten().round().astype(int)
        true = test_batch.edata['labels'].cpu().numpy().flatten().astype(int)
        tpi, tni, fpi, fni = evaluate_results(true, pred)
        tp += tpi
        tn += tni
        fp += fpi
        fn += fni

    #print test results
    print("Test results: {} TP, {} FP, {} TN, {} FN.".format(tp, fp, tn, fn))
    print('Accuracy: {:.4f}'.format((tp+tn)/(tp+tn+fp+fn)))
    print('Fake Rate (1-Precision): {:.4f}'.format(1.-tp/(tp+fp)))
    print('Efficiency (TPR): {:.4f}'.format(tp/(tp+fn)))
    print('True Negative Rate: {:.4f}'.format(tn/(tn+fp)))
    print('F1 Score {:.4f}'.format(2*tp/(2*tp+fp+fn)))

    #plot loss
    plt.ioff()
    plt.plot(range(nepochs), train_loss_array, label="Training")
    plt.plot(range(nepochs), val_loss_array, label="Validation")
    plt.legend()
    plt.xlabel("Epoch")
    plt.savefig("lossplot.png")


if __name__ == '__main__':
    main(sys.argv)
