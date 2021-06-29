#!/usr/bin/env python

#GNN for secondary vertex reconstructions

from GNN_model import *
from GNN_eval import *

import matplotlib as mpl
mpl.use('Agg')

import dgl
import torch as th
import torch.nn as nn
import os,sys,math,glob,time
import numpy as np
import argparse
import ROOT
from ROOT import gROOT, gStyle, TFile, TH1D, TLegend, TCanvas
import matplotlib.pyplot as plt

#th.set_printoptions(edgeitems=10000)
np.set_printoptions(threshold=sys.maxsize)


#############################################SCRIPT PARAMS#################################################

#training parameters
batch_size = 10000 #number of jets in a single training batch

#model parameters
attention_heads = 1 #number of attention heads in GAT layer -> these are averaged over
in_features = 13 #number of unique features per node
nodemlp_sizes = [in_features, 40]
gat_sizes = [in_features, 128, 256]
edgemlp_sizes = [2*(gat_sizes[-1]+nodemlp_sizes[-1]), 256, 128, 64] #excluding output features

#script options
reweight = True #reweight positive labels in loss to make positives and negatives equally important
load_checkpoint = True
use_lr_scheduler = True
multi_class = False

#bad jet criteria for each category
mult_threshold = [0.5, 0.1, 0.1, 0.7, 0.1] #0, 1, 2, 3, 4 (none, b, c, b->c, other)
bin_threshold = [0.5, 0.7] #False, True
score_threshold = 0.6 #threshold for two edges to be able to be associated

###########################################################################################################


def main(argv):
    gROOT.SetBatch(True)

    #parse command line arguments
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("-r", "--runnumber", type=str, default=0, dest="runnumber", help="unique identifier for current run")
    parser.add_argument("-l", "--lr", type=float, default=0.001, dest="learning_rate", help="learning rate for training")
    parser.add_argument("-e", "--epochs", type=int, default=20, dest="nepochs", help="number of epochs for training")
    parser.add_argument("-d", "--data_dir", type=str, required=True, dest="data_dir", help="name of directory where data is stored")
    parser.add_argument("-o", "--output_dir", type=str, required=True, dest="output_dir", help="name of directory where GNN output is stored")
    parser.add_argument("-s", "--dataset", type=str, required=True, dest="infile_name", help="name of dataset to train on (without hdf5 extension)")
    parser.add_argument("-n", "--normed", type=int, default=1, dest="use_normed", help="choose whether to use normalized features or not")
    args = parser.parse_args()

    runnumber = args.runnumber
    learning_rate = args.learning_rate
    nepochs = args.nepochs
    infile_name = args.infile_name
    infile_path = args.data_dir
    outfile_path = args.output_dir
    use_normed = args.use_normed

    print("Importing input data.")
    start_time = time.time()
    
    #set relevant filenames
    if use_normed:
        ext = ".normed"
    else:
        ext = ""
    paramfile_name = infile_path+infile_name+"_params"
    train_infile_name = infile_path+infile_name+"_train"+ext+".bin"
    val_infile_name = infile_path+infile_name+"_val"+ext+".bin"
    test_infile_name = infile_path+infile_name+"_test"+ext+".bin"
    registerfile_name = outfile_path+runnumber+"/"+infile_name+"_"+runnumber+"_register"
    checkpointfile_name = outfile_path+runnumber+"/"+infile_name+"_"+runnumber+"_model.pt"

    nnfeatures = (dgl.load_graphs(train_infile_name, [0]))[0][0].ndata['features'].size()[1]
    
    #read in values from parameter file
    truth_frac = 0
    if os.path.isfile(paramfile_name):
        paramfile = open(paramfile_name, "r")
        test_len = int(paramfile.readline())
        val_len = int(paramfile.readline())
        train_len = int(paramfile.readline())
        truth_frac = float(paramfile.readline())
        b_frac = float(paramfile.readline())
        c_frac = float(paramfile.readline())
        btoc_frac = float(paramfile.readline())
        o_frac = float(paramfile.readline())
    else:
        print("ERROR: Specified parameter file not found")
        return 1

    p_time = time.time()-start_time
    print("Finished importing input data. Time elapsed: {}s.\n".format(p_time))

    #reweight positive labels automatically if desired
    if reweight:
        pos_weight = th.tensor([0.5*(1-truth_frac)/truth_frac])
        mult_weights = th.tensor([1./(1-b_frac-c_frac-btoc_frac-o_frac), 1./b_frac, 1./c_frac, 1./btoc_frac, 1./o_frac])
        print("Setting positive weight to {}".format(pos_weight))
    else:
        pos_weight = th.tensor([1])
        mult_weights = th.tensor([1., 1., 1., 1., 1.])

    #calculate number of testing, training and validation batches
    test_batches = int(math.ceil(test_len/batch_size))
    val_batches = int(math.ceil(val_len/batch_size))
    train_batches = int(math.ceil(train_len/batch_size))

    device = th.device('cuda' if th.cuda.is_available() else 'cpu') #automatically run on GPU if available

    if not multi_class:
        loss = nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction='sum').to(device)
        outfeats = 1
        cm = np.zeros((2,2),dtype=int)
        activation = nn.Sigmoid()
        labeltype = 'bin_labels'
    else:
        loss = nn.CrossEntropyLoss(weight=mult_weights).double().to(device)
        outfeats = 5
        cm = np.zeros((5,5),dtype=int)
        activation = nn.Softmax(dim=1)
        labeltype = 'mult_labels'

    model = EdgePredModel(nodemlp_sizes, gat_sizes, edgemlp_sizes, outfeats, attention_heads).double().to(device)
    opt = th.optim.Adam(model.parameters(), lr=learning_rate)
    if use_lr_scheduler: scheduler = th.optim.lr_scheduler.OneCycleLR(opt,0.1, epochs=nepochs, steps_per_epoch=train_batches) #th.optim.lr_scheduler.ReduceLROnPlateau(opt,patience=5)
        
    train_loss_array = np.zeros(nepochs)
    val_loss_array = np.zeros(nepochs)

    #print model parameters
    print("Model built. Parameters:")
    for name, param in model.named_parameters():
        print(name, param.size(), param.requires_grad)
    print("")

    #load existing checkpoint
    if load_checkpoint and os.path.exists(checkpointfile_name):
        checkpoint = th.load(checkpointfile_name)
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
            target = batch.edata[labeltype]
            if multi_class: target = target[:,0].long()
            pred_lt = loss(pred, target)

            opt.zero_grad()
            pred_lt.backward()
            opt.step()

            #evaluate loss
            batch_labels = batch.edata['bin_labels'].size()[0]
            total_labels += batch_labels
            print("Training loss: {}".format(pred_lt.item()/batch_labels))
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

            #process batch
            val_batch = val_batch.to(device)
            pred = model(val_batch, val_batch.ndata['features'])
            target = val_batch.edata[labeltype]
            if multi_class: target = target[:,0].long()
            pred_lv = loss(pred, target)

            #evaluate loss
            batch_labels = val_batch.edata['bin_labels'].size()[0]
            total_labels += batch_labels
            print("Validation loss: {}".format(pred_lv.item()/batch_labels))
            val_loss_array[epoch-1] += pred_lv.item()

        #normalize loss
        val_loss_array[epoch-1] = val_loss_array[epoch-1]/total_labels

        #print validation results
        e_time = time.time()-start_time
        print('Time elapsed: {}s.\n'.format(e_time))

    print("Training finished. Evaluating model.\n")
    
    registerfile = open(registerfile_name, "w")
    if not multi_class:
        pos_r_hist = TH1D("TPR", "Results for each jet;Rate;Entries",10,0,1.001) #1.001 is the upper bound so this is inclusive of 1
        neg_r_hist = TH1D("TNR", "Results for each jet;Rate;Entries",10,0,1.001)
        edge_score_hist = TH1D("", "Edges scores;Score;Entries",10,0,1.001)
        sv1_corrp_hist = TH1D("SV1", "Percentage of tracks from true SV in predicted SV;Percentage;Entries",10,-0.001,1.001)
        sv1_fakep_hist = TH1D("SV1 ", "Percentage of tracks in predicted SV not found in true SV;Percentage;Entries",10,-0.001,1.001)
        gnn_corrp_hist = TH1D("GNN", "Percentage of tracks from true SV in predicted SV;Percentage;Entries",10,-0.001,1.001)
        gnn_fakep_hist = TH1D("GNN ", "Percentage of tracks in predicted SV not found in true SV;Percentage;Entries",10,-0.001,1.001)

        hist_r_list = [neg_r_hist, pos_r_hist]
        hist_s_list = [edge_score_hist]
        hist_pc_list = [gnn_corrp_hist, sv1_corrp_hist]
        hist_pf_list = [gnn_fakep_hist, sv1_fakep_hist]
    else:
        neg_r_hist = TH1D("Class 0 recall", "Results for each jet;Rate;Entries",10,0,1.001)
        b_r_hist = TH1D("Class 1 recall", "Results for each jet;Rate;Entries",10,0,1.001)
        c_r_hist = TH1D("Class 2 recall", "Results for each jet;Rate;Entries",10,0,1.001)
        btoc_r_hist = TH1D("Class 3 recall", "Results for each jet;Rate;Entries",10,0,1.001)
        o_r_hist = TH1D("Class 4 recall", "Results for each jet;Rate;Entries",10,0,1.001)
        neg_score_hist = TH1D("Class 0 scores", "Class scores;Score;Entries",10,0,1.001)
        b_score_hist = TH1D("Class 1 scores", "Class scores;Score;Entries",10,0,1.001)
        c_score_hist = TH1D("Class 2 scores", "Class scores;Score;Entries",10,0,1.001)
        btoc_score_hist = TH1D("Class 3 scores", "Class scores;Score;Entries",10,0,1.001)
        o_score_hist = TH1D("Class 4 scores", "Class scores;Score;Entries",10,0,1.001)

        hist_r_list = [neg_r_hist, b_r_hist, c_r_hist, btoc_r_hist, o_r_hist]
        hist_s_list = [neg_score_hist, b_score_hist, c_score_hist, btoc_score_hist, o_score_hist]

    #initialize matrices for overall SV predictions and plots
    gnn_cm = np.zeros((3,3), dtype=int)
    sv1_cm = np.zeros((3,3), dtype=int)

    #initialize overall bad events matrix
    bad_events = np.empty((0,3), dtype=np.int)
    total_bad = total_jets = 0

    #testing
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
        node_info = test_batch.ndata['info'].cpu().numpy()
        edge_labels = test_batch.edata[labeltype]
        
        #evaluate results
        pred = activation(model(test_batch, test_batch.ndata['features']).float()).cpu().detach().numpy()
        true = test_batch.edata[labeltype].cpu().numpy().astype(int)

        #fill edge score histograms
        if not multi_class:
            for i in range(pred.shape[0]):
                edge_score_hist.Fill(pred[i,0])
            pred = pred.round().astype(int)
        else:
            for i in range(pred.shape[0]):
                neg_score_hist.Fill(pred[i,0])
                b_score_hist.Fill(pred[i,1])
                c_score_hist.Fill(pred[i,2])
                btoc_score_hist.Fill(pred[i,3])
                o_score_hist.Fill(pred[i,4])
            pred = np.argmax(pred, axis=1).astype(int)

        test_batch.edata['pred'] = activation(test_batch.edata['pred'])

        g_test_list = dgl.unbatch(test_batch)

        #evaluate bad events and find secondary vertices
        for g in g_test_list:
            bad_events = evaluate_jet(g, hist_r_list, bad_events, multi_class, bin_threshold, mult_threshold)
            
            if not multi_class:
                gnn_vertices = find_vertices_bin(g, 'gnn', score_threshold)
                true_vertices = find_vertices_bin(g, 'truth', score_threshold)
                sv1_vertex = np.argwhere(g.ndata['reco_labels'].cpu().numpy().astype(int)[:,1]).flatten()
                if sv1_vertex.size > 0: sv1_vertex = [sv1_vertex]

                jet_gnn_cm, gnn_vertex_metrics = compare_vertices(true_vertices, gnn_vertices)
                jet_sv1_cm, sv1_vertex_metrics = compare_vertices(true_vertices, sv1_vertex)
                gnn_cm += jet_gnn_cm
                sv1_cm += jet_sv1_cm

                for vertex_metric in gnn_vertex_metrics:
                    hist_pc_list[0].Fill(vertex_metric[0])
                    hist_pf_list[0].Fill(vertex_metric[1])

                for vertex_metric in sv1_vertex_metrics:
                    hist_pc_list[1].Fill(vertex_metric[0])
                    hist_pf_list[1].Fill(vertex_metric[1])
        
        total_jets += len(g_test_list)
        total_bad += np.shape(bad_events)[0]

        if not multi_class:
            cm += evaluate_confusion_bin(true, pred)
        else:
            cm += evaluate_confusion_mult(true, pred)

    #print test results
    if not multi_class:
        hist_list_list = [hist_r_list, hist_s_list, hist_pc_list, hist_pf_list]
        print('\nTesting results:')
        print('             ||  Pred False  |  Pred True   |')
        print('---------------------------------------------')
        print(f'Actual False || {cm[0,0]:12} | {cm[0,1]:12} |')
        print(f'Actual True  || {cm[1,0]:12} | {cm[1,1]:12} |')
        print('---------------------------------------------')
        print('Accuracy: {:.4f}'.format((cm[1,1]+cm[0,0])/(cm[1,1]+cm[0,0]+cm[0,1]+cm[1,0]))) #(tp+tn)/(tp+tn+fp+fn)
        print('Fake Rate (1-Precision): {:.4f}'.format(1.-cm[1,1]/(cm[1,1]+cm[0,1]))) #1-tp/(tp+fp)
        print('Efficiency (TPR): {:.4f}'.format(cm[1,1]/(cm[1,1]+cm[1,0]))) #tp/(tp+fn)
        print('True Negative Rate: {:.4f}'.format(cm[0,0]/(cm[0,0]+cm[0,1]))) #tn/(tn+fp)
        print('F1 Score {:.4f}\n'.format(2*cm[1,1]/(2*cm[1,1]+cm[0,1]+cm[1,0]))) #2*tp/(2*tp+fp+fn)
        print('Secondary Vertex prediction results per jet (GNN/SV1):')
        print('             ||  Pred no SV   |   Pred 1 SV   |   Pred >1 SV  |')
        print('---------------------------------------------------------------')
        print(f'Actual no SV || {gnn_cm[0,0]:5} /{sv1_cm[0,0]:6} | {gnn_cm[0,1]:5} /{sv1_cm[0,1]:6} | {gnn_cm[0,2]:5} /{sv1_cm[0,2]:6} |')
        print(f'Actual 1 SV  || {gnn_cm[1,0]:5} /{sv1_cm[1,0]:6} | {gnn_cm[1,1]:5} /{sv1_cm[1,1]:6} | {gnn_cm[1,2]:5} /{sv1_cm[1,2]:6} |')
        print(f'Actual >1 SV || {gnn_cm[2,0]:5} /{sv1_cm[2,0]:6} | {gnn_cm[2,1]:5} /{sv1_cm[2,1]:6} | {gnn_cm[2,2]:5} /{sv1_cm[2,2]:6} |')
        print('---------------------------------------------------------------')

    else:
        hist_list_list = [hist_r_list, hist_s_list]
        print('\nTesting results:')
        print('       ||    Pred 0    |    Pred 1    |    Pred 2    |    Pred 3    |    Pred 4    ||  Recall ')
        print('----------------------------------------------------------------------------------------------')
        print(f'True 0 || {cm[0,0]:12d} | {cm[0,1]:12d} | {cm[0,2]:12d} | {cm[0,3]:12d} | {cm[0,4]:12d} || {cm[0,0]/(cm[0,0]+cm[0,1]+cm[0,2]+cm[0,3]+cm[0,4]):.4f}')
        print(f'True 1 || {cm[1,0]:12d} | {cm[1,1]:12d} | {cm[1,2]:12d} | {cm[1,3]:12d} | {cm[1,4]:12d} || {cm[1,1]/(cm[1,0]+cm[1,1]+cm[1,2]+cm[1,3]+cm[1,4]):.4f}')
        print(f'True 2 || {cm[2,0]:12d} | {cm[2,1]:12d} | {cm[2,2]:12d} | {cm[2,3]:12d} | {cm[2,4]:12d} || {cm[2,2]/(cm[2,0]+cm[2,1]+cm[2,2]+cm[2,3]+cm[2,4]):.4f}')
        print(f'True 3 || {cm[3,0]:12d} | {cm[3,1]:12d} | {cm[3,2]:12d} | {cm[3,3]:12d} | {cm[3,4]:12d} || {cm[3,3]/(cm[3,0]+cm[3,1]+cm[3,2]+cm[3,3]+cm[3,4]):.4f}')
        print(f'True 4 || {cm[4,0]:12d} | {cm[4,1]:12d} | {cm[4,2]:12d} | {cm[4,3]:12d} | {cm[4,4]:12d} || {cm[4,4]/(cm[4,0]+cm[4,1]+cm[4,2]+cm[4,3]+cm[4,4]):.4f}')
        print('----------------------------------------------------------------------------------------------')
        print(f'Prec   ||       {cm[0,0]/(cm[0,0]+cm[1,0]+cm[2,0]+cm[3,0]+cm[4,0]):.4f} |       {cm[1,1]/(cm[0,1]+cm[1,1]+cm[2,1]+cm[3,1]+cm[4,1]):.4f} |       {cm[2,2]/(cm[0,2]+cm[1,2]+cm[2,2]+cm[3,2]+cm[4,2]):.4f} |       {cm[3,3]/(cm[0,3]+cm[1,3]+cm[2,3]+cm[3,3]+cm[4,3]):.4f} |       {cm[4,4]/(cm[0,4]+cm[1,4]+cm[2,4]+cm[3,4]+cm[4,4]):.4f} ||\n')

    #sort array with bad events by file, event and jet
    indices = np.lexsort((bad_events[:,2], bad_events[:,1], bad_events[:,0]))
    bad_events = bad_events[indices]
    
    #output bad events to file
    for i in range(total_bad):
        registerfile.write(str(bad_events[i,0])+' '+str(bad_events[i,1])+' '+str(bad_events[i,2])+'\n')
    print("\nMarked {}% of {} jets as bad".format(100*total_bad/total_jets, total_jets))

    #plot loss
    plt.ioff()
    plt.plot(range(nepochs), train_loss_array, label="Training")
    plt.plot(range(nepochs), val_loss_array, label="Validation")
    plt.legend()
    plt.xlabel("Epoch")
    plt.savefig(outfile_path+runnumber+"/"+infile_name+"_"+runnumber+"_lossplot.png")

    ext = ["_recall.png", "_score.png", "_corr_p.png", "_fake_p.png"]
    outfile_name = outfile_path+runnumber+"/"+infile_name+"_"+runnumber
    for i in range(len(hist_list_list)):
        plot_hist(hist_list_list[i], outfile_name, ext[i])


if __name__ == '__main__':
    main(sys.argv)
