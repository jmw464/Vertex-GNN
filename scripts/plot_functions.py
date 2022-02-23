#!/usr/bin/env python

######################################### plot_functions.py #########################################
# PURPOSE: contains helper functions related to plotting
# EDIT TO: /
# ---------------------------------------------Summary-----------------------------------------------
# This script contains a collection of functions used to make plots throughout the plotting scripts.
#####################################################################################################

import os,sys,math,ROOT,glob
import numpy as np
import argparse
import matplotlib.pyplot as plt
from ROOT import TFile, TH1D, TH1I, gROOT, TCanvas, gStyle, gPad, TLegend, TGaxis, THStack, TMultiGraph, TLine


colorlist = [432,600,604,401,419]


#plot list of histograms with specified labels - scaling draws second axis shifted by -scaling[0] and scaled by 1/scaling[1]
def plot_hist(histlist, labellist, norm, log, overflow, filename, options, scaling=[]):
    canv = TCanvas("c1", "c1", 800, 600)
    canv.SetTopMargin(10)
    canv.SetRightMargin(100)
    canv.SetLeftMargin(0.15)
    canv.SetBottomMargin(0.15)
    gStyle.SetOptStat(0)
    if log: gPad.SetLogy()
    if len(scaling) > 1 and scaling[1] == 0: scaling = []

    histstack = THStack("stack",histlist[0].GetTitle())
    nostack = 'NOSTACK' in options or 'nostack' in options

    legend = TLegend(0.75-0.15*(math.ceil(len(histlist)/2)-1),0.72,0.9,0.88,'','NDC')
    legend.SetNColumns(math.ceil(len(histlist)/2))

    for i in range(len(histlist)):
        entries = histlist[i].GetEntries()
        mean = histlist[i].GetMean()
        nbins = histlist[i].GetNbinsX()

        histlist[i].SetLineColorAlpha(colorlist[i],.65)
        histlist[i].SetLineWidth(3)
        if not nostack: histlist[i].SetFillColorAlpha(colorlist[i],.65)

        if labellist: legend.AddEntry(histlist[i], "#splitline{"+labellist[i]+"}{#splitline{%d entries}{mean=%.2f}}"%(entries, mean), "F")

        if overflow:
            histlist[i].SetBinContent(nbins, histlist[i].GetBinContent(nbins) + histlist[i].GetBinContent(nbins+1)) #overflow
            histlist[i].SetBinContent(1, histlist[i].GetBinContent(0) + histlist[i].GetBinContent(1)) #underflow
        if entries and norm: histlist[i].Scale(1./entries)

        histstack.Add(histlist[i])

    #get maximum value of y-axis
    if nostack:
        maximum_val = histstack.GetMaximum('nostack')
    else:
        maximum_val = histstack.GetMaximum()

    #increase maximum value of y-axis to make room for legend
    if not log: maximum_val = maximum_val*1.2
    elif not norm: maximum_val = maximum_val**1.2
    else: maximum_val = maximum_val*10

    histstack.SetMaximum(maximum_val)
    
    histstack.Draw(options)
    histstack.GetXaxis().SetTitle(histlist[0].GetXaxis().GetTitle())
    histstack.GetYaxis().SetTitle(histlist[0].GetYaxis().GetTitle())

    xlimit = [histlist[0].GetBinLowEdge(1), histlist[0].GetBinLowEdge(histlist[0].GetNbinsX())+histlist[0].GetBinWidth(histlist[0].GetNbinsX())]
    if len(scaling) != 0:
        top_axis = TGaxis(xlimit[0],maximum_val*0.9999,xlimit[1],maximum_val*0.9999,(xlimit[0]-scaling[0])/scaling[1],(xlimit[1]-scaling[0])/scaling[1],510,"-")
        top_axis.SetTitle("normalized scale")
        top_axis.SetTickSize(0)
        top_axis.Draw("SAME")

    legend.SetTextSize(0.02)
    legend.SetFillStyle(0)
    legend.SetBorderSize(0)
    legend.Draw("SAME")

    canv.SaveAs(filename)
    if log: gPad.Clear()
    canv.Clear()
    del canv


#plot difference between two lists of normalized histograms
def plot_hist_diff(histlist1, histlist2, labellist, overflow, filename, options):
    canv = TCanvas("c1", "c1", 800, 600)
    canv.SetTopMargin(10)
    canv.SetRightMargin(100)
    canv.SetLeftMargin(0.15)
    canv.SetBottomMargin(0.15)
    gStyle.SetOptStat(0)

    histstack = THStack("stack",histlist1[0].GetTitle())

    legend = TLegend(0.75-0.15*(math.ceil(len(histlist1)/2)-1),0.72,0.9,0.88,'','NDC')
    legend.SetNColumns(math.ceil(len(histlist1)/2))

    if options: options += " P9 NOSTACK"
    else: options = "P9 NOSTACK"

    for i in range(len(histlist1)):
        entries = histlist1[i].GetEntries()
        bad_entries = histlist2[i].GetEntries()
        nbins = histlist1[i].GetNbinsX()

        histlist1[i].SetLineColorAlpha(colorlist[i],0.65)
        histlist1[i].SetLineWidth(3)
        histlist1[i].SetMarkerStyle(3)
        histlist1[i].SetMarkerColorAlpha(colorlist[i],0.65)
        
        if entries: histlist1[i].Scale(1./entries)
        if bad_entries: histlist2[i].Scale(1./bad_entries)
        histlist1[i].Add(histlist2[i],-1.)

        legend.AddEntry(histlist1[i], "#splitline{"+labellist[i]+"}{#splitline{%d total jets}{%d bad jets}}"%(entries, bad_entries), "p")

        if overflow:
            histlist1[i].SetBinContent(nbins, histlist1[i].GetBinContent(nbins) + histlist1[i].GetBinContent(nbins+1)) #overflow
            histlist1[i].SetBinContent(1, histlist1[i].GetBinContent(0) + histlist1[i].GetBinContent(1)) #underflow

        histstack.Add(histlist1[i])

    #get minimum/maximum value of y-axis
    maximum_val = histstack.GetMaximum('nostack')
    minimum_val = histstack.GetMinimum('nostack')

    #increase minimum/maximum value of y-axis to make sure whole graph is visible
    maximum_val = maximum_val*1.2
    minimum_val = minimum_val*1.2
    if maximum_val > 0: histstack.SetMaximum(maximum_val)
    if minimum_val < 0: histstack.SetMinimum(minimum_val)

    histstack.Draw(options)
    histstack.GetXaxis().SetTitle(histlist1[0].GetXaxis().GetTitle())
    histstack.GetYaxis().SetTitle(histlist1[0].GetYaxis().GetTitle())

    legend.SetTextSize(0.02)
    legend.SetFillStyle(0)
    legend.SetBorderSize(0)
    legend.Draw("SAME")

    line = TLine(canv.GetUxmin(),0,canv.GetUxmax(),0)
    line.SetLineColorAlpha(2,0.65)
    line.Draw("SAME")

    canv.SaveAs(filename)
    canv.Clear()
    del canv


def plot_profile(profile_list, labellist, filename):
    canv = TCanvas("c1", "c1", 800, 600)
    canv.SetTopMargin(10)
    canv.SetRightMargin(100)
    canv.SetLeftMargin(0.15)
    canv.SetBottomMargin(0.15)
    canv.SetGrid()
    gStyle.SetOptStat(0)

    legend = TLegend(0.75-0.15*(math.ceil(len(profile_list)/2)-1),0.72,0.9,0.88,'','NDC')
    legend.SetNColumns(math.ceil(len(profile_list)/2))

    maximum = max([profile.GetMaximum() for profile in profile_list])
    minimum = min([profile.GetMinimum() for profile in profile_list])

    for i in range(len(profile_list)):
        entries = profile_list[i].GetEntries()
        mean = profile_list[i].GetMean(2)
        nbins = profile_list[i].GetNbinsX()

        profile_list[i].SetLineColorAlpha(colorlist[i],.65)
        profile_list[i].SetLineWidth(3)

        legend.AddEntry(profile_list[i], "#splitline{"+labellist[i]+"}{#splitline{%d entries}{y-mean=%.2f}}"%(entries, mean), "F")

        if maximum > 0: profile_list[i].SetMaximum(maximum*1.2)
        profile_list[i].SetMinimum(minimum)
        
        if i == 0: profile_list[i].Draw()
        else: profile_list[i].Draw("SAMES")

    legend.SetTextSize(0.02)
    legend.SetFillStyle(0)
    legend.SetBorderSize(0)
    legend.Draw("SAME")

    canv.SaveAs(filename)
    canv.Clear()
    del canv


def plot_bar(histlist, axislabels, labels, norm, log, filename, options):
    canv = TCanvas("c1", "c1", 800, 600)
    canv.SetTopMargin(10)
    canv.SetRightMargin(100)
    canv.SetLeftMargin(0.15)
    canv.SetBottomMargin(0.15)
    gStyle.SetOptStat(0)
    if log: gPad.SetLogy()

    histstack = THStack("stack",histlist[0].GetTitle())
    nostack = 'NOSTACK' in options or 'nostack' in options

    legend = TLegend(0.75-0.15*(math.ceil(len(histlist)/2)-1),0.72,0.9,0.88,'','NDC')
    legend.SetNColumns(math.ceil(len(histlist)/2))

    for i in range(len(histlist)):
        entries = histlist[i].GetEntries()
        mean = histlist[i].GetMean()
        nbins = histlist[i].GetNbinsX()

        histlist[i].SetLineColorAlpha(colorlist[i],.65)
        histlist[i].SetLineWidth(3)
        if not nostack: histlist[i].SetFillColorAlpha(colorlist[i],.65)

        if labels: legend.AddEntry(histlist[i], "#splitline{"+labels[i]+"}{#splitline{%d entries}{mean=%.2f}}"%(entries, mean), "F")

        if entries and norm: histlist[i].Scale(1./entries)

        histstack.Add(histlist[i])

    #get maximum value of y-axis
    if nostack:
        maximum_val = histstack.GetMaximum('nostack')
    else:
        maximum_val = histstack.GetMaximum()

    #increase maximum value of y-axis to make room for legend
    if not log: maximum_val = maximum_val*1.2
    elif not norm: maximum_val = maximum_val**1.2
    else: maximum_val = maximum_val*10

    histstack.Draw(options)
    histstack.GetXaxis().SetTitle(histlist[0].GetXaxis().GetTitle())
    histstack.GetYaxis().SetTitle(histlist[0].GetYaxis().GetTitle())

    histstack.GetXaxis().SetNdivisions(len(axislabels))
    histstack.GetXaxis().CenterLabels(True)
    for i, label in enumerate(axislabels):
        histstack.GetXaxis().ChangeLabel(i+1,-1,-1,-1,-1,-1,str(label))

    if labels:
        legend.SetTextSize(0.02)
        legend.SetFillStyle(0)
        legend.SetBorderSize(0)
        legend.Draw("SAME")

    canv.SaveAs(filename)
    if log: gPad.Clear()
    canv.Clear()
    del canv


def plot_confusion_matrix(conf_matrix, xlabels, ylabels, title, axis_labels, filename):
    fig, ax = plt.subplots(figsize=(7.5, 7.5))
    xlab = [""]
    xlab.extend(xlabels)
    ylab = [""]
    ylab.extend(ylabels)
    ax.matshow(conf_matrix, cmap=plt.cm.Blues, alpha=0.3)
    ax.set_xticklabels(xlab)
    ax.set_yticklabels(ylab)

    for i in range(conf_matrix.shape[0]):
        for j in range(conf_matrix.shape[1]):
            ax.text(x=j, y=i,s=conf_matrix[i, j], va='center', ha='center', size='xx-large')
 
    plt.xlabel(axis_labels[0], fontsize=18)
    plt.ylabel(axis_labels[1], fontsize=18)
    plt.title(title, fontsize=18)
    plt.savefig(filename)


def plot_roc_curve(roc_b, roc_c, comp_roc_b, comp_roc_c, xlimits, ylimits, ctype, filename):
    canv = TCanvas("c1", "c1", 800, 600)

    roc_b.SetLineColor(1)
    roc_c.SetLineColor(4)
    comp_roc_b.SetLineColor(1)
    comp_roc_c.SetLineColor(4)
    roc_b.SetMarkerColor(1)
    roc_c.SetMarkerColor(4)
    comp_roc_b.SetMarkerColor(1)
    comp_roc_c.SetMarkerColor(4)
    roc_b.SetMarkerStyle(20)
    roc_c.SetMarkerStyle(20)
    comp_roc_b.SetMarkerStyle(22)
    comp_roc_c.SetMarkerStyle(22)

    mg = TMultiGraph()
    mg.Add(roc_b)
    mg.Add(roc_c)
    mg.Add(comp_roc_b)
    mg.Add(comp_roc_c)
    mg.SetTitle("; "+ ctype +" fake rate; " + ctype + " efficiency")
    mg.Draw("ALP")
    mg.GetXaxis().SetLimits(xlimits[0],xlimits[1])
    mg.SetMinimum(ylimits[0])
    mg.SetMaximum(ylimits[1])

    legend = TLegend(0.75,0.88-0.08*4,0.9,0.88,'','NDC')
    legend.AddEntry(roc_b, "b-jets (GNN)","lp")
    legend.AddEntry(roc_c, "c-jets (GNN)","lp")
    legend.AddEntry(comp_roc_b, "b-jets (SV1)","p")
    legend.AddEntry(comp_roc_c, "c-jets (SV1)","p")
    legend.SetTextSize(0.025)
    legend.SetFillStyle(0)
    legend.SetBorderSize(0)
    legend.Draw("SAME")

    canv.SaveAs(filename)
    canv.Clear()
    del canv