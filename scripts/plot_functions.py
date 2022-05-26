#!/usr/bin/env python

######################################### plot_functions.py #########################################
# PURPOSE: contains helper functions related to plotting
# EDIT TO: /
# ---------------------------------------------Summary-----------------------------------------------
# This script contains a collection of functions used to make plots throughout the plotting scripts.
#####################################################################################################

import os,sys,math,glob
import ROOT
import numpy as np
import argparse
import matplotlib.pyplot as plt
from ROOT import TFile, TH1D, TH1I, gROOT, TCanvas, gStyle, gPad, TLegend, TGaxis, THStack, TMultiGraph, TLine, TLatex, TPad

#set ATLAS style for plots
gROOT.LoadMacro("/global/homes/j/jmw464/ATLAS/Vertex-GNN/scripts/include/AtlasStyle.C")
gROOT.LoadMacro("/global/homes/j/jmw464/ATLAS/Vertex-GNN/scripts/include/AtlasLabels.C")
from ROOT import SetAtlasStyle


colorlist = [432,600,604,401,419]


def plot_hist(canv, hist_list, labellist, cutstring, norm, log, filename, scaling=[]):
    canv.Clear()
    canv.SetGrid()
    SetAtlasStyle()
    gStyle.SetOptStat(0)
    gStyle.SetErrorX(0.5)
    pad1 = TPad("pad1", "pad1", 0.,0.,1.,1.)
    if log: pad1.SetLogy()
    if len(scaling) > 1 and scaling[1] == 0: scaling = []
    pad1.SetTopMargin(0.1)
    pad1.Draw()
    pad1.cd()

    legend = TLegend(0.75-0.15*(math.ceil(len(hist_list)/3)-1),0.68,0.9,0.86,'','NDC')
    legend.SetNColumns(math.ceil(len(hist_list)/3))
    logo = TLatex(0.2,0.82, "#bf{#it{ATLAS}} #it{Internal}")
    add_text = TLatex(0.2,0.77,cutstring)
    logo.SetNDC(True)
    add_text.SetNDC(True)

    #normalize histograms
    for i in range(len(hist_list)):
        scale = 1.
        entries = hist_list[i].GetEntries()
        if norm and entries: scale = scale/entries
        hist_list[i].Scale(scale)

    maximum_val = max([hist.GetMaximum() for hist in hist_list])
    minimum_val = min([hist.GetMinimum(0.) for hist in hist_list])
    if maximum_val > 0 and not log: maximum_val = maximum_val*1.5
    elif maximum_val > 0 and not norm: maximum_val = maximum_val**1.5
    elif maximum_val > 0: maximum_val = maximum_val*15
    if not log: minimum_val = 0
    else: minimum_val = minimum_val/10

    for i in range(len(hist_list)):
        hist_list[i].SetMarkerColorAlpha(colorlist[i],.75)
        hist_list[i].SetLineColorAlpha(colorlist[i],.65)
        hist_list[i].SetLineWidth(3)
        hist_list[i].SetTitle("")
        hist_list[i].GetXaxis().SetLabelSize(0.05)
        hist_list[i].GetXaxis().SetTitleSize(0.05)
        hist_list[i].GetYaxis().SetLabelSize(0.05)
        hist_list[i].GetYaxis().SetTitleSize(0.05)
        hist_list[i].GetYaxis().SetTitleOffset(1.5)

        entries = hist_list[i].GetEntries()
        legend.AddEntry(hist_list[i], "#splitline{"+labellist[i]+"}{#bf{#scale[0.7]{%d entries}}}"%(entries), "l")

        hist_list[i].SetMaximum(maximum_val)
        hist_list[i].SetMinimum(minimum_val)
        hist_list[i].Draw("SAMES")

    #draw second axis
    xlimit = [hist_list[0].GetBinLowEdge(1), hist_list[0].GetBinLowEdge(hist_list[0].GetNbinsX())+hist_list[0].GetBinWidth(hist_list[0].GetNbinsX())]
    ndivs = hist_list[0].GetNdivisions()
    if len(scaling) != 0:
        gStyle.SetPadTickX(0)
        top_axis = TGaxis(xlimit[0],maximum_val*0.9999,xlimit[1],maximum_val*0.9999,(xlimit[0]-scaling[0])/scaling[1],(xlimit[1]-scaling[0])/scaling[1],ndivs,"-",0)
        top_axis.SetTitle("normalized scale")
        top_axis.SetTitleOffset(1.0)
        top_axis.Draw("SAME")

    legend.SetTextSize(0.03)
    legend.SetFillStyle(0)
    legend.SetBorderSize(0)
    legend.Draw("SAME")
    logo.Draw("SAME")
    add_text.Draw("SAME")
    canv.cd()
    canv.SaveAs(filename)
    if log: gPad.Clear()
    canv.Clear()


def plot_histratio(canv, hist_list, div_list, labellist, cutstring, log, filename):
    canv.Clear()
    gStyle.SetOptStat(0)
    gStyle.SetErrorX(0.5)
    SetAtlasStyle()
    pad1 = TPad("pad1", "pad1", 0.05,0.35,1.0,1.0)
    pad1.SetGrid()
    if log: pad1.SetLogy()
    pad1.Draw()
    pad2 = TPad("pad2", "pad2", 0.05,0.0,1.0,0.35)
    pad2.SetGrid()
    pad2.Draw("SAME")
    pad1.SetBottomMargin(0.05)
    pad2.SetTopMargin(0.05)
    pad2.SetBottomMargin(0.25)

    pad1.cd()
    legend = TLegend(0.75-0.2*(math.ceil(len(hist_list)/3)-1),0.68,0.9,0.92,'','NDC')
    legend.SetNColumns(math.ceil(len(hist_list)/3))
    logo = TLatex(0.2,0.88, "#bf{#it{ATLAS}} #it{Internal}")
    add_text = TLatex(0.2,0.83,cutstring)
    logo.SetNDC(True)
    add_text.SetNDC(True)

    #normalize histograms
    for i in range(len(hist_list)):
        entries = hist_list[i].GetEntries()
        if entries: hist_list[i].Scale(1./entries)
    for i in range(len(div_list)):
        entries = div_list[i].GetEntries()
        if entries: div_list[i].Scale(1./entries)

    #generate ratio histograms
    ratio_list = [None]*len(hist_list)
    for i in range(len(hist_list)):
        ratio_list[i] = hist_list[i].Clone(hist_list[i].GetName()+"_ratio")
        ratio_list[i].Divide(div_list[i])

    maximum = max([hist.GetMaximum() for hist in hist_list])
    minimum = min([hist.GetMinimum(0.) for hist in hist_list])
    if maximum > 0 and not log: maximum = maximum*1.5
    elif maximum > 0: maximum = maximum*15
    if not log: minimum = 0
    else: minimum = minimum/10

    for i in range(len(hist_list)):
        entries = hist_list[i].GetEntries()
        bad_entries = div_list[i].GetEntries() 
        hist_list[i].SetMarkerColorAlpha(colorlist[i],.75)
        hist_list[i].SetLineColorAlpha(colorlist[i],.65)
        hist_list[i].SetLineWidth(3)
        hist_list[i].SetTitle("")
        hist_list[i].GetXaxis().SetLabelSize(0)
        hist_list[i].GetXaxis().SetTitleSize(0)
        hist_list[i].GetYaxis().SetLabelSize(0.05)
        hist_list[i].GetYaxis().SetTitleSize(0.05)
        hist_list[i].GetYaxis().SetTitleOffset(1.0)

        legend.AddEntry(hist_list[i], "#splitline{"+labellist[i]+"}{#bf{#scale[0.7]{#splitline{%d total jets}{%d bad jets}}}}"%(entries, bad_entries), "p")

        hist_list[i].SetMaximum(maximum)
        hist_list[i].SetMinimum(minimum)
        hist_list[i].Draw("SAMES")

    legend.SetTextSize(0.03)
    legend.SetFillStyle(0)
    legend.SetBorderSize(0)
    legend.Draw("SAME")
    logo.Draw("SAME")
    add_text.Draw("SAME")

    pad2.cd()
    
    maximum = max([ratio.GetMaximum() for ratio in ratio_list])*1.1
    minimum = min([ratio.GetMinimum() for ratio in ratio_list])*0.9
    
    for i in range(len(ratio_list)):
        ratio_list[i].SetMarkerColorAlpha(colorlist[i],.75)
        ratio_list[i].SetLineColorAlpha(colorlist[i],.65)
        ratio_list[i].SetLineWidth(3)
        ratio_list[i].SetTitle("")
        ratio_list[i].GetXaxis().SetTitleSize(0.1)
        ratio_list[i].GetXaxis().SetLabelSize(0.1)
        ratio_list[i].GetXaxis().SetTitleOffset(0.9)
        ratio_list[i].GetYaxis().SetLabelSize(0.05)
        ratio_list[i].GetYaxis().SetTitleSize(0.1)
        ratio_list[i].GetYaxis().SetTitle("Bad/overall")
        ratio_list[i].GetYaxis().SetTitleOffset(0.5)

        if maximum > 0: ratio_list[i].SetMaximum(maximum)
        ratio_list[i].SetMinimum(minimum)
        ratio_list[i].Draw("SAMES")

    canv.cd()
    canv.SaveAs(filename)
    if log: gPad.Clear()

    for ratio in ratio_list:
        ratio.SetDirectory(0)
        del ratio
    canv.Clear()


def plot_bar(canv, hist_list, axislabels, labels, cutstring, norm, log, filename):
    canv.Clear()
    canv.SetGrid()
    SetAtlasStyle()
    gStyle.SetOptStat(0)
    gStyle.SetErrorX(0.5)
    pad1 = TPad("pad1", "pad1", 0.,0.,1.,1.)
    if log: pad1.SetLogy()
    pad1.SetTopMargin(0.1)
    pad1.Draw()
    pad1.cd()

    legend = TLegend(0.75-0.15*(math.ceil(len(hist_list)/3)-1),0.68,0.9,0.86,'','NDC')
    legend.SetNColumns(math.ceil(len(hist_list)/3))
    logo = TLatex(0.2,0.82, "#bf{#it{ATLAS}} #it{Internal}")
    add_text = TLatex(0.2,0.77,cutstring)
    logo.SetNDC(True)
    add_text.SetNDC(True)

    #normalize histograms
    for i in range(len(hist_list)):
        scale = 1.
        entries = hist_list[i].GetEntries()
        if norm and entries: scale = scale/entries
        hist_list[i].Scale(scale)

    maximum_val = max([hist.GetMaximum() for hist in hist_list])
    minimum_val = min([hist.GetMinimum(0.) for hist in hist_list])
    if maximum_val > 0 and not log: maximum_val = maximum_val*1.5
    elif maximum_val > 0 and not norm: maximum_val = maximum_val**1.5
    elif maximum_val > 0: maximum_val = maximum_val*15
    if not log: minimum_val = 0
    else: minimum_val = minimum_val/10

    for i in range(len(hist_list)):
        hist_list[i].SetMarkerColorAlpha(colorlist[i],.75)
        hist_list[i].SetLineColorAlpha(colorlist[i],.65)
        hist_list[i].SetLineWidth(3)
        hist_list[i].SetTitle("")
        hist_list[i].GetXaxis().SetLabelSize(0.03)
        hist_list[i].GetXaxis().SetTitleSize(0.05)
        hist_list[i].GetYaxis().SetLabelSize(0.03)
        hist_list[i].GetYaxis().SetTitleSize(0.05)
        hist_list[i].GetYaxis().SetTitleOffset(1.5)

        entries = hist_list[i].GetEntries()
        legend.AddEntry(hist_list[i], "#splitline{"+labels[i]+"}{#bf{#scale[0.7]{%d entries}}}"%(entries), "l")

        hist_list[i].GetXaxis().SetNdivisions(len(axislabels))
        hist_list[i].GetXaxis().CenterLabels(True)
        for j, label in enumerate(axislabels):
            hist_list[i].GetXaxis().ChangeLabel(j+1,-1,-1,-1,-1,-1,str(label))

        hist_list[i].SetMaximum(maximum_val)
        hist_list[i].SetMinimum(minimum_val)
        hist_list[i].Draw("SAMES")

    legend.SetTextSize(0.03)
    legend.SetFillStyle(0)
    legend.SetBorderSize(0)
    legend.Draw("SAME")
    logo.Draw("SAME")
    add_text.Draw("SAME")
    canv.cd()
    canv.SaveAs(filename)
    if log: gPad.Clear()
    canv.Clear()


def plot_profile(canv, profile_list, labellist, cutstring, filename):
    canv.Clear()
    canv.SetGrid()
    SetAtlasStyle()
    gStyle.SetOptStat(0)
    gStyle.SetErrorX(0.5)
    pad1 = TPad("pad1", "pad1", 0.,0.,1.,1.)
    pad1.SetTopMargin(0.1)
    pad1.Draw()
    pad1.cd()

    legend = TLegend(0.75-0.15*(math.ceil(len(profile_list)/3)-1),0.68,0.9,0.86,'','NDC')
    legend.SetNColumns(math.ceil(len(profile_list)/3))
    logo = TLatex(0.2,0.82, "#bf{#it{ATLAS}} #it{Internal}")
    add_text = TLatex(0.2,0.77,cutstring)
    logo.SetNDC(True)
    add_text.SetNDC(True)

    maximum = max(0,max([profile.GetMaximum() for profile in profile_list]))*1.5
    minimum = min(0,min([profile.GetMinimum() for profile in profile_list]))*0.9

    for i in range(len(profile_list)):
        profile_list[i].SetMarkerColorAlpha(colorlist[i],.75)
        profile_list[i].SetLineColorAlpha(colorlist[i],.65)
        profile_list[i].SetLineWidth(3)
        profile_list[i].SetTitle("")
        profile_list[i].GetXaxis().SetLabelSize(0.05)
        profile_list[i].GetXaxis().SetTitleSize(0.05)
        profile_list[i].GetYaxis().SetLabelSize(0.05)
        profile_list[i].GetYaxis().SetTitleSize(0.05)
        profile_list[i].GetYaxis().SetTitleOffset(1.5)

        entries = profile_list[i].GetEntries()
        legend.AddEntry(profile_list[i], "#splitline{"+labellist[i]+"}{#bf{#scale[0.7]{%d entries}}}"%(entries), "l")

        profile_list[i].SetMaximum(maximum)
        profile_list[i].SetMinimum(minimum)
        profile_list[i].Draw("SAMES")
        
    legend.SetTextSize(0.03)
    legend.SetFillStyle(0)
    legend.SetBorderSize(0)
    legend.Draw("SAME")
    logo.Draw("SAME")
    add_text.Draw("SAME")
    canv.cd()
    canv.SaveAs(filename)
    canv.Clear()


def plot_profileratio(canv, profile_list, div_list, profile_labels, div_labels, cut_string, log, plot_div, filename):
    canv.Clear()
    gStyle.SetOptStat(0)
    gStyle.SetErrorX(.5)
    SetAtlasStyle()
    pad1 = TPad("pad1", "pad1", 0.0,0.35,1.0,1.0)
    pad1.SetGrid()
    if log: pad1.SetLogy()
    pad1.Draw()
    pad2 = TPad("pad2", "pad2", 0.0,0.0,1.0,0.35)
    pad2.SetGrid()
    pad2.Draw("SAME")
    pad1.SetBottomMargin(0.05)
    pad2.SetTopMargin(0.05)
    pad2.SetBottomMargin(0.25)

    pad1.cd()
    legend = TLegend(0.75-0.15*(math.ceil(len(profile_list)/3)-1),0.76,0.9,0.92,'','NDC')
    legend.SetNColumns(math.ceil(len(profile_list)/3))
    logo = TLatex(0.2,0.88, "#bf{#it{ATLAS}} #it{Internal}")
    cut_text = TLatex(0.2,0.81,cut_string)
    logo.SetNDC(True)
    cut_text.SetNDC(True)

    #generate ratio histograms
    ratio_list = [None]*len(profile_list)
    for i in range(len(profile_list)):
        ratio_list[i] = profile_list[i].Clone(profile_list[i].GetName()+"_ratio").ProjectionX()
        ratio_list[i].Divide(div_list[i].ProjectionX())

    maximum = max([profile.GetMaximum() for profile in profile_list])
    minimum = min([profile.GetMinimum(0.) for profile in profile_list])
    if maximum > 0 and not log: maximum = maximum*1.5
    elif maximum > 0: maximum = maximum*15
    if not log: minimum = 0
    else: minimum = minimum/10

    for i in range(len(profile_list)):
        profile_list[i].SetMarkerColorAlpha(colorlist[i],.75)
        profile_list[i].SetLineColorAlpha(colorlist[i],.65)
        profile_list[i].SetLineWidth(3)
        profile_list[i].SetTitle("")
        profile_list[i].GetXaxis().SetLabelSize(0)
        profile_list[i].GetXaxis().SetTitleSize(0)
        profile_list[i].GetYaxis().SetLabelSize(0.05)
        profile_list[i].GetYaxis().SetTitleSize(0.05)
        profile_list[i].GetYaxis().SetTitleOffset(1.0)

        entries = profile_list[i].GetEntries()
        legend.AddEntry(profile_list[i], "#splitline{"+profile_labels[i]+"}{#bf{#scale[0.7]{%d entries}}}"%(entries), "l")

        if maximum > 0: profile_list[i].SetMaximum(maximum*1.3)
        profile_list[i].SetMinimum(minimum*0.7)
        profile_list[i].Draw("SAMES")

    if plot_div:
        for i in range(len(div_list)):
            div_list[i].SetMarkerColorAlpha(colorlist[i+len(profile_list)],.75)
            div_list[i].SetLineColorAlpha(colorlist[i+len(profile_list)],.65)
            div_list[i].SetLineWidth(3)
            div_list[i].SetTitle("")
            div_list[i].GetXaxis().SetLabelSize(0)
            div_list[i].GetXaxis().SetTitleSize(0)
            div_list[i].GetYaxis().SetLabelSize(0.05)
            div_list[i].GetYaxis().SetTitleSize(0.05)
            div_list[i].GetYaxis().SetTitleOffset(1.0)

            entries = div_list[i].GetEntries()
            legend.AddEntry(div_list[i], "#splitline{"+div_labels[i]+"}{#bf{#scale[0.7]{%d entries}}}"%(entries), "l")

            if maximum > 0: div_list[i].SetMaximum(maximum*1.3)
            div_list[i].SetMinimum(minimum*0.7)
            div_list[i].Draw("SAMES")

    legend.SetTextSize(0.03)
    legend.SetFillStyle(0)
    legend.SetBorderSize(0)
    legend.Draw("SAME")
    logo.Draw("SAME")
    cut_text.Draw("SAME")

    pad2.cd()

    maximum = max([ratio.GetMaximum() for ratio in ratio_list])*1.1
    minimum = min([ratio.GetMinimum() for ratio in ratio_list])*0.9
    
    for i in range(len(ratio_list)):
        ratio_list[i].SetMarkerColorAlpha(colorlist[i+len(profile_list)+len(div_list)],.75)
        ratio_list[i].SetLineColorAlpha(colorlist[i+len(profile_list)+len(div_list)],.65)
        ratio_list[i].SetLineWidth(3)
        ratio_list[i].SetTitle("")
        ratio_list[i].GetXaxis().SetTitleSize(0.1)
        ratio_list[i].GetXaxis().SetLabelSize(0.1)
        ratio_list[i].GetXaxis().SetTitleOffset(0.9)
        ratio_list[i].GetYaxis().SetLabelSize(0.05)
        ratio_list[i].GetYaxis().SetTitleSize(0.1)
        ratio_list[i].GetYaxis().SetTitle("GNN/SV1")
        ratio_list[i].GetYaxis().SetTitleOffset(0.5)

        if maximum > 0: ratio_list[i].SetMaximum(maximum)
        ratio_list[i].SetMinimum(minimum)
        ratio_list[i].Draw("SAMES")

    canv.cd()
    canv.SaveAs(filename)
    if log: gPad.Clear()

    for ratio in ratio_list:
        ratio.SetDirectory(0)
        del ratio
    canv.Clear()


def plot_confusion_matrix(conf_matrix, xlabels, ylabels, title, axis_labels, filename):
    fig, ax = plt.subplots(figsize=(7.5, 7.5))
    ax.matshow(conf_matrix, cmap=plt.cm.Blues, alpha=0.3)
    #ax.set_xlim([-0.5,len(xlabels)-0.5])
    ax.set_xticks(list(range(len(xlabels))))
    ax.set_xticklabels(xlabels)
    #ax.set_ylim([-0.5,len(ylabels)-0.5])
    ax.set_yticks(list(range(len(ylabels))))
    ax.set_yticklabels(ylabels)

    for i in range(conf_matrix.shape[0]):
        for j in range(conf_matrix.shape[1]):
            ax.text(x=j, y=i,s=conf_matrix[i, j], va='center', ha='center', size='xx-large')
 
    plt.xlabel(axis_labels[0], fontsize=18)
    plt.ylabel(axis_labels[1], fontsize=18)
    plt.title(title, fontsize=18)
    plt.savefig(filename)


def plot_roc_curve(canv, roc_b, roc_c, comp_roc_b, comp_roc_c, xlimits, ylimits, ctype, filename):
    canv.Clear()
    SetAtlasStyle()

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

    logo = TLatex(0.2,0.88, "#bf{#it{ATLAS}} #it{Internal}")
    logo.SetNDC(True)

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
    logo.Draw("SAME")

    canv.SaveAs(filename)
    canv.Clear()
