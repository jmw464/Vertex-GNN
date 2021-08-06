import os,sys,math,ROOT,glob
import numpy as np
import argparse
from ROOT import TFile, TH1D, TH1I, gROOT, TCanvas, gStyle, gPad, TLegend, TGaxis, THStack


#plot list of histograms with specified labels - scaling draws second axis shifted by -scaling[0] and scaled by 1/scaling[1]
def plot_hist(histlist, labellist, norm, log, overflow, filename, options, scaling=[]):
    canv = TCanvas("c1", "c1", 800, 600)
    canv.SetTopMargin(10)
    canv.SetRightMargin(100)
    if log: gPad.SetLogy()
    histstack = THStack("stack",histlist[0].GetTitle())
    legend = TLegend(0.76,0.88-0.08*len(histlist),0.91,0.88,'','NDC')
    colorlist = [4,8,2,6,1]

    if options: options += " NOSTACK"
    else: options = "NOSTACK"

    maximum = 0
    for i in range(len(histlist)):
        entries = histlist[i].GetEntries()
        mean = histlist[i].GetMean()
        histlist[i].SetLineColorAlpha(colorlist[i],0.65)
        histlist[i].SetLineWidth(3)
        nbins = histlist[i].GetNbinsX()
        legend.AddEntry(histlist[i], "#splitline{"+labellist[i]+"}{#splitline{%d entries}{mean=%.2f}}"%(entries, mean), "l")

        if overflow:
            histlist[i].SetBinContent(nbins, histlist[i].GetBinContent(nbins) + histlist[i].GetBinContent(nbins+1)) #overflow
            histlist[i].SetBinContent(1, histlist[i].GetBinContent(0) + histlist[i].GetBinContent(1)) #underflow
        if entries and norm: histlist[i].Scale(1./entries)

        if histlist[i].GetMaximum() > maximum: maximum = histlist[i].GetMaximum()
        histstack.Add(histlist[i])
        #if i == 0: histlist[i].Draw(options)
        #else: histlist[i].Draw(same+options)

    histstack.SetMaximum(maximum*1.4)
    histstack.Draw(options)
    histstack.GetXaxis().SetTitle(histlist[0].GetXaxis().GetTitle())
    histstack.GetYaxis().SetTitle(histlist[0].GetYaxis().GetTitle())
    xlimit = [histlist[0].GetBinLowEdge(1), histlist[0].GetBinLowEdge(histlist[0].GetNbinsX())+histlist[0].GetBinWidth(histlist[0].GetNbinsX())]
    if len(scaling) != 0:
        top_axis = TGaxis(xlimit[0],maximum*1.4,xlimit[1],maximum*1.4,(xlimit[0]-scaling[0])/scaling[1],(xlimit[1]-scaling[0])/scaling[1],510,"-")
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


#plot difference between two lists of normalized histograms
def plot_hist_diff(histlist1, histlist2, labellist, log, overflow, filename, options):
    canv = TCanvas("c1", "c1", 800, 600)
    canv.SetTopMargin(10)
    canv.SetRightMargin(100)
    if log: gPad.SetLogy()
    histstack = THStack("stack",histlist1[0].GetTitle())
    legend = TLegend(0.76,0.88-0.08*len(histlist1),0.91,0.88,'','NDC')
    colorlist = [4,8,2,6,1]

    if options: options += " NOSTACK"
    else: options = "NOSTACK"

    maximum = 0
    for i in range(len(histlist1)):
        entries = histlist1[i].GetEntries()
        bad_entries = histlist2[i].GetEntries()
        histlist1[i].SetLineColorAlpha(colorlist[i],0.65)
        histlist1[i].SetLineWidth(3)
        
        if entries: histlist1[i].Scale(1./entries)
        if bad_entries: histlist2[i].Scale(1./bad_entries)
        histlist1[i].Add(histlist2[i],-1.)

        nbins = histlist1[i].GetNbinsX()
        legend.AddEntry(histlist1[i], "#splitline{"+labellist[i]+"}{#splitline{%d total jets}{%d bad jets}}"%(entries, bad_entries), "l")

        if overflow:
            histlist1[i].SetBinContent(nbins, histlist1[i].GetBinContent(nbins) + histlist1[i].GetBinContent(nbins+1)) #overflow
            histlist1[i].SetBinContent(1, histlist1[i].GetBinContent(0) + histlist1[i].GetBinContent(1)) #underflow

        if histlist1[i].GetMaximum() > maximum: maximum = histlist1[i].GetMaximum()
        histstack.Add(histlist1[i])
        #if i == 0: histlist[i].Draw(options)
        #else: histlist[i].Draw(same+options)

    histstack.SetMaximum(maximum*1.4)
    histstack.Draw(options)
    histstack.GetXaxis().SetTitle(histlist1[0].GetXaxis().GetTitle())
    histstack.GetYaxis().SetTitle(histlist1[0].GetYaxis().GetTitle())

    legend.SetTextSize(0.02)
    legend.SetFillStyle(0)
    legend.SetBorderSize(0)
    legend.Draw("SAME")
    canv.SaveAs(filename)
    if log: gPad.Clear()
    canv.Clear()


def plot_metric_hist(hist_list, ylimit, filename):
    canv = TCanvas("c1", "c1", 800, 600)
    canv.SetGrid()
    colorlist = [1,4,8,2,6]
    gStyle.SetOptStat(0)
    legend = TLegend(0.76,0.88-0.08*len(hist_list),0.91,0.88,'','NDC')

    for i in range(len(hist_list)):
        legend.AddEntry(hist_list[i], "#splitline{%s}{#splitline{%d entries}{mean=%.2f}}"%(hist_list[i].GetName(), hist_list[i].GetEntries(), hist_list[i].GetMean()), "l")
        hist_list[i].SetLineColorAlpha(colorlist[i],0.65)
        hist_list[i].SetLineWidth(3)

        if hist_list[i].GetEntries(): hist_list[i].Scale(1./(hist_list[i].GetEntries()))
        hist_list[i].SetMaximum(ylimit[1])
        hist_list[i].SetMinimum(ylimit[0])
        
        if i == 0: hist_list[i].Draw()
        else:hist_list[i].Draw("SAMES")
    
    legend.SetTextSize(0.02)
    legend.SetFillStyle(0)
    legend.SetBorderSize(0)
    legend.Draw("SAME")
    canv.SaveAs(filename)
    canv.Clear()


def plot_profile(profile_list, labels, ylimit, overflow, filename):
    canv = TCanvas("c1", "c1", 800, 600)
    canv.SetGrid()
    colorlist = [1,4,8,2,6]
    gStyle.SetOptStat(0)
    legend = TLegend(0.76,0.88-0.08*len(profile_list),0.91,0.88,'','ndc')

    for i in range(len(profile_list)):
        legend.AddEntry(profile_list[i], "#splitline{%s}{#splitline{%d entries}{y-mean=%.2f}}"%(labels[i], profile_list[i].GetEntries(), profile_list[i].GetMean(2)), "l")
        profile_list[i].SetLineColorAlpha(colorlist[i],0.65)
        profile_list[i].SetLineWidth(3)
        profile_list[i].SetMaximum(ylimit[1])
        profile_list[i].SetMinimum(ylimit[0])

        nbins = profile_list[i].GetNbinsX()
        if overflow:
            profile_list[i].SetBinContent(nbins, profile_list[i].GetBinContent(nbins) + profile_list[i].GetBinContent(nbins+1)) #overflow
            profile_list[i].SetBinContent(1, profile_list[i].GetBinContent(0) + profile_list[i].GetBinContent(1)) #underflow

        if i == 0: profile_list[i].Draw()
        else:profile_list[i].Draw("sames")

    legend.SetTextSize(0.02)
    legend.SetFillStyle(0)
    legend.SetBorderSize(0)
    legend.Draw("same")
    canv.SaveAs(filename)
    canv.Clear()
