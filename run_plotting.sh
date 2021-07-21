#!/bin/bash

RUN=25
NTUPLES="user.jmwagner.26222492.Akt4EMPf_BTagging201903._000003" #"user.jmwagner.24900045.Akt4EMPf_BTagging201903._000007 user.jmwagner.24900045.Akt4EMPf_BTagging201903._000016"
DATADIR=/global/cfs/cdirs/atlas/jmw464/data/
OUTPUTDIR=/global/homes/j/jmw464/ATLAS/Vertex-GNN/output/
DATA=btag_zhllcc_03_cut_v5c

ENTRIES=10000
BADJETS=0

#create output directory if not already there
if [ ! -d ${OUTPUTDIR}${RUN}/ ] && [ $BADJETS != 0 ]
then
	mkdir ${OUTPUTDIR}${RUN}/
fi

if $PLOT
then
	printf "##########BEGINNING PLOTTING##########\n"
	python scripts/generate_plots.py -r $RUN -e $ENTRIES -d $DATA -o $OUTPUTDIR -n "$NTUPLES" -b $BADJETS -i $DATADIR
fi
