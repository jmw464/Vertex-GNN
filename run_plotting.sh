#!/bin/bash

RUN=6
NTUPLES="user.jmwagner.24900045.Akt4EMPf_BTagging201903._000007 user.jmwagner.24900045.Akt4EMPf_BTagging201903._000016"
DATADIR=/global/homes/j/jmw464/ATLAS/Vertex-GNN/data/
OUTPUTDIR=/global/homes/j/jmw464/ATLAS/Vertex-GNN/output/
DATA=btag_07_19_cut_v2c

ENTRIES=100000
BADJETS=1

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
