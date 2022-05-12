#!/bin/bash

NTUPLES="user.jmwagner.27266826.Akt4EMPf_BTagging201903._000006" # user.jmwagner.27266824.Akt4EMPf_BTagging201903._000006"
DATADIR=/global/cfs/cdirs/atlas/jmw464/gnn_data/

ENVNAME=dgl-env #name of conda environment that contains packages
ENTRIES=0

source activate $ENVNAME

printf "##########BEGINNING PROCESSING##########\n\n"

for NTUPLE in $NTUPLES
do
	printf "Running process_ntuple.py to create ${NTUPLE}.hdf5\n"
	python scripts/process_ntuple.py -n ${NTUPLE} -i ${DATADIR} -o ${DATADIR} -v ${ENTRIES}
	printf "\n"
done
