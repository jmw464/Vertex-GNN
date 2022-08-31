#!/bin/bash
#SBATCH -n 2
#SBATCH --job-name=h5
#SBATCH --qos=regular
#SBATCH -C haswell
#SBATCH --time=11:59:00
#SBATCH --error=jobs/h5_%j.err
#SBATCH --output=jobs/h5_%j.out

NTUPLES="user.jmwagner.29159347.Akt4EMPf_BTagging201903._000004"
DATADIR=/global/cfs/cdirs/atlas/jmw464/gnn_data/NTUPLE/
#NTUPLES="user.jmwagner.29152222.Akt4EMPf_BTagging201903._000004"
#DATADIR=/global/cfs/cdirs/atlas/jmw464/gnn_data/NTUPLE/

ENVNAME=dgl-env #name of conda environment that contains packages
MAX_EVENTS=0
EVENTS_PER_FILE=2000
REPROCESS=0

source activate $ENVNAME

printf "##########BEGINNING PROCESSING##########\n\n"

for NTUPLE in $NTUPLES
do
	printf "Running process_ntuple.py to create ${NTUPLE}.hdf5\n"
	python scripts/process_ntuple.py -n ${NTUPLE} -i ${DATADIR} -o ${DATADIR} -e ${MAX_EVENTS} -v ${EVENTS_PER_FILE} -r ${REPROCESS}
	printf "\n"
done
