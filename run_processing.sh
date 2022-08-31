#!/bin/bash
#SBATCH -n 12
#SBATCH --qos=regular
#SBATCH -C knl
#SBATCH --time=19:59:00
#SBATCH --error=proc_%j.err
#SBATCH --output=proc_%j.out

NTUPLES="user.jmwagner.29159347.Akt4EMPf_BTagging201903._000001 user.jmwagner.29159347.Akt4EMPf_BTagging201903._000002 user.jmwagner.29159347.Akt4EMPf_BTagging201903._000003 user.jmwagner.29152222.Akt4EMPf_BTagging201903._000001 user.jmwagner.29152222.Akt4EMPf_BTagging201903._000002 user.jmwagner.29152222.Akt4EMPf_BTagging201903._000003"
INDIR=/global/cfs/cdirs/atlas/jmw464/gnn_data/NTUPLE/
OUTDIR=/global/cfs/cdirs/atlas/jmw464/gnn_data/
SAMPLE=btag_zh_tt_2m_v1
OPTIONFILE=options

ENVNAME=dgl-env #name of conda environment that contains packages

ENTRIES=( 0 0 0 0 0 0 ) #jets used per file (after cuts)

#create output directory if not already there
if [ ! -d ${OUTDIR}${SAMPLE}/ ]
then
	mkdir ${OUTDIR}${SAMPLE}/
fi

if [ "$YN" == "N" ] || [ "$YN" == "n" ]
then
	exit 1
fi

source activate $ENVNAME

printf "##########BEGINNING PROCESSING##########\n\n"

file_counter=0
for NTUPLE in $NTUPLES
do
	printf "Running create_graphs.py to transform ${NTUPLE}.hdf5 into GNN compatible data\n\n"
	#python scripts/create_graphs.py -n ${NTUPLE} -i ${INDIR} -o ${OUTDIR}${SAMPLE}/ -e ${ENTRIES[$file_counter]} -d ${SAMPLE} -f ${OPTIONFILE}
	printf "\n"
	file_counter=$(expr $file_counter + 1)
done

printf "Running combine_graphs.py to combine individual files\n\n"
#python scripts/combine_graphs.py -d ${OUTDIR}${SAMPLE}/ -s $SAMPLE -n "$NTUPLES" -f ${OPTIONFILE}

printf "Running plot_data.py to generate plots\n\n"
#python scripts/plot_data.py -d ${OUTDIR}${SAMPLE}/ -s $SAMPLE -f ${OPTIONFILE}

printf "Running prune_graphs.py to remove cut tracks\n\n"
python scripts/prune_graphs.py -d ${OUTDIR}${SAMPLE}/ -s $SAMPLE
