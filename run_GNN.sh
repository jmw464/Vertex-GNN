#!/bin/bash
##SBATCH -n 1
#SBATCH --job-name=SVGNN
#SBATCH --qos=regular
##SBATCH -C haswell
#SBATCH --time=5:59:00
#SBATCH --error=jobs/GNN_%j.err
#SBATCH --output=jobs/GNN_%j.out
#SBATCH -A atlas_g
#SBATCH -G 1
#SBATCH -C gpu

RUN=30 #run number - used to distinguish output
DATADIR=/global/cfs/cdirs/atlas/jmw464/gnn_data/
OUTPUTDIR=/global/homes/j/jmw464/ATLAS/Vertex-GNN/output/
DATANAME=btag_zh06_tt06_cut_v7_1 #btag_ttbar_05_cut_v5 #btag_zhllcc_07_cut_v5
OPTIONFILE=options30

ENVNAME=test #name of conda environment that contains packages

EPOCHS=50

NORMED=1
MULTICLASS=0

#create output directory if not already there
if [ ! -d ${OUTPUTDIR}${RUN}/ ]
then
	mkdir ${OUTPUTDIR}${RUN}/
fi

source activate $ENVNAME

printf "##########BEGINNING TRAINING##########\n"
python scripts/GNN_main.py -r $RUN -e $EPOCHS -s $DATANAME -d ${DATADIR}${DATANAME}/ -o $OUTPUTDIR -n $NORMED -m $MULTICLASS -f ${OPTIONFILE} | tee ${OUTPUTDIR}${RUN}/${DATANAME}_results.txt

printf "##########PLOTTING RESULTS##########\n"
python scripts/plot_results.py -r $RUN -s $DATANAME -d ${DATADIR}${DATANAME}/ -o $OUTPUTDIR -f ${OPTIONFILE}
python scripts/visualize_attention.py -r $RUN -s $DATANAME -d ${DATADIR}${DATANAME}/ -o $OUTPUTDIR -f ${OPTIONFILE}

if [[ $MULTICLASS == 0 ]]
then
	python scripts/compare_performance.py -r $RUN -s $DATANAME -d ${DATADIR}${DATANAME}/ -o $OUTPUTDIR -n $NORMED -f ${OPTIONFILE} | tee ${OUTPUTDIR}${RUN}/${DATANAME}_comparisons.txt
fi
