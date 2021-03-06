#!/bin/bash

pipeline=$1
dataset=$2

IFS='/'

echo $pipeline
read -ra PRIM <<< ${pipeline}
for i in "${PRIM[@]}"; do
    if [[ $i == *"d3m.primitives"* ]]; then
        echo $i
        export PRIMITIVE=$i
    fi
done
unset IFS

export PROBLEM="/input/${dataset}/${dataset}_problem/problemDoc.json"
export DATA="/input/${dataset}/${dataset}_dataset/datasetDoc.json"
export DATA_TEST="/input/${dataset}/TEST/dataset_TEST/datasetDoc.json"
export DATA_SCORE="/input/${dataset}/SCORE/dataset_SCORE/datasetDoc.json"

echo $PRIMITIVE
cd /src/d3m/
echo "ARGS "
echo $pipeline
echo $PROBLEM
echo $DATA
echo $DATA_TEST
echo $DATA_SCORE

python3 -m d3m runtime fit-score -p ${pipeline} -r $PROBLEM -i $DATA -t $DATA_TEST -a $DATA_SCORE -o /output/${dataset}_results.csv -O /output/$PRIMITIVE_${dataset}_run.yml

