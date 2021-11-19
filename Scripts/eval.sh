#!/bin/bash

echo "Automatic results : "
./trec_eval-9.0.7/trec_eval 2020qrels.txt ../data/baselines/y2_automatic_results_500.v1.0.run
echo "----------------------------------------------------------"

echo "Manual results : "
./trec_eval-9.0.7/trec_eval 2020qrels.txt ../data/baselines/y2_manual_results_500.v1.0.run

echo "----------------------------------------------------------"
echo "Raw results : "

./trec_eval-9.0.7/trec_eval 2020qrels.txt ../data/baselines/y2_raw_results_500.v1.0.run
echo "----------------------------------------------------------"
echo "Our rerank : "
./trec_eval-9.0.7/trec_eval 2020qrels.txt ../data/trec_result.run

