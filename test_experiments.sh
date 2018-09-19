#!/usr/bin/env bash
export PYTHONPATH=.
python src/data_processing/delexicalise_data.py -c config_test.json
python src/data_processing/vectorize_data.py -c config_test.json
python src/data_processing/surface_feature_vectors.py -c config_test.json
python src/data_processing/generate_evaluation_data.py -c config_test.json

python src/training/train_classifiers.py -c config_test.json
python src/training/train_sclstm.py -c config_test.json


