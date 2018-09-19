#!/usr/bin/env bash
export PYTHONPATH=.
python src/data_processing/delexicalise_data.py -c config.json
python src/data_processing/vectorize_data.py -c config.json
python src/data_processing/surface_feature_vectors.py -c config.json
python src/data_processing/generate_evaluation_data.py -c config.json
