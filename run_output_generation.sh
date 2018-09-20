#!/usr/bin/env bash
export PYTHONPATH=.

python src/evaluation/generate_output.py -c config.json
python src/evaluation/generate_output.py -c config_utt_fw.json
python src/evaluation/generate_output.py -c config_follow_fw.json
python src/evaluation/generate_output.py -c config_form.json
python src/evaluation/generate_output.py -c config_utt_follow_fw.json
python src/evaluation/generate_output.py -c config_utt_fw_form.json
python src/evaluation/generate_output.py -c config_follow_fw_form.json
python src/evaluation/generate_output.py -c config_full.json
