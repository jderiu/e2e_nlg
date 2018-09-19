#!/usr/bin/env bash
export PYTHONPATH=.
python src/training/train_classifiers.py -c config_test.json
python src/training/train_sclstm.py -c config_test.json

python src/training/train_sclstm.py -c config_utt_fw.json
python src/training/train_sclstm.py -c config_follow_fw.json
python src/training/train_sclstm.py -c config_form.json
python src/training/train_sclstm.py -c config_utt_follow_fw.json
python src/training/train_sclstm.py -c config_utt_fw_form.json
python src/training/train_sclstm.py -c config_follow_fw_form.json
python src/training/train_sclstm.py -c config_full.json

