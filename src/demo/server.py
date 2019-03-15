# -*- coding: utf-8 -*-
"""
Purpose:    API which calls the keyword_extractor
Usage:      python serving_client.py
Author:     Jan Deriu
"""

import os, argparse
import random
import flask
import sys
from src.demo.demo_utils import NLGModel
import json

host = "127.0.0.1"
port = 5000


def main(args):
    config_fname = args.config

    config_dict = json.load(open(os.path.join('configurations', config_fname)))
    model = NLGModel(config_dict)

    app = flask.Flask(__name__)

    @app.route('/', methods=['GET'])
    def index():
        return flask.render_template('form.html')

    @app.route('/generate_utterance', methods=['POST'])
    def generate_utterance():
        process_mr = flask.request.form.to_dict()
        process_mr = {k: v for k,v in process_mr.items() if not v == ''}
        outputs, scores = model.generate_utterance_for_mr([process_mr])
        random.shuffle(outputs)
        oline = ''
        for o in outputs[:10]:
            oline += '<p>{}</p>'.format(o)

        return oline

    app.run(port=port, debug=False, host=host)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Preprocess Data')
    parser.add_argument('-c, --config_file', dest='config', default='config_full.json', type=str,  help='Path to the config file')
    args = parser.parse_args()
    main(args)
