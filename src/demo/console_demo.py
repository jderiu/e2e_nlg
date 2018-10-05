import argparse, json, os, random

from src.demo.demo_utils import NLGModel


def isInt(s):
    try:
        int(s)
        return True
    except ValueError:
        return False


def get_mr_from_user(mr_data_ontology, delex_attributes):

    mr = dict()
    for attr, values in mr_data_ontology.items():
        if attr in delex_attributes:
            aval = input('Type {} '.format(attr))
            if aval:
                mr[attr] = aval
        else:
            oline = ''
            for val, idx in values.items():
                oline += ' {} [{}],'.format(val, idx)
            non_val = len(values)
            oline += ' None [{}]'.format(non_val)

            inverse_ontology = {v: k for k, v in values.items()}

            aval = input('Select {}:{} '.format(attr, oline))
            if isInt(aval):
                aval = int(aval)

            aval = inverse_ontology.get(aval, non_val)
            if not aval == non_val:
                mr[attr] = aval

    process_mr = [mr]

    return process_mr


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Preprocess Data')
    parser.add_argument('-c, --config_file', dest='config', default='config.json', type=str,
                        help='Path to the config file')

    args = parser.parse_args()

    config_fname = args.config

    config_dict = json.load(open(os.path.join('configurations', config_fname)))
    model = NLGModel(config_dict)

    while True:
        process_mr = get_mr_from_user(model.mr_data_ontology, model.delex_attributes)
        oline = model.generate_utterance_for_mr(process_mr)

        random.shuffle(oline)

        print(process_mr[0])
        for line in oline[:10]:
            print(line)

        cont = input('Press 0 to end')
        if cont == '0':
            break