import argparse, json, os
import numpy as np
from src.data_processing.surface_feature_vectors import _load_vectorized_data
from src.data_processing.utils import PREPROCESSED_DATA


def attr_value_in_line(attr, value, line, ignore=False):
    mentions_atts = False

    if attr == 'familyFriendly':
        mentions_atts = False
        for y in yes_fam_firendly:
            if y.lower() in line:
                mentions_atts = True

    elif attr == 'priceRange':
        mentions_atts = False
        if value.lower() in line:
            mentions_atts = True

        if value in ['cheap', 'less than £20']:
            if 'less than £20' in line or 'low-priced' in line or 'under £20' in line or 'less than 20 pounds' in line or 'cheap' in line or 'inexpensive' in line or 'low price' in line:
                mentions_atts = True

        if value in ['moderate', '£20-25']:
            if 'moderate' in line or '£20-25' in line or 'moderately' in line or 'medium priced' in line or 'average' in line or 'mid priced' in line or 'mid-priced' in line or 'medium-priced' in line:
                mentions_atts = True

        if value in ['high', 'more than £30']:
            if 'more than £30' in line or 'high' in line or 'expensive' in line or '£30' in line or 'above average' in line or '30' in line:
                mentions_atts = True

    elif attr == 'customer rating':
        mentions_atts = False
        if value.lower() in line:
            mentions_atts = True

        if value in ['low', '1 out of 5']:
            if 'low' in line or 'less than £20' in line or 'low-priced' in line or 'one out of five' in line or 'one star' in line or '1 star' in line or 'poor customer' in line:
                mentions_atts = True

        if value in ['average', '3 out of 5']:
            if 'moderate' in line or '£20-25' in line or 'moderately' in line or 'three out of five' in line or 'three star' in line or 'average' in line or '3 star' in line:
                mentions_atts = True

        if value in ['high', '5 out of 5']:
            if 'more than £30' in line or 'high' in line or 'excellent' in line or 'five out of five' in line or 'five star' in line or '5 star' in line:
                mentions_atts = True

    elif attr == 'food':
        mentions_atts = False
        if value.lower() in line:
            mentions_atts = True

        if value == 'Japanese':
            if 'japanese' in line or 'sushi' in line:
                mentions_atts = True

        if value == 'Italian':
            if 'italian' in line or 'pasta' in line:
                mentions_atts = True

        if value == 'French':
            if 'french' in line or 'wine' in line:
                mentions_atts = True
        if value == 'English':
            if 'english' in line or 'breakfast' in line or 'british' in line:
                mentions_atts = True

    elif attr == 'area':
        mentions_atts = False
        if value in line:
            mentions_atts = True

        if value == 'riverside':
            if 'riverside' in line or 'river' in line:
                mentions_atts = True

    else:
        if value.lower() in line:
            mentions_atts = True

    return mentions_atts


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Preprocess Data')
    parser.add_argument('-c, --config_file', dest='config', default='config.json', type=str, help='Path to the config file')

    args = parser.parse_args()

    config_fname = args.config

    config_dict = json.load(open(os.path.join('configurations', config_fname)))

    # filename
    valid_data_mr_only = config_dict['valid_data_mr_only']
    test_data_mr_only = config_dict['test_data_mr_only']

    out_tag = config_dict['processing_tag']
    data_base_path = config_dict['data_base_path']

    char_vocab_fname = config_dict['character_vocabulary']
    max_sentence_len = config_dict['max_sentence_len']

    # path to data
    delex_data_path = os.path.join(data_base_path, PREPROCESSED_DATA, out_tag)

    test_data_delex = _load_vectorized_data(delex_data_path, test_data_mr_only)#parsed_mrs, vectorised_mrs, sample (syntactic man type-> vector)
    print(test_data_delex.keys())
    parsed_mrs = test_data_delex['parsed_mrs']
    test_data_delex = None

    mr_file = open(os.path.join(data_base_path, PREPROCESSED_DATA, out_tag, 'mr_data_ontology.json'), 'rt', encoding='utf-8')
    mr_data_ontology = json.load(mr_file) #atribute -> values -> idx

    raw_input = [x.replace('\n', '') for x in open('outputs/sc_{}/final_output_test.txt'.format(config_dict['logging_tag']), 'rt', encoding='utf-8').readlines()]

    eval_file = open('outputs/sc_{}/rule_based_eval.txt'.format(config_dict['logging_tag']), 'wt', encoding='utf-8')

    yes_fam_firendly = [
        'child friendly',
        'kid friendly',
        'kids friendly',
        'family friendly',
        'children friendly',
        'family-friendly',
        'no kids',
        'adult only',
        'children are welcome',
        'child-friendly',
        'kid-friendly',
    ]

    scores = []
    for line, mr in zip(raw_input, parsed_mrs):
        name = mr.get('name')
        line = line.replace(name, 'X-name')

        near = mr.get('near', None)
        if near:
            line = line.replace(near, 'X-near')

        line = line.lower()
        curr_score = 8
        for attr, value in mr.items():
            if attr == 'name' or attr == 'near':
                continue

            mentions_atts = attr_value_in_line(attr, value, line)

            if not mentions_atts:
                curr_score -= 1
                eval_file.write('Attribute: {}, Value: {} should be mentioned\n'.format(attr, value))

        for attr, values in mr_data_ontology.items():
            if attr in mr.keys():
                continue

            for value in values:
                if attr == 'priceRange' and 'customer rating' in mr.keys():
                    if value in ['cheap', 'less than £20'] and mr['customer rating'] in ['low', '1 out of 5']:
                        continue
                    if value in ['moderate', '£20-25'] and mr['customer rating'] in ['average', '3 out of 5']:
                        continue
                    if value in ['high', 'more than £30'] and mr['customer rating'] in ['high', '5 out of 5']:
                        continue

                elif attr == 'customer rating' and 'priceRange' in mr.keys():
                    if mr['priceRange'] in ['cheap', 'less than £20'] and value in ['low', '1 out of 5']:
                        continue
                    if mr['priceRange'] in ['moderate', '£20-25'] and value in ['average', '3 out of 5']:
                        continue
                    if mr['priceRange'] in ['high', 'more than £30'] and value in ['high', '5 out of 5']:
                        continue

                mentions_atts = attr_value_in_line(attr, value, line)

                if mentions_atts:
                    curr_score -=1
                    eval_file.write('Attribute: {}, Value: {} should not be mentioned\n'.format(attr, value))
                    break

        scores.append(curr_score)
        if curr_score < 8:
            eval_file.write('Utterance: {}\nScore:{}\nMR: {}\n\n'.format(line, curr_score, mr))

    eval_file.write('Average Scores: {}\tCorrect Rate: {}\tError Rate: {}\n'.format(np.mean(scores), np.mean(scores) / 8, 1 - np.mean(scores) / 8))
    eval_file.close()