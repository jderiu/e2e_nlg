import numpy as np


def output_predefined_feautres(model, discriminator_models, model_input, da_acts, text_lex, delex_vocab, vocab, step='final', delimiter='', fname='logging/test_output'):
    ofile = open('{}_{}.txt'.format(fname, step), 'wt', encoding='utf-8')
    inverse_vocab = {v: k for (k, v) in vocab.items()}

    predictions = model.predict(model_input, batch_size=1024, verbose=1)
    predicted_sentences = predictions[0]
    confidence_scores = predictions[1]

    discr_scores = []
    for attribute, value_vector in da_acts.items():
        discriminator = discriminator_models[attribute]

        ypred_full = discriminator.predict(predicted_sentences, batch_size=1024, verbose=1)
        scores = []
        for ytrue, ypred in zip(value_vector, ypred_full):

            pred_lbl = np.argmax(ypred, axis=0)
            true_lbl = np.argmax(ytrue, axis=0)

            if pred_lbl == true_lbl:
                scores.append(1)
            else:
                scores.append(0)
        discr_scores.append(scores)

    sen_dict = []
    for i, (sentence, confidence_score, dscore) in enumerate(zip(predicted_sentences, confidence_scores, zip(*discr_scores))):
        scores = '\t'.join([str(x) for x in dscore] + [str(sum(dscore))])
        list_txt_idx = [int(x) for x in sentence.tolist()]
        txt_list = [inverse_vocab.get(int(x), '') for x in list_txt_idx]
        oline = delimiter.join(txt_list)
        for lex_key, val in text_lex.items():
            original_token = delex_vocab[lex_key][i]
            oline = oline.replace(val, original_token)

        sen_dict.append((oline, confidence_score, scores))

    for sentence, confidence_score, scores in sen_dict:
        ofile.write('{}\t{:.4f}\t{}'.format(sentence, confidence_score, scores) + '\n')
    ofile.write('\n')
    ofile.close()