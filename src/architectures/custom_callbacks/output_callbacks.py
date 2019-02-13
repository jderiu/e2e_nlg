import logging
from keras.callbacks import Callback
from src.architectures.custom_callbacks.output_text import  output_predefined_feautres


class LexFeatureOutputCallbackVanilla(Callback):
    def __init__(self, config_data, test_model, discriminators, model_input, da_acts, lex_dict, delex_vocab, out_frequency, char_vocab, delimiter, fname='logging/test_output'):
        self.model_input = model_input
        self.char_vocab = char_vocab
        self.test_model = test_model
        self.out_frequency = out_frequency
        self.delimiter = delimiter
        self.fname = fname
        self.lex_dict = lex_dict
        self.discriminators = discriminators
        self.da_acts = da_acts
        self.delex_vocab = delex_vocab

        super(LexFeatureOutputCallbackVanilla, self).__init__()

    def on_epoch_end(self, epoch, logs={}):
        if epoch % self.out_frequency == 0:
            output_predefined_feautres(self.test_model, self.discriminators, self.model_input, self.da_acts, self.lex_dict, self.delex_vocab, self.char_vocab, str(epoch), delimiter=self.delimiter, fname=self.fname)

        loss = logs.get('loss', '-')
        main_loss = logs.get('main_loss', '-')
        dialogue_act_loss = logs.get('dialogue_act_loss', '-')
        dialogue_history_loss = logs.get('dialogue_history_loss', '-')
        val_loss = logs.get('val_loss', '-')
        val_main_loss = logs.get('val_main_loss', '-')
        val_dialogue_act_loss = logs.get('val_dialogue_act_loss', '-')
        val_dialogue_history_loss = logs.get('val_dialogue_history_loss', '-')

        logging.info('{0} TRAINING: Loss: {1: <32}\tReconstruction Loss: {2: <32}\tDA Loss: {3: <32}\tDA Hist Loss{4: <32}'.format(epoch,loss, main_loss, dialogue_act_loss, dialogue_history_loss))
        logging.info('{0} VALIDATION: Loss: {1: <32}\tReconstruction Loss: {2: <32}\tDA Loss: {3: <32}\tDA Hist Loss{4: <32}'.format(epoch,val_loss, val_main_loss, val_dialogue_act_loss, val_dialogue_history_loss))


