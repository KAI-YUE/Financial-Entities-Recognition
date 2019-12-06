# Python Libraries
import re
import os
import logging
import random
import pandas as pd
import numpy as np

class dataProcessor(object):
    def __init__(self, config):
 
        self.train_dir = config.Train_Dir
        self.test_dir = config.Test_Dir
        self.data_dir = config.bert_Data_Dir

        self.num_dev = int(config.num_dev)
        self.num_test = int(config.num_test)
        self.data_dir = str(config.bert_Data_Dir)
        self.logger = logging.getLogger()
        self.logger.setLevel(logging.INFO)
        sh = logging.StreamHandler()
        sh.setLevel(logging.INFO)
        self.logger.addHandler(sh)

    ####################### Public Methods ########################
    def set_param(self, **kwargs):
        """
        Set parameters of the model
        """
        for arg, value in kwargs.items():
            command = "self.{}".format(arg)
            exec(command + "={}".format(value))
    
    def split_data(self, output_dir):
        """
        Split the train.csv data for Bert model, in case that the test data do not have labels.
        """
        train_df = pd.read_csv(self.train_dir, encoding='utf-8')
        train_df = train_df.fillna("")

        # Concatenate the title and text if they are different
        for i in range(train_df.shape[0]):
            if (train_df.iloc[i]['title'] != train_df.iloc[i]['text']):
                train_df.iloc[i]['text'] = "".join((train_df.iloc[i]['title'], 
                                        train_df.iloc[i]['text']))
            else:
                train_df.iloc[i]['text'] = train_df.iloc[i]['text']

        train_df['text'] = train_df['text'].apply(self._word_filter)
        
        indices = np.arange(0, train_df.shape[0])
        random.shuffle(indices)
        np.savetxt(os.path.join(self.data_dir, 'indices.txt'),indices)

        with open(os.path.join(output_dir, "dev.txt"), 'w', encoding='utf-8') as fp:    
            self._dump_data(train_df.iloc[indices[:self.num_dev]], fp)
        
        with open(os.path.join(output_dir, "test.txt"), 'w', encoding='utf-8') as fp:
            self._dump_data(train_df.iloc[indices[self.num_dev: self.num_dev+self.num_test]], fp)

        with open(os.path.join(output_dir, "train.txt"), 'w', encoding='utf-8') as fp:
            self._dump_data(train_df.iloc[indices[self.num_dev+self.num_test:]], fp)

    def transfer_origin(self, output_dir):
        """
        Transform the .csv data for Bert model
        """
        train_df = pd.read_csv(self.train_dir, encoding='utf-8')
        test_df = pd.read_csv(self.test_dir, encoding='utf-8')
        
        train_df = train_df.fillna("")
        test_df = test_df.fillna("")
        
        # Concatenate the title and text if they are different
        for i in range(train_df.shape[0]):
            if (train_df.iloc[i]['title'] != train_df.iloc[i]['text']):
                train_df.iloc[i]['text'] = "".join((train_df.iloc[i]['title'], 
                                        train_df.iloc[i]['text']))
            else:
                train_df.iloc[i]['text'] = train_df.iloc[i]['text']
        
        for i in range(test_df.shape[0]):
            if (test_df.iloc[i]['title'] != test_df.iloc[i]['text']):
                test_df.iloc[i]['text'] = "".join((test_df.iloc[i]['title'], 
                                         test_df.iloc[i]['text']))
            else:
                test_df.iloc[i]['text'] = test_df.iloc[i]['text']
        
        train_df['text'] = train_df['text'].apply(self._word_filter)
        test_df['text'] = test_df['text'].apply(self._word_filter)
    
        indices = np.arange(0, train_df.shape[0])
        random.shuffle(indices)
        np.savetxt(os.path.join(self.data_dir, 'indices.txt'),indices)
        
        with open(os.path.join(output_dir, "train.txt"), 'w', encoding='utf-8') as fp:    
            self._dump_data(train_df.iloc[indices[:-self.num_dev]], fp)
        
        with open(os.path.join(output_dir, "dev.txt"), 'w', encoding='utf-8') as fp:
            self._dump_data(train_df.iloc[indices[-self.num_dev:]], fp)
        
        with open(os.path.join(output_dir, "test.txt"), 'w', encoding='utf-8') as fp:
            self._dump_data_without_label(test_df, fp)
                
    
    ###################### Private Methods ############################
    @staticmethod
    def _word_filter(x):
        try:
            x = x.strip()
        except:
            return ''
        x = re.sub('\?{2,}|\{IMG:.*\}','',x)   # Remove ??**?? and IMG{:*}
        x = re.sub('<.*>', '', x)              # Remove html
        x = re.sub('http.*\w', '', x)          # Remove http 
        return x
    
    def _dump_data_without_label(self, dataFrame, fp):
        for row in dataFrame.itertuples():
            for c in row.text:
                try:
                    fp.write(c)
                except Exception as e:
                    self.logger.warning(e)
                    continue

        fp.write('\n')

    def _dump_data(self, dataFrame, fp):
        for row in dataFrame.itertuples():
        
            text_lbl = row.text
            entities = str(row.unknownEntities).split(';')
            if entities == ['']:
                for c in row.text:
                    fp.write('{} O\n'.format(c))
                
                fp.write('\n')
            else:
                for entity in entities:
                    text_lbl = text_lbl.replace(entity, '\0' + (len(entity)-1)*'\1')
                
                for c1, c2 in zip(row.text, text_lbl):
                    if c2 == '\0':
                        fp.write('{0} {1}\n'.format(c1, 'B-ORG'))
                    elif c2 == '\1':
                        fp.write('{0} {1}\n'.format(c1, 'I-ORG'))
                    else:
                        try:
                            fp.write('{0} {1}\n'.format(c1, 'O'))
                        except Exception as e:
                            self.logger.warning(e)
                            continue
                
                fp.write('\n')

   
    