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
        self.num_dev = config.num_dev
        
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
    
    
    def transfer(self, output_dir):
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
        
        train_df['title'] = train_df['text'].apply(self._word_filter)
        test_df['title'] = test_df['text'].apply(self._word_filter)
    
        indices = np.arange(0, train_df.shape[0])
        random.shuffle(indices)
        
        with open(os.path.join(output_dir, "train.txt"), 'w', encoding='utf-8') as fp:    
            self._dump_data(train_df.iloc[indices[:-self.num_dev]], fp)
        
        with open(os.path.join(output_dir, "dev.txt"), 'w', encoding='utf-8') as fp:
            self._dump_data(train_df.iloc[indices[-self.num_dev:]], fp)
        
        with open(os.path.join(output_dir, "test.txt"), 'w', encoding='utf-8') as fp:
            self._dump_data(test_df, fp)
                
    
    ###################### Private Methods ############################
    @staticmethod
    def _word_filter(x):
        try:
            x = x.strip()
        except:
            return ''
        x = re.sub('\?\?+|\{IMG:\d\}','',x)
        return x
    
    def _dump_data(self, dataFrame, fp):
        for row in dataFrame.itertuples():
        
            text_lbl = row.text
            entities = str(row.unknownEntities).split(';')
            for entity in entities:
                text_lbl = text_lbl.replace(entity, 'Ё' + (len(entity)-1)*'Ж')
            
            for c1, c2 in zip(row.text, text_lbl):
                if c2 == 'Ё':
                    fp.write('{0} {1}\n'.format(c1, 'B-ORG'))
                elif c2 == 'Ж':
                    fp.write('{0} {1}\n'.format(c1, 'I-ORG'))
                else:
                    try:
                        fp.write('{0} {1}\n'.format(c1, 'O'))
                    except Exception as e:
                        self.logger.info(e)
                        continue
                
            fp.write('\n')

   
    