"""
Data Preprocess
"""
import pandas as pd
import codecs
import re

train_dir = 'F:\\Projects\\FinancialEntities\\data\\Train_Data.csv'
test_dir = 'F:\\Projects\\FinancialEntities\\data\\Test_Data.csv'

train_df = pd.read_csv(train_dir)
test_df = pd.read_csv(test_dir)

def stop_words(x):
    try:
        x = x.strip()
    except:
        return ''
    x = re.sub('\?\?+','',x)
    x = re.sub('\{IMG:.?.?.?\}','',x)
    return x

train_df['text'] =  train_df['title'].fillna('') + train_df['text'].fillna('')
test_df['text'] =  test_df['title'].fillna('') + test_df['text'].fillna('')

train_df['text'] = train_df['text'].apply(stop_words)
test_df['text'] = test_df['text'].apply(stop_words)

train_df = train_df[~train_df['unknownEntities'].isnull()]

with codecs.open('./bert-chinese-ner/data/train.txt', 'w') as up:
    for row in train_df.iloc[:-200].itertuples():
        # print(row.unknownEntities)

        text_lbl = row.text
        entitys = str(row.unknownEntities).split(';')
        for entity in entitys:
            text_lbl = text_lbl.replace(entity, 'Ё' + (len(entity)-1)*'Ж')
        
        for c1, c2 in zip(row.text, text_lbl):
            if c2 == 'Ё':
                up.write('{0} {1}\n'.format(c1, 'B-ORG'))
            elif c2 == 'Ж':
                up.write('{0} {1}\n'.format(c1, 'I-ORG'))
            else:
                up.write('{0} {1}\n'.format(c1, 'O'))
        
        up.write('\n')
        
with codecs.open('./bert-chinese-ner/data/dev.txt', 'w') as up:
    for row in train_df.iloc[-200:].itertuples():
        # print(row.unknownEntities)

        text_lbl = row.text
        entitys = str(row.unknownEntities).split(';')
        for entity in entitys:
            text_lbl = text_lbl.replace(entity, 'Ё' + (len(entity)-1)*'Ж')
        
        for c1, c2 in zip(row.text, text_lbl):
            if c2 == 'Ё':
                up.write('{0} {1}\n'.format(c1, 'B-ORG'))
            elif c2 == 'Ж':
                up.write('{0} {1}\n'.format(c1, 'I-ORG'))
            else:
                up.write('{0} {1}\n'.format(c1, 'O'))
        
        up.write('\n')
        

with codecs.open('./bert-chinese-ner/data/test.txt', 'w') as up:
    for row in test_df.iloc[:].itertuples():

        text_lbl = row.text
        for c1 in text_lbl:
            up.write('{0} {1}\n'.format(c1, 'O'))
        
        up.write('\n')
        

with codecs.open('./bert-chinese-ner/data/dev.txt', 'w') as up:
    for row in train_df.iloc[-500:].itertuples():
        # print(row.unknownEntities)

        text_lbl = row.text
        entitys = str(row.unknownEntities).split(';')
        for entity in entitys:
            text_lbl = text_lbl.replace(entity, 'Ё' + (len(entity)-1)*'Ж')
        
        for c1, c2 in zip(row.text, text_lbl):
            if c2 == 'Ё':
                up.write('{0} {1}\n'.format(c1, 'B-ORG'))
            elif c2 == 'Ж':
                up.write('{0} {1}\n'.format(c1, 'I-ORG'))
            else:
                up.write('{0} {1}\n'.format(c1, 'O'))
        
        up.write('\n')
        
with codecs.open('./bert-chinese-ner/data/test.txt', 'w') as up:
    for row in test_df.iloc[:].itertuples():

        text_lbl = row.text
        for c1 in text_lbl:
            up.write('{0} {1}\n'.format(c1, 'O'))
        
        up.write('\n')