# Python Libraries
import os

# My Libraries
from src.loadConfig import loadConfig
from src.dataProcess import dataProcessor

def main(**kwargs):
#    config = loadConfig(kwargs["file_name"])
#    Processor = dataProcessor(config)
#    Processor.transfer(config.bert_Data_Dir) 
    pass

def train(config):
    train_command = \
    r"""python BERT_NER.py
   --task_name=NER
   --do_train=True
   --do_eval=True
   --data_dir={}
   --bert_config_file={}
   --init_checkpoint={}
   --vocab_file={}
   --output_dir={}
   --train_batch_size={}
   --num_train_epochs={}
   """.format(config.bert_Data_Dir,
       config.bert_config_file,
       config.init_checkpoint,
       config.vocab_file,
       config.output_dir,
       config.train_batch_size,
       config.num_train_epochs)

    train_command = train_command.replace('\n', '')
    os.system(train_command)

def predict(config):
    test_command = \
    r"""python BERT_NER.py
   --task_name=NER
   --do_predict=true
   --data_dir={}
   --bert_config_file={}
   --init_checkpoint={}
   --vocab_file={}
   --output_dir={}
   """.format(config.bert_Data_Dir,
       config.bert_config_file,
       config.output_dir,
       config.vocab_file,
       config.output_dir)
    
    test_command = test_command.replace('\n', '')
    print(test_command)
    os.system(test_command)

if __name__ == "__main__":
    config_file = 'config.json'
    config = loadConfig(config_file)
        
    processor = dataProcessor(config)
    processor.split_data(config.bert_Data_Dir)

    # train(config)