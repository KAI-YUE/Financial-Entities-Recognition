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

def train():
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

def test():
    test_command = \
    r"""python BERT_NER.py
   --task_name=NER
   --do_predict=True
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

if __name__ == "__main__":
    config_file = 'config.json'
    config = loadConfig(config_file)
        


    print(train_command)

    # 
     