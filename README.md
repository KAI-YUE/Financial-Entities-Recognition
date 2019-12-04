# DS2019互联网金融新实体发现

环境：python3.5 + tensorflow1.10 + BERT

##### BERT模型介绍

**BERT**全称**B**idirectional **E**ncoder **R**epresentations from **T**ransformers，是预训练语言表示的方法，可以在大型文本语料库（如维基百科）上训练通用的“语言理解”模型，然后将该模型用于下游NLP任务，比如机器翻译、问答等等。 

报告中可以加多一些这方面的介绍。



互联网金融新实体发现其实是一个典型的命名实体识别(NER)问题，属于下游NLP问题。直接使用BERT

模型就可以得到一个初步的结果。首先我们需要下载BERT源码（从[BERT-TF](https://github.com/google-research/bert)下载bert源代码）。因为BERT模型是基于预训练模型进行微调的，我们自己是不可能完成模型的预训练的，我们需要直接下载使用Google提供的关于中文的bert预训练模型（从[BERT-Base Chinese](https://storage.googleapis.com/bert_models/2018_11_03/chinese_L-12_H-768_A-12.zip)下载模型，存放在checkpoint文件夹下）

进行这样的相关准备后，我们需要编写自己的BERT_NER.py文件，基本参考bert源码中的run_classifier.py文件，主要是需要编写我们的自己的`DataProcessor`和`NerProcessor`类，还有修改其他一些地方的引用关系，这是因为bert源码里面没有我们需要的专门的NER识别的操纵代码，所以需要自己去编写。编写时也参考了网上一些BERT_NER代码。`DataProcessor`和`NerProcessor`类具体代码如下，整个代码详见BERT_NER.py文件。

```python
class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    @classmethod
    def _read_data(cls, input_file):
        """Reads a BIO data."""
        with open(input_file) as f:
            lines = []
            words = []
            labels = []
            for line in f:
                contends = line.strip()
                word = line.strip().split(' ')[0]
                label = line.strip().split(' ')[-1]
                if contends.startswith("-DOCSTART-"):
                    words.append('')
                    continue
                # if len(contends) == 0 and words[-1] == '。':
                if len(contends) == 0:
                    l = ' '.join([label for label in labels if len(label) > 0])
                    w = ' '.join([word for word in words if len(word) > 0])
                    lines.append([l, w])
                    words = []
                    labels = []
                    continue
                words.append(word)
                labels.append(label)
            return lines


class NerProcessor(DataProcessor):
    def get_train_examples(self, data_dir):
        return self._create_example(
            self._read_data(os.path.join(data_dir, "train.txt")), "train"
        )

    def get_dev_examples(self, data_dir):
        return self._create_example(
            self._read_data(os.path.join(data_dir, "dev.txt")), "dev"
        )

    def get_test_examples(self,data_dir):
        return self._create_example(
            self._read_data(os.path.join(data_dir, "test.txt")), "test")


    def get_labels(self):
        # prevent potential bug for chinese text mixed with english text
        # return ["O", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC", "[CLS]","[SEP]"]
        return ["O", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC", "X","[CLS]","[SEP]"]

    def _create_example(self, lines, set_type):
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            text = tokenization.convert_to_unicode(line[1])
            label = tokenization.convert_to_unicode(line[0])
            examples.append(InputExample(guid=guid, text=text, label=label))
        return examples
```

这样子，我们配置好环境后，我们可以使用命令行来运行程序。

`python BERT_NER.py --task_name=NER --do_train=true --do_eval=true --data_dir=data/ --bert_config_file=checkpoint/bert_config.json --init_checkpoint=checkpoint/bert_model.ckpt --vocab_file=vocab.txt --output_dir=output/`

这是进行训练微调的，output会输出训练后的模型。data文件夹下就是我们的数据。分别为train.txt,dev.txt,test.txt数据集。训练完，再进行

`python BERT_NER.py --task_name=NER --do_predict=true --data_dir=data/ --bert_config_file=checkpoint/bert_config.json --init_checkpoint=output/ --vocab_file=vocab.txt --output_dir=output/`

来进行测试，即对test.txt文件进行预测输出最终结果。

比赛的数据都是比较原始的，再进行以上训练及测试操作时，需要对数据进行处理，得到能够进入程序运行的数据才行。

处理原始数据的代码详见`dataprocess.ipynb`

由于我们机器资源有限，比赛的数据跑不下来，并且我们只有简单的笔记本是由tensorflow_cpu跑程序，后来找了一些NER数据集来跑，依然无法胜任。所以我们没有一个完整的结果。实测程序没有问题，因为每次都是跑到了电脑卡死而被迫停止。