import random
import os
import csv
from transformers import AutoTokenizer
import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

class Data:
    def __init__(self, args):
        set_seed(args.seed)
        self.args = args
        MAX_SEQ_LEN = {'clinc':30, 'stackoverflow':45, 'banking':55}
        TRAIN_EPOCH = {'clinc':60, 'stackoverflow':40, 'banking':50}
        LAMBDA_T = {'clinc': 0.7, 'stackoverflow': 0.1, 'banking': 0.3}
        LAMBDA_V = {'clinc': 0.51, 'stackoverflow': 0.50, 'banking': 0.51}
        args.num_train_epochs = TRAIN_EPOCH[args.dataset]
        args.lambda_t = LAMBDA_T[args.dataset]
        args.lambda_v = LAMBDA_V[args.dataset]
        self.max_seq_length = MAX_SEQ_LEN[args.dataset]
        
        self.data_dir = os.path.join(args.data_dir, args.dataset)
        self.all_label_list = self.get_label_list()
        self.n_known_cls = round(len(self.all_label_list) * args.known_cls_ratio)
        self.known_label_list = list(np.random.choice(np.array(self.all_label_list), self.n_known_cls, replace=False))
        self.num_labels = int(len(self.all_label_list) * args.cluster_num_factor)

        self.train_labeled_examples, self.train_unlabeled_examples = self.get_examples('train')
        print('num_labeled_samples',len(self.train_labeled_examples))
        print('num_unlabeled_samples',len(self.train_unlabeled_examples))
        self.eval_examples = self.get_examples('eval')
        self.test_examples = self.get_examples('test')
        self.semi_examples = self.get_semi(self.train_labeled_examples, self.train_unlabeled_examples)
        self.semi_input_ids, self.semi_input_mask, self.semi_segment_ids, self.semi_label_ids = self.semi_examples
        self.train_labeled_dataloader = self.get_data_loader(self.train_labeled_examples, 'train')
        self.train_semi_dataloader = self.get_semi_loader()
        self.eval_dataloader = self.get_data_loader(self.eval_examples, 'eval')
        self.test_dataloader = self.get_data_loader(self.test_examples, 'test')

    def get_label_list(self):
        data = self.read_tsv(os.path.join(self.data_dir, "train.tsv"))
        label_list = np.unique(np.array([i[1] for i in data], dtype=str))  
        return label_list

    def create_examples(self, lines, set_type):
        examples = []
        for (i, line) in enumerate(lines):
            if len(line) != 2:
                continue
            guid = "%s-%s" % (set_type, i)
            text_a = line[0]
            label = line[1]

            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
        return examples

    def get_examples(self, mode):
        if mode == 'train':
            ori_examples = self.create_examples(self.read_tsv(os.path.join(self.data_dir, "train.tsv")), 'train')
            train_labels = np.array([example.label for example in ori_examples])
            train_labeled_ids = []
            for label in self.known_label_list:
                num = round(len(train_labels[train_labels == label]) * self.args.labeled_ratio)
                pos = list(np.where(train_labels == label)[0])                
                train_labeled_ids.extend(random.sample(pos, num))

            train_labeled_examples, train_unlabeled_examples = [], []
            for idx, example in enumerate(ori_examples):
                if idx in train_labeled_ids:
                    train_labeled_examples.append(example)
                else:
                    train_unlabeled_examples.append(example)

            return train_labeled_examples, train_unlabeled_examples

        elif mode == 'eval':
            ori_examples = self.create_examples(self.read_tsv(os.path.join(self.data_dir, "dev.tsv")), 'train')
            eval_examples = []
            for example in ori_examples:
                if example.label in self.known_label_list:
                    eval_examples.append(example)
            return eval_examples

        elif mode == 'test':
            ori_examples = self.create_examples(self.read_tsv(os.path.join(self.data_dir, "test.tsv")), 'test')
            return ori_examples

    def get_semi(self, labeled_examples, unlabeled_examples):
        tokenizer = AutoTokenizer.from_pretrained(self.args.bert_model, do_lower_case=True)    
        labeled_features = self.convert_examples_to_features(labeled_examples, self.known_label_list, self.max_seq_length, tokenizer)
        unlabeled_features = self.convert_examples_to_features(unlabeled_examples, self.all_label_list, self.max_seq_length, tokenizer)

        labeled_input_ids = torch.tensor([f.input_ids for f in labeled_features], dtype=torch.long)
        labeled_input_mask = torch.tensor([f.input_mask for f in labeled_features], dtype=torch.long)
        labeled_segment_ids = torch.tensor([f.segment_ids for f in labeled_features], dtype=torch.long)
        labeled_label_ids = torch.tensor([f.label_id for f in labeled_features], dtype=torch.long)      

        unlabeled_input_ids = torch.tensor([f.input_ids for f in unlabeled_features], dtype=torch.long)
        unlabeled_input_mask = torch.tensor([f.input_mask for f in unlabeled_features], dtype=torch.long)
        unlabeled_segment_ids = torch.tensor([f.segment_ids for f in unlabeled_features], dtype=torch.long)
        unlabeled_label_ids = torch.tensor([f.label_id for f in unlabeled_features], dtype=torch.long)      

        semi_input_ids = torch.cat([labeled_input_ids, unlabeled_input_ids])
        semi_input_mask = torch.cat([labeled_input_mask, unlabeled_input_mask])
        semi_segment_ids = torch.cat([labeled_segment_ids, unlabeled_segment_ids])
        semi_label_ids = torch.cat([labeled_label_ids, unlabeled_label_ids])
        return semi_input_ids, semi_input_mask, semi_segment_ids, semi_label_ids

    def get_semi_loader(self):
        semi_input_ids, semi_input_mask, semi_segment_ids, semi_label_ids = self.semi_examples
        semi_data = TensorDataset(semi_input_ids, semi_input_mask, semi_segment_ids, semi_label_ids)
        semi_sampler = SequentialSampler(semi_data)
        semi_dataloader = DataLoader(semi_data, sampler=semi_sampler, batch_size = self.args.train_batch_size) 

        return semi_dataloader

    def get_data_loader(self, examples, mode):
        tokenizer = AutoTokenizer.from_pretrained(self.args.bert_model, do_lower_case=True)    
        
        if mode == 'train' or mode == 'eval':
            features = self.convert_examples_to_features(examples, self.known_label_list, self.max_seq_length, tokenizer)
        elif mode == 'test':
            features = self.convert_examples_to_features(examples, self.all_label_list, self.max_seq_length, tokenizer)

        input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
        input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
        segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
        label_ids = torch.tensor([f.label_id for f in features], dtype=torch.long)
        data = TensorDataset(input_ids, input_mask, segment_ids, label_ids)
        
        if mode == 'train':
            sampler = RandomSampler(data)
            dataloader = DataLoader(data, sampler=sampler, batch_size = self.args.train_batch_size)    
        elif mode == 'eval' or mode == 'test':
            sampler = SequentialSampler(data)
            dataloader = DataLoader(data, sampler=sampler, batch_size = self.args.eval_batch_size) 
        
        return dataloader

    def convert_examples_to_features(self, examples, label_list, max_seq_length, tokenizer):
        label_map = {}
        for i, label in enumerate(label_list):
            label_map[label] = i

        features = []
        for _, example in enumerate(examples):
            tokens_a = tokenizer(example.text_a, padding='max_length', max_length=max_seq_length, truncation=True)

            input_ids = tokens_a['input_ids']
            input_mask = tokens_a['attention_mask']
            segment_ids = tokens_a['token_type_ids']
            assert len(input_ids) == max_seq_length
            assert len(input_mask) == max_seq_length
            assert len(segment_ids) == max_seq_length

            label_id = label_map[example.label]

            features.append(
                InputFeatures(input_ids=input_ids,
                            input_mask=input_mask,
                            segment_ids=segment_ids,
                            label_id=label_id))
        return features

    def read_tsv(self, file):
        """Reads a tab separated value file."""
        with open(file, "r") as f:
            reader = csv.reader(f, delimiter="\t")
            lines = []
            for line in reader:
                lines.append(line)
            # skip the headline
            return lines[1:]
    

class InputExample(object):
    """Convert data to inputs for bert"""

    def __init__(self, guid, text_a, text_b=None, label=None):
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label

class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_id):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id