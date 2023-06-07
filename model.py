import os
import copy
import random
import numpy as np
import torch
import torch.nn as nn
from transformers import AdamW, BertModel
import torch.nn.functional as F
from transformers import WEIGHTS_NAME, CONFIG_NAME
from sklearn.metrics import accuracy_score
from transformers import get_linear_schedule_with_warmup

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

class PretrainModelManager:
    def __init__(self, args, data):
        set_seed(args.seed)
        self.args = args
        self.data = data
        self.model = BertForModel(args, data.n_known_cls)
        if args.freeze_bert_parameters:
            self.freeze_parameters(self.model)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.optimizer = self.get_optimizer(args)
        self.best_eval_score = 0
        self.num_training_steps = int(
            len(data.train_labeled_examples) / args.train_batch_size) * args.num_pretrain_epochs
        self.num_warmup_steps= int(self.args.warmup_proportion * self.num_training_steps) 
        self.scheduler = get_linear_schedule_with_warmup(optimizer=self.optimizer, num_warmup_steps=self.num_warmup_steps, num_training_steps=self.num_training_steps) 

    def freeze_parameters(self, model):
        for name, param in model.bert.named_parameters():
            param.requires_grad = False
            if "encoder.layer.11" in name or "pooler" in name:
                param.requires_grad = True

    def get_optimizer(self, args):
        param_optimizer = list(self.model.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=args.lr_pre, correct_bias=False, no_deprecation_warning=True)
        return optimizer

    def save_model(self):
        if not os.path.exists(self.args.pretrain_dir):
            os.makedirs(self.args.pretrain_dir)
        self.save_model = self.model.module if hasattr(self.model, 'module') else self.model
        model_file = os.path.join(self.args.pretrain_dir, WEIGHTS_NAME)
        model_config_file = os.path.join(self.args.pretrain_dir, CONFIG_NAME)
        torch.save(self.save_model.state_dict(), model_file)
        with open(model_config_file, "w") as f:
            f.write(self.save_model.config.to_json_string())

    def train(self):
        wait = 0
        best_model = None
        for epoch in range(int(self.args.num_pretrain_epochs)):
            self.model.train()
            tr_loss = 0
            nb_tr_examples, nb_tr_steps = 0, 0

            for step, batch in enumerate(self.data.train_labeled_dataloader):
                batch = tuple(t.to(self.device) for t in batch)
                input_ids, input_mask, segment_ids, label_ids = batch
                with torch.set_grad_enabled(True):
                    loss, _ = self.model(input_ids, segment_ids, input_mask, label_ids, mode="train")
                    loss.backward()
                    tr_loss += loss.item()
                    nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

                    self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad()

                    nb_tr_examples += input_ids.size(0)
                    nb_tr_steps += 1

            loss = tr_loss / nb_tr_steps
            print('Epoch {} train_loss: {}'.format(epoch, loss))

            eval_score = self.eval()
            print('Epoch {} eval_score: {}'.format(epoch, eval_score))

            if eval_score > self.best_eval_score:
                best_model = copy.deepcopy(self.model)
                wait = 0
                self.best_eval_score = eval_score
            else:
                wait += 1
                if wait >= self.args.wait_patient:
                    break
        self.model = best_model
        if self.args.save_model:
            self.save_model()

    def eval(self):
        self.model.eval()

        total_labels = torch.empty(0, dtype=torch.long).to(self.device)
        total_logits = torch.empty((0, self.data.n_known_cls)).to(self.device)

        for batch in self.data.eval_dataloader:
            batch = tuple(t.to(self.device) for t in batch)
            input_ids, input_mask, segment_ids, label_ids = batch
            with torch.set_grad_enabled(False):
                _, logits = self.model(input_ids, segment_ids, input_mask, mode='eval')
                total_labels = torch.cat((total_labels, label_ids))
                total_logits = torch.cat((total_logits, logits))

        total_probs, total_preds = F.softmax(total_logits.detach(), dim=1).max(dim=1)
        y_pred = total_preds.cpu().numpy()
        y_true = total_labels.cpu().numpy()
        acc = round(accuracy_score(y_true, y_pred) * 100, 2)

        return acc

class BertForModel(nn.Module):
    def __init__(self, args, num_labels):
        super(BertForModel, self).__init__()
        self.num_labels = num_labels
        self.bert = BertModel.from_pretrained(args.bert_model)
        self.config = self.bert.config
        self.dense = nn.Linear(self.config.hidden_size, self.config.hidden_size)
        self.activation = nn.Tanh()
        self.dropout = nn.Dropout(self.config.hidden_dropout_prob)
        self.classifier = nn.Linear(self.config.hidden_size, num_labels)

    def forward(self, input_ids = None, token_type_ids = None, attention_mask=None , labels = None, mode = None, feature_ext = False):
        encoded_layer_12, pooled_output = self.bert(input_ids, attention_mask, token_type_ids,  return_dict=False)
        pooled_output = self.dense(encoded_layer_12.mean(dim = 1))
        pooled_output = self.activation(pooled_output)
        pooled_output = self.dropout(pooled_output)        
        logits = self.classifier(pooled_output)
        prob = F.softmax(logits, 1)
        _, pred = torch.max(prob, 1)

        if feature_ext:
            return pooled_output, pred, prob
        elif mode == 'train':
            loss = nn.CrossEntropyLoss()(logits, labels)
            return loss, prob
        else:
            return pooled_output, logits



