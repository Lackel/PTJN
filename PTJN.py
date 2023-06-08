import os
import torch
from init_parameter import init_model
from data import Data
from model import PretrainModelManager, BertForModel
from util import clustering_score
from transformers import get_linear_schedule_with_warmup
from transformers import AdamW, logging, WEIGHTS_NAME
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import (DataLoader, SequentialSampler, TensorDataset)
import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans


class ModelManager:
    def __init__(self, args, data, pretrained_model=None):
        if pretrained_model is None:
            pretrained_model = BertForModel.from_pretrained(args.bert_model, num_labels = data.n_known_cls)
            if os.path.exists(args.pretrain_dir):
                pretrained_model = self.restore_model(args.pretrained_model)
        self.pretrained_model = pretrained_model           
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.num_labels = data.num_labels       
        self.centroids = None

        self.extractor = BertForModel(args, self.num_labels)
        self.corrector = BertForModel(args, self.num_labels)
        if args.pretrain:
            self.load_pretrained_model()
        if args.freeze_bert_parameters:
            self.freeze_parameters(self.extractor)
            self.freeze_parameters(self.corrector)
        self.extractor.to(self.device)
        self.corrector.to(self.device)

        self.optimizer = self.get_optimizer(args)
        num_train_examples = len(data.train_labeled_examples) + len(data.train_unlabeled_examples)
        self.num_training_steps = int(num_train_examples / args.train_batch_size) * 100
        self.num_warmup_steps= int(args.warmup_proportion * self.num_training_steps) 
        self.scheduler = get_linear_schedule_with_warmup(optimizer=self.optimizer, num_warmup_steps=self.num_warmup_steps, num_training_steps=self.num_training_steps) 

    def get_features_labels(self, dataloader, model, args, mode=False):      
        model.eval()
        total_features = torch.empty((0,args.feat_dim)).to(self.device)
        total_labels = torch.empty(0,dtype=torch.long).to(self.device)
        total_pred = torch.empty(0,dtype=torch.long).to(self.device)
        total_soft = torch.empty((0, self.num_labels)).to(self.device)
        for _, batch in enumerate(dataloader):
            batch = tuple(t.to(self.device) for t in batch)
            input_ids, input_mask, segment_ids, label_ids = batch
            with torch.no_grad():
                feature, label, logits_softmax = model(input_ids, segment_ids, input_mask, feature_ext = True)
            total_features = torch.cat((total_features, feature))
            total_labels = torch.cat((total_labels, label_ids))
            total_pred = torch.cat((total_pred, label))
            total_soft = torch.cat((total_soft, logits_softmax))
        if mode:
            return total_features, total_labels, total_soft
        return total_features, total_labels, total_pred
    
    def get_optimizer(self, args):
        param_optimizer = list(self.extractor.named_parameters())
        param_optimizer.extend(list(self.corrector.named_parameters()))
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=args.lr, correct_bias=False, no_deprecation_warning=True)
        return optimizer
 
    def evaluation(self, args, data):
        feats, labels, pred_e = self.get_features_labels(data.test_dataloader, self.extractor, args, mode=True)
        _, _, pred_c = self.get_features_labels(data.test_dataloader, self.corrector, args, mode=True)
        feats = feats.cpu().numpy()
        km = KMeans(n_clusters = self.num_labels, n_init=30).fit(feats)
        y_true = labels.cpu().numpy()

        align_labels = self.alignment(km, args)
        y_g = np.zeros((len(feats), self.num_labels), dtype=int)
        lambda_i = torch.zeros(len(feats), 1).to(self.device)
        
        for i in range(len(align_labels)):
            y_g[i][align_labels[i]] = 1
            lambda_i[i] = (F.kl_div(pred_e[i].log(), (pred_e[i] + pred_c[i])/2, reduction='batchmean') + F.kl_div(pred_c[i].log(), (pred_e[i] + pred_c[i])/2, reduction='batchmean'))/2
        y_g = torch.tensor(y_g).to(self.device)
            
        vote_labels = y_g + args.lambda_v * (1 - lambda_i) * (pred_e + pred_c)
        y_pred = torch.softmax(vote_labels, 1)
        _, y_pred = torch.max(y_pred, 1)
        y_pred = y_pred.cpu().numpy()
        results = clustering_score(y_true, y_pred)
        print(results)
        
    def alignment(self, km, args):
        if self.centroids is not None:
            old_centroids = self.centroids.cpu().numpy()
            new_centroids = km.cluster_centers_
            DistanceMatrix = np.linalg.norm(old_centroids[:,np.newaxis,:]-new_centroids[np.newaxis,:,:],axis=2) 
            _, col_ind = linear_sum_assignment(DistanceMatrix)
            new_centroids = torch.tensor(new_centroids).to(self.device)
            self.centroids = torch.empty(self.num_labels ,args.feat_dim).to(self.device)
            alignment_labels = list(col_ind)

            for i in range(self.num_labels):
                label = alignment_labels[i]
                self.centroids[i] = new_centroids[label]

            pseudo2label = {label:i for i,label in enumerate(alignment_labels)}
            pseudo_labels = np.array([pseudo2label[label] for label in km.labels_])
        else:
            self.centroids = torch.tensor(km.cluster_centers_).to(self.device)        
            pseudo_labels = km.labels_ 

        pseudo_labels = torch.tensor(pseudo_labels, dtype=torch.long).to(self.device)

        return pseudo_labels

    def update_pseudo_labels(self, pseudo_labels, args, data):
        train_data = TensorDataset(data.semi_input_ids, data.semi_input_mask, data.semi_segment_ids, pseudo_labels)
        train_sampler = SequentialSampler(train_data)
        train_dataloader = DataLoader(train_data, sampler = train_sampler, batch_size = args.train_batch_size)
        return train_dataloader

    def train(self, args, data):
        labelediter = iter(data.train_labeled_dataloader)
        
        for epoch in range(1, int(args.num_train_epochs) + 1):  
            feats, _, _ = self.get_features_labels(data.train_semi_dataloader, self.extractor, args)
            feats = feats.cpu().numpy()
            km = KMeans(n_clusters = self.num_labels).fit(feats)
            print('KMeans finished')
            pseudo_labels = self.alignment(km, args)
            train_dataloader = self.update_pseudo_labels(pseudo_labels, args, data)
            tr_loss = 0
            nb_tr_steps = 0
            self.extractor.train()
            self.corrector.train()

            for _, batch in  enumerate(train_dataloader):
                batch = tuple(t.to(self.device) for t in batch)
                input_ids, input_mask, segment_ids, label_ids = batch
                loss_h, logits_e = self.extractor(input_ids, segment_ids, input_mask, label_ids, mode='train')
                _, pred_e = torch.max(logits_e.data, 1)
                loss_c, logits_c = self.corrector(input_ids, segment_ids, input_mask, pred_e, mode='train')
                _, pred_c = torch.max(logits_c.data, 1)
                loss_e, _ = self.extractor(input_ids, segment_ids, input_mask, pred_c, mode='train')
                loss_pseudo = 1/3 * loss_e +  1/3 * loss_c +  1/3 * loss_h
    
                try:
                    batch = labelediter.next()
                except StopIteration:
                    labelediter = iter(data.train_labeled_dataloader)
                    batch = labelediter.next()
                batch = tuple(t.to(self.device) for t in batch)
                input_ids, input_mask, segment_ids, label_ids = batch
                loss_je, _ = self.extractor(input_ids, segment_ids, input_mask, label_ids, mode="train")
                loss_jc, _ = self.corrector(input_ids, segment_ids, input_mask, label_ids, mode="train")
                loss_labeled = 0.5 * loss_je + 0.5 * loss_jc
                
                loss = args.lambda_t * loss_pseudo + loss_labeled

                nb_tr_steps += 1
                tr_loss += loss.item()
                loss.backward()
                nn.utils.clip_grad_norm_(self.extractor.parameters(), 1.0)
                nn.utils.clip_grad_norm_(self.corrector.parameters(), 1.0)
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()
            print('Epoch ' + str(epoch) + ' loss: ' + str(tr_loss/nb_tr_steps))

    def load_pretrained_model(self):
        pretrained_dict = self.pretrained_model.state_dict()
        classifier_params = ['classifier.weight','classifier.bias']
        pretrained_dict =  {k: v for k, v in pretrained_dict.items() if k not in classifier_params}
        self.extractor.load_state_dict(pretrained_dict, strict=False)
        self.corrector.load_state_dict(pretrained_dict, strict=False)
        
    def restore_model(self, args, model):
        output_model_file = os.path.join(args.pretrain_dir, WEIGHTS_NAME)
        model.load_state_dict(torch.load(output_model_file))
        return model
    
    def freeze_parameters(self,model):
        for name, param in model.bert.named_parameters():  
            param.requires_grad = False
            if "encoder.layer.11" in name or "pooler" in name:
                param.requires_grad = True

if __name__ == '__main__':
    
    logging.set_verbosity_error()
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    print('Data and Parameters Initialization...')
    parser = init_model()
    args = parser.parse_args()
    data = Data(args)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    
    if args.pretrain:
        print('Pre-training begin...')
        manager_p = PretrainModelManager(args, data)
        manager_p.train()
        print('Pre-training finished!')
        manager = ModelManager(args, data, manager_p.model)
    else:
        manager_p = PretrainModelManager(args, data)
        manager = ModelManager(args, data, manager_p.model)
    
    print('Training begin...')
    manager.train(args,data)
    print('Training finished!')

    print('Evaluation begin...')
    manager.evaluation(args, data)
    print('Evaluation finished!')

