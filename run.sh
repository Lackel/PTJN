#!/usr/bin bash
python PTJN.py \
    --dataset banking \
    --known_cls_ratio 0.75 \
    --cluster_num_factor 1 \
    --seed 0 \
    --gpu_id 0 \
    --freeze_bert_parameters \
    --pretrain \
