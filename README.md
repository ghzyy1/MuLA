# Graph Neural Networks with Multi-level Augmentation for Low-degree Graph

This repository contains the implementation of our method proposed in the paper  "Graph Neural Networks with Multi-level Augmentation for Low-degree Graph".
## Dependencies
- PyTorch >= 1.10.0
- torch-geometric >= 2.0.0
- tqdm
- numpy
-  CUDA >= 10.2.89
-  python >=  3.6.8

## Usage
To integrate the existing knowledge, please first execute the following script:
```
python run_aug_data.py
```
## Scripts
dataset-ACM
```
python run_results.py --gnn_type gcn --em_dim_1 16 --dropout 0.5 --lr 0.01 --weight_decay 0.0005 --gamma 0.7 --beta_d 0.5 --mixup_alpha 0.8  --num_concat 1 --tem 0.5 --alpha 1.0 --beta 1.0 --dataset_name ACM`

python run_results.py --gnn_type gat --em_dim_1 8 --dropout 0.6 --lr 0.005 --weight_decay 0.0005 --gamma 0.5 --beta_d 0.5 --mixup_alpha 0.6  --num_concat 1 --tem 0.5 --alpha 0.8 --beta 0.2 --dataset_name ACM`

python run_results.py --gnn_type appnp --em_dim_1 64 --dropout 0.5 --lr 0.01 --weight_decay 0.0005 --gamma 0.7 --beta_d 1.0 --mixup_alpha 0.6  --num_concat 2 --tem 0.5 --alpha 0.6 --beta 0.4 --dataset_name ACM`

python run_results.py --gnn_type sage --em_dim_1 256 --dropout 0.5 --lr 0.01 --weight_decay 0.0005 --gamma 0.7 --beta_d 1.5 --mixup_alpha 0.4  --num_concat 1 --tem 0.5 --alpha 0.5 --beta 0.5 --dataset_name ACM`
```
dataset-CiteSeer
```
python run_results.py --gnn_type gcn --em_dim_1 16 --dropout 0.5 --lr 0.01 --weight_decay 0.0005 --gamma 0.5 --beta_d 2.0 --mixup_alpha 0.6  --num_concat 1 --tem 0.2 --alpha 0.6 --beta 0.4 --dataset_name CiteSeer`

python run_results.py --gnn_type gat --em_dim_1 8 --dropout 0.6 --lr 0.005 --weight_decay 0.0005 --gamma 0.5 --beta_d 0.5 --mixup_alpha 0.6  --num_concat 1 --tem 0.5 --alpha 0.7 --beta 0.3 --dataset_name CiteSeer`

python run_results.py --gnn_type appnp --em_dim_1 64 --dropout 0.5 --lr 0.01 --weight_decay 0.0005 --gamma 0.5 --beta_d 0.5 --mixup_alpha 0.6  --num_concat 1 --tem 0.5 --alpha 0.6 --beta 0.4 --dataset_name CiteSeer`

python run_results.py --gnn_type sage --em_dim_1 256 --dropout 0.5 --lr 0.01 --weight_decay 0.0005 --gamma 0.5 --beta_d 1.5 --mixup_alpha 0.8  --num_concat 2 --tem 0.5 --alpha 0.2 --beta 0.8 --dataset_name CiteSeer`
```
dataset-Cora_ML
```
python run_results.py --gnn_type gcn --em_dim_1 16 --dropout 0.5 --lr 0.01 --weight_decay 0.0005 --gamma 0.5 --beta_d 1.5 --mixup_alpha 0.8  --num_concat 1 --tem 0.2 --alpha 1.0 --beta 1.0 --dataset_name Cora_ML`

python run_results.py --gnn_type gat --em_dim_1 8 --dropout 0.6 --lr 0.005 --weight_decay 0.0005 --gamma 0.5 --beta_d 0.5 --mixup_alpha 0.2  --num_concat 3 --tem 0.2 --alpha 1.0 --beta 1.0 --dataset_name Cora_ML`

python run_results.py --gnn_type appnp --em_dim_1 64 --dropout 0.5 --lr 0.01 --weight_decay 0.0005 --gamma 0.9 --beta_d 1.0 --mixup_alpha 0.2  --num_concat 1 --tem 0.5 --alpha 0.9 --beta 0.1 --dataset_name Cora_ML`

python run_results.py --gnn_type sage --em_dim_1 256 --dropout 0.5 --lr 0.01 --weight_decay 0.0005 --gamma 0.9 --beta_d 1.5 --mixup_alpha 0.6  --num_concat 4 --tem 0.5 --alpha 0.9 --beta 0.1 --dataset_name Cora_ML`
```

dataset-DBLP
```
python run_results.py --gnn_type gcn --em_dim_1 16 --dropout 0.5 --lr 0.01 --weight_decay 0.0005 --gamma 0.5 --beta_d 0.5 --mixup_alpha 0.2  --num_concat 3 --tem 0.2 --alpha 0.3 --beta 0.7 --dataset_name DBLP`

python run_results.py --gnn_type gat --em_dim_1 8 --dropout 0.6 --lr 0.005 --weight_decay 0.0005 --gamma 0.9 --beta_d 1.5 --mixup_alpha 0.4  --num_concat 4 --tem 0.5 --alpha 0.3 --beta 0.7 --dataset_name DBLP`

python run_results.py --gnn_type appnp --em_dim_1 64 --dropout 0.5 --lr 0.01 --weight_decay 0.0005 --gamma 0.5 --beta_d 1.5 --mixup_alpha 0.8  --num_concat 4 --tem 0.2 --alpha 0.6 --beta 0.4 --dataset_name DBLP`

python run_results.py --gnn_type sage --em_dim_1 256 --dropout 0.5 --lr 0.01 --weight_decay 0.0005 --gamma 0.5 --beta_d 1.5 --mixup_alpha 0.8  --num_concat 4 --tem 0.5 --alpha 0.8 --beta 0.2 --dataset_name DBLP`
```
