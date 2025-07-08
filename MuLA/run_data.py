import gc
import sys
import copy
import torch
import os
# os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:32'
import random
import argparse
import numpy as np
from tqdm import tqdm
import torch.optim as optim
import torch.nn.functional as F
from tqdm import trange
from scipy.sparse import csr_matrix
from util.ops_io import get_pyg_dataset, normalize_features,softmax
from torch.utils.data import TensorDataset, DataLoader, RandomSampler
torch.nn.functional.softmax
from models.gcn import GCN
from models.cvae import VAE
from util.ops_cvae_pt import feature_tensor_normalize, loss_fn

exc_path = sys.path[0]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cuda', action='store_true', default=True, help='Disables CUDA training.')
    parser.add_argument('--cuda_device', type=str, default='0', help='The number of cuda device.')
    parser.add_argument('--seed', type=int, default=1009, help='Number of seed.')
    parser.add_argument('--gnn_type', type=str, default='gcn', help='The used basic model')
    parser.add_argument('--direction', type=str, default='/home/gh/MuLA/data', help='direction of datasets')
    parser.add_argument('--dataset_name', type=str, default='Cora_ML', help='The dataset used for training/testing')
                  
    parser.add_argument('--hidden_dim', type=int, default=16, help='Number of hidden units.')
    parser.add_argument('--dropout', type=float, default=0.5, help='Dropout rate (1 - keep probability).')
    parser.add_argument('--lr', type=float, default=0.01, help='Initial learning rate.')
    parser.add_argument('--weight_decay', type=float, default=0.0005, help='Weight decay (L2 loss on parameters).')
    parser.add_argument('--epochs', type=int, default=200, help='Number of epochs to train.')
    parser.add_argument("--batch_size", type=int, default=64)#改成512

    parser.add_argument("--latent_size", type=int, default=10)
    parser.add_argument("--conditional", action='store_true', default=True)
    parser.add_argument('--num_models', type=int, default=100, help='The number of models for choice')
    parser.add_argument('--warmup', type=int, default=200, help='Warmup')
    parser.add_argument('--update_epochs', type=int, default=20, help='Update training epochs')
    parser.add_argument("--pretrain_lr", type=float, default=0.01)
    parser.add_argument("--pretrain_epochs", type=int, default=10)
    args = parser.parse_args()

    device = torch.device("cuda:" + args.cuda_device if args.cuda else "cpu")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    # Load datatop
    dataset = get_pyg_dataset( args.direction, args.dataset_name, args.gnn_type)
    data = dataset[0]
    
    features = data.x.numpy()
    data.x = torch.Tensor(normalize_features(features))

    # Construct adj and adj_normalized
    edge_index = data.edge_index.numpy()
    values = np.ones(data.num_edges)
    adj = csr_matrix((values, (edge_index[0], edge_index[1])), shape=(data.num_nodes, data.num_nodes))
    features_tensor = torch.from_numpy(features).float()

    x_list, c_list = [], []
    for i in trange(adj.shape[0]):
        connected_indices = adj[i].nonzero()[1]
        x = features[adj[i].nonzero()[1]]
        c = np.tile(features[i], (x.shape[0], 1))
        x_list.append(x)
        c_list.append(c)

    features_x = np.vstack(x_list)
    features_c = np.vstack(c_list)
    del x_list
    del c_list
    gc.collect()
    features_x = torch.tensor(features_x, dtype=torch.float32)
    features_c = torch.tensor(features_c, dtype=torch.float32)

    cvae_features = torch.tensor(features, dtype=torch.float32)
    
    cvae_dataset = TensorDataset(features_x, features_c)
    cvae_dataset_sampler = RandomSampler(cvae_dataset)
    cvae_dataset_dataloader = DataLoader(cvae_dataset, sampler=cvae_dataset_sampler, batch_size=args.batch_size)

    model = GCN(input_dim=data.num_node_features, hidden_dim=args.hidden_dim, num_classes=dataset.num_classes,
                dropout=args.dropout)
    model_optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    cvae = VAE(encoder_layer_sizes=[features.shape[1], 256],
               latent_size=args.latent_size,
               decoder_layer_sizes=[256, features.shape[1]],
               conditional=args.conditional,
               conditional_size=features.shape[1])
    cvae_optimizer = optim.Adam(cvae.parameters(), lr=args.pretrain_lr)

    if args.cuda:
        model = model.to(device)
        data = data.to(device)
        cvae = cvae.to(device)
        cvae_features = cvae_features.to(device)

    for _ in range(int(args.epochs/2)):
        model.train()
        model_optimizer.zero_grad()
        output = model(data.x, data.edge_index)
        output = torch.log_softmax(output, dim=1)
        loss_train = F.nll_loss(output[data.train_mask], data.y[data.train_mask])
        loss_train.backward()
        model_optimizer.step()
    
    t = 0
    best_augmented_features = None
    cvae_model = None
    best_score = -float("inf")
    for _ in trange(args.pretrain_epochs, desc='Run CVAE Train'):
        for _, (x, c) in enumerate(tqdm(cvae_dataset_dataloader)):
            cvae.train()
            x, c = x.to(device), c.to(device)
            # x = min_max_normalize(x)
            # x = feature_tensor_normalize(x)
            if args.conditional:
                recon_x, mean, log_var, _ = cvae(x, c)
                
            else:
                recon_x, mean, log_var, _ = cvae(x)
            cvae_loss = loss_fn(recon_x, x, mean, log_var)

            cvae_optimizer.zero_grad()
            cvae_loss.backward()
            cvae_optimizer.step()
            z = torch.randn([cvae_features.size(0), args.latent_size]).to(device)
            if torch.isnan(z).any() or torch.isinf(z).any():
               print("z contains NaN or Inf values")
            augmented_features = cvae.inference(z, cvae_features)
            augmented_features = feature_tensor_normalize(augmented_features).detach()
            total_logits = 0
            cross_entropy = 0
            for i in range(args.num_models):
                logits = model(augmented_features, data.edge_index)
                total_logits += F.softmax(logits, dim=1)
                output = F.log_softmax(logits, dim=1)
                cross_entropy += F.nll_loss(output[data.train_mask], data.y[data.train_mask])
            output = torch.log(total_logits / args.num_models)
            U_score = F.nll_loss(output[data.train_mask], data.y[data.train_mask]) - cross_entropy / args.num_models
            t += 1
            print("U Score: ", U_score, " Best Score: ", best_score)
            if U_score > best_score:
                best_score = U_score
                if t > args.warmup:
                    cvae_model = copy.deepcopy(cvae)
                    print("U_score: ", U_score, " t: ", t)
                    best_augmented_features = copy.deepcopy(augmented_features)
                    for i in range(args.update_epochs):
                        model.train()
                        model_optimizer.zero_grad()
                        output = model(best_augmented_features, data.edge_index)
                        output = torch.log_softmax(output, dim=1)
                        loss_train = F.nll_loss(output[data.train_mask], data.y[data.train_mask])
                        loss_train.backward()
                        model_optimizer.step()
                        

    torch.save(cvae_model, "/home/gh/MuLA/CVAE_run_model/%s.pkl"%args.dataset_name)
    
