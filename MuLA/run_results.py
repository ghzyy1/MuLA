import copy
import time
import torch
import numpy as np
import random
import os
from torch.optim import Adam
from torch_geometric.nn.conv.gcn_conv import gcn_norm
import argparse
from models.node_mixup import Mixup
from util.ops_io import get_pyg_dataset
from util.ops_ev import get_evaluation_results
from util.ops_al import Mixer, Cross_entropy_loss, sharpen
from util.ops_loss import consis_loss
from util.ops_io import normalize_features, get_augmented_features
import torch.nn.functional as F

def validate(args,features,model, data, mixup_dict):
    model.eval()
    device = torch.device("cuda:" + args.cuda_device if args.cuda else "cpu")
    val_x_list = get_augmented_features(args.load_saved, args.dataset_name, features, device, args.num_concat)
    logits_list = []
    outputgh = val_x_list+[data.x]
    for Tensor in outputgh:
        logits, _, _ = model(Tensor, data.edge_index, mixup_dict)
        logits_list.append(logits)
    stacked_tensors = torch.stack(logits_list)
    average_tensor = torch.mean(stacked_tensors, dim=0)
    pred = average_tensor[data.val_mask].max(1)[1].cpu().numpy()
    gold = data.y[data.val_mask].cpu().numpy()
    ACC, MaP, MaR, MaF, MiF = get_evaluation_results(gold, pred)
    return ACC, MaP, MaR, MaF, MiF

def test(args,features, model, data, mixup_dict):
    model.eval()
    device = torch.device("cuda:" + args.cuda_device if args.cuda else "cpu")
    val_x_list = get_augmented_features(args.load_saved, args.dataset_name, features, device, args.num_concat)
    logits_list = []
    outputgh = val_x_list+[data.x]
    for Tensor in outputgh:
        logits, _, _ = model(Tensor, data.edge_index, mixup_dict)
        logits_list.append(logits)
    stacked_tensors = torch.stack(logits_list)
    average_tensor = torch.mean(stacked_tensors, dim=0)
    pred = average_tensor[data.val_mask].max(1)[1].cpu().numpy()
    gold = data.y[data.val_mask].cpu().numpy()
    ACC, MaP, MaR, MaF, MiF = get_evaluation_results(gold, pred)
    return ACC, MaP, MaR, MaF, MiF

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cuda', action='store_true', default=True, help='Disables CUDA training.')
    parser.add_argument('--cuda_device', type=str, default='2', help='The number of cuda device.')
    parser.add_argument('--num_repeat', type=int, default=1, help='Number of repeated experiments')
    parser.add_argument('--seed', type=int, default=1009, help='Number of seed.')
    parser.add_argument('--direction', type=str, default='/home/gh/data/', help='direction of datasets')
    parser.add_argument('--dataset_name', type=str, default='Cora_ML', help='The dataset used for training/testing')
    parser.add_argument('--gnn_type', type=str, default='gcn', help='The used basic model')
    # common parameters of mode
    parser.add_argument("--num_concat", type=int, default=1)
    parser.add_argument("--samples", type=int, default=4)
    parser.add_argument('--epochs', type=int, default=2000, help='Number of epochs to train.')
    parser.add_argument('--patience', type=int, default=10000, help='Patience')
    parser.add_argument('--save_patience', type=int, default=10,
                        help='the frequency about saving the training result in the finetune stage')
    parser.add_argument('--em_dim_1', type=int, default=16, help='Dimension of the first encoder layer.')
    parser.add_argument('--lr', type=float, default=0.01, help='Initial learning rate.')
    parser.add_argument('--weight_decay', type=float, default=0.0005, help='Weight decay (L2 loss on parameters).')
    parser.add_argument('--dropout', type=float, default=0.5, help='Dropout rate (1 - keep probability).')
    # Params: ChebNet
    parser.add_argument('--cheb_hop', type=int, default=3, help='Number of hops')
    # Params: GraphSAGE
    parser.add_argument('--sage_num_layers', type=int, default=2, help='Number of hops')
    # Params: GAT
    parser.add_argument('--gat_head_1', type=int, default=8, help='Number of attention heads of 1st layer.')
    parser.add_argument('--gat_head_2', type=int, default=1, help='Number of attention heads of 2nd layer.')
    # Params: APPNP
    parser.add_argument('--appnp_K', type=int, default=10, help='Number of propagation layers.')
    parser.add_argument('--appnp_alpha', type=float, default=0.1, help='the value of alpha.')
    # Params: SGC
    parser.add_argument('--sgc_K', type=int, default=2, help='Number of graph convolution layers')
    # Params: GCN2
    parser.add_argument('--gcn2_num_layers', type=int, default=64, help='Number of layers')
    parser.add_argument('--gcn2_alpha', type=float, default=0.1, help='Value of alpha')
    parser.add_argument('--gcn2_theta', type=float, default=0.5, help='Value of theta')
    # specific for NodeMixup
    parser.add_argument('--gamma', type=float, default=0.7, help="threshold for pseudo labeling")
    parser.add_argument('--beta_s', type=float, default=1.5,
                        help="tuning strength of NLP similarity in NLD-aware sampling")
    parser.add_argument('--beta_d', type=float, default=0.5, help="tuning strength of node degree in NLD-aware sampling")
    parser.add_argument('--temp', type=float, default=0.1, help="sharpness of distribution")
    parser.add_argument('--tem', type=float, default=0.5, help='Sharpening temperature')
    parser.add_argument('--lam', type=float, default=1.0, help='Lamda')
    parser.add_argument('--alpha', type=float, default=0.9, help='alpha')
    parser.add_argument('--beta', type=float, default=0.1, help='beta')
    parser.add_argument('--gh', type=float, default=0.1, help='gh')
    parser.add_argument('--mixup_alpha', type=float, default=0.8, help="determing the Beta distribution")
    parser.add_argument('--lam_intra', type=float, default=1.0, help="balance hyperparameter of intra-class mixup loss")
    parser.add_argument('--lam_inter', type=float, default=1.0, help="balance hyperparameter of inter-class mixup loss")
    args = parser.parse_args()
    device = torch.device("cuda:" + args.cuda_device if args.cuda else "cpu")
    random.seed(args.seed)
    os.environ['PYTHONHASHSEED'] = str(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = False
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    torch.use_deterministic_algorithms(True)

    all_ACC = []
    all_MaP = []
    all_MaR = []
    all_MaF = []
    all_MiF = []
    all_Time = []
    logits_list = []

    for i in range(args.num_repeat):
        print("################ Current repeat: ", i + 1, "  ################")
        if i == 0:
            # args.load_saved = False
            args.load_saved = True
        else:
            args.load_saved = True
        # Load data
        dataset = get_pyg_dataset(direction=args.direction, dataset_name=args.dataset_name, gnn_type=args.gnn_type)
        args.input_dim = dataset.num_features
        args.num_classes = dataset.num_classes
        data = dataset[0]
        features = data.x.numpy()
        data.x = torch.Tensor(normalize_features(features))
        labels = data.y

        # create the model and optimizer
        model = Mixup(args)
        optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        
        if args.cuda:
            model = model.to(device)
            data = data.to(device)
            labels = labels.to(device)
        mixup_dict = None
        use_mixup = args.lam_intra != 0 or args.lam_inter != 0
        if use_mixup:
            t_idx = torch.nonzero(data.train_mask).squeeze(-1).to(device)  # [0, 1, ..., 139]
            t_labels = data.y[data.train_mask].unsqueeze(1).to(device)  # shape[140, 1]
            t_y = torch.zeros(t_idx.shape[0], args.num_classes, device=device).scatter_(1, t_labels, 1).to(device)

            unlabeled_mask = (1 - data.train_mask.float()).bool()
            un_idx = torch.nonzero(unlabeled_mask).squeeze(-1).to(device)
            mixup_y = torch.zeros(labels.shape[0], args.num_classes).to(device)
            mixup_y[un_idx] = 1. / args.num_classes

            mixer = Mixer(t_idx, un_idx, beta_d=args.beta_d, beta_s=args.beta_s, temp=args.temp,
                          train_size=int(t_idx.shape[0] / args.num_classes), nclass=args.num_classes,
                          alpha=args.mixup_alpha, gamma=args.gamma, device=device)

            mixup_dict = dict()
            mixup_dict['t_idx'], mixup_dict['t_y'], mixup_dict['un_idx'], mixup_dict['all_idx'], mixup_dict['mixer'] = \
                t_idx, t_y, un_idx, torch.cat([t_idx, un_idx]), mixer
            mixup_dict['lam_intra'], mixup_dict['lam_inter'] = args.lam_intra, args.lam_inter

        # the information which need to be recorded
        start_time = time.time()
        best_valid_f1 = 0.0
        least_loss = float("inf")
        bad_counter = 0
        best_epoch = 0
        best_model = None
        # Graph Representation Augmentation beging training
        for epoch in range(args.epochs):
            if use_mixup:
                mixup_y[t_idx] = t_y
                mixup_dict['eq_mixup_x'], mixup_dict['eq_mixup_y'], mixup_dict['neq_mixup_x'], mixup_dict[
                    'neq_mixup_y'], \
                mixup_dict['mixup_adj'], mixup_dict['E'], mixup_dict['eq_idx'] = \
                    mixer.mixup_data( data.x, mixup_y, data.edge_index)
            
            model.train()
            optimizer.zero_grad()
            #Knowledge Integration
            val_x_list = get_augmented_features(args.load_saved, args.dataset_name, features, device, args.num_concat)
            outputgh = val_x_list+[data.x]
            output_list = []
            
            loss_2 = 0
    
            for Tensor in outputgh:
               
                logits, eq_mixup_logits, neq_mixup_logits = model(Tensor, data.edge_index, mixup_dict)
                logits_list.append(logits)
   
                loss_2 += Cross_entropy_loss(logits, labels, data.train_mask)
                if use_mixup:
                    mixer = mixup_dict['mixer']
                    eq_mixup_loss, neq_mixup_loss = mixer.mixup_loss(eq_mixup_logits, neq_mixup_logits, mixup_dict)
                    mixup_loss = mixup_dict['lam_intra'] * eq_mixup_loss + mixup_dict['lam_inter'] * neq_mixup_loss

            loss_2 = loss_2 / len(outputgh)
            mixup_loss = mixup_loss / len(outputgh)

            #Knowledge Integration consis_loss   
            for Tensor in outputgh:
                output111,_,_ = model(Tensor, data.edge_index)
                output111_log_softmax = torch.log_softmax(output111, dim=-1)
            output_list.append(output111_log_softmax)

            loss_consis = consis_loss(output_list, args.tem, args.lam)
    
            loss_2 = 0.

            if use_mixup:
                y = logits.softmax(-1).detach()
                y_val, _ = torch.max(y, 1)
                mask = y_val >= args.gamma
                # mask = mask.cpu()
                if mask[unlabeled_mask].sum() > 0:
                    target_y = y[mask * unlabeled_mask]
                    mixup_y[mask * unlabeled_mask] = sharpen(target_y, 0.5)
        
            loss_fin = loss_2 + args.alpha * loss_consis + args.beta * mixup_loss
            current_loss = loss_fin.item()
            loss_fin.backward()
            optimizer.step() 
            _, _, _, _, f1 = validate(args,features,model, data, mixup_dict)
            if best_valid_f1 < f1:
                best_epoch = epoch + 1
                best_valid_f1 = f1
                best_model = copy.deepcopy(model)
            if current_loss < least_loss:
                least_loss = current_loss
                bad_counter = 0
                print('Obtain best loss at Epoch: {:04d}'.format(epoch + 1), 'f1: {:.4f}'.format(f1), 'loss: {:.4f}'.format(current_loss))
            else:
                bad_counter += 1


        print("Optimization Finished!")
        used_time = time.time() - start_time
        print("Best epochs: {:2d}".format(best_epoch))
        print("Best epoch's validate f1: {:.2f}".format(best_valid_f1 * 100))
        print("Total time elapsed: {:.2f}s".format(used_time))

        # test the best trained model
        ACC, MaP, MaR, MaF, MiF = test(args,features,best_model, data, mixup_dict)
        print("Test MaF: ", MaF)
        all_ACC.append(ACC)
        all_MaP.append(MaP)
        all_MaR.append(MaR)
        all_MaF.append(MaF)
        all_MiF.append(MiF)
        all_Time.append(used_time)

    fp = open("results_" + args.gnn_type + '_' + args.dataset_name + ".txt", "a+", encoding="utf-8")
    fp.write(args.dataset_name)
    fp.write("ACC: {:.2f}\t{:.2f}\n".format(np.mean(all_ACC) * 100, np.std(all_ACC) * 100))
    fp.write("MaP: {:.2f}\t{:.2f}\n".format(np.mean(all_MaP) * 100, np.std(all_MaP) * 100))
    fp.write("MaR: {:.2f}\t{:.2f}\n".format(np.mean(all_MaR) * 100, np.std(all_MaR) * 100))
    fp.write("MaF: {:.2f}\t{:.2f}\n".format(np.mean(all_MaF) * 100, np.std(all_MaF) * 100))
    fp.write("Time: {:.2f}\t{:.2f}\n\n".format(np.mean(all_Time), np.std(all_Time)))
    fp.close()


    
