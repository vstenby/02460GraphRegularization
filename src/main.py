import argparse

import torch
from torch_geometric.data import Data
from torch_geometric.datasets import Planetoid
from sklearn.metrics import mean_squared_error, roc_auc_score, accuracy_score
import wandb

#Import different models.
from GCN import GCN
from GAT import GAT
from GATv2 import GATv2

from get_masks import get_masks
from PRegLoss import PRegLoss
from conf_penalty import conf_penalty
from LapLoss import LapLoss
from label_smoothing import label_smoothing

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--dataset', type=str, choices=['Cora', 'CiteSeer', 'PubMed'])
    parser.add_argument('--lr', default=0.01, type=float)
    parser.add_argument('--weight-decay', default=5e-4, type=float)
    parser.add_argument('--epochs', default=200, type=int, help='Number of epochs to train.')
    parser.add_argument('--phi', default='cross_entropy', type=str, choices=['cross_entropy', 'squared_error', 'KL_divergence'])
    parser.add_argument('--mu', default=0.0, type=float, help='regularisation weight for the p-reg loss')
    parser.add_argument('--sweep', default=0, type=int, help='whether or not to do a WandB sweep.')
    parser.add_argument('--beta', default=0, type=float, help='conf penalty parameter.')
    parser.add_argument('--tau', default=0, type=float, help='p-reg thresholding')
    parser.add_argument('--unmask-alpha', default=1, type=float, help='value of alpha for the unmasking. 1 means p-reg is applied to all nodes, 0 means p-reg is applied to no nodes.')
    parser.add_argument('--unmask-random-nodes-every-call', default=1, type=int, choices=[0, 1])
    parser.add_argument('--kappa', default=0, type=float, help='Laplacian reg weight')
    parser.add_argument('--epsilon', default=0, type=float, help='Label smoothing parameter')
    parser.add_argument('--model', type=str, default='GCN', choices=['GCN', 'GAT', 'GATv2'])
    parser.add_argument('--num_hidden_features', default=16, type=int)
    parser.add_argument('--early-stopping', default=0, type=int, choices=[0,1], help='whether or not to do early stopping')

    #Specify A and B arguments for the split values.
    parser.add_argument('--A', default=None, type=int)
    parser.add_argument('--B', default=None, type=int)

    args = parser.parse_args()

    if args.sweep:
        wandb.init(project="02460AdvancedML", entity="rasgaard")

    assert (args.A is None and args.B is None) or (args.A is not None and args.B is not None), 'A and B should be either given or not given'

    assert args.dataset is not None, 'Dataset should not be None.'

    if args.dataset.lower() == 'cora':  
        dataset = Planetoid(root=f'/tmp/Cora', name='Cora')
    elif args.dataset.lower() == 'citeseer':
        dataset = Planetoid(root=f'/tmp/CiteSeer', name='CiteSeer')
    elif args.dataset.lower() == 'pubmed':
        dataset = Planetoid(root=f'/tmp/PubMed', name='PubMed') 
    else:
        raise NotImplementedError('Invalid dataset.')

    #Unpack the dataset to get the data.
    data = dataset[0]
    C = data.y.unique().size()[0]

    if args.A is not None:
        #Then args.B is not none either.
        train_mask, val_mask, test_mask = get_masks(args.A, args.B, dataset, args.seed)
    else:
        train_mask, val_mask, test_mask = data.train_mask, data.val_mask, data.test_mask

    #Create the unmask dictionary. 
    unmask_dict = {'alpha' : args.unmask_alpha, 'random_nodes_every_call' : args.unmask_random_nodes_every_call, 'seed' : args.seed}
    
    torch.manual_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Running on {device}.')

    if args.model == 'GCN':
        model = GCN(num_node_features = dataset.num_node_features, num_classes = dataset.num_classes, num_hidden_features = args.num_hidden_features).to(device)
    elif args.model == 'GAT':
        model = GAT(num_node_features = dataset.num_node_features, num_classes = dataset.num_classes).to(device)
    elif args.model == 'GATv2':
        model = GATv2(num_node_features = dataset.num_node_features, num_classes = dataset.num_classes).to(device)
    else:
        raise NotImplementedError('Invalid model.')

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)    

    loss_fn = torch.nn.CrossEntropyLoss()
    preg_loss_fn = PRegLoss(phi = args.phi, edge_index = data.edge_index, unmask_dict=unmask_dict, device=device)
    lap_loss_fn  = LapLoss(edge_index = data.edge_index, device=device)
    model.train()

    if args.early_stopping:
        if val_mask.sum() == 0:
            raise ValueError('No validation set for early stopping.')

        #Set the early stopping validation accuracy.
        val_acc_prev = 0.0
        model_prev = model

    for epoch in range(args.epochs):
        optimizer.zero_grad()
        out = model(data.to(device)) 

        #Calculate the loss, which is the CrossEntropy for the training, \mu and the P-reg loss as well as the confidence penalty term.
        #torch.maximum for tau > 0, C.2: Thresholding of P-reg
        loss = loss_fn(out[train_mask], label_smoothing(data.y[train_mask], C, args.epsilon)) \
             + args.mu * torch.maximum(torch.tensor([0]).to(device), preg_loss_fn(out) - args.tau)\
             + args.kappa * lap_loss_fn(out)\
             + args.beta  * conf_penalty(out)

        #Backpropagate 
        loss.backward()

        #Take a step with the optimizer.
        optimizer.step()

        #Check for early stopping every 200 epochs.
        if ((epoch+1) % 200) == 0:
            #Check for early stopping.
            if args.early_stopping:
                model.eval()
                out = model(data)
                pred = out.argmax(dim=1)
                score = torch.softmax(out, dim=1)

                #Check the validation accuracy.
                val_acc = accuracy_score(y_true = data.y[val_mask].cpu(), y_pred = pred[val_mask].cpu())

                #If the validation accuracy is better than the previous one, then continue training.
                if val_acc > val_acc_prev:
                    val_acc_prev = val_acc
                    model_prev   = model
                    model.train()
                else:
                    #Otherwise, stop training.
                    model = model_prev
                    break

    with torch.no_grad():
        model.eval()
        out = model(data)
        pred = out.argmax(dim=1)
        score = torch.softmax(out, dim=1)

        #RMSE is calculated as seen here: https://stackoverflow.com/a/18623635/17389949
        train_rms = mean_squared_error(y_true = data.y[train_mask].cpu(), y_pred = pred[train_mask].cpu(), squared=False)
        train_roc_auc_score = roc_auc_score(y_true = data.y[train_mask].cpu(), y_score = score[train_mask, :].cpu(), multi_class='ovr')
        train_acc = accuracy_score(y_true = data.y[train_mask].cpu(), y_pred = pred[train_mask].cpu())

        if args.B != 0:
            val_rms = mean_squared_error(y_true = data.y[val_mask].cpu(), y_pred = pred[val_mask].cpu(), squared=False)
            val_roc_auc_score = roc_auc_score(y_true = data.y[val_mask].cpu(), y_score = score[val_mask, :].cpu(), multi_class='ovr')
            val_acc = accuracy_score(y_true = data.y[val_mask].cpu(), y_pred = pred[val_mask].cpu())
        else:
            val_rms           = None
            val_roc_auc_score = None
            val_acc           = None

        test_rms = mean_squared_error(y_true = data.y[test_mask].cpu(), y_pred = pred[test_mask].cpu(), squared=False)
        test_roc_auc_score = roc_auc_score(y_true = data.y[test_mask].cpu(), y_score = score[test_mask, :].cpu(), multi_class='ovr')
        test_acc = accuracy_score(y_true = data.y[test_mask].cpu(), y_pred = pred[test_mask].cpu())

    #Log it right here!
    if args.sweep:
        wandb.log({ "train_rms": train_rms,
                    "train_roc_auc_score": train_roc_auc_score,
                    "train_acc": train_acc,
                    "val_rms": val_rms,
                    "val_roc_auc_score": val_roc_auc_score,
                    "val_acc": val_acc,
                    "test_rms": test_rms,
                    "test_roc_auc_score": test_roc_auc_score,
                    "test_acc": test_acc})
    else:
        print(f'Test RMSE: {test_rms} | Test ROC-AUC-Score: {test_roc_auc_score} | Test Acc: {test_acc}')

if __name__ == '__main__':
    main()