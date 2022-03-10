import argparse

import torch
from torch_geometric.data import Data
from torch_geometric.datasets import Planetoid
from sklearn.metrics import mean_squared_error, roc_auc_score, accuracy_score
import wandb

from GCN import GCN
from RegularizedLoss import RegularizedLoss
from get_masks import get_masks

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--dataset', type=str, choices=['Cora', 'CiteSeer', 'PubMed'])
    parser.add_argument('--lr', default=0.01, type=float)
    parser.add_argument('--weight-decay', default=5e-4, type=float)
    parser.add_argument('--epochs', default=200, type=int)
    
    #Specify A and B arguments for the split values.
    parser.add_argument('--A', default=None, type=int)
    parser.add_argument('--B', default=None, type=int)

    args = parser.parse_args()

    assert (args.A is None and args.B is None) or (args.A is not None and args.B is not None), 'A and B should be either given or not given'
    #Fetch the dataset.
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

    if args.A is not None:
        #Then args.B is not none either.
        train_mask, val_mask, test_mask = get_masks(args.A, args.B, dataset, args.seed)
    else:
        train_mask, val_mask, test_mask = data.train_mask, data.val_mask, data.test_mask
    
    torch.manual_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = GCN(num_node_features = dataset.num_node_features, num_classes = dataset.num_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)    

    loss_fn = RegularizedLoss(phi = 'squared_error', mu = 0, edge_index = data.edge_index, train_mask = data.train_mask)
    model.train()
    for epoch in range(args.epochs):
        optimizer.zero_grad()
        out = model(data)
        
        loss = loss_fn(out, data.y)
        
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        model.eval()
        out = model(data)
        pred = out.argmax(dim=1)
        score = torch.softmax(out, dim=1)

        #RMSE is calculated as seen here: https://stackoverflow.com/a/18623635/17389949
        train_rms = mean_squared_error(y_true = data.y[train_mask], y_pred = pred[train_mask], squared=False)
        train_roc_auc_score = roc_auc_score(y_true = data.y[train_mask], y_score = score[train_mask, :], multi_class='ovr')
        train_acc = accuracy_score(y_true = data.y[train_mask], y_pred = pred[train_mask])

        val_rms = mean_squared_error(y_true = data.y[val_mask], y_pred = pred[val_mask], squared=False)
        val_roc_auc_score = roc_auc_score(y_true = data.y[val_mask], y_score = score[val_mask, :], multi_class='ovr')
        val_acc = accuracy_score(y_true = data.y[val_mask], y_pred = pred[val_mask])

        test_rms = mean_squared_error(y_true = data.y[test_mask], y_pred = pred[test_mask], squared=False)
        test_roc_auc_score = roc_auc_score(y_true = data.y[test_mask], y_score = score[test_mask, :], multi_class='ovr')
        test_acc = accuracy_score(y_true = data.y[test_mask], y_pred = pred[test_mask])

    #TODO: Log it right here!
    wandb.log({"dataset": args.dataset,
                "learning rate": args.lr,
                "weight-decay": args.weight_decay,
                "epochs": args.epochs,
                "seed": args.seed,
                "A": args.A,
                "B": args.B,
                "train_rms": train_rms,
                "train_roc_auc_score": train_roc_auc_score,
                "train_acc": train_acc,
                "val_rms": val_rms,
                "val_roc_auc_score": val_roc_auc_score,
                "val_acc": val_acc,
                "test_rms": test_rms,
                "test_roc_auc_score": test_roc_auc_score,
                "test_acc": test_acc})

if __name__ == '__main__':
    wandb.init(project="02460AdvancedML", entity="rasgaard")
    main()
