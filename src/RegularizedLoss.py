import torch
import torch_geometric.utils as utils



class RegularizedLoss:
    '''
    Used to calculate the regularized loss.

    Inputs:
        edge_index, edge index of the graph.
        train_mask, train indices that should be used for the training loss.

    '''
    
    def __init__(self, phi, mu, edge_index, train_mask):

        if phi == 'squared_error': 
            self.phi = lambda Zprime, Z : 0.5 * (torch.norm(Zprime - Z, p=2, dim=1) ** 2).sum()
        else:
            raise NotImplementedError()

        #Set the value of mu.
        self.mu = mu

        self.train_mask = train_mask
    
        #Calculate the adjacency matrix.
        A = utils.to_dense_adj(edge_index).squeeze(0)

        #Set N, the number of nodes in the graph.
        self.N, _ = A.shape

        #Calculate the degree matrix, https://github.com/pyg-team/pytorch_geometric/issues/1261#issuecomment-633913984
        D = torch.diag(utils.degree(edge_index[0])) 

        #Here we calculate the Ahat matrix basically two ways depending on whether or not we have isolated nodes.
        if utils.contains_isolated_nodes(edge_index):
            #In the case of isolated nodes, the degree of the node is 0 meaning D is singular. We get around this by replacing inf with 0.
            Dinv = 1./torch.diag(D)
            Dinv[Dinv == float('inf')] = 0
            Dinv = torch.diag(Dinv)
            self.Ahat = torch.matmul(Dinv, A)
        else:
            #If there are no isolated nodes, then all nodes have a degree larger than zero meaning that we can calculate the inverse of D. 
            self.Ahat = torch.linalg.solve(D, A)

        #Use the CrossEntropyLoss. 
        self.cross_entropy = torch.nn.CrossEntropyLoss()

    def __call__(self, Z, y):
        '''
        Evaluate the loss function. 
        '''
        N = len(y)
        
        L1 = self.cross_entropy(Z[self.train_mask], y[self.train_mask])

        Zprime = torch.matmul(self.Ahat, Z)

        #Equation (2) in the paper.
        L2 = self.mu * 1/self.N * self.phi(Zprime, Z)

        return L1 + self.mu * L2