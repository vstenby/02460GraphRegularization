import torch
import torch_geometric.utils as utils

class PRegLoss:
    '''
    Calculate the P-Reg loss as defined in "Rethinking Graph Regularisation for Graph Neural Networks"
    '''

    def __init__(self, phi, edge_index):
        '''
        Input:
            phi, either 'squared_error', 'cross_entropy' or 'KL_divergence'
            edge_index, used to construct the adjacency matrix and the diagonal degree matrix.
        '''

        self.not_isolated = utils.degree(edge_index[0]) > 0

        #Calculate the adjacency matrix.
        A = utils.to_dense_adj(edge_index).squeeze(0)
        A = A[self.not_isolated, :][:, self.not_isolated]

        #Not sure whether or not N is the number of nodes,
        #or the number of isolated nodes. Probably doesn't matter.
        self.N = len(self.not_isolated)

        #Calculate the degree matrix, https://github.com/pyg-team/pytorch_geometric/issues/1261#issuecomment-633913984
        D = torch.diag(utils.degree(edge_index[0])) 
        D = D[self.not_isolated, :][:, self.not_isolated]

        self.Ahat = torch.linalg.solve(D, A)
        self.Ahat.requires_grad = False

        if phi == 'squared_error': 
            def phi(Z):
                #Define the squared error phi function.
                #Remove the Z values of the not isolated values.
                Z = Z[self.not_isolated, :]

                #Calculate the "averaging" of the neighborhood.
                Zprime = torch.matmul(self.Ahat, Z)
                return 0.5 * (torch.norm(Zprime - Z, p=2, dim=1)**2).sum()

        elif phi == 'cross_entropy':
            def phi(Z):
                #Define the cross entropy phi function.
                Z = Z[self.not_isolated, :]
                P = torch.softmax(Z, dim=1)
                
                #Calculate the "averaging" of the neighborhood.
                Zprime = torch.matmul(self.Ahat, Z)
                Q = torch.softmax(self.Ahat, Zprime)
                return - (P * torch.log(Q)).sum()

        elif phi == 'KL_divergence':
            def phi(Z):
                #Define the KL divergence phi function
                Z = Z[self.not_isolated, :]
                P = torch.softmax(Z, dim=1)
                
                #Calculate the "averaging" of the neighborhood.
                Zprime = torch.matmul(self.Ahat, Z)
                Q = torch.softmax(self.Ahat, Zprime)

                return (P * torch.log(P / Q)).sum()
                
        else:
            raise NotImplementedError()

        #Set phi accordingly.
        self.phi = lambda Z : phi(Z)

    def __call__(self, Z):
        '''
        Evaluate the loss function. 
        '''
    
        #Equation (2) in the paper. 
        return 1/self.N * self.phi(Z)