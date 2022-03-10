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

        #Set the value of mu.   
        self.mu = mu

        #Set the train mask.
        self.train_mask = train_mask

        #Find the indices of the not isolated nodes.
        self.not_isolated = utils.degree(edge_index[0]) > 0

        #Calculate the adjacency matrix.
        A = utils.to_dense_adj(edge_index).squeeze(0)
        A = A[self.not_isolated, :][:, self.not_isolated]
        
        #Set N, the number of nodes in the graph.
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

        self.phi = lambda Z : phi(Z)

        #Use the CrossEntropyLoss. 
        self.cross_entropy = torch.nn.CrossEntropyLoss()

    def __call__(self, Z, y):
        '''
        Evaluate the loss function. 
        '''
        N = len(y)
        
        L1 = self.cross_entropy(Z[self.train_mask], y[self.train_mask])

        #Equation (2) in the paper. 
        L2 = self.mu * 1/self.N * self.phi(Z)

        return L1 + self.mu * L2