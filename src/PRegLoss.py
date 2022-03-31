import torch
import torch_geometric.utils as utils
import numpy as np

class TensorUnmasker:
    def __init__(self, alpha, N, random_nodes_every_call, seed):
        #Takes alpha, the unmasking coefficient and N, the number of nodes. (excluding self isolated.)
        assert random_nodes_every_call in [0, 1], 'random_nodes_every_call should be 0 or 1.'

        torch.manual_seed(seed)
        
        self.alpha = alpha
        self.N = N
        self.random_nodes_every_call = random_nodes_every_call

        #Calculate k, the number of nodes to apply it on.
        self.k = np.floor(alpha * N).astype(int)
        self.idx = None

    def __call__(self, tensors):
        #Tensors should be a list of tensors that we should slice.
        if self.random_nodes_every_call:
            perm = torch.randperm(self.N)
            idx  = perm[:self.k]
            return [T[idx,] for T in tensors]
        else:
            if self.idx is None:
                perm = torch.randperm(self.N)
                self.idx  = perm[:self.k]
            return [T[self.idx,] for T in tensors]

class PRegLoss:
    '''
    Calculate the P-Reg loss as defined in "Rethinking Graph Regularisation for Graph Neural Networks"
    '''

    def __init__(self, phi, edge_index, unmask_dict, device=None):
        '''
        Input:
            phi, either 'squared_error', 'cross_entropy' or 'KL_divergence'
            edge_index, used to construct the adjacency matrix and the diagonal degree matrix.
        '''

        assert np.all([x in unmask_dict.keys() for x in ['alpha', 'random_nodes_every_call', 'seed']]), 'unmask_dict should contain alpha, random_nodes_every_call and seed.'

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

        
        if unmask_dict['alpha'] != 1:
            self.unmasker = TensorUnmasker(alpha=unmask_dict['alpha'], N=self.N, random_nodes_every_call=unmask_dict['random_nodes_every_call'], seed=unmask_dict['seed'])
        else:
            #No need to call the unmasker if alpha=1.
            self.unmasker = None

        self.device = device

        if phi == 'squared_error': 
            def phi(Z):
                #Define the squared error phi function.
                #Remove the Z values of the not isolated values.
                Z = Z[self.not_isolated, :]

                #Calculate the "averaging" of the neighborhood.
                Zprime = torch.matmul(self.Ahat.to(device), Z.to(device))

                #Do the unmasking!
                if self.unmasker is not None:
                    Z, Zprime = self.unmasker([Z, Zprime])

                return 0.5 * (torch.norm(Zprime - Z, p=2, dim=1)**2).sum()

        elif phi == 'cross_entropy':
            def phi(Z):
                #Define the cross entropy phi function.
                Z = Z[self.not_isolated, :]
                P = torch.softmax(Z, dim=1)
                
                #Calculate the "averaging" of the neighborhood.
                Zprime = torch.matmul(self.Ahat.to(device), Z.to(device))
                Q = torch.softmax(Zprime, dim=1)

                #Do the unmasking!
                if self.unmasker is not None:
                    P, Q = self.unmasker([P, Q])

                return - (P * torch.log(Q)).sum()

        elif phi == 'KL_divergence':
            def phi(Z):
                #Define the KL divergence phi function
                Z = Z[self.not_isolated, :]
                P = torch.softmax(Z, dim=1)
                
                #Calculate the "averaging" of the neighborhood.
                Zprime = torch.matmul(self.Ahat.to(device), Z)
                Q = torch.softmax(Zprime, dim=1)

                logP = torch.log(P)
                logQ = torch.log(Q)

                #Do the unmasking!
                if self.unmasker is not None:
                    P, logP, logQ = self.unmasker([P, logP, logQ])

                return (P * torch.exp(logP - logQ)).sum()

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