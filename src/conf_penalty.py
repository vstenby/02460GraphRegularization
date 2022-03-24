import torch

def conf_penalty(Z):
    '''
    Calculates the confidence penalty.
    '''
    #Confidence penalization from the blackboard:
    #loss = loss - beta*H(P), where H(P) = - \sum_{i=1}^{N} \sum_{j=1}^{C} P_{ij} log P_{ij}

    P = torch.softmax(Z, dim=1)

    return (-1.0) * (P * torch.log(P)).sum()