import torch
import torch.nn as nn

# A radom Gaussian noise uses in the ZINB-based denoising autoencoder.
class GaussianNoise(nn.Module):
    def __init__(self, device=torch.device('cpu'), sigma=1, is_relative_detach=True):
        super(GaussianNoise,self).__init__()
        self.sigma = sigma
        self.is_relative_detach = is_relative_detach
        self.noise = torch.tensor(0, device = device)

    def forward(self, x):
        if self.training and self.sigma != 0:
            scale = self.sigma * x.detach() if self.is_relative_detach else self.sigma * x
            sampled_noise = self.noise.repeat(*x.size()).float().normal_() * scale
            x = x + sampled_noise
        return x 
    
class model1(nn.Module):
    def __init__(self, n_input, n_hidden, dropout=0.1):
        super(model1, self).__init__()
        self.bulk_encoder = nn.Sequential(nn.Linear(n_input, n_hidden[0]),
                                          nn.ReLU(),
                                          nn.BatchNorm1d(n_hidden[0]),
                                          nn.Dropout(p=dropout),
                                          nn.Linear(n_hidden[0], n_hidden[1]))
        self.sc_encoder = nn.Sequential(nn.Linear(n_input, n_hidden[0]),
                                        nn.ReLU(),
                                        nn.BatchNorm1d(n_hidden[0]),
                                        nn.Dropout(p=dropout),
                                        nn.Linear(n_hidden[0], n_hidden[1]))

        self.decoder = nn.Sequential(nn.Linear(n_hidden[1], n_hidden[0]),
                                    nn.ReLU(),
                                    nn.BatchNorm1d(n_hidden[0]),
                                    nn.Dropout(p=dropout),
                                    nn.Linear(n_hidden[0], n_input))
        
    def encode(self, x_bulk, x_sc):
        return self.bulk_encoder(x_bulk), self.sc_encoder(x_sc)
    
    def forward(self, x_bulk, x_sc):
        z_bulk = self.bulk_encoder(x_bulk)
        z_sc = self.sc_encoder(x_sc)
        x_bulk_bar = self.decoder(z_bulk)
        x_sc_bar = self.decoder(z_sc)
        return z_bulk, z_sc, x_bulk_bar, x_sc_bar

class model2(nn.Module):
    def __init__(self, n_input, n_hidden, dropout=0.1):
        super(model2, self).__init__()
        self.bulk_encoder = nn.Sequential(nn.Linear(n_input, n_hidden[0]),
                                          nn.ReLU(),
                                          nn.BatchNorm1d(n_hidden[0]),
                                          nn.Dropout(p=dropout),
                                          nn.Linear(n_hidden[0], n_hidden[1]))
        self.sc_encoder = nn.Sequential(nn.Linear(n_input, n_hidden[0]),
                                        nn.ReLU(),
                                        nn.BatchNorm1d(n_hidden[0]),
                                        nn.Dropout(p=dropout),
                                        nn.Linear(n_hidden[0], n_hidden[1]))
        self.share_encoder = nn.Sequential(nn.Linear(n_input, n_hidden[0]),
                                        nn.ReLU(),
                                        nn.BatchNorm1d(n_hidden[0]),
                                        nn.Dropout(p=dropout),
                                        nn.Linear(n_hidden[0], n_hidden[1]))
        
        self.decoder = nn.Sequential(nn.Linear(n_hidden[1]*2, n_hidden[0]),
                                    nn.ReLU(),
                                    nn.BatchNorm1d(n_hidden[0]),
                                    nn.Dropout(p=dropout),
                                    nn.Linear(n_hidden[0], n_input))
        
    def encode(self, x_bulk, x_sc):
        return self.share_encoder(x_bulk), self.share_encoder(x_sc)
    
    def forward(self, x_bulk, x_sc):
        z_bulk_p = self.bulk_encoder(x_bulk)
        z_sc_p = self.sc_encoder(x_sc)
        z_bulk_s = self.share_encoder(x_bulk)
        z_sc_s = self.share_encoder(x_sc)
        z_bulk = torch.cat((z_bulk_p, z_bulk_s), dim=1)
        z_sc = torch.cat((z_sc_p, z_sc_s), dim=1)
        x_bulk_bar = self.decoder(z_bulk)
        x_sc_bar = self.decoder(z_sc)
        return z_bulk_s, z_sc_s, z_bulk_p, z_sc_p, x_bulk_bar, x_sc_bar
    
class model3(nn.Module):
    '''
    increase the mask matrix to execute a sparsity decoder
    '''
    def __init__(self, pathway_mask, n_hidden, dropout=0.1):
        super(model3, self).__init__()
        self.pathway_mask = pathway_mask
        self.n_input = self.pathway_mask.shape[0]
        self.n_pathways = self.pathway_mask.shape[1]
        self.Gnoise = GaussianNoise(sigma=0.1)
        self.bulk_encoder = nn.Sequential(nn.Linear(self.n_input, n_hidden[0]),
                                          nn.ReLU(),
                                          nn.BatchNorm1d(n_hidden[0]),
                                          nn.Dropout(p=dropout),
                                          nn.Linear(n_hidden[0], self.n_pathways))
        self.sc_encoder = nn.Sequential(nn.Linear(self.n_input, n_hidden[0]),
                                        nn.ReLU(),
                                        nn.BatchNorm1d(n_hidden[0]),
                                        nn.Dropout(p=dropout),
                                        nn.Linear(n_hidden[0], self.n_pathways))
        self.share_encoder = nn.Sequential(nn.Linear(self.n_input, n_hidden[0]),
                                        nn.ReLU(),
                                        nn.BatchNorm1d(n_hidden[0]),
                                        nn.Dropout(p=dropout),
                                        nn.Linear(n_hidden[0], self.n_pathways))
        self.decoder = nn.Sequential(nn.Linear(self.n_pathways*2, n_hidden[0]),
                                    nn.ReLU(),
                                    nn.BatchNorm1d(n_hidden[0]),
                                    nn.Dropout(p=dropout),
                                    nn.Linear(n_hidden[0], self.n_input))
        
        self.sparse_decoder = CustomizedLinear(self.pathway_mask.T)
        
    def share_encode(self, x):
        return self.share_encoder(x)
    
    def forward(self, x_bulk, x_sc):
        # x_bulk = self.Gnoise(x_bulk)
        x_sc = self.Gnoise(x_sc)
        z_bulk_p = self.bulk_encoder(x_bulk)
        z_sc_p = self.sc_encoder(x_sc)
        z_bulk_s = self.share_encoder(x_bulk)
        z_sc_s = self.share_encoder(x_sc)
        
        z_bulk = torch.cat((z_bulk_p, z_bulk_s), dim=1)
        z_sc = torch.cat((z_sc_p, z_sc_s), dim=1)
        
        x_bulk_bar_1 = self.sparse_decoder(z_bulk_s)
        x_sc_bar_1 = self.sparse_decoder(z_sc_s)
        x_bulk_bar_2 = self.decoder(z_bulk)
        x_sc_bar_2 = self.decoder(z_sc)
        return [z_bulk_s, z_sc_s, z_bulk_p, z_sc_p, x_bulk_bar_1, x_sc_bar_1, x_bulk_bar_2, x_sc_bar_2]
    
class critic(nn.Module):
    def __init__(self, n_hidden, n_classifier_hidden, dropout=0.1):
        super(critic, self).__init__()
        self.critic_mlp = nn.Sequential(nn.Linear(n_hidden, n_classifier_hidden),
                                        nn.ReLU(),
                                        nn.BatchNorm1d(n_classifier_hidden),
                                        nn.Dropout(p=dropout),
                                        nn.Linear(n_classifier_hidden, 1))
        
    def forward(self, embedding):
        return self.critic_mlp(embedding)

import math

class CustomizedLinear(nn.Module):
    
    def __init__(self, mask, bias=True):
        """
        extended torch.nn module which mask connection.
        Arguments
        ------------------
        mask [torch.tensor]:
            the shape is (n_input_feature, n_output_feature).
            the elements are 0 or 1 which declare un-connected or
            connected.
        bias [bool]:
            flg of bias.
        """
        super(CustomizedLinear, self).__init__()
        self.input_features = mask.shape[0]
        self.output_features = mask.shape[1]
        if isinstance(mask, torch.Tensor):
            self.mask = mask.type(torch.float).t()
        else:
            self.mask = torch.tensor(mask, dtype=torch.float).t()

        self.mask = nn.Parameter(self.mask, requires_grad=False)

        # nn.Parameter is a special kind of Tensor, that will get
        # automatically registered as Module's parameter once it's assigned
        # as an attribute. Parameters and buffers need to be registered, or
        # they won't appear in .parameters() (doesn't apply to buffers), and
        # won't be converted when e.g. .cuda() is called. You can use
        # .register_buffer() to register buffers.
        # nn.Parameters require gradients by default.
        self.weight = nn.Parameter(torch.Tensor(self.output_features, self.input_features))

        if bias:
            self.bias = nn.Parameter(torch.Tensor(self.output_features))
        else:
            # You should always register all possible parameters, but the
            # optional ones can be None if you want.
            self.register_parameter('bias', None)
        self.reset_parameters()

        # mask weight
        self.weight.data = self.weight.data * self.mask

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def reset_params_pos(self):
        """ Same as reset_parameters, but only initialize to positive values. """
        stdv = 1./math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(0,stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)
    
    def forward(self, input):
        # See the autograd section for explanation of what happens here.
        return CustomizedLinearFunction.apply(input, self.weight, self.bias, self.mask)

    def extra_repr(self):
        # (Optional)Set the extra information about this module. You can test
        # it by printing an object of this class.
        return 'input_features={}, output_features={}, bias={}'.format(
            self.input_features, self.output_features, self.bias is not None
        )
    
class CustomizedLinearFunction(torch.autograd.Function):
    """
    autograd function which masks it's weights by 'mask'.
    """

    # Note that both forward and backward are @staticmethods
    @staticmethod
    # bias, mask is an optional argument
    def forward(ctx, input, weight, bias=None, mask=None):
        if mask is not None:
            # change weight to 0 where mask == 0
            weight = weight * mask
        output = input.mm(weight.t())
        if bias is not None:
            output += bias.unsqueeze(0).expand_as(output)
        ctx.save_for_backward(input, weight, bias, mask)
        return output

    # This function has only a single output, so it gets only one gradient
    @staticmethod
    def backward(ctx, grad_output):
        # This is a pattern that is very convenient - at the top of backward
        # unpack saved_tensors and initialize all gradients w.r.t. inputs to
        # None. Thanks to the fact that additional trailing Nones are
        # ignored, the return statement is simple even when the function has
        # optional inputs.
        input, weight, bias, mask = ctx.saved_tensors
        grad_input = grad_weight = grad_bias = grad_mask = None

        # These needs_input_grad checks are optional and there only to
        # improve efficiency. If you want to make your code simpler, you can
        # skip them. Returning gradients for inputs that don't require it is
        # not an error.
        if ctx.needs_input_grad[0]:
            grad_input = grad_output.mm(weight)
        if ctx.needs_input_grad[1]:
            grad_weight = grad_output.t().mm(input)
            if mask is not None:
                # change grad_weight to 0 where mask == 0
                grad_weight = grad_weight * mask
        #if bias is not None and ctx.needs_input_grad[2]:
        if ctx.needs_input_grad[2]:
            grad_bias = grad_output.sum(0).squeeze(0)

        return grad_input, grad_weight, grad_bias, grad_mask
    
class fcn(nn.Module):
    def __init__(self, n_input, n_hidden, dropout=0.1):
        super(fcn, self).__init__()
        self.linear_1 = nn.Linear(n_input, n_hidden[0])
        # self.BN = nn.BatchNorm1d(n_hidden[0])
        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout)
        self.linear2 = nn.Linear(n_hidden[0], n_hidden[1])
        self.linear3 = nn.Linear(n_hidden[1], 1)
        self.sigmoid = nn.Sigmoid()
        # nn.init.xavier_uniform_(self.linear2.weight, gain=nn.init.calculate_gain('relu'))
    def forward(self, x):
        emb = self.linear2(self.dropout(self.relu1(self.linear_1(x))))
        y_pred = self.sigmoid(self.linear3(self.relu2(emb)))
        return y_pred
    
    
class fcn2(nn.Module):
    def __init__(self, n_input, n_hidden, dropout=0.1):
        super(fcn2, self).__init__()
        self.linear_1 = nn.Linear(n_input, n_hidden[0])
        # self.BN = nn.BatchNorm1d(n_hidden[0])
        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout)
        self.linear2 = nn.Linear(n_hidden[0], n_hidden[1])
        self.linear3 = nn.Linear(n_hidden[1], 2)
        # nn.init.xavier_uniform_(self.linear2.weight, gain=nn.init.calculate_gain('relu'))
    def forward(self, x):
        emb = self.linear2(self.dropout(self.relu1(self.linear_1(x))))
        y_pred = self.linear3(self.relu2(emb))
        return y_pred