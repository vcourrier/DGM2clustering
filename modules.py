import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
import torch.nn.init as weight_init
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

import os
import numpy as np
import random
import sys
from torch.autograd import Variable
import math



parentPath = os.path.abspath("..")
sys.path.insert(0, parentPath)# add parent folder to path so as to import common modules
from helper import SOS_ID, EOS_ID

class MLP(nn.Module):
    """
    Multi-Layer Perceptron (MLP) module.
    """
    def __init__(self, input_size, arch, output_size, activation=nn.ReLU(), batch_norm=True, init_w=0.02, discriminator=False):
        """
        Initialize the MLP.

        Args:
            input_size (int): Dimensionality of the input data.
            arch (str): Architecture specification for hidden layers.
            output_size (int): Dimensionality of the output.
            activation (nn.Module): Activation function.
            batch_norm (bool): Whether to use batch normalization.
            init_w (float): Initialization scale for weights.
            discriminator (bool): Whether the MLP is used as a discriminator.
        """
        super(MLP, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.init_w= init_w

        layer_sizes = [input_size] + [int(x) for x in arch.split('-')]
        self.layers = []

        for i in range(len(layer_sizes)-1):
            layer = nn.Linear(layer_sizes[i], layer_sizes[i+1])
            self.layers.append(layer)
            self.add_module("layer"+str(i+1), layer)            
            if batch_norm and not(discriminator and i==0):  # if used as discriminator, then there is no batch norm in the first layer
                bn = nn.BatchNorm1d(layer_sizes[i+1], eps=1e-05, momentum=0.1)
                self.layers.append(bn)
                self.add_module("bn"+str(i+1), bn)
            self.layers.append(activation)
            self.add_module("activation"+str(i+1), activation)

        layer = nn.Linear(layer_sizes[-1], output_size)
        self.layers.append(layer)
        self.add_module("layer"+str(len(self.layers)), layer)

        self.init_weights()

    def forward(self, x):
        """
        Forward pass of the MLP.

        Args:
            x (tensor): Input tensor.

        Returns:
            x (tensor): Output tensor.
        """
        for i, layer in enumerate(self.layers):
            x = layer(x)
        return x

    def init_weights(self):
        """
        Initialize the weights of the MLP.
        """
        for layer in self.layers:
            try:
                layer.weight.data.normal_(0, self.init_w)
                layer.bias.data.fill_(0)
            except: pass

class Encoder(nn.Module):
    """
    Encoder module for sequence data.
    """
    def __init__(self, embedder, input_size, hidden_size, bidir, n_layers, dropout=0.5, noise_radius=0.2):
        """
        Initialize the Encoder.

        Args:
            embedder: Embedding layer (nn.Module).
            input_size (int): Dimensionality of the input data.
            hidden_size (int): Size of the hidden state in the RNN.
            bidir (bool): Whether the RNN is bidirectional.
            n_layers (int): Number of RNN layers.
            dropout (float): Dropout rate.
            noise_radius (float): Radius of Gaussian noise.
        """
        super(Encoder, self).__init__()
        
        self.hidden_size = hidden_size
        self.noise_radius=noise_radius
        self.n_layers = n_layers
        self.bidir = bidir
        assert type(self.bidir)==bool
        self.dropout=dropout
        
        self.embedding = embedder  # nn.Embedding(vocab_size, emb_size)
        self.rnn = nn.GRU(input_size, hidden_size, n_layers, batch_first=True, bidirectional=bidir)
        self.init_h = nn.Parameter(torch.randn(self.n_layers*(1+self.bidir), 1, self.hidden_size), requires_grad=True)#learnable h0
        self.init_weights()
        
    def init_weights(self):
        """
        Initialize the weights of the Encoder.
        """
        for w in self.rnn.parameters():  # initialize the gate weights with orthogonal
            if w.dim()>1:
                weight_init.orthogonal_(w)
                
    def store_grad_norm(self, grad):
        """
        Store the gradient norm.

        Args:
            grad (tensor): Gradient tensor.

        Returns:
            grad (tensor): Gradient tensor.
        """
        norm = torch.norm(grad, 2, 1)
        self.grad_norm = norm.detach().data.mean()
        return grad
    
    def forward(self, inputs, input_lens=None, init_h=None, noise=False): 
        """
        Forward pass of the Encoder.

        Args:
            inputs (tensor): Input tensor.
            input_lens (tensor): Input sequence lengths.
            init_h (tensor): Initial hidden state.
            noise (bool): Whether to add noise to the output.

        Returns:
            enc (tensor): Encoded representation.
            hids (tensor): Hidden states.
        """
        # init_h: [n_layers*n_dir x batch_size x hid_size]
        if self.embedding is not None:
            inputs=self.embedding(inputs)  # input: [batch_sz x seq_len] -> [batch_sz x seq_len x emb_sz]
        
        batch_size, seq_len, emb_size=inputs.size()
        inputs=F.dropout(inputs, self.dropout, self.training)  # dropout
        
        if input_lens is not None :  # sort and pack sequence 
            input_lens_sorted, indices = input_lens.sort(descending=True)
            inputs_sorted = inputs.index_select(0, indices)        
            inputs = pack_padded_sequence(inputs_sorted, input_lens_sorted.data.tolist(), batch_first=True)
        
        if init_h is None:
            init_h = self.init_h.expand(-1,batch_size,-1).contiguous()  # use learnable initial states, expanding along batches
        # self.rnn.flatten_parameters() # time consuming!!
        hids, h_n = self.rnn(inputs, init_h)  # hids: [b x seq x (n_dir*hid_sz)]  

        if input_lens is not None: # reorder and pad
            _, inv_indices = indices.sort()
            hids, lens = pad_packed_sequence(hids, batch_first=True)     
            hids = hids.index_select(0, inv_indices)
            h_n = h_n.index_select(1, inv_indices)
        h_n = h_n.view(self.n_layers, (1+self.bidir), batch_size, self.hidden_size) #[n_layers x n_dirs x batch_sz x hid_sz]
        h_n = h_n[-1] # get the last layer [n_dirs x batch_sz x hid_sz]
        enc = h_n.transpose(0,1).contiguous().view(batch_size,-1) #[batch_sz x (n_dirs*hid_sz)]

        if noise and self.noise_radius > 0:
            gauss_noise = torch.normal(means=torch.zeros(enc.size(), device=inputs.device),std=self.noise_radius)
            enc = enc + gauss_noise

        return enc, hids


class Encoder_cluster(nn.Module):
    """
    Cluster Encoder module for sequence data.
    """
    def __init__(self, input_dim, hidden_size, dropout, device, bidir = True, block = 'LSTM', n_layers=1):
        """
        Initialize the Cluster Encoder.

        Args:
            input_dim (int): Dimensionality of the input data.
            hidden_size (int): Size of the hidden state in the RNN.
            dropout (float): Dropout rate.
            device: Device for computation.
            bidir (bool): Whether the RNN is bidirectional.
            block (str): Type of RNN block ('LSTM' or 'GRU').
            n_layers (int): Number of RNN layers.
        """
        super(Encoder_cluster, self).__init__()

        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.bidir = bidir
        self.device = device
        assert type(self.bidir)==bool
        self.dropout=dropout

        self.block = block
        if self.block == 'GRU':
            self.rnn = nn.GRU(input_dim, hidden_size, n_layers, batch_first=True, bidirectional=bidir, dropout = self.dropout)
        else:
            self.rnn = nn.LSTM(input_dim, hidden_size, n_layers, batch_first=True, bidirectional=bidir, dropout = self.dropout)
        self.init_h = torch.zeros([self.n_layers*(1+self.bidir), 1, self.hidden_size], device = self.device)

        if self.block == 'LSTM':
            self.init_c = torch.zeros([self.n_layers*(1+self.bidir), 1, self.hidden_size],device = self.device)
        
    def init_weights(self):
        """
        Initialize the weights of the Cluster Encoder.
        """
        for w in self.rnn.parameters():  # initialize the gate weights with orthogonal
            if w.dim()>1:
                weight_init.orthogonal_(w)

    def store_grad_norm(self, grad):
        """
        Store the gradient norm.

        Args:
            grad (tensor): Gradient tensor.

        Returns:
            grad (tensor): Gradient tensor.
        """
        norm = torch.norm(grad, 2, 1)
        self.grad_norm = norm.detach().data.mean()
        return grad

    def forward2(self, inputs, input_lens, rnn_out, exp_last_h_n, exp_last_c_n, init_h=None, init_c = None, noise=False):
        """
        Perform forward pass for training with packing and sorting.

        Args:
            inputs (tensor): Input tensor.
            input_lens (tensor): Input sequence lengths.
            rnn_out (tensor): RNN output tensor.
            exp_last_h_n (tensor): Expected last hidden states.
            exp_last_c_n (tensor): Expected last cell states.
            init_h (tensor): Initial hidden state.
            init_c (tensor): Initial cell state.
            noise (bool): Whether to add noise to the output.

        Returns:
            output_hiddens (tensor): Output hidden states.
            last_hidden_states (tuple): Last hidden and cell states.
        """
        batch_size, seq_len, emb_size=inputs.size()
#         if input_lens is not None:# sort and pack sequence

        output = torch.zeros_like(inputs)

        input_lens_sorted, indices = input_lens.sort(descending=True)
        inputs_sorted = inputs.index_select(0, indices)

        if init_h is None:
            init_h = self.init_h.expand(-1,batch_size,-1).contiguous()# use learnable initial states, expanding along batches
        if self.block == 'LSTM':
            if init_c is None:            
                init_c = self.init_c.expand(-1,batch_size,-1).contiguous()

        last_len = None
        last_id = None

        last_h_n = torch.zeros([(1+self.bidir)*self.n_layers, batch_size, self.hidden_size])
        last_c_n = torch.zeros([(1+self.bidir)*self.n_layers, batch_size, self.hidden_size])

        output_list = torch.zeros([batch_size, seq_len,(1+self.bidir)*self.hidden_size])

        for k in range(len(input_lens_sorted)):
            if last_len is None:
                last_len = input_lens_sorted[k]
                last_id = k 
            else:
                if last_len == input_lens_sorted[k]:
                    continue
                else:
                    if self.block == 'LSTM':
                        hids, (h_n, c_n) = self.rnn(inputs_sorted[last_id:k, 0:last_len], (init_h[:,last_id:k,:], init_c[:,last_id:k,:]))
                        last_c_n[:,last_id:k] = c_n
                    else:
                        hids, h_n = self.rnn(inputs_sorted[last_id:k, 0:last_len], init_h[:,last_id:k,:])

                    output_list[last_id:k, 0:last_len] = hids
                    last_h_n[:,last_id:k] = h_n

                    last_id = k
                    last_len = input_lens_sorted[k]

        if self.block == 'LSTM':
            hids, (h_n, c_n) = self.rnn(inputs_sorted[last_id:k+1, 0:last_len], (init_h[:,last_id:k+1,:], init_c[:,last_id:k+1,:]))
            last_c_n[:,last_id:k+1] = c_n
        else:
            hids, h_n = self.rnn(inputs_sorted[last_id:k+1, 0:last_len], init_h[:,last_id:k+1,:])

        output_list[last_id:k+1, 0:last_len] = hids
        last_h_n[:,last_id:k+1] = h_n

        _, inv_indices = indices.sort()

        output_hiddens = output_list[inv_indices]

        print(torch.norm(output_hiddens[0, 0:input_lens[0]] - rnn_out[0, 0:input_lens[0]]))

        print(torch.norm(output_hiddens[-1, 0:input_lens[-1]] - rnn_out[-1, 0:input_lens[-1]]))

        print(torch.norm(last_h_n[:,inv_indices] - exp_last_h_n))

        print(torch.norm(last_c_n[:,inv_indices] - exp_last_c_n))

        return output_hiddens, (last_h_n[:,inv_indices], last_c_n[:, inv_indices])

  
    def check_hidden_states(self, x, x_lens, init_h, init_c, hids, h_n, c_n):
        """
        Check the hidden states' consistency.

        Args:
            x (tensor): Input tensor.
            x_lens (tensor): Input sequence lengths.
            init_h (tensor): Initial hidden state.
            init_c (tensor): Initial cell state.
            hids (tensor): Hidden states.
            h_n (tensor): Hidden state tensor.
            c_n (tensor): Cell state tensor.
        """

        T_max = x_lens.max()

        for i in range(x.shape[0]):
            origin_hids, (curr_h_n, curr_c_n) = self.rnn(x[i, 0:x_lens[i]].view(1, x_lens[i], x.shape[2]), (init_h[:,i,:].view(init_h[:,i,:].shape[0], 1, init_h[:,i,:].shape[1]), init_c[:,i,:].view(init_c[:,i,:].shape[0], 1, init_c[:,i,:].shape[1])))

            print('lens::', T_max, x_lens[i])

            print(torch.norm(curr_h_n.view(-1) - h_n[:,i,:].reshape((-1))))

            print(torch.norm(curr_c_n.view(-1) - c_n[:,i,:].reshape((-1))))


    def forward(self, inputs, input_lens=None, init_h=None, init_c = None, noise=False, test = False): 
        """
        Forward pass of the Cluster Encoder.

        Args:
            inputs (tensor): Input tensor.
            input_lens (tensor): Input sequence lengths.
            init_h (tensor): Initial hidden state.
            init_c (tensor): Initial cell state.
            noise (bool): Whether to add noise to the output.
            test (bool): Whether it's a test forward pass.

        Returns:
            hids (tensor): Hidden states.
            hidden_states (tuple): Hidden and cell states.
        """
        # init_h: [n_layers*n_dir x batch_size x hid_size]

        origin_inputs = inputs.clone()
        
        batch_size, seq_len, emb_size=inputs.size()
        
        if input_lens is not None:# sort and pack sequence 
            input_lens_sorted, indices = input_lens.sort(descending=True)
            inputs_sorted = inputs.index_select(0, indices)        
            inputs = pack_padded_sequence(inputs_sorted, input_lens_sorted.data.tolist(), batch_first=True)
            
            if init_h is not None:
                
                init_h = init_h.index_select(1, indices)
                if self.block == 'LSTM':

                    init_c = init_c.index_select(1, indices)

        if init_h is None:
            init_h = self.init_h.expand(-1,batch_size,-1).contiguous()  # use learnable initial states, expanding along batches
        if self.block == 'LSTM':
            if init_c is None:            
                init_c = self.init_c.expand(-1,batch_size,-1).contiguous()

            origin_hids, (h_n, c_n) = self.rnn(inputs, (init_h, init_c))

        else:
            origin_hids, h_n = self.rnn(inputs, init_h)
            
        #self.rnn.flatten_parameters() # time consuming!!
         # hids: [b x seq x (n_dir*hid_sz)]  
                                                  # h_n: [(n_layers*n_dir) x batch_sz x hid_sz] (2=fw&bw)
        if input_lens is not None: # reorder and pad
            _, inv_indices = indices.sort()
            hids, lens = pad_packed_sequence(origin_hids, batch_first=True)     
            hids = hids.index_select(0, inv_indices)
            h_n = h_n.index_select(1, inv_indices)
            
            if self.block == 'LSTM':
                c_n = c_n.index_select(1, inv_indices)

        if self.block =='LSTM':    
            return hids, (h_n, c_n)
        else:
            return hids, h_n


class GatedTransition2(nn.Module):
    """
    A module for parameterizing the gaussian latent transition probability `p(z_t | z_{t-1})`
    """
    def __init__(self, z_dim, h_dim, trans_dim):
        super(GatedTransition2, self).__init__()
        self.gate = nn.Sequential( 
            nn.Linear(h_dim, trans_dim),
            nn.ReLU(),
            nn.Linear(trans_dim, z_dim),
            nn.Sigmoid()
        )
        self.proposed_mean = nn.Sequential(
            nn.Linear(h_dim, trans_dim),
            nn.ReLU(),
            nn.Linear(trans_dim, z_dim)
        )           

        self.lstm = torch.nn.LSTM(z_dim, h_dim)

        self.z_to_mu = nn.Linear(z_dim, z_dim)
        # modify the default initialization of z_to_mu so that it starts out as the identity function
        self.z_to_mu.weight.data = torch.eye(z_dim)
        self.z_to_mu.bias.data = torch.zeros(z_dim)
        self.z_to_logvar = nn.Linear(z_dim, z_dim)
        self.relu = nn.ReLU()


    def forward(self, z_t_1, h_t_1, c_t_1):
        """
        Forward pass of the GatedTransition2 module. Given the latent `z_{t-1}` corresponding to the time step t-1.
        
        Args:
            z_t_1 (Tensor): Latent variable at time step t-1.
            h_t_1 (Tensor): Hidden state at time step t-1.
            c_t_1 (Tensor): Cell state at time step t-1.

        Returns:
            z_t (Tensor): Sampled latent variable at time step t.
            mu (Tensor): Mean vector for the gaussian distribution.
            logvar (Tensor): Log-variance vector for the gaussian distribution.
            h_t (Tensor): Hidden state of the LSTM at time step t.
            c_t (Tensor): Cell state of the LSTM at time step t.        """        
        gate = self.gate(h_t_1)  # compute the gating function
        
        _, (h_t, c_t) = self.lstm(z_t_1.view(1, z_t_1.shape[0], z_t_1.shape[1]).contiguous(), (h_t_1, c_t_1))

        proposed_mean = self.proposed_mean(h_t)  # compute the 'proposed mean'
        mu = (1 - gate) * self.z_to_mu(z_t_1) + gate * proposed_mean # compute the scale used to sample z_t, using the proposed mean from
        logvar = self.z_to_logvar(self.relu(proposed_mean)) 
        epsilon = torch.randn(z_t_1.size(), device=z_t_1.device)  # sampling z by re-parameterization
        z_t = mu + epsilon * torch.exp(0.5 * logvar)    # [batch_sz x z_sz]
        return z_t, mu.view(mu.shape[1], mu.shape[2]), logvar.view(logvar.shape[1], logvar.shape[2]), h_t, c_t


class GatedTransition(nn.Module):
    """
    A module for parameterizes the gaussian latent transition probability `p(z_t | z_{t-1})`
    """
    def __init__(self, z_dim, trans_dim):
        super(GatedTransition, self).__init__()
        self.gate = nn.Sequential( 
            nn.Linear(z_dim, z_dim),
            nn.Sigmoid()
        )
        self.proposed_mean = nn.Sequential(
            nn.Linear(z_dim,  z_dim)
        )           
        self.z_to_mu = nn.Linear(z_dim, z_dim)
        # modify the default initialization of z_to_mu so that it starts out as the identity function
        self.z_to_mu.weight.data = torch.eye(z_dim)
        self.z_to_mu.bias.data = torch.zeros(z_dim)
        self.z_to_logvar = nn.Linear(z_dim, z_dim)
        self.relu = nn.ReLU()


    def forward(self, z_t_1):
        """
        Forward pass of the GatedTransition module. Given the latent `z_{t-1}` corresponding to the time step t-1.

        Args:
            z_t_1 (Tensor): Latent variable at time step t-1.

        Returns:
            z_t (Tensor): Sampled latent variable at time step t.
            mu (Tensor): Mean vector for the gaussian distribution.
            logvar (Tensor): Log-variance vector for the gaussian distribution.
        """  
        gate = self.gate(z_t_1) # compute the gating function
        proposed_mean = self.proposed_mean(z_t_1) # compute the 'proposed mean'
        mu = (1 - gate) * self.z_to_mu(z_t_1) + gate * proposed_mean # compute the scale used to sample z_t, using the proposed mean from
        logvar = F.softplus(self.z_to_logvar(self.relu(proposed_mean))) 
        epsilon = torch.randn(z_t_1.size(), device=z_t_1.device) # sampling z by re-parameterization
#         z_t = mu + epsilon * torch.exp(0.5 * logvar)    # [batch_sz x z_sz]
        z_t = mu + epsilon * logvar
        return z_t, mu, logvar 


class PostNet_cluster(nn.Module):
    """
    Parameterizes `q(z_t|z_{t-1}, x_{t:T})`, which is the basic building block of the inference (i.e. the variational distribution). 
    The dependence on `x_{t:T}` is through the hidden state of the RNN
    """
    def __init__(self, z_dim, h_dim, cluster_num, dropout, sampling_times, bidirt = True):
        super(PostNet_cluster, self).__init__()

        self.h_to_z = nn.Sequential(
            nn.Linear((1+bidirt)*h_dim + z_dim, cluster_num),
            nn.Dropout(p = dropout)
        )

        self.sampling_times = sampling_times

    def gen_z_t_dist_now(self, z_t_1, h_x):
        h_combined = torch.cat([z_t_1, h_x], -1)

        z_category = F.softmax(self.h_to_z(h_combined), dim = -1)
        return z_category,z_category 

    def get_z_t_from_samples(self, z_t, phi_table):

        return torch.mm(z_t, torch.t(phi_table))

    def forward(self, z_t_1, h_x, phi_table, t, temp=0):
        """
        Given the latent z at a particular time step t-1 as well as the hidden
        state of the RNN `h(x_{t:T})` we return the mean and scale vectors that
        parameterize the (diagonal) gaussian distribution `q(z_t|z_{t-1}, x_{t:T})`
        """

        z_category, z_category_sparse = self.gen_z_t_dist_now(z_t_1, h_x)
        z_t = z_category

        if len(z_t.shape) == 2:
            phi_z = torch.mm(z_t, torch.t(phi_table))
        else:
            phi_table_full = (torch.t(phi_table)).view(1, phi_table.shape[1], phi_table.shape[0])

            phi_table_full = phi_table_full.repeat(phi_table.shape[1], 1, 1)

            phi_z = torch.bmm(z_t, phi_table_full)

        return z_t, z_category, phi_z, z_category_sparse


class PostNet_cluster_time(nn.Module):
    """
    Parameterizes `q(z_t|z_{t-1}, x_{t:T})`, which is the basic building block of the inference (i.e. the variational distribution). 
    The dependence on `x_{t:T}` is through the hidden state of the RNN
    """
    def __init__(self, z_dim, h_dim, cluster_num, dropout, use_gumbel_softmax, sampling_times, bidirt = True):
        """
        Initialize the PostNet_cluster module.

        Parameters:
            z_dim (int): Dimensionality of the latent variable z.
            h_dim (int): Dimensionality of the hidden state of the RNN.
            cluster_num (int): Number of clusters/categories for z.
            dropout (float): Dropout probability applied to the hidden state.
            sampling_times (int): Number of times to sample z during forward pass.
            bidirt (bool, optional): If True, consider bidirectional RNN. Defaults to True.
        """
        super(PostNet_cluster_time, self).__init__()
        self.z_to_h = nn.Sequential(
            nn.Linear(z_dim+1, (1+bidirt)*h_dim),
            nn.Tanh(),
            nn.Dropout(p = dropout)
        )

        self.h_to_z = nn.Sequential(
            nn.Linear((1+bidirt)*h_dim, cluster_num),
            nn.Dropout(p = dropout)
            )

        self.use_gumbel_softmax = use_gumbel_softmax

        self.sampling_times = sampling_times

    def gen_z_t_dist_now(self, z_t_1, h_x):
        """
        Generate the distribution over z_t categories based on z_t_1 and h_x.

        Parameters:
            z_t_1 (torch.Tensor): Latent variable z at time step t-1.
            h_x (torch.Tensor): Hidden state of the RNN.

        Returns:
            z_category (torch.Tensor): Distribution over z_t categories.
        """

        h_combined = 0.5*(self.z_to_h(z_t_1) + h_x)# combine the rnn hidden state with a transformed version of z_t_1

        z_category = F.softmax(self.h_to_z(h_combined), dim = 1)

        return z_category

    def get_z_t_from_samples(self, z_t, phi_table):
        """
        Transform z_t samples to z_t values based on phi_table.

        Parameters:
            z_t (torch.Tensor): Sampled z_t values.
            phi_table (torch.Tensor): Table for transforming z.

        Returns:
            transformed_z_t (torch.Tensor): Transformed z_t values.
        """
        return torch.mm(z_t, torch.t(phi_table))


    def forward(self, z_t_1, h_x, phi_table):
        """
        Forward pass of the module. Given the latent z at a particular time step t-1 as well as the hidden
        state of the RNN `h(x_{t:T})` we return the mean and scale vectors that
        parameterize the (diagonal) gaussian distribution `q(z_t|z_{t-1}, x_{t:T})`

        Parameters:
            z_t_1 (torch.Tensor): Latent z at time step t-1.
            h_x (torch.Tensor): Hidden state of the RNN.
            phi_table (torch.Tensor): Table for transforming z.
            t (int): Time step.
            temp (float, optional): Temperature for sampling. Defaults to 0.

        Returns:
            z_t (torch.Tensor): Sampled z at time step t.
            z_category (torch.Tensor): Distribution over z categories.
            phi_z (torch.Tensor): Transformed z values.
            z_category_sparse (torch.Tensor): Distribution over z categories (sparse version).
        """
        z_category = self.gen_z_t_dist_now(z_t_1, h_x)

        if self.use_gumbel_softmax:
            
            averaged_z_t = 0
            
            log_prob = Variable(torch.log(z_category))
            
            for k in range(self.sampling_times):           
                curr_z_t = F.gumbel_softmax(log_prob, tau = 0.1)
                
                averaged_z_t += curr_z_t
                
                del curr_z_t

            z_t = averaged_z_t/self.sampling_times
        else:
            z_t = z_category
        
        phi_z = torch.mm(z_t, torch.t(phi_table))

        return z_t, z_category, phi_z


class PostNet_cluster2(nn.Module):
    """
    Parameterizes `q(z_t|z_{t-1}, x_{t:T})`, which is the basic building block of the inference (i.e. the variational distribution). 
    The dependence on `x_{t:T}` is through the hidden state of the RNN
    """
    def __init__(self, z_dim, h_dim, z_std, dropout):
        """
        Initialize the PostNet_cluster module.

        Parameters:
            z_dim (int): Dimensionality of the latent variable z.
            h_dim (int): Dimensionality of the hidden state of the RNN.
            cluster_num (int): Number of clusters/categories for z.
            dropout (float): Dropout probability applied to the hidden state.
            sampling_times (int): Number of times to sample z during forward pass.
            bidirt (bool, optional): If True, consider bidirectional RNN. Defaults to True.
        """
        super(PostNet_cluster2, self).__init__()
        self.z_to_h = nn.Sequential(
            nn.Linear(z_dim, 2*h_dim),
            nn.Tanh(),
            nn.Dropout(p = dropout)
        )

        self.h_to_z_mean = nn.Sequential(
            nn.Linear(2*h_dim, z_dim),
            nn.Dropout(p = dropout)
            )

        self.h_to_z_var = nn.Sequential(
            nn.Linear(2*h_dim, z_dim),
            nn.Dropout(p = dropout)
            )

    def forward(self, z_t_1, h_x):
        """
        Generate the distribution over z_t categories based on z_t_1 and h_x.
        Given the latent z at a particular time step t-1 as well as the hidden
        state of the RNN `h(x_{t:T})` we return the mean and scale vectors that
        parameterize the (diagonal) gaussian distribution `q(z_t|z_{t-1}, x_{t:T})`

        Parameters:
            z_t_1 (torch.Tensor): Latent variable z at time step t-1.
            h_x (torch.Tensor): Hidden state of the RNN.

        Returns:
            z_category (torch.Tensor): Distribution over z_t categories.
        """
        h_combined = 0.5*(self.z_to_h(z_t_1) + h_x)  # combine the rnn hidden state with a transformed version of z_t_1

        z_mean = self.h_to_z_mean(h_combined)

        z_var = self.h_to_z_var(h_combined)

        epsilon = torch.randn(z_t_1.size(), device=z_t_1.device)  # sampling z by re-parameterization
        z_t = z_mean + epsilon * torch.exp(0.5 * z_var)    # [batch_sz x z_sz]

        return z_t, z_mean, z_var
  

class PostNet(nn.Module):
    """
    Parameterizes `q(z_t|z_{t-1}, x_{t:T})`, which is the basic building block of the inference (i.e. the variational distribution). 
    The dependence on `x_{t:T}` is through the hidden state of the RNN
    """
    def __init__(self, z_dim, h_dim):
        """
        Initialize the PostNet module.

        Parameters:
            z_dim (int): Dimensionality of the latent variable z.
            h_dim (int): Dimensionality of the hidden state of the RNN.
        """
        super(PostNet, self).__init__()
        self.z_to_h = nn.Sequential(
            nn.Linear(z_dim, h_dim),
            nn.Tanh()
        )
        self.h_to_mu = nn.Linear(h_dim, z_dim)
        self.h_to_logvar = nn.Linear(h_dim, z_dim)

    def forward(self, z_t_1, h_x):
        """
        Forward pass of the module.
        Given the latent z at a particular time step t-1 as well as the hidden
        state of the RNN `h(x_{t:T})` we return the mean and scale vectors that
        parameterize the (diagonal) gaussian distribution `q(z_t|z_{t-1}, x_{t:T})`

        Parameters:
            z_t_1 (torch.Tensor): Latent z at time step t-1.
            h_x (torch.Tensor): Hidden state of the RNN.

        Returns:
            z_t (torch.Tensor): Sampled z at time step t.
            mu (torch.Tensor): Mean of the latent distribution.
            logvar (torch.Tensor): Log-variance of the latent distribution.
        """
        h_combined = 0.5*(self.z_to_h(z_t_1) + h_x)  # combine the rnn hidden state with a transformed version of z_t_1
        mu = self.h_to_mu(h_combined)
        logvar = self.h_to_logvar(h_combined)
        std = F.softplus(logvar)     
        epsilon = torch.randn(z_t_1.size(), device=z_t_1.device)  # sampling z by re-parameterization
        z_t = epsilon * std + mu   # [batch_sz x z_sz]
        return z_t, mu, logvar 


class FilterLinear(nn.Module):
    """
    A custom linear layer that applies a filter square matrix to the weight matrix before performing linear transformation.
    """
    def __init__(self, in_features, out_features, filter_square_matrix, bias=True):
        """
            Args:
            in_features (int): Number of input features.
            out_features (int): Number of output features.
            filter_square_matrix (torch.Tensor): Filter square matrix with elements 0 or 1.
            bias (bool): Whether to include bias in the linear transformation.

        Note:
            The filter square matrix is applied element-wise to the weight matrix before performing the linear transformation.
        """

        super(FilterLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features

        use_gpu = torch.cuda.is_available()
        self.filter_square_matrix = None
        if use_gpu:
            self.filter_square_matrix = Variable(filter_square_matrix.cuda(), requires_grad=False)
        else:
            self.filter_square_matrix = Variable(filter_square_matrix, requires_grad=False)

        self.weight = Parameter(torch.Tensor(out_features, in_features))
        if bias:
            self.bias = Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        """
        Initialize the weights and biases of the layer.
        """
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input):
        """
        Perform the forward pass through the layer.

        Args:
            input (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after linear transformation.
        """
        return F.linear(input, self.filter_square_matrix.mul(self.weight), self.bias)

    def __repr__(self):
        """
        Generate a string representation of the layer.

        Returns:
            str: String representation of the layer.
        """
        return self.__class__.__name__ + '(' \
            + 'in_features=' + str(self.in_features) \
            + ', out_features=' + str(self.out_features) \
            + ', bias=' + str(self.bias is not None) + ')'

class GRU_module(nn.Module):
    """
    Custom GRU module that operates on sequences and handles variable sequence lengths.
    """
    def __init__(self, input_size, hidden_size, device, num_layers=1, x_mean=0,\
                 bias=True, batch_first=False, bidirectional=False, dropout=0, \
                 dropout_type='mloss', return_hidden = False):
        """
        Initialize
        
        Args:
            input_size (int): Number of expected features in the input.
            hidden_size (int): Number of features in the hidden state.
            device (torch.device): The device to which tensors will be moved.
            num_layers (int, optional): Number of recurrent layers. Default is 1.
            x_mean (float, optional): Mean value of the input. Default is 0.
            bias (bool, optional): If False, then the layer does not use bias weights. Default is True.
            batch_first (bool, optional): If True, then the input and output tensors are provided as (batch, seq, feature). Default is False.
            bidirectional (bool, optional): If True, becomes a bidirectional GRU. Default is False.
            dropout (float, optional): If non-zero, introduces a `Dropout` layer on the outputs of each GRU layer except the last layer, with dropout probability equal to dropout. Default is 0.
            dropout_type (str, optional): The type of dropout to use ('mloss' or 'standard'). Default is 'mloss'.
            return_hidden (bool, optional): If True, return the hidden state output for all time steps. Default is False.
        """
        super(GRU_module, self).__init__()

        self.hidden_size = hidden_size
        self.n_layers = num_layers
        self.bidir = bidirectional
        self.device = device
        assert type(self.bidir)==bool
        self.dropout=dropout

        self.rnn = GRUD_cell(input_size, hidden_size, device, self.n_layers, x_mean, bias = True, batch_first=True, bidirectional=self.bidir, dropout = self.dropout)
        self.init_h = torch.zeros([self.n_layers, 1, self.hidden_size], device = self.device)

    def forward2(self, inputs, masks, input_lens, deltas, init_h=None):
        """
        Perform the forward pass through the GRU module with customized handling of variable sequence lengths.

        Args:
            inputs (torch.Tensor): Input tensor of shape (batch_size, seq_len, emb_size).
            masks (torch.Tensor): Mask tensor indicating valid time steps.
            input_lens (torch.Tensor): Tensor containing lengths of input sequences.
            deltas (torch.Tensor): Delta tensor.
            init_h (torch.Tensor, optional): Initial hidden state. Default is None.

        Returns:
            torch.Tensor: Output hidden states for all time steps.
            torch.Tensor: Final hidden state for each sequence.
        """
        batch_size, seq_len, emb_size=inputs.size()

        input_lens_sorted, indices = input_lens.sort(descending=True)
        inputs_sorted = inputs.index_select(0, indices)

        mask_sorted = masks.index_select(0, indices)

        delta_sorted = deltas.index_select(0, indices)

        if init_h is None:
            init_h = self.init_h.expand(-1,batch_size,-1).contiguous()  # use learnable initial states, expanding along batches

        hids, _ = self.rnn(inputs_sorted, mask_sorted, delta_sorted, init_h)

        last_len = None

        last_id = None

        last_h_n = torch.zeros([self.n_layers, batch_size, self.hidden_size], device = self.device)

        output_list = torch.zeros([batch_size, seq_len,1*self.hidden_size], device = self.device)

        for k in range(len(input_lens_sorted)):
            if last_len is None:
                last_len = input_lens_sorted[k]
                last_id = k 
            else:
                if last_len == input_lens_sorted[k]:
                    continue
                else:
                    '''Mask, Delta, init_h'''
                    last_h_n[:,last_id:k] = hids[last_id:k, last_len - 1]

                    last_id = k
                    last_len = input_lens_sorted[k]

        last_h_n[:,last_id:k+1] = hids[last_id:k+1, last_len - 1]

        _, inv_indices = indices.sort()
        
        output_hiddens = hids[inv_indices]

        return output_hiddens, last_h_n[:,inv_indices]


    def forward(self, inputs, masks, input_lens, deltas, init_h=None):
        """
        Perform the forward pass through the GRU module with customized handling of variable sequence lengths.

        Args:
            inputs (torch.Tensor): Input tensor of shape (batch_size, seq_len, emb_size).
            masks (torch.Tensor): Mask tensor indicating valid time steps.
            input_lens (torch.Tensor): Tensor containing lengths of input sequences.
            deltas (torch.Tensor): Delta tensor.
            init_h (torch.Tensor, optional): Initial hidden state. Default is None.

        Returns:
            torch.Tensor: Output hidden states for all time steps.
            torch.Tensor: Final hidden state for each sequence.
        """
        batch_size, seq_len, emb_size=inputs.size()

        input_lens_sorted, indices = input_lens.sort(descending=True)
        inputs_sorted = inputs.index_select(0, indices)

        mask_sorted = masks.index_select(0, indices)

        delta_sorted = deltas.index_select(0, indices)

        if init_h is None:
            init_h = self.init_h.expand(-1,batch_size,-1).contiguous()  # use learnable initial states, expanding along batches

        last_len = None
        
        last_id = None
        
        last_h_n = torch.zeros([self.n_layers, batch_size, self.hidden_size], device = self.device)

        output_list = torch.zeros([batch_size, seq_len,1*self.hidden_size], device = self.device)

        for k in range(len(input_lens_sorted)):
            if last_len is None:
                last_len = input_lens_sorted[k]
                last_id = k 
            else:
                if last_len == input_lens_sorted[k]:
                    continue
                else:
                    '''Mask, Delta, init_h'''
                    hids, h_n = self.rnn(inputs_sorted[last_id:k, 0:last_len], mask_sorted[last_id:k, 0:last_len], delta_sorted[last_id:k, 0:last_len], init_h[:,last_id:k,:])

                    output_list[last_id:k, 0:last_len] = hids
                    last_h_n[:,last_id:k] = h_n

                    last_id = k
                    last_len = input_lens_sorted[k]

        hids, h_n = self.rnn(inputs_sorted[last_id:k+1, 0:last_len], mask_sorted[last_id:k+1, 0:last_len], delta_sorted[last_id:k+1, 0:last_len], init_h[:,last_id:k+1,:])

        output_list[last_id:k+1, 0:last_len] = hids
        last_h_n[:,last_id:k+1] = h_n

        _, inv_indices = indices.sort()

        output_hiddens = output_list[inv_indices]

        return output_hiddens, last_h_n[:,inv_indices]

class GRUI_cell(nn.Module):
    """
    Implementation of GRUD cell.
    Inputs:
        input_size (int): Number of expected features in the input.
        hidden_size (int): Number of features in the hidden state.
        device (torch.device): The device to which tensors will be moved.
        num_layers (int, optional): Number of recurrent layers. Default is 1.
        x_mean (float, optional): Mean value of the input. Default is 0.
        bias (bool, optional): If False, then the layer does not use bias weights. Default is True.
        batch_first (bool, optional): If True, then the input and output tensors are provided as (batch, seq, feature). Default is False.
        bidirectional (bool, optional): If True, becomes a bidirectional GRUD. Default is False.
        dropout (float, optional): If non-zero, introduces a `Dropout` layer on the outputs of each GRUD layer except the last layer, with dropout probability equal to dropout. Default is 0.
        dropout_type (str, optional): The type of dropout to use ('mloss' or 'standard'). Default is 'mloss'.
        return_hidden (bool, optional): If True, return the hidden state output for all time steps. Default is False.
    """
    def __init__(self, input_size, hidden_size, device, num_layers=1, x_mean=0,\
                 bias=True, batch_first=False, bidirectional=False, dropout=0, dropout_type='mloss', return_hidden = False):

        super(GRUI_cell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.return_hidden = return_hidden #controls the output, True if another GRU-D layer follows
        self.device = device

        x_mean = torch.tensor(x_mean, requires_grad = True, device = self.device)
        self.register_buffer('x_mean', x_mean)
        self.bias = bias
        self.batch_first = batch_first
        self.dropout_type = dropout_type
        self.dropout = dropout
        self.bidirectional = bidirectional
        num_directions = 2 if bidirectional else 1

        self.w_dg_x = torch.nn.Linear(input_size, hidden_size, bias=True)
        self.w_dg_h = torch.nn.Linear(input_size, hidden_size, bias = True)

        self.w_mu = torch.nn.Linear(input_size + hidden_size, hidden_size, bias = True)

        self.w_r = torch.nn.Linear(input_size + hidden_size, hidden_size, bias = True)

        self.w_h = torch.nn.Linear(input_size + hidden_size, hidden_size, bias = True)

        Hidden_State = torch.zeros(self.hidden_size, requires_grad = True, device = self.device)
        #we use buffers because pytorch will take care of pushing them to GPU for us
        self.register_buffer('Hidden_State', Hidden_State)
        self.register_buffer('X_last_obs', torch.zeros(input_size, device = self.device)) #torch.tensor(x_mean) #TODO: what to initialize last observed values with?, also check broadcasting behaviour

    # TODO: check usefulness of everything below here, just copied skeleton
        self.reset_parameters()


    def reset_parameters(self):
        """
        Reset the parameters of the GRUI cell.
        """
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            torch.nn.init.uniform_(weight, -stdv, stdv)

    def check_forward_args(self, input, hidden, batch_sizes):
        """
        Check the forward arguments compatibility.

        Args:
            input (torch.Tensor): Input tensor.
            hidden (torch.Tensor): Hidden state tensor.
            batch_sizes (torch.Tensor): Batch size tensor.
        """
        is_input_packed = batch_sizes is not None
        expected_input_dim = 2 if is_input_packed else 3
        if input.dim() != expected_input_dim:
            raise RuntimeError(
                'input must have {} dimensions, got {}'.format(
                    expected_input_dim, input.dim()))
        if self.input_size != input.size(-1):
            raise RuntimeError(
                'input.size(-1) must be equal to input_size. Expected {}, got {}'.format(
                    self.input_size, input.size(-1)))

        if is_input_packed:
            mini_batch = int(batch_sizes[0])
        else:
            mini_batch = input.size(0) if self.batch_first else input.size(1)

        num_directions = 2 if self.bidirectional else 1
        expected_hidden_size = (self.num_layers * num_directions,
                                mini_batch, self.hidden_size)

        def check_hidden_size(hx, expected_hidden_size, msg='Expected hidden size {}, got {}'):
            if tuple(hx.size()) != expected_hidden_size:
                raise RuntimeError(msg.format(expected_hidden_size, tuple(hx.size())))

        if self.mode == 'LSTM':
            check_hidden_size(hidden[0], expected_hidden_size,
                              'Expected hidden[0] size {}, got {}')
            check_hidden_size(hidden[1], expected_hidden_size,
                              'Expected hidden[1] size {}, got {}')
        else:
            check_hidden_size(hidden, expected_hidden_size)

    def extra_repr(self):
        """
        Return the extra representation of the GRUI cell.
        """
        s = '{input_size}, {hidden_size}'
        if self.num_layers != 1:
            s += ', num_layers={num_layers}'
        if self.bias is not True:
            s += ', bias={bias}'
        if self.batch_first is not False:
            s += ', batch_first={batch_first}'
        if self.dropout != 0:
            s += ', dropout={dropout}'
        if self.bidirectional is not False:
            s += ', bidirectional={bidirectional}'
        return s.format(**self.__dict__)

    @property
    def _flat_weights(self):
        return list(self._parameters.values())


    def forward(self, input, Mask, Delta, init_h = None):
        """
        Perform the forward pass through the GRUI cell.

        Args:
            input (torch.Tensor): Input tensor of shape (3, seq_len, input_size) where the first dimension includes data, mask, and deltat.
            Mask (torch.Tensor): Mask tensor indicating valid time steps.
            Delta (torch.Tensor): Delta tensor.
            init_h (torch.Tensor, optional): Initial hidden state. Default is None.

        Returns:
            torch.Tensor: Output hidden states for all time steps.
            torch.Tensor: Final hidden state.
        """
        X = input

        step_size = X.size(1) # 49
        
        output = None
        
        if init_h is None:
            h = getattr(self, 'Hidden_State')
        else:
            h = init_h
        x_mean = getattr(self, 'x_mean')
        x_last_obsv = getattr(self, 'X_last_obs')

        hidden_tensor = torch.empty(X.size()[0], X.size()[1], self.hidden_size, dtype=X.dtype, device = self.device)

        #iterate over seq
        for timestep in range(X.size()[1]):

            x = (X[:,timestep,:]).unsqueeze(0)
            m = (Mask[:,timestep,:]).unsqueeze(0)
            d = (Delta[:,timestep,:]).unsqueeze(0)

            #(4)
            gamma_x = torch.exp(-1* torch.nn.functional.relu( self.w_dg_x(d) ))

            h_prime = gamma_x*h
            #(5)
            #standard mult handles case correctly, this should work - maybe broadcast x_mean, seems to be taking care of that anyway

            #update x_last_obsv
            x_last_obsv = torch.where(m>0,x,x_last_obsv)

            gate = F.sigmoid(self.w_mu(torch.cat([h_prime, x], -1)))

            reset = F.sigmoid(self.w_r(torch.cat([h_prime, x], -1)))

            h_bar = F.tanh(self.w_h(torch.cat([h_prime*reset,x], -1)))

            h = (1-gate)*h + gate*h_bar

            dropout = torch.nn.Dropout(p=self.dropout)
            h = dropout(h)

            hidden_tensor[:,timestep,:] = h

        output = hidden_tensor

        return output, h


class GRUI_module(nn.Module):
    """
    Implementation of the GRUI module.

    Inputs:
        input_size (int): Number of expected features in the input.
        hidden_size (int): Number of features in the hidden state.
        device (torch.device): The device to which tensors will be moved.
        num_layers (int, optional): Number of recurrent layers. Default is 1.
        x_mean (float, optional): Mean value of the input. Default is 0.
        bias (bool, optional): If False, then the layer does not use bias weights. Default is True.
        batch_first (bool, optional): If True, then the input and output tensors are provided as (batch, seq, feature). Default is False.
        bidirectional (bool, optional): If True, becomes a bidirectional GRUD. Default is False.
        dropout (float, optional): If non-zero, introduces a `Dropout` layer on the outputs of each GRUD layer except the last layer, with dropout probability equal to dropout. Default is 0.
        dropout_type (str, optional): The type of dropout to use ('mloss' or 'standard'). Default is 'mloss'.
        return_hidden (bool, optional): If True, return the hidden state output for all time steps. Default is False.
    """
    def __init__(self, input_size, hidden_size, device, num_layers=1, x_mean=0,\
                 bias=True, batch_first=False, bidirectional=False, dropout=0, dropout_type='mloss', return_hidden = False):

        super(GRUI_module, self).__init__()

        self.hidden_size = hidden_size
        self.n_layers = num_layers
        self.bidir = bidirectional
        self.device = device
        assert type(self.bidir)==bool
        self.dropout=dropout
        self.rnn = GRUI_cell(input_size, hidden_size, device, self.n_layers, x_mean, bias = True, batch_first=True, bidirectional=self.bidir, dropout = self.dropout)
        self.init_h = torch.zeros([self.n_layers, 1, self.hidden_size], device = self.device)

    def forward2(self, inputs, masks, input_lens, deltas, init_h=None):
        """
        Perform the forward pass through the GRUI module.

        Args:
            inputs (torch.Tensor): Input tensor.
            masks (torch.Tensor): Mask tensor indicating valid time steps.
            input_lens (torch.Tensor): Input sequence lengths.
            deltas (torch.Tensor): Delta tensor.
            init_h (torch.Tensor, optional): Initial hidden state. Default is None.

        Returns:
            torch.Tensor: Output hidden states for all time steps.
            torch.Tensor: Final hidden state.
        """
        batch_size, seq_len, emb_size=inputs.size()

        input_lens_sorted, indices = input_lens.sort(descending=True)
        inputs_sorted = inputs.index_select(0, indices)

        mask_sorted = masks.index_select(0, indices)

        delta_sorted = deltas.index_select(0, indices)

        if init_h is None:
            init_h = self.init_h.expand(-1, batch_size,-1).contiguous()  # use learnable initial states, expanding along batches

        hids, _ = self.rnn(inputs_sorted, mask_sorted, delta_sorted, init_h)

        last_len = None

        last_id = None

        last_h_n = torch.zeros([self.n_layers, batch_size, self.hidden_size], device = self.device)

        output_list = torch.zeros([batch_size, seq_len,1*self.hidden_size], device = self.device)

        for k in range(len(input_lens_sorted)):
            if last_len is None:
                last_len = input_lens_sorted[k]
                last_id = k 
            else:
                if last_len == input_lens_sorted[k]:
                    continue
                else:
                    '''Mask, Delta, init_h'''
                    last_h_n[:,last_id:k] = hids[last_id:k, last_len - 1]

                    last_id = k
                    last_len = input_lens_sorted[k]

        last_h_n[:,last_id:k+1] = hids[last_id:k+1, last_len - 1]

        _, inv_indices = indices.sort()
        
        output_hiddens = hids[inv_indices]

        return output_hiddens, last_h_n[:,inv_indices]#, last_c_n[:, inv_indices])

    def forward(self, inputs, masks, input_lens, deltas, init_h=None):
        """
        Perform the forward pass through the GRUI module.

        Args:
            inputs (torch.Tensor): Input tensor.
            masks (torch.Tensor): Mask tensor indicating valid time steps.
            input_lens (torch.Tensor): Input sequence lengths.
            deltas (torch.Tensor): Delta tensor.
            init_h (torch.Tensor, optional): Initial hidden state. Default is None.

        Returns:
            torch.Tensor: Output hidden states for all time steps.
            torch.Tensor: Final hidden state.
        """
        batch_size, seq_len, emb_size=inputs.size()

        input_lens_sorted, indices = input_lens.sort(descending=True)
        inputs_sorted = inputs.index_select(0, indices)
        
        mask_sorted = masks.index_select(0, indices)
        
        delta_sorted = deltas.index_select(0, indices)
            
        if init_h is None:
            init_h = self.init_h.expand(-1,batch_size,-1).contiguous()  # use learnable initial states, expanding along batches
        
        last_len = None
        
        last_id = None
        
        last_h_n = torch.zeros([self.n_layers, batch_size, self.hidden_size], device = self.device)

        output_list = torch.zeros([batch_size, seq_len,1*self.hidden_size], device = self.device)
        
        for k in range(len(input_lens_sorted)):
            if last_len is None:
                last_len = input_lens_sorted[k]
                last_id = k 
            else:
                if last_len == input_lens_sorted[k]:
                    continue
                else:
                    '''Mask, Delta, init_h'''
                    hids, h_n = self.rnn(inputs_sorted[last_id:k, 0:last_len], mask_sorted[last_id:k, 0:last_len], delta_sorted[last_id:k, 0:last_len], init_h[:,last_id:k,:])
                        
                    output_list[last_id:k, 0:last_len] = hids
                    last_h_n[:,last_id:k] = h_n
                    
                    last_id = k
                    last_len = input_lens_sorted[k]

        hids, h_n = self.rnn(inputs_sorted[last_id:k+1, 0:last_len], mask_sorted[last_id:k+1, 0:last_len], delta_sorted[last_id:k+1, 0:last_len], init_h[:,last_id:k+1,:])

        output_list[last_id:k+1, 0:last_len] = hids
        last_h_n[:,last_id:k+1] = h_n


        _, inv_indices = indices.sort()

        output_hiddens = output_list[inv_indices]

        return output_hiddens, last_h_n[:,inv_indices]#, last_c_n[:, inv_indices])

class GRUD_cell(nn.Module):
    """
    Implementation of the GRUD cell.
    Inputs:
        input_size (int): Number of expected features in the input.
        hidden_size (int): Number of features in the hidden state.
        device (torch.device): The device to which tensors will be moved.
        num_layers (int, optional): Number of recurrent layers. Default is 1.
        x_mean (float, optional): Mean value of the input. Default is 0.
        bias (bool, optional): If False, then the layer does not use bias weights. Default is True.
        batch_first (bool, optional): If True, then the input and output tensors are provided as (batch, seq, feature). Default is False.
        bidirectional (bool, optional): If True, becomes a bidirectional GRUD. Default is False.
        dropout (float, optional): If non-zero, introduces a `Dropout` layer on the outputs of each GRUD layer except the last layer, with dropout probability equal to dropout. Default is 0.
        dropout_type (str, optional): The type of dropout to use ('mloss' or 'standard'). Default is 'mloss'.
        return_hidden (bool, optional): If True, return the hidden state output for all time steps. Default is False.
    """
    def __init__(self, input_size, hidden_size, device, num_layers=1, x_mean=0,\
                 bias=True, batch_first=False, bidirectional=False, dropout=0, dropout_type='mloss', return_hidden = False):

        super(GRUD_cell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.return_hidden = return_hidden  #controls the output, True if another GRU-D layer follows
        self.device = device

        x_mean = torch.tensor(x_mean, requires_grad = True, device = self.device)
        self.register_buffer('x_mean', x_mean)
        self.bias = bias
        self.batch_first = batch_first
        self.dropout_type = dropout_type
        self.dropout = dropout
        self.bidirectional = bidirectional
        num_directions = 2 if bidirectional else 1

        #set up all the operations that are needed in the forward pass
        self.w_dg_x = torch.nn.Linear(input_size,input_size, bias=True)
        self.w_dg_h = torch.nn.Linear(input_size, hidden_size, bias = True)

        self.w_xz = torch.nn.Linear(input_size, hidden_size, bias=False)
        self.w_hz = torch.nn.Linear(hidden_size, hidden_size, bias=False)
        self.w_mz = torch.nn.Linear(input_size, hidden_size, bias=True)

        self.w_xr = torch.nn.Linear(input_size, hidden_size, bias=False)
        self.w_hr = torch.nn.Linear(hidden_size, hidden_size, bias=False)
        self.w_mr = torch.nn.Linear(input_size, hidden_size, bias=False)
        self.w_xh = torch.nn.Linear(input_size, hidden_size, bias=False)
        self.w_hh = torch.nn.Linear(hidden_size, hidden_size, bias=False)
        self.w_mh = torch.nn.Linear(input_size, hidden_size, bias=True)

        Hidden_State = torch.zeros(self.hidden_size, requires_grad = True, device = self.device)
        #we use buffers because pytorch will take care of pushing them to GPU for us
        self.register_buffer('Hidden_State', Hidden_State)
        self.register_buffer('X_last_obs', torch.zeros(input_size, device = self.device)) #torch.tensor(x_mean) #TODO: what to initialize last observed values with?, also check broadcasting behaviour

    #TODO: check usefulness of everything below here, just copied skeleton
        self.reset_parameters()


    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            torch.nn.init.uniform_(weight, -stdv, stdv)

    def check_forward_args(self, input, hidden, batch_sizes):
        is_input_packed = batch_sizes is not None
        expected_input_dim = 2 if is_input_packed else 3
        if input.dim() != expected_input_dim:
            raise RuntimeError(
                'input must have {} dimensions, got {}'.format(
                    expected_input_dim, input.dim()))
        if self.input_size != input.size(-1):
            raise RuntimeError(
                'input.size(-1) must be equal to input_size. Expected {}, got {}'.format(
                    self.input_size, input.size(-1)))

        if is_input_packed:
            mini_batch = int(batch_sizes[0])
        else:
            mini_batch = input.size(0) if self.batch_first else input.size(1)

        num_directions = 2 if self.bidirectional else 1
        expected_hidden_size = (self.num_layers * num_directions,
                                mini_batch, self.hidden_size)

        def check_hidden_size(hx, expected_hidden_size, msg='Expected hidden size {}, got {}'):
            if tuple(hx.size()) != expected_hidden_size:
                raise RuntimeError(msg.format(expected_hidden_size, tuple(hx.size())))

        if self.mode == 'LSTM':
            check_hidden_size(hidden[0], expected_hidden_size,
                              'Expected hidden[0] size {}, got {}')
            check_hidden_size(hidden[1], expected_hidden_size,
                              'Expected hidden[1] size {}, got {}')
        else:
            check_hidden_size(hidden, expected_hidden_size)

    def extra_repr(self):
        s = '{input_size}, {hidden_size}'
        if self.num_layers != 1:
            s += ', num_layers={num_layers}'
        if self.bias is not True:
            s += ', bias={bias}'
        if self.batch_first is not False:
            s += ', batch_first={batch_first}'
        if self.dropout != 0:
            s += ', dropout={dropout}'
        if self.bidirectional is not False:
            s += ', bidirectional={bidirectional}'
        return s.format(**self.__dict__)

    @property
    def _flat_weights(self):
        return list(self._parameters.values())

    def forward(self, input, Mask, Delta, init_h = None):
        """
        Perform the forward pass through the GRUD cell.

        Args:
            input (torch.Tensor): Input tensor.
            Mask (torch.Tensor): Mask tensor indicating valid time steps.
            Delta (torch.Tensor): Delta tensor.
            init_h (torch.Tensor, optional): Initial hidden state. Default is None.

        Returns:
            torch.Tensor: Output hidden states for all time steps.
            torch.Tensor: Final hidden state.
        """
        X = input

        step_size = X.size(1) # 49

        output = None

        if init_h is None:
            h = getattr(self, 'Hidden_State')
        else:
            h = init_h
        #felix - buffer system from newer pytorch version
        x_mean = getattr(self, 'x_mean')
        x_last_obsv = getattr(self, 'X_last_obs')

        hidden_tensor = torch.empty(X.size()[0], X.size()[1], self.hidden_size, dtype=X.dtype, device = self.device)

        #iterate over seq
        for timestep in range(X.size()[1]):
            x = torch.squeeze(X[:,timestep,:])
            m = torch.squeeze(Mask[:,timestep,:])
            d = torch.squeeze(Delta[:,timestep,:])
            
            #(4)
            gamma_x = torch.exp(-1* torch.nn.functional.relu( self.w_dg_x(d) ))
            gamma_h = torch.exp(-1* torch.nn.functional.relu( self.w_dg_h(d) ))

            #(5)
            #standard mult handles case correctly, this should work - maybe broadcast x_mean, seems to be taking care of that anyway
            #update x_last_obsv
            x_last_obsv = torch.where(m>0,x,x_last_obsv)
            x = m * x + (1 - m) * (gamma_x * x + (1 - gamma_x) * x_mean)
            x = m * x + (1 - m) * (gamma_x * x_last_obsv + (1 - gamma_x) * x_mean)

            #(6)
            if self.dropout == 0:

                h = gamma_h*h

                z = F.sigmoid( self.w_xz(x) + self.w_hz(h) + self.w_mz(m))
                r = F.sigmoid( self.w_xr(x) + self.w_hr(h) + self.w_mr(m))

                h_tilde = F.tanh( self.w_xh(x) + self.w_hh( r*h ) + self.w_mh(m))

                h = (1 - z) * h + z * h_tilde

            #TODO: not adapted yet
            elif self.dropout_type == 'Moon':
                '''
                RNNDROP: a novel dropout for rnn in asr(2015)
                '''
                h = gamma_h * h

                z = F.sigmoid((self.w_xz(x) + self.w_hz(h) + self.w_mz(m)))
                r = F.sigmoid((self.w_xr(x) + self.w_hr(h) + self.w_mr(m)))

                h_tilde = F.tanh((self.w_xh(x) + self.w_hh(r * h) + self.w_mh(m)))

                h = (1 - z) * h + z * h_tilde
                dropout = torch.nn.Dropout(p=self.dropout)
                h = dropout(h)

            elif self.dropout_type == 'Gal':
                '''
                A Theoretically grounded application of dropout in recurrent neural networks(2015)
                '''
                dropout = torch.nn.Dropout(p=self.dropout)
                h = dropout(h)

                h = gamma_h * h

                z = F.sigmoid((self.w_xz(x) + self.w_hz(h) + self.w_mz(m)))
                r = F.sigmoid((self.w_xr(x) + self.w_hr(h) + self.w_mr(m)))
                h_tilde = F.tanh((self.w_xh(x) + self.w_hh(r * h) + self.w_mh(m)))

                h = (1 - z) * h + z * h_tilde

            elif self.dropout_type == 'mloss':
                '''
                recurrent dropout without memory loss arXiv 1603.05118
                g = h_tilde, p = the probability to not drop a neuron
                '''
                h = gamma_h*h
                z = F.sigmoid( self.w_xz(x) + self.w_hz(h) + self.w_mz(m))
                r = F.sigmoid( self.w_xr(x) + self.w_hr(h) + self.w_mr(m))

                dropout = torch.nn.Dropout(p=self.dropout)
                h_tilde = F.tanh( self.w_xh(x) + self.w_hh( r*h ) + self.w_mh(m))

                h = (1 - z) * h + z * h_tilde
                h = dropout(h)

            else:
                h = gamma_h * h

                z = F.sigmoid((self.w_xz(x) + self.w_hz(h) + self.w_mz(m)))
                r = F.sigmoid((self.w_xr(x) + self.w_hr(h) +self.w_mr(m)))
                h_tilde = F.tanh((self.w_xh(x) + self.w_hh((r * h)) + self.w_mh(m)))

                h = (1 - z) * h + z * h_tilde

            hidden_tensor[:,timestep,:] = h

        output = hidden_tensor

        return output, h


class TimeLSTM_module(nn.Module):
    """
    Implementation of the TimeLSTM module.

    Inputs:
        input_size (int): Number of expected features in the input.
        hidden_size (int): Number of features in the hidden state.
        dropout (float): Dropout rate.
        device (torch.device): The device to which tensors will be moved.
        bidirectional (bool, optional): If True, becomes a bidirectional TimeLSTM. Default is False.
    """
    def __init__(self, input_size, hidden_size, dropout, device, bidirectional=False):
        super(TimeLSTM_module, self).__init__()
        self.hidden_size = hidden_size
        self.input_size = input_size

        self.device = device
        self.dropout=dropout
        self.init_h = torch.zeros([self.hidden_size], device = self.device)
        self.init_c = torch.zeros([self.hidden_size], device = self.device)
        self.lstm = TimeLSTMCell(input_size, hidden_size, dropout, device, bidirectional)
        
    def forward(self, inputs, time_stamps, input_lens, init_h=None, init_c=None):
        """
        Perform the forward pass through the TimeLSTM module.

        Args:
            inputs (torch.Tensor): Input tensor.
            time_stamps (torch.Tensor): Time stamps tensor.
            input_lens (torch.Tensor): Input sequence lengths.
            init_h (torch.Tensor, optional): Initial hidden state. Default is None.
            init_c (torch.Tensor, optional): Initial cell state. Default is None.

        Returns:
            torch.Tensor: Output hidden states for all time steps.
            torch.Tensor: Final hidden state.
            torch.Tensor: Final cell state.
        """
        batch_size = inputs.shape[0]

        seq_len = inputs.shape[1]

        if init_h is None:
            init_h = self.init_h.expand(batch_size, self.init_h.shape[0])

        if init_c is None:
            init_c = self.init_c.expand(batch_size, self.init_h.shape[0])

        input_lens_sorted, indices = input_lens.sort(descending=True)
        inputs_sorted = inputs.index_select(0, indices)

        time_stamps_sorted = time_stamps.index_select(0, indices)

        last_len = None

        last_id = None

        last_h_n = torch.zeros([batch_size, self.hidden_size], device = self.device)

        last_c_n = torch.zeros([batch_size, self.hidden_size], device = self.device)

        output_list = torch.zeros([batch_size, seq_len,1*self.hidden_size], device = self.device)

        for k in range(len(input_lens_sorted)):
            if last_len is None:
                last_len = input_lens_sorted[k]
                last_id = k 
            else:
                if last_len == input_lens_sorted[k]:
                    continue
                else:
                    '''Mask, Delta, init_h'''
                    hids, h_n, c_n = self.lstm(inputs_sorted[last_id:k, 0:last_len], time_stamps_sorted[last_id:k, 0:last_len], init_h[last_id:k,:], init_c[last_id:k,:])
                    output_list[last_id:k, 0:last_len] = hids
                    last_h_n[last_id:k] = h_n
                    last_c_n[last_id:k] = c_n

                    last_id = k
                    last_len = input_lens_sorted[k]

        hids, h_n, c_n = self.lstm(inputs_sorted[last_id:k+1, 0:last_len], time_stamps_sorted[last_id:k+1, 0:last_len], init_h[last_id:k+1,:], init_c[last_id:k+1,:])
            
        output_list[last_id:k+1, 0:last_len] = hids
        last_h_n[last_id:k+1] = h_n
        last_c_n[last_id:k+1] = c_n

        _, inv_indices = indices.sort()

        output_hiddens = output_list[inv_indices]

        return output_hiddens, last_h_n[inv_indices], last_c_n[inv_indices]


    def forward2(self, inputs, time_stamps, input_lens, init_h=None, init_c=None):
        """
        Perform an alternate forward pass through the TimeLSTM module.

        Args:
            inputs (torch.Tensor): Input tensor.
            time_stamps (torch.Tensor): Time stamps tensor.
            input_lens (torch.Tensor): Input sequence lengths.
            init_h (torch.Tensor, optional): Initial hidden state. Default is None.
            init_c (torch.Tensor, optional): Initial cell state. Default is None.

        Returns:
            torch.Tensor: Output hidden states for all time steps.
            torch.Tensor: Final hidden state.
            torch.Tensor: Final cell state.
        """
        batch_size = inputs.shape[0]

        seq_len = inputs.shape[1]

        if init_h is None:
            init_h = self.init_h.expand(batch_size, self.init_h.shape[0])

        if init_c is None:
            init_c = self.init_c.expand(batch_size, self.init_h.shape[0])

        input_lens_sorted, indices = input_lens.sort(descending=True)
        inputs_sorted = inputs.index_select(0, indices)
        time_stamps_sorted = time_stamps.index_select(0, indices)

        h_outputs, c_outputs = self.lstm.forward2(inputs_sorted, time_stamps_sorted, init_h, init_c)

        last_len = None

        last_id = None

        last_h_n = torch.zeros([batch_size, self.hidden_size], device = self.device)

        last_c_n = torch.zeros([batch_size, self.hidden_size], device = self.device)

        output_list = torch.zeros([batch_size, seq_len,1*self.hidden_size], device = self.device)

        for k in range(len(input_lens_sorted)):
            if last_len is None:
                last_len = input_lens_sorted[k]
                last_id = k 
            else:
                if last_len == input_lens_sorted[k]:
                    continue
                else:
                    '''Mask, Delta, init_h'''

                    last_h_n[last_id:k] = h_outputs[last_id:k,last_len-1]
                    last_c_n[last_id:k] = c_outputs[last_id:k,last_len-1]

                    last_id = k
                    last_len = input_lens_sorted[k]

        output_list = h_outputs
        last_h_n[last_id:k+1] = h_outputs[last_id:k+1,last_len-1]
        last_c_n[last_id:k+1] = c_outputs[last_id:k+1,last_len-1]

        _, inv_indices = indices.sort()

        output_hiddens = output_list[inv_indices]

        return output_hiddens, last_h_n[inv_indices], last_c_n[inv_indices]


class TimeLSTMCell(nn.Module):
    """
    Implementation of the TimeLSTM cell.

    Inputs:
        input_size (int): Number of expected features in the input.
        hidden_size (int): Number of features in the hidden state.
        dropout (float): Dropout rate.
        device (torch.device): The device to which tensors will be moved.
        bidirectional (bool, optional): If True, becomes a bidirectional TimeLSTM. Default is False.
    """
    def __init__(self, input_size, hidden_size, dropout, device, bidirectional=False):
        # assumes that batch_first is always true
        super(TimeLSTMCell, self).__init__()
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.device = device
        self.W_all = nn.Linear(hidden_size, hidden_size * 4)
        self.U_all = nn.Linear(input_size, hidden_size * 4)
        self.W_d = nn.Linear(hidden_size, hidden_size)
        self.bidirectional = bidirectional

    def elapse_function(self, delta_t):
        """
        Calculate the elapse function based on the given delta_t.

        Args:
            delta_t (torch.Tensor): Delta time.

        Returns:
            torch.Tensor: Elapse function values.
        """
        return 1.0/(torch.log(np.e + delta_t))

    def forward2(self, inputs, timestamps, init_h=None, init_c=None, reverse=False):
        """
        Perform the forward pass through the TimeLSTM cell using an alternate implementation.

        Args:
            inputs (torch.Tensor): Input tensor.
            timestamps (torch.Tensor): Timestamps tensor.
            init_h (torch.Tensor, optional): Initial hidden state. Default is None.
            init_c (torch.Tensor, optional): Initial cell state. Default is None.
            reverse (bool, optional): If True, reverse the sequence. Default is False.

        Returns:
            torch.Tensor: Output hidden states for all time steps.
            torch.Tensor: Output cell states for all time steps.
        """
        # inputs: [b, seq, embed]
        # h: [b, hid]
        # c: [b, hid]
        b, seq, embed = inputs.size()
        if init_h is None:
            h = torch.zeros(b, self.hidden_size, requires_grad=False, device = self.device)
        else:
            h = init_h

        if init_c is None:
            c = torch.zeros(b, self.hidden_size, requires_grad=False, device = self.device)
        else:
            c = init_c
        outputs = []
        c_outputs = []

        for s in range(seq):
            c_s1 = F.tanh(self.W_d(c))
            c_s2 = c_s1 * self.elapse_function(timestamps[:, s]).expand_as(c_s1)
            c_l = c - c_s1
            c_adj = c_l + c_s2
            outs = self.W_all(h) + self.U_all(inputs[:, s])
            f, i, o, c_tmp = torch.chunk(outs, 4, 1)
            f = F.sigmoid(f)
            i = F.sigmoid(i)
            o = F.sigmoid(o)
            c_tmp = F.tanh(c_tmp)
            c = f * c_adj + i * c_tmp
            h = o * F.tanh(c)
            outputs.append(h)
            c_outputs.append(c)

        if reverse:
            outputs.reverse()
        outputs = torch.stack(outputs, 1)

        c_outputs = torch.stack(c_outputs, 1)

        return outputs, c_outputs

    def forward(self, inputs, timestamps, init_h=None, init_c=None, reverse=False):
        """
        Perform the forward pass through the TimeLSTM cell.

        Args:
            inputs (torch.Tensor): Input tensor.
            timestamps (torch.Tensor): Timestamps tensor.
            init_h (torch.Tensor, optional): Initial hidden state. Default is None.
            init_c (torch.Tensor, optional): Initial cell state. Default is None.
            reverse (bool, optional): If True, reverse the sequence. Default is False.

        Returns:
            torch.Tensor: Output hidden states for all time steps.
            torch.Tensor: Final hidden state.
            torch.Tensor: Final cell state.
        """
        # inputs: [b, seq, embed]
        # h: [b, hid]
        # c: [b, hid]
        b, seq, embed = inputs.size()
        if init_h is None:
            h = torch.zeros(b, self.hidden_size, requires_grad=False, device=self.device)
        else:
            h = init_h

        if init_c is None:
            c = torch.zeros(b, self.hidden_size, requires_grad=False, device=self.device)
        else:
            c = init_c

        outputs = []
        for s in range(seq):
            c_s1 = F.tanh(self.W_d(c))
            c_s2 = c_s1 * self.elapse_function(timestamps[:, s]).expand_as(c_s1)
            c_l = c - c_s1
            c_adj = c_l + c_s2
            outs = self.W_all(h) + self.U_all(inputs[:, s])
            f, i, o, c_tmp = torch.chunk(outs, 4, 1)
            f = F.sigmoid(f)
            i = F.sigmoid(i)
            o = F.sigmoid(o)
            c_tmp = F.tanh(c_tmp)
            c = f * c_adj + i * c_tmp
            h = o * F.tanh(c)
            outputs.append(h)
        if reverse:
            outputs.reverse()
        outputs = torch.stack(outputs, 1)
        return outputs, h, c
