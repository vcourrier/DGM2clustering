
import numpy as np
import sys, os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam

from modules import PostNet_cluster, Encoder_cluster
from imputation.inverse_distance_weighting import *
from lib.utils import *

sys.path.append(os.path.dirname(os.path.abspath(__file__))+'/imputation')
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/lib')
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.abspath(__file__))


data_folder = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + "/" + data_dir


class DGM2_L(nn.Module):
    """
    The Deep Markov Model
    """
    def __init__(self, config ):
        """
        Initialize the DGM2-L model with the provided configuration.

        Args:
            config (dict): Configuration parameters for the model.
        """
        super(DGM2_L, self).__init__()

        self.input_dim = config['input_dim']
        self.h_dim = config['h_dim']
        self.s_dim = config['s_dim']
        self.centroid_max = config['d']
        self.device = config['device']
        self.dropout = config['dropout']
        self.e_dim = config['e_dim']
        self.lstm_layer = 1
        self.cluster_num = config['cluster_num']
        self.h_0 = torch.zeros(self.h_dim, device = config['device'])
        self.s_0 = torch.zeros(self.s_dim, device = config['device'])
        self.x_std = config['x_std']
        self.sample_times = config['sampling_times']
        self.use_gate = config['use_gate']
        self.evaluate = False
        self.loss_on_missing = False  # config['loss_missing']
        self.transfer_prob = False
        self.gaussian_prior_coeff = config['gaussian']
        self.pre_impute = True
        self.cluster_mask = False
        self.clip_norm = config['clip_norm']

        self.emitter_z = nn.Sequential(  # Parameterizes the bernoulli observation likelihood `p(x_t|z_t)`
            nn.Linear(self.h_dim, self.cluster_num),
            nn.Dropout(p = self.dropout)
            )
        self.block = 'LSTM'
        self.lstm_latent = True
        self.z_dim = self.input_dim

        if config['is_missing']:
            self.impute = IDW_impute(self.input_dim, self.device)
        self.is_missing = config['is_missing'] 
        
        self.use_mask = True
        if self.use_mask:
            self.x_encoder = Encoder_cluster(
                2*self.z_dim, self.s_dim, self.dropout, self.device, bidir = False, block = self.block, n_layers =self.lstm_layer
                )
        else:
            self.x_encoder = Encoder_cluster(
                self.z_dim, self.s_dim, self.dropout, self.device, bidir = False, block = self.block, n_layers =self.lstm_layer
                )
        
        if self.transfer_prob:
            self.postnet = PostNet_cluster(
                self.cluster_num, self.s_dim, self.cluster_num, self.dropout, self.sample_times, bidirt = self.x_encoder.bidir
                )
        else:
            self.postnet = PostNet_cluster(
                self.z_dim, self.s_dim, self.cluster_num, self.dropout, self.sample_times, bidirt = self.x_encoder.bidir
                )

        if self.transfer_prob:
            if self.block == 'GRU':
                self.trans = torch.nn.GRU(
                    self.cluster_num, self.h_dim, num_layers = self.lstm_layer, dropout = self.dropout, batch_first = True
                    )
            else:
                self.trans = torch.nn.LSTM(
                    self.cluster_num, self.h_dim, num_layers = self.lstm_layer, dropout = self.dropout, batch_first = True
                    )
        
        else:
            if self.block == 'GRU':
                self.trans = torch.nn.GRU(
                    self.z_dim, self.h_dim, num_layers = self.lstm_layer, dropout = self.dropout, batch_first = True
                    )
            else:
                self.trans = torch.nn.LSTM(
                    self.z_dim, self.h_dim, num_layers = self.lstm_layer,dropout = self.dropout, batch_first = True
                    )

        if self.use_gate:
            self.gate_func = nn.Sequential(
                nn.Linear((1+self.x_encoder.bidir)*self.s_dim, 1),
                nn.Dropout(p = self.dropout)
                )

        self.z_dim = self.input_dim

        self.phi_table = torch.zeros([self.z_dim, self.cluster_num], dtype = torch.float, device = config['device']) 
         
        self.phi_table = torch.nn.Parameter(self.phi_table)

        if self.transfer_prob:
            self.z_q_0 = torch.zeros(self.cluster_num, device = config['device'])
        else:
            self.z_q_0 = torch.zeros(self.z_dim, device = config['device'])
        
        self.optimizer = Adam(self.parameters(), lr=config['lr'], betas= (config['beta1'], config['beta2']))


    def init_phi_table(self, init_values, is_tensor):
        """
        Initialize the phi_table with the given initial values.

        Args:
            init_values (numpy.ndarray or torch.Tensor): Initial values for the phi_table.
            is_tensor (bool): Whether init_values is a torch.Tensor or not.
        """
        if not is_tensor:
            self.phi_table.data.copy_(torch.t(torch.from_numpy(init_values)))
        else:
            self.phi_table.data.copy_(init_values)
         
        torch.save(self.phi_table, data_folder + '/' + output_dir + 'init_phi_table')

    def calculate_gaussian_coeff(self, h_now):
        """
        Calculate the Gaussian coefficient based on the gate function and model's configuration.

        Args:
            h_now (torch.Tensor): Hidden state representation.

        Returns:
            torch.Tensor: Computed Gaussian coefficient.
        """
        if self.config['use_gate']:
            return (self.config['gaussian'] + torch.sigmoid(self.gate_func(h_now[-1]))) / 2
        else:
            return self.config['gaussian']

    def generate_z(self, z_category, prior_cluster_probs, h_now=None):
        """
        Generate latent variable phi_z using the given z_category and prior_cluster_probs.

        Args:
            z_category (torch.Tensor): Categorical representation of latent variable.
            prior_cluster_probs (torch.Tensor): Prior probabilities of cluster categories.
            h_now (torch.Tensor, optional): Current hidden state representation.

        Returns:
            phi_z (torch.Tensor): Generated phi_z.
            z_representation (torch.Tensor): Latent variable representation.
        """
        curr_gaussian_coeff = self.calculate_gaussian_coeff(h_now)

        z_category = (1 - curr_gaussian_coeff)*z_category + curr_gaussian_coeff*prior_cluster_probs

        z_representation = z_category  # .view(z_category.shape[1], z_category.shape[2])

        phi_z = torch.t(torch.mm(self.phi_table, torch.t(z_representation)))

        return phi_z, z_representation
    
    
#    def compute_cluster_obj(self, all_distances, all_probabs, T_max, x_lens, x_dim):
#        '''all_probabs: 1*cluster_num'''
#        
#        
#        all_probabs_copy = all_probabs.view(1, 1, all_probabs.shape[0])
#        
#        all_probabs_copy = all_probabs_copy.repeat(all_distances.shape[0], T_max, 1)
#        
#        '''batch_size*T_max*cluster_num'''
#        
#        cluster_obj = torch.sum(all_distances**2*all_probabs_copy/((self.x_std**2)*(torch.sum(x_lens)*all_probabs.shape[0]*x_dim))) + 2*np.log(self.x_std) + 2*np.log(np.sqrt(2*np.pi))
#
#        return torch.sum(all_distances**2*all_probabs_copy/((self.x_std**2)*(torch.sum(x_lens)*all_probabs.shape[0]*x_dim))), cluster_obj
#    
#    
#    def compute_cluster_obj_full2(self, all_distances, all_probabs, T_max, x_masks, x_lens):
#        
#        '''all_distances:: T_max, self.cluster_num, batch_size, input_dim'''
#        
#        '''all_probabs: 1*cluster_num'''
#        
#        
#        all_probabs_copy = all_probabs.view(1, all_probabs.shape[0], 1, 1)
#        
#        all_probabs_copy = all_probabs_copy.repeat(T_max, 1, all_distances.shape[2], all_distances.shape[3])
#        
#        
#        all_probabs_copy2 = all_probabs.view(1, all_probabs.shape[0], 1)
#        
#        all_probabs_copy2 = all_probabs_copy2.repeat(T_max, 1, all_distances.shape[2])
#        
#        all_masks = torch.transpose(x_masks, 0, 1)
#        
#        all_masks = all_masks.view(all_masks.shape[0],1, all_masks.shape[1], all_masks.shape[2])
#        
#        all_masks_copy = all_masks.repeat(1, all_probabs.shape[0], 1, 1)
#
#        cluster_obj = 0.5*(torch.sum((torch.sum(all_distances*all_masks_copy*all_probabs_copy, 1)/(self.x_std**2)/torch.sum(all_masks))) + 2*np.log(self.x_std) + 2*np.log(np.sqrt(2*np.pi)))
#        
#        return 0.5*torch.sum((all_distances*all_masks_copy/(self.x_std**2)*all_probabs_copy/torch.sum(all_masks_copy))), cluster_obj
#
#    def compute_distance_per_cluster(self, x_t):
#        
#        '''cluster_num*batch_size*dim'''
#        
#        curr_x_t = x_t.repeat(self.cluster_num, 1, 1)
#        
#        phi_table_transpose = torch.t(self.phi_table)
#        
#        phi_table_transpose = phi_table_transpose.reshape(phi_table_transpose.shape[0], 1, phi_table_transpose.shape[1])
#        
#        phi_table_transpose = phi_table_transpose.repeat(1,x_t.shape[0], 1)
#        
#        '''cluster_num*batch_size*dim'''
#        
#        all_distances = torch.norm(curr_x_t - phi_table_transpose, dim=2)
#        return torch.t(all_distances)
##         '''batch_size * cluster_num'''
#
#
#    def compute_distance_per_cluster_all(self, x_t, x_mask):
#        
#        '''cluster_num*batch_size*dim'''
#        
#        curr_x_t = x_t.repeat(self.cluster_num, 1, 1)
#        
#        curr_x_mask = x_mask.repeat(self.cluster_num, 1, 1)
#        
#        phi_table_transpose = torch.t(self.phi_table)
#        
#        phi_table_transpose = phi_table_transpose.reshape(phi_table_transpose.shape[0], 1, phi_table_transpose.shape[1])
#        
#        phi_table_transpose = phi_table_transpose.repeat(1,x_t.shape[0], 1)
#        
#        '''cluster_num*batch_size*dim'''
#        all_distances = (curr_x_t*curr_x_mask - phi_table_transpose*curr_x_mask)**2
#        
#        return all_distances
#         '''batch_size * cluster_num'''
        
    def calculate_imputation_losses(self, origin_x_to_pred, origin_x, x_to_predict_new_mask, imputed_x2, objective):
        """
        Calculate and print imputation losses based on the provided data and imputed values.

        Args:
            origin_x_to_pred (torch.Tensor): Original data for imputation.
            origin_x (torch.Tensor): Original data for comparison.
            x_to_predict_new_mask (torch.Tensor): Mask for new predictions.
            imputed_x2 (torch.Tensor): Imputed values.
            objective (str): Name of the objective for logging.

        Returns:
            tuple: Tuple containing imputation losses and their variations.
        """
        imputed_loss = torch.sum((torch.abs(origin_x_to_pred - origin_x)*(1-x_to_predict_new_mask)))/torch.sum(1-x_to_predict_new_mask)
        imputed_loss2 = torch.sum((torch.abs(origin_x_to_pred - imputed_x2)*(1-x_to_predict_new_mask)))/torch.sum(1-x_to_predict_new_mask)
        imputed_mse_loss = torch.sqrt(torch.sum((((origin_x_to_pred - origin_x)**2)*(1-x_to_predict_new_mask)))/torch.sum(1-x_to_predict_new_mask))
        imputed_mse_loss2 = torch.sqrt(torch.sum((((origin_x_to_pred - imputed_x2)**2)*(1-x_to_predict_new_mask)))/torch.sum(1-x_to_predict_new_mask))
        
        print(objective + ' imputation rmse loss::', imputed_mse_loss)
        print(objective + ' imputation rmse loss 2::', imputed_mse_loss2)
        print(objective + ' imputation mae loss::', imputed_loss)
        print(objective + ' imputation mae loss 2::', imputed_loss2)
        
        return imputed_loss, imputed_loss2, imputed_mse_loss, imputed_mse_loss


    def sample_x(self, z_prob):
        """
        Sample x values based on the provided z_prob.

        Args:
            z_prob (torch.Tensor): Probabilities for latent variable z.

        Returns:
            torch.Tensor: Sampled x values.
        """
        sampled_x = torch.multinomial(z_prob, 1000, replacement = True)
        
        selected_phi_z = self.phi_table[:,sampled_x]
        
        epsilon = torch.randn(selected_phi_z.shape, device=selected_phi_z.device)
        
        std = self.x_std
        
        avg_selected_phi_z = torch.mean(selected_phi_z + std*epsilon, -1)

        return torch.t(avg_selected_phi_z)


    def generate_x(self, phi_z):
        """
        Generate x values based on the provided phi_z.

        Args:
            phi_z (torch.Tensor): Latent variable representation.

        Returns:
            tuple: Tuple containing mean, logvar, and logit_x_t.
        """
        mean = phi_z

        std = self.x_std*torch.ones_like(mean, device = self.device)

        logvar = 2*torch.log(std)

        return mean, logvar, mean


    def kl_div(self, cat_1, cat_2):
        """
        Compute the Kullback-Leibler divergence between two categorical distributions.

        Args:
            cat_1 (torch.Tensor): First categorical distribution.
            cat_2 (torch.Tensor): Second categorical distribution.

        Returns:
            torch.Tensor: KL divergence values.
        """
        epsilon = 1e-5*torch.ones_like(cat_1)
        kl_div = torch.sum((cat_1+epsilon)*torch.log((cat_1 + epsilon)/(cat_2+epsilon)), 1)
        return kl_div


    def entropy(self, cat):
        """
        Compute the entropy of a categorical distribution.

        Args:
            cat (torch.Tensor): Categorical distribution.

        Returns:
            torch.Tensor: Entropy values.
        """
        epsilon = 1e-5*torch.ones_like(cat)
        kl_div = -torch.sum(cat*torch.log(cat+epsilon), 1)
        return kl_div

    def compute_gaussian_probs0(self, x, mean, logvar, mask):
        """
        Compute Gaussian probabilities based on mean, logvar, and mask.

        Args:
            x (torch.Tensor): Input data.
            mean (torch.Tensor): Mean values.
            logvar (torch.Tensor): Log variance values.
            mask (torch.Tensor): Mask for valid values.

        Returns:
            tuple: Tuple containing computed probabilities and differences.
        """
        std = torch.exp(0.5 * logvar)

        prob = 0.5*(((x - mean)/std)**2 + logvar + 2*np.log(np.sqrt(2*np.pi)))  # + torch.log((std*np.sqrt(2*np.pi)))
        
        return prob*mask, (x - mean)**2*mask


    def compute_rec_loss(self, joint_probs, prob_sums, full_curr_rnn_input, x_t, x_t_mask, h_now = None, curr_z_t_category_infer=None):
        """
        Compute reconstruction loss based on various inputs.

        Args:
            joint_probs (torch.Tensor): Joint probabilities.
            prob_sums (torch.Tensor): Sum of probabilities.
            full_curr_rnn_input (torch.Tensor): Full RNN input.
            x_t (torch.Tensor): Current x values.
            x_t_mask (torch.Tensor): Mask for x values.
            h_now (torch.Tensor, optional): Current hidden state representation.
            curr_z_t_category_infer (torch.Tensor, optional): Current inferred latent category.

        Returns:
            tuple: Tuple containing computed losses and components.
        """
        phi_table_extend = (torch.t(self.phi_table)).clone()

        phi_table_extend = phi_table_extend.view(1, self.phi_table.shape[1], self.phi_table.shape[0])

        phi_table_extend = phi_table_extend.repeat(self.cluster_num, 1, 1) 

        phi_z_infer_full = torch.bmm(full_curr_rnn_input, phi_table_extend)
        mean_full, logvar_full, logit_x_t_full = self.generate_x(phi_z_infer_full)

        x_t_full = x_t.view(1, x_t.shape[0], x_t.shape[1])

        x_t_full = x_t_full.repeat(self.cluster_num, 1, 1)

        x_t_mask_full = x_t_mask.view(1, x_t_mask.shape[0], x_t_mask.shape[1])
        
        x_t_mask_full = x_t_mask_full.repeat(self.cluster_num, 1, 1)

        curr_full_rec_loss, curr_distances = self.compute_gaussian_probs0(x_t_full, mean_full, logvar_full, x_t_mask_full)

        curr_gaussian_coeff = self.calculate_gaussian_coeff(h_now)

        full_rec_loss1 = torch.sum(curr_full_rec_loss*(1-curr_gaussian_coeff)*torch.t(joint_probs).unsqueeze(-1), 0)
        
        full_rec_loss2 = torch.sum(curr_full_rec_loss*(curr_gaussian_coeff)*prob_sums.view(prob_sums.shape[0], 1, 1), 0)
        
        l2_norm_loss = full_rec_loss1/(1-curr_gaussian_coeff)
        
        cluster_loss = full_rec_loss2/curr_gaussian_coeff

        if self.cluster_mask:

            full_logit_x_t = torch.mm((1-curr_gaussian_coeff)*curr_z_t_category_infer + curr_gaussian_coeff*prob_sums.view(1,-1), torch.t(self.phi_table))[:, 0:x_t.shape[-1]]
        else:
            full_logit_x_t = torch.mm((1-curr_gaussian_coeff)*curr_z_t_category_infer + curr_gaussian_coeff*prob_sums.view(1,-1), torch.t(self.phi_table))

        return full_rec_loss1, full_rec_loss2, full_logit_x_t, curr_full_rec_loss, l2_norm_loss, cluster_loss
    '''x, origin_x, x_mask, T_max, x_to_predict, origin_x_to_pred, x_to_predict_mask, tp_to_predict, is_GPU, device'''        


    def update_joint_probability2(self, joint_probs, curr_rnn_output, batch_size, t, h_prev, c_prev, prev_z_t_category_infer):
        """
        Update joint probabilities based on various inputs.

        Args:
            joint_probs (torch.Tensor): Joint probabilities.
            curr_rnn_output (torch.Tensor): Current RNN output.
            batch_size (int): Batch size.
            t (int): Current time step.
            h_prev (torch.Tensor): Previous hidden state.
            c_prev (torch.Tensor): Previous cell state.
            prev_z_t_category_infer (torch.Tensor): Previous inferred latent category.

        Returns:
            tuple: Tuple containing updated joint probabilities, full KL divergence, updated hidden state and cell state, and inferred latent category.
        """
        z_t_category_gen = F.softmax(self.emitter_z(h_prev[-1]), dim = 1)

        full_kl = self.kl_div(prev_z_t_category_infer, z_t_category_gen)

        if self.transfer_prob:

            full_curr_rnn_input = torch.zeros((self.cluster_num, batch_size, self.cluster_num), dtype = torch.float, device = self.device)

            for k in range(self.cluster_num):
                curr_rnn_input = torch.zeros((batch_size, self.cluster_num), dtype = torch.float, device = self.device)

                curr_rnn_input[:,k] = 1

                full_curr_rnn_input[k] = curr_rnn_input

        else:
            full_curr_rnn_input = torch.zeros((self.cluster_num, batch_size, self.z_dim), dtype = torch.float, device = self.device)

            for k in range(self.cluster_num):
                curr_rnn_input = torch.zeros((batch_size, self.cluster_num), dtype = torch.float, device = self.device)

                curr_rnn_input[:,k] = 1

                full_curr_rnn_input[k] = torch.mm(curr_rnn_input, torch.t(self.phi_table))

        curr_rnn_output_full = curr_rnn_output.view(1, curr_rnn_output.shape[0], curr_rnn_output.shape[1])

        curr_rnn_output_full = curr_rnn_output_full.repeat(self.cluster_num, 1, 1)

        z_t, z_t_category_infer_full, _,z_category_infer_sparse = self.postnet(full_curr_rnn_input, curr_rnn_output_full, self.phi_table, t)

        updated_joint_probs = torch.sum(z_t_category_infer_full*torch.t(joint_probs).view(joint_probs.shape[1], joint_probs.shape[0], 1), 0)

        z_t_transfer_infer = prev_z_t_category_infer

        if self.transfer_prob:

            if self.block == 'GRU':
                output, h_now = self.trans(z_t_transfer_infer.view(z_t_transfer_infer.shape[0], 1, z_t_transfer_infer.shape[1]), h_prev.contiguous())# p(z_t| z_{t-1})
            else:
                output, (h_now, c_now) = self.trans(z_t_transfer_infer.view(z_t_transfer_infer.shape[0], 1, z_t_transfer_infer.shape[1]), (h_prev.contiguous(), c_prev.contiguous()))# p(z_t| z_{t-1})

            z_t, z_t_category_infer, _,z_category_infer_sparse = self.postnet(z_t_transfer_infer, curr_rnn_output, self.phi_table, t)
        else:

            phi_z_infer = torch.mm(z_t_transfer_infer, torch.t(self.phi_table))

            if self.block == 'GRU':
                output, h_now = self.trans(phi_z_infer.view(phi_z_infer.shape[0], 1, phi_z_infer.shape[1]).contiguous(), h_prev.contiguous())# p(z_t| z_{t-1})

            else:
                output, (h_now, c_now) = self.trans(phi_z_infer.view(phi_z_infer.shape[0], 1, phi_z_infer.shape[1]).contiguous(), (h_prev.contiguous(), c_prev.contiguous()))# p(z_t| z_{t-1})

            phi_z_infer2 = torch.mm(z_t_transfer_infer, torch.t(self.phi_table))
            z_t, z_t_category_infer, _,z_category_infer_sparse = self.postnet(phi_z_infer2, curr_rnn_output, self.phi_table, t)

        if np.isnan(full_kl.cpu().detach().numpy()).any():
            print('distribution 1::', z_t_category_gen)
            print('distribution 2::', z_t_category_infer)

        return updated_joint_probs, full_kl, h_now, c_now, z_t_category_infer, z_t_category_gen

    def infer(self, x, origin_x, x_mask, origin_x_mask, new_x_mask, x_lens, x_to_predict, \
              origin_x_to_pred, x_to_predict_mask, x_to_predict_origin_mask, x_to_predict_new_mask, \
                x_to_predict_lens, is_GPU, device, x_time_stamps, x_to_predict_time_stamps):
        """
        Infer q(z_{1:T}|x_{1:T}) (i.e. the variational distribution).

        Args:
            x (torch.Tensor): Observed data.
            origin_x (torch.Tensor): Original data.
            x_mask (torch.Tensor): Mask for observed data.
            origin_x_mask (torch.Tensor): Mask for original data.
            new_x_mask (torch.Tensor): Mask for new data.
            x_lens (torch.Tensor): Sequence lengths of observed data.
            x_to_predict (torch.Tensor): Data to predict.
            origin_x_to_pred (torch.Tensor): Original data for prediction.
            x_to_predict_mask (torch.Tensor): Mask for predicted data.
            x_to_predict_origin_mask (torch.Tensor): Mask for original predicted data.
            x_to_predict_new_mask (torch.Tensor): Mask for new predicted data.
            x_to_predict_lens (torch.Tensor): Sequence lengths of predicted data.
            is_GPU (bool): Flag for GPU usage.
            device: Device to use for computation.
            x_time_stamps (torch.Tensor): Time stamps for observed data.
            x_to_predict_time_stamps (torch.Tensor): Time stamps for predicted data.

        Returns:
            tuple: Tuple containing various losses and evaluation results.
        """

        T_max = x_lens.max().item()
        if is_GPU:
            x = x.to(device)
            
            x_to_predict = x_to_predict.to(device)
            
            x_mask = x_mask.to(device)
            
            x_to_predict_mask = x_to_predict_mask.to(device)
            
            origin_x = origin_x.to(device)
            
            origin_x_to_pred = origin_x_to_pred.to(device)
            
            origin_x_mask = origin_x_mask.to(device)
            
            new_x_mask = new_x_mask.to(device)

            x_to_predict_origin_mask = x_to_predict_origin_mask.to(device)
            
            x_to_predict_new_mask = x_to_predict_new_mask.to(device) 
        
            x_lens = x_lens.to(device)
        
            x_to_predict_lens = x_to_predict_lens.to(device)
            
            x_time_stamps = x_time_stamps.to(device)
            
            x_to_predict_time_stamps = x_to_predict_time_stamps.to(device)
        
        if self.is_missing:

            if self.pre_impute:
                imputed_x, interpolated_x = self.impute.forward2(x, x_mask, x_time_stamps[0].type(torch.float))

                x = imputed_x
            else:
                x = x_mask*x
                 
                interpolated_x = x
        
        batch_size, _, input_dim = x.size()
        
        h_prev = self.h_0.expand(self.trans.num_layers, batch_size, self.h_0.size(0))
        
        c_prev = self.h_0.expand(self.trans.num_layers, batch_size, self.h_0.size(0))
        
        
        z_1_category = torch.ones(self.cluster_num, dtype = torch.float, device = x.device)/self.cluster_num
        
        z_t_category_gen = z_1_category.expand(1, batch_size, z_1_category.size(0))

        if self.use_mask:
            input_x = torch.cat([x, x_mask], -1)
        
        else:
            input_x = x
        rnn_out,(last_h_n, last_c_n)= self.x_encoder(input_x, x_lens) # push the observed x's through the rnn;

        
        rec_losses = torch.zeros((batch_size, T_max-1, self.z_dim), device=x.device)
        
        cluster_losses = torch.zeros((batch_size, T_max  -1, self.z_dim), device=x.device) 
        
        rec_losses_no_coeff = torch.zeros((batch_size, T_max-1, self.z_dim), device=x.device)
        
        cluster_losses_no_coeff = torch.zeros((batch_size, T_max  -1, self.z_dim), device=x.device)

        kl_states = torch.zeros((batch_size, T_max), device=x.device)
        
        rmse_losses = torch.zeros((batch_size, input_dim, T_max - 1), device=x.device)
        
        mae_losses = torch.zeros((batch_size, input_dim, T_max - 1), device=x.device)   
        
        
#        cluster_distances = torch.zeros([batch_size, T_max, self.cluster_num], device = x.device)
#        cluster_distances2 = torch.zeros([T_max, self.cluster_num, batch_size, input_dim], device = x.device)
        
        prob_sums = 0
            
        '''z_q_*''' 
        z_prev = self.z_q_0.expand(batch_size,self.z_q_0.size(0)) # set z_prev=z_q_0 to setup the recursive conditioning in q(z_t|...)
        
        curr_x_lens = x_lens.clone()
        
        shrinked_x_lens = x_lens.clone()
        
        x_t = 0
        
        x_t_mask = 0
        
        curr_rnn_out = rnn_out[curr_x_lens > 0,0,:]
                
        last_h_now = torch.zeros_like(h_prev)
        
        last_c_now = torch.zeros_like(c_prev)
        
        imputed_x2 = torch.zeros_like(x)
        
        imputed_x2[:,0] = x[:,0] 
        
        last_rnn_out = torch.zeros(batch_size, self.s_dim*(1+self.x_encoder.bidir), device = self.device)
        
        joint_probs = torch.zeros([T_max, batch_size, self.cluster_num], dtype = torch.float, device = self.device)
        
        h_now_list = []
        
        last_decoded_h_n = torch.zeros([batch_size, self.z_dim], device = self.device)
        
        last_decoded_c_n = torch.zeros([batch_size, self.z_dim], device = self.device)

        all_z_t_category_infer = torch.zeros([T_max, batch_size, self.cluster_num], device = self.device)
        
        for t in range(T_max):
            
            '''phi_z_infer: phi_{z_t}'''

            if t == 0:
                z_t, z_t_category_infer, _, z_category_infer_sparse = self.postnet(z_prev, curr_rnn_out, self.phi_table, t) #q(z_t | z_{t-1}, x_{t:T})
                
                joint_probs[t] = z_t_category_infer
                
                all_z_t_category_infer[t] = z_t_category_infer
                
                kl = self.kl_div(z_t_category_infer, z_t_category_gen.view(z_t_category_gen.shape[1], z_t_category_gen.shape[2]))
                
                if np.isnan(kl.cpu().detach().numpy()).any():
                    print('distribution 1::', z_t_category_gen)
                    
                    print('distribution 2::', z_t_category_infer)

                z_t_category_trans =  z_t_category_infer
                
                if self.transfer_prob:
                    
                    if self.block == 'GRU':
                        output, h_now = self.trans(z_t_category_trans.view(z_t_category_trans.shape[0], 1, z_t_category_trans.shape[1]), h_prev.contiguous())# p(z_t| z_{t-1})
                        
                    else:
                        output, (h_now, c_now) = self.trans(z_t_category_trans.view(z_t_category_trans.shape[0], 1, z_t_category_trans.shape[1]), (h_prev.contiguous(), c_prev.contiguous()))# p(z_t| z_{t-1})

                else:
                    phi_z_infer = torch.mm(z_t_category_trans, torch.t(self.phi_table))

                    if self.block == 'GRU':
                        output, h_now = self.trans(phi_z_infer.view(phi_z_infer.shape[0], 1, phi_z_infer.shape[1]).contiguous(), h_prev.contiguous())# p(z_t| z_{t-1})
                        
                    else:
                        output, (h_now, c_now) = self.trans(phi_z_infer.view(phi_z_infer.shape[0], 1, phi_z_infer.shape[1]).contiguous(), (h_prev.contiguous(), c_prev.contiguous()))# p(z_t| z_{t-1})

            else:
                '''joint_probs, curr_rnn_output, batch_size, t, h_prev, c_prev, shrinked_x_lens'''
                '''updated_joint_probs, full_kl, h_now, c_now, full_rec_loss, full_logit_x_t'''
                updated_joint_probs, kl, h_now, c_now, z_t_category_infer, z_t_category_gen \
                    = self.update_joint_probability2(joint_probs[t-1, curr_x_lens > 0], curr_rnn_out, torch.sum(curr_x_lens > 0), \
                                                     t, h_prev, c_prev,z_t_category_infer)

                all_z_t_category_infer[t, curr_x_lens > 0] = z_t_category_gen
                
                joint_probs[t, curr_x_lens > 0] = updated_joint_probs

                kl_states[curr_x_lens > 0,t] = kl

            prob_sums += torch.sum(z_t_category_infer, 0)

            curr_x_lens -= 1

            shrinked_x_lens -= 1

            h_now_list.append(h_now)
            
            h_prev = h_now[:,shrinked_x_lens > 0,:]
            
            if (curr_x_lens == 0).any():
                last_h_now[:, curr_x_lens == 0, :] = h_now[:,shrinked_x_lens <= 0,:]
                last_rnn_out[curr_x_lens == 0] = rnn_out[curr_x_lens == 0,t,:]
            if self.block == 'LSTM':
                c_prev = c_now[:,shrinked_x_lens > 0,:]
                
                if (curr_x_lens == 0).any():
                    last_c_now[:, curr_x_lens == 0, :] = c_now[:,shrinked_x_lens <= 0,:] 

            if t < T_max - 1:
                x_t = x[curr_x_lens > 0,t+1,:]
                
                x_t_mask = x_mask[curr_x_lens > 0,t+1,:]
                
                curr_rnn_out = rnn_out[curr_x_lens > 0,t+1,:]

            shrinked_x_lens = shrinked_x_lens[shrinked_x_lens > 0]
        
        full_curr_rnn_input = torch.zeros((self.cluster_num, batch_size, self.cluster_num), dtype = torch.float, device = self.device)
        
        for k in range(self.cluster_num):
            curr_rnn_input = torch.zeros((batch_size, self.cluster_num), dtype = torch.float, device = self.device)
            
            curr_rnn_input[:,k] = 1
            
            full_curr_rnn_input[k] = curr_rnn_input
        
        curr_x_lens = x_lens.clone()
        
        shrinked_x_lens = x_lens.clone()
        
        x_t = x[curr_x_lens > 0,0,:]
                
        x_t_mask = x_mask[curr_x_lens > 0,0,:]
        
        for t in range(T_max):

            if t >= 1:

                input_x_t = x_t

                full_rec_loss1, full_rec_loss2, full_logit_x_t, curr_full_rec_loss, l2_norm_loss, cluster_loss \
                    = self.compute_rec_loss(joint_probs[t, curr_x_lens > 0], prob_sums/torch.sum(x_lens), \
                                            full_curr_rnn_input, input_x_t, x_t_mask, curr_rnn_out, all_z_t_category_infer[t])

                rec_losses[curr_x_lens > 0,t-1] = full_rec_loss1
                
                cluster_losses[curr_x_lens > 0,t-1] = full_rec_loss2
                
                rec_losses_no_coeff[curr_x_lens > 0,t-1] = l2_norm_loss
                
                cluster_losses_no_coeff[curr_x_lens > 0,t-1] = cluster_loss

                rmse_loss = (x_t*x_t_mask - full_logit_x_t*x_t_mask)**2
                
                mae_loss = torch.abs(x_t*x_t_mask - full_logit_x_t*x_t_mask)

                imputed_x2[curr_x_lens > 0,t] = full_logit_x_t
                
                rmse_losses[curr_x_lens > 0,:, t-1] = rmse_loss
                
                mae_losses[curr_x_lens > 0,:,t-1] = mae_loss

            curr_rnn_out = rnn_out[curr_x_lens > 0,t,:]

            curr_x_lens -= 1
            shrinked_x_lens -=1
            
            
            if t < T_max - 1:
                x_t = x[curr_x_lens > 0,t+1,:]
                
                x_t_mask = x_mask[curr_x_lens > 0,t+1,:]

            shrinked_x_lens = shrinked_x_lens[shrinked_x_lens > 0]

        rec_loss1 = torch.sum(rec_losses)/torch.sum(x_mask[:,1:,:])

        rec_loss2 = torch.sum(cluster_losses)/torch.sum(x_mask[:,1:,:])

        final_rec_loss = torch.sum(rec_losses_no_coeff)/torch.sum(x_mask[:,1:,:])

        final_cluster_loss = torch.sum(cluster_losses_no_coeff)/torch.sum(x_mask[:,1:,:])

        first_kl_loss = kl_states[:, 0].view(-1).mean()
        
        kl_loss = torch.sum(kl_states[:, 1:])/torch.sum(x_lens-1)
        final_rmse_loss = torch.sqrt(torch.sum(rmse_losses)/torch.sum(x_mask[:,1:,:]))
        
        final_mae_losses = torch.sum(mae_losses)/torch.sum(x_mask[:,1:,:])
        
        print('loss::', final_rec_loss, kl_loss)
        
        print('loss with coefficient::', rec_loss1, kl_loss)
        
        print('rmse loss::', final_rmse_loss)
        
        print('mae loss::', final_mae_losses)
        
        print('cluster objective::', final_cluster_loss)
        
        print('cluster objective with coefficient::', rec_loss2)
        
        final_ae_loss = 0

        interpolated_loss = 0
        
        if self.is_missing:
            interpolated_loss = torch.norm(interpolated_x*x_mask - x*x_mask)
            
            print('interpolate loss::', interpolated_loss)
        
        if torch.sum(1-new_x_mask[:,1:]) > 0:
            
            imputed_loss, imputed_loss2, imputed_mse_loss, imputed_mse_loss2 = self.calculate_imputation_losses(origin_x[:,1:], x[:,1:], new_x_mask[:,1:], imputed_x2[:,1:], objective = 'training')
        
        prior_cluster_probs = prob_sums/torch.sum(x_lens)
        
        if self.block == 'GRU':
            
            if self.evaluate:
                forecasting_imputed_data = self.evaluate_forecasting_errors(x_to_predict, origin_x_to_pred, x_to_predict_mask, x_to_predict_origin_mask, x_to_predict_new_mask, x_to_predict_lens, last_h_now, None, T_max, prior_cluster_probs, last_rnn_out, last_h_n, last_c_n, last_decoded_h_n.unsqueeze(0), last_decoded_c_n.unsqueeze(0))
            else:
                self.evaluate_forecasting_errors(x_to_predict, origin_x_to_pred, x_to_predict_mask, x_to_predict_origin_mask, x_to_predict_new_mask, x_to_predict_lens, last_h_now, None, T_max, prior_cluster_probs, last_rnn_out, last_h_n, last_c_n, last_decoded_h_n.unsqueeze(0), last_decoded_c_n.unsqueeze(0))
        else:
            if self.evaluate:
                forecasting_imputed_data = self.evaluate_forecasting_errors(x_to_predict, origin_x_to_pred, x_to_predict_mask, x_to_predict_origin_mask, x_to_predict_new_mask, x_to_predict_lens, last_h_now, last_c_now, T_max, prior_cluster_probs, last_rnn_out, last_h_n, last_c_n, last_decoded_h_n.unsqueeze(0), last_decoded_c_n.unsqueeze(0))
            else:
                self.evaluate_forecasting_errors(x_to_predict, origin_x_to_pred, x_to_predict_mask, x_to_predict_origin_mask, x_to_predict_new_mask, x_to_predict_lens, last_h_now, last_c_now, T_max, prior_cluster_probs, last_rnn_out, last_h_n, last_c_n, last_decoded_h_n.unsqueeze(0), last_decoded_c_n.unsqueeze(0))

        if not self.evaluate:
            return rec_loss1, kl_loss, first_kl_loss, final_rmse_loss, interpolated_loss, rec_loss2, final_ae_loss
        else:
            
            if torch.sum(1-new_x_mask[:,1:]) > 0:
                return imputed_x2*(1-x_mask) + x*x_mask, forecasting_imputed_data*(1-x_to_predict_mask) + x_to_predict*x_to_predict_mask, (imputed_mse_loss, imputed_mse_loss2, imputed_loss, imputed_loss2)
            else:
                return imputed_x2*(1-x_mask) + x*x_mask, forecasting_imputed_data*(1-x_to_predict_mask) + x_to_predict*x_to_predict_mask, None


    def test_samples(self, x, origin_x, x_mask, origin_x_mask, new_x_mask, x_lens, x_to_predict, origin_x_to_pred, \
                     x_to_predict_mask, x_to_predict_origin_mask, x_to_predict_new_mask, x_to_predict_lens, is_GPU, \
                        device, x_delta_time_stamps, x_to_predict_delta_time_stamps, x_time_stamps, x_to_predict_time_stamps):
        """
        Perform testing on the trained model by inferring q(z_{1:T}|x_{1:T}) (i.e. the variational distribution).

        Args:
            x (Tensor): Observed data sequence tensor of shape (batch_size, T_max, input_dim).
            origin_x (Tensor): Original observed data sequence tensor of shape (batch_size, T_max, input_dim).
            x_mask (Tensor): Mask indicating observed values in x of shape (batch_size, T_max, input_dim).
            origin_x_mask (Tensor): Mask indicating observed values in origin_x of shape (batch_size, T_max, input_dim).
            new_x_mask (Tensor): Mask indicating newly imputed values in x of shape (batch_size, T_max, input_dim).
            x_lens (Tensor): Lengths of sequences in x after masking, of shape (batch_size,).
            x_to_predict (Tensor): Data sequence to forecast of shape (batch_size, T_max, input_dim).
            origin_x_to_pred (Tensor): Original data sequence to forecast of shape (batch_size, T_max, input_dim).
            x_to_predict_mask (Tensor): Mask indicating observed values in x_to_predict of shape (batch_size, T_max, input_dim).
            x_to_predict_origin_mask (Tensor): Mask indicating observed values in origin_x_to_pred of shape (batch_size, T_max, input_dim).
            x_to_predict_new_mask (Tensor): Mask indicating newly imputed values in x_to_predict of shape (batch_size, T_max, input_dim).
            x_to_predict_lens (Tensor): Lengths of sequences in x_to_predict after masking, of shape (batch_size,).
            is_GPU (bool): Whether to use GPU for computation.
            device (str): Device to be used ('cpu' or 'cuda').
            x_delta_time_stamps (Tensor): Time differences between consecutive observations in x, of shape (batch_size, T_max-1).
            x_to_predict_delta_time_stamps (Tensor): Time differences between consecutive observations in x_to_predict, of shape (batch_size, T_max-1).
            x_time_stamps (Tensor): Absolute time stamps of observations in x, of shape (batch_size, T_max).
            x_to_predict_time_stamps (Tensor): Absolute time stamps of observations in x_to_predict, of shape (batch_size, T_max).

        Returns:
            Tuple or Tensor: Depending on the evaluation mode, returns evaluation metrics or imputed data.
        """

        T_max = x_lens.max().item()

        # Transfer data to GPU if necessary
        if is_GPU:
            x = x.to(device)
            x_to_predict = x_to_predict.to(device)
            x_mask = x_mask.to(device)
            x_to_predict_mask = x_to_predict_mask.to(device)
            origin_x = origin_x.to(device)
            origin_x_to_pred = origin_x_to_pred.to(device)
            origin_x_mask = origin_x_mask.to(device)
            new_x_mask = new_x_mask.to(device)
            x_to_predict_origin_mask = x_to_predict_origin_mask.to(device)
            x_to_predict_new_mask = x_to_predict_new_mask.to(device) 
            x_lens = x_lens.to(device)
            x_to_predict_lens = x_to_predict_lens.to(device)
            x_time_stamps = x_time_stamps.to(device)
            x_to_predict_time_stamps = x_to_predict_time_stamps.to(device)
        
        # Handle missing data
        if self.is_missing:

            if self.pre_impute:
                imputed_x, interpolated_x = self.impute.forward2(x, x_mask, x_time_stamps[0].type(torch.float))
                x = imputed_x

            else:      
                x = x_mask*x
                 
                interpolated_x = x

        batch_size, _, input_dim = x.size()

        # Initialize hidden states
        h_prev = self.h_0.expand(self.trans.num_layers, batch_size, self.h_0.size(0))
        c_prev = self.h_0.expand(self.trans.num_layers, batch_size, self.h_0.size(0))
        
        z_1_category = torch.ones(self.cluster_num, dtype = torch.float, device = x.device)/self.cluster_num
        z_t_category_gen = z_1_category.expand(1, batch_size, z_1_category.size(0))
        
        # Concatenate masks if required
        if self.use_mask:
            input_x = torch.cat([x, x_mask], -1)
        else:
            input_x = x
        
        rnn_out,(last_h_n, last_c_n)= self.x_encoder(input_x, x_lens) # push the observed x's through the rnn;

        '''to be done'''
#         rnn_out2,(last_h_n2, last_c_n2)= self.x_encoder.forward2(x, x_lens) # push the observed x's through the rnn;
        
#         print(torch.norm(rnn_out - rnn_out2), torch.norm(last_h_n - last_h_n2), torch.norm(last_c_n - last_c_n2))
        
#         rnn_out = reverse_sequence(rnn_out, x_lens) # reverse the time-ordering in the hidden state and un-pack it
        
#         if self.latent and self.lstm_latent:
#             rec_losses = torch.zeros((batch_size, T_max-1, (1+self.x_encoder.bidir)*self.s_dim), device=x.device)
#         
#             cluster_losses = torch.zeros((batch_size, T_max  -1, (1+self.x_encoder.bidir)*self.s_dim), device=x.device) 
#             
#             rec_losses_no_coeff = torch.zeros((batch_size, T_max-1, (1+self.x_encoder.bidir)*self.s_dim), device=x.device)
#             
#             cluster_losses_no_coeff = torch.zeros((batch_size, T_max  -1, (1+self.x_encoder.bidir)*self.s_dim), device=x.device)
#             
#         else:
            
#         if self.cluster_mask:
#             rec_losses = torch.zeros((batch_size, T_max-1, 2*self.z_dim), device=x.device)
#             
#             cluster_losses = torch.zeros((batch_size, T_max  -1, 2*self.z_dim), device=x.device) 
#             
#             rec_losses_no_coeff = torch.zeros((batch_size, T_max-1, 2*self.z_dim), device=x.device)
#             
#             cluster_losses_no_coeff = torch.zeros((batch_size, T_max  -1, 2*self.z_dim), device=x.device)
#         else:

        rec_losses = torch.zeros((batch_size, T_max-1, self.z_dim), device=x.device)
        cluster_losses = torch.zeros((batch_size, T_max  -1, self.z_dim), device=x.device) 
        rec_losses_no_coeff = torch.zeros((batch_size, T_max-1, self.z_dim), device=x.device)
        cluster_losses_no_coeff = torch.zeros((batch_size, T_max  -1, self.z_dim), device=x.device)
        kl_states = torch.zeros((batch_size, T_max), device=x.device)
        rmse_losses = torch.zeros((batch_size, input_dim, T_max - 1), device=x.device)
        mae_losses = torch.zeros((batch_size, input_dim, T_max - 1), device=x.device)   
        
#        cluster_distances = torch.zeros([batch_size, T_max, self.cluster_num], device = x.device)
#        cluster_distances2 = torch.zeros([T_max, self.cluster_num, batch_size, input_dim], device = x.device)
        
        prob_sums = 0
                
        '''z_q_*''' 
        z_prev = self.z_q_0.expand(batch_size,self.z_q_0.size(0)) # set z_prev=z_q_0 to setup the recursive conditioning in q(z_t|...)
        
        curr_x_lens = x_lens.clone()
        
        shrinked_x_lens = x_lens.clone()
        
        x_t = 0
        
        x_t_mask = 0
        
        curr_rnn_out = rnn_out[curr_x_lens > 0,0,:]
                
        last_h_now = torch.zeros_like(h_prev)
        
        last_c_now = torch.zeros_like(c_prev)
        
        imputed_x2 = torch.zeros_like(x)
        
        imputed_x2[:,0] = x[:,0] 
        
        last_rnn_out = torch.zeros(batch_size, self.s_dim*(1+self.x_encoder.bidir), device = self.device)
        
        joint_probs = torch.zeros([T_max, batch_size, self.cluster_num], dtype = torch.float, device = self.device)
        
        all_z_t_category_infer = torch.zeros([T_max, batch_size, self.cluster_num], dtype = torch.float, device = self.device)
        
        h_now_list = []
        
        last_decoded_h_n = torch.zeros([batch_size, self.z_dim], device = self.device)
        
        last_decoded_c_n = torch.zeros([batch_size, self.z_dim], device = self.device)

        
        for t in range(T_max):
            
            '''phi_z_infer: phi_{z_t}'''

            if t == 0:
                z_t, z_t_category_infer, _, z_category_infer_sparse = self.postnet(z_prev, curr_rnn_out, self.phi_table, t) #q(z_t | z_{t-1}, x_{t:T})
                
                joint_probs[t] = z_t_category_infer
                
                all_z_t_category_infer[t] = z_t_category_infer
                
                kl = self.kl_div(z_t_category_infer, z_t_category_gen.view(z_t_category_gen.shape[1], z_t_category_gen.shape[2]))
                
                if np.isnan(kl.cpu().detach().numpy()).any():
                    print('distribution 1::', z_t_category_gen)
                    
                    print('distribution 2::', z_t_category_infer)

                z_t_category_trans =  z_t_category_infer
                
                if self.transfer_prob:
                    
                    if self.block == 'GRU':
                        output, h_now = self.trans(z_t_category_trans.view(z_t_category_trans.shape[0], 1, z_t_category_trans.shape[1]), h_prev.contiguous())# p(z_t| z_{t-1})
                        
                    else:
                        output, (h_now, c_now) = self.trans(z_t_category_trans.view(z_t_category_trans.shape[0], 1, z_t_category_trans.shape[1]), (h_prev.contiguous(), c_prev.contiguous()))# p(z_t| z_{t-1})
                else:
                    
                    phi_z_infer = torch.mm(z_t_category_trans, torch.t(self.phi_table))
                    
                    if self.block == 'GRU':
                        output, h_now = self.trans(phi_z_infer.view(phi_z_infer.shape[0], 1, phi_z_infer.shape[1]).contiguous(), h_prev.contiguous())# p(z_t| z_{t-1})
                        
                    else:
                        output, (h_now, c_now) = self.trans(phi_z_infer.view(phi_z_infer.shape[0], 1, phi_z_infer.shape[1]).contiguous(), (h_prev.contiguous(), c_prev.contiguous()))# p(z_t| z_{t-1})

            else:
                '''joint_probs, curr_rnn_output, batch_size, t, h_prev, c_prev, shrinked_x_lens'''
                '''updated_joint_probs, full_kl, h_now, c_now, full_rec_loss, full_logit_x_t'''
                updated_joint_probs, kl, h_now, c_now, z_t_category_infer, z_t_category_gen \
                    = self.update_joint_probability2(joint_probs[t-1, curr_x_lens > 0], curr_rnn_out, \
                                                     torch.sum(curr_x_lens > 0), t, h_prev, c_prev,z_t_category_infer)

                all_z_t_category_infer[t] = z_t_category_gen
                
                joint_probs[t, curr_x_lens > 0] = updated_joint_probs

                kl_states[curr_x_lens > 0,t] = kl

            prob_sums += torch.sum(z_t_category_infer, 0)

            curr_x_lens -= 1

            shrinked_x_lens -= 1

            h_now_list.append(h_now)
            
            h_prev = h_now[:,shrinked_x_lens > 0,:]
            
            if (curr_x_lens == 0).any():
                last_h_now[:, curr_x_lens == 0, :] = h_now[:,shrinked_x_lens <= 0,:]
                last_rnn_out[curr_x_lens == 0] = rnn_out[curr_x_lens == 0,t,:]
            if self.block == 'LSTM':
                c_prev = c_now[:,shrinked_x_lens > 0,:]
                
                if (curr_x_lens == 0).any():
                    last_c_now[:, curr_x_lens == 0, :] = c_now[:,shrinked_x_lens <= 0,:] 

            if t < T_max - 1:
                x_t = x[curr_x_lens > 0,t+1,:]
                
                x_t_mask = x_mask[curr_x_lens > 0,t+1,:]
                
                curr_rnn_out = rnn_out[curr_x_lens > 0,t+1,:]

            shrinked_x_lens = shrinked_x_lens[shrinked_x_lens > 0]

            
        full_curr_rnn_input = torch.zeros((self.cluster_num, batch_size, self.cluster_num), dtype = torch.float, device = self.device)
        
        for k in range(self.cluster_num):
            curr_rnn_input = torch.zeros((batch_size, self.cluster_num), dtype = torch.float, device = self.device)
            
            curr_rnn_input[:,k] = 1
            
            full_curr_rnn_input[k] = curr_rnn_input
        
        
        curr_x_lens = x_lens.clone()
        
        shrinked_x_lens = x_lens.clone()
        
        x_t = x[curr_x_lens > 0,0,:]
                
        x_t_mask = x_mask[curr_x_lens > 0,0,:]
        
        for t in range(T_max):

            if t >= 1:

                input_x_t = x_t

                full_rec_loss1, full_rec_loss2, full_logit_x_t, curr_full_rec_loss, l2_norm_loss, cluster_loss \
                    = self.compute_rec_loss(joint_probs[t, curr_x_lens > 0], prob_sums/torch.sum(x_lens), \
                                            full_curr_rnn_input, input_x_t, x_t_mask, curr_rnn_out, all_z_t_category_infer[t])

                rec_losses[curr_x_lens > 0,t-1] = full_rec_loss1
                
                cluster_losses[curr_x_lens > 0,t-1] = full_rec_loss2
                
                rec_losses_no_coeff[curr_x_lens > 0,t-1] = l2_norm_loss
                
                cluster_losses_no_coeff[curr_x_lens > 0,t-1] = cluster_loss

                rmse_loss = (x_t*x_t_mask - full_logit_x_t*x_t_mask)**2
                
                mae_loss = torch.abs(x_t*x_t_mask - full_logit_x_t*x_t_mask)

                imputed_x2[curr_x_lens > 0,t] = full_logit_x_t
            
                rmse_losses[curr_x_lens > 0,:, t-1] = rmse_loss
                
                mae_losses[curr_x_lens > 0,:,t-1] = mae_loss
            
            curr_rnn_out = rnn_out[curr_x_lens > 0,t,:]

            curr_x_lens -= 1
            shrinked_x_lens -=1
            
            
            if t < T_max - 1:
                x_t = x[curr_x_lens > 0,t+1,:]
                
                x_t_mask = x_mask[curr_x_lens > 0,t+1,:]

            shrinked_x_lens = shrinked_x_lens[shrinked_x_lens > 0]

        rec_loss1 = torch.sum(rec_losses)/torch.sum(x_mask[:,1:,:])

        rec_loss2 = torch.sum(cluster_losses)/torch.sum(x_mask[:,1:,:])

        final_rec_loss = torch.sum(rec_losses_no_coeff)/torch.sum(x_mask[:,1:,:])

        final_cluster_loss = torch.sum(cluster_losses_no_coeff)/torch.sum(x_mask[:,1:,:])
        
        kl_loss = torch.sum(kl_states[:, 1:])/torch.sum(x_lens-1)
        final_rmse_loss = torch.sqrt(torch.sum(rmse_losses)/torch.sum(x_mask[:,1:,:]))
        
        final_mae_losses = torch.sum(mae_losses)/torch.sum(x_mask[:,1:,:])
        
        print('loss::', final_rec_loss, kl_loss)
        
        print('loss with coefficient::', rec_loss1, kl_loss)
        
        print('rmse loss::', final_rmse_loss)
        
        print('mae loss::', final_mae_losses)
        
        print('cluster objective::', final_cluster_loss)
        
        print('cluster objective with coefficient::', rec_loss2)

        imputed_loss = 0
        
        # Handle missing data and evaluate imputation losses if required
        if self.is_missing:
            imputed_loss = torch.norm(interpolated_x*x_mask - x*x_mask)
            
            print('interpolate loss::', imputed_loss)
        
        if torch.sum(1-new_x_mask[:,1:]) > 0:
            
            curr_new_mask = origin_x_mask*(1-new_x_mask)
            
            assert (torch.nonzero(~(curr_new_mask ==  (1-new_x_mask)))).shape[0] == 0
            
            imputed_loss, imputed_loss2, imputed_mse_loss, imputed_mse_loss2 = \
                self.calculate_imputation_losses(origin_x[:,1:], x[:,1:], new_x_mask[:,1:], imputed_x2[:,1:], objective = 'training')

        prior_cluster_probs = prob_sums/torch.sum(x_lens)
        
        if not self.evaluate:
            final_rmse_loss, rmse_loss_count, final_mae_losses, mae_loss_count, final_nll_loss, nll_loss_count, list_res = self.evaluate_forecasting_errors(x_to_predict, origin_x_to_pred, x_to_predict_mask, x_to_predict_origin_mask, x_to_predict_new_mask, x_to_predict_lens, last_h_now, last_c_now, T_max, prior_cluster_probs, last_rnn_out, last_h_n, last_c_n, last_decoded_h_n.unsqueeze(0), last_decoded_c_n.unsqueeze(0))
        else:
            imputed_x2 = self.evaluate_forecasting_errors(x_to_predict, origin_x_to_pred, x_to_predict_mask, x_to_predict_origin_mask, x_to_predict_new_mask, x_to_predict_lens, last_h_now, last_c_now, T_max, prior_cluster_probs, last_rnn_out,last_h_n, last_c_n, last_decoded_h_n.unsqueeze(0), last_decoded_c_n.unsqueeze(0))
        
        # Save cluster centroids and return results
        if not os.path.exists(data_folder + '/' + output_dir):
            os.makedirs(data_folder + '/' + output_dir)

        torch.save(self.phi_table, data_folder + '/' + output_dir + 'cluster_centroids')

        if not self.evaluate:
            if not torch.sum(1-new_x_mask[:,1:]) > 0:
                return final_rmse_loss, rmse_loss_count, final_mae_losses, mae_loss_count, final_nll_loss, nll_loss_count, list_res, None
            else:
                return final_rmse_loss, rmse_loss_count, final_mae_losses, mae_loss_count, final_nll_loss, nll_loss_count, list_res, ((imputed_loss, imputed_loss2), torch.sum(1-new_x_mask[:,1:]), (imputed_mse_loss, imputed_mse_loss2), torch.sum(1-new_x_mask[:,1:]))

        else:
            
            if torch.sum(1-new_x_mask[:,1:]) > 0:
                return imputed_x2, (imputed_mse_loss, imputed_mse_loss2, imputed_loss, imputed_loss2)
            else:
                return imputed_x2, None

    '''self, x, origin_x, x_mask, x_origin_mask, x_new_mask, x_lens, x_to_predict, origin_x_to_predict, x_to_predict_mask, x_to_predict_origin_mask, x_to_predict_new_mask, x_to_predict_lens, is_GPU, device, x_delta_time_stamps, x_to_predict_delta_time_stamps, x_time_stamps, x_to_predict_time_stamps'''


    def evaluate_forecasting_errors(self, x, origin_x_to_pred, x_mask, x_to_predict_origin_mask, x_to_predict_new_mask, x_to_predict_lens, h_now, c_now, T_max_train, prior_cluster_probs, curr_rnn_out, last_h_n, last_c_n, last_decoded_h_n, last_decoded_c_n):
        """
        Evaluate forecasting errors on the trained model.

        Args:
            x (Tensor): Observed data sequence tensor of shape (batch_size, T_max, input_dim).
            origin_x_to_pred (Tensor): Original data sequence to forecast of shape (batch_size, T_max, input_dim).
            x_mask (Tensor): Mask indicating observed values in x of shape (batch_size, T_max, input_dim).
            x_to_predict_origin_mask (Tensor): Mask indicating observed values in origin_x_to_pred of shape (batch_size, T_max, input_dim).
            x_to_predict_new_mask (Tensor): Mask indicating newly imputed values in x_to_predict of shape (batch_size, T_max, input_dim).
            x_to_predict_lens (Tensor): Lengths of sequences in x_to_predict after masking, of shape (batch_size,).
            h_now (Tensor): Hidden state tensor for forecasting, of shape (num_layers, batch_size, hidden_size).
            c_now (Tensor): Cell state tensor for forecasting (if LSTM), of shape (num_layers, batch_size, hidden_size).
            T_max_train (int): Maximum time step for training data.
            prior_cluster_probs (Tensor): Prior cluster probabilities for z_t_category_gen, of shape (cluster_num,).
            curr_rnn_out (Tensor): RNN output for forecasting, of shape (batch_size, hidden_size).
            last_h_n (Tensor): Last hidden state tensor for forecasting, of shape (num_layers, batch_size, hidden_size).
            last_c_n (Tensor): Last cell state tensor for forecasting (if LSTM), of shape (num_layers, batch_size, hidden_size).
            last_decoded_h_n (Tensor): Last decoded hidden state tensor, of shape (batch_size, z_dim).
            last_decoded_c_n (Tensor): Last decoded cell state tensor (if LSTM), of shape (batch_size, z_dim).

        Returns:
            Tensor: Evaluated forecasting results (imputed data).
        """

        T_max = x_to_predict_lens.max().cpu().item()

        # Initialize various tensors for storing losses
        rmse_losses = torch.zeros((x.shape[0], x.shape[2], T_max), device=x.device)
        mae_losses = torch.zeros((x.shape[0], x.shape[2], T_max), device=x.device)
        rmse_losses2 = torch.zeros((x.shape[0], x.shape[2], T_max), device=x.device)
        mae_losses2 = torch.zeros((x.shape[0], x.shape[2], T_max), device=x.device)
        neg_nll_losses = torch.zeros((x.shape[0], self.z_dim, T_max), device=x.device)
        
         # Initialize lens and hidden states for forecasting
        curr_x_to_predict_lens = x_to_predict_lens.clone()
        shrinked_x_to_predict_lens = x_to_predict_lens.clone() 
        imputed_x2 = torch.zeros_like(x)
        last_h_n2 = last_h_n.clone()
        last_c_n2 = last_c_n.clone()
        
        # Perform computations for each time step
        for t in range(T_max):

            z_t_category_gen = F.softmax(self.emitter_z(h_now[-1]), dim = 1)

            phi_z, z_representation = self.generate_z(z_t_category_gen, prior_cluster_probs, last_h_n)

            mean, std, logit_x_t = self.generate_x(phi_z)
            
            
            
            avg_selected_phi_z = self.sample_x(z_representation)
            
            
            if self.cluster_mask:
                mean, std, logit_x_t = mean[:,0:self.z_dim], std[:,0:self.z_dim], logit_x_t[:,0:self.z_dim]
                avg_selected_phi_z = avg_selected_phi_z[:, 0:self.z_dim]

            imputed_x2[curr_x_to_predict_lens>0,t,:] = logit_x_t

            if self.use_gate:
                
                if self.use_mask:
                    
                    input_x = torch.cat([logit_x_t, torch.ones_like(logit_x_t)], -1)
                    curr_rnn_out, (last_h_n, last_c_n) = self.x_encoder(input_x.view(input_x.shape[0], 1, input_x.shape[1]), torch.ones(logit_x_t.shape[0], device = self.device), init_h = last_h_n, init_c = last_c_n)
                else:
                    input_x = logit_x_t
                    
                    curr_rnn_out, (last_h_n, last_c_n) = self.x_encoder(input_x.view(input_x.shape[0], 1, input_x.shape[1]), torch.ones(logit_x_t.shape[0], device = self.device), init_h = last_h_n, init_c = last_c_n)

            
            rec_loss = (x[curr_x_to_predict_lens>0,t,:]*x_mask[curr_x_to_predict_lens>0,t,:] - logit_x_t*x_mask[curr_x_to_predict_lens>0,t,:])**2
            
            mae_loss = torch.abs(x[curr_x_to_predict_lens>0,t,:]*x_mask[curr_x_to_predict_lens>0,t,:] - logit_x_t*x_mask[curr_x_to_predict_lens>0,t,:])
            
            rec_loss2 = (x[curr_x_to_predict_lens>0,t,:]*x_mask[curr_x_to_predict_lens>0,t,:] - avg_selected_phi_z*x_mask[curr_x_to_predict_lens>0,t,:])**2
            
            mae_loss2 = torch.abs(x[curr_x_to_predict_lens>0,t,:]*x_mask[curr_x_to_predict_lens>0,t,:] - avg_selected_phi_z*x_mask[curr_x_to_predict_lens>0,t,:])
            
            rmse_losses[curr_x_to_predict_lens>0,:,t] = rec_loss
            
            mae_losses[curr_x_to_predict_lens>0,:,t] = mae_loss
            
            rmse_losses2[curr_x_to_predict_lens>0,:,t] = rec_loss2
            
            mae_losses2[curr_x_to_predict_lens>0,:,t] = mae_loss2

            neg_nll_losses[curr_x_to_predict_lens>0,:,t] = compute_gaussian_probs0(x[curr_x_to_predict_lens>0,t,:], mean, std, x_mask[curr_x_to_predict_lens>0,t,:])

            z_t_category_gen_trans = z_t_category_gen

            if self.transfer_prob:
                if self.block == 'GRU':
                    output, h_now = self.trans(z_t_category_gen_trans.view(z_t_category_gen_trans.shape[0], 1, z_t_category_gen_trans.shape[1]), h_now)# p(z_t| z_{t-1})
                else:
                    output, (h_now, c_now) = self.trans(z_t_category_gen_trans.view(z_t_category_gen_trans.shape[0], 1, z_t_category_gen_trans.shape[1]), (h_now, c_now))# p(z_t| z_{t-1})
            
            else:
                
                phi_z_transfer = torch.t(torch.mm(self.phi_table, torch.t(z_t_category_gen_trans.squeeze(0))))
                
                if self.block == 'GRU':
                    output, h_now = self.trans(phi_z_transfer.view(phi_z_transfer.shape[0], 1, phi_z_transfer.shape[1]), h_now)# p(z_t| z_{t-1})
                    
                else:
                    output, (h_now, c_now) = self.trans(phi_z_transfer.view(phi_z_transfer.shape[0], 1, phi_z_transfer.shape[1]), (h_now, c_now))# p(z_t| z_{t-1})

            curr_x_to_predict_lens -= 1
            
            shrinked_x_to_predict_lens -= 1
            
            h_now = h_now[:,shrinked_x_to_predict_lens > 0]


            if self.use_gate:
                curr_rnn_out = curr_rnn_out[shrinked_x_to_predict_lens > 0,0]
                
                last_h_n = last_h_n[:,shrinked_x_to_predict_lens > 0]
            
                last_c_n = last_c_n[:,shrinked_x_to_predict_lens > 0]
                
                last_h_n2 = last_h_n2[:,shrinked_x_to_predict_lens > 0]
            
                last_c_n2 = last_c_n2[:,shrinked_x_to_predict_lens > 0]
            
            if self.block == 'LSTM':
                c_now = c_now[:,shrinked_x_to_predict_lens > 0]
            
            phi_z = phi_z[shrinked_x_to_predict_lens > 0]
            
            
            
            shrinked_x_to_predict_lens = shrinked_x_to_predict_lens[shrinked_x_to_predict_lens > 0]
        
        # Calculate various forecasting losses
        final_rmse_loss = torch.sqrt(torch.sum(rmse_losses)/torch.sum(x_mask))
        
        final_mae_losses = torch.sum(mae_losses)/torch.sum(x_mask)
        
        final_rmse_loss2 = torch.sqrt(torch.sum(rmse_losses2)/torch.sum(x_mask))
        
        final_mae_losses2 = torch.sum(mae_losses2)/torch.sum(x_mask)
        
        final_nll_loss = torch.sum(neg_nll_losses)/torch.sum(x_mask)
        
        print('forecasting rmse loss::', final_rmse_loss)
        
        print('forecasting mae loss::', final_mae_losses)
        
        print('forecasting rmse loss 2::', final_rmse_loss2)
        
        print('forecasting mae loss 2::', final_mae_losses2)
        
        print('forecasting neg likelihood::', final_nll_loss)

        # Calculate imputation losses if required
        if torch.sum(1-x_to_predict_new_mask) > 0:
            imputed_loss, imputed_loss2, imputed_mse_loss, imputed_mse_loss2 = \
                self.calculate_imputation_losses(origin_x_to_pred, x, x_to_predict_new_mask, imputed_x2, objective = 'forecasting')
        
        # Calculate forecasting results by time steps
        rmse_list, mae_list = get_forecasting_res_by_time_steps(x, imputed_x2, x_mask)
        
        if not self.evaluate:
            return final_rmse_loss, torch.sum(x_mask), final_mae_losses, torch.sum(x_mask), final_nll_loss, torch.sum(x_mask), (rmse_list, mae_list, torch.sum(x_mask)) 
        else:
            return imputed_x2*(1-x_mask) + x*x_mask


    def init_params(self):
        """
        Initialize model parameters using Xavier normal initialization.
        """
        for p in self.parameters():
            if len(p.shape) >= 2:
                torch.nn.init.xavier_normal_(p, 1)


    def compute_phi_table_gap(self):
        """
        Compute the gap between cluster centroids in the phi_table.

        Returns:
            Tensor: The computed L2 regularization term.
        """
        d = self.centroid_max
        cluster_diff = torch.norm(self.phi_table.view(self.phi_table.shape[0], 1, self.phi_table.shape[1]) - self.phi_table.view(self.phi_table.shape[0], self.phi_table.shape[1], 1), dim=0)
        distance = torch.sum(torch.triu(F.relu(d - cluster_diff)**2, diagonal=1))
        return distance


    def get_regularization_term(self):
        """
        Compute the L2 regularization term for model parameters.

        Returns:
            Tensor: The computed L2 regularization term.
        """
        reg_term = 0
        
        for param in self.parameters():
            reg_term += torch.norm(param.view(-1))**2
            
        return reg_term


    def train_AE(self, x, origin_x, x_mask, origin_x_mask, new_x_mask, x_lens, kl_anneal, 
                 x_to_predict, origin_x_to_pred, x_to_predict_mask, x_to_predict_origin_mask,
                 x_to_predict_new_mask, x_to_predict_lens, is_GPU, device, x_time_stamps, x_to_predict_time_stamps):
        """
        Train the autoencoder model.

        Args:
            x (Tensor): Observed data sequence tensor.
            origin_x (Tensor): Original data sequence tensor.
            x_mask (Tensor): Mask indicating observed values in x.
            origin_x_mask (Tensor): Mask indicating observed values in origin_x.
            new_x_mask (Tensor): Mask indicating newly imputed values.
            x_lens (Tensor): Lengths of sequences in x.
            kl_anneal (float): Annealing factor for KL divergence.
            x_to_predict (Tensor): Data sequence to forecast.
            origin_x_to_pred (Tensor): Original data sequence to forecast.
            x_to_predict_mask (Tensor): Mask indicating observed values in x_to_predict.
            x_to_predict_origin_mask (Tensor): Mask indicating observed values in origin_x_to_pred.
            x_to_predict_new_mask (Tensor): Mask indicating newly imputed values in x_to_predict.
            x_to_predict_lens (Tensor): Lengths of sequences in x_to_predict.
            is_GPU (bool): Flag indicating whether to use GPU.
            device (str): Device to use (e.g., "cuda" or "cpu").
            x_time_stamps (Tensor): Time stamps for x.
            x_to_predict_time_stamps (Tensor): Time stamps for x_to_predict.

        Returns:
            dict: A dictionary containing training loss values.
        """
        self.x_encoder.train() # put the RNN back into training mode (i.e. turn on drop-out if applicable)

        rec_loss1, kl_loss, first_kl_loss, final_rmse_loss, interpolated_loss, rec_loss2, final_ae_loss = \
            self.infer(x, origin_x, x_mask, origin_x_mask, new_x_mask, x_lens, x_to_predict, origin_x_to_pred, 
                       x_to_predict_mask, x_to_predict_origin_mask, x_to_predict_new_mask, x_to_predict_lens, 
                       is_GPU, device, x_time_stamps, x_to_predict_time_stamps)
        loss = rec_loss1 + kl_anneal*kl_loss + 5e-5*first_kl_loss + rec_loss2 + 0.0001*interpolated_loss  # (10 + 10* kl_anneal)*final_ae_loss# + 0.01*interpolated_loss# + 0.001*self.get_regularization_term()#+ 0.001*self.compute_phi_table_gap()# + 0.001*final_entropy_loss#

        self.optimizer.zero_grad()

        loss.backward()
        nn.utils.clip_grad_norm_(self.parameters(), self.clip_norm)
        self.optimizer.step()

        return {'train_loss_AE':loss.item(), 'train_loss_KL':kl_loss.item()}


    def valid(self, x, x_rev, x_lens):
        """
        Validate the model.

        Args:
            x (Tensor): Observed data sequence tensor.
            x_rev (Tensor): Reversed data sequence tensor.
            x_lens (Tensor): Lengths of sequences in x.
        
        Returns:
            Tensor: The computed validation loss.
        """
        self.eval()
        rec_loss, kl_loss = self.infer(x, x_rev, x_lens)
        loss = rec_loss + kl_loss
        return loss