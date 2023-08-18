
import time
import numpy as np
import sys, os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam

from helper import reverse_sequence, sequence_mask
from torch.distributions import normal

from torch.distributions import Independent
from torch.distributions.normal import Normal

from ODE_modules import *
from imputation.inverse_distance_weighting import *
from lib.utils import *
from lib.encoder_decoder_cluster import *

sys.path.append(os.path.dirname(os.path.abspath(__file__))+'/imputation')
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/lib')
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.abspath(__file__))


data_folder = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + "/" + data_dir

class DGM2_O(nn.Module):
    """
    The Deep Markov Model
    """
    def __init__(self, config ):
        super(DGM2_O, self).__init__()

        self.input_dim = config['input_dim']
        
        self.h_dim = config['h_dim']
        
        self.s_dim = config['s_dim']
        
        self.centroid_max = config['d']
        
        self.device = config['device']
        
        self.dropout = config['dropout']

        self.e_dim = config['e_dim']

        self.cluster_num = config['cluster_num']

        self.h_0 = torch.zeros(self.h_dim, device = config['device'])
        
        self.s_0 = torch.zeros(self.s_dim, device = config['device'])
        
        self.x_std = config['x_std']
        
        self.use_gate = config['use_gate']
        
        self.evaluate = False
        
        self.transfer_prob = True
        
        self.gaussian_prior_coeff = config['gaussian']
        
        self.pre_impute = True
        
        self.shift = 5
        
        self.use_shift = True
        
        self.clip_norm = config['clip_norm']
        
        self.emitter_z = nn.Sequential( #Parameterizes the bernoulli observation likelihood `p(x_t|z_t)`
            nn.Linear(self.h_dim, self.cluster_num),
            nn.Dropout(p = self.dropout)
        )

        self.block = 'LSTM'

        self.z_dim = self.input_dim

        self.use_transition_gen = True
        
        if config['is_missing']:
            self.impute = IDW_impute(self.input_dim, self.device)
        self.is_missing = config['is_missing'] 
        
        
        z0_dim = self.s_dim

        self.use_mask = True

        n_rec_dims = self.s_dim
        
        ode_func_net = create_net(n_rec_dims, n_rec_dims, nonlinear = nn.Tanh, dropout = self.dropout, n_layers = 0)
    
        rec_ode_func = ODEFunc(
            input_dim = self.s_dim, 
            latent_dim = n_rec_dims,
            ode_func_net = ode_func_net,
            device = self.device).to(self.device)

        z0_diffeq_solver = DiffeqSolver(self.s_dim, rec_ode_func, "euler", self.s_dim, 
            odeint_rtol = 1e-3, odeint_atol = 1e-4, device = self.device)

        self.gru_update = GRU_unit_cluster(self.s_dim, self.input_dim, n_units = self.h_dim, device = self.device, use_mask = self.use_mask, dropout = self.dropout)

        self.method_2 = False
        
        if self.method_2:
            self.postnet = Encoder_z0_ODE_RNN_cluster2(n_rec_dims, self.input_dim, self.cluster_num, z0_diffeq_solver, 
            z0_dim = z0_dim, n_gru_units = self.h_dim, GRU_update=self.gru_update, device = self.device, use_mask = self.use_mask, dropout = self.dropout).to(self.device)
        else:self.postnet = Encoder_z0_ODE_RNN_cluster(n_rec_dims, self.input_dim, self.cluster_num, z0_diffeq_solver, 
            z0_dim = z0_dim, n_gru_units = self.h_dim, GRU_update=self.gru_update, device = self.device, use_mask = self.use_mask, dropout = self.dropout).to(self.device)

        ode_func_net2 = create_net(self.s_dim, self.s_dim,  nonlinear = nn.Tanh, dropout = self.dropout, n_layers = 0, n_units = 20)
     
        self.gen_ode_func = ODEFunc(
                input_dim = self.s_dim, 
                latent_dim = self.s_dim, 
                ode_func_net = ode_func_net2,
                device = self.device).to(self.device)

        self.prob_to_states = nn.Sequential(nn.Linear(self.cluster_num, self.s_dim), nn.Dropout(p = self.dropout))

        self.diffeq_solver = DiffeqSolver(self.cluster_num, self.gen_ode_func, 'dopri5', self.s_dim, device = self.device, odeint_rtol = 1e-3, odeint_atol = 1e-4)

        self.gen_emitter_z = nn.Sequential( #Parameterizes the bernoulli observation likelihood `p(x_t|z_t)`
            nn.Linear(self.s_dim, self.cluster_num),
        )
        
        self.trans = Decoder_ODE_RNN_cluster(n_rec_dims, self.cluster_num, self.cluster_num, self.diffeq_solver, 
            z0_dim = z0_dim, n_gru_units = self.h_dim, device = self.device, dropout = self.dropout).to(self.device)
        
        if self.use_gate:
            self.gate_func = nn.Sequential(
                nn.Linear(self.s_dim, 1),
                nn.Dropout(p = self.dropout)
                )
        
        self.phi_table = torch.zeros([self.input_dim, self.cluster_num], dtype = torch.float, device = config['device']) 
        
        self.phi_table = torch.nn.Parameter(self.phi_table)

        if self.transfer_prob:
            self.z_q_0 = torch.zeros(self.cluster_num, device = config['device'])
        else:
            self.z_q_0 = torch.zeros(self.z_dim, device = config['device'])
        
        self.optimizer = Adam(self.parameters(), lr=config['lr'], betas= (config['beta1'], config['beta2']))

    def get_reconstruction(self, time_steps_to_predict, truth, truth_time_steps, 
        mask = None, n_traj_samples = 1, run_backwards = False, mode = None):
        """
        Get the reconstruction results for the model.
        
        Args:
            time_steps_to_predict (Tensor): Time steps to predict.
            truth (Tensor): Ground truth data.
            truth_time_steps (Tensor): Time steps for the truth data.
            mask (Tensor, optional): Mask indicating observed values. Defaults to None.
            n_traj_samples (int, optional): Number of trajectory samples. Defaults to 1.
            run_backwards (bool, optional): Flag indicating if to run backwards. Defaults to False.
            mode (str, optional): Mode for reconstruction. Defaults to None.
        
        Returns:
            tuple: A tuple containing the inference probabilities, generated y probabilities, and latent y states.
        """
        if isinstance(self.postnet, Encoder_z0_ODE_RNN_cluster) or \
            isinstance(self.postnet, Encoder_z0_RNN):

            truth_w_mask = truth
            if self.use_mask:
                truth_w_mask = torch.cat((truth, mask), -1)
            infer_probs, latent_y_states = self.postnet(
                truth_w_mask, truth_time_steps, run_backwards = run_backwards)

        else:
            raise Exception("Unknown encoder type {}".format(type(self.encoder_z0).__name__))
        
        if run_backwards:
            new_infer_probs = torch.flip(infer_probs, [1])
            
            infer_probs = new_infer_probs
            
            new_lat_y_states = torch.flip(latent_y_states, [1])

            latent_y_states = new_lat_y_states

        sol_y = self.diffeq_solver(latent_y_states[:,0], time_steps_to_predict)

        gen_y_probs = F.softmax(self.gen_emitter_z(sol_y), -1)

        return torch.transpose(infer_probs.squeeze(0), 0, 1), gen_y_probs.squeeze(0), latent_y_states.squeeze(0)
    
    
    def get_reconstruction1(self, time_steps_to_predict, truth, truth_time_steps, 
        mask = None, n_traj_samples = 1, run_backwards = False, mode = None):
        """
        Get the reconstruction results for the model using an alternative method.
        
        Args:
            time_steps_to_predict (Tensor): Time steps to predict.
            truth (Tensor): Ground truth data.
            truth_time_steps (Tensor): Time steps for the truth data.
            mask (Tensor, optional): Mask indicating observed values. Defaults to None.
            n_traj_samples (int, optional): Number of trajectory samples. Defaults to 1.
            run_backwards (bool, optional): Flag indicating if to run backwards. Defaults to False.
            mode (str, optional): Mode for reconstruction. Defaults to None.
        
        Returns:
            tuple: A tuple containing the inference probabilities, generated y probabilities, and latent y states.
        """
        if isinstance(self.postnet, Encoder_z0_ODE_RNN_cluster2) or \
            isinstance(self.postnet, Encoder_z0_ODE_RNN_cluster):

            truth_w_mask = truth
            if self.use_mask:
                truth_w_mask = torch.cat((truth, mask), -1)
            infer_probs, latent_y_states = self.postnet(
                truth_w_mask, truth_time_steps, run_backwards = run_backwards)

        else:
            raise Exception("Unknown encoder type {}".format(type(self.encoder_z0).__name__))
        
        if run_backwards:
            new_infer_probs = torch.flip(infer_probs, [1])
             
            infer_probs = new_infer_probs
             
            new_lat_y_states = torch.flip(latent_y_states, [1])
 
            latent_y_states = new_lat_y_states
        
        sol_y = self.diffeq_solver(latent_y_states[:,0], time_steps_to_predict)

        gen_y_probs = F.softmax(self.gen_emitter_z(sol_y), -1)

        return torch.transpose(infer_probs.squeeze(0), 0, 1), gen_y_probs.squeeze(0), latent_y_states.squeeze(0)
    
    
    def get_reconstruction2(self, time_steps_to_predict, truth, truth_time_steps, 
        mask = None, n_traj_samples = 1, run_backwards = False, mode = None):
        """
        Get the reconstruction results for the model using another alternative method.
        
        Args:
            time_steps_to_predict (Tensor): Time steps to predict.
            truth (Tensor): Ground truth data.
            truth_time_steps (Tensor): Time steps for the truth data.
            mask (Tensor, optional): Mask indicating observed values. Defaults to None.
            n_traj_samples (int, optional): Number of trajectory samples. Defaults to 1.
            run_backwards (bool, optional): Flag indicating if to run backwards. Defaults to False.
            mode (str, optional): Mode for reconstruction. Defaults to None.
        
        Returns:
            tuple: A tuple containing the inference probabilities, generated y probabilities, latent y states, 
                   generated latent y states, and extra KL divergence.
        """
        if isinstance(self.postnet, Encoder_z0_ODE_RNN_cluster) or \
            isinstance(self.postnet, Encoder_z0_ODE_RNN_cluster2):

            truth_w_mask = truth
            if self.use_mask:
                truth_w_mask = torch.cat((truth, mask), -1)

            if torch.isnan(truth_w_mask).any():
                print('here')
            if torch.isnan(truth_time_steps).any():
                print('here')

            infer_probs, latent_y_states,_ = self.postnet.run_odernn(
                truth_w_mask, truth_time_steps, run_backwards = run_backwards)

        else:
            raise Exception("Unknown encoder type {}".format(type(self.encoder_z0).__name__))
        
        '''return lenth: T_max - 1'''
        gen_y_probs, gen_latent_y_states = self.trans(torch.transpose(infer_probs.squeeze(0), 1, 0), truth_time_steps, run_backwards = run_backwards)
        extra_kl_div = 0

        squeezed_infer_probs = torch.transpose(infer_probs.squeeze(0), 1, 0)

        selected_time_count = truth_time_steps.shape[0] -  self.shift

        count = 0
        
        for k in range(selected_time_count - 1):
            time_steps = torch.tensor([truth_time_steps[k].item(), truth_time_steps[k + self.shift + 1].item()])
            
            if k == 0:
                last_gen_probs, last_gen_states = self.trans.run_odernn_single_step(infer_probs[:,k].clone(), time_steps, prev_y_state = torch.zeros_like(gen_latent_y_states[:,0]))
            else:
                last_gen_probs, last_gen_states = self.trans.run_odernn_single_step(infer_probs[:,k].clone(), time_steps, prev_y_state = gen_latent_y_states[:,k-1].clone())

            extra_kl_div += self.kl_div(squeezed_infer_probs[:, k+self.shift+1].clone(), last_gen_probs.squeeze(0))
            
            count += 1

        extra_kl_div = torch.sum(extra_kl_div)/(count*truth.shape[0])
        print(infer_probs.shape, gen_y_probs.shape)
        return torch.transpose(infer_probs.squeeze(0), 0, 1), torch.transpose(gen_y_probs.squeeze(0), 0, 1), latent_y_states.squeeze(0), gen_latent_y_states.squeeze(0), extra_kl_div
    
    
    def init_phi_table(self, init_values, is_tensor):
        """
        Initialize the phi_table with given initial values.
        
        Args:
            init_values (numpy.ndarray or Tensor): Initial values for the phi_table.
            is_tensor (bool): Indicates whether the init_values are already tensors.
        """
        if not is_tensor:
            self.phi_table.data.copy_(torch.t(torch.from_numpy(init_values)))
        else:
            self.phi_table.data.copy_(init_values)
         
        torch.save(self.phi_table, data_folder + '/' + output_dir + 'init_phi_table')


    def generate_z(self, z_category, prior_cluster_probs, t, h_now=None):
        """
        Generate latent variables z using the given z_category and prior cluster probabilities.
        
        Args:
            z_category (Tensor): Categorical distribution over cluster categories.
            prior_cluster_probs (Tensor): Prior cluster probabilities.
            t (int): Current time step.
            h_now (Tensor, optional): Current hidden state. Defaults to None.
        
        Returns:
            Tuple[Tensor, Tensor]: Tuple containing phi_z (cluster centroids), and z_representation (latent representation).
        """
        if self.use_gate:
            curr_gaussian_coeff = (self.gaussian_prior_coeff + F.sigmoid(self.gate_func(h_now)))/2

        else:
            curr_gaussian_coeff = self.gaussian_prior_coeff

        z_category = (1 - curr_gaussian_coeff)*z_category + curr_gaussian_coeff*prior_cluster_probs

        z_representation = z_category#.view(z_category.shape[1], z_category.shape[2])

        phi_z = torch.t(torch.mm(self.phi_table, torch.t(z_representation)))

        return phi_z, z_representation
    
    
#    def compute_cluster_obj(self, all_distances, all_probabs, T_max, x_lens, x_dim):
#        '''all_probabs: 1*cluster_num'''
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

    def calculate_imputation_losses(self, origin_x_to_pred, origin_x, x_to_predict_new_mask, imputed_x2, objective):
        """
        Calculate various imputation losses for the given imputed data.
        
        Args:
            origin_x_to_pred (Tensor): Original x data for imputation.
            origin_x (Tensor): Original x data.
            x_to_predict_new_mask (Tensor): Mask indicating new data for imputation.
            imputed_x2 (Tensor): Imputed x data.
            objective (str): Name of the objective (e.g., forecasting).
        
        Returns:
            Tuple of imputation loss values.
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
    

    def init_GPU(device, x, x_to_predict, x_mask, x_to_predict_mask, origin_x, origin_x_to_pred, 
                 origin_x_mask, new_x_mask, x_to_predict_origin_mask, x_to_predict_new_mask, x_lens,
                 x_to_predict_lens, x_time_stamps, x_to_predict_time_stamps):
        """
        Move data tensors to the specified GPU device.
        
        Args:
            device (str): Target GPU device.
            ... (various input tensors): Tensors to be moved to the GPU device.
        
        Returns:
            Tuple of tensors moved to the GPU device.
        """
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
        return x, x_to_predict, x_mask, x_to_predict_mask, origin_x, origin_x_to_pred, \
                 origin_x_mask, new_x_mask, x_to_predict_origin_mask, x_to_predict_new_mask, x_lens, \
                 x_to_predict_lens, x_time_stamps, x_to_predict_time_stamps
    
    def generate_x(self, phi_z, z_t):
        """
        Generate x data given phi_z and z_t.
        
        Args:
            phi_z (Tensor): Cluster centroids.
            z_t: Not used.
        
        Returns:
            Tuple of generated mean and logvar for x data.
        """
        mean = phi_z

        std = self.x_std*torch.ones_like(mean, device = self.device)
        
        logvar = 2*torch.log(std)
        
        return mean, logvar, mean
        
    def kl_div(self, cat_1, cat_2):
        """
        Compute the KL divergence between two categorical distributions.
        
        Args:
            cat_1 (Tensor): First categorical distribution.
            cat_2 (Tensor): Second categorical distribution.
        
        Returns:
            Tensor containing KL divergence values.
        """
        epsilon = 1e-5*torch.ones_like(cat_1)
        kl_div = torch.sum((cat_1+epsilon)*torch.log((cat_1 + epsilon)/(cat_2+epsilon)), 1)
        
        return kl_div
    
    
    def entropy(self, cat):
        """
        Compute the entropy of a categorical distribution.
        
        Args:
            cat (Tensor): Categorical distribution.
        
        Returns:
            Tensor containing entropy values.
        """
        epsilon = 1e-5*torch.ones_like(cat)
        kl_div = -torch.sum(cat*torch.log(cat+epsilon), 1)
        
        return kl_div


    def compute_gaussian_probs0(self, x, mean, logvar, mask):
        """
        Compute Gaussian probabilities for x data.
        
        Args:
            x (Tensor): x data.
            mean (Tensor): Mean values.
            logvar (Tensor): Log variance values.
            mask (Tensor): Mask indicating available data.
        
        Returns:
            Tuple of computed probabilities and squared differences.
        """
        std = torch.exp(0.5 * logvar)
        
        prob = 0.5*(((x - mean)/std)**2 + logvar + 2*np.log(np.sqrt(2*np.pi)))  # + torch.log((std*np.sqrt(2*np.pi)))
                
        return prob*mask, (x - mean)**2*mask


    def compute_rec_loss(self, joint_probs, prob_sums, full_curr_rnn_input, x_t, x_t_mask, h_now = None, curr_z_t_category_infer = None):
        """
        Compute reconstruction loss components.
        
        Args:
            joint_probs (Tensor): Joint probabilities.
            prob_sums (Tensor): Sum of probabilities.
            full_curr_rnn_input (Tensor): Full current RNN input.
            x_t (Tensor): Current x data.
            x_t_mask (Tensor): Mask for current x data.
            h_now (Tensor, optional): Current hidden state. Defaults to None.
            curr_z_t_category_infer (Tensor, optional): Inferred z category probabilities. Defaults to None.
        
        Returns:
            Tuple of various computed loss components.
        """
        phi_table_extend = (torch.t(self.phi_table)).clone()
        
        phi_table_extend = phi_table_extend.view(1, self.phi_table.shape[1], self.phi_table.shape[0])
        
        phi_table_extend = phi_table_extend.repeat(self.cluster_num, 1, 1) 
        
        phi_z_infer_full = torch.bmm(full_curr_rnn_input, phi_table_extend)
        mean_full, logvar_full, logit_x_t_full = self.generate_x(phi_z_infer_full, None)

        x_t_full = x_t.view(1, x_t.shape[0], x_t.shape[1])
        
        x_t_full = x_t_full.repeat(self.cluster_num, 1, 1)
        
        x_t_mask_full = x_t_mask.view(1, x_t_mask.shape[0], x_t_mask.shape[1])
        
        x_t_mask_full = x_t_mask_full.repeat(self.cluster_num, 1, 1)

        curr_full_rec_loss, curr_distances = self.compute_gaussian_probs0(x_t_full, mean_full, logvar_full, x_t_mask_full)

        
        if self.use_gate:
            curr_gaussian_coeff = (self.gaussian_prior_coeff + F.sigmoid(self.gate_func(h_now)))/2
        else:
            curr_gaussian_coeff = self.gaussian_prior_coeff
        
        full_rec_loss1 = torch.sum(curr_full_rec_loss*(1-curr_gaussian_coeff)*torch.t(joint_probs).unsqueeze(-1), 0)
        
        full_rec_loss2 = torch.sum(curr_full_rec_loss*(curr_gaussian_coeff)*prob_sums.view(prob_sums.shape[0], 1, 1), 0)
        
        l2_norm_loss = full_rec_loss1/(1-curr_gaussian_coeff)
        
        cluster_loss = full_rec_loss2/curr_gaussian_coeff

        full_logit_x_t = torch.mm((1-curr_gaussian_coeff)*curr_z_t_category_infer + curr_gaussian_coeff*prob_sums.view(1,-1), torch.t(self.phi_table))
        

        return full_rec_loss1, full_rec_loss2, full_logit_x_t, curr_full_rec_loss, l2_norm_loss, cluster_loss
    '''x, origin_x, x_mask, T_max, x_to_predict, origin_x_to_pred, x_to_predict_mask, tp_to_predict, is_GPU, device'''        


    def update_joint_probability(self, joint_probs, curr_rnn_output, batch_size, t, h_prev, c_prev, z_t_category_infer, shrinked_x_lens, x_t, x_t_mask):
        """
        Update the joint probability distribution of z_t across clusters.
        Args:
            joint_probs (Tensor): Current joint probability distribution over clusters.
            curr_rnn_output (Tensor): Current output of the RNN.
            batch_size (int): Batch size.
            t (int): Time step.
            h_prev (Tensor): Previous hidden state of the RNN.
            c_prev (Tensor): Previous cell state of the RNN (if applicable).
            z_t_category_infer (Tensor): Categorical distribution over clusters obtained from the inference network.
            shrinked_x_lens (Tensor): Shrinked sequence lengths for the current batch.
            x_t (Tensor): The input data at time step t.
            x_t_mask (Tensor): The mask for the input data at time step t.
        Returns:
            Tuple of updated joint probabilities, KL divergence loss, updated hidden state, updated cell state, reconstruction loss, logit x_t, and inferred cluster probabilities.
        """
        z_t_category_gen = F.softmax(self.emitter_z(h_prev), dim = 2)

        full_kl = self.kl_div(z_t_category_infer, z_t_category_gen.view(z_t_category_gen.shape[1], z_t_category_gen.shape[2]))

        full_rec_loss = 0
        
        full_logit_x_t = 0

        full_curr_rnn_input = torch.zeros((self.cluster_num, batch_size, self.cluster_num), dtype = torch.float, device = self.device)
        
        for k in range(self.cluster_num):
            curr_rnn_input = torch.zeros((batch_size, self.cluster_num), dtype = torch.float, device = self.device)
            
            curr_rnn_input[:,k] = 1
            
            full_curr_rnn_input[k] = curr_rnn_input
        
        t1 = time.time()
        
        phi_table_extend = (torch.t(self.phi_table)).clone()
        
        phi_table_extend = phi_table_extend.view(1, self.cluster_num, self.z_dim)
        
        phi_table_extend = phi_table_extend.repeat(self.cluster_num, 1, 1) 
            
        phi_z_infer_full = torch.bmm(full_curr_rnn_input, phi_table_extend)
        
        curr_rnn_output_full = curr_rnn_output.view(1, curr_rnn_output.shape[0], curr_rnn_output.shape[1])
        
        curr_rnn_output_full = curr_rnn_output_full.repeat(self.cluster_num, 1, 1)
        
        z_t, z_t_category_infer_full, _, z_category_infer_sparse = self.postnet(full_curr_rnn_input, curr_rnn_output_full, self.phi_table, t, self.temp)
        
        mean_full, logvar_full, logit_x_t_full = self.generate_x(phi_z_infer_full, None)

        updated_joint_probs = torch.sum(z_t_category_infer_full*torch.t(joint_probs).view(joint_probs.shape[1], joint_probs.shape[0], 1), 0)

        x_t_full = x_t.view(1, x_t.shape[0], x_t.shape[1])
        
        x_t_full = x_t_full.repeat(self.cluster_num, 1, 1)
        
        x_t_mask_full = x_t_mask.view(1, x_t.shape[0], x_t.shape[1])
        
        x_t_mask_full = x_t_mask_full.repeat(self.cluster_num, 1, 1)
        
        curr_full_rec_loss = compute_gaussian_probs0(x_t_full, mean_full, logvar_full, x_t_mask_full)

        if self.transfer_prob:
                
            z_t_transfer = updated_joint_probs
            
            if self.block == 'GRU':
                output, h_now = self.trans(z_t_transfer.view(z_t_transfer.shape[0], 1, z_t_transfer.shape[1]), h_prev.contiguous())# p(z_t| z_{t-1})
                
            else:
                output, (h_now, c_now) = self.trans(z_t_transfer.view(z_t_transfer.shape[0], 1, z_t_transfer.shape[1]), (h_prev.contiguous(), c_prev.contiguous()))# p(z_t| z_{t-1})

            z_t, z_t_category_infer, _ , z_category_infer_sparse= self.postnet(z_t_transfer, curr_rnn_output, self.phi_table, t, self.temp)
        else:
            
            phi_z_infer = torch.mm(updated_joint_probs, torch.t(self.phi_table))
            
            if self.block == 'GRU':
                output, h_now = self.trans(phi_z_infer.view(phi_z_infer.shape[0], 1, phi_z_infer.shape[1]).contiguous(), h_prev.contiguous())# p(z_t| z_{t-1})
                
            else:
                output, (h_now, c_now) = self.trans(phi_z_infer.view(phi_z_infer.shape[0], 1, phi_z_infer.shape[1]).contiguous(), (h_prev.contiguous(), c_prev.contiguous()))# p(z_t| z_{t-1})
            
            z_t, z_t_category_infer, _,z_category_infer_sparse = self.postnet(phi_z_infer, curr_rnn_output, self.phi_table, t, self.temp)
        
        for k in range(self.cluster_num):
            full_rec_loss += curr_full_rec_loss[k]*updated_joint_probs[:,k].view(curr_full_rec_loss[k].shape[0],1)
        full_logit_x_t = torch.mm(updated_joint_probs, torch.t(self.phi_table))
        
        if np.isnan(full_kl.cpu().detach().numpy()).any():
            print('distribution 1::', z_t_category_gen)
            
            print('distribution 2::', z_t_category_infer)
        

        return updated_joint_probs, full_kl, h_now, c_now, full_rec_loss, full_logit_x_t, z_t_category_infer


    def infer(self, x, origin_x, x_mask, origin_x_mask, new_x_mask, x_lens, x_to_predict, origin_x_to_pred, x_to_predict_mask, x_to_predict_origin_mask, x_to_predict_new_mask, x_to_predict_lens, is_GPU, device, x_time_stamps, x_to_predict_time_stamps):
        """
        Infer the variational distribution q(z_{1:T}|x_{1:T}).
        Args:
            x (Tensor): Observed data.
            origin_x (Tensor): Original observed data (without imputation).
            x_mask (Tensor): Mask for observed data.
            origin_x_mask (Tensor): Mask for original observed data.
            new_x_mask (Tensor): Mask for new observed data.
            x_lens (Tensor): Sequence lengths of observed data.
            x_to_predict (Tensor): Data to predict.
            origin_x_to_pred (Tensor): Original data to predict.
            x_to_predict_mask (Tensor): Mask for data to predict.
            x_to_predict_origin_mask (Tensor): Mask for original data to predict.
            x_to_predict_new_mask (Tensor): Mask for new data to predict.
            x_to_predict_lens (Tensor): Sequence lengths of data to predict.
            is_GPU (bool): Flag indicating whether to use GPU.
            device (str): Device for computation (e.g., 'cuda' or 'cpu').
            x_time_stamps (Tensor): Time stamps of observed data.
            x_to_predict_time_stamps (Tensor): Time stamps of data to predict.
        Returns:
            If not in evaluation mode, returns reconstruction loss, KL loss, first KL loss, final RMSE loss, interpolated loss, cluster loss, and final AE loss.
            If in evaluation mode, returns imputed data and imputation losses (if applicable).
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
                imputed_x, interpolated_x = self.impute.forward2(x, x_mask, x_time_stamps[0]*100)

                x = imputed_x

            else:      
                x = x_mask*x
                 
                interpolated_x = x
        
        batch_size, _, input_dim = x.size()
        
        h_0 = self.h_0.expand(1, batch_size, self.h_dim).contiguous()
        
        infer_probs, gen_probs, latent_y_states = self.get_reconstruction1(torch.cat([x_time_stamps[0].type(torch.FloatTensor), x_to_predict_time_stamps[0].type(torch.FloatTensor)], 0), x,x_time_stamps[0].type(torch.FloatTensor) ,x_mask, n_traj_samples = 1, run_backwards = True)
        
        rec_losses = torch.zeros((batch_size, T_max-1, self.z_dim), device=x.device)
            
        cluster_losses = torch.zeros((batch_size, T_max  -1, self.z_dim), device=x.device) 
        
        rec_losses_no_coeff = torch.zeros((batch_size, T_max-1, self.z_dim), device=x.device)
        
        cluster_losses_no_coeff = torch.zeros((batch_size, T_max  -1, self.z_dim), device=x.device) 
        
        imputed_x2 = torch.zeros_like(x)
        
        imputed_x2[:,0] = x[:,0]

        kl_states = torch.zeros((batch_size, T_max), device=x.device)
        
        rmse_losses = torch.zeros((batch_size, input_dim, T_max - 1), device=x.device)
        
        mae_losses = torch.zeros((batch_size, input_dim, T_max - 1), device=x.device)   
        
        joint_probs = torch.zeros([T_max, batch_size, self.cluster_num], dtype = torch.float, device = self.device)

        curr_x_lens = x_lens.clone()
        
        shrinked_x_lens = x_lens.clone()
        
        x_t = None
        
        x_t_mask = None
        
        prob_sums = 0
        
        gen_prior = torch.ones([batch_size, self.cluster_num], device = self.device, dtype = torch.float)/self.cluster_num
        
        for t in range(T_max):

            full_curr_rnn_input = torch.zeros((self.cluster_num, batch_size, self.cluster_num), dtype = torch.float, device = self.device)
            
            for k in range(self.cluster_num):
                curr_rnn_input = torch.zeros((batch_size, self.cluster_num), dtype = torch.float, device = self.device)
                curr_rnn_input[:,k] = 1
                full_curr_rnn_input[k] = curr_rnn_input
            
            '''phi_z_infer: phi_{z_t}'''
            
            if t > 0:
                kl = self.kl_div(infer_probs[:,t], gen_probs[:,t])
            else:
                kl = self.kl_div(infer_probs[:,t], gen_prior)
                
            kl_states[:,t] = kl
                        
            if t == 0:
                joint_probs[t, curr_x_lens > 0] = infer_probs[:,t].clone()
            else:

                if isinstance(self.postnet, Encoder_z0_ODE_RNN_cluster):
                    updated_joint_probs = self.postnet.update_joint_probs(x_t.shape[0], joint_probs[t-1, curr_x_lens > 0], t, latent_y_states, (x_time_stamps[0,t] - x_time_stamps[0,t-1]).type(torch.float).to(self.device), full_curr_rnn_input)
                    
                    joint_probs[t, curr_x_lens > 0] = updated_joint_probs
                else:
                    joint_probs[t, curr_x_lens > 0] = gen_probs[:,t].clone()

            if t < T_max - 1:
                x_t = x[curr_x_lens > 0,t+1,:]
                
                x_t_mask = x_mask[curr_x_lens > 0,t+1,:]
            if self.method_2 and isinstance(self.postnet, Encoder_z0_ODE_RNN_cluster):
                prob_sums += torch.sum(gen_probs[:,t], 0)
            else:
                prob_sums += torch.sum(infer_probs[:,t], 0)
            shrinked_x_lens = shrinked_x_lens[shrinked_x_lens > 0]
        
        prior_cluster_probs = prob_sums/torch.sum(x_lens)
        
        for t in range(T_max):
            
            full_curr_rnn_input = torch.zeros((self.cluster_num, batch_size, self.cluster_num), dtype = torch.float, device = self.device)
            
            for k in range(self.cluster_num):
                curr_rnn_input = torch.zeros((batch_size, self.cluster_num), dtype = torch.float, device = self.device)
                curr_rnn_input[:,k] = 1
                full_curr_rnn_input[k] = curr_rnn_input
            
            if t >= 1:                
                full_rec_loss1, full_rec_loss2, full_logit_x_t, curr_full_rec_loss, l2_norm_loss, cluster_loss = self.compute_rec_loss(joint_probs[t], prior_cluster_probs, full_curr_rnn_input, x_t, x_t_mask)
            
                rec_losses[curr_x_lens > 0,t-1] = full_rec_loss1
                
                cluster_losses[curr_x_lens > 0,t-1] = full_rec_loss2
                
                rec_losses_no_coeff[curr_x_lens > 0,t-1] = l2_norm_loss
                
                cluster_losses_no_coeff[curr_x_lens > 0,t-1] = cluster_loss

                imputed_x2[curr_x_lens > 0,t] = full_logit_x_t
                rmse_loss = (x_t*x_t_mask - full_logit_x_t*x_t_mask)**2
                
                mae_loss = torch.abs(x_t*x_t_mask - full_logit_x_t*x_t_mask)

                rmse_losses[curr_x_lens > 0,:, t-1] = rmse_loss
                
                mae_losses[curr_x_lens > 0,:,t-1] = mae_loss
            
            if t < T_max - 1:
                x_t = x[curr_x_lens > 0,t+1,:]
                
                x_t_mask = x_mask[curr_x_lens > 0,t+1,:]
            
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
            
            imputed_loss, imputed_loss2, imputed_mse_loss, imputed_mse_loss2 = \
                self.calculate_imputation_losses(origin_x[:,1:], x[:,1:], new_x_mask[:,1:], imputed_x2[:,1:], objective = 'training')

        self.evaluate_forecasting_errors0(x_to_predict, origin_x_to_pred, x_to_predict_mask, x_to_predict_origin_mask, x_to_predict_new_mask, x_to_predict_lens, T_max, prior_cluster_probs, gen_probs[:, T_max:], x_to_predict_time_stamps)
        print()
        
        if not self.evaluate:
            return rec_loss1, kl_loss, first_kl_loss, final_rmse_loss, interpolated_loss, rec_loss2, final_ae_loss
        else:
            
            if torch.sum(1-new_x_mask[:,1:]) > 0:
                return imputed_x2*(1-x_mask) + x*x_mask, (imputed_mse_loss, imputed_mse_loss2, imputed_loss, imputed_loss2)
            else:
                return imputed_x2*(1-x_mask) + x*x_mask, None


    def infer0(self, x, origin_x, x_mask, origin_x_mask, new_x_mask, x_lens, x_to_predict, origin_x_to_pred, x_to_predict_mask, x_to_predict_origin_mask, x_to_predict_new_mask, x_to_predict_lens, is_GPU, device, x_time_stamps, x_to_predict_time_stamps):
        """
        Infer the variational distribution q(z_{1:T}|x_{1:T}) using an alternative method.
        Args:
            x (Tensor): Observed data.
            origin_x (Tensor): Original observed data (without imputation).
            x_mask (Tensor): Mask for observed data.
            origin_x_mask (Tensor): Mask for original observed data.
            new_x_mask (Tensor): Mask for new observed data.
            x_lens (Tensor): Sequence lengths of observed data.
            x_to_predict (Tensor): Data to predict.
            origin_x_to_pred (Tensor): Original data to predict.
            x_to_predict_mask (Tensor): Mask for data to predict.
            x_to_predict_origin_mask (Tensor): Mask for original data to predict.
            x_to_predict_new_mask (Tensor): Mask for new data to predict.
            x_to_predict_lens (Tensor): Sequence lengths of data to predict.
            is_GPU (bool): Flag indicating whether to use GPU.
            device (str): Device for computation (e.g., 'cuda' or 'cpu').
            x_time_stamps (Tensor): Time stamps of observed data.
            x_to_predict_time_stamps (Tensor): Time stamps of data to predict.
        Returns:
            If not in evaluation mode, returns reconstruction loss, KL loss, first KL loss, final RMSE loss, interpolated loss, cluster loss, and final AE loss.
            If in evaluation mode, returns imputed data, forecasting-imputed data, and imputation losses (if applicable).
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
                imputed_x, interpolated_x = self.impute.forward2(x, x_mask, x_time_stamps[0]*100)

                x = imputed_x

            else:      
                x = x_mask*x
                 
                interpolated_x = x
            
        
        batch_size, _, input_dim = x.size()
        
        h_0 = self.h_0.expand(1, batch_size, self.h_dim).contiguous()
        
        last_state = None
        
        if self.method_2:
            infer_probs, gen_probs, latent_y_states, gen_latent_y_states, extra_kl_div = self.get_reconstruction2(torch.cat([x_time_stamps[0].type(torch.FloatTensor), x_to_predict_time_stamps[0].type(torch.FloatTensor)], 0), x,x_time_stamps[0].type(torch.FloatTensor) ,x_mask, n_traj_samples = 1, run_backwards = True)
        else:
            infer_probs, gen_probs, latent_y_states, gen_latent_y_states, extra_kl_div = self.get_reconstruction2(torch.cat([x_time_stamps[0].type(torch.FloatTensor), x_to_predict_time_stamps[0].type(torch.FloatTensor)], 0), x,x_time_stamps[0].type(torch.FloatTensor) ,x_mask, n_traj_samples = 1, run_backwards = False)
            last_state = latent_y_states[-1]
            
            
        rec_losses = torch.zeros((batch_size, T_max-1, self.z_dim), device=x.device)
            
        cluster_losses = torch.zeros((batch_size, T_max  -1, self.z_dim), device=x.device) 
        
        rec_losses_no_coeff = torch.zeros((batch_size, T_max-1, self.z_dim), device=x.device)
        
        cluster_losses_no_coeff = torch.zeros((batch_size, T_max  -1, self.z_dim), device=x.device) 
        
        imputed_x2 = torch.zeros_like(x)
        
        imputed_x2[:,0] = x[:,0]

        kl_states = torch.zeros((batch_size, T_max), device=x.device)
        
        rmse_losses = torch.zeros((batch_size, input_dim, T_max - 1), device=x.device)
        
        mae_losses = torch.zeros((batch_size, input_dim, T_max - 1), device=x.device)   
        
        joint_probs = torch.zeros([T_max, batch_size, self.cluster_num], dtype = torch.float, device = self.device)

        curr_x_lens = x_lens.clone()
        
        shrinked_x_lens = x_lens.clone()
        
        x_t = None
        
        x_t_mask = None
        
        prob_sums = 0
        
        gen_prior = torch.ones([batch_size, self.cluster_num], device = self.device, dtype = torch.float)/self.cluster_num
        
        for t in range(T_max):
            
            full_curr_rnn_input = torch.zeros((self.cluster_num, batch_size, self.cluster_num), dtype = torch.float, device = self.device)
            
            for k in range(self.cluster_num):
                curr_rnn_input = torch.zeros((batch_size, self.cluster_num), dtype = torch.float, device = self.device)
                curr_rnn_input[:,k] = 1
                full_curr_rnn_input[k] = curr_rnn_input
            
            '''phi_z_infer: phi_{z_t}'''
            
            if t > 0:
                kl = self.kl_div(infer_probs[:,t], gen_probs[:,t-1])
            else:
                kl = self.kl_div(infer_probs[:,t], gen_prior)

            kl_states[:,t] = kl

            if t == 0:
                
                joint_probs[t, curr_x_lens > 0] = infer_probs[:,t].clone()
            else:
                updated_joint_probs = self.postnet.update_joint_probs(x_t.shape[0], joint_probs[t-1, curr_x_lens > 0], t, latent_y_states, (x_time_stamps[0,t] - x_time_stamps[0,t-1]).type(torch.float).to(self.device), full_curr_rnn_input)
                                
                joint_probs[t, curr_x_lens > 0] = updated_joint_probs
            
            if t < T_max - 1:
                x_t = x[curr_x_lens > 0,t+1,:]
                
                x_t_mask = x_mask[curr_x_lens > 0,t+1,:]
                
            prob_sums += torch.sum(infer_probs[:,t], 0)
            shrinked_x_lens = shrinked_x_lens[shrinked_x_lens > 0]
        
        prior_cluster_probs = prob_sums/torch.sum(x_lens)
        
        for t in range(T_max):
            
            full_curr_rnn_input = torch.zeros((self.cluster_num, batch_size, self.cluster_num), dtype = torch.float, device = self.device)
            
            for k in range(self.cluster_num):
                curr_rnn_input = torch.zeros((batch_size, self.cluster_num), dtype = torch.float, device = self.device)
                curr_rnn_input[:,k] = 1
                full_curr_rnn_input[k] = curr_rnn_input
            
            if t >= 1:
                
                full_rec_loss1, full_rec_loss2, full_logit_x_t, curr_full_rec_loss, l2_norm_loss, cluster_loss = self.compute_rec_loss(joint_probs[t], prior_cluster_probs, full_curr_rnn_input, x_t, x_t_mask, latent_y_states[t-1], gen_probs[:,t-1])
            
                rec_losses[curr_x_lens > 0,t-1] = full_rec_loss1
                
                cluster_losses[curr_x_lens > 0,t-1] = full_rec_loss2
                
                rec_losses_no_coeff[curr_x_lens > 0,t-1] = l2_norm_loss
                
                cluster_losses_no_coeff[curr_x_lens > 0,t-1] = cluster_loss

                imputed_x2[curr_x_lens > 0,t] = full_logit_x_t
                rmse_loss = (x_t*x_t_mask - full_logit_x_t*x_t_mask)**2
                
                mae_loss = torch.abs(x_t*x_t_mask - full_logit_x_t*x_t_mask)

                rmse_losses[curr_x_lens > 0,:, t-1] = rmse_loss
                
                mae_losses[curr_x_lens > 0,:,t-1] = mae_loss
            
            if t < T_max - 1:
                x_t = x[curr_x_lens > 0,t+1,:]
                
                x_t_mask = x_mask[curr_x_lens > 0,t+1,:]
            
        rec_loss1 = torch.sum(rec_losses)/torch.sum(x_mask[:,1:,:])

        rec_loss2 = torch.sum(cluster_losses)/torch.sum(x_mask[:,1:,:])

        final_rec_loss = torch.sum(rec_losses_no_coeff)/torch.sum(x_mask[:,1:,:])

        final_cluster_loss = torch.sum(cluster_losses_no_coeff)/torch.sum(x_mask[:,1:,:])

        
        first_kl_loss = kl_states[:, 0].view(-1).mean()
        
        kl_loss = torch.sum(kl_states[:, 1:])/torch.sum(x_lens-1)
        final_rmse_loss = torch.sqrt(torch.sum(rmse_losses)/torch.sum(x_mask[:,1:,:]))
        
        final_mae_losses = torch.sum(mae_losses)/torch.sum(x_mask[:,1:,:])

        print('loss::', final_rec_loss, kl_loss, extra_kl_div)
        
        print('loss with coefficient::', rec_loss1, kl_loss, extra_kl_div)
        
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
            
            imputed_loss, imputed_loss2, imputed_mse_loss, imputed_mse_loss2 = \
                self.calculate_imputation_losses(origin_x[:,1:], x[:,1:], new_x_mask[:,1:], imputed_x2[:,1:], objective = 'training')

        forecasting_imputed_data = self.evaluate_forecasting_errors1(x_to_predict, origin_x_to_pred, x_to_predict_mask, x_to_predict_origin_mask, x_to_predict_new_mask, x_to_predict_lens, T_max, prior_cluster_probs, infer_probs[:, -1].unsqueeze(0), gen_latent_y_states[-1].unsqueeze(0) , x_time_stamps[0,-1], x_to_predict_time_stamps, last_state.unsqueeze(0))

        if not self.evaluate:
            
            if self.use_shift:
                return rec_loss1, kl_loss + extra_kl_div, first_kl_loss, final_rmse_loss, interpolated_loss, rec_loss2, final_ae_loss
            else:
                return rec_loss1, kl_loss, first_kl_loss, final_rmse_loss, interpolated_loss, rec_loss2, final_ae_loss
        else:
            
            if torch.sum(1-new_x_mask[:,1:]) > 0:
                return imputed_x2*(1-x_mask) + x*x_mask, forecasting_imputed_data*(1-x_to_predict_mask) + x_to_predict*x_to_predict_mask, (imputed_mse_loss, imputed_mse_loss2, imputed_loss, imputed_loss2)
            else:
                return imputed_x2*(1-x_mask) + x*x_mask, forecasting_imputed_data*(1-x_to_predict_mask) + x_to_predict*x_to_predict_mask, None


    def test_samples(self, x, origin_x, x_mask, origin_x_mask, new_x_mask, x_lens, x_to_predict, origin_x_to_pred, x_to_predict_mask, x_to_predict_origin_mask, x_to_predict_new_mask, x_to_predict_lens, is_GPU, device, x_delta_time_stamps, x_to_predict_delta_time_stamps, x_time_stamps, x_to_predict_time_stamps):
        """
        Computes various evaluation metrics and losses for forecasting and imputation tasks.

        Args:
            x (Tensor): Input data of shape (batch_size, sequence_length, input_dim).
            origin_x (Tensor): Original input data without any imputations.
            x_mask (Tensor): Mask indicating missing values in x of shape (batch_size, sequence_length, input_dim).
            origin_x_mask (Tensor): Mask indicating missing values in origin_x of shape (batch_size, sequence_length, input_dim).
            new_x_mask (Tensor): Mask indicating newly missing values in x_to_predict of shape (batch_size, sequence_length, input_dim).
            x_lens (Tensor): Lengths of sequences in x.
            x_to_predict (Tensor): Data to predict of shape (batch_size, sequence_length, input_dim).
            origin_x_to_pred (Tensor): Original x_to_predict data without any imputations.
            x_to_predict_mask (Tensor): Mask indicating missing values in x_to_predict of shape (batch_size, sequence_length, input_dim).
            x_to_predict_origin_mask (Tensor): Mask indicating missing values in origin_x_to_pred of shape (batch_size, sequence_length, input_dim).
            x_to_predict_new_mask (Tensor): Mask indicating newly missing values in x_to_predict of shape (batch_size, sequence_length, input_dim).
            x_to_predict_lens (Tensor): Lengths of sequences in x_to_predict.
            is_GPU (bool): Flag indicating whether GPU is being used.
            device (str): Device to perform computations on (e.g., 'cuda' or 'cpu').
            x_delta_time_stamps (Tensor): Time differences between consecutive time steps in x.
            x_to_predict_delta_time_stamps (Tensor): Time differences between consecutive time steps in x_to_predict.
            x_time_stamps (Tensor): Time stamps associated with x data.
            x_to_predict_time_stamps (Tensor): Time stamps associated with x_to_predict data.

        Returns:
            Tuple of evaluation metrics and imputation information:
            - final_rmse_loss (Tensor): Final RMSE loss for forecasting.
            - rmse_loss_count (Tensor): Total count of RMSE loss values used for computation.
            - final_mae_losses (Tensor): Final MAE loss for forecasting.
            - mae_loss_count (Tensor): Total count of MAE loss values used for computation.
            - final_nll_loss (Tensor): Final negative log likelihood loss for forecasting.
            - nll_loss_count (Tensor): Total count of NLL loss values used for computation.
            - list_res (Tuple): A tuple containing lists of RMSE and MAE losses for each time step, and the total count of losses.
            - imputation_info (Tuple): A tuple containing imputation loss values, counts, MSE losses, and counts, based on imputed and original data.
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
                imputed_x, interpolated_x = self.impute.forward2(x, x_mask, x_time_stamps[0]*100)
                x = imputed_x
            else:
                x = x_mask*x
                interpolated_x = x

        batch_size, _, input_dim = x.size()
        h_0 = self.h_0.expand(1, batch_size, self.h_dim).contiguous()
        last_states = None
        if self.method_2:
            infer_probs, gen_probs, latent_y_states, gen_latent_y_states, extra_kl_div = self.get_reconstruction2(torch.cat([x_time_stamps[0].type(torch.FloatTensor), x_to_predict_time_stamps[0].type(torch.FloatTensor)], 0), x, x_time_stamps[0].type(torch.FloatTensor), x_mask, n_traj_samples = 1, run_backwards = True)
        else:
            infer_probs, gen_probs, latent_y_states, gen_latent_y_states, extra_kl_div = self.get_reconstruction2(torch.cat([x_time_stamps[0].type(torch.FloatTensor), x_to_predict_time_stamps[0].type(torch.FloatTensor)], 0), x, x_time_stamps[0].type(torch.FloatTensor), x_mask, n_traj_samples = 1, run_backwards = False)
            last_states = latent_y_states[-1]

        rec_losses = torch.zeros((batch_size, T_max-1, self.z_dim), device=x.device)
        cluster_losses = torch.zeros((batch_size, T_max - 1, self.z_dim), device=x.device)
        rec_losses_no_coeff = torch.zeros((batch_size, T_max-1, self.z_dim), device=x.device)
        cluster_losses_no_coeff = torch.zeros((batch_size, T_max - 1, self.z_dim), device=x.device)
        imputed_x2 = torch.zeros_like(x)
        imputed_x2[:,0] = x[:,0]
        kl_states = torch.zeros((batch_size, T_max), device=x.device)
        rmse_losses = torch.zeros((batch_size, input_dim, T_max - 1), device=x.device)
        mae_losses = torch.zeros((batch_size, input_dim, T_max - 1), device=x.device)
        joint_probs = torch.zeros([T_max, batch_size, self.cluster_num], dtype=torch.float, device=self.device)
        curr_x_lens = x_lens.clone()
        shrinked_x_lens = x_lens.clone()
        x_t = None
        x_t_mask = None
        prob_sums = 0
        gen_prior = torch.ones([batch_size, self.cluster_num], device=self.device, dtype=torch.float) / self.cluster_num

        for t in range(T_max):
            full_curr_rnn_input = torch.zeros((self.cluster_num, batch_size, self.cluster_num), dtype=torch.float, device=self.device)
            for k in range(self.cluster_num):
                curr_rnn_input = torch.zeros((batch_size, self.cluster_num), dtype=torch.float, device=self.device)
                curr_rnn_input[:,k] = 1
                full_curr_rnn_input[k] = curr_rnn_input

            if t > 0:
                kl = self.kl_div(infer_probs[:,t], gen_probs[:,t-1])
            else:
                kl = self.kl_div(infer_probs[:,t], gen_prior)
            kl_states[:,t] = kl

            if t == 0:
                joint_probs[t, curr_x_lens > 0] = infer_probs[:,t].clone()
            else:
                updated_joint_probs = self.postnet.update_joint_probs(x_t.shape[0], joint_probs[t-1, curr_x_lens > 0], t, latent_y_states, (x_time_stamps[0,t] - x_time_stamps[0,t-1]).type(torch.float).to(self.device), full_curr_rnn_input)
                joint_probs[t, curr_x_lens > 0] = updated_joint_probs

            if t < T_max - 1:
                x_t = x[curr_x_lens > 0,t+1,:]
                x_t_mask = x_mask[curr_x_lens > 0,t+1,:]
            prob_sums += torch.sum(infer_probs[:,t], 0)
            shrinked_x_lens = shrinked_x_lens[shrinked_x_lens > 0]

        prior_cluster_probs = prob_sums / torch.sum(x_lens)

        for t in range(T_max):
            full_curr_rnn_input = torch.zeros((self.cluster_num, batch_size, self.cluster_num), dtype=torch.float, device=self.device)
            for k in range(self.cluster_num):
                curr_rnn_input = torch.zeros((batch_size, self.cluster_num), dtype=torch.float, device=self.device)
                curr_rnn_input[:,k] = 1
                full_curr_rnn_input[k] = curr_rnn_input

            if t >= 1:
                full_rec_loss1, full_rec_loss2, full_logit_x_t, curr_full_rec_loss, l2_norm_loss, cluster_loss = self.compute_rec_loss(joint_probs[t], prior_cluster_probs, full_curr_rnn_input, x_t, x_t_mask, latent_y_states[t-1], gen_probs[:,t-1])
                rec_losses[curr_x_lens > 0,t-1] = full_rec_loss1
                cluster_losses[curr_x_lens > 0,t-1] = full_rec_loss2
                rec_losses_no_coeff[curr_x_lens > 0,t-1] = l2_norm_loss
                cluster_losses_no_coeff[curr_x_lens > 0,t-1] = cluster_loss
                imputed_x2[curr_x_lens > 0,t] = full_logit_x_t
                rmse_loss = (x_t*x_t_mask - full_logit_x_t*x_t_mask)**2
                mae_loss = torch.abs(x_t*x_t_mask - full_logit_x_t*x_t_mask)
                rmse_losses[curr_x_lens > 0,:, t-1] = rmse_loss
                mae_losses[curr_x_lens > 0,:,t-1] = mae_loss

            if t < T_max - 1:
                x_t = x[curr_x_lens > 0,t+1,:]
                x_t_mask = x_mask[curr_x_lens > 0,t+1,:]

        rec_loss1 = torch.sum(rec_losses) / torch.sum(x_mask[:,1:,:])
        rec_loss2 = torch.sum(cluster_losses) / torch.sum(x_mask[:,1:,:])
        final_rec_loss = torch.sum(rec_losses_no_coeff) / torch.sum(x_mask[:,1:,:])
        final_cluster_loss = torch.sum(cluster_losses_no_coeff) / torch.sum(x_mask[:,1:,:])
        first_kl_loss = kl_states[:, 0].view(-1).mean()
        kl_loss = torch.sum(kl_states[:, 1:]) / torch.sum(x_lens-1)
        final_rmse_loss = torch.sqrt(torch.sum(rmse_losses) / torch.sum(x_mask[:,1:,:]))
        final_mae_losses = torch.sum(mae_losses) / torch.sum(x_mask[:,1:,:])
        final_ae_loss = 0
        interpolated_loss = 0

        if self.is_missing:
            interpolated_loss = torch.norm(interpolated_x*x_mask - x*x_mask)

        if torch.sum(1-new_x_mask[:,1:]) > 0:
            imputed_mse_loss = torch.sqrt(torch.sum((((origin_x[:,1:] - x[:,1:])**2)*(1-new_x_mask[:,1:])))/torch.sum(1-new_x_mask[:,1:]))
            imputed_mse_loss2 = torch.sqrt(torch.sum((((origin_x[:,1:] - imputed_x2[:,1:])**2)*(1-new_x_mask[:,1:])))/torch.sum(1-new_x_mask[:,1:]))
            imputed_loss = torch.sum((torch.abs(origin_x[:,1:] - x[:,1:])*(1-new_x_mask[:,1:])))/torch.sum(1-new_x_mask[:,1:])
            imputed_loss2 = torch.sum((torch.abs(origin_x[:,1:] - imputed_x2[:,1:])*(1-new_x_mask[:,1:])))/torch.sum(1-new_x_mask[:,1:])

        print('loss::', final_rec_loss, kl_loss, extra_kl_div)
        print('loss with coefficient::', rec_loss1, kl_loss, extra_kl_div)
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
            imputed_loss, imputed_loss2, imputed_mse_loss, imputed_mse_loss2 = \
                self.calculate_imputation_losses(origin_x[:,1:], x[:,1:], new_x_mask[:,1:], imputed_x2[:,1:], objective = 'training')

        final_rmse_loss, rmse_loss_count, final_mae_losses, mae_loss_count, final_nll_loss, nll_loss_count, list_res = self.evaluate_forecasting_errors1(x_to_predict, origin_x_to_pred, x_to_predict_mask, x_to_predict_origin_mask, x_to_predict_new_mask, x_to_predict_lens, T_max, prior_cluster_probs, infer_probs[:, -1].unsqueeze(0), gen_latent_y_states[-1].unsqueeze(0) , x_time_stamps[0,-1], x_to_predict_time_stamps, last_states.unsqueeze(0))
        print()
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

    def evaluate_forecasting_errors0(self, x, origin_x_to_pred, x_mask, x_to_predict_origin_mask, x_to_predict_new_mask, x_to_predict_lens, T_max_train, prior_cluster_probs, gen_probs, x_to_predict_time_stamps):
        """
        Evaluates forecasting errors and losses using a generative model approach without gating.

        Args:
            x (Tensor): Input data of shape (batch_size, sequence_length, input_dim).
            origin_x_to_pred (Tensor): Original x_to_predict data without any imputations.
            x_mask (Tensor): Mask indicating missing values in x of shape (batch_size, sequence_length, input_dim).
            x_to_predict_origin_mask (Tensor): Mask indicating missing values in x_to_predict of shape (batch_size, sequence_length, input_dim).
            x_to_predict_new_mask (Tensor): Mask indicating newly missing values in x_to_predict of shape (batch_size, sequence_length, input_dim).
            x_to_predict_lens (Tensor): Lengths of sequences in x_to_predict.
            T_max_train (int): Maximum time step in training data.
            prior_cluster_probs (Tensor): Prior probabilities for cluster assignments.
            gen_probs (Tensor): Generated probabilities for cluster assignments.
            x_to_predict_time_stamps (Tensor): Time stamps associated with x_to_predict data.

        Returns:
            Tuple of evaluation metrics and imputation information:
            - final_rmse_loss (Tensor): Final RMSE loss for forecasting.
            - rmse_loss_count (Tensor): Total count of RMSE loss values used for computation.
            - final_mae_losses (Tensor): Final MAE loss for forecasting.
            - mae_loss_count (Tensor): Total count of MAE loss values used for computation.
            - final_nll_loss (Tensor): Final negative log likelihood loss for forecasting.
            - nll_loss_count (Tensor): Total count of NLL loss values used for computation.
            - list_res (Tuple): A tuple containing lists of RMSE and MAE losses for each time step, and the total count of losses.
            - imputation_info (Tuple): A tuple containing imputation loss values, counts, MSE losses, and counts, based on imputed and original data.
        """
        T_max = x_to_predict_lens.max().cpu().item()

        rmse_losses = torch.zeros((x.shape[0], x.shape[2], T_max), device=x.device)
        
        mae_losses = torch.zeros((x.shape[0], x.shape[2], T_max), device=x.device)

        neg_nll_losses = torch.zeros((x.shape[0], self.z_dim, T_max), device=x.device)
        
        curr_x_to_predict_lens = x_to_predict_lens.clone()
        
        
        shrinked_x_to_predict_lens = x_to_predict_lens.clone() 
        
        imputed_x2 = torch.zeros_like(x)

        for t in range(T_max):

            phi_z, z_representation = self.generate_z(gen_probs[:,t], prior_cluster_probs, T_max_train + t, None)
            
            
            mean, std, logit_x_t = self.generate_x(phi_z, z_representation)
            
            imputed_x2[curr_x_to_predict_lens>0,t,:] = logit_x_t

            rec_loss = (x[curr_x_to_predict_lens>0,t,:]*x_mask[curr_x_to_predict_lens>0,t,:] - logit_x_t*x_mask[curr_x_to_predict_lens>0,t,:])**2
            
            mae_loss = torch.abs(x[curr_x_to_predict_lens>0,t,:]*x_mask[curr_x_to_predict_lens>0,t,:] - logit_x_t*x_mask[curr_x_to_predict_lens>0,t,:])
            
            rmse_losses[curr_x_to_predict_lens>0,:,t] = rec_loss
            
            mae_losses[curr_x_to_predict_lens>0,:,t] = mae_loss

            neg_nll_losses[curr_x_to_predict_lens>0,:,t] = compute_gaussian_probs0(x[curr_x_to_predict_lens>0,t,:], mean, std, x_mask[curr_x_to_predict_lens>0,t,:])

            curr_x_to_predict_lens -= 1
             
            shrinked_x_to_predict_lens -= 1
            
            shrinked_x_to_predict_lens = shrinked_x_to_predict_lens[shrinked_x_to_predict_lens > 0]
            
        final_rmse_loss = torch.sqrt(torch.sum(rmse_losses)/torch.sum(x_mask))
        
        final_mae_losses = torch.sum(mae_losses)/torch.sum(x_mask)
        
        final_nll_loss = torch.sum(neg_nll_losses)/torch.sum(x_mask)
        
        if np.isnan(neg_nll_losses.cpu().detach().numpy()).any():
            print('here')
        
        
        print('forecasting rmse loss::', final_rmse_loss)
        
        print('forecasting mae loss::', final_mae_losses)
        
        print('forecasting neg likelihood::', final_nll_loss)

        if torch.sum(1-x_to_predict_new_mask) > 0:
            imputed_loss, imputed_loss2, imputed_mse_loss, imputed_mse_loss2 = \
                self.calculate_imputation_losses(origin_x_to_pred, x, x_to_predict_new_mask, imputed_x2, objective = 'forecasting')
        
        all_masks = torch.sum(x_mask)
        
        rmse_list, mae_list = get_forecasting_res_by_time_steps(x, imputed_x2, x_mask)
        
        if not self.evaluate:
            return final_rmse_loss, torch.sum(x_mask), final_mae_losses, torch.sum(x_mask), final_nll_loss, torch.sum(x_mask), (rmse_list, mae_list, torch.sum(x_mask))
        else:
            return imputed_x2*(1-x_mask) + x*x_mask
        

    def evaluate_forecasting_errors1(self, x, origin_x_to_pred, x_mask, x_to_predict_origin_mask, x_to_predict_new_mask, x_to_predict_lens, T_max_train, prior_cluster_probs, last_gen_probs, last_gen_states, last_time_step, x_to_predict_time_stamps, last_infer_states):
        """
        Evaluates forecasting errors and losses using a generative model approach with gating.

        Args:
            x (Tensor): Input data of shape (batch_size, sequence_length, input_dim).
            origin_x_to_pred (Tensor): Original x_to_predict data without any imputations.
            x_mask (Tensor): Mask indicating missing values in x of shape (batch_size, sequence_length, input_dim).
            x_to_predict_origin_mask (Tensor): Mask indicating missing values in x_to_predict of shape (batch_size, sequence_length, input_dim).
            x_to_predict_new_mask (Tensor): Mask indicating newly missing values in x_to_predict of shape (batch_size, sequence_length, input_dim).
            x_to_predict_lens (Tensor): Lengths of sequences in x_to_predict.
            T_max_train (int): Maximum time step in training data.
            prior_cluster_probs (Tensor): Prior probabilities for cluster assignments.
            last_gen_probs (Tensor): Generated probabilities for cluster assignments at the last time step.
            last_gen_states (Tensor): Generated states at the last time step.
            last_time_step (Tensor): Time stamp associated with the last time step.
            x_to_predict_time_stamps (Tensor): Time stamps associated with x_to_predict data.
            last_infer_states (Tensor): Last inferred states.

        Returns:
            Tuple of evaluation metrics and imputation information:
            - final_rmse_loss (Tensor): Final RMSE loss for forecasting.
            - rmse_loss_count (Tensor): Total count of RMSE loss values used for computation.
            - final_mae_losses (Tensor): Final MAE loss for forecasting.
            - mae_loss_count (Tensor): Total count of MAE loss values used for computation.
            - final_nll_loss (Tensor): Final negative log likelihood loss for forecasting.
            - nll_loss_count (Tensor): Total count of NLL loss values used for computation.
            - list_res (Tuple): A tuple containing lists of RMSE and MAE losses for each time step, and the total count of losses.
            - imputation_info (Tuple): A tuple containing imputation loss values, counts, MSE losses, and counts, based on imputed and original data.
        """
        T_max = x_to_predict_lens.max().cpu().item()

        rmse_losses = torch.zeros((x.shape[0], x.shape[2], T_max), device=x.device)
        
        mae_losses = torch.zeros((x.shape[0], x.shape[2], T_max), device=x.device)
        
        rmse_losses2 = torch.zeros((x.shape[0], x.shape[2], T_max), device=x.device)
        
        mae_losses2 = torch.zeros((x.shape[0], x.shape[2], T_max), device=x.device)

        neg_nll_losses = torch.zeros((x.shape[0], self.z_dim, T_max), device=x.device)
        
        neg_nll_losses2 = torch.zeros((x.shape[0], self.z_dim, T_max), device=x.device)
        
        curr_x_to_predict_lens = x_to_predict_lens.clone()
                
        shrinked_x_to_predict_lens = x_to_predict_lens.clone() 
        
        imputed_x2 = torch.zeros_like(x)
        
        imputed_x2_2 = torch.zeros_like(x)

        predicted_time_stamp_list = []
        
        predicted_time_stamp_list.append(last_time_step)
        
        predicted_time_stamp_list.extend(x_to_predict_time_stamps[0].tolist())

        ode_sol = self.trans.z0_diffeq_solver(last_gen_states, torch.tensor(predicted_time_stamp_list, dtype =torch.float))
        
        print(ode_sol.shape)
        
        gen_probs = self.trans.emit_probs(ode_sol[:,:,1:].squeeze(0))
        
        print(gen_probs.shape)
        
        for t in range(T_max):

            time_steps = torch.tensor([last_time_step.item(), x_to_predict_time_stamps[0,t].item()])
            
            last_gen_probs, last_gen_states = self.trans.run_odernn_single_step(last_gen_probs[:,curr_x_to_predict_lens>0,:], time_steps, prev_y_state = last_gen_states)
            
            if self.use_gate:
                phi_z, z_representation = self.generate_z(last_gen_probs.squeeze(0), prior_cluster_probs, T_max_train + t, last_infer_states.squeeze(0))
                
                phi_z2, z_representation2 = self.generate_z(gen_probs[:,t], prior_cluster_probs, T_max_train + t, last_infer_states.squeeze(0))
            else:
                phi_z, z_representation = self.generate_z(last_gen_probs.squeeze(0), prior_cluster_probs, T_max_train + t, None)
                
                phi_z2, z_representation2 = self.generate_z(gen_probs[:,t], prior_cluster_probs, T_max_train + t, None)
                        
            mean, std, logit_x_t = self.generate_x(phi_z, z_representation)
            
            mean2, std2, logit_x_t2 = self.generate_x(phi_z2, z_representation2)
            
            last_time_step = x_to_predict_time_stamps[0,t]
            
            imputed_x2[curr_x_to_predict_lens>0,t,:] = logit_x_t
            
            imputed_x2_2[curr_x_to_predict_lens>0,t,:] = logit_x_t2
            
            
            if self.use_gate:
                
                if self.use_mask:
                                        
                    input_x = torch.cat([logit_x_t, torch.ones_like(logit_x_t)], -1)
                    
                    last_infer_states = self.postnet.run_odernn_single_step(input_x.unsqueeze(0), time_steps, prev_y_state = last_infer_states)
                    
                else:
                    input_x = logit_x_t
                    
                    last_infer_states = self.postnet.run_odernn_single_step(input_x.unsqueeze(0), time_steps, prev_y_state = last_infer_states)

            rec_loss = (x[curr_x_to_predict_lens>0,t,:]*x_mask[curr_x_to_predict_lens>0,t,:] - logit_x_t*x_mask[curr_x_to_predict_lens>0,t,:])**2
            
            mae_loss = torch.abs(x[curr_x_to_predict_lens>0,t,:]*x_mask[curr_x_to_predict_lens>0,t,:] - logit_x_t*x_mask[curr_x_to_predict_lens>0,t,:])
            
            rec_loss2 = (x[curr_x_to_predict_lens>0,t,:]*x_mask[curr_x_to_predict_lens>0,t,:] - logit_x_t2*x_mask[curr_x_to_predict_lens>0,t,:])**2
            
            mae_loss2 = torch.abs(x[curr_x_to_predict_lens>0,t,:]*x_mask[curr_x_to_predict_lens>0,t,:] - logit_x_t2*x_mask[curr_x_to_predict_lens>0,t,:])
            
            rmse_losses[curr_x_to_predict_lens>0,:,t] = rec_loss
            
            mae_losses[curr_x_to_predict_lens>0,:,t] = mae_loss
            
            rmse_losses2[curr_x_to_predict_lens>0,:,t] = rec_loss2
            
            mae_losses2[curr_x_to_predict_lens>0,:,t] = mae_loss2
            
            neg_nll_losses[curr_x_to_predict_lens>0,:,t] = compute_gaussian_probs0(x[curr_x_to_predict_lens>0,t,:], mean, std, x_mask[curr_x_to_predict_lens>0,t,:])
            
            neg_nll_losses2[curr_x_to_predict_lens>0,:,t] = compute_gaussian_probs0(x[curr_x_to_predict_lens>0,t,:], mean2, std2, x_mask[curr_x_to_predict_lens>0,t,:])

            curr_x_to_predict_lens -= 1
             
            shrinked_x_to_predict_lens -= 1
            
            shrinked_x_to_predict_lens = shrinked_x_to_predict_lens[shrinked_x_to_predict_lens > 0]

        final_rmse_loss = torch.sqrt(torch.sum(rmse_losses)/torch.sum(x_mask))
        
        final_mae_losses = torch.sum(mae_losses)/torch.sum(x_mask)
        
        final_nll_loss = torch.sum(neg_nll_losses)/torch.sum(x_mask)
        
        final_rmse_loss2 = torch.sqrt(torch.sum(rmse_losses2)/torch.sum(x_mask))
        
        final_mae_losses2 = torch.sum(mae_losses2)/torch.sum(x_mask)
        
        final_nll_loss2 = torch.sum(neg_nll_losses2)/torch.sum(x_mask)
        
        if np.isnan(neg_nll_losses.cpu().detach().numpy()).any():
            print('here')
        
        
        print('forecasting rmse loss::', final_rmse_loss)
        
        print('forecasting mae loss::', final_mae_losses)
        
        print('forecasting neg likelihood::', final_nll_loss)

        print('forecasting rmse loss 2::', final_rmse_loss2)
        
        print('forecasting mae loss 2::', final_mae_losses2)
        
        print('forecasting neg likelihood 2::', final_nll_loss2)
        
        if torch.sum(1-x_to_predict_new_mask) > 0:
            imputed_loss, imputed_loss2, imputed_mse_loss, imputed_mse_loss2 = \
                self.calculate_imputation_losses(origin_x_to_pred, x, x_to_predict_new_mask, imputed_x2, objective = 'forecasting')
        
        all_masks = torch.sum(x_mask)
        
        
        rmse_list, mae_list = get_forecasting_res_by_time_steps(x, imputed_x2, x_mask)
        
        rmse_list2, mae_list2 = get_forecasting_res_by_time_steps(x, imputed_x2_2, x_mask)
        
        if not self.evaluate:
            return (final_rmse_loss, final_rmse_loss2), torch.sum(x_mask), (final_mae_losses, final_mae_losses2), torch.sum(x_mask), (final_nll_loss, final_nll_loss2), torch.sum(x_mask), ((rmse_list, rmse_list2), (mae_list, mae_list2), torch.sum(x_mask))
        else:
            return imputed_x2*(1-x_mask) + x*x_mask


    def evaluate_forecasting_errors(self, x, origin_x_to_pred, x_mask, x_to_predict_origin_mask, x_to_predict_new_mask, x_to_predict_lens, h_now, c_now, T_max_train, prior_cluster_probs, curr_rnn_out, last_h_n, last_c_n, last_decoded_h_n, last_decoded_c_n):
        """
        Evaluates forecasting errors and losses using the generative model approach.

        Args:
            x (Tensor): Input data of shape (batch_size, sequence_length, input_dim).
            origin_x_to_pred (Tensor): Original x_to_predict data without any imputations.
            x_mask (Tensor): Mask indicating missing values in x of shape (batch_size, sequence_length, input_dim).
            x_to_predict_origin_mask (Tensor): Mask indicating missing values in x_to_predict of shape (batch_size, sequence_length, input_dim).
            x_to_predict_new_mask (Tensor): Mask indicating newly missing values in x_to_predict of shape (batch_size, sequence_length, input_dim).
            x_to_predict_lens (Tensor): Lengths of sequences in x_to_predict.
            h_now (Tensor): Current hidden states of the model.
            c_now (Tensor): Current cell states of the model (if applicable).
            T_max_train (int): Maximum time step in the training data.
            prior_cluster_probs (Tensor): Prior probabilities for cluster assignments.
            curr_rnn_out (Tensor): Current RNN outputs.
            last_h_n (Tensor): Last hidden states of the model.
            last_c_n (Tensor): Last cell states of the model (if applicable).
            last_decoded_h_n (Tensor): Last decoded hidden states of the model.
            last_decoded_c_n (Tensor): Last decoded cell states of the model (if applicable).

        Returns:
            Tuple of evaluation metrics and imputation information:
            - final_rmse_loss (Tensor): Final RMSE loss for forecasting.
            - final_mae_losses (Tensor): Final MAE loss for forecasting.
            - final_nll_loss (Tensor): Final negative log likelihood loss for forecasting.
            - imputed_x2 (Tensor): Imputed x data.
            - (Optional) other evaluation metrics or losses based on the implementation.
        """
        T_max = x_to_predict_lens.max().cpu().item()
        
        if self.is_missing:
            imputed_x, interpolated_x = self.impute.forward2(x[:,0:T_max,:], x_mask[:,0:T_max,:], T_max)
            x = imputed_x
        
        rmse_losses = torch.zeros((x.shape[0], x.shape[2], T_max), device=x.device)
        
        mae_losses = torch.zeros((x.shape[0], x.shape[2], T_max), device=x.device)
        
        neg_nll_losses = torch.zeros((x.shape[0], self.z_dim, T_max), device=x.device)
        
        curr_x_to_predict_lens = x_to_predict_lens.clone()
                
        shrinked_x_to_predict_lens = x_to_predict_lens.clone() 
        
        imputed_x2 = torch.zeros_like(x)
        
        last_h_n2 = last_h_n.clone()
        
        last_c_n2 = last_c_n.clone()
        
        for t in range(T_max):

            z_t_category_gen = F.softmax(self.emitter_z(h_now), dim = 2)

            phi_z, z_representation = self.generate_z(z_t_category_gen, prior_cluster_probs, T_max_train + t, curr_rnn_out)

            
            mean, std, logit_x_t = self.generate_x(phi_z, z_representation)
            

            imputed_x2[curr_x_to_predict_lens>0,t,:] = logit_x_t

            if self.use_gate:
                curr_rnn_out, (last_h_n, last_c_n) = self.x_encoder(logit_x_t.view(logit_x_t.shape[0], 1, logit_x_t.shape[1]), torch.ones(logit_x_t.shape[0], device = self.device), init_h = last_h_n, init_c = last_c_n)

            rec_loss = (x[curr_x_to_predict_lens>0,t,:]*x_mask[curr_x_to_predict_lens>0,t,:] - logit_x_t*x_mask[curr_x_to_predict_lens>0,t,:])**2
            
            mae_loss = torch.abs(x[curr_x_to_predict_lens>0,t,:]*x_mask[curr_x_to_predict_lens>0,t,:] - logit_x_t*x_mask[curr_x_to_predict_lens>0,t,:])
            
            rmse_losses[curr_x_to_predict_lens>0,:,t] = rec_loss

            mae_losses[curr_x_to_predict_lens>0,:,t] = mae_loss

            neg_nll_losses[curr_x_to_predict_lens>0,:,t] = compute_gaussian_probs0(x[curr_x_to_predict_lens>0,t,:], mean, std, x_mask[curr_x_to_predict_lens>0,t,:])

            z_t_category_gen_trans = z_t_category_gen

            if self.transfer_prob:
                if self.block == 'GRU':
                    output, h_now = self.trans(z_t_category_gen_trans.view(z_t_category_gen_trans.shape[1], 1, z_t_category_gen_trans.shape[2]), h_now)# p(z_t| z_{t-1})
                else:
                    output, (h_now, c_now) = self.trans(z_t_category_gen_trans.view(z_t_category_gen_trans.shape[1], 1, z_t_category_gen_trans.shape[2]), (h_now, c_now))# p(z_t| z_{t-1})
            
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

        final_rmse_loss = torch.sqrt(torch.sum(rmse_losses)/torch.sum(x_mask))
        
        final_mae_losses = torch.sum(mae_losses)/torch.sum(x_mask)
        
        final_nll_loss = torch.sum(neg_nll_losses)/torch.sum(x_mask)
        
        if np.isnan(neg_nll_losses.cpu().detach().numpy()).any():
            print('here')
        
        
        print('forecasting rmse loss::', final_rmse_loss)
        
        print('forecasting mae loss::', final_mae_losses)
        
        print('forecasting neg likelihood::', final_nll_loss)
        
        if self.is_missing:
            print('forecasting interpolate loss:', torch.norm(interpolated_x*x_mask - x*x_mask))
        
        if torch.sum(1-x_to_predict_new_mask) > 0:
            imputed_loss, imputed_loss2, imputed_mse_loss, imputed_mse_loss2 = \
                self.calculate_imputation_losses(origin_x_to_pred, x, x_to_predict_new_mask, imputed_x2, objective = 'forecasting')
        
        all_masks = torch.sum(x_mask)
        
        rmse_list, mae_list = get_forecasting_res_by_time_steps(x, imputed_x2, x_mask)
        
        if not self.evaluate:
            return final_rmse_loss, torch.sum(x_mask), final_mae_losses, torch.sum(x_mask), final_nll_loss, torch.sum(x_mask), (rmse_list, mae_list, torch.sum(x_mask))
        else:
            return imputed_x2*(1-x_mask) + x*x_mask


    def init_params(self):
        """
        Initializes the parameters of the model.
        """
        for p in self.parameters():
            if len(p.shape) >= 2:
                torch.nn.init.xavier_normal_(p, 1)
    
    
    def compute_phi_table_gap(self):
        """
        Computes the gap in the phi table.

        Returns:
            distance2 (Tensor): Gap in the phi table.
        """
        d = self.centroid_max
        cluster_diff = torch.norm(self.phi_table.view(self.phi_table.shape[0], 1, self.phi_table.shape[1]) 
                                  - self.phi_table.view(self.phi_table.shape[0], self.phi_table.shape[1], 1), dim=0)
        
        distance2 = torch.sum(torch.triu(F.relu(d - cluster_diff)**2, diagonal=1))

        return distance2


    def get_regularization_term(self):
        """
        Computes the regularization term for the model's parameters.

        Returns:
            reg_term (Tensor): Regularization term.
        """
        reg_term = 0
        
        for param in self.parameters():
            reg_term += torch.norm(param.view(-1))**2
            
        return reg_term


    def train_AE(self, x, origin_x, x_mask, origin_x_mask, new_x_mask, x_lens, kl_anneal, 
                 x_to_predict, origin_x_to_pred, x_to_predict_mask, x_to_predict_origin_mask,
                 x_to_predict_new_mask, x_to_predict_lens, is_GPU, device, x_time_stamps, x_to_predict_time_stamps):
        """
        Trains the autoencoder model.

        Args:
            x (Tensor): Input data of shape (batch_size, sequence_length, input_dim).
            origin_x (Tensor): Original input data without any imputations.
            x_mask (Tensor): Mask indicating missing values in x of shape (batch_size, sequence_length, input_dim).
            origin_x_mask (Tensor): Mask indicating missing values in origin_x of shape (batch_size, sequence_length, input_dim).
            new_x_mask (Tensor): Mask indicating newly missing values in x_to_predict of shape (batch_size, sequence_length, input_dim).
            x_lens (Tensor): Lengths of sequences in x.
            kl_anneal (float): KL annealing factor.
            x_to_predict (Tensor): Data to predict of shape (batch_size, sequence_length, input_dim).
            origin_x_to_pred (Tensor): Original x_to_predict data without any imputations.
            x_to_predict_mask (Tensor): Mask indicating missing values in x_to_predict of shape (batch_size, sequence_length, input_dim).
            x_to_predict_origin_mask (Tensor): Mask indicating missing values in origin_x_to_pred of shape (batch_size, sequence_length, input_dim).
            x_to_predict_new_mask (Tensor): Mask indicating newly missing values in x_to_predict of shape (batch_size, sequence_length, input_dim).
            x_to_predict_lens (Tensor): Lengths of sequences in x_to_predict.
            is_GPU (bool): Flag indicating whether GPU is being used.
            device (str): Device to perform computations on (e.g., 'cuda' or 'cpu').
            x_time_stamps (Tensor): Time stamps associated with x data.
            x_to_predict_time_stamps (Tensor): Time stamps associated with x_to_predict data.

        Returns:
            train_losses (dict): Dictionary containing training losses.
        """
        rec_loss1, kl_loss, first_kl_loss, final_rmse_loss, interpolated_loss, rec_loss2, final_ae_loss = \
            self.infer0(x, origin_x, x_mask, origin_x_mask, new_x_mask, x_lens, x_to_predict, origin_x_to_pred, 
                        x_to_predict_mask, x_to_predict_origin_mask, x_to_predict_new_mask, x_to_predict_lens, 
                        is_GPU, device, x_time_stamps, x_to_predict_time_stamps)
        
        loss = rec_loss1 + kl_anneal*kl_loss + 5e-5*first_kl_loss + rec_loss2 + 0.0001*interpolated_loss+ (10 + 10* kl_anneal)*final_ae_loss# + 0.01*interpolated_loss# + 0.001*self.get_regularization_term()#+ 0.001*self.compute_phi_table_gap()# + 0.001*final_entropy_loss#

        self.optimizer.zero_grad()

        loss.backward()
        nn.utils.clip_grad_norm_(self.parameters(), self.clip_norm)
        self.optimizer.step()
        
        return {'train_loss_AE':loss.item(), 'train_loss_KL':kl_loss.item()}

    
    def valid(self, x, x_rev, x_lens):
        """
        Evaluates the validation loss of the model.

        Args:
            x (Tensor): Input data of shape (batch_size, sequence_length, input_dim).
            x_rev (Tensor): Reversed input data.
            x_lens (Tensor): Lengths of sequences in x.

        Returns:
            loss (Tensor): Validation loss.
        """
        self.eval()
        rec_loss, kl_loss = self.infer(x, x_rev, x_lens)
        loss = rec_loss + kl_loss
        return loss
    

