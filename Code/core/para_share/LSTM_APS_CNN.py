from __future__ import division
import numpy as np
import tensorflow as tf
from sklearn.metrics.cluster import normalized_mutual_info_score
from sklearn.cluster import KMeans
import scipy.io as sio

# import core.general.LSTM_APS_CNN_net as LAC_net

def get_key(item):
    return item[0]


class APS:
    # LAC.APS(self.aps_config, att_config, gat_config, mlp_task_config, cnn_config, training_iters, batch_size, lr, lambda_l2_reg, theta)
    def __init__(self, T, aps_config):

        self.aps_config = aps_config
        self.T = T
        # print('T = ', self.T)
        self.num_slstm = self.aps_config[0][0]
        self.num_layer = len(self.aps_config[1])
        self.H_p_s = []
        self.c_p_s = []
        self.Att = []
        self.in_mlp  = []
        self.out_mlp = []
        self.h_list_tensor_temp_tsr = []

        for j in range(self.T):
            Att_tmp = []
            in_mlp_tmp = []
            out_mlp_tmp = []
            for i in range(self.num_layer):
                Att_tmp.append([])
                in_mlp_tmp.append([])
                out_mlp_tmp.append([])
            self.Att.append(Att_tmp)
            self.in_mlp.append(in_mlp_tmp)
            self.out_mlp.append(out_mlp_tmp)

    @staticmethod
    def gaussian_normalization(train_x):
        mu = np.mean(train_x, axis=0)
        dev = np.std(train_x, axis=0)
        norm_x = (train_x - mu) / (dev + 1e-12)
        # print norm_x
        return norm_x

    @staticmethod
    def minmax_normalization(x, base):
        min_val = np.min(base, axis=0)
        max_val = np.max(base, axis=0)
        norm_x = (x - min_val) / (max_val - min_val + 1e-12)
        # print norm_x
        return norm_x

    def multilayer_perceptron(self, x, config):
        inpt = x
        for i in range(len(config)-1):
            W = tf.Variable(tf.truncated_normal(shape=[config[i], config[i+1]], mean=-0.1, stddev=0.1, dtype=tf.float32))
            b = tf.Variable(tf.constant(0.1, shape=[config[i+1]], dtype=tf.float32))

            # print('config: ', config)
            # print('W: ', W)

            outp = tf.nn.leaky_relu(tf.add(tf.matmul(inpt, W), b))
            inpt = outp

        return outp

    def run(self, X, n_hidden_units, n_hidden_units_s, weights, biases):

        n_dim = X.get_shape()[2].value # n_dim = 42
        d = tf.cast(n_hidden_units/n_dim, tf.int32) # d = 4

        for i in range(self.T):
            # print("index i in T:", i)
            if i == 0:
                #--------- shared LSTM ---------#
                for j in range(self.num_slstm):
                    # Eq.(1)
                    Wx_temp_sj  = tf.transpose(weights['Wx_s'+str(j+1)]) # 50*42
                    xt_temp_sj  = X[:,i,:] # 6400*42
                    xt_sj       = tf.transpose(xt_temp_sj) # 42*6400
                    J_t_temp_sj = tf.matmul(Wx_temp_sj, xt_sj) # 50*6400
                    J_t_sj   = tf.tanh(tf.transpose(J_t_temp_sj) + biases['bj_s'+str(j+1)]) # 6400*50 + 50 = 6400*50
                    vec_J_t_sj  = J_t_sj # 6400*50
                    
                    # Eq.(3)
                    c_c_sj      = vec_J_t_sj # 6400*50
                    
                    # Eq.(4)
                    H_c_temp_sj = tf.tanh(c_c_sj) # 6400*50
                    H_c_sj      = tf.transpose(H_c_temp_sj) # 50*6400
                    
                    # update H and c
                    self.H_p_s.append(H_c_sj)  # [36*6400, 36*6400, 36*6400]
                    self.c_p_s.append(c_c_sj)  # [6400*36, 6400*36, 6400*36]

                #--------- sub-network modularization ---------#
                for j in range(self.num_slstm):
                    self.H_p_s[j] = tf.transpose(self.H_p_s[j]) # [6400*36, 6400*36, 6400*36]

                for i_ly in range(self.num_layer):
                    
                    if i_ly == 0:
                        H_list = self.H_p_s
                        num_pre_subs = self.num_slstm
                    else:
                        H_list = self.out_mlp[i][i_ly-1]
                        num_pre_subs = len(self.out_mlp[i][i_ly-1])

                    list_H_extend = []
                    for j in range(num_pre_subs):
                        list_H_extend.append(tf.expand_dims(H_list[j], 2)) # [6400*36*1, 6400*36*1, 6400*36*1]
                    H_com = tf.concat(list_H_extend, 2) # 6400*36*3
                    H_com = tf.transpose(H_com, perm=[1, 2, 0]) # 36*3*6400

                    for k in range(self.aps_config[1][i_ly][0]):

                        temp_sk = tf.tanh(tf.einsum('ij,jkl->ikl', weights['V_mlp'+str(i_ly+1)+str(k+1)], H_com)) # L*3*6400
                        self.Att[i][i_ly].append(tf.nn.softmax(tf.einsum('i,ijk->jk', weights['W_mlp'+str(i_ly+1)+str(k+1)], temp_sk), dim=0)) # len = n_sub'; [3*6400, 3*6400, 3*6400]

                        # 1st layer
                        self.in_mlp[i][i_ly].append(tf.matmul(tf.transpose(H_com, perm=[2, 0, 1]), tf.expand_dims(tf.transpose(self.Att[i][i_ly][k]), 2))) # 6400*36*3 * 6400*3*1 -> 6400*36*1

                        self.out_mlp[i][i_ly].append(self.multilayer_perceptron(self.in_mlp[i][i_ly][k][:,:,0], self.aps_config[1][i_ly][1])) # 6400*20

                #--------- tensorized LSTM ---------#
                # Eq.(1)
                Wx_temp_tsr  = tf.expand_dims(weights['Wx_tsr'], 2) # 42*4*1
                xt_temp_tsr  = X[:,i,:] # 6400*42
                xt_tsr       = tf.expand_dims(tf.transpose(xt_temp_tsr), 1) # 42*1*6400
                J_t_temp_tsr = tf.matmul(Wx_temp_tsr, xt_tsr) # 42*4*6400
                J_t_tsr      = tf.tanh(tf.transpose(J_t_temp_tsr, perm=[2, 0, 1]) + biases['bj_tsr']) # 6400*42*4 + 42*4 = 6400*42*4
                # vec_J_t_tsr  = tf.reshape(J_t_tsr, [-1, n_hidden_units]) # 6400*168
                
                # Eq.(3)
                c_c_tsr      = J_t_tsr # 6400*42*4

                # Eq.(4)
                # generate gate 'st_tsr'
                # self.out_mlp[i][-1][0]: 6400*50
                # weights['U_mlp_C1']: 42*4*50
                st_tsr_p1 = tf.matmul(weights['U_cc'], tf.transpose(c_c_tsr, perm=[1, 2, 0])) # 42*4*4 * 42*4*6400 = 42*4*6400
                stack_out_mlp = tf.transpose(tf.stack(self.out_mlp[i][-1]), perm=[0, 2, 1]) # 42*50*6400 
                # print(stack_out_mlp)
                # st_tsr_p2 = tf.einsum('ijk,kl->ijl', weights['U_mlp_C1'], tf.transpose(self.out_mlp[i][-1][0])) # 42*4*50 (*) 50*6400 = 42*4*6400
                st_tsr_p2 = tf.matmul(weights['U_mlp_C1'], stack_out_mlp) # 42*4*50 * 42*50*6400 = 42*4*6400
                st_tsr = tf.sigmoid(st_tsr_p1 + st_tsr_p2) # 42*4*6400
                st_tsr = tf.transpose(st_tsr, perm=[2, 0, 1]) # 6400*42*4
                # mem_ex_tmp = tf.einsum('ijk,kl->ijl', weights['U_tsr'], tf.transpose(self.out_mlp[i][-1][0])) # 42*4*6400
                mem_ex_tmp = tf.matmul(weights['U_tsr'], stack_out_mlp) # 42*4*50 * 42*50*6400 = 42*4*6400
                mem_ex = tf.multiply(st_tsr, tf.transpose(mem_ex_tmp, perm=[2, 0, 1])) # 6400*42*4

                H_c_temp_tsr = tf.tanh(c_c_tsr + mem_ex) # 6400*42*4
                H_c_tsr      = tf.transpose(H_c_temp_tsr, perm=[1, 2, 0]) # 42*4*6400

                # update H and c
                H_p_tsr = H_c_tsr # 42*4*6400
                c_p_tsr = c_c_tsr # 6400*42*4

                self.h_list_tensor_temp_tsr.append(tf.transpose(H_p_tsr, perm=[2, 0, 1]))      


            else:
                #--------- shared LSTM ---------#
                for j in range(self.num_slstm):
                    # Eq.(1)
                    Wx_temp_sj  = tf.transpose(weights['Wx_s'+str(j+1)]) # 50*42
                    xt_temp_sj  = X[:,i,:] # 6400*42
                    xt_sj       = tf.transpose(xt_temp_sj) # 42*6400
                    # self.H_p_s[j]: 6400*36
                    J_t_temp_sj = tf.matmul(Wx_temp_sj, xt_sj) + tf.matmul(weights['Wh_s'+str(j+1)], tf.transpose(self.H_p_s[j]))  # (36*42)*(42*6400) + (36*36)*(36*6400)  ->  (36*6400)
                    J_t_sj   = tf.tanh(tf.transpose(J_t_temp_sj) + biases['bj_s'+str(j+1)]) # 6400*50 + 50 = 6400*50
                    vec_J_t_sj  = J_t_sj # 6400*50
                    
                    # Eq.(2)        
                    ifo_temp_sj = tf.matmul(weights['W3_s'+str(j+1)], tf.concat([xt_sj, tf.transpose(self.H_p_s[j])], 0)) # ((3*50)*(42+50)) * ((42+50)*6400) = (3*50)*6400
                    ifo_sj      = tf.sigmoid(tf.transpose(ifo_temp_sj) + biases['b3_s'+str(j+1)]) # 6400*(3*50)

                    # Eq.(3)
                    it_sj       = ifo_sj[:,0:n_hidden_units_s] # 6400*50
                    ft_sj       = ifo_sj[:,n_hidden_units_s:n_hidden_units_s*2] # 6400*50
                    ot_sj       = ifo_sj[:,n_hidden_units_s*2:n_hidden_units_s*3] # 6400*50
                    c_c_sj      = tf.multiply(ft_sj, self.c_p_s[j]) + tf.multiply(it_sj, vec_J_t_sj) # 6400*50
                    
                    # Eq.(4)
                    H_c_temp_sj = tf.multiply(ot_sj, tf.tanh(c_c_sj)) # 6400*50
                    H_c_sj      = tf.transpose(H_c_temp_sj) # 50*6400

                    # update H and c
                    self.H_p_s[j] = H_c_sj # 36*6400                    
                    self.c_p_s[j] = c_c_sj # 6400*50

                #--------- sub-network modularization ---------#
                for j in range(self.num_slstm):
                    self.H_p_s[j] = tf.transpose(self.H_p_s[j]) # 6400*50

                for i_ly in range(self.num_layer):
                    # print("i_ly:", i_ly)
                    if i_ly == 0:
                        H_list = self.H_p_s
                        num_pre_subs = self.num_slstm
                    else:
                        H_list = self.out_mlp[i][i_ly-1]
                        num_pre_subs = len(self.out_mlp[i][i_ly-1])

                    list_H_extend = []
                    for j in range(num_pre_subs):
                        list_H_extend.append(tf.expand_dims(H_list[j], 2))
                    H_com = tf.concat(list_H_extend, 2) # 6400*50*3
                    H_com = tf.transpose(H_com, perm=[1, 2, 0]) # 50*3*6400

                    for k in range(self.aps_config[1][i_ly][0]):
                        # print("k:", k)
                        temp_sk = tf.tanh(tf.einsum('ij,jkl->ikl', weights['V_mlp'+str(i_ly+1)+str(k+1)], H_com)) # L*3*6400
                        self.Att[i][i_ly].append(tf.nn.softmax(tf.einsum('i,ijk->jk', weights['W_mlp'+str(i_ly+1)+str(k+1)], temp_sk), dim=0)) # len = n_sub'; [3*6400, 3*6400, 3*6400]

                        # 1st layer
                        self.in_mlp[i][i_ly].append(tf.matmul(tf.transpose(H_com, perm=[2, 0, 1]), tf.expand_dims(tf.transpose(self.Att[i][i_ly][k]), 2))) # 6400*50*3 * 6400*3*1 -> 6400*50*1
                        self.out_mlp[i][i_ly].append(self.multilayer_perceptron(self.in_mlp[i][i_ly][k][:,:,0], self.aps_config[1][i_ly][1])) # 6400*50

                #--------- tensorized LSTM ---------#
                # Eq.(1)
                Wx_temp_tsr  = tf.expand_dims(weights['Wx_tsr'], 2) # 42*4*1
                xt_temp_tsr  = X[:,i,:] # 6400*42
                xt_tsr       = tf.expand_dims(tf.transpose(xt_temp_tsr), 1) # 42*1*6400
                J_t_temp_tsr = tf.matmul(weights['Wh_tsr'], H_p_tsr) + tf.matmul(Wx_temp_tsr, xt_tsr) # 42*4*6400 + 42*4*6400
                J_t_tsr      = tf.tanh(tf.transpose(J_t_temp_tsr, perm=[2, 0, 1]) + biases['bj_tsr']) # 6400*42*4 + 42*4 = 6400*42*4
                # vec_J_t_tsr  = tf.reshape(J_t_tsr, [-1, n_hidden_units]) # 6400*168
                
                # Eq.(2)
                Wf_temp_tsr = tf.expand_dims(weights['Wf_tsr'], 2) # 42*4*1
                Ft_temp_tsr = tf.matmul(weights['Uf_tsr'], H_p_tsr) + tf.matmul(Wf_temp_tsr, xt_tsr) # 42*4*6400 + 42*4*6400
                Ft_tsr      = tf.sigmoid(tf.transpose(Ft_temp_tsr, perm=[2, 0, 1]) + biases['bf_tsr']) # 6400*42*4 + 42*4 = 6400*42*4

                Wi_temp_tsr = tf.expand_dims(weights['Wi_tsr'], 2) # 42*4*1
                It_temp_tsr = tf.matmul(weights['Ui_tsr'], H_p_tsr) + tf.matmul(Wi_temp_tsr, xt_tsr) # 42*4*6400 + 42*4*6400
                It_tsr      = tf.sigmoid(tf.transpose(It_temp_tsr, perm=[2, 0, 1]) + biases['bi_tsr']) # 6400*42*4 + 42*4 = 6400*42*4

                Wo_temp_tsr = tf.expand_dims(weights['Wo_tsr'], 2) # 42*4*1
                Ot_temp_tsr = tf.matmul(weights['Uo_tsr'], H_p_tsr) + tf.matmul(Wo_temp_tsr, xt_tsr) # 42*4*6400 + 42*4*6400
                Ot_tsr      = tf.sigmoid(tf.transpose(Ot_temp_tsr, perm=[2, 0, 1]) + biases['bo_tsr']) # 6400*42*4 + 42*4 = 6400*42*4

                # Eq.(3)
                c_c_tsr      = tf.multiply(Ft_tsr, c_p_tsr) + tf.multiply(It_tsr, J_t_tsr) # 6400*42*4

                # Eq.(4)
                # generate gate 'st_tsr'
                # self.out_mlp[i][-1][0]: 6400*50
                # weights['U_mlp_C1']: 42*4*50
                st_tsr_p1 = tf.matmul(weights['U_cc'], tf.transpose(c_c_tsr, perm=[1, 2, 0])) # 42*4*4 * 42*4*6400 = 42*4*6400
                stack_out_mlp = tf.transpose(tf.stack(self.out_mlp[i][-1]), perm=[0, 2, 1]) # 42*50*6400
                # st_tsr_p2 = tf.einsum('ijk,kl->ijl', weights['U_mlp_C1'], tf.transpose(self.out_mlp[i][-1][0])) # 42*4*50 (*) 50*6400 = 42*4*6400
                st_tsr_p2 = tf.matmul(weights['U_mlp_C1'], stack_out_mlp) # 42*4*50 * 42*50*6400 = 42*4*6400
                st_tsr = tf.sigmoid(st_tsr_p1 + st_tsr_p2) # 42*4*6400
                st_tsr = tf.transpose(st_tsr, perm=[2, 0, 1]) # 6400*42*4
                # mem_ex_tmp = tf.einsum('ijk,kl->ijl', weights['U_tsr'], tf.transpose(self.out_mlp[i][-1][0])) # 42*4*6400
                mem_ex_tmp = tf.matmul(weights['U_tsr'], stack_out_mlp) # 42*4*50 * 42*50*6400 = 42*4*6400
                mem_ex = tf.multiply(st_tsr, tf.transpose(mem_ex_tmp, perm=[2, 0, 1])) # 6400*42*4

                H_c_temp_tsr = tf.multiply(Ot_tsr, tf.tanh(c_c_tsr + mem_ex)) # 6400*42*4
                H_c_tsr      = tf.transpose(H_c_temp_tsr, perm=[1, 2, 0]) # 42*4*6400

                # update H and c
                H_p_tsr = H_c_tsr # 42*4*6400
                c_p_tsr = c_c_tsr # 6400*42*4

                self.h_list_tensor_temp_tsr.append(tf.transpose(H_p_tsr, perm=[2, 0, 1]))  # 6400*42*4

        return self.h_list_tensor_temp_tsr[-1], self.Att

