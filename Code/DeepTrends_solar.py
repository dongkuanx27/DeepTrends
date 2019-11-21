from __future__ import division
import os
import tensorflow as tf
import core.para_share.LSTM_APS_CNN as LAC
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import math as m

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def eval_rmse(y_true, y_pred):
    """ RMSE """
    return np.sqrt(np.mean(np.square(y_true - y_pred)))

def eval_rrse(y_true, y_pred):
    """ Root Relative Squared Error """
    return np.sqrt(np.sum(np.square(y_true - y_pred)) / np.sum(np.square(y_true - np.mean(y_true))) + 1e-8)

def constr_data_copyl(loc_s, ws, l, s, local):
    # l: (n_trend) s: (n_trend, 9), local: n_trend*(trend_l', 9)
    # loc_s: the local size
    n_trd, n_ts = np.shape(s)
    n_data = n_trend - ws + 1

    trd_list = []
    loc_list = []
    nxt_list = []
    for i in range(n_data):
        tmp = np.tile(l[i:i+ws], (n_ts, 1)).T # size: (ws, n_ts)
        trd_list.append(np.concatenate((tmp, s[i:i+ws]), axis=1)) # size: (ws, 2*n_ts)
        
        tmp = np.concatenate(local[i:i+ws-1]) # !!!!
        loc_list.append(tmp[-loc_s:])

        tmp = local[i+ws-1]
        nxt_list.append(tmp)

    # data_trd: (n_trend-ws+1, ws, 2*n_ts) 
    # data_loc: (n_trend-ws+1, loc_size, n_ts)
    # data_next: (n_trend-ws+1, trend_l', n_ts)
    return np.asarray(trd_list), np.asarray(loc_list), np.asarray(nxt_list)

def get_Batch(data, batch_size, n_epochs):
    #print(data.shape, label.shape)
    input_queue = tf.train.slice_input_producer([data], num_epochs=n_epochs, shuffle=True, capacity=500) 
    x_batch = tf.train.batch(input_queue, batch_size=batch_size, num_threads=1, capacity=1000, allow_smaller_final_batch=True)
    return x_batch

def CNN_local(data_local, config):
    inpt = data_local
    for i in range(len(config)):
        if i == 0:
            conv = tf.layers.conv1d(inpt, filters=config[i][0], kernel_size=config[i][2], strides=1, padding='same', activation = tf.nn.leaky_relu)
            max_pool = tf.layers.max_pooling1d(inputs=conv, pool_size=config[i][1], strides=2, padding='same')
            inpt = max_pool

        else:
            conv = tf.layers.conv1d(inpt, filters=config[i][0], kernel_size=config[i][2], strides=1, padding='same', activation = tf.nn.leaky_relu)
            max_pool = tf.layers.max_pooling1d(inputs=conv, pool_size=config[i][1], strides=2, padding='same')
            inpt = max_pool

    return max_pool

def CNN_local_shared(data_local, config):
    inpt = data_local
    with tf.variable_scope("CNN_local_shared", reuse=tf.AUTO_REUSE): 
        for i in range(len(config)):
            if i == 0:
                conv = tf.layers.conv1d(inputs=tf.expand_dims(inpt, 2), filters=config[i][0], kernel_size=config[i][2], strides=1, padding='same', activation = tf.nn.leaky_relu, name='cnn_shared_layer_'+str(i))
                max_pool = tf.layers.max_pooling1d(inputs=conv, pool_size=config[i][1], strides=2, padding='same')
                inpt = max_pool

            else:
                conv = tf.layers.conv1d(inpt, filters=config[i][0], kernel_size=config[i][2], strides=1, padding='same', activation = tf.nn.leaky_relu, name='cnn_shared_layer_'+str(i))
                max_pool = tf.layers.max_pooling1d(inputs=conv, pool_size=config[i][1], strides=2, padding='same')
                inpt = max_pool

    return max_pool

if __name__ == '__main__':

    # -------------------------- Read Data ------------------------
    filedpath = '/home/dsi/dxu/Backups/Research_Server/Working/Trend/Data/Processed/'
    list_filename = \
            [['solar_bgn0_end3000_lambda400_trends839_union', \
            'solar_bgn3000_end6000_lambda400_trends721_union', \
            'solar_bgn6000_end9000_lambda400_trends665_union', \
            'solar_bgn9000_end12000_lambda400_trends815_union', \
            'solar_bgn12000_end15000_lambda400_trends764_union', \
            'solar_bgn15000_end18000_lambda400_trends801_union', \
            'solar_bgn18000_end21000_lambda400_trends774_union', \
            'solar_bgn21000_end24000_lambda400_trends818_union', \
            'solar_bgn24000_end27000_lambda400_trends813_union', \
            'solar_bgn27000_end30000_lambda400_trends880_union', \
            'solar_bgn30000_end33000_lambda400_trends860_union', \
            'solar_bgn33000_end36000_lambda400_trends872_union', \
            'solar_bgn36000_end39000_lambda400_trends791_union', \
            'solar_bgn39000_end42000_lambda400_trends789_union', \
            'solar_bgn42000_end45000_lambda400_trends792_union', \
            'solar_bgn45000_end48000_lambda400_trends814_union', \
            'solar_bgn48000_end51000_lambda400_trends931_union', \
            'solar_bgn51000_end52560_lambda400_trends218_union'], \
                ['solar_bgn0_end3000_lambda600_trends417_union', \
                'solar_bgn3000_end6000_lambda600_trends471_union', \
                'solar_bgn6000_end9000_lambda600_trends398_union', \
                'solar_bgn9000_end12000_lambda600_trends479_union', \
                'solar_bgn12000_end15000_lambda600_trends516_union', \
                'solar_bgn15000_end18000_lambda600_trends481_union', \
                'solar_bgn18000_end21000_lambda600_trends461_union', \
                'solar_bgn21000_end24000_lambda600_trends543_union', \
                'solar_bgn24000_end27000_lambda600_trends553_union', \
                'solar_bgn27000_end30000_lambda600_trends567_union', \
                'solar_bgn30000_end33000_lambda600_trends554_union', \
                'solar_bgn33000_end36000_lambda600_trends554_union', \
                'solar_bgn36000_end39000_lambda600_trends488_union', \
                'solar_bgn39000_end42000_lambda600_trends475_union', \
                'solar_bgn42000_end45000_lambda600_trends478_union', \
                'solar_bgn45000_end48000_lambda600_trends530_union', \
                'solar_bgn48000_end51000_lambda600_trends494_union', \
                'solar_bgn51000_end52560_lambda600_trends95_union'], \
                    ['solar_bgn0_end3000_lambda800_trends278_union', \
                    'solar_bgn3000_end6000_lambda800_trends274_union', \
                    'solar_bgn6000_end9000_lambda800_trends274_union', \
                    'solar_bgn9000_end12000_lambda800_trends276_union', \
                    'solar_bgn12000_end15000_lambda800_trends273_union', \
                    'solar_bgn15000_end18000_lambda800_trends262_union', \
                    'solar_bgn18000_end21000_lambda800_trends248_union', \
                    'solar_bgn21000_end24000_lambda800_trends270_union', \
                    'solar_bgn24000_end27000_lambda800_trends285_union', \
                    'solar_bgn27000_end30000_lambda800_trends288_union', \
                    'solar_bgn30000_end33000_lambda800_trends268_union', \
                    'solar_bgn33000_end36000_lambda800_trends283_union', \
                    'solar_bgn36000_end39000_lambda800_trends246_union', \
                    'solar_bgn39000_end42000_lambda800_trends312_union', \
                    'solar_bgn42000_end45000_lambda800_trends275_union', \
                    'solar_bgn45000_end48000_lambda800_trends362_union', \
                    'solar_bgn48000_end51000_lambda800_trends306_union', \
                    'solar_bgn51000_end52560_lambda800_trends76_union']]

    list_lambda = ['400', '600', '800']
    
    for i_lambda in range(3):
        print('i_lambda = ', i_lambda)
        lbda = list_lambda[i_lambda]

        L_trend_l    = []
        L_trend_s    = []
        L_list_local = []
        for filename in list_filename[i_lambda]:
            file       = np.load(filedpath+filename+'.npz', allow_pickle=True)
            tmp_trend_l    = file['trend_l']    # (n_trend)
            tmp_trend_s    = file['trend_s']    # (n_trend, 9)
            tmp_list_local = file['list_local'] # list: n_trend*(trend_l', 9)

            L_trend_l.append(tmp_trend_l)
            L_trend_s.append(tmp_trend_s)
            L_list_local = L_list_local + list(tmp_list_local)

        trend_l = np.concatenate(L_trend_l)
        trend_s = np.concatenate(L_trend_s, axis=0)
        list_local = L_list_local

        n_trend, n_ts = np.shape(trend_s)

        # -------------------------- Parameter Setting ------------------------
        w_size = 32+1 # window size for long-term, last one is used for prediction; 8,16,32,64,128
        loc_size = 16 # the length of local data; 4,8,16,32,64
        k = 32 # hidden dim for each dim
        mt_h1, mt_h2= 32, 2 # '2': length' and 'slope'

        n_dim = 2 * n_ts # '2': slope + length
        L = 32 # for attention
        d_indi = k # hidden dimen for each time series variable

        n_steps = w_size - 1 # the input length for RNN
        n_hidden_units = n_dim * d_indi # tensorized lstm
        n_hidden_units_s = k * n_dim # shared lstm

        h_share = n_hidden_units_s # dimen for shared lstm
        d_mlp_h1a, d_mlp_outa = 32, 32
        d_mlp_h1b, d_mlp_outb = 32, 32
        d_mlp_h1d, d_mlp_outd = 32, 32
        d_mlp_h1c, d_mlp_outc = 32, 32

        training_iters = 560 # 700 #1000
        batch_size = 1000 #1000
        lr = 0.0005
        # num_stacked_layers = 1
        display_step = 1
        in_keep_prob  = 1 #0.5
        out_keep_prob = 1
        lambda_l2_reg = 0.00005
        theta = 0.25
        rounds = 1

        # for CNN
        filter1, filter2 = 16, 16
        ps1, ps2 = 2, 2
        ks1, ks2 = 2, 4
        # for shared CNN
        filter1_s, filter2_s = 32, 32
        ps1_s, ps2_s = 2, 2
        ks1_s, ks2_s = 2, 4

        # -------------------------- Contruct Training/Val/Test ------------------------
        # type: array, array
        # data_trd: (n_trend-ws+1, ws, 2*n_ts) 
        # data_loc: (n_trend-ws+1, loc_size, n_ts)
        # data_next: (n_trend-ws+1, trend_l', n_ts)
        data_trd, data_loc, data_next = constr_data_copyl(loc_size, w_size, trend_l, trend_s, list_local)
        n_data = len(data_loc)

        Data_idx = np.arange(n_data)
        X_train_idx, X_test_idx = train_test_split(Data_idx, test_size=0.1, random_state=42)
        X_train_idx, X_val_idx  = train_test_split(X_train_idx, test_size=0.1, random_state=42)

        # -------------------------- Model Configuration ------------------------
        # 0: shared lstm; 1: tensorized lstm; 2: sub networks
        aps_config = [[4, [n_dim, h_share]], \
                        [[4, [h_share, d_mlp_h1a, d_mlp_outa]], \
                         [4, [d_mlp_outa, d_mlp_h1b, d_mlp_outb]], \
                         [4, [d_mlp_outb, d_mlp_h1d, d_mlp_outd]], \
                         [n_dim, [d_mlp_outd, d_mlp_h1c, d_mlp_outc]]]]

        # cnn for each task
        cnn_config = [[filter1, ps1, ks1], [filter2, ps2, ks2]] # [filter1, pool_size1, kernel_size1]
        # shared cnn for all tasks
        cnn_config_s = [[filter1_s, ps1_s, ks1_s], [filter2_s, ps2_s, ks2_s]]
        # for each task
        mlp_task_config = [2*k+int((loc_size/(ps1_s*ps2_s))*filter2/(ps1*ps2)), mt_h1, mt_h2]

        # -------------------------- Training ------------------------
        # !! need to revise 'var-name' if k_round > 1
        for k_round in range(rounds):
            tf.reset_default_graph()

            # -------------- Parameter Initialization ------------
            Wx_s = []
            Wh_s = []
            W3_s = []
            for i in range(aps_config[0][0]):
                Wx_s.append(['Wx_s'+str(i+1), tf.get_variable('Weights_'+'Wx_s'+str(i+1), shape=[n_dim, n_hidden_units_s], dtype=tf.float32, initializer=tf.truncated_normal_initializer(mean=-0.1, stddev=0.1))])
                Wh_s.append(['Wh_s'+str(i+1), tf.get_variable('Weights_'+'Wh_s'+str(i+1), shape=[n_hidden_units_s, n_hidden_units_s], dtype=tf.float32, initializer=tf.truncated_normal_initializer(mean=-0.1, stddev=0.1))])
                W3_s.append(['W3_s'+str(i+1), tf.get_variable('Weights_'+'W3_s'+str(i+1), shape=[3*n_hidden_units_s, n_dim+n_hidden_units_s], dtype=tf.float32, initializer=tf.truncated_normal_initializer(mean=-0.1, stddev=0.1))])

            V_mlp = []
            W_mlp = []
            for i in range(len(aps_config[1])): # num_layer
                for j in range(aps_config[1][i][0]): # num_subnet of each layer
                    V_mlp.append(['V_mlp'+str(i+1)+str(j+1), tf.get_variable('Weights_'+'V_mlp'+str(i+1)+str(j+1), shape=[L, aps_config[1][i][1][0]], dtype=tf.float32, initializer=tf.truncated_normal_initializer(mean=-0.1, stddev=0.1))])
                    W_mlp.append(['W_mlp'+str(i+1)+str(j+1), tf.get_variable('Weights_'+'W_mlp'+str(i+1)+str(j+1), shape=[L], dtype=tf.float32, initializer=tf.truncated_normal_initializer(mean=-0.1, stddev=0.1))])

            weights = {'Wh_tsr': tf.get_variable('Weights_Wh_tsr', shape=[n_dim, d_indi, d_indi], dtype=tf.float32, initializer=tf.truncated_normal_initializer(mean=-0.1, stddev=0.1)), \
                        'Uf_tsr': tf.get_variable('Weights_Uf_tsr', shape=[n_dim, d_indi, d_indi], dtype=tf.float32, initializer=tf.truncated_normal_initializer(mean=-0.1, stddev=0.1)), \
                        'Ui_tsr': tf.get_variable('Weights_Ui_tsr', shape=[n_dim, d_indi, d_indi], dtype=tf.float32, initializer=tf.truncated_normal_initializer(mean=-0.1, stddev=0.1)), \
                        'Uo_tsr': tf.get_variable('Weights_Uo_tsr', shape=[n_dim, d_indi, d_indi], dtype=tf.float32, initializer=tf.truncated_normal_initializer(mean=-0.1, stddev=0.1)), \
                            'Wx_tsr': tf.get_variable('Weights_Wx_tsr', shape=[n_dim, d_indi], dtype=tf.float32, initializer=tf.truncated_normal_initializer(mean=-0.1, stddev=0.1)), \
                            'Wf_tsr': tf.get_variable('Weights_Wf_tsr', shape=[n_dim, d_indi], dtype=tf.float32, initializer=tf.truncated_normal_initializer(mean=-0.1, stddev=0.1)), \
                            'Wi_tsr': tf.get_variable('Weights_Wi_tsr', shape=[n_dim, d_indi], dtype=tf.float32, initializer=tf.truncated_normal_initializer(mean=-0.1, stddev=0.1)), \
                            'Wo_tsr': tf.get_variable('Weights_Wo_tsr', shape=[n_dim, d_indi], dtype=tf.float32, initializer=tf.truncated_normal_initializer(mean=-0.1, stddev=0.1)), \
                            # 'W3_tsr': tf.get_variable('Weights_W3_tsr', shape=[3*(n_dim*d_indi),n_dim*(d_indi+1)], dtype=tf.float32, initializer=tf.truncated_normal_initializer(mean=-0.1, stddev=0.1)), \
                            'U_cc': tf.get_variable('Weights_W_cc', shape=[n_dim, d_indi, d_indi], dtype=tf.float32, initializer=tf.truncated_normal_initializer(mean=-0.1, stddev=0.1)), \
                            'U_mlp_C1': tf.get_variable('Weights_W_mlp_C1', shape=[n_dim, d_indi, d_mlp_outc], dtype=tf.float32, initializer=tf.truncated_normal_initializer(mean=-0.1, stddev=0.1)), \
                            'U_tsr': tf.get_variable('Weights_W_tsr', shape=[n_dim, d_indi, d_mlp_outc], dtype=tf.float32, initializer=tf.truncated_normal_initializer(mean=-0.1, stddev=0.1))}
            
            weights_tmp = Wx_s + Wh_s + W3_s + V_mlp + W_mlp
            weights_tmp = dict(weights_tmp)
            weights.update(weights_tmp)

            bj_s = []
            b3_s = []
            for i in range(aps_config[0][0]):
                bj_s.append(['bj_s'+str(i+1), tf.get_variable('Biases_'+'bj_s'+str(i+1), shape=[n_hidden_units_s], dtype=tf.float32, initializer=tf.constant_initializer(0.0))])
                b3_s.append(['b3_s'+str(i+1), tf.get_variable('Biases_'+'b3_s'+str(i+1), shape=[3*n_hidden_units_s], dtype=tf.float32, initializer=tf.constant_initializer(0.0))])

            biases = {'bj_tsr': tf.get_variable('Weights_bj_tsr', shape=[n_dim, d_indi], dtype=tf.float32, initializer=tf.truncated_normal_initializer(mean=-0.1, stddev=0.1)), \
                        'bf_tsr': tf.get_variable('Weights_bf_tsr', shape=[n_dim, d_indi], dtype=tf.float32, initializer=tf.truncated_normal_initializer(mean=-0.1, stddev=0.1)), \
                        'bi_tsr': tf.get_variable('Weights_bi_tsr', shape=[n_dim, d_indi], dtype=tf.float32, initializer=tf.truncated_normal_initializer(mean=-0.1, stddev=0.1)), \
                        'bo_tsr': tf.get_variable('Weights_bo_tsr', shape=[n_dim, d_indi], dtype=tf.float32, initializer=tf.truncated_normal_initializer(mean=-0.1, stddev=0.1))}
                        #'b3_tsr': tf.get_variable('Weights_b3_tsr', shape=[3*(n_dim*d_indi)], dtype=tf.float32, initializer=tf.truncated_normal_initializer(mean=-0.1, stddev=0.1))}
            biases_tmp = bj_s + b3_s
            biases_tmp = dict(biases_tmp)
            biases.update(biases_tmp)


            # placeholder
            x_idx = tf.placeholder(tf.int64, [None,])

            trd_tf = tf.convert_to_tensor(data_trd, dtype=tf.float32) # array: (n_trend-ws+1, ws, 2*n_ts)
            loc_tf = tf.convert_to_tensor(data_loc, dtype=tf.float32) # array: (n_trend-ws+1, loc_size, n_ts)

            X = tf.gather(trd_tf, x_idx)[:, 0:-1, :] # size: (batch_size, ws-1, 2*n_ts)
            Y = tf.gather(trd_tf, x_idx)[:, -1, :] # size: (batch_size, 2*n_ts); form: [9*l,9*s]

            # -------------- TLASM for long ------------
            machine = LAC.APS(X.get_shape()[1].value, aps_config)
            h_list_batch, list_att = machine.run(X, n_hidden_units, n_hidden_units_s, weights, biases)

            # -------------- Multi-task CNN for local ------------
            X_loc = tf.cast(tf.gather(loc_tf, x_idx), tf.float32) # size: (batch_size, loc_size, n_ts)
            
            # shared CNN for all
            max_pool_s = []
            for i in range(n_ts):
                # CNN with 2 layers:
                # (batch_size, loc_size, 1) -> (batch, loc_size/2, 64)
                # (batch, loc_size/2, 64) -> (batch, loc_size/4, 64)
                max_pool_s.append(CNN_local_shared(X_loc[:,:,i], cnn_config_s))
            out_cnn_s = tf.stack(max_pool_s, axis=1) # (batch_size, n_ts, loc_size/4, filter2_s)

            # task-specific CNN for each
            max_pool = []
            for i in range(n_ts):
                # CNN with 2 layers:
                # (batch_size, loc_size/4, filter2_s) -> (batch, loc_size/4/2, 32)
                # (batch, loc_size/4/2, 32) -> (batch, loc_size/4/2/2, 32)
                max_pool.append(tf.reshape(CNN_local(out_cnn_s[:,i,:,:], cnn_config), [-1,1,int((loc_size/(ps1_s*ps2_s))*filter2/(ps1*ps2))])) # (batch,1,loc_size/4/2/2*32)
            out_cnn = tf.concat(max_pool, 1) # size = (batch, n_ts, loc_size/4/2/2*32)

            # -------------- Task-specific subnetwork for each ------------
            # h_list_batch： 6400*(9+9)*4
            ly_fus = tf.concat([tf.reshape(h_list_batch, [-1, n_ts, 2*k]), out_cnn], 2) # (batch_size, n_ts, 2*k + loc_size/4/2/2*32)

            out_t = []
            for i in range(n_ts):
                out_t.append(machine.multilayer_perceptron(ly_fus[:,i,:], mlp_task_config))
            
            # -------------- Loss & optimization ------------
            # loss
            l_pred = []
            s_pred = []
            for i in range(n_ts):
                l_pred.append(tf.expand_dims(out_t[i][:,0], 1)) # (batch_size, 1)
                s_pred.append(tf.expand_dims(out_t[i][:,1], 1))

            # L2 regularization for weights and biases
            reg_loss = 0
            for tf_var in tf.trainable_variables():
                reg_loss += tf.reduce_mean(tf.nn.l2_loss(tf_var))

            # Define loss and optimization
            # AdamOptimizer, GradientDescentOptimizer, AdagradOptimizer
            prediction = tf.concat(l_pred+s_pred, 1) # (batch_size, 2*n_ts)

            loss_op = tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(Y, prediction))))
            train_op = tf.train.AdamOptimizer(learning_rate=lr).minimize(loss_op + lambda_l2_reg*reg_loss)

            # -------------- confidence score, aging score ------------
            # confidence score; greater -> more confident
            pred_l, pred_s = prediction[:, 0:n_ts], prediction[:, n_ts:2*n_ts] # size: (batch_size, n_ts)
            true_l, true_s = Y[:, 0:n_ts], Y[:, n_ts:2*n_ts]

            conf_sco_l = tf.tanh(tf.divide(tf.abs(tf.subtract(pred_l, true_l)), true_l)) # size: (batch_size, n_ts); range: [0,1]
            conf_sco_s = tf.atan(tf.abs(tf.subtract(pred_s, true_s))) / tf.constant(m.pi, dtype=tf.float32)
            conf_sco = 1 - tf.add(theta*conf_sco_l, (1 - theta)*conf_sco_s) # size: (batch_size, n_ts)

            # aging score; greater -> more trending
            aging_sco = tf.abs(tf.tanh(pred_s)) # size: (batch_size, n_ts)

            # -------------- rmse of each task------------
            # size: (n_ts)
            rmse_l = tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(true_l, pred_l)), axis=0))
            rmse_s = tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(true_s, pred_s)), axis=0))

            # 'Saver' 操作将保存所有变量以供恢复
            saver = tf.train.Saver()

            x_batch_idx = get_Batch(X_train_idx, batch_size, training_iters)
        
            with tf.Session() as sess:
                sess.run(tf.global_variables_initializer())
                sess.run(tf.local_variables_initializer())
                # 开启协调器
                coord = tf.train.Coordinator()
                # 使用start_queue_runners 启动队列填充
                threads = tf.train.start_queue_runners(sess, coord)

                try:
                    ite_tag = 0
                    List_tr_B_loss = []
                    while not coord.should_stop():
                        # 获取训练用的每一个batch中batch_size个样本idx和标签
                        # with tf.device('/CPU:0'):

                        x_batch_idx_feed = sess.run([x_batch_idx])
                        x_batch_idx_feed = x_batch_idx_feed[0]
                        # print('x_batch_idx_feed: ', type(x_batch_idx_feed))
                        # print('X_val_idx: ', type(X_val_idx))

                        # training
                        train_op.run(feed_dict={x_idx:x_batch_idx_feed})

                        fetch = {'x_idx': x_idx, 'Y':Y, 'prediction':prediction, 'h_list_batch':h_list_batch, 'reg_loss':reg_loss, 'loss_op':loss_op, 'conf_sco':conf_sco, 'aging_sco':aging_sco, 'list_att':list_att, 'rmse_l':rmse_l, 'rmse_s':rmse_s, 'pred_l':pred_l, 'pred_s':pred_s, 'true_l':true_l, 'true_s':true_s}
                        
                        # training
                        Res = sess.run(fetch, feed_dict={x_idx:x_batch_idx_feed})
                        tr_rmse = eval_rmse(Res['Y'], Res['prediction'])
                        tr_rrse = eval_rrse(Res['Y'], Res['prediction'])

                        # validation
                        Res_val = sess.run(fetch, feed_dict={x_idx:X_val_idx})
                        val_rmse= eval_rmse(Res_val['Y'], Res_val['prediction'])
                        val_rrse= eval_rrse(Res_val['Y'], Res_val['prediction'])

                        if ite_tag % display_step == 0 or ite_tag == 0:
                            print("Epoch %d, Ite %d, tr_loss=%g, L2=%g, tr_rmse=%g, val_rmse=%g, tr_rrse=%g, val_rrse=%g" % (ite_tag//(len(X_train_idx)//batch_size + 1), ite_tag, Res['loss_op'], Res['reg_loss'], tr_rmse, val_rmse, tr_rrse, val_rrse))
                        ite_tag += 1

                        List_tr_B_loss = np.append(List_tr_B_loss, Res['loss_op'])

                except tf.errors.OutOfRangeError: # num_epochs 次数用完会抛出此异常
                    print("---Train end---")
                finally:
                    # 协调器coord发出所有线程终止信号
                    coord.request_stop()
                    print('---Programm end---')
                coord.join(threads) # 把开启的线程加入主线程，等待threads结束

                # testing
                Res_test = sess.run(fetch, feed_dict={x_idx:X_test_idx})
                te_rmse_l = eval_rmse(Res_test['pred_l'], Res_test['true_l'])
                te_rmse_s = eval_rmse(Res_test['pred_s'], Res_test['true_s'])
                te_rrse_l = eval_rrse(Res_test['pred_l'], Res_test['true_l'])
                te_rrse_s = eval_rrse(Res_test['pred_s'], Res_test['true_s'])

                te_rmse = eval_rmse(Res_test['Y'], Res_test['prediction'])
                te_rrse = eval_rrse(Res_test['Y'], Res_test['prediction'])

                # print("test_rmse=%g" % (te_rmse))
                # print("test_rrse=%g" % (te_rrse))
                print("test_rmse_l=%g, test_rmse_s=%g, test_rrse_l=%g, test_rrse_s=%g" % (te_rmse_l, te_rmse_s, te_rrse_l, te_rrse_s))

                print("test_rmse_l_task: ", Res_test['rmse_l'])
                print("test_rmse_s_task: ", Res_test['rmse_s'])

                # for case study based on training set
                Res_train = sess.run(fetch, feed_dict={x_idx:X_train_idx})
                # Res_train['conf_sco']: (batch_size, n_ts)
                # Res_train['aging_sco']: (batch_size, n_ts)
                # Res_train['list_att']: [time-step -> layer-num -> att]
                
                # Res_test['Y']: (batch_size, 2*n_ts)
                # Res_test['rmse_l']: (n_ts)
                # Res_test['rmse_s']: (n_ts)

                # data_trd: (n_trend-ws+1, ws, 2*n_ts) 
                # data_loc: (n_trend-ws+1, loc_size, n_ts)

                np.savez("/home/dsi/dxu/Backups/Research_Server/Working/Trend/Res/Routing/TLASM_"+filename[0:5]+"_lambda_"+lbda+".npz", List_tr_B_loss=List_tr_B_loss, data_trd=data_trd, data_loc=data_loc, data_next=data_next, conf_sco=Res_train['conf_sco'], aging_sco=Res_train['aging_sco'], Y=Res_test['Y'], prediction=Res_test['prediction'], X_train_idx=X_train_idx, list_att=Res_train['list_att'], te_rmse_l_task=Res_test['rmse_l'], te_rmse_s_task=Res_test['rmse_s'], te_rmse_l=te_rmse_l, te_rmse_s=te_rmse_s, te_rrse_l=te_rrse_l, te_rrse_s=te_rrse_s)




