import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from model.ours.SPS_Model_Phy import SPS_Model_Phy
from model.ours.spatial_block import *
from model.ours.temporal_block import *
from model.ours.SPS_Model_Phy_wo_BCR import SPS_Model_Phy_wo_BCR




class SPS_Model_PINN(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.node_num = args.node_num
        self.sensor_num = args.sensor_num

        self.H_last = None
        '()    是否已经初始化初始遥测量'

        # ### temporal_block for temporal modeling
        # if args.transformation_block == 'MLP':
        #     self.temporal_block = MLP_dim2(in_dim=self.sensor_num,
        #                                          hidden_dim=args.transf_MLP_hidden_dim,
        #                                          out_dim=self.sensor_num,
        #                                          layer_num=args.transf_MLP_layer_num,
        #                                          dropout=args.dropout,
        #                                          LeakyReLU_slope=args.LeakyReLU_slope)
        #     # (batch_size, in_node_num, lag) -> (batch_size, out_node_num, lag)
        # elif args.transformation_block == 'TCN':
        #     self.temporal_block = TCN(num_inputs=self.sensor_num,
        #                                     num_channels=args.transf_TCN_num_channels,
        #                                     kernel_size=args.transf_TCN_kernel_size,
        #                                     dropout=args.dropout)
        #     # (batch, node_num, lag) -> (batch, TCN_layers_channels[-1], lag)
        #     if args.transf_TCN_num_channels[-1] != self.sensor_num:
        #         self.temporal_block_end = MLP_dim2(in_dim=args.transf_TCN_num_channels[-1],
        #                                                  hidden_dim=args.transformation_block_end_mlp_hidden_dim,
        #                                                  layer_num=args.transformation_block_end_mlp_layer_num,
        #                                                  out_dim=self.sensor_num,
        #                                                  dropout=args.dropout,
        #                                                  LeakyReLU_slope=args.LeakyReLU_slope)
        # elif args.transformation_block == 'GRU':
        #     self.temporal_block = GRU(input_size=self.sensor_num,
        #                                     hidden_size=args.transf_GRU_hidden_size,
        #                                     num_layers=args.transf_GRU_layers,
        #                                     dropout=args.dropout)
        #     # (batch, node_num, lag) -> (batch, hidden_size, lag)
        #     if args.transf_GRU_hidden_size != self.sensor_num:
        #         self.temporal_block_end = MLP_dim2(in_dim=args.transf_GRU_hidden_size,
        #                                                  hidden_dim=args.transformation_block_end_mlp_hidden_dim,
        #                                                  layer_num=args.transformation_block_end_mlp_layer_num,
        #                                                  out_dim=self.sensor_num,
        #                                                  dropout=args.dropout,
        #                                                  LeakyReLU_slope=args.LeakyReLU_slope)
        # else:
        #     raise Exception("No such transformation_block! must be 'MLP' or 'TCN' or 'GRU'!")
        "if your forecast task is long-term, you may need to use temporal_block, " \
        "but in our model, we don't set a temporal_block, detail in the paper"


        """为了batch_size不再局限于1，这里的SPS_Model_Phy不再使用，而是直接用PI_info_batch了
        如果未来要再启用，记得set_init_value里面的注释了的也要改回来"""
        if self.args.how_deploy_phy == 'cal':
            if args.SPS_Model_PINN_if_has_Phy_of_BCR:
                self.SPS_Model_Phy = SPS_Model_Phy(exp_frequency=args.exp_frequency,
                                                   lag_step=args.lag_step,
                                                   BAT_QU_curve_app_order=args.BAT_QU_curve_app_order,
                                                   Load_TP_curve_app_order=args.Load_TP_curve_app_order,
                                                   SOC_init=args.SOC_init,
                                                   SOC_if_trickle_charge=args.SOC_if_trickle_charge,
                                                   SPS_Model_PINN_if_has_wrong_Phy=args.SPS_Model_PINN_if_has_wrong_Phy,
                                                   wrong_Phy_error=args.wrong_Phy_error,
                                                   random_seed=args.random_seed)
                # 4 * (batch_size, 1, lag) -> [sensor_num * (batch_size, 1, lag)]
            else:
                self.SPS_Model_Phy = SPS_Model_Phy_wo_BCR(exp_frequency=args.exp_frequency,
                                                          lag_step=args.lag_step,
                                                          BAT_QU_curve_app_order=args.BAT_QU_curve_app_order,
                                                          Load_TP_curve_app_order=args.Load_TP_curve_app_order,
                                                          SOC_init=args.SOC_init,
                                                          SOC_if_trickle_charge=args.SOC_if_trickle_charge,
                                                          BCR_MLP_hidden_dim=args.BCR_MLP_hidden_dim,
                                                          BCR_MLP_lay_num=args.BCR_MLP_lay_num,
                                                          replace_BCR_with_MLP_or_0=args.replace_BCR_with_MLP_or_0,
                                                          dropout=args.dropout,
                                                          LeakyReLU_slope=args.LeakyReLU_slope)
        elif self.args.how_deploy_phy == 'sim':
            pass

        ### spatial_block for spatial modeling
        if args.graph_ca_meth == 'Training':
            self.A = Parameter(torch.Tensor(args.node_num+args.sensor_num, args.node_num+args.sensor_num))
            nn.init.xavier_uniform_(self.A)
            # 'self.A: (node_num+sensor_num, node_num+sensor_num)'
        else:
            self.A = None
        if args.spatial_block == 'G3CN':
            self.spatial_block = CMTS_GCN(CMTS_GCN_K_nums=args.CMTS_GCN_K_nums,
                                          node_num=args.node_num+args.sensor_num,
                                          CMTS_GCN_residual=args.CMTS_GCN_residual,
                                          LeakyReLU_slope=args.LeakyReLU_slope)
            # (batch_size, node_num, lag) -> (batch_size, node_num, lag)
        elif args.spatial_block == 'GCN':
            self.spatial_block = GCN_s(GCN_layer_nums=args.GCN_layer_nums,
                                       node_num=args.node_num+args.sensor_num,
                                       lag=args.lag,
                                       LeakyReLU_slope=args.LeakyReLU_slope)
            # (batch_size, node_num, lag) -> (batch_size, node_num, lag)
        elif args.spatial_block == 'GAT':
            self.spatial_block = Muti_S_GAT(Muti_S_GAT_K=args.Muti_S_GAT_K,
                                            Muti_S_GAT_embed_dim=args.Muti_S_GAT_embed_dim,
                                            node_num=args.node_num+args.sensor_num,
                                            lag=args.lag,
                                            use_gatv2=args.use_gatv2,
                                            dropout=args.dropout,
                                            LeakyReLU_slope=args.LeakyReLU_slope)
            # (batch_size, node_num, lag) -> (batch_size, node_num, lag)
        elif args.spatial_block == 'GIN':
            self.spatial_block = GIN(GIN_layer_nums=args.GIN_layer_nums,
                                     GIN_MLP_layer_num=args.GIN_MLP_layer_num,
                                     lag=args.lag,
                                     dropout=args.dropout,
                                     LeakyReLU_slope=args.LeakyReLU_slope)
            # (batch_size, node_num, lag) -> (batch_size, node_num, lag)
        elif args.spatial_block == 'SGC':
            self.spatial_block = SGC(SGC_K=args.SGC_K,
                                     SGC_hidden_dim=args.SGC_hidden_dim,
                                     lag=args.lag,
                                     LeakyReLU_slope=args.LeakyReLU_slope)
            # (batch_size, node_num, lag) -> (batch_size, node_num, lag)
        else:
            raise Exception("No such spatial_block! must be 'G3CN' or 'GCN' or 'GAT' or 'GIN' or 'SGC'!")


    def set_init_value(self, init_sample):
        """
        用于设置初始遥测量, the initial state at the beginning of the simulation
        Args:
            init_sample: (batch_size, sensor_num, lag)   the initial state at the beginning of the simulation
        """
        # self.SPS_Model_Phy.set_init_value(init_sample)
        pass


    def forward(self, A, X, X_norm, WC, WC_norm, T, init_sample, init_sample_norm, scaler_info, PI_info_batch):
        """
        :param A: (node_num, node_num)
        :param X: (batch_size, sensor_num, lag)
        :param X_norm: (batch_size, sensor_num, lag)
        :param WC: (batch_size, 4, lag), actually are working conditions: irradiance, temperature, wind speed, load
        :param WC_norm: (batch_size, 4, lag), normalized working conditions
        :param T: (batch_size, 5, lag), the time information, DAY, HOUR, MINUTE, SECOND, TIMESTAMP
        :param init_sample: (batch_size, sensor_num, lag) the initial state at the beginning of the simulation
        :param init_sample_norm: (batch_size, sensor_num, lag) the normalized initial state at the beginning of the simulation
        :param scaler_info: dict, {'scale_list': scale_list, 'mean_list': mean_list, 'scale_list_work_condition': scale_list_work_condition, 'mean_list_work_condition': mean_list_work_condition}
        :param PI_info_batch: (batch_size, sensor_num, lag), PI_info_batch is the PI information, pure physical simluation data

        """
        if self.A is not None:
            A = self.A
            'A: (node_num+sensor_num, node_num+sensor_num)'


        # # temporal_block
        # # in our model, we don't set a temporal_block
        # X_norm = self.temporal_block(X_norm)
        # 'X_norm: (batch_size, TCN_layers_channels[-1], lag)'
        # if X_norm.size(1) != self.sensor_num
        #     X_norm = self.temporal_block_end(X_norm)
        # 'X_norm: (batch_size, sensor_num, lag)'
        "if your forecast task is long-term, you may need to use temporal_block, " \
        "but in our model, we don't set a temporal_block, detail in the paper"


        ### Phy
        if self.args.how_deploy_phy == 'cal':
            """ *** calculate by the physical feature *** """
            # # if self.H_last is None:
            # #     self.H_last = init_sample.data
            # # if self.H_last.size(0) != X.size(0):
            # #     self.H_last = self.H_last[:X.size(0), :, :]
            # #     # 因为在数据导入时，如果末尾凑不够一个batch_size，就会丢弃，所以这里也要丢弃
            # #     # 这样也不行，训练集裁成不完整的传入验证集又会连不起来而报错，根源错误不在这里
            # # self.SPS_Model_Phy.set_init_value(self.H_last)
            self.SPS_Model_Phy.set_init_value(X)
            H_Phy_list = self.SPS_Model_Phy(S_irr_SA=WC[:, 0:1, :],
                                            T_SA=WC[:, 1:2, :],
                                            theta=WC[:, 2:3, :],
                                            Load_Signal=WC[:, 3:, :])
            # (batch_size, 4, lag) -> [sensor_num * (batch_size, 1, lag)]
            H_Phy = torch.cat(H_Phy_list, dim=1)
            # [sensor_num * (batch_size, 1, lag)] -> (batch_size, sensor_num, lag)
            'H_Phy: (batch_size, sensor_num, lag)  not de normalized'
            # # # self.H_last = H_Phy.data
            # # self.H_last = H_Phy.detach()
            # # 'self.H_last: (batch_size, sensor_num, lag)  not de normalized'
        elif self.args.how_deploy_phy == 'sim':
            """ *** or you can just use PI_info_batch directly *** """
            """actually, how_deploy_phy == 'sim' is recommended because it supports batch size greater than 1."""
            """PI_info_batch is pure physical simluation data, which can be got by running [args.model_selection='SPS_Model_Phy', args.if_save_simulate_result=True, args.BasedOn='reconstruct'] first"""
            H_Phy = PI_info_batch
            'H_Phy: (batch_size, sensor_num, lag)  not de normalized'


        ### Adapt
        # use scaler_info to normalize H_Phy
        # scale_tensor = torch.tensor(scaler_info['scale_list'], dtype=H_Phy.dtype, device=H_Phy.device).unsqueeze(0).unsqueeze(2)
        scale_tensor = scaler_info['scale_list'].unsqueeze(0).unsqueeze(2)
        'scale_tensor: (1, sensor_num, 1)'
        # mean_tensor = torch.tensor(scaler_info['mean_list'], dtype=H_Phy.dtype, device=H_Phy.device).unsqueeze(0).unsqueeze(2)
        mean_tensor = scaler_info['mean_list'].unsqueeze(0).unsqueeze(2)
        'mean_tensor: (1, sensor_num, 1)'
        H_Phy_norm = (H_Phy - mean_tensor) / scale_tensor
        'H_Phy_norm: (batch_size, sensor_num, lag)  normalized'
        # denoise by conv1d
        if self.args.if_adapt_denoise:
            # smooth_kernel = torch.tensor([0.1, 0.2, 0.4, 0.2, 0.1], device=H_Phy_norm.device).unsqueeze(0).unsqueeze(0)
            smooth_kernel = torch.tensor([0.0, 0.0, 1.0, 0.0, 0.0], device=H_Phy_norm.device).unsqueeze(0).unsqueeze(0)
            'smooth_kernel: (1, 1, 5)'
            smooth_kernel = smooth_kernel.repeat(H_Phy_norm.size(1), 1, 1)
            'smooth_kernel: (sensor_num, 1, 5)'
            # set the 6, 9, 12, 15, 18 th channels to [0.2, 0.2, 0.2, 0.2, 0.2]
            smooth_kernel[6, 0, :] = torch.tensor([0.2, 0.2, 0.2, 0.2, 0.2], device=H_Phy_norm.device)
            smooth_kernel[9, 0, :] = torch.tensor([0.2, 0.2, 0.2, 0.2, 0.2], device=H_Phy_norm.device)
            smooth_kernel[12, 0, :] = torch.tensor([0.2, 0.2, 0.2, 0.2, 0.2], device=H_Phy_norm.device)
            smooth_kernel[15, 0, :] = torch.tensor([0.2, 0.2, 0.2, 0.2, 0.2], device=H_Phy_norm.device)
            smooth_kernel[18, 0, :] = torch.tensor([0.2, 0.2, 0.2, 0.2, 0.2], device=H_Phy_norm.device)
            # Pad the tensor with replication of the edge values
            H_Phy_norm = F.pad(H_Phy_norm, (2, 2), mode='replicate')
            'H_Phy_norm: (batch_size, sensor_num, lag+4)'
            H_Phy_norm = F.conv1d(H_Phy_norm, smooth_kernel, padding=0, groups=H_Phy_norm.size(1))
            'H_Phy_norm: (batch_size, sensor_num, lag)  normalized'
        "if deploy other GNN, not G3CN, you may need to feature transform here, it is part of Adapt(), detail in the paper"


        ### Aggregate and Update
        H_in  = torch.cat((X_norm, H_Phy_norm), dim=1)
        'H_in: (batch_size, sensor_num*2, lag)  normalized'
        if self.args.if_timestamp:
            H_in = torch.cat((H_in, T), dim=1)
            'H_in: (batch_size, sensor_num*2+5, lag)  normalized'
        if self.args.if_add_work_condition:
            H_in = torch.cat((H_in, WC_norm), dim=1)
            'H_in: (batch_size, sensor_num*2+5+4, lag)  normalized'
        H_gnn = self.spatial_block(H_in, A)
        'H_gnn: (batch_size, sensor_num*2+5+4, lag)  normalized'
        if H_gnn.size(1) != H_Phy_norm.size(1):
            H_gnn = H_gnn[:, :H_Phy_norm.size(1), :]
            'H_gnn: (batch_size, sensor_num, lag)  normalized'


        ### which mode
        if self.args.SPS_Model_PINN_if_simplified:
            H_out = H_gnn
            'H_out: (batch_size, sensor_num, lag)  normalized'
        else:
            H_out = H_gnn + H_Phy_norm
            'H_out: (batch_size, sensor_num, lag)  normalized'

        return H_out
























