import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter


class SPS_Model_Phy(nn.Module):
    def __init__(self, exp_frequency=1,
                 lag_step=1,
                 BAT_QU_curve_app_order=3,
                 Load_TP_curve_app_order=3,
                 SOC_init=39.63193175 / 40,
                 SOC_if_trickle_charge=0.99,
                 SPS_Model_PINN_if_has_wrong_Phy=False,
                 wrong_Phy_error=0.00001,
                 random_seed=42):
        super().__init__()

        self.V_mp_ref_all_SA = 24
        self.I_mp_ref_all_SA = 2.5
        self.SA = Solar_Array(V_mp_ref_all=self.V_mp_ref_all_SA, I_mp_ref_all=self.I_mp_ref_all_SA)

        self.SR = Shunt_Regulator()

        self.SOC_init = SOC_init
        self.BAT_n_p = 5
        self.BAT_n_s = 4
        self.Q_ref_BAT = 40
        self.SOC_if_trickle_charge = SOC_if_trickle_charge
        self.charge_Q_thershold = self.Q_ref_BAT * self.SOC_if_trickle_charge
        self.exp_frequency = exp_frequency
        self.lag_step = lag_step
        self.BAT_QU_curve_app_order = BAT_QU_curve_app_order
        self.efficiency = 0.99
        self.dt = 1 / exp_frequency / 3600
        self.Q_start = None
        self.BAT = Battery(BAT_n_p=self.BAT_n_p, BAT_n_s=self.BAT_n_s, 
                           Q_ref_BAT=self.Q_ref_BAT, 
                           charge_Q_thershold=self.charge_Q_thershold, 
                           lag_step=self.lag_step, app_order=self.BAT_QU_curve_app_order, 
                           efficiency=self.efficiency, dt=self.dt,
                            SPS_Model_PINN_if_has_wrong_Phy=False,
                            wrong_Phy_error=0.00001,
                            random_seed=42)

        self.BDR = Battery_Discharge_Regulator()

        self.BCR = Battery_Charging_Regulator(Q_ref_BAT=self.Q_ref_BAT,
                                              SOC_if_trickle_charge=self.SOC_if_trickle_charge,
                                                SPS_Model_PINN_if_has_wrong_Phy=False,
                                                wrong_Phy_error=0.00001,
                                                random_seed=42)

        self.Bus = Bus_Flow()

        self.PDM = Power_Distribution_Module()

        self.Load_TP_curve_app_order = Load_TP_curve_app_order
        self.Load = Load(Load_TP_curve_app_order=self.Load_TP_curve_app_order, dt=self.dt)

        self.init_sample_mark = False

        self.U_SA = None
        self.I_SA = None
        self.P_SA = None
        self.U_Load_output = None
        self.I_Load_output = None
        self.P_Load_output = None
        self.U_BCR = None
        self.I_BCR = None
        self.P_BCR = None
        self.U_BAT2 = None
        self.I_BAT2 = None
        self.T_BAT2 = None
        self.U_BAT3 = None
        self.I_BAT3 = None
        self.T_BAT3 = None
        self.U_BAT4 = None
        self.I_BAT4 = None
        self.T_BAT4 = None
        self.U_Bus = None
        self.I_Bus = None
        self.P_Bus = None
        self.U_Load1 = None
        self.I_Load1 = None
        self.T_Load1 = None
        self.P_Load1 = None
        self.U_Load2 = None
        self.I_Load2 = None
        self.T_Load2 = None
        self.P_Load2 = None
        self.U_Load3 = None
        self.I_Load3 = None
        self.T_Load3 = None
        self.P_Load3 = None

        self.reset_parameters()

    def reset_parameters(self):
        pass

    def set_init_value(self, init_sample):
        self.init_sample_mark = True
        self.U_SA = init_sample[:, 0:1, :]
        self.I_SA = init_sample[:, 1:2, :]
        self.P_SA = init_sample[:, 2:3, :]
        self.U_Load_output = init_sample[:, 3:4, :]
        self.I_Load_output = init_sample[:, 4:5, :]
        self.P_Load_output = init_sample[:, 5:6, :]
        self.U_BCR = init_sample[:, 6:7, :]
        self.I_BCR = init_sample[:, 7:8, :]
        self.P_BCR = init_sample[:, 8:9, :]
        self.U_BAT2 = init_sample[:, 9:10, :]
        self.I_BAT2 = init_sample[:, 10:11, :]
        self.T_BAT2 = init_sample[:, 11:12, :]
        self.U_BAT3 = init_sample[:, 12:13, :]
        self.I_BAT3 = init_sample[:, 13:14, :]
        self.T_BAT3 = init_sample[:, 14:15, :]
        self.U_BAT4 = init_sample[:, 15:16, :]
        self.I_BAT4 = init_sample[:, 16:17, :]
        self.T_BAT4 = init_sample[:, 17:18, :]
        self.U_Bus = init_sample[:, 18:19, :]
        self.I_Bus = init_sample[:, 19:20, :]
        self.P_Bus = init_sample[:, 20:21, :]
        self.U_Load1 = init_sample[:, 21:22, :]
        self.I_Load1 = init_sample[:, 22:23, :]
        self.T_Load1 = init_sample[:, 23:24, :]
        self.P_Load1 = init_sample[:, 24:25, :]
        self.U_Load2 = init_sample[:, 25:26, :]
        self.I_Load2 = init_sample[:, 26:27, :]
        self.T_Load2 = init_sample[:, 27:28, :]
        self.P_Load2 = init_sample[:, 28:29, :]
        self.U_Load3 = init_sample[:, 29:30, :]
        self.I_Load3 = init_sample[:, 30:31, :]
        self.T_Load3 = init_sample[:, 31:32, :]
        self.P_Load3 = init_sample[:, 32:33, :]
        
        Q_start_00 = self.Q_ref_BAT * self.SOC_init
        I_BAT = (self.I_BAT2 + self.I_BAT3 + self.I_BAT4) / 3 * self.BAT_n_p
        self.Q_start_init = Q_start_00 - torch.cumsum(I_BAT[0:1, :, :] * self.dt, dim=-1) * self.efficiency

    def forward(self, S_irr_SA, T_SA, theta, Load_Signal):
        self.P_Load_output = torch.where(Load_Signal == 0,
                                         torch.tensor(0.0, device=S_irr_SA.device),
                                         torch.where(Load_Signal == 2,
                                                     torch.tensor(14.5, device=S_irr_SA.device),
                                                     torch.where(Load_Signal == 1,
                                                                 torch.tensor(28.8, device=S_irr_SA.device),
                                                                 torch.tensor(14.5, device=S_irr_SA.device))))

        I_mp_all, V_mp_all, P_mp_all, I_all_sample, V_all_sample, P_all_sample \
            = self.SA(S_irr_SA=S_irr_SA, T_SA=T_SA, theta=theta)

        self.U_BCR, self.I_BCR, I_Remain, Charge_Signal, Discharge_Signal, I_Shunt \
            = self.SR(P_Load=self.P_Load_output, U_Bus=self.U_Bus, I_SA=I_mp_all, U_SA=V_mp_all,
                      I_BAT=(self.I_BAT2 + self.I_BAT3 + self.I_BAT4) / 3 * self.BAT_n_p,
                      U_BAT=(self.U_BAT2 + self.U_BAT3 + self.U_BAT4) / 3)

        self.U_SA, self.I_SA, self.P_SA, self.U_BCR, self.I_BCR, self.P_BCR \
            = DET2PPT(I_Shunt, I_mp_all, V_mp_all, P_all_sample, I_all_sample, V_all_sample, self.I_BCR, self.U_BCR)

        Q_now, SOC, U_BAT, Pres_BAT, T_BAT, I_BAT_each_set \
            = self.BAT(Signal_BAT=None,
                       I_BAT=(self.I_BAT2 + self.I_BAT3 + self.I_BAT4) / 3 * self.BAT_n_p,
                       Q_start_init=self.Q_start_init)
        self.U_BAT2 = U_BAT
        self.T_BAT2 = T_BAT
        self.U_BAT3 = U_BAT
        self.T_BAT3 = T_BAT
        self.U_BAT4 = U_BAT
        self.T_BAT4 = T_BAT

        Duty_Factor, I_Discharge, I_BAT2Load, U_BAT2Load \
            = self.BDR(Discharge_Signal=Discharge_Signal, U_BAT=U_BAT, I_Remain=I_Remain)

        I_BAT, Signal_BAT \
            = self.BCR(I_Remain=I_Remain, Charge_Signal=Charge_Signal, I_BAT_Discharge=I_Discharge, SOC=SOC)
        self.I_BAT2 = I_BAT / self.BAT_n_p
        self.I_BAT3 = I_BAT / self.BAT_n_p
        self.I_BAT4 = I_BAT / self.BAT_n_p

        self.I_Bus, self.U_Bus = self.Bus(SOC=SOC, U_BAT=U_BAT, P_Load=self.P_Load_output)
        self.P_Bus = self.I_Bus * self.U_Bus

        self.I_Load1, self.U_Load1, self.I_Load2, self.U_Load2, self.I_Load3, self.U_Load3, \
            P_Load, I_PDM2Load, U_PDM2Load = self.PDM(self.U_Bus, self.I_Bus, Load_Signal)

        self.P_Load1 = self.I_Load1 * self.U_Load1
        self.P_Load2 = self.I_Load2 * self.U_Load2
        self.P_Load3 = self.I_Load3 * self.U_Load3
        self.U_Load_output = torch.where(Load_Signal == 0,
                                         torch.tensor(0.0, device=S_irr_SA.device),
                                         torch.where(Load_Signal == 2,
                                                     torch.tensor(5, device=S_irr_SA.device),
                                                     torch.where(Load_Signal == 1,
                                                                 torch.tensor(12, device=S_irr_SA.device),
                                                                 torch.tensor(5, device=S_irr_SA.device))))
        self.I_Load_output = self.I_Load1 + self.I_Load2 + self.I_Load3
        self.P_Load_output = self.P_Load1 + self.P_Load2 + self.P_Load3

        if self.I_Load_output.shape[0] != self.U_Bus.shape[0]:
            raise Exception("I_Load_output.shape[0] != U_Bus.shape[0]")

        self.T_Load1, self.T_Load2, self.T_Load3 = self.Load(self.P_Load_output)

        return [self.U_SA, self.I_SA, self.P_SA,
                self.U_Load_output, self.I_Load_output, self.P_Load_output,
                self.U_BCR, self.I_BCR, self.P_BCR,
                self.U_BAT2, self.I_BAT2, self.T_BAT2,
                self.U_BAT3, self.I_BAT3, self.T_BAT3,
                self.U_BAT4, self.I_BAT4, self.T_BAT4,
                self.U_Bus, self.I_Bus, self.P_Bus,
                self.U_Load1, self.I_Load1, self.T_Load1, self.P_Load1,
                self.U_Load2, self.I_Load2, self.T_Load2, self.P_Load2,
                self.U_Load3, self.I_Load3, self.T_Load3, self.P_Load3]


def DET2PPT(I_Shunt, I_mp_all, V_mp_all, P_all_sample, I_all_sample, V_all_sample, I_BCR, U_BCR):
    I_BCR = I_BCR - I_Shunt
    P_BCR = I_BCR * U_BCR

    P_SA = I_mp_all * V_mp_all - I_Shunt * U_BCR

    index = torch.argmin(torch.abs(P_all_sample - P_SA.unsqueeze(0)), dim=0)

    U_SA = V_all_sample[index
    , torch.arange(V_all_sample.shape[1]).unsqueeze(1).unsqueeze(2).to(I_Shunt.device)
    , torch.arange(V_all_sample.shape[2]).unsqueeze(0).unsqueeze(2).to(I_Shunt.device)
    , torch.arange(V_all_sample.shape[3]).unsqueeze(0).unsqueeze(1).to(I_Shunt.device)]
    I_SA = I_all_sample[index
    , torch.arange(I_all_sample.shape[1]).unsqueeze(1).unsqueeze(2).to(I_Shunt.device)
    , torch.arange(I_all_sample.shape[2]).unsqueeze(0).unsqueeze(2).to(I_Shunt.device)
    , torch.arange(I_all_sample.shape[3]).unsqueeze(0).unsqueeze(1).to(I_Shunt.device)]

    return U_SA, I_SA, P_SA, U_BCR, I_BCR, P_BCR


class Solar_Array(nn.Module):
    def __init__(self, V_mp_ref_all=24, I_mp_ref_all=2.5):
        super().__init__()

        self.V_mp_ref_all = V_mp_ref_all
        self.I_mp_ref_all = I_mp_ref_all

        self.I_sc_ref_one = 0.306
        self.V_oc_ref_one = 0.55
        self.I_mp_ref_one = 0.2886
        self.V_mp_ref_one = 0.45
        self.S_irr_ref = 1000
        self.T_ref = 25

        self.param_a = 0.0025
        self.param_b = 0.00288
        self.param_c = 0.0005

        self.I_red = 0.001
        self.V_red = 0.001

        self.reset_parameters()

    def reset_parameters(self):
        pass

    def forward(self, S_irr_SA, T_SA, theta):
        I_sc_ref_one = self.I_sc_ref_one
        V_oc_ref_one = self.V_oc_ref_one
        I_mp_ref_one = self.I_mp_ref_one
        V_mp_ref_one = self.V_mp_ref_one
        V_mp = self.V_mp_ref_all
        I_mp = self.I_mp_ref_all
        param_a = self.param_a
        param_b = self.param_b
        param_c = self.param_c
        e = torch.exp(torch.tensor(1.0, device=S_irr_SA.device))

        self.n_p = (self.I_mp_ref_all + self.I_red) // self.I_mp_ref_one
        n_p = self.n_p
        self.n_s = (self.V_mp_ref_all + self.V_red) // self.V_mp_ref_one
        n_s = self.n_s

        S_irr_ref = self.S_irr_ref
        T_ref = self.T_ref
        S_irr = S_irr_SA
        T = T_SA

        D_i = (S_irr / (S_irr_ref + 1e-6)) * (1 + param_a * (T - T_ref))
        D_v = (1 - param_b * (T - T_ref)) * torch.log(e + param_c * (S_irr - S_irr_ref))
        I_sc_one = I_sc_ref_one * D_i
        V_oc_one = V_oc_ref_one * D_v
        I_mp_one = I_mp_ref_one * D_i
        V_mp_one = V_mp_ref_one * D_v

        V_mp_all = n_s * V_mp_one
        I_mp_all = torch.cos(theta) * n_p * I_mp_one
        P_mp_all = V_mp_all * I_mp_all

        C_2 = (V_mp_one / (V_oc_one + 1e-6) - 1) * (1 / torch.log(1 - I_mp_one / (I_sc_one + 1e-6)))
        C_1 = (1 - I_mp_one / (I_sc_one + 1e-6)) * torch.exp(-V_mp_one / ((C_2 * V_oc_one) + 1e-6))

        sample_num = 1000
        sample_tensor = torch.linspace(0, 1, sample_num).unsqueeze(1).unsqueeze(2).unsqueeze(3).to(S_irr_SA.device)
        V_all_sample = sample_tensor * (n_s * V_oc_one - 0.514 * n_s * V_oc_one).unsqueeze(0) + 0.514 * n_s * V_oc_one
        I_all_sample = n_p * torch.cos(theta).unsqueeze(0) * I_sc_one.unsqueeze(0) \
                       * (1 - C_1.unsqueeze(0) * (
                    torch.exp(V_all_sample / ((C_2 * n_s * V_oc_one) + 1e-6).unsqueeze(0)) - 1))
        V_all_sample = V_all_sample * torch.where(theta == 90,
                                                  torch.tensor(0.0, device=S_irr_SA.device),
                                                  torch.tensor(1.0, device=S_irr_SA.device)).unsqueeze(0)
        P_all_sample = V_all_sample * I_all_sample

        if torch.isnan(V_mp_all).any() or torch.isnan(I_mp_all).any() or torch.isnan(P_mp_all).any() \
                or torch.isnan(I_all_sample).any() or torch.isnan(V_all_sample).any() or torch.isnan(
            P_all_sample).any():
            raise ValueError("NAN in Solar Array Model")

        return I_mp_all, V_mp_all, P_mp_all, I_all_sample, V_all_sample, P_all_sample


class Battery(nn.Module):
    def __init__(self, BAT_n_p=5, BAT_n_s=4, Q_ref_BAT=40, charge_Q_thershold=39,
                 lag_step=1, app_order=3, efficiency=0.99, dt=1 / 3600,
                 SPS_Model_PINN_if_has_wrong_Phy=False,
                 wrong_Phy_error=0.00001,
                 random_seed=42):
        super().__init__()

        self.BAT_n_p = BAT_n_p
        self.BAT_n_s = BAT_n_s

        self.Q_ref_BAT = Q_ref_BAT

        self.efficiency = efficiency
        self.dt = dt
        self.Q_start = None
        self.lag_step = lag_step

        self.app_coff_bat_temperature = torch.tensor([-8.375354868, 16.10352945, -15.4012349, -357.3665664])
        self.normal_temperature = 29.6

        self.charge_Q_thershold = charge_Q_thershold
        self.app_coff_fast_charge = torch.tensor([1.7518449429, -210.0860393943, 8398.0398828205, -111885.2255328090])
        self.app_coff_slow_charge = torch.tensor(
            [37.4475839939, -4496.5396273707, 179974.9059945020, -2401161.4298347400])
        self.app_down_charge = 0.08
        self.app_coff_discharge = torch.tensor([0.96106309, -21.87748203])
        self.app_down_discharge = 0.1
        self.app_up_discharge = 0.1

        if SPS_Model_PINN_if_has_wrong_Phy:
            torch.manual_seed(random_seed)
            self.app_coff_bat_temperature = self.app_coff_bat_temperature * (1 + wrong_Phy_error*(torch.rand_like(self.app_coff_bat_temperature)-0.5))
            self.app_coff_fast_charge = self.app_coff_fast_charge * (1 + wrong_Phy_error*(torch.rand_like(self.app_coff_fast_charge)-0.5))
            self.app_coff_slow_charge = self.app_coff_slow_charge * (1 + wrong_Phy_error*(torch.rand_like(self.app_coff_slow_charge)-0.5))
            self.app_coff_discharge = self.app_coff_discharge * (1 + wrong_Phy_error*(torch.rand_like(self.app_coff_discharge)-0.5))

        self.last_U_BAT = torch.tensor([16.8])
        self.last_I_BAT = torch.tensor([-0.6])

        self.reset_parameters()

    def reset_parameters(self):
        pass

    def forward(self, Signal_BAT, I_BAT, Q_start_init):
        if self.Q_start is None:
            self.Q_start = Q_start_init
        Q_now = self.Q_start - torch.cumsum(I_BAT * self.dt * self.lag_step, dim=0) * self.efficiency
        smooth_kernel = torch.tensor([0.1, 0.2, 0.4, 0.2, 0.1], device=Q_now.device)
        Q_now_padded = F.pad(Q_now, (2, 2), mode='replicate')
        Q_now = F.conv1d(Q_now_padded, smooth_kernel.unsqueeze(0).unsqueeze(0), padding=0)
        Q_now = torch.where(Q_now > self.Q_ref_BAT, torch.tensor(self.Q_ref_BAT, device=Q_now.device), Q_now)
        Q_now = torch.where(Q_now < 0, torch.tensor(0.0, device=Q_now.device), Q_now)

        self.Q_start = Q_now[-1:, :, :].data

        Pres_BAT = 2.1258 * Q_now - 1.39915

        SOC = Q_now / self.Q_ref_BAT
        SOC = torch.where(SOC > 1, torch.tensor(1.0, device=SOC.device), SOC)
        SOC = torch.where(SOC < 0, torch.tensor(0.0, device=SOC.device), SOC)

        U_BAT_fast_charge = torch.zeros_like(Q_now)
        for i in range(self.app_coff_fast_charge.shape[0]):
            U_BAT_fast_charge = U_BAT_fast_charge \
                                + self.app_coff_fast_charge[i] * Q_now ** (self.app_coff_fast_charge.shape[0] - 1 - i)

        U_BAT_slow_charge = torch.zeros_like(Q_now)
        for i in range(self.app_coff_slow_charge.shape[0]):
            U_BAT_slow_charge = U_BAT_slow_charge \
                                + self.app_coff_slow_charge[i] * Q_now ** (self.app_coff_slow_charge.shape[0] - 1 - i)

        U_BAT_discharge = torch.zeros_like(Q_now)
        for i in range(self.app_coff_discharge.shape[0]):
            U_BAT_discharge = U_BAT_discharge + self.app_coff_discharge[i] * Q_now ** (self.app_coff_discharge.shape[0] - 1 - i)

        U_BAT_charge = torch.where(Q_now <= self.charge_Q_thershold, U_BAT_fast_charge, U_BAT_slow_charge)

        charge_end_mask = (I_BAT[:, :, :-1] < -0.02) & (I_BAT[:, :, 1:] > -0.02)
        charge_end_mask = torch.cat((charge_end_mask, torch.zeros_like(I_BAT[:, :, :1])), dim=-1).float()
        discharge_start_mask = (I_BAT[:, :, :-1] < 0.05) & (I_BAT[:, :, 1:] > 0.05)
        discharge_start_mask = torch.cat((discharge_start_mask, torch.zeros_like(I_BAT[:, :, :1])), dim=-1).float()
        discharge_end_mask = (I_BAT[:, :, :-1] > 0.05) & (I_BAT[:, :, 1:] < 0.05)
        discharge_end_mask = torch.cat((discharge_end_mask, torch.zeros_like(I_BAT[:, :, :1])), dim=-1).float()
        U_BAT_charge = U_BAT_charge \
                       - charge_end_mask * self.app_down_charge \
                       - discharge_start_mask * self.app_down_discharge
        U_BAT_discharge = U_BAT_discharge \
                          + discharge_end_mask * self.app_up_discharge
        
        self.last_U_BAT = self.last_U_BAT.to(U_BAT_discharge.device)
        self.last_I_BAT = self.last_I_BAT.to(U_BAT_discharge.device)
        
        if (self.last_I_BAT < -0.02).all() and (I_BAT[:1, :, :-1] < -0.02).all() and (I_BAT[:1, :, -1:] > -0.02).all():
            self.last_U_BAT[:, :, -1:] = self.last_U_BAT[:, :, -1:] - self.app_down_charge
        if (self.last_I_BAT < 0.05).all() and (I_BAT[:1, :, :-1] < 0.05).all() and (I_BAT[:1, :, -1:] > 0.05).all():
            self.last_U_BAT[:, :, -1:] = self.last_U_BAT[:, :, -1:] - self.app_down_discharge
        if (self.last_I_BAT > 0.05).all() and (I_BAT[:1, :, :-1] > 0.05).all() and (I_BAT[:1, :, -1:] < 0.05).all():
            self.last_U_BAT[:, :, -1:] = self.last_U_BAT[:, :, -1:] + self.app_up_discharge

        U_BAT = torch.where(I_BAT < -0.02,
                            U_BAT_charge,
                            torch.where(I_BAT > 0.05,
                                        U_BAT_discharge,
                                        torch.tensor(float('nan'), device=I_BAT.device)))
        U_BAT[:1, :, :] = torch.where(torch.isnan(U_BAT[:1, :, :]),
                                      self.last_U_BAT,
                                      U_BAT[:1, :, :])
        
        NAN_mask = torch.isnan(U_BAT)
        not_NAN_mask = ~NAN_mask
        not_NAN_mask01 = not_NAN_mask.float()
        values, indices = not_NAN_mask01.cummax(dim=0)
        U_BAT = torch.gather(U_BAT, dim=0, index=indices)
        if torch.isnan(U_BAT).any():
            raise ValueError("NAN in (U_BAT)")
        
        self.last_U_BAT = U_BAT[-1:, :, :].data
        self.last_I_BAT = I_BAT[-1:, :, :].data

        I_BAT_each_set = I_BAT / self.BAT_n_p

        T_BAT = self.app_coff_bat_temperature[0] * I_BAT \
            + self.app_coff_bat_temperature[1] * Q_now \
            + self.app_coff_bat_temperature[2] * U_BAT \
            + self.app_coff_bat_temperature[3]

        if torch.isnan(Q_now).any() or torch.isnan(SOC).any() or torch.isnan(U_BAT).any() \
                or torch.isnan(Pres_BAT).any() or torch.isnan(T_BAT).any() or torch.isnan(I_BAT_each_set).any():
            raise ValueError("NAN in Battery Model")

        return Q_now, SOC, U_BAT, Pres_BAT, T_BAT, I_BAT_each_set


class Power_Distribution_Module(nn.Module):
    def __init__(self):
        super().__init__()
        self.reset_parameters()

    def reset_parameters(self):
        pass

    def forward(self, U_Bus, I_Bus, Load_Signal):
        P_Bus = U_Bus * I_Bus

        I_Load2 = torch.where(Load_Signal == 2, torch.tensor(2.9, device=U_Bus.device),
                              torch.tensor(0.0, device=U_Bus.device))
        U_Load2 = torch.where(Load_Signal == 2, P_Bus / (I_Load2 + 1e-6), torch.tensor(5.3, device=U_Bus.device))

        I_Load3 = torch.where(Load_Signal == 3, torch.tensor(2.9, device=U_Bus.device),
                              torch.tensor(0.0, device=U_Bus.device))
        U_Load3 = torch.where(Load_Signal == 3, P_Bus / (I_Load3 + 1e-6), torch.tensor(5.4, device=U_Bus.device))

        I_Load1 = torch.where(Load_Signal == 1, torch.tensor(2.4, device=U_Bus.device),
                              torch.tensor(0.0, device=U_Bus.device))
        U_Load1 = torch.where(Load_Signal == 1, P_Bus / (I_Load1 + 1e-6), torch.tensor(12.1, device=U_Bus.device))

        P_Load = U_Load1 * I_Load1 + U_Load2 * I_Load2 + U_Load3 * I_Load3
        I_PDM2Load = I_Load1 + I_Load2 + I_Load3
        U_PDM2Load = P_Load / (I_PDM2Load + 1e-6)

        if torch.isnan(I_Load1).any() or torch.isnan(U_Load1).any() or torch.isnan(I_Load2).any() \
                or torch.isnan(U_Load2).any() or torch.isnan(I_Load3).any() or torch.isnan(U_Load3).any() \
                or torch.isnan(P_Load).any() or torch.isnan(I_PDM2Load).any() or torch.isnan(U_PDM2Load).any():
            raise ValueError("NAN in Power Distribution Module Model")

        return I_Load1, U_Load1, \
            I_Load2, U_Load2, \
            I_Load3, U_Load3, \
            P_Load, I_PDM2Load, U_PDM2Load


class Shunt_Regulator(nn.Module):
    def __init__(self):
        super().__init__()
        self.BCR_voltage_drop = 0.02
        self.reset_parameters()

    def reset_parameters(self):
        pass

    def forward(self, P_Load, U_Bus, I_SA, U_SA, I_BAT, U_BAT):
        U_BCR = U_BAT + self.BCR_voltage_drop
        I_BCR = I_SA * U_SA / (U_BCR + 1e-6)

        I_Remain = I_BCR - (P_Load / (U_Bus + 1e-6))

        Charge_Signal = torch.where(I_Remain > 0,
                                    torch.tensor(1.0, device=I_Remain.device),
                                    torch.tensor(0.0, device=I_Remain.device))
        Discharge_Signal = torch.where(I_Remain < 0,
                                       torch.tensor(1.0, device=I_Remain.device),
                                       torch.tensor(0.0, device=I_Remain.device))

        I_BAT_Charge = torch.where(I_BAT < 0,
                                   I_BAT,
                                   torch.tensor(0.0, device=I_BAT.device))
        I_Shunt = torch.where(I_Remain + I_BAT_Charge > 0,
                              I_Remain + I_BAT_Charge,
                              torch.tensor(0.0, device=I_Remain.device))

        if torch.isnan(U_BCR).any() or torch.isnan(I_BCR).any() or torch.isnan(I_Remain).any() \
                or torch.isnan(Charge_Signal).any() or torch.isnan(Discharge_Signal).any() or torch.isnan(
            I_Shunt).any():
            raise ValueError("NAN in Shunt Regulator Model")

        return U_BCR, I_BCR, I_Remain, Charge_Signal, Discharge_Signal, I_Shunt


class Battery_Charging_Regulator(nn.Module):
    def __init__(self, Q_ref_BAT=40, SOC_if_trickle_charge=0.99,
                 SPS_Model_PINN_if_has_wrong_Phy=False,
                 wrong_Phy_error=0.00001,
                 random_seed=42):
        super().__init__()

        self.Q_ref_BAT = Q_ref_BAT

        self.SOC_if_trickle_charge = SOC_if_trickle_charge
        self.Rate_Charge_Fast = 60.606
        self.Rate_Charge_Trickle = 1000
        self.Rate_Charge_Fast_app_coff = torch.tensor([348.7745365064, -0.3635008852])

        if SPS_Model_PINN_if_has_wrong_Phy:
            torch.manual_seed(random_seed)
            self.Rate_Charge_Fast_app_coff = self.Rate_Charge_Fast_app_coff * \
                                             (1 + wrong_Phy_error * (torch.rand_like(self.Rate_Charge_Fast_app_coff) - 0.5))

        self.reset_parameters()

    def reset_parameters(self):
        pass

    def forward(self, I_Remain, Charge_Signal, I_BAT_Discharge, SOC):
        I_BAT_Charge_wanted = torch.where(SOC <= self.SOC_if_trickle_charge,
                                          self.Rate_Charge_Fast_app_coff[0] * torch.log(SOC) +
                                          self.Rate_Charge_Fast_app_coff[1],
                                          torch.where(SOC < 1,
                                                      -1 * self.Q_ref_BAT / self.Rate_Charge_Trickle,
                                                      torch.tensor(0.0, device=SOC.device)))
        I_BAT_Charge_able = torch.where(I_Remain <= (-1 * I_BAT_Charge_wanted),
                                        -1 * I_Remain,
                                        I_BAT_Charge_wanted)
        I_BAT_Charge = torch.where(Charge_Signal > 0, I_BAT_Charge_able, torch.tensor(0.0, device=I_Remain.device))
        I_BAT = torch.where(I_BAT_Charge != 0, I_BAT_Charge, I_BAT_Discharge)
        
        Signal_BAT = torch.where(I_BAT > 0,
                                 torch.tensor(1.0, device=I_Remain.device),
                                 torch.where(I_BAT < 0,
                                             torch.tensor(-1.0, device=I_Remain.device),
                                             torch.tensor(0.0, device=I_Remain.device)))

        if torch.isnan(I_BAT).any() or torch.isnan(Signal_BAT).any():
            raise ValueError("NAN in Battery Charging Regulator Model")

        return I_BAT, Signal_BAT


class Battery_Discharge_Regulator(nn.Module):
    def __init__(self):
        super().__init__()
        self.reset_parameters()

    def reset_parameters(self):
        pass

    def forward(self, Discharge_Signal, U_BAT, I_Remain):
        I_BAT2Load = -1 * torch.where(Discharge_Signal > 0,
                                      I_Remain,
                                      torch.tensor(0.0, device=I_Remain.device))

        Duty_Factor = torch.tensor(0.0, device=I_Remain.device)

        I_Discharge = I_BAT2Load

        U_BAT2Load = U_BAT / (1 - Duty_Factor)
        U_BAT2Load = torch.where(Discharge_Signal > 0, U_BAT2Load, torch.tensor(0.0, device=I_Remain.device))

        if torch.isnan(Duty_Factor).any() or torch.isnan(I_Discharge).any() or torch.isnan(I_BAT2Load).any() \
                or torch.isnan(U_BAT2Load).any():
            raise ValueError("NAN in Battery Discharge Regulator Model")

        return Duty_Factor, I_Discharge, I_BAT2Load, U_BAT2Load


class Bus_Flow(nn.Module):
    def __init__(self):
        super().__init__()

        self.Bus_U_Drop = 0.05
        self.SOC_threshold = 0.2

        self.reset_parameters()

    def reset_parameters(self):
        pass

    def forward(self, SOC, U_BAT, P_Load):
        U_Bus = torch.where(SOC <= self.SOC_threshold,
                            torch.tensor(0.0, device=SOC.device),
                            U_BAT - self.Bus_U_Drop)

        I_Bus = P_Load / (U_Bus + 1e-6)

        if torch.isnan(I_Bus).any() or torch.isnan(U_Bus).any():
            raise ValueError("NAN in Bus Flow Model")

        return I_Bus, U_Bus


class Load(nn.Module):
    def __init__(self, Load_TP_curve_app_order=3, dt=1 / 3600):
        super().__init__()
        self.P_Load_reminder = None

        self.param_a = (1 / 0.5 / 3600) / dt

        self.TP_app_coff_load1 = torch.tensor(
            [-4.7560399174e-24, 4.6927350101e-19, -1.6336017995e-14, 2.4922206954e-10, -1.6669744152e-6,
             4.6887279008e-3, 29.719951905])
        self.TP_app_coff_load2 = torch.tensor(
            [-1.3100104821e-23, 9.1852969377e-19, -2.3210088147e-14, 2.5163775164e-10, -1.0234581423e-6,
             4.3372246013e-4, 34.774505284])
        self.TP_app_coff_load3 = torch.tensor(
            [-2.0420602233e-25, 1.4375347108e-19, -8.0874245993e-15, 1.6608841814e-10, -1.4692362383e-6,
             5.5559090911e-3, 30.126537016])

        self.reset_parameters()

    def reset_parameters(self):
        pass

    def forward(self, P_Load_output):
        if self.P_Load_reminder is None:
            self.P_Load_reminder = torch.zeros_like(P_Load_output)
        P_Load_reminder = torch.cumsum(P_Load_output, dim=0) / self.param_a + self.P_Load_reminder
        self.P_Load_reminder = P_Load_reminder[-1:, :, :].data
        P_Load_reminder = torch.fmod(P_Load_reminder, 24945)

        T_Load1 = torch.zeros_like(P_Load_reminder)
        T_Load2 = torch.zeros_like(P_Load_reminder)
        T_Load3 = torch.zeros_like(P_Load_reminder)

        for i in range(self.TP_app_coff_load1.shape[0]):
            T_Load1 = T_Load1 + self.TP_app_coff_load1[i] * P_Load_reminder ** (self.TP_app_coff_load1.shape[0] - 1 - i)
            T_Load2 = T_Load2 + self.TP_app_coff_load2[i] * P_Load_reminder ** (self.TP_app_coff_load2.shape[0] - 1 - i)
            T_Load3 = T_Load3 + self.TP_app_coff_load3[i] * P_Load_reminder ** (self.TP_app_coff_load3.shape[0] - 1 - i)

        if torch.isnan(T_Load1).any() or torch.isnan(T_Load2).any() or torch.isnan(T_Load3).any():
            raise ValueError("NAN in Load Model")

        return T_Load1, T_Load2, T_Load3
