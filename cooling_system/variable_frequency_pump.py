import numpy as np
from utils import principles
from sklearn.linear_model import LinearRegression
from constant import string_constant, physical_constant


class VariableFrequencyPump:

    def __init__(self, pole_pairs: int, rated_speed: float, power: float, flow: float,
                 pump_head: float, freq: float = 50, density: float = 1000,
                 hardware_paras: list = None, VE: float = None, eff: float = 0.8):
        """
        :param pole_pairs:          电机极对数
        :param power:               额定功率
        :param flow:                额定流量(m3/h)
        :param rated_speed:         电机额定转速
        :param pump_head:           额定扬程
        :param freq:                额定频率
        :param density:             液体密度(default--水)
        :param VE                   容积效率
        :param eff                  水泵效率(default=0.8)
        :param hardware_paras:      硬件参数：1、叶轮外径; 2、叶轮出口宽度; 3、排挤系数
        """
        self.flow = flow
        self.pole_pairs = pole_pairs
        self.rated_speed = rated_speed
        self.pump_head = pump_head
        self.freq = freq
        self.density = density
        self.power = power
        self.hardware_paras = hardware_paras
        self.impeller_diameter = None
        self.impeller_outlet_width = None
        self.VE = VE
        self.eff = eff
        self.excretion_coefficient = None
        self.flow_ratio_model = FlowRatioModel()
        self.pump_head_model = PumpHeadRatioModel()

        if self.hardware_paras:
            self.range_hardware_paras()

    def range_hardware_paras(self):
        self.impeller_diameter = self.hardware_paras[0]
        self.impeller_outlet_width = self.hardware_paras[1]
        self.excretion_coefficient = self.hardware_paras[2]

    # 流量
    def get_flow(self, in_power=None, h=None, paras=None, n=None, v_2r=None):
        """
        :param v_2r:            流体绝对速度的径向分速度
        :param in_power:        轴功率
        :param h:               扬程
        :param paras:           计算扬程的参数
        :param n:               转速
        :return:                实时流量 (m3/h)
        """
        if paras and h is None and n is None:
            h = principles.get_pump_head(paras=paras, density=self.density)
        elif n and h is None and paras is None:
            flow, _, _ = self.get_values_by_rotate(n)
            return flow

        if v_2r:
            return self.__get_flow_accurately(v_2r)

        return in_power * self.eff / 2.73 * h

    def __get_flow_accurately(self, v_2r):
        return self.VE * np.pi * self.impeller_diameter * self.impeller_outlet_width * self.excretion_coefficient * v_2r

    # 轴功率
    def get_in_power(self, q, h=None, paras=None, n=None):
        """
        :param q:               实时流量 (m3/h)
        :param h:               扬程
        :param paras:           计算扬程的参数
        :param n:               转速
        :return:                轴功率
        """
        if paras and h is None:
            h = principles.get_pump_head(paras=paras, density=self.density)
        elif h is None and paras is None and n:
            _, _, power = self.get_values_by_rotate(n)
            return power

        return q * 2.73 * h / self.eff

    # 扬程
    def get_pump_head(self, paras: list = None, n=None, in_power=None, q=None):
        """
        :param q:               实时流量 (m3/h)
        :param in_power:        轴功率
        :param paras:           液体压力差、流速差，出入口高度差
        :param n:               转速
        :return:                扬程
        """
        if paras:
            return principles.get_pump_head(paras=paras, density=self.density)
        elif n:
            _, head, _ = self.get_values_by_rotate(n)
            return head
        elif q and in_power:
            return in_power * self.eff / 2.73 * q

    def __get_values_by_f_n(self, para_0, para_n):
        flow = para_n / para_0 * self.flow if self.flow and self.rated_speed else None
        head = (para_n / para_0) ** 2 * self.pump_head if self.pump_head and self.rated_speed else None
        in_power = (para_n / para_0) ** 3 * self.power if self.power and self.rated_speed else None
        return flow, head, in_power

    # 转速 ---- 流量、扬程、功率
    def get_values_by_rotate(self, n):
        return self.__get_values_by_f_n(self.rated_speed, n)

    # 频率 ---- 流量、扬程、功率
    def get_values_by_freq(self, f):
        return self.__get_values_by_f_n(self.freq, f)

    # 容积效率
    def cal_ve(self, q=None, kv=None, cv=None, volume_loss_paras: list = None):
        """
        :param q:                   容积损失量
        :param kv:                  流量系数（国标）
        :param cv:                  流量系数（欧标）
        :param volume_loss_paras:   计算容积损失量的参数

        :return:                    容积效率
        """
        if kv is None and cv:
            kv = cv / 1.156
        if q is None and volume_loss_paras:
            delta_p, f = volume_loss_paras[0], volume_loss_paras[1]
            q = principles.get_volume_loss(delta_p, f, kv) if kv else None

        ve = self.flow / q + self.flow if q else None
        return ve


# 流量比模型
class FlowRatioModel:
    def __init__(self):
        self.a = 1
        self.b = 0
        self.model = LinearRegression()

    def fit(self, X, y):
        X = np.array(X)
        if len(X.shape) != 1:
            raise ValueError('X维度错误！')

        self.model.fit(X.reshape(-1, 1), y)
        self.a = self.model.coef_[0][0]
        self.b = self.model.intercept_[0]

    def predict(self, X):
        X = np.array(X)
        if len(X.shape) != 1:
            raise ValueError('X维度错误！')

        y = self.model.predict(X)
        return y

    def get_flow_by_rotate(self, n_0, n_N, flow_0):
        X = np.array([n_N / n_0]).reshape(-1, 1)
        ratio = self.model.predict(X)
        return ratio * flow_0

    def get_flow_by_freq(self, f_0, f_N, flow_0):
        X = np.array([f_N / f_0]).reshape(-1, 1)
        ratio = self.model.predict(X)
        return ratio * flow_0

    def get_freq_n_rotate_by_flow(self, flow_0, flow_N, f_0=None, n_0=None):
        ratio = flow_N / flow_0
        fn_ratio = (ratio - self.b) / self.a
        f_N = fn_ratio * f_0 if f_0 else None
        n_N = fn_ratio * n_0 if n_0 else None

        return f_N, n_N


class PumpHeadRatioModel():
    def __init__(self):
        self.a = 1
        self.b = 0
        self.c = 0
        self.model = LinearRegression()

    def fit(self, X, y):
        X = np.array(X)
        if X.shape[-1] != 2:
            raise ValueError('X维度错误！')

        self.model.fit(X, y)
        self.a = self.model.coef_[0]
        self.b = self.model.coef_[1]
        self.c = self.model.intercept_

    def predict(self, X):
        X = np.array(X)
        if X.shape[-1] != 2:
            raise ValueError('X维度错误！')

        y = self.model.predict(X)
        return y

    def get_head_by_rotate(self, n_N, n_0, h_0):
        n_N = np.array(n_N).reshape(-1, 1)
        X = np.array([(n_N / n_0) ** 2, n_N / n_0]).T
        ratio = self.model.predict(X)
        return ratio * h_0

    def get_head_by_freq(self, f_0, f_N, h_0):
        f_N = np.array(f_N).reshape(-1, 1)
        X = np.array([(f_N / f_0) ** 2, f_N / f_0]).T
        ratio = self.model.predict(X)
        return ratio * h_0

    def get_freq_n_rotate_by_head(self, h_0, h_N, f_0=None, n_0=None):
        ratio = h_N / h_0
        c = ratio - self.c
        d = np.square(self.b) - (4 * self.a * c)
        x1 = (-self.b - np.sqrt(d)) / (2 * self.a)
        x2 = (-self.b + np.sqrt(d)) / (2 * self.a)

        fn_ratio = x1 if x1 >= 0 else x2
        f_N = fn_ratio * f_0 if f_0 else None
        n_N = fn_ratio * n_0 if n_0 else None

        return f_N, n_N
