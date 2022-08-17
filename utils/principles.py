import numpy as np

import constant.physical_constant
from constant import physical_constant


def rotate_speed_of_pump(s, p, f=50):
    """
    :param s:           电机转差率
    :param p:           电动机极对数
    :param f:           交流电频率

    :return:            水泵转速
    """
    n = (1 - s) * (60 * f) / p
    return n


def quantity_of_flow(vol_eff, d, b, exc_coef, v_2r):
    """
    :param vol_eff:     容积效率
    :param d:           叶轮外径
    :param b:           叶轮出口宽度
    :param exc_coef:    排挤系数
    :param v_2r:        流体绝对速度的径向分速度

    :return:            水泵机组流量 (m3/h)
    """

    g = vol_eff * d * b * exc_coef * v_2r * np.pi
    return g * 3600


def get_pump_head(paras=None, density=1000):
    if paras is not None:
        """
        p1,p2 = 出口处液体压力
        c1,c2 = 进出口处的流速
        z1,z2 = 进出口高度
        """
        p1, p2, c1, c2, z1, z2 = paras
        return (p2 - p1) / density * physical_constant.GRAVITY + (
                c2 - c1) / 2 * physical_constant.GRAVITY + z2 - z1


# 容积损失量
def get_volume_loss(delta_p, f, kv):
    """
    :param delta_p:     间隙两端压力差
    :param f:           间隙断面面积
    :param kv:          流量系数

    :return:            容积损失量
    """
    return kv * f * np.sqrt(2 * constant.physical_constant.GRAVITY * delta_p)
