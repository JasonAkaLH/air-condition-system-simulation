from cooling_system.variable_frequency_pump import VariableFrequencyPump
import scipy

if __name__ == '__main__':

    test = VariableFrequencyPump(pole_pairs=2, rated_speed=400, power=30, flow=600, pump_head=10)

    pump_head = test.get_pump_head(n=40)
    flow = test.get_flow(n=40)
    print(pump_head, flow)
