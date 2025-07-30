from scaling.scaling_qam_comm import *
from settings.soc_struct import *
from settings.socs_to_test import *

import sys

if __name__ == "__main__":


  #Print into a log file
  original_stdout = sys.stdout
  log_file = open("logs/log_qam_output.txt", "w")

  #parameters
  qam_efficiency = 0
  qam_max = 99
  qam_min = 1
  plot_name = 'qam'
  step = 128

  channel_list = list()

  for i in range(9216):
    channel_list.append(i)

  for soc in socs:

    sys.stdout = log_file #print into log file

    channel_accurate_list = list()
    efficiency_list = list()
    efficiency_accurate_list = list()
    for i in range(9216):
      efficiency_list.append(0)

    for eff in range(qam_min, qam_max+1):

      qam_efficiency = eff
      soc[1]["qam_efficiency"] = qam_efficiency

      x_inter, y_inter, ni_channels, sensing_power_consumption, non_sensing_power_consumption, total_power_consumption, total_power_budget = \
      scaling_qam_comm(soc[0], physical_specs45, soc[1], show=False, maximum_channels=8192+1024, step=step)

      if len(x_inter) != 0:
        print("INTERSECTION ", x_inter[0])

        if int(x_inter[0]) not in channel_accurate_list:
          channel_accurate_list.append(int(x_inter[0]))
          efficiency_accurate_list.append(qam_efficiency)

    name = soc[0]["Name"]

    np.savetxt(f'data/qam_data{name}.txt', np.column_stack((channel_accurate_list, efficiency_accurate_list)), header='x y')

    for i in range(len(efficiency_accurate_list)):
      if efficiency_accurate_list[i] != 0:
        print("efficiency ", efficiency_accurate_list[i])
        print("channels ", channel_accurate_list[i])

    sys.stdout = original_stdout #Print only final results to stdout
    print(f"Finished SoC {name}")
