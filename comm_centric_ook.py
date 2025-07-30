from scaling.scaling_ook_comm import *
from settings.soc_struct import *
from settings.socs_to_test import *

import sys

if __name__ == "__main__":

  scaling_arg = 'high_margin'
  if len(sys.argv) > 1:
    if sys.argv[1] == 'naive' or sys.argv[1] == 'high_margin':
      scaling_arg = sys.argv[1]

  #Print into a log file
  original_stdout = sys.stdout
  log_file = open("logs/log_ook_output.txt", "w")
  sys.stdout = log_file #print into log file

  #parameters
  fontscale = 1
  channel_density = 1

  #change to get different plots
  if scaling_arg == 'naive':
    antenna_density = 1
  else:
    antenna_density = 0

  extra_sensing_power = 1
  extra_non_sensing_power = 1

  max_ch = (8192 + 1024 + 8192)*2
  plot_name = scaling_arg

  for soc in socs:

    x_inter, y_inter, ni_channels, sensing_power_consumption, non_sensing_power_consumption, total_power_consumption, total_power_budget, sensing_area, non_sensing_area, total_area = \
      scaling_ook_comm(soc[0], physical_specs45, soc[1], channel_density, antenna_density, extra_sensing_power, extra_non_sensing_power, show=False, maximum_channels=max_ch)

    name = soc[0]["Name"]

    #Save data for next run
    np.savetxt(f'data/ook_data{name}_{antenna_density}.txt', np.column_stack((ni_channels, sensing_power_consumption, total_power_budget, non_sensing_power_consumption, total_power_consumption, sensing_area, total_area)))

