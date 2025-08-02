#from run_networks import *
from scaling.scaling_layers_comp import *
from settings.soc_struct import *
from settings.socs_to_test import *

import sys

if __name__ == "__main__":

  #Print into a log file
  original_stdout = sys.stdout
  log_file = open("logs/log_dnn_layers_output.txt", "w")

  #parameters
  dnn_types = ["mlp", "dense"]
  step = 32
  max_ch = 8192+1024+1024+1024
  standard = 1024

  with open('socs_intersections_layers.py', 'w') as file:
    file.write(f'\n')

  for soc in socs:

    for i in range(len(dnn_types)):
      sys.stdout = log_file #print into log file

      #Choose a DNN architecture to run
      plot_name = dnn_types[i]
      if dnn_types[i] == "mlp":
        dnn_arch = mlp_architecture.copy()
      elif dnn_types[i] == "dense":
        dnn_arch = densenet_architecture.copy()
      else:
        dnn_arch = densenet_architecture.copy()

      x_inter, y_inter, ni_channels, sensing_power_consumption, non_sensing_power_consumption, total_power_consumption, total_power_budget = \
        scaling_layers_comp(soc[0], dnn_arch, physical_specs45, soc[1], show=False, maximum_channels=max_ch, step=step)

      name = soc[0]["Name"]

      np.savetxt(f'data/{plot_name}_comp_layers_data{name}.txt', np.column_stack((total_power_budget, total_power_consumption, ni_channels)))

      print("INTERSECTION ", x_inter, y_inter)

      soc[0]["budget_cutoff"][i] = x_inter

      if i == len(dnn_types)-1:
        intersections_list = []
        for intersection in soc[0]["budget_cutoff"]:
          if len(intersection) != 0:
            if intersection[0] > standard + 25:
              intersections_list.append(intersection[0])
            else:
              intersections_list.append(None)
          else:
            intersections_list.append(None)
        with open('socs_intersections_layers.py', 'a') as file:
          file.write(f"list_layers_{name} = [")
          file.write(','.join(map(str,intersections_list)))
          file.write(']\n')


      sys.stdout = original_stdout
      print(f"Finished SoC {name} {plot_name}")

