from scaling.scaling_all_opt_comp import *
from settings.soc_struct import *
from settings.socs_to_test import *

if __name__ == "__main__":

  #Print into a log file
  original_stdout = sys.stdout
  log_file = open("logs/log_opt_output.txt", "w")

  #parameters
  dnn_types = ["mlp"]
  step = 1024
  max_ch = 9472
  min_ch = 1024

  for soc in socs:

    for i in range(len(dnn_types)):
      if dnn_types[i] == "mlp":
        dnn_arch = mlp_architecture.copy()
      elif dnn_types[i] == "dense":
        dnn_arch = densenet_architecture.copy()
      else:
        dnn_arch = densenet_architecture.copy()

      name = soc[0]["Name"]
      dnn_type = dnn_arch["DNN"]

      print(f"Starting SoC {name} {dnn_type}")
      sys.stdout = log_file #print into log file

      total_power_budget_plot, default_size, default_power_budget, dropout_size, dropout_power_budget, layers_dropout_size, layers_dropout_power_budget, tech_size, tech_power_budget, density_size, density_power_budget, power_consumption_default_plot, power_consumption_dropout_plot, power_consumption_layers_dropout_plot, power_consumption_tech_plot, power_consumption_density_plot, x_axis = \
        scaling_all_opt_comp(soc[0], dnn_arch, physical_specs45, soc[1], show=False, maximum_channels=max_ch, step=step, minimum_channels=min_ch)

      np.savetxt(f'data/1_{dnn_type}_all_opt_data{name}.txt', np.column_stack((total_power_budget_plot, power_consumption_default_plot, power_consumption_dropout_plot, power_consumption_layers_dropout_plot, power_consumption_tech_plot, power_consumption_density_plot, x_axis)))

      np.savetxt(f'data/2_{dnn_type}_all_opt_data{name}.txt', np.column_stack((default_size, default_power_budget, dropout_size, dropout_power_budget, layers_dropout_size, layers_dropout_power_budget, tech_size, tech_power_budget, density_size, density_power_budget)))

      sys.stdout = original_stdout
      print(f"Finished SoC {name} {dnn_type}")


