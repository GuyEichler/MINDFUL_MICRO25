from scaling.scaling_all_opt_comp import *
from settings.soc_struct import *
from settings.socs_to_test import *

if __name__ == "__main__":

  width = 10
  height = 5
  fig, ax = plt.subplots(figsize=(width, height))
  ctr = 0

  #Print into a log file
  original_stdout = sys.stdout
  log_file = open("logs/log_opt_show.txt", "w")
  sys.stdout = log_file #print into log file

  #parameters
  dnn_types = ["mlp"]

  for soc in socs:

    for i in range(len(dnn_types)):
      if dnn_types[i] == "mlp":
        dnn_arch = mlp_architecture.copy()
      elif dnn_types[i] == "dense":
        dnn_arch = densenet_architecture.copy()
      else:
        dnn_arch = densenet_architecture.copy()    #dnn_arch = s2s_architecture.copy()

      name = soc[0]["Name"]
      dnn_type = dnn_arch["DNN"]

      data = np.loadtxt(f'data/1_{dnn_type}_all_opt_data{name}.txt')
      total_power_budget_plot, power_consumption_default_plot, power_consumption_dropout_plot, power_consumption_layers_dropout_plot, power_consumption_tech_plot, power_consumption_density_plot, x_axis = data.T  # Transpose to unpack columns

      data = np.loadtxt(f'data/2_{dnn_type}_all_opt_data{name}.txt')
      default_size, default_power_budget, dropout_size, dropout_power_budget, layers_dropout_size, layers_dropout_power_budget, tech_size, tech_power_budget, density_size, density_power_budget = data.T

      x_axis = x_axis.tolist()


      plt.figure(2*ctr+1, figsize=(width, height))

      mid_index = len(total_power_budget_plot) // 2
      label_x, label_y = x_axis[mid_index], total_power_budget_plot[mid_index]
      angle = np.arctan2(total_power_budget_plot[mid_index+1] - total_power_budget_plot[mid_index-1], x_axis[mid_index+1] - x_axis[mid_index-1]) * 35 #/ np.pi
      print(angle)
      angle_degrees = np.degrees(angle)
      print(angle_degrees)


      plt.xlabel('Number of NI Channels', fontsize=22)
      plt.ylabel('Normalized Power', fontsize=22)


      bar_width = 0.2
      categories = ['2048', '4096', '8192']
      x = np.arange(len(categories))
      default_power = list()
      # layer_power = list()
      dropout_power = list()
      layers_dropout_power = list()
      tech_power = list()
      density_power = list()
      power_budget_list = list()

      dropout_model_size = list()
      layers_dropout_model_size = list()
      tech_model_size = list()
      density_model_size = list()

      for i in range(len(categories)):
        channels = 2048 * 2**i
        power_budget = total_power_budget_plot[x_axis.index(channels)]
        power_budget_default = default_power_budget[channels]
        power_budget_dropout = dropout_power_budget[channels]
        power_budget_layers_dropout = layers_dropout_power_budget[channels]
        power_budget_tech = tech_power_budget[channels]
        power_budget_density = density_power_budget[channels]

        print("power budget:", power_budget)

        print("power budget dense:", power_budget_density)
        print("power consumption dense", power_consumption_density_plot[x_axis.index(channels)])
        default_power = default_power + [power_consumption_default_plot[x_axis.index(channels)]/power_budget]
        print(default_power)

        dropout_power = dropout_power + [power_consumption_dropout_plot[x_axis.index(channels)]/power_budget_dropout]
        print(dropout_power)
        layers_dropout_power = layers_dropout_power + [power_consumption_layers_dropout_plot[x_axis.index(channels)]/power_budget_layers_dropout]
        print(layers_dropout_power)
        tech_power = tech_power + [power_consumption_tech_plot[x_axis.index(channels)]/power_budget_tech]
        print(tech_power)
        density_power = density_power + [power_consumption_density_plot[x_axis.index(channels)]/power_budget_density]
        print(density_power)
        power_budget_list = power_budget_list + [1]

      plt.bar(x - 1.5*bar_width, dropout_power, width=bar_width, label='ChDr')
      plt.bar(x - 0.5*bar_width, layers_dropout_power, width=bar_width, label='La+ChDr')
      plt.bar(x + 0.5*bar_width, tech_power, width=bar_width, label='La+ChDr+Tech')
      plt.bar(x + 1.5*bar_width, density_power, width=bar_width, label='La+ChDr+Tech+Dense')

      for i in range(len(categories)):
        channels = 2048 * 2**i
        dropout_model_size.append(dropout_size[channels]/default_size[channels]*100)
        layers_dropout_model_size.append(layers_dropout_size[channels]/default_size[channels]*100)
        tech_model_size.append(tech_size[channels]/default_size[channels]*100)
        density_model_size.append(density_size[channels]/default_size[channels]*100)
        plt.text(x[i] - 1.5*bar_width, dropout_power[i], str(round(dropout_power[i], 2)), ha='center', va='bottom', weight='normal', fontsize=16) #dropout
        plt.text(x[i] - 0.5*bar_width, layers_dropout_power[i], str(round(layers_dropout_power[i],2)), ha='center', va='bottom', weight='normal', fontsize=16) #layer+dropout
        plt.text(x[i] + 0.5*bar_width, tech_power[i], str(round(tech_power[i],2)), ha='center', va='bottom', weight='normal', fontsize=16) #tech
        plt.text(x[i] + 1.5*bar_width, density_power[i], str(round(density_power[i],2)), ha='center', va='bottom', weight='normal', fontsize=16) #density

      #plt.yscale('log')
      plt.xticks(x, categories, fontsize=18)
      plt.yticks(fontsize=18)
      plt.ylim(0, 1.1)
      plt.legend(fontsize=16,loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=4,columnspacing=0.5, frameon=False)

      plt.tight_layout()

      plt.savefig(f"figures/{name}_{dnn_type}_all_opt_scale_power.pdf")

      plt.figure(2*ctr+2, figsize=(width, height))
      bar_width = 0.2

      plt.xlabel('Number of NI Channels', fontsize=24)
      plt.ylabel('Norm. Model Size [%]', fontsize=24)


      plt.bar(x - 1.5*bar_width, dropout_model_size, width=bar_width, label='ChDr')
      plt.bar(x - 0.5*bar_width, layers_dropout_model_size, width=bar_width, label='La+ChDr')
      plt.bar(x + 0.5*bar_width, tech_model_size, width=bar_width, label='La+ChDr+Tech')
      plt.bar(x + 1.5*bar_width, density_model_size, width=bar_width, label='La+ChDr+Tech+Dense')

      for i in range(len(categories)):
        channels = 2048 * 2**i

        if int(math.ceil(dropout_size[channels]/default_size[channels]*100)) != 100:
          plt.text(x[i] - 1.5*bar_width, dropout_model_size[i], str(int(math.ceil(dropout_size[channels]/default_size[channels]*100))), ha='center', va='bottom', weight='normal', fontsize=20) #dropout

          if int(math.ceil(layers_dropout_size[channels]/default_size[channels]*100)) != 100:
            if i==0:
              plt.text(x[i] - 0.5*bar_width, layers_dropout_model_size[i], str(int(math.ceil(layers_dropout_size[channels]/default_size[channels]*100))), ha='center', va='bottom', weight='normal', fontsize=20) #layer+dropout
            else:
              plt.text(x[i] - 0.5*bar_width, layers_dropout_model_size[i], str(int(math.ceil(layers_dropout_size[channels]/default_size[channels]*100))), ha='center', va='bottom', weight='normal', fontsize=20) #layer+dropout

          if int(math.ceil(tech_size[channels]/default_size[channels]*100)) != 100:
            plt.text(x[i] + 0.5*bar_width, tech_model_size[i], str(int(math.ceil(tech_size[channels]/default_size[channels]*100))), ha='center', va='bottom', weight='normal', fontsize=20) #tech

          if int(math.ceil(density_size[channels]/default_size[channels]*100)) != 100:
            if i==0:
              plt.text(x[i] + 1.5*bar_width, density_model_size[i], str(int(math.ceil(density_size[channels]/default_size[channels]*100))), ha='center', va='bottom', weight='normal', fontsize=20) #density
            else:
              plt.text(x[i] + 1.5*bar_width, density_model_size[i], str(int(math.ceil(density_size[channels]/default_size[channels]*100))), ha='center', va='bottom', weight='normal', fontsize=20) #density

      plt.xticks(x, categories, fontsize=22)
      plt.yticks(fontsize=22)
      plt.ylim(0, 110)

      plt.text(
        1.0, -0.15,                # X and Y coordinates (normalized to axes)
        f'SoC {name}',         # Text to display
        fontsize=30,               # Font size
        ha='right', va='top',      # Align horizontally to the right and vertically to the top
        transform=plt.gca().transAxes,  # Use axis coordinates (0 to 1)
        fontweight='bold'          # Make the text bold (optional)
      )

      plt.legend(fontsize=18,loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=4,columnspacing=0.5, frameon=False, handlelength=0.7, labelspacing=0.1)

      #fig.set_size_inches(20, 6)

      plt.tight_layout()

      #plt.savefig('comp_dropout2.pdf')
      plt.savefig(f"figures/{name}_{dnn_type}_all_opt_scale_model.pdf")

      ctr = ctr + 1


  #Plot only the normalized model size plots
  for num in plt.get_fignums():
    if num % 2 != 0:
        plt.close(num)

  plt.show()
