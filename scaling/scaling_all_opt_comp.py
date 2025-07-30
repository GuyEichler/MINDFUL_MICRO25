from framework.networks import *
import matplotlib.pyplot as plt
from shapely.geometry import LineString
import numpy as np
from scipy.optimize import minimize_scalar
from scipy.optimize import differential_evolution

def scaling_all_opt_comp(soc, dnn_arch, physical_specs, comm_specs, show=True, maximum_channels=9472, step=1024, minimum_channels=1024):

  physical = physical_specs.copy()
  comm = comm_specs.copy()

  max_channels = maximum_channels
  min_channels = minimum_channels
  step = step

  orig_channels = soc["active_channels"]
  max_orig_channels = soc["max_channels"]
  sensing_area = soc["sensing_area"]
  total_area = soc["total_area"]
  non_sensing_area = total_area - sensing_area
  power_consumption = soc["power_consumption"]
  max_comm_channels = soc["max_comm_channels"]


  #max_data_rate = 300 #Mb/sec
  data_type = soc["data_type"]
  network_time = soc["sampling_period"]
  #max_comm_channels = 300*10**6 / data_type * network_time * 10**-9
  power_budget_per_area = soc["power_density_budget"]

  non_sensing_area_per_channel = non_sensing_area / max_comm_channels
  sensing_area_per_channel = sensing_area / orig_channels #0.000768 #mm^2
  area_per_channel = total_area / orig_channels
  area_per_channel_density = non_sensing_area_per_channel + sensing_area_per_channel / 2

  #power_per_channel_recording = 0.01325 #mW
  power_per_channel_recording = power_consumption * sensing_area/total_area / orig_channels
  power_per_channel_comm = power_consumption * non_sensing_area/total_area / max_comm_channels
  #power_per_channel = 0.03789 #mW

  name = soc["Name"]
  print(f"{name}: Non sensing area per channel:", non_sensing_area_per_channel)
  print(f"{name}: Sensing area per channel:", sensing_area_per_channel)
  print(f"{name}: Max communication channels:", max_comm_channels)


  # total_area = area_per_channel * channels
  # total_power_budget = power_budget_per_area * total_area

  #Check what happens when number of channels grows
  channel_numbers = list(range(min_channels, max_channels, step))
  x_axis = list()
  total_power_budget_plot = list()
  total_power_budget_density_plot = list()
  sensing_power_consumption_plot = list()
  power_consumption_plot = list()
  power_consumption_default_plot = list()
  power_consumption_dropout_plot = list()
  power_consumption_layers_dropout_plot = list()
  power_consumption_tech_plot = list()
  power_consumption_density_plot = list()

  #tick_ratio = math.ceil(max_channels / 256 * 16 / step)
  #tick_ratio = 2

  default_size = [0] * len(range(max_channels))
  default_power_budget = [0] * len(range(max_channels))
  dropout_size = [0] * len(range(max_channels))
  dropout_power_budget = [0] * len(range(max_channels))
  layers_dropout_size = [0] * len(range(max_channels))
  layers_dropout_power_budget = [0] * len(range(max_channels))
  tech_size = [0] * len(range(max_channels))
  tech_power_budget = [0] * len(range(max_channels))
  density_size = [0] * len(range(max_channels))
  density_power_budget = [0] * len(range(max_channels))

  for channels in channel_numbers:
    print("Num channels:", channels)

    if channels <= orig_channels:
      #area_per_channel = non_sensing_area_per_channel + sensing_area_per_channel
      if channels == 0:
        total_power_budget_plot.append(0)
        total_power_budget_density_plot.append(0)
        sensing_power_consumption_plot.append(0)
        power_consumption_plot.append(0)
        power_consumption_default_plot.append(0)
        power_consumption_dropout_plot.append(0)
        power_consumption_layers_dropout_plot.append(0)
        power_consumption_tech_plot.append(0)
        power_consumption_density_plot.append(0)

        #Prepare x axis - channel number
        x_val = channels
        x_axis.append(x_val)
        continue

      total_area = area_per_channel * channels
      total_area_density = area_per_channel_density * channels
      total_power_budget = power_budget_per_area * total_area
      total_power_budget_density = power_budget_per_area * total_area_density

      sensing_power_consumption = power_per_channel_recording * channels
      # non_sensing_power_consumption = power_per_channel_comm * channels
      power_consumption_channels = (power_per_channel_recording + power_per_channel_comm) * channels

      # print("\nPOWER BUDGET:", new_power_budget, "mW")
      # print("TOTAL AREA:", total_area, "mm^2")
      # print("SENSING AREA:", total_sensing_area, "mm^2")
      # print("NON SENSING AREA:", total_non_sensing_area, "mm^2\n")

      total_power_budget_plot.append(total_power_budget)
      total_power_budget_density_plot.append(total_power_budget_density)
      sensing_power_consumption_plot.append(sensing_power_consumption)

      power_consumption_plot.append(power_consumption_channels)
      power_consumption_default_plot.append(power_consumption_channels)
      power_consumption_dropout_plot.append(power_consumption_channels)
      power_consumption_layers_dropout_plot.append(power_consumption_channels)
      power_consumption_tech_plot.append(power_consumption_channels)
      power_consumption_density_plot.append(power_consumption_channels)

    else: #computation scaling

      #parameters
      reduction = False #doesn't need to be set true can just set aggressive to true
      aggressive = True
      budget=True # sets power budget to the maximum
      check_data_rate = True
      network_input_dependency = 1

      #sensing power and area
      sensing_power_consumption = power_per_channel_recording * channels
      sensing_area = sensing_area_per_channel * channels
      sensing_area_const_density = sensing_area_per_channel * channels/2
      non_sensing_area_density = non_sensing_area #non_sensing_area_per_channel * channels

      #non sensing area
      #non_sensing_area - stays constant with 1024 channels
      #non_sensing_area = non_sensing_area_per_channel * channels

      #non-sensing power - computation
      #calculate the communication - no QAM
      physical["network_time"] = network_time #8KHz
      physical["data_type"] = data_type #+2.5
      physical["mac_power"] = 0.05 #45nm
      physical["mac_time"] = 2 #45nm
      physical["mac_area"] = 0.000783 * 1.5 #mm^2 , 45nm + an overhead for wiring
      communication_power_orig, data_rate_orig = calc_communication(orig_channels, 0, physical, comm, enable_qam=False)
      comm["max_data_rate"] = data_rate_orig / 10**6
      print("MAX DATA RATE:", comm["max_data_rate"])

      #for the model we can increase the network time
      physical["network_time"] = 1 * 10**6 / 2 #network_time * 4 #2KHz


      total_power_budget = power_budget_per_area * (sensing_area + non_sensing_area)
      total_power_budget_const_density = power_budget_per_area * (sensing_area_const_density + non_sensing_area_density)
      non_sensing_power_budget = total_power_budget - sensing_power_consumption
      non_sensing_power_budget_const_density = total_power_budget_const_density - sensing_power_consumption

      total_power_budget_plot.append(total_power_budget) #

      arch = dnn_arch.copy()

      original_channels = arch["input_channels"]
      ratio = (channels / original_channels)
      arch["input_channels"] = channels
      physical["num_channels"] = channels

      scale_network(arch, ratio, network_input_dependency)

      _, _, res_default, macs = \
        run_paper_network_wrapper(arch, physical, comm, soc, allow_reduction=reduction, aggressive=False, budget=budget, check_data_rate=check_data_rate)

      computation_area = physical["mac_area"] * macs

      layer_operations, layer_sequences, output_per_layer = calc_dnn(arch)
      total_accumulations = total_macs(layer_operations, layer_sequences)
      default_size[channels] = total_accumulations #store the default size if the model in MACs

      default_power_budget[channels] = total_power_budget


      #Channel Dropout
      found = False
      res_dropout = 0
      high = 1.0
      low = 0.0
      epsilon = 1e-4

      while found == False:
        mid = (high + low) / 2
        if high - low < epsilon: #finished
          found = True
          print(f"Dropout value for {name}:", low)
          network_input_dependency = low
        else:
          network_input_dependency = mid

        arch = dnn_arch.copy()

        original_channels = arch["input_channels"]
        ratio = (channels / original_channels)
        #dropout_channels = math.ceil(channels * network_input_dependency)
        #physical["num_channels"] = channels
        arch["input_channels"] = channels

        scale_network(arch, ratio, network_input_dependency)

        _, _, res_dropout, macs = \
          run_paper_network_wrapper(arch, physical, comm, soc, allow_reduction=reduction, aggressive=False, budget=budget, check_data_rate=check_data_rate)

        layer_operations, layer_sequences, output_per_layer = calc_dnn(arch)
        total_accumulations = total_macs(layer_operations, layer_sequences)
        dropout_size[channels] = total_accumulations #store the default size if the model in MACs

        computation_area = physical["mac_area"] * macs
        total_power_budget = power_budget_per_area * (sensing_area + non_sensing_area + computation_area)
        non_sensing_power_budget = total_power_budget - sensing_power_consumption

        dropout_power_budget[channels] = total_power_budget


        if res_dropout > non_sensing_power_budget: #failed - need lower
          high = mid
        else:
          low = mid

      #end while

      #Channel Layer+Dropout
      found = False
      res_layers_dropout = 0
      high = 1.0
      low = 0.0
      epsilon = 1e-4

      while found == False:
        mid = (high + low) / 2
        if high - low < epsilon: #finished
          found = True
          print(f"Dropout value for {name}:", low)
          network_input_dependency = low
        else:
          network_input_dependency = mid

        arch = dnn_arch.copy()

        original_channels = arch["input_channels"]
        ratio = (channels / original_channels)
        #dropout_channels = math.ceil(channels * network_input_dependency)
        #physical["num_channels"] = channels
        arch["input_channels"] = channels

        scale_network(arch, ratio, network_input_dependency)

        _, _, res_layers_dropout, macs = \
          run_paper_network_wrapper(arch, physical, comm, soc, allow_reduction=reduction, aggressive=True, budget=budget, check_data_rate=check_data_rate)

        layer_operations, layer_sequences, output_per_layer = calc_dnn(arch)
        total_accumulations = total_macs(layer_operations, layer_sequences)
        layers_dropout_size[channels] = total_accumulations #store the default size if the model in MACs

        computation_area = physical["mac_area"] * macs
        total_power_budget = power_budget_per_area * (sensing_area + non_sensing_area + computation_area)
        non_sensing_power_budget = total_power_budget - sensing_power_consumption

        layers_dropout_power_budget[channels] = total_power_budget

        if res_layers_dropout > non_sensing_power_budget: #failed - need lower
          high = mid
        else:
          low = mid


      #Channel Layer+Dropout+technology
      found = False
      res_tech = 0
      high = 1.0
      low = 0.0
      epsilon = 1e-4

      physical["mac_power"] = 0.026 #12nm
      physical["mac_time"] = 1 #12nm
      physical["mac_area"] = 0.000248 * 1.5 #mm^2 , 12nm + an overhead for wiring

      while found == False:
        mid = (high + low) / 2
        if high - low < epsilon: #finished
          found = True
          print(f"Dropout value for {name}:", low)
          network_input_dependency = low
        else:
          network_input_dependency = mid

        arch = dnn_arch.copy()

        original_channels = arch["input_channels"]
        ratio = (channels / original_channels)
        #dropout_channels = math.ceil(channels * network_input_dependency)
        #physical["num_channels"] = channels
        arch["input_channels"] = channels

        scale_network(arch, ratio, network_input_dependency)

        _, _, res_tech, macs = \
          run_paper_network_wrapper(arch, physical, comm, soc, allow_reduction=reduction, aggressive=True, budget=budget, check_data_rate=check_data_rate)

        layer_operations, layer_sequences, output_per_layer = calc_dnn(arch)
        total_accumulations = total_macs(layer_operations, layer_sequences)
        tech_size[channels] = total_accumulations #store the default size if the model in MACs

        computation_area = physical["mac_area"] * macs
        total_power_budget = power_budget_per_area * (sensing_area + non_sensing_area + computation_area)
        non_sensing_power_budget = total_power_budget - sensing_power_consumption

        tech_power_budget[channels] = total_power_budget

        if res_tech > non_sensing_power_budget: #failed - need lower
          high = mid
        else:
          low = mid



      #Channel Layer+Dropout+technology+density
      found = False
      res_density = 0
      high = 1.0
      low = 0.0
      epsilon = 1e-4

      physical["mac_power"] = 0.026 #12nm
      physical["mac_time"] = 1 #12nm

      while found == False:
        mid = (high + low) / 2
        if high - low < epsilon: #finished
          found = True
          print(f"Dropout value for {name}:", low)
          network_input_dependency = low
        else:
          network_input_dependency = mid

        arch = dnn_arch.copy()

        original_channels = arch["input_channels"]
        ratio = (channels / original_channels)
        #dropout_channels = math.ceil(channels * network_input_dependency)
        #physical["num_channels"] = channels
        arch["input_channels"] = channels

        scale_network(arch, ratio, network_input_dependency)

        _, _, res_density, macs = \
          run_paper_network_wrapper(arch, physical, comm, soc, allow_reduction=reduction, aggressive=True, budget=budget, check_data_rate=check_data_rate)

        layer_operations, layer_sequences, output_per_layer = calc_dnn(arch)
        total_accumulations = total_macs(layer_operations, layer_sequences)
        density_size[channels] = total_accumulations #store the default size if the model in MACs

        computation_area = physical["mac_area"] * macs
        total_power_budget = power_budget_per_area * (sensing_area_const_density + non_sensing_area + computation_area)
        non_sensing_power_budget = total_power_budget - sensing_power_consumption

        density_power_budget[channels] = total_power_budget

        if res_density > non_sensing_power_budget: #failed - need lower
          high = mid
        else:
          low = mid



      print("non sensing power budget dense is:",non_sensing_power_budget_const_density, "it was:", non_sensing_power_budget)
      print("non sensing power dense:", res_density)
      print("sensing power:", sensing_power_consumption)
      print("total power budget density:", total_power_budget_const_density)


      # non_sensing_power_consumption = res_layers
      non_sensing_power_consumption_default = res_default
      non_sensing_power_consumption_dropout = res_dropout
      non_sensing_power_consumption_layers_dropout = res_layers_dropout
      non_sensing_power_consumption_tech = res_tech
      non_sensing_power_consumption_density = res_density

      #power consumption
      # power_consumption_channels = sensing_power_consumption + non_sensing_power_consumption
      power_consumption_channels_default = sensing_power_consumption + non_sensing_power_consumption_default
      power_consumption_channels_dropout = sensing_power_consumption + non_sensing_power_consumption_dropout
      power_consumption_channels_layers_dropout = sensing_power_consumption + non_sensing_power_consumption_layers_dropout
      power_consumption_channels_tech = sensing_power_consumption + non_sensing_power_consumption_tech
      power_consumption_channels_density = sensing_power_consumption + non_sensing_power_consumption_density
      # if channels == orig_channels + step:
      #   power_consumption_plot.append(None)
      # else:
      #power_consumption_plot.append(power_consumption_channels)
      power_consumption_default_plot.append(power_consumption_channels_default)
      power_consumption_dropout_plot.append(power_consumption_channels_dropout)
      power_consumption_layers_dropout_plot.append(power_consumption_channels_layers_dropout)
      power_consumption_tech_plot.append(power_consumption_channels_tech)
      power_consumption_density_plot.append(power_consumption_channels_density)

      print("Final powers: default:", power_consumption_channels_default, "layers:", power_consumption_channels, "dropout:", power_consumption_channels_dropout)

    #Prepare x axis - channel number
    x_val = channels
    x_axis.append(x_val)

  dnn_type = dnn_arch["DNN"]

  if show == True:
    fig, ax = plt.subplots(figsize=(10, 5))

    mid_index = len(total_power_budget_plot) // 2
    label_x, label_y = x_axis[mid_index], total_power_budget_plot[mid_index]
    angle = np.arctan2(total_power_budget_plot[mid_index+1] - total_power_budget_plot[mid_index-1], x_axis[mid_index+1] - x_axis[mid_index-1]) * 35 #/ np.pi
    print(angle)
    angle_degrees = np.degrees(angle)
    print(angle_degrees)

    #ax.axvline(x=1024, color='b', linestyle=':', linewidth=1, label=None)
    #label_xx, label_yy = 2048, power_consumption_plot[mid_index * 2]
    #print(total_power_budget_plot[-1])

    # plt.plot(x_axis, total_power_budget_plot, linestyle='--', color='red', label=None)
    # plt.text(label_x, label_y, 'Power Budget', rotation=angle_degrees-360, ha='center', va='center', color='red', weight='bold', fontsize=12)
    #plt.text(label_xx, label_yy, 'Zeng et al.', rotation=90, ha='center', va='center', color='blue', fontsize=12)
    #plt.text(x_axis[int(max_channels/2 / step)] + 0.1, total_power_budget_plot[int(max_channels/2 / step)], 'Power Budget', color='red')

    # plt.plot(x_axis, sensing_power_consumption_plot, label='Sensing Power Consumption', color='blue')
    # ax.axvline(x=2230 ,ymin=0, ymax=0.8, color='gray', linestyle='--', linewidth=1, label=None)
    # plt.plot(x_axis, power_consumption_plot, label='Total Power Consumption', color='green')
    # plt.xticks(x_axis[::tick_ratio], fontsize=12)
    # plt.yticks(fontsize=12)
    plt.xlabel('Number of Active Channels', fontsize=22)
    plt.ylabel('Normalized Power', fontsize=22)
    #ax.grid(axis='y',zorder=0, linewidth=0.5)
    #plt.title('Number of Active Channels vs. Power')
    # plt.legend(fontsize=12,loc='upper left', bbox_to_anchor=(0, 1), ncol=1)


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
      #power_budget_density = total_power_budget_density_plot[x_axis.index(channels)]
      print("power budget:", power_budget)
      # print("power consumption", power_consumption_plot[x_axis.index(channels)])
      print("power budget dense:", power_budget_density)
      print("power consumption dense", power_consumption_density_plot[x_axis.index(channels)])
      default_power = default_power + [power_consumption_default_plot[x_axis.index(channels)]/power_budget]
      print(default_power)
      # layer_power = layer_power + [power_consumption_plot[x_axis.index(channels)]/power_budget]
      dropout_power = dropout_power + [power_consumption_dropout_plot[x_axis.index(channels)]/power_budget_dropout]
      print(dropout_power)
      layers_dropout_power = layers_dropout_power + [power_consumption_layers_dropout_plot[x_axis.index(channels)]/power_budget_layers_dropout]
      print(layers_dropout_power)
      tech_power = tech_power + [power_consumption_tech_plot[x_axis.index(channels)]/power_budget_tech]
      print(tech_power)
      density_power = density_power + [power_consumption_density_plot[x_axis.index(channels)]/power_budget_density]
      print(density_power)
      power_budget_list = power_budget_list + [1]

    #plt.bar(x - 2*bar_width, power_budget_list, width=bar_width, label='Power budget', color='red')
    #plt.bar(x - 2*bar_width, default_power, width=bar_width, label='Default')
    #plt.bar(x - bar_width, layer_power, width=bar_width, label='Layer opt.')
    plt.bar(x - 1.5*bar_width, dropout_power, width=bar_width, label='ChDr')
    plt.bar(x - 0.5*bar_width, layers_dropout_power, width=bar_width, label='La+ChDr')
    plt.bar(x + 0.5*bar_width, tech_power, width=bar_width, label='La+ChDr+Tech')
    plt.bar(x + 1.5*bar_width, density_power, width=bar_width, label='La+ChDr+Tech+Dense')

    for i in range(len(categories)):
      channels = 2048 * 2**i
      #plt.text(x[i] - 3 * bar_width, data1[i] + 1, str(data1[i]), ha='center', va='bottom')
      # plt.text(x[i] - 2*bar_width, default_power[i], str(int(default_size[channels]/default_size[channels]*100))+'%', ha='center', va='bottom', weight='normal', fontsize=11) #default
      # plt.text(x[i] - bar_width, layer_power[i], str(int(default_size[channels]/default_size[channels]*100))+'%', ha='center', va='bottom', weight='normal', fontsize=11) #layers
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
    #plt.legend(fontsize=12,loc='upper left', bbox_to_anchor=(0, 1), ncol=1)
    #plt.legend(fontsize=16,loc='center left', bbox_to_anchor=(1, 0.5), ncol=1)
    plt.legend(fontsize=16,loc='upper center', bbox_to_anchor=(0.5, 1.25), ncol=4,columnspacing=0.5)

    #plt.subplots_adjust(left=0.1, right=0.9, top=1.2, bottom=0.1)
    #fig.set_size_inches(20, 6)

    plt.tight_layout()

    #plt.savefig('comp_dropout1.pdf')
    plt.savefig(f"figures/{name}_{dnn_type}_all_opt_scale_power.pdf")

    plt.show()

    fig, ax = plt.subplots(figsize=(10, 5))
    bar_width = 0.2

    plt.xlabel('Number of Active Channels', fontsize=22)
    plt.ylabel('Normalized Model Size [%]', fontsize=22)
    #ax.grid(axis='y',zorder=0, linewidth=0.5)

    #plt.bar(x - 2*bar_width, power_budget_list, width=bar_width, label='Power budget', color='red')
    #plt.bar(x - 2*bar_width, default_power, width=bar_width, label='Default')
    #plt.bar(x - bar_width, layer_power, width=bar_width, label='Layer opt.')
    plt.bar(x - 1.5*bar_width, dropout_model_size, width=bar_width, label='ChDr')
    plt.bar(x - 0.5*bar_width, layers_dropout_model_size, width=bar_width, label='La+ChDr')
    plt.bar(x + 0.5*bar_width, tech_model_size, width=bar_width, label='La+ChDr+Tech')
    plt.bar(x + 1.5*bar_width, density_model_size, width=bar_width, label='La+ChDr+Tech+Dense')

    for i in range(len(categories)):
      channels = 2048 * 2**i
      plt.text(x[i] - 1.5*bar_width, dropout_model_size[i], str(int(math.ceil(dropout_size[channels]/default_size[channels]*100))), ha='center', va='bottom', weight='normal', fontsize=16) #dropout

      if i==0:
        plt.text(x[i] - 0.5*bar_width, layers_dropout_model_size[i], str(int(math.ceil(layers_dropout_size[channels]/default_size[channels]*100))), ha='center', va='bottom', weight='normal', fontsize=16) #layer+dropout
      else:
        plt.text(x[i] - 0.5*bar_width, layers_dropout_model_size[i], str(int(math.ceil(layers_dropout_size[channels]/default_size[channels]*100))), ha='center', va='bottom', weight='normal', fontsize=16) #layer+dropout

      plt.text(x[i] + 0.5*bar_width, tech_model_size[i], str(int(math.ceil(tech_size[channels]/default_size[channels]*100))), ha='center', va='bottom', weight='normal', fontsize=16) #tech

      if i==0:
        plt.text(x[i] + 1.5*bar_width, density_model_size[i], str(int(math.ceil(density_size[channels]/default_size[channels]*100))), ha='center', va='bottom', weight='normal', fontsize=16) #density
      else:
        plt.text(x[i] + 1.5*bar_width, density_model_size[i], str(int(math.ceil(density_size[channels]/default_size[channels]*100))), ha='center', va='bottom', weight='normal', fontsize=16) #density

    plt.xticks(x, categories, fontsize=18)
    plt.yticks(fontsize=18)
    plt.ylim(0, 110)
    #plt.legend(fontsize=12,loc='upper left', bbox_to_anchor=(0, 1), ncol=1)
    #plt.legend(fontsize=15,loc='center left', bbox_to_anchor=(1, 0.5), ncol=1)
    plt.legend(fontsize=16,loc='upper center', bbox_to_anchor=(0.5, 1.25), ncol=4,columnspacing=0.5)

    #fig.set_size_inches(20, 6)

    plt.tight_layout()

    #plt.savefig('comp_dropout2.pdf')
    plt.savefig(f"figures/{name}_{dnn_type}_all_opt_scale_model.pdf")

    plt.show()

  return total_power_budget_plot, default_size, default_power_budget, dropout_size, dropout_power_budget, layers_dropout_size, layers_dropout_power_budget, tech_size, tech_power_budget, density_size, density_power_budget, power_consumption_default_plot, power_consumption_dropout_plot, power_consumption_layers_dropout_plot, power_consumption_tech_plot, power_consumption_density_plot, x_axis
