# import math
# import sys
# from helper_functions import *

#from structs import *
from networks import *
import matplotlib.pyplot as plt
from shapely.geometry import LineString
import numpy as np
from scipy.optimize import minimize_scalar
from scipy.optimize import differential_evolution


def run_network(network, arch, physical, comm, allow_reduction=False, aggressive=False, budget=False, check_data_rate=False):

  no_recursive_res = results("not recursive")
  recursive_res = results("recursive")
  pipeline_res = results("pipeline")
  merge_pipeline_res = results("merged pipeline")
  no_compute_res = results("no computation")

  no_compute, no_pipeline = network(physical, comm, arch, allow_reduction)

  #print(no_compute)
  no_compute_res.communication_power = no_compute["communication_power"]
  no_compute_res.communication_data_rate = no_compute["data_rate"]
  no_compute_res.computation_power = 0
  no_compute_res.computation_time = 0
  no_comp_power = results_summary(no_compute_res, physical, comm)

  if no_compute != no_pipeline:
    recursive_res.communication_power = no_pipeline["recursion_communication_power"]
    recursive_res.communication_data_rate = no_pipeline["recursion_communication_data_rate"]
    recursive_res.computation_power = no_pipeline["recursion_compute_power"]
    recursive_res.computation_time = no_pipeline["recursion_compute_time"]
    rec_power = results_summary(recursive_res, physical, comm)

    no_recursive_res.communication_power = no_pipeline["no_recursion_communication_power"]
    no_recursive_res.communication_data_rate = no_pipeline["no_recursion_communication_data_rate"]
    no_recursive_res.computation_power = no_pipeline["no_recursion_compute_power"]
    no_recursive_res.computation_time = no_pipeline["no_recursion_compute_time"]
    no_rec_power = results_summary(no_recursive_res, physical, comm)
  else: #network did not fit on chip
    rec_power = None #no_comp_power
    no_rec_power = None #no_comp_power

  bound = no_comp_power if budget==False else physical["power_budget"]

  if aggressive == True:

    minimum_pg = 0
    minimum_no_pg = 0
    new_num_layers = 0
    if no_compute != no_pipeline:
      new_num_layers = len(no_pipeline["time_per_layer"])
      minimum_pg = new_num_layers
      minimum_no_pg = new_num_layers
      min_rec_power = rec_power
      min_no_rec_power = no_rec_power
    else:
      min_rec_power = no_comp_power
      min_no_rec_power = no_comp_power

    while new_num_layers > 0:
      new_num_layers = new_num_layers - 1
      no_compute_tmp, no_pipeline_check = network(physical, comm, arch, allow_reduction, new_num_layers)

      if no_compute_tmp != no_pipeline_check:
        recursive_res.communication_power = no_pipeline_check["recursion_communication_power"]
        recursive_res.communication_data_rate = no_pipeline_check["recursion_communication_data_rate"]
        recursive_res.computation_power = no_pipeline_check["recursion_compute_power"]
        recursive_res.computation_time = no_pipeline_check["recursion_compute_time"]
        rec_power_check = results_summary(recursive_res, physical, comm)
        if rec_power_check < min_rec_power:
          min_rec_power = rec_power_check
          minimum_pg = new_num_layers

        no_recursive_res.communication_power = no_pipeline_check["no_recursion_communication_power"]
        no_recursive_res.communication_data_rate = no_pipeline_check["no_recursion_communication_data_rate"]
        no_recursive_res.computation_power = no_pipeline_check["no_recursion_compute_power"]
        no_recursive_res.computation_time = no_pipeline_check["no_recursion_compute_time"]
        no_rec_power_check = results_summary(no_recursive_res, physical, comm)
        if no_rec_power_check < min_no_rec_power:
          min_no_rec_power = no_rec_power_check
          minimum_no_pg = new_num_layers

    #Store minimums
    if min_rec_power < bound and min_rec_power != no_comp_power:
      rec_power = min_rec_power
    else:
      rec_power = None #no_comp_power
    if min_no_rec_power < bound and min_no_rec_power != no_comp_power:
      no_rec_power = min_no_rec_power
    else:
      no_rec_power = None #no_comp_power

    print("\nMinimum power gating (recursive):", minimum_pg, rec_power ,"Minimum no pawer gating (not recursive):", minimum_no_pg, no_rec_power, "\n")

    # while bound < rec_power or bound < no_rec_power:
    #   #If final budget is not satisfied run again
    #   print("Aggressive power budget - must be less than bound")
    #   new_num_layers = len(no_pipeline["time_per_layer"]) - 1 #remove one layer manually
    #   #print("\nnew num layers:", new_num_layers)
    #   no_compute_tmp, no_pipeline = network(physical, comm, arch, allow_reduction, new_num_layers)

    #   if bound < rec_power: #check previous
    #     print("\nNeed aggressive reduction for no pipeline + power gating option")
    #     if no_compute_tmp != no_pipeline:
    #       recursive_res.communication_power = no_pipeline["recursion_communication_power"]
    #       recursive_res.communication_data_rate = no_pipeline["recursion_communication_data_rate"]
    #       recursive_res.computation_power = no_pipeline["recursion_compute_power"]
    #       recursive_res.computation_time = no_pipeline["recursion_compute_time"]
    #       rec_power = results_summary(recursive_res, physical, comm)
    #     else:
    #       rec_power = no_comp_power

    #   if bound < no_rec_power: #check previous
    #     print("\nNeed aggressive reduction for full network + no power gating option")
    #     if no_compute_tmp != no_pipeline:
    #       no_recursive_res.communication_power = no_pipeline["no_recursion_communication_power"]
    #       no_recursive_res.communication_data_rate = no_pipeline["no_recursion_communication_data_rate"]
    #       no_recursive_res.computation_power = no_pipeline["no_recursion_compute_power"]
    #       no_recursive_res.computation_time = no_pipeline["no_recursion_compute_time"]
    #       no_rec_power = results_summary(no_recursive_res, physical, comm)
    #     else:
    #       no_rec_power = no_comp_power

  return no_comp_power, rec_power, no_rec_power

def run_pipeline_network(network, arch, physical, comm, allow_reduction=False, aggressive=False, budget=False, check_data_rate=False):

  no_recursive_res = results("not recursive")
  recursive_res = results("recursive")
  pipeline_res = results("pipeline")
  merge_pipeline_res = results("merged pipeline")
  no_compute_res = results("no computation")

  no_compute, pipeline = network(physical, comm, arch, allow_reduction)

  #print(no_compute)
  no_compute_res.communication_power = no_compute["communication_power"]
  no_compute_res.communication_data_rate = no_compute["data_rate"]
  no_compute_res.computation_power = 0
  no_compute_res.computation_time = 0
  no_comp_power = results_summary(no_compute_res, physical, comm)

  if no_compute != pipeline:
    pipeline_res.communication_power = pipeline["communication_power"]
    pipeline_res.communication_data_rate = pipeline["communication_data_rate"]
    pipeline_res.computation_power = pipeline["full_compute_power"]
    pipeline_res.computation_time = pipeline["max_time_per_layer"]
    full_pipe_power = results_summary(pipeline_res, physical, comm)

    merge_pipeline_res.communication_power = pipeline["communication_power"]
    merge_pipeline_res.communication_data_rate = pipeline["communication_data_rate"]
    merge_pipeline_res.computation_power = pipeline["merge_compute_power"]
    merge_pipeline_res.computation_time = pipeline["max_time_per_stage"]
    merge_pipe_power = results_summary(merge_pipeline_res, physical, comm)
  else:
    full_pipe_power = None #no_comp_power
    merge_pipe_power = None #no_comp_power

  bound = no_comp_power if budget==False else physical["power_budget"]

  #print("\nBOUND IS:", bound, "\n")

  #no_compute_tmp = no_compute
  if aggressive == True:

    minimum_full = 0
    minimum_merge = 0
    new_num_layers = 0
    if no_compute != pipeline:
      new_num_layers = len(pipeline["time_per_layer"])
      minimum_full = new_num_layers
      minimum_merge = new_num_layers
      min_full_power = full_pipe_power
      min_merge_power = merge_pipe_power
    else:
      min_full_power = no_comp_power
      min_merge_power = no_comp_power

    while new_num_layers > 0:
      new_num_layers = new_num_layers - 1
      #print("\nAGGRESSIVE OPTIMIZATION LAYERS:", new_num_layers, "\n")
      no_compute_tmp, pipeline_check = network(physical, comm, arch, allow_reduction, new_num_layers)

      if no_compute_tmp != pipeline_check:
        if (check_data_rate == True and pipeline_check["communication_data_rate"] < comm["max_data_rate"]) or check_data_rate == False:
          pipeline_res.communication_power = pipeline_check["communication_power"]
          pipeline_res.communication_data_rate = pipeline_check["communication_data_rate"]
          pipeline_res.computation_power = pipeline_check["full_compute_power"]
          pipeline_res.computation_time = pipeline_check["max_time_per_layer"]
          full_power_check = results_summary(pipeline_res, physical, comm)
          if full_power_check < min_full_power:
            min_full_power = full_power_check
            minimum_full = new_num_layers

          merge_pipeline_res.communication_power = pipeline_check["communication_power"]
          merge_pipeline_res.communication_data_rate = pipeline_check["communication_data_rate"]
          merge_pipeline_res.computation_power = pipeline_check["merge_compute_power"]
          merge_pipeline_res.computation_time = pipeline_check["max_time_per_stage"]
          merge_power_check = results_summary(merge_pipeline_res, physical, comm)
          if merge_power_check < min_merge_power:
            min_merge_power = merge_power_check
            minimum_merge = new_num_layers

    #Store minimums
    if min_full_power < bound and min_full_power != no_comp_power:
      #print("\nAGGRESSIVE OPTIMIZATION: FULL POWER LOWER THAN BOUND")
      full_pipe_power = min_full_power
    else:
      full_pipe_power = None #no_comp_power
      #print("\nAGGRESSIVE OPTIMIZATION: FULL POWER HIGHER THAN BOUND", full_pipe_power)
    if min_merge_power < bound and min_merge_power != no_comp_power:
      #print("\nAGGRESSIVE OPTIMIZATION: MERGE POWER LOWER THAN BOUND")
      merge_pipe_power = min_merge_power
    else:
      merge_pipe_power = None #no_comp_power
      #print("\nAGGRESSIVE OPTIMIZATION: MERGE POWER HIGHER THAN BOUND")

    print("\nMinimum full:", minimum_full, full_pipe_power ,"Minimum merge:", minimum_merge, merge_pipe_power, "\n")
    # while bound < full_pipe_power or bound < merge_pipe_power:
    #   #If final budget is not satisfied run again
    #   print("\nAggressive power budget - must be less than communication bound chip")
    #   new_num_layers = len(pipeline["time_per_layer"]) - 1 #remove one layer manually
    #   no_compute_tmp, pipeline = network(physical, comm, arch, allow_reduction, new_num_layers)

    #   if bound < full_pipe_power:
    #     print("\nNeed aggressive reduction for full pipeline")
    #     if no_compute_tmp != pipeline:
    #       pipeline_res.communication_power = pipeline["communication_power"]
    #       pipeline_res.communication_data_rate = pipeline["communication_data_rate"]
    #       pipeline_res.computation_power = pipeline["full_compute_power"]
    #       pipeline_res.computation_time = pipeline["max_time_per_layer"]
    #       full_pipe_power = results_summary(pipeline_res, physical, comm)
    #     else:
    #       full_pipe_power = no_comp_power

    #   if bound < merge_pipe_power:
    #     print("\nNeed aggressive reduction for merge pipeline")
    #     if no_compute_tmp != pipeline:
    #       merge_pipeline_res.communication_power = pipeline["communication_power"]
    #       merge_pipeline_res.communication_data_rate = pipeline["communication_data_rate"]
    #       merge_pipeline_res.computation_power = pipeline["merge_compute_power"]
    #       merge_pipeline_res.computation_time = pipeline["max_time_per_layer"]
    #       merge_pipe_power = results_summary(merge_pipeline_res, physical, comm)
    #     else:
    #       merge_pipe_power = no_comp_power

  return no_comp_power, full_pipe_power, merge_pipe_power


def run_paper_network(network, arch, physical, comm, total_power_budget=np.inf, allow_reduction=False, aggressive=False, budget=False, check_data_rate=False):

  no_recursive_res = results("not recursive")
  recursive_res = results("recursive")
  pipeline_res = results("pipeline")
  merge_pipeline_res = results("merged pipeline")
  no_compute_res = results("no computation")

  no_compute, pipeline = network(physical, comm, arch, allow_reduction)

  #print(no_compute)
  no_compute_res.communication_power = no_compute["communication_power"]
  no_compute_res.communication_data_rate = no_compute["data_rate"]
  no_compute_res.computation_power = 0
  no_compute_res.computation_time = 0
  no_comp_power = results_summary(no_compute_res, physical, comm)

  if no_compute != pipeline: #should always enter
    pipeline_res.communication_power = pipeline["communication_power"]
    pipeline_res.communication_data_rate = pipeline["communication_data_rate"]
    pipeline_res.computation_power = pipeline["full_compute_power"]
    pipeline_res.computation_time = pipeline["max_time_per_layer"]
    full_pipe_power = results_summary(pipeline_res, physical, comm)

    merge_pipeline_res.communication_power = pipeline["communication_power"]
    merge_pipeline_res.communication_data_rate = pipeline["communication_data_rate"]
    merge_pipeline_res.computation_power = pipeline["merge_compute_power"]
    merge_pipeline_res.computation_time = pipeline["max_time_per_stage"]
    merge_pipe_power = results_summary(merge_pipeline_res, physical, comm)
  else: #should never enter this
    full_pipe_power = None #no_comp_power
    merge_pipe_power = None #no_comp_power

  bound = no_comp_power if budget==False else total_power_budget #physical["power_budget"]

  #print("\nBOUND IS:", bound, "\n")

  #no_compute_tmp = no_compute
  if aggressive == True:

    minimum_full = 0
    minimum_merge = 0
    new_num_layers = 0
    if no_compute != pipeline: #should always enter
      new_num_layers = len(pipeline["time_per_layer"])
      minimum_full = new_num_layers
      minimum_merge = new_num_layers
      min_full_power = full_pipe_power
      min_merge_power = merge_pipe_power
    else:
      min_full_power = no_comp_power
      min_merge_power = no_comp_power

    while new_num_layers > 0: # and min_merge_power > bound:
      new_num_layers = new_num_layers - 1
      print("\nAGGRESSIVE OPTIMIZATION LAYERS:", new_num_layers, "MINIMUM POWER:", min_merge_power, "BUDGET:", bound)
      no_compute_tmp, pipeline_check = network(physical, comm, arch, allow_reduction, new_num_layers)

      if no_compute_tmp != pipeline_check: #should always enter
        if (check_data_rate == True and pipeline_check["communication_data_rate"]/10**6 < comm["max_data_rate"]) or check_data_rate == False:
          pipeline_res.communication_power = pipeline_check["communication_power"]
          pipeline_res.communication_data_rate = pipeline_check["communication_data_rate"]
          pipeline_res.computation_power = pipeline_check["full_compute_power"]
          pipeline_res.computation_time = pipeline_check["max_time_per_layer"]
          full_power_check = results_summary(pipeline_res, physical, comm)
          if full_power_check < min_full_power:
            min_full_power = full_power_check
            minimum_full = new_num_layers

          merge_pipeline_res.communication_power = pipeline_check["communication_power"]
          merge_pipeline_res.communication_data_rate = pipeline_check["communication_data_rate"]
          merge_pipeline_res.computation_power = pipeline_check["merge_compute_power"]
          merge_pipeline_res.computation_time = pipeline_check["max_time_per_stage"]
          merge_power_check = results_summary(merge_pipeline_res, physical, comm)
          if merge_power_check < min_merge_power:
            print("\nAGGRESSIVE OPTIMIZATION NEW MINIMUM LAYERs:", new_num_layers)
            min_merge_power = merge_power_check
            minimum_merge = new_num_layers
            arch["reduced_layers"] = new_num_layers
        else:
          print("REDUCED NETWORK DOES NOT MEET NEEDS:", pipeline_check["communication_data_rate"]/10**6, comm["max_data_rate"], "LAYERS:", new_num_layers)

    #Store minimums
    #if min_full_power < bound and min_full_power != no_comp_power:
    if min_full_power != no_comp_power:
      #print("\nAGGRESSIVE OPTIMIZATION: FULL POWER LOWER THAN BOUND")
      full_pipe_power = min_full_power
    else:
      full_pipe_power = None #no_comp_power
      #print("\nAGGRESSIVE OPTIMIZATION: FULL POWER HIGHER THAN BOUND", full_pipe_power)
    #if min_merge_power < bound and min_merge_power != no_comp_power:
    if min_merge_power != no_comp_power:
      print("\nAGGRESSIVE OPTIMIZATION: MINIMUM LAYERS:", arch["reduced_layers"])
      merge_pipe_power = min_merge_power
    else:
      merge_pipe_power = None #no_comp_power
      #print("\nAGGRESSIVE OPTIMIZATION: MERGE POWER HIGHER THAN BOUND")

    print("\nMinimum full:", minimum_full, full_pipe_power ,"Minimum merge:", minimum_merge, merge_pipe_power, "\n")

  return no_comp_power, full_pipe_power, merge_pipe_power


def find_maximum_value(*args):#(original_function, *args):

  #result = minimize_scalar(objective_function, bounds=(0, 1), method='bounded', args=args)

  result = differential_evolution(objective_function, bounds=[(0,1)], args=args)

  print(args)
  # The result will contain the input value that maximizes the output of the original function
  print("RESULT IS:", result)
  print("RESULT IS:", result.x)
  return result.x

def objective_function(x, *args):
    """
    Objective function to minimize. It returns the negative of the output of your original function.
    """
    #print(args)
    other_args = args[1:]
    original_function = args[0]
    res = original_function(x, *other_args)
    print("RESULT from original function:", res)
    if res is None:
      return 0
    else:
      return -1*res
    #return res


def mlp_function(x, channels, physical, comm, reduction, aggressive, budget, check_data_rate):

  mlp_arch = mlp_architecture.copy()
  original_channels = mlp_arch["input_channels"]
  dropout_channels = math.ceil(channels * x)
  physical["num_channels"] = channels
  mlp_arch["input_channels"] = dropout_channels

  mlp_arch["input_size"] = dropout_channels * mlp_arch["timestamps"] #mlp
  ratio = dropout_channels / original_channels
  # if ratio < 10:
  update_hidden = [round(layer * ratio) for layer in mlp_arch["hidden_size"]]
  mlp_arch["hidden_size"] = update_hidden
  # else:
  #   int_ratio = int(round(ratio))
  #   ratio = ratio - int_ratio + 1
  #   orig_hidden = mlp_arch["hidden_size"][0]
  #   #enlarge hidden layers
  #   update_hidden = [round(layer * ratio) for layer in mlp_arch["hidden_size"]]
  #   #add hidden layers
  #   int_ratio = round(int_ratio/10)
  #   update_hidden = update_hidden + [round(orig_hidden * ratio)] * int_ratio
  #   mlp_arch["hidden_size"] = update_hidden

  print("\nMLP INPUT SIZE:", mlp_arch["input_size"])
  print("\nMLP HIDDEN SIZE:", mlp_arch["hidden_size"], "\n")

  _, _, mlp_res = \
    run_pipeline_network(mlp_pipe, mlp_arch, physical, comm, allow_reduction=reduction, aggressive=aggressive, budget=budget, check_data_rate=check_data_rate)

  return mlp_res



def mlp_channel_test():

  arch = mlp_architecture.copy()
  physical = physical_constraints.copy()
  comm = communication.copy()

  max_channels = 256
  min_channels = 16
  step = 16

  #Check what happens when number of channels grows
  channel_numbers = list(range(min_channels, max_channels, step))
  x_axis = list()
  no_comp = list()
  rec = list()
  no_rec = list()
  # pipe = list()
  # merge_pipe = list()

  constraint = [physical["power_budget"]] * len(channel_numbers)

  for channels in channel_numbers:
    print("Num channels:", channels)
    physical["num_channels"] = channels
    arch["input_size"] = channels * arch["timestamps"] #mlp
    #arch["input_channels"] = channels #dense

    no_compute_res, recursive_res, no_recursive_res = \
      run_network(mlp, arch, physical, comm)

    #Prepare x axis - channel number
    x_val = channels
    x_axis.append(x_val)

    #prepare results
    no_comp.append(no_compute_res)
    rec.append(recursive_res)
    no_rec.append(no_recursive_res)
    # pipe.append(pipeline_res)
    # merge_pipe.append(merge_pipeline_res)

  plt.figure()
  plt.plot(x_axis, no_comp, label='no computation power')
  #plt.plot(x_axis, rec, label='no pipeline with power gating')
  plt.plot(x_axis, no_rec, label='no pipeline')
  # plt.plot(x_axis, pipe, label='full pipeline')
  # plt.plot(x_axis, merge_pipe, label='merge pipeline')
  plt.xticks(x_axis)
  plt.xlabel('Number of channels')
  plt.ylabel('Total comm+comp power [mW]')
  plt.title('Number of Channels VS. Power')
  plt.legend()

  plt.figure()
  plt.plot(x_axis, no_comp, label='no computation power')
  plt.plot(x_axis, rec, label='no pipeline with power gating')
  #plt.plot(x_axis, no_rec, label='no pipeline')
  # plt.plot(x_axis, pipe, label='full pipeline')
  # plt.plot(x_axis, merge_pipe, label='merge pipeline')
  #plt.plot(x_axis, constraint, label='power budget')
  plt.xticks(x_axis)
  plt.xlabel('Number of channels')
  plt.ylabel('Total comm+comp power [mW]')
  plt.title('Number of Channels VS. Power')
  plt.legend()

  plt.figure()
  #plt.plot(x_axis, no_comp, label='no computation power')
  plt.plot(x_axis, rec, label='no pipeline with power gating')
  plt.plot(x_axis, no_rec, label='no pipeline')
  # plt.plot(x_axis, pipe, label='full pipeline')
  # plt.plot(x_axis, merge_pipe, label='merge pipeline')
  plt.xticks(x_axis)
  plt.xlabel('Number of channels')
  plt.ylabel('Total comm+comp power [mW]')
  plt.title('Number of Channels VS. Power')
  plt.legend()

  plt.show()


def mlp_pipeline_channel_test():

  arch = mlp_architecture.copy()
  physical = physical_constraints.copy()
  comm = communication.copy()

  max_channels = 256
  min_channels = 16
  step = 16

  #Check what happens when number of channels grows
  channel_numbers = list(range(min_channels, max_channels, step))
  x_axis = list()
  no_comp = list()
  #rec = list()
  #no_rec = list()
  full_pipe = list()
  merge_pipe = list()

  for channels in channel_numbers:
    print("Num channels:", channels)
    physical["num_channels"] = channels
    arch["input_size"] = channels * arch["timestamps"] #mlp
    #arch["input_channels"] = channels #dense

    no_compute_res, full_pipe_res, merge_pipe_res = \
      run_pipeline_network(mlp_pipe, arch, physical, comm)

    #Prepare x axis - channel number
    x_val = channels
    x_axis.append(x_val)

    #prepare results
    no_comp.append(no_compute_res)
    full_pipe.append(full_pipe_res)
    merge_pipe.append(merge_pipe_res)
    # pipe.append(pipeline_res)
    # merge_pipe.append(merge_pipeline_res)

  plt.figure()
  plt.plot(x_axis, no_comp, label='no computation power')
  #plt.plot(x_axis, rec, label='no pipeline with power gating')
  plt.plot(x_axis, merge_pipe, label='merge pipeline')
  # plt.plot(x_axis, pipe, label='full pipeline')
  # plt.plot(x_axis, merge_pipe, label='merge pipeline')
  plt.xticks(x_axis)
  plt.xlabel('Number of channels')
  plt.ylabel('Total comm+comp power [mW]')
  plt.title('Number of Channels VS. Power')
  plt.legend()

  plt.figure()
  plt.plot(x_axis, no_comp, label='no computation power')
  plt.plot(x_axis, full_pipe, label='full_pipeline')
  #plt.plot(x_axis, no_rec, label='no pipeline')
  # plt.plot(x_axis, pipe, label='full pipeline')
  # plt.plot(x_axis, merge_pipe, label='merge pipeline')
  plt.xticks(x_axis)
  plt.xlabel('Number of channels')
  plt.ylabel('Total comm+comp power [mW]')
  plt.title('Number of Channels VS. Power')
  plt.legend()

  plt.figure()
  #plt.plot(x_axis, no_comp, label='no computation power')
  plt.plot(x_axis, full_pipe, label='full pipeline')
  plt.plot(x_axis, merge_pipe, label='merge pipeline')
  # plt.plot(x_axis, pipe, label='full pipeline')
  # plt.plot(x_axis, merge_pipe, label='merge pipeline')
  plt.xticks(x_axis)
  plt.xlabel('Number of channels')
  plt.ylabel('Total comm+comp power [mW]')
  plt.title('Number of Channels VS. Power')
  plt.legend()

  plt.show()

def mlp_compare_all_channel_test():

  arch = mlp_architecture.copy()
  physical = physical_constraints.copy()
  comm = communication.copy()

  max_channels = 256
  min_channels = 16
  step = 16

  #Check what happens when number of channels grows
  channel_numbers = list(range(min_channels, max_channels, step))
  x_axis = list()
  no_comp = list()
  rec = list()
  no_rec = list()
  full_pipe = list()
  merge_pipe = list()

  for channels in channel_numbers:
    print("Num channels:", channels)
    physical["num_channels"] = channels
    arch["input_size"] = channels * arch["timestamps"] #mlp
    #arch["input_channels"] = channels #dense

    no_compute_res, full_pipe_res, merge_pipe_res = \
      run_pipeline_network(mlp_pipe, arch, physical, comm)

    _, recursive_res, no_recursive_res = \
      run_network(mlp, arch, physical, comm)

    #Prepare x axis - channel number
    x_val = channels
    x_axis.append(x_val)

    #prepare results
    no_comp.append(no_compute_res)
    full_pipe.append(full_pipe_res)
    merge_pipe.append(merge_pipe_res)
    rec.append(recursive_res)
    no_rec.append(no_recursive_res)

  plt.figure()
  plt.plot(x_axis, no_comp, label='no computation power')
  plt.plot(x_axis, full_pipe, label='full pipeline')
  plt.plot(x_axis, merge_pipe, label='merge pipeline')
  plt.plot(x_axis, no_rec, label='full network no pipeline')
  plt.plot(x_axis, rec, label='full network no pipeline + PG')
  plt.xticks(x_axis)
  plt.xlabel('Number of channels')
  plt.ylabel('Total comm+comp power [mW]')
  plt.title('Number of Channels VS. Power')
  plt.legend()


  plt.show()


def mlp_hidden_channel_test():

  # arch = mlp_architecture.copy()
  physical = physical_constraints.copy()
  comm = communication.copy()

  original_channels = physical["num_channels"]
  max_channels = 512
  min_channels = 16
  step = 16

  #Check what happens when number of channels grows
  channel_numbers = list(range(min_channels, max_channels, step))
  x_axis = list()
  no_comp = list()
  rec = list()
  no_rec = list()
  full_pipe = list()
  merge_pipe = list()

  for channels in channel_numbers:
    print("Num channels:", channels)
    arch = mlp_architecture.copy()
    physical["num_channels"] = channels
    arch["input_size"] = channels * arch["timestamps"] #mlp
    ratio = channels / original_channels
    #arch["hidden_size"][:] = [round(layer * ratio) for layer in arch["hidden_size"]]
    update_hidden = [round(layer * ratio) for layer in arch["hidden_size"]]
    arch["hidden_size"] = update_hidden

    #arch["input_channels"] = channels #dense

    no_compute_res, full_pipe_res, merge_pipe_res = \
      run_pipeline_network(mlp_pipe, arch, physical, comm)

    _, recursive_res, no_recursive_res = \
      run_network(mlp, arch, physical, comm)

    #Prepare x axis - channel number
    x_val = channels
    x_axis.append(x_val)

    #prepare results
    no_comp.append(no_compute_res)
    full_pipe.append(full_pipe_res)
    merge_pipe.append(merge_pipe_res)
    rec.append(recursive_res)
    no_rec.append(no_recursive_res)

  np_no_comp = np.array(no_comp)
  np_full_pipe = np.array(full_pipe)
  np_merge_pipe = np.array(merge_pipe)
  np_rec = np.array(rec)
  np_no_rec = np.array(no_rec)

  plt.figure()
  plt.plot(x_axis, np_no_comp, label='no computation power')
  plt.plot(x_axis, np_full_pipe, label='full pipeline')
  plt.plot(x_axis, np_merge_pipe, label='merge pipeline')
  plt.plot(x_axis, np_no_rec, label='full network no pipeline')
  plt.plot(x_axis, np_rec, label='full network no pipeline + PG')

  no_comp_line = LineString(np.column_stack((x_axis, np_no_comp)))
  full_pipe_line = LineString(np.column_stack((x_axis, np_full_pipe)))
  merge_pipe_line = LineString(np.column_stack((x_axis, np_merge_pipe)))
  no_rec_line = LineString(np.column_stack((x_axis, np_no_rec)))
  rec_line = LineString(np.column_stack((x_axis, np_rec)))

  #intersection
  inter_merge_pipe = no_comp_line.intersection(merge_pipe_line)
  if inter_merge_pipe.geom_type == 'MultiPoint':
    plt.plot(*LineString(inter_merge_pipe.geoms).xy, 'o')
    x, y = LineString(inter_merge_pipe.geoms).xy
    print("Intersection at:\n", x,"\n", y)
  elif inter_merge_pipe.geom_type == 'Point':
    plt.plot(*inter_merge_pipe.xy, 'o')
    x, y = inter_merge_pipe.xy
    print("Intersection at:\n", x,"\n", y)



  plt.xticks(x_axis)
  plt.xlabel('Number of channels')
  plt.ylabel('Total comm+comp power [mW]')
  plt.title('Number of Channels VS. Power (Hidden Layer Scales)')
  plt.legend()


  plt.show()

def mlp_hidden_reduction_channel_test():

  # arch = mlp_architecture.copy()
  physical = physical_constraints.copy()
  comm = communication.copy()

  original_channels = physical["num_channels"]
  max_channels = 512
  min_channels = 16
  step = 16

  #Check what happens when number of channels grows
  channel_numbers = list(range(min_channels, max_channels, step))
  x_axis = list()
  no_comp = list()
  rec = list()
  no_rec = list()
  full_pipe = list()
  merge_pipe = list()

  for channels in channel_numbers:
    print("Num channels:", channels)
    arch = mlp_architecture.copy()
    physical["num_channels"] = channels
    arch["input_size"] = channels * arch["timestamps"] #mlp
    ratio = channels / original_channels
    #arch["hidden_size"][:] = [round(layer * ratio) for layer in arch["hidden_size"]]

    #arch["input_channels"] = channels #dense
    update_hidden = [round(layer * ratio) for layer in arch["hidden_size"]]
    arch["hidden_size"] = update_hidden

    no_compute_res, full_pipe_res, merge_pipe_res = \
      run_pipeline_network(mlp_pipe, arch, physical, comm, allow_reduction=True)

    _, recursive_res, no_recursive_res = \
      run_network(mlp, arch, physical, comm, allow_reduction=True)

    #Prepare x axis - channel number
    x_val = channels
    x_axis.append(x_val)

    #prepare results
    no_comp.append(no_compute_res)
    full_pipe.append(full_pipe_res)
    merge_pipe.append(merge_pipe_res)
    rec.append(recursive_res)
    no_rec.append(no_recursive_res)

  np_no_comp = np.array(no_comp)
  np_full_pipe = np.array(full_pipe)
  np_merge_pipe = np.array(merge_pipe)
  np_rec = np.array(rec)
  np_no_rec = np.array(no_rec)

  plt.figure()
  plt.plot(x_axis, np_no_comp, label='no computation power')
  plt.plot(x_axis, np_full_pipe, label='full pipeline')
  plt.plot(x_axis, np_merge_pipe, label='merge pipeline')
  plt.plot(x_axis, np_no_rec, label='full network no pipeline')
  plt.plot(x_axis, np_rec, label='full network no pipeline + PG')

  no_comp_line = LineString(np.column_stack((x_axis, np_no_comp)))
  full_pipe_line = LineString(np.column_stack((x_axis, np_full_pipe)))
  merge_pipe_line = LineString(np.column_stack((x_axis, np_merge_pipe)))
  no_rec_line = LineString(np.column_stack((x_axis, np_no_rec)))
  rec_line = LineString(np.column_stack((x_axis, np_rec)))

  #intersection
  inter_merge_pipe = no_comp_line.intersection(merge_pipe_line)
  if inter_merge_pipe.geom_type == 'MultiPoint':
    plt.plot(*LineString(inter_merge_pipe.geoms).xy, 'o')
    x, y = LineString(inter_merge_pipe.geoms).xy
    print("Intersection at:\n", x,"\n", y)
  elif inter_merge_pipe.geom_type == 'Point':
    plt.plot(*inter_merge_pipe.xy, 'o')
    x, y = inter_merge_pipe.xy
    print("Intersection at:\n", x,"\n", y)


  plt.xticks(x_axis)
  #plt.yscale('log')
  plt.xlabel('Number of channels')
  plt.ylabel('Total comm+comp power [mW]')
  plt.title('Number of Channels VS. Power (Hidden Layer Scales)')
  plt.legend()


  plt.show()


def mlp_hidden_reduction_comm_channel_test():

  #arch = mlp_architecture.copy()
  physical = physical_constraints.copy()
  comm = communication.copy()

  original_channels = physical["num_channels"]
  max_channels = 512
  min_channels = 16
  step = 16

  #Check what happens when number of channels grows
  channel_numbers = list(range(min_channels, max_channels, step))
  x_axis = list()
  no_comp = list()
  rec = list()
  no_rec = list()
  full_pipe = list()
  merge_pipe = list()

  #Patch the first phase until first intersection to keep network size nad budget
  reduction = False
  budget = False
  first_intersection = False

  for channels in channel_numbers:
    print("\nNum channels:", channels)
    arch = mlp_architecture.copy()
    physical["num_channels"] = channels
    arch["input_size"] = channels * arch["timestamps"] #mlp
    ratio = channels / original_channels
    # arch["hidden_size"][:] = [round(layer * ratio) for layer in arch["hidden_size"]]
    update_hidden = [round(layer * ratio) for layer in arch["hidden_size"]]
    arch["hidden_size"] = update_hidden

    print(arch["hidden_size"])

    #arch["input_channels"] = channels #dense

    no_compute_res, full_pipe_res, merge_pipe_res = \
      run_pipeline_network(mlp_pipe, arch, physical, comm, allow_reduction=reduction)

    _, recursive_res, no_recursive_res = \
      run_network(mlp, arch, physical, comm, allow_reduction=reduction)

    #Prepare x axis - channel number
    x_val = channels
    x_axis.append(x_val)

    #prepare results
    no_comp.append(no_compute_res)
    full_pipe.append(full_pipe_res)
    merge_pipe.append(merge_pipe_res)
    rec.append(recursive_res)
    no_rec.append(no_recursive_res)

    #Patch
    if first_intersection == False:
      if no_compute_res > full_pipe_res and \
      no_compute_res > merge_pipe_res and \
      no_compute_res > recursive_res and \
      no_compute_res > no_recursive_res:
      #Now allow reduction and budgeting
        reduction = True
        budget = True
        first_intersection = True

  np_no_comp = np.array(no_comp)
  np_full_pipe = np.array(full_pipe)
  np_merge_pipe = np.array(merge_pipe)
  np_rec = np.array(rec)
  np_no_rec = np.array(no_rec)

  plt.figure()
  plt.plot(x_axis, np_no_comp, label='no computation power')
  plt.plot(x_axis, np_full_pipe, label='full pipeline')
  plt.plot(x_axis, np_merge_pipe, label='merge pipeline')
  plt.plot(x_axis, np_no_rec, label='full network no pipeline')
  plt.plot(x_axis, np_rec, label='full network no pipeline + PG')

  no_comp_line = LineString(np.column_stack((x_axis, np_no_comp)))
  full_pipe_line = LineString(np.column_stack((x_axis, np_full_pipe)))
  merge_pipe_line = LineString(np.column_stack((x_axis, np_merge_pipe)))
  no_rec_line = LineString(np.column_stack((x_axis, np_no_rec)))
  rec_line = LineString(np.column_stack((x_axis, np_rec)))

  #intersection
  inter_merge_pipe = no_comp_line.intersection(merge_pipe_line)
  if inter_merge_pipe.geom_type == 'MultiPoint':
    plt.plot(*LineString(inter_merge_pipe.geoms).xy, 'o')
    x, y = LineString(inter_merge_pipe.geoms).xy
    print("Intersection at:\n", x,"\n", y)
  elif inter_merge_pipe.geom_type == 'Point':
    plt.plot(*inter_merge_pipe.xy, 'o')
    x, y = inter_merge_pipe.xy
    print("Intersection at:\n", x,"\n", y)



  plt.xticks(x_axis)
  #plt.yscale('log')
  plt.xlabel('Number of channels')
  plt.ylabel('Total comm+comp power [mW]')
  plt.title('Number of Channels VS. Power (Hidden Layer Scales)')
  plt.legend()


  plt.show()


def mlp_hidden_reduction_aggressive_channel_test():

  #arch = mlp_architecture.copy()
  physical = physical_constraints.copy()
  comm = communication.copy()

  original_channels = physical["num_channels"]
  max_channels = 4096
  min_channels = 16
  step = 16

  #Check what happens when number of channels grows
  channel_numbers = list(range(min_channels, max_channels, step))
  x_axis = list()
  no_comp = list()
  rec = list()
  no_rec = list()
  full_pipe = list()
  merge_pipe = list()

  #Patch the first phase until first intersection to keep network size nad budget
  # reduction = False
  # aggressive = True
  # first_intersection = False

  #parameters
  reduction = True
  aggressive = True
  budget=True
  check_data_rate = False
  network_input_dependency = 1

  for channels in channel_numbers:
    print("\nNum channels:", channels)
    arch = mlp_architecture.copy()
    physical["num_channels"] = channels
    physical["power_budget"] = 100
    arch["input_size"] = channels * arch["timestamps"] #mlp
    ratio = channels / original_channels * network_input_dependency
    update_hidden = [round(layer * ratio) for layer in arch["hidden_size"]]
    arch["hidden_size"] = update_hidden

    #arch["input_channels"] = channels #dense

    no_compute_res, full_pipe_res, merge_pipe_res = \
      run_pipeline_network(mlp_pipe, arch, physical, comm, allow_reduction=reduction, aggressive=aggressive, budget=budget, check_data_rate=check_data_rate)

    _, recursive_res, no_recursive_res = \
      run_network(mlp, arch, physical, comm, allow_reduction=reduction, aggressive=aggressive, budget=budget, check_data_rate=check_data_rate)

    #Prepare x axis - channel number
    x_val = channels
    x_axis.append(x_val)

    #prepare results
    no_comp.append(no_compute_res)
    full_pipe.append(full_pipe_res)
    merge_pipe.append(merge_pipe_res)
    rec.append(recursive_res)
    no_rec.append(no_recursive_res)

    # #Patch
    # if first_intersection == False:
    #   if no_compute_res > full_pipe_res and \
    #   no_compute_res > merge_pipe_res and \
    #   no_compute_res > recursive_res and \
    #   no_compute_res > no_recursive_res:
    #   #Now allow reduction and budgeting
    #     #reduction = True
    #     #aggressive = True
    #     first_intersection = True

  np_no_comp = np.array(no_comp)
  np_full_pipe = np.array(full_pipe)
  np_merge_pipe = np.array(merge_pipe)
  np_rec = np.array(rec)
  np_no_rec = np.array(no_rec)

  plt.figure()
  plt.plot(x_axis, np_no_comp, label='no computation power')
  plt.plot(x_axis, np_full_pipe, label='full pipeline')
  plt.plot(x_axis, np_merge_pipe, label='merge pipeline')
  plt.plot(x_axis, np_no_rec, label='full network no pipeline')
  plt.plot(x_axis, np_rec, label='full network no pipeline + PG')

  no_comp_line = LineString(np.column_stack((x_axis, np_no_comp)))
  full_pipe_line = LineString(np.column_stack((x_axis, np_full_pipe)))
  merge_pipe_line = LineString(np.column_stack((x_axis, np_merge_pipe)))
  no_rec_line = LineString(np.column_stack((x_axis, np_no_rec)))
  rec_line = LineString(np.column_stack((x_axis, np_rec)))

  #intersection
  inter_merge_pipe = no_comp_line.intersection(merge_pipe_line)
  if inter_merge_pipe.geom_type == 'MultiPoint':
    plt.plot(*LineString(inter_merge_pipe.geoms).xy, 'o')
    x, y = LineString(inter_merge_pipe.geoms).xy
    print("Intersection at:\n", x,"\n", y)
  elif inter_merge_pipe.geom_type == 'Point':
    plt.plot(*inter_merge_pipe.xy, 'o')
    x, y = inter_merge_pipe.xy
    print("Intersection at:\n", x,"\n", y)



  plt.xticks(x_axis)
  #plt.yscale('log')
  plt.xlabel('Number of channels')
  plt.ylabel('Total comm+comp power [mW]')
  plt.title('Number of Channels VS. Power (Hidden Layer Scales)')
  plt.legend()


  plt.show()


def densenet_compare_all_channel_test():

  physical = physical_constraints.copy()
  comm = communication.copy()

  max_channels = 512
  min_channels = 16
  step = 16

  #Check what happens when number of channels grows
  channel_numbers = list(range(min_channels, max_channels, step))
  x_axis = list()
  no_comp = list()
  rec = list()
  no_rec = list()
  full_pipe = list()
  merge_pipe = list()

  for channels in channel_numbers:
    print("Num channels:", channels)
    arch = densenet_architecture.copy()
    physical["num_channels"] = channels
    arch["input_channels"] = channels #dense

    no_compute_res, full_pipe_res, merge_pipe_res = \
      run_pipeline_network(densenet_pipe, arch, physical, comm)

    _, recursive_res, no_recursive_res = \
      run_network(densenet, arch, physical, comm)

    #Prepare x axis - channel number
    x_val = channels
    x_axis.append(x_val)

    #prepare results
    no_comp.append(no_compute_res)
    full_pipe.append(full_pipe_res)
    merge_pipe.append(merge_pipe_res)
    rec.append(recursive_res)
    no_rec.append(no_recursive_res)

  plt.figure()
  plt.plot(x_axis, no_comp, label='no computation power')
  plt.plot(x_axis, full_pipe, label='full pipeline')
  plt.plot(x_axis, merge_pipe, label='merge pipeline')
  plt.plot(x_axis, no_rec, label='full network no pipeline')
  plt.plot(x_axis, rec, label='full network no pipeline + PG')
  plt.xticks(x_axis)
  plt.xlabel('Number of channels')
  plt.ylabel('Total comm+comp power [mW]')
  plt.title('Number of Channels VS. Power')
  plt.legend()


  plt.show()

def densenet_pipeline_channel_test():

  physical = physical_constraints.copy()
  comm = communication.copy()

  max_channels = 2048
  min_channels = 16
  step = 16

  #Check what happens when number of channels grows
  channel_numbers = list(range(min_channels, max_channels, step))
  x_axis = list()
  no_comp = list()
  #rec = list()
  #no_rec = list()
  full_pipe = list()
  merge_pipe = list()

  for channels in channel_numbers:
    print("Num channels:", channels)
    arch = densenet_architecture.copy()
    physical["num_channels"] = channels
    arch["input_channels"] = channels #dense

    no_compute_res, full_pipe_res, merge_pipe_res = \
      run_pipeline_network(densenet_pipe, arch, physical, comm)

    #Prepare x axis - channel number
    x_val = channels
    x_axis.append(x_val)

    #prepare results
    no_comp.append(no_compute_res)
    full_pipe.append(full_pipe_res)
    merge_pipe.append(merge_pipe_res)
    # pipe.append(pipeline_res)
    # merge_pipe.append(merge_pipeline_res)

  plt.figure()
  plt.plot(x_axis, no_comp, label='no computation power')
  #plt.plot(x_axis, rec, label='no pipeline with power gating')
  plt.plot(x_axis, merge_pipe, label='merge pipeline')
  # plt.plot(x_axis, pipe, label='full pipeline')
  # plt.plot(x_axis, merge_pipe, label='merge pipeline')
  plt.xticks(x_axis)
  plt.xlabel('Number of channels')
  plt.ylabel('Total comm+comp power [mW]')
  plt.title('Number of Channels VS. Power')
  plt.legend()

  plt.figure()
  plt.plot(x_axis, no_comp, label='no computation power')
  plt.plot(x_axis, full_pipe, label='full_pipeline')
  #plt.plot(x_axis, no_rec, label='no pipeline')
  # plt.plot(x_axis, pipe, label='full pipeline')
  # plt.plot(x_axis, merge_pipe, label='merge pipeline')
  plt.xticks(x_axis)
  plt.xlabel('Number of channels')
  plt.ylabel('Total comm+comp power [mW]')
  plt.title('Number of Channels VS. Power')
  plt.legend()

  plt.figure()
  #plt.plot(x_axis, no_comp, label='no computation power')
  plt.plot(x_axis, full_pipe, label='full pipeline')
  plt.plot(x_axis, merge_pipe, label='merge pipeline')
  # plt.plot(x_axis, pipe, label='full pipeline')
  # plt.plot(x_axis, merge_pipe, label='merge pipeline')
  plt.xticks(x_axis)
  plt.xlabel('Number of channels')
  plt.ylabel('Total comm+comp power [mW]')
  plt.title('Number of Channels VS. Power')
  plt.legend()

  plt.show()

def densenet_channel_test():

  physical = physical_constraints.copy()
  comm = communication.copy()

  max_channels = 512
  min_channels = 16
  step = 16

  #Check what happens when number of channels grows
  channel_numbers = list(range(min_channels, max_channels, step))
  x_axis = list()
  no_comp = list()
  rec = list()
  no_rec = list()
  # pipe = list()
  # merge_pipe = list()

  for channels in channel_numbers:
    print("Num channels:", channels)
    arch = densenet_architecture.copy()
    physical["num_channels"] = channels
    arch["input_channels"] = channels #dense

    no_compute_res, recursive_res, no_recursive_res = \
      run_network(densenet, arch, physical, comm)

    #Prepare x axis - channel number
    x_val = channels
    x_axis.append(x_val)

    #prepare results
    no_comp.append(no_compute_res)
    rec.append(recursive_res)
    no_rec.append(no_recursive_res)
    # pipe.append(pipeline_res)
    # merge_pipe.append(merge_pipeline_res)

  plt.figure()
  plt.plot(x_axis, no_comp, label='no computation power')
  #plt.plot(x_axis, rec, label='no pipeline with power gating')
  plt.plot(x_axis, no_rec, label='no pipeline')
  # plt.plot(x_axis, pipe, label='full pipeline')
  # plt.plot(x_axis, merge_pipe, label='merge pipeline')
  # intersection_points = [(xi, yi) for xi, yi in zip(x_axis, no_comp) if yi in no_rec]

  # # Plot intersection points
  # for xi, yi in intersection_points:
  #   plt.plot(xi, yi, 'ro')  # Plot red dots at intersection points

  plt.xticks(x_axis)
  plt.xlabel('Number of channels')
  plt.ylabel('Total comm+comp power [mW]')
  plt.title('Number of Channels VS. Power')
  plt.legend()



  plt.figure()
  plt.plot(x_axis, no_comp, label='no computation power')
  plt.plot(x_axis, rec, label='no pipeline with power gating')
  #plt.plot(x_axis, no_rec, label='no pipeline')
  # plt.plot(x_axis, pipe, label='full pipeline')
  # plt.plot(x_axis, merge_pipe, label='merge pipeline')
  plt.xticks(x_axis)
  plt.xlabel('Number of channels')
  plt.ylabel('Total comm+comp power [mW]')
  plt.title('Number of Channels VS. Power')
  plt.legend()

  # plt.figure()
  # #plt.plot(x_axis, no_comp, label='no computation power')
  # plt.plot(x_axis, rec, label='no pipeline with power gating')
  # plt.plot(x_axis, no_rec, label='no pipeline')
  # # plt.plot(x_axis, pipe, label='full pipeline')
  # # plt.plot(x_axis, merge_pipe, label='merge pipeline')
  # plt.xticks(x_axis)
  # plt.xlabel('Number of channels')
  # plt.ylabel('Total comm+comp power [mW]')
  # plt.title('Number of Channels VS. Power')
  # plt.legend()

  plt.show()

def densenet_scale_channel_test():

  physical = physical_constraints.copy()
  comm = communication.copy()

  max_channels = 4096
  min_channels = 16
  step = 16

  #Check what happens when number of channels grows
  channel_numbers = list(range(min_channels, max_channels, step))
  x_axis = list()
  no_comp = list()
  rec = list()
  no_rec = list()
  full_pipe = list()
  merge_pipe = list()

  for channels in channel_numbers:
    print("Num channels:", channels)
    arch = densenet_architecture.copy()
    original_channels = arch["input_channels"]
    physical["num_channels"] = channels
    arch["input_channels"] = channels #dense
    ratio = channels / original_channels
    arch["growth_factor"] = math.ceil(ratio * arch["growth_factor"])

    no_compute_res, full_pipe_res, merge_pipe_res = \
      run_pipeline_network(densenet_pipe, arch, physical, comm, aggressive=True, budget=True)

    _, recursive_res, no_recursive_res = \
      run_network(densenet, arch, physical, comm, aggressive=True, budget=True)

    #Prepare x axis - channel number
    x_val = channels
    x_axis.append(x_val)

    #prepare results
    no_comp.append(no_compute_res)
    full_pipe.append(full_pipe_res)
    merge_pipe.append(merge_pipe_res)
    rec.append(recursive_res)
    no_rec.append(no_recursive_res)

  plt.figure()
  plt.plot(x_axis, no_comp, label='no computation power')
  plt.plot(x_axis, full_pipe, label='full pipeline')
  plt.plot(x_axis, merge_pipe, label='merge pipeline')
  plt.plot(x_axis, no_rec, label='full network no pipeline')
  plt.plot(x_axis, rec, label='full network no pipeline + PG')
  plt.xticks(x_axis)
  plt.xlabel('Number of channels')
  plt.ylabel('Total comm+comp power [mW]')
  plt.title('Number of Channels VS. Power')
  plt.legend()


  plt.show()


def densenet_bisc_scale_channel_test():

  physical = physical_constraints.copy()
  comm = communication.copy()

  max_channels = 4096
  min_channels = 48
  step = 16

  #power_per_channel_recording = 0.01325 #mW
  #power_per_channel = 0.03789 #mW

  #BISC parameters
  orig_channels = 1024
  max_orig_channels = 65536
  max_data_rate = 300 #Mb/sec
  data_type = 12.5
  network_time = 0.5*10**6 / 4 #8KHz sampling
  max_comm_channels = 300*10**6 / data_type * network_time * 10**-9
  power_budget_per_area = 0.4 #mW/mm^2 - from literature
  sensing_area = 6.8 * 7.4 #mm^2 - BISC
  total_area = 144 #mm^2 - BISC
  non_sensing_area = total_area - sensing_area
  non_sensing_area_per_channel = non_sensing_area / max_comm_channels
  sensing_area_per_channel = sensing_area / max_orig_channels #0.000768 #mm^2

  print("Non sensing area per channel:", non_sensing_area_per_channel)
  print("Sensing area per channel:", sensing_area_per_channel)
  print("Max communication channels:", max_comm_channels)

  area_per_channel = non_sensing_area_per_channel + sensing_area_per_channel
  # total_area = area_per_channel * channels
  # total_power_budget = power_budget_per_area * total_area

  #Check what happens when number of channels grows
  channel_numbers = list(range(min_channels, max_channels, step))
  x_axis = list()
  no_comp = list()
  rec = list()
  no_rec = list()
  full_pipe = list()
  merge_pipe = list()
  power_budget = list()
  tick_ratio = math.ceil(max_channels / 128 * 16 / step)

  #parameters
  reduction = True
  aggressive = True
  budget=True
  check_data_rate = True
  network_input_dependency = 0.05

  for channels in channel_numbers:
    print("Num channels:", channels)
    arch = densenet_architecture.copy()
    original_channels = arch["input_channels"]
    physical["num_channels"] = channels
    arch["input_channels"] = channels #dense

    ratio = (channels / original_channels * network_input_dependency)
    arch["growth_factor"] = math.ceil(ratio * arch["growth_factor"])

    # new_power_budget = channels * (power_per_channel - power_per_channel_recording)
    total_area = area_per_channel * channels
    total_power_budget = power_budget_per_area * total_area
    new_power_budget = total_power_budget

    print("\nPOWER BUDGET:", new_power_budget, "\n")

    physical["power_budget"] = new_power_budget
    power_budget.append(new_power_budget)

    comm["energy_per_bit"] = 50 * 10**-12 #J
    comm["max_data_rate"] = max_data_rate #MHz
    physical["network_time"] = network_time
    physical["data_type"] = data_type

    no_compute_res, full_pipe_res, merge_pipe_res = \
      run_pipeline_network(densenet_pipe, arch, physical, comm, allow_reduction=reduction, aggressive=aggressive, budget=budget, check_data_rate=check_data_rate)

    _, recursive_res, no_recursive_res = \
      run_network(densenet, arch, physical, comm, allow_reduction=reduction, aggressive=aggressive, budget=budget, check_data_rate=check_data_rate)

    #Prepare x axis - channel number
    x_val = channels
    x_axis.append(x_val)

    #prepare results
    no_comp.append(no_compute_res)
    full_pipe.append(full_pipe_res)
    merge_pipe.append(merge_pipe_res)
    rec.append(recursive_res)
    no_rec.append(no_recursive_res)


  plt.figure()
  plt.plot(x_axis, no_comp, label='no computation power')
  plt.plot(x_axis, full_pipe, label='full pipeline')
  plt.plot(x_axis, merge_pipe, label='merge pipeline')
  plt.plot(x_axis, no_rec, label='full network no pipeline')
  plt.plot(x_axis, rec, label='full network no pipeline + PG')
  plt.plot(x_axis, power_budget, label='power budget')
  plt.xticks(x_axis[::tick_ratio])
  plt.xlabel('Number of channels')
  plt.ylabel('Total comm+comp power [mW]')
  plt.title('Number of Channels VS. Power')
  plt.legend()


  plt.show()


def mlp_bisc_scale_channel_test():

  physical = physical_constraints.copy()
  comm = communication.copy()

  max_channels = 1024
  min_channels = 16
  step = 16

  #power_per_channel_recording = 0.01325 #mW
  #power_per_channel = 0.03789 #mW

  #BISC parameters
  orig_channels = 1024
  max_orig_channels = 65536
  max_data_rate = 300 #Mb/sec
  data_type = 12.5
  network_time = 0.5*10**6 / 4 #8KHz sampling
  max_comm_channels = 300*10**6 / data_type * network_time * 10**-9
  power_budget_per_area = 0.4 #mW/mm^2 - from literature
  sensing_area = 6.8 * 7.4 #mm^2 - BISC
  total_area = 144 #mm^2 - BISC
  non_sensing_area = total_area - sensing_area
  non_sensing_area_per_channel = non_sensing_area / max_comm_channels
  sensing_area_per_channel = sensing_area / max_orig_channels #0.000768 #mm^2

  print("Non sensing area per channel:", non_sensing_area_per_channel)
  print("Sensing area per channel:", sensing_area_per_channel)
  print("Max communication channels:", max_comm_channels)

  area_per_channel = non_sensing_area_per_channel + sensing_area_per_channel
  # total_area = area_per_channel * channels
  # total_power_budget = power_budget_per_area * total_area

  #Check what happens when number of channels grows
  channel_numbers = list(range(min_channels, max_channels, step))
  x_axis = list()
  no_comp = list()
  rec = list()
  no_rec = list()
  full_pipe = list()
  merge_pipe = list()
  power_budget = list()
  tick_ratio = math.ceil(max_channels / 128 * 16 / step)

  #parameters
  reduction = True
  aggressive = True
  budget=True
  check_data_rate = False
  network_input_dependency = 1

  for channels in channel_numbers:
    print("Num channels:", channels)
    arch = mlp_architecture.copy()
    original_channels = arch["input_channels"]
    physical["num_channels"] = channels
    arch["input_channels"] = channels

    arch["input_size"] = channels * arch["timestamps"] #mlp
    ratio = channels / original_channels * network_input_dependency
    update_hidden = [round(layer * ratio) for layer in arch["hidden_size"]]
    arch["hidden_size"] = update_hidden

    # new_power_budget = channels * (power_per_channel - power_per_channel_recording)
    total_area = area_per_channel * channels
    total_power_budget = power_budget_per_area * total_area
    new_power_budget = total_power_budget
    total_sensing_area = sensing_area_per_channel * channels
    total_non_sensing_area = non_sensing_area_per_channel * channels

    print("\nPOWER BUDGET:", new_power_budget)
    print("TOTAL AREA:", total_area)
    print("SENSING AREA:", total_sensing_area)
    print("NON SENSING AREA:", total_non_sensing_area, "\n")

    physical["power_budget"] = new_power_budget
    power_budget.append(new_power_budget)

    comm["energy_per_bit"] = 50 * 10**-12 #J
    comm["max_data_rate"] = max_data_rate #MHz
    physical["network_time"] = network_time
    physical["data_type"] = data_type

    no_compute_res, full_pipe_res, merge_pipe_res = \
      run_pipeline_network(mlp_pipe, arch, physical, comm, allow_reduction=reduction, aggressive=aggressive, budget=budget, check_data_rate=check_data_rate)

    _, recursive_res, no_recursive_res = \
      run_network(mlp, arch, physical, comm, allow_reduction=reduction, aggressive=aggressive, budget=budget, check_data_rate=check_data_rate)

    #Prepare x axis - channel number
    x_val = channels
    x_axis.append(x_val)

    #prepare results
    no_comp.append(no_compute_res)
    full_pipe.append(full_pipe_res)
    merge_pipe.append(merge_pipe_res)
    rec.append(recursive_res)
    no_rec.append(no_recursive_res)


  plt.figure()
  plt.plot(x_axis, no_comp, label='no computation power')
  plt.plot(x_axis, full_pipe, label='full pipeline')
  plt.plot(x_axis, merge_pipe, label='merge pipeline')
  plt.plot(x_axis, no_rec, label='full network no pipeline')
  plt.plot(x_axis, rec, label='full network no pipeline + PG')
  plt.plot(x_axis, power_budget, label='power budget')
  plt.xticks(x_axis[::tick_ratio])
  plt.xlabel('Number of channels')
  plt.ylabel('Total comm+comp power [mW]')
  plt.title('Number of Channels VS. Power')
  plt.legend()


  plt.show()



def mlp_per_layer_test():

  #check what happens when redusing a specific model to less layers

  arch = mlp_architecture.copy()
  physical = physical_constraints.copy()
  comm = communication.copy()

  channels = 128
  # min_channels = 16
  # step = 16

  #Check what happens when number of channels grows
  #channel_numbers = list(range(min_channels, max_channels, step))
  x_axis = list()
  no_comp = list()
  rec = list()
  no_rec = list()
  # pipe = list()
  # merge_pipe = list()

  #physical constraints for the network
  mac_time = physical["mac_time"]
  mac_power = physical["mac_power"]
  network_time = physical["network_time"]
  min_macs = physical["min_macs"] #1
  num_channels = physical["num_channels"]

  #network architecture parameters
  arch["input_size"] = channels * arch["timestamps"] #mlp
  input_size = arch["input_size"] #2048
  hidden_size = arch["hidden_size"] #[256, 256]
  output_size = arch["output_size"] #40

  #get the model for the network
  layer_operations, layer_sequences, output_per_layer = \
    calc_mlp(input_size, hidden_size, output_size)
  total_accumulations = total_macs(layer_operations, layer_sequences)
  original_network_length = len(layer_operations)

  #Store results
  no_compute = no_compute_dict.copy()
  no_pipeline_per_layer = list()
  pipeline_per_layer = list()
  power_per_subset = list()
  pipeline_power_per_subset = list()
  recursive_res = results("recursive")
  merge_pipeline_res = results("merge pipeline")
  no_compute_res = results("no computation")

  print("\nCommunication power without any on-chip computation:")
  communication_power_no_compute, data_rate_no_compute = \
    calc_communication(num_channels, 0, physical, comm)

  no_compute["communication_power"] = communication_power_no_compute
  no_compute["data_rate"] = data_rate_no_compute

  no_compute_res.communication_power = communication_power_no_compute
  no_compute_res.communication_data_rate = data_rate_no_compute
  no_compute_res.computation_power = 0
  no_compute_res.computation_time = 0
  no_comp_power = results_summary(no_compute_res, physical, comm)

  for length in range(original_network_length+1):
    if length == 0:
      no_pipeline_per_layer.append(no_compute)
      power_per_subset.append(no_comp_power)
      pipeline_power_per_subset.append(no_comp_power)
    else:
      no_pipeline = \
      optimize_full_network_parameters(layer_operations[0:length], layer_sequences[0:length], mac_power, output_per_layer[0:length], physical, comm)
      pipeline = \
      optimize_pipelined_network_parameters(layer_operations[0:length], layer_sequences[0:length], mac_power, output_per_layer[0:length], physical, comm)

      no_pipeline_per_layer.append(no_pipeline)
      pipeline_per_layer.append(no_pipeline)

      recursive_res.communication_power = no_pipeline["recursion_communication_power"]
      recursive_res.communication_data_rate = no_pipeline["recursion_communication_data_rate"]
      recursive_res.computation_power = no_pipeline["recursion_compute_power"]
      recursive_res.computation_time = no_pipeline["recursion_compute_time"]
      no_pipeline_power = results_summary(recursive_res, physical, comm)
      power_per_subset.append(no_pipeline_power)

      merge_pipeline_res.communication_power = pipeline["communication_power"]
      merge_pipeline_res.communication_data_rate = pipeline["communication_data_rate"]
      merge_pipeline_res.computation_power = pipeline["merge_compute_power"]
      merge_pipeline_res.computation_time = pipeline["max_time_per_layer"]
      merge_pipe_power = results_summary(merge_pipeline_res, physical, comm)
      pipeline_power_per_subset.append(merge_pipe_power)

  #create plot
  categories = np.array([i for i in range(original_network_length+1)])
  width = 0.2

  print(no_pipeline_per_layer)
  print(categories)
  print(power_per_subset)
  print(pipeline_power_per_subset)

  plt.bar(categories, power_per_subset, width=width, label='no pipeline')
  plt.bar(categories + width, pipeline_power_per_subset, width=width, label='pipeline')

  plt.xlabel('Layers included')
  plt.ylabel('Power [mW]')
  plt.title('Power per subset of layers for MLP-speech')

  plt.legend()

  plt.show()


def s2s_bisc_scale_channel_test():

  physical = physical_constraints.copy()
  comm = communication.copy()

  max_channels = 4096
  min_channels = 16
  step = 16

  #power_per_channel_recording = 0.01325 #mW
  #power_per_channel = 0.03789 #mW

  #BISC parameters
  orig_channels = 1024
  max_orig_channels = 65536
  max_data_rate = 300 #Mb/sec
  data_type = 12.5
  network_time = 0.5*10**6 / 4 #8KHz sampling
  max_comm_channels = 300*10**6 / data_type * network_time * 10**-9
  power_budget_per_area = 0.4 #mW/mm^2 - from literature
  sensing_area = 6.8 * 7.4 #mm^2 - BISC
  total_area = 144 #mm^2 - BISC
  non_sensing_area = total_area - sensing_area
  non_sensing_area_per_channel = non_sensing_area / max_comm_channels
  sensing_area_per_channel = sensing_area / max_orig_channels #0.000768 #mm^2

  print("Non sensing area per channel:", non_sensing_area_per_channel)
  print("Sensing area per channel:", sensing_area_per_channel)
  print("Max communication channels:", max_comm_channels)

  area_per_channel = non_sensing_area_per_channel + sensing_area_per_channel
  # total_area = area_per_channel * channels
  # total_power_budget = power_budget_per_area * total_area

  #Check what happens when number of channels grows
  channel_numbers = list(range(min_channels, max_channels, step))
  x_axis = list()
  no_comp = list()
  rec = list()
  no_rec = list()
  full_pipe = list()
  merge_pipe = list()
  power_budget = list()
  tick_ratio = math.ceil(max_channels / 128 * 16 / step)

  #parameters
  reduction = True
  aggressive = True
  budget=True
  check_data_rate = False
  network_input_dependency = 1

  for channels in channel_numbers:
    print("Num channels:", channels)
    arch = s2s_architecture.copy()
    original_channels = arch["input_channels"]
    physical["num_channels"] = channels
    arch["input_channels"] = channels

    arch["input_size"] = channels
    ratio = channels / original_channels * network_input_dependency
    original_enc_layers = arch["encoder_layers"]
    arch["encoder_layers"] = math.ceil(original_enc_layers * ratio)

    # new_power_budget = channels * (power_per_channel - power_per_channel_recording)
    total_area = area_per_channel * channels
    total_power_budget = power_budget_per_area * total_area
    new_power_budget = total_power_budget

    print("\nPOWER BUDGET:", new_power_budget, "\n")

    physical["power_budget"] = new_power_budget
    power_budget.append(new_power_budget)

    comm["energy_per_bit"] = 50 * 10**-12 #J
    comm["max_data_rate"] = max_data_rate #MHz
    physical["network_time"] = network_time
    physical["data_type"] = data_type

    no_compute_res, full_pipe_res, merge_pipe_res = \
      run_pipeline_network(s2s_pipe, arch, physical, comm, allow_reduction=reduction, aggressive=aggressive, budget=budget, check_data_rate=check_data_rate)

    _, recursive_res, no_recursive_res = \
      run_network(s2s, arch, physical, comm, allow_reduction=reduction, aggressive=aggressive, budget=budget, check_data_rate=check_data_rate)

    #Prepare x axis - channel number
    x_val = channels
    x_axis.append(x_val)

    #prepare results
    no_comp.append(no_compute_res)
    full_pipe.append(full_pipe_res)
    merge_pipe.append(merge_pipe_res)
    rec.append(recursive_res)
    no_rec.append(no_recursive_res)


  plt.figure()
  plt.plot(x_axis, no_comp, label='no computation power')
  plt.plot(x_axis, full_pipe, label='full pipeline')
  plt.plot(x_axis, merge_pipe, label='merge pipeline')
  plt.plot(x_axis, no_rec, label='full network no pipeline')
  plt.plot(x_axis, rec, label='full network no pipeline + PG')
  plt.plot(x_axis, power_budget, label='power budget')
  plt.xticks(x_axis[::tick_ratio])
  plt.xlabel('Number of channels')
  plt.ylabel('Total comm+comp power [mW]')
  plt.title('Number of Channels VS. Power')
  plt.legend()


  plt.show()

def bisc_communication_test():

  physical = physical_constraints.copy()
  comm = communication.copy()

  max_channels = 4096*2
  min_channels = 16
  step = 16

  #power_per_channel_recording = 0.01325 #mW
  #power_per_channel = 0.03789 #mW

  #BISC parameters
  orig_channels = 1024
  max_orig_channels = 65536
  max_data_rate = 300 #Mb/sec
  data_type = 12.5
  network_time = 0.5*10**6 / 4 #8KHz sampling
  max_comm_channels = 300*10**6 / data_type * network_time * 10**-9
  power_budget_per_area = 0.4 #mW/mm^2 - from literature
  sensing_area = 6.8 * 7.4 #mm^2 - BISC
  total_area = 144 #mm^2 - BISC
  non_sensing_area = total_area - sensing_area
  non_sensing_area_per_channel = non_sensing_area / max_comm_channels
  sensing_area_per_channel = sensing_area / max_orig_channels #0.000768 #mm^2

  print("Non sensing area per channel:", non_sensing_area_per_channel)
  print("Sensing area per channel:", sensing_area_per_channel)
  print("Max communication channels:", max_comm_channels)

  area_per_channel = non_sensing_area_per_channel + sensing_area_per_channel
  # total_area = area_per_channel * channels
  # total_power_budget = power_budget_per_area * total_area

  #Check what happens when number of channels grows
  channel_numbers = list(range(min_channels, max_channels, step))
  x_axis = list()
  linear_comm = list()
  not_supported_linear_comm = list()
  qam_communication = list()
  power_budget = list()
  tick_ratio = math.ceil(max_channels / 128 * 16 / step)

  #parameters
  reduction = True
  aggressive = True
  budget=True
  check_data_rate = False
  network_input_dependency = 1

  for channels in channel_numbers:
    print("Num channels:", channels)

    # new_power_budget = channels * (power_per_channel - power_per_channel_recording)
    total_area = area_per_channel * channels
    total_power_budget = power_budget_per_area * total_area
    new_power_budget = total_power_budget
    total_sensing_area = sensing_area_per_channel * channels
    total_non_sensing_area = non_sensing_area_per_channel * channels

    print("\nPOWER BUDGET:", new_power_budget, "mW")
    print("TOTAL AREA:", total_area, "mm^2")
    print("SENSING AREA:", total_sensing_area, "mm^2")
    print("NON SENSING AREA:", total_non_sensing_area, "mm^2\n")

    physical["power_budget"] = new_power_budget
    power_budget.append(new_power_budget)

    comm["energy_per_bit"] = 50 * 10**-12 #J
    comm["max_data_rate"] = max_data_rate #300MHz
    physical["network_time"] = network_time
    physical["data_type"] = data_type

    #calculate the communication - no QAM
    communication_power, data_rate = calc_communication(channels, 0, physical, comm, enable_qam=False)
    if(data_rate/10**6 < max_data_rate):
      linear_comm.append(communication_power*10**3)
      not_supported_linear_comm.append(None)
      qam_communication.append(None)
    else:
      linear_comm.append(None)
      not_supported_linear_comm.append(communication_power*10**3)
      communication_power, data_rate = calc_communication(channels, 0, physical, comm, enable_qam=True)
      qam_communication.append(communication_power*10**3)

    #Prepare x axis - channel number
    x_val = channels
    x_axis.append(x_val)


  plt.figure()
  plt.plot(x_axis, linear_comm, label='Linear communication scaling')
  plt.plot(x_axis, not_supported_linear_comm, label='Unallowed linear scaling')
  plt.plot(x_axis, qam_communication, label='QAM communication scaling')
  plt.plot(x_axis, power_budget, label='power budget')
  plt.xticks(x_axis[::tick_ratio])
  plt.xlabel('Number of channels')
  plt.ylabel('Power [mW]')
  plt.title('Number of Channels VS. Power')
  plt.legend()


  plt.show()


def bisc_communication_and_computation_test():

  physical = physical_constraints.copy()
  comm = communication.copy()

  max_channels = 4096*2
  min_channels = 16
  step = 16

  #power_per_channel_recording = 0.01325 #mW
  #power_per_channel = 0.03789 #mW

  #BISC parameters
  orig_channels = 1024
  max_orig_channels = 65536
  max_data_rate = 300 #Mb/sec
  data_type = 12.5
  network_time = 0.5*10**6 / 4 #8KHz sampling
  max_comm_channels = 300*10**6 / data_type * network_time * 10**-9
  power_budget_per_area = 0.4 #mW/mm^2 - from literature
  sensing_area = 6.8 * 7.4 #mm^2 - BISC
  total_area = 144 #mm^2 - BISC
  non_sensing_area = total_area - sensing_area
  non_sensing_area_per_channel = non_sensing_area / max_comm_channels
  sensing_area_per_channel = sensing_area / max_orig_channels #0.000768 #mm^2

  print("Non sensing area per channel:", non_sensing_area_per_channel)
  print("Sensing area per channel:", sensing_area_per_channel)
  print("Max communication channels:", max_comm_channels)

  area_per_channel = non_sensing_area_per_channel + sensing_area_per_channel
  # total_area = area_per_channel * channels
  # total_power_budget = power_budget_per_area * total_area

  #Check what happens when number of channels grows
  channel_numbers = list(range(min_channels, max_channels, step))
  x_axis = list()
  linear_comm = list()
  not_supported_linear_comm = list()
  qam_communication = list()
  densenet_comp = list()
  mlp_comp = list()
  power_budget = list()
  tick_ratio = math.ceil(max_channels / 128 * 16 / step)

  #parameters
  reduction = True
  aggressive = True
  budget=True
  check_data_rate = True
  network_input_dependency_densenet = 0.04
  network_input_dependency_mlp = 0.2
  found = False

  for channels in channel_numbers:
    print("Num channels:", channels)

    # new_power_budget = channels * (power_per_channel - power_per_channel_recording)
    total_area = area_per_channel * channels
    total_power_budget = power_budget_per_area * total_area
    new_power_budget = total_power_budget
    total_sensing_area = sensing_area_per_channel * channels
    total_non_sensing_area = non_sensing_area_per_channel * channels

    print("\nPOWER BUDGET:", new_power_budget, "mW")
    print("TOTAL AREA:", total_area, "mm^2")
    print("SENSING AREA:", total_sensing_area, "mm^2")
    print("NON SENSING AREA:", total_non_sensing_area, "mm^2\n")

    physical["power_budget"] = new_power_budget
    power_budget.append(new_power_budget)

    comm["energy_per_bit"] = 50 * 10**-12 #J
    comm["max_data_rate"] = max_data_rate #300MHz
    physical["network_time"] = network_time
    physical["data_type"] = data_type

    #calculate the communication - no QAM
    communication_power, data_rate = calc_communication(channels, 0, physical, comm, enable_qam=False)
    if(data_rate/10**6 < max_data_rate):
      linear_comm.append(communication_power*10**3)
      not_supported_linear_comm.append(None)
      qam_communication.append(None)
      densenet_comp.append(None)
      mlp_comp.append(None)
    else:
      linear_comm.append(None)
      not_supported_linear_comm.append(communication_power*10**3)
      communication_power, data_rate = calc_communication(channels, 0, physical, comm, enable_qam=True)
      qam_communication.append(communication_power*10**3)

      #Also try pipelined computation - densenet

      dense_arch = densenet_architecture.copy()
      original_channels = dense_arch["input_channels"]
      physical["num_channels"] = channels
      dense_arch["input_channels"] = channels #dense

      ratio = (channels / original_channels * network_input_dependency_densenet)
      dense_arch["growth_factor"] = math.ceil(ratio * dense_arch["growth_factor"])

      print("\nDENSENET INPUT CHANNELS:", dense_arch["input_channels"], "\n")

      _, _, densenet_res = \
      run_pipeline_network(densenet_pipe, dense_arch, physical, comm, allow_reduction=reduction, aggressive=aggressive, budget=budget, check_data_rate=check_data_rate)

      densenet_comp.append(densenet_res)

      #MLP
      mlp_arch = mlp_architecture.copy()
      original_channels = mlp_arch["input_channels"]
      dropout_channels = channels * network_input_dependency_mlp
      physical["num_channels"] = dropout_channels
      mlp_arch["input_channels"] = dropout_channels

      mlp_arch["input_size"] = dropout_channels * mlp_arch["timestamps"] #mlp
      ratio = dropout_channels / original_channels
      update_hidden = [round(layer * ratio) for layer in mlp_arch["hidden_size"]]
      mlp_arch["hidden_size"] = update_hidden

      print("\nMLP INPUT SIZE:", mlp_arch["input_size"], "\n")

      _, _, mlp_res = \
      run_pipeline_network(mlp_pipe, mlp_arch, physical, comm, allow_reduction=reduction, aggressive=aggressive, budget=budget, check_data_rate=check_data_rate)

      mlp_comp.append(mlp_res)
      # if found == True and mlp_res == None:
      #   x_val = channels
      #   x_axis.append(x_val)
      #   break
      # if mlp_res != None:
      #   found = True

    #Prepare x axis - channel number
    x_val = channels
    x_axis.append(x_val)


  plt.figure()
  plt.plot(x_axis, linear_comm, label='Linear communication scaling')
  plt.plot(x_axis, not_supported_linear_comm, label='Unsupported linear scaling')
  plt.plot(x_axis, qam_communication, label='QAM communication scaling')
  plt.plot(x_axis, densenet_comp, label='DenseNet speech computation')
  plt.plot(x_axis, mlp_comp, label='MLP speech computation')
  plt.plot(x_axis, power_budget, label='power budget')
  plt.xticks(x_axis[::tick_ratio])
  plt.xlabel('Number of channels')
  plt.ylabel('Power [mW]')
  plt.title('Number of Channels VS. Power')
  plt.legend()


  plt.show()

def bisc_optimized_dropout_test():

  physical = physical_constraints.copy()
  comm = communication.copy()

  max_channels = 4096 * 2
  min_channels = 64
  step = 64

  #power_per_channel_recording = 0.01325 #mW
  #power_per_channel = 0.03789 #mW

  #BISC parameters
  orig_channels = 1024
  max_orig_channels = 65536
  max_data_rate = 300 #Mb/sec
  data_type = 12.5
  network_time = 0.5*10**6 / 4 #8KHz sampling
  max_comm_channels = 300*10**6 / data_type * network_time * 10**-9
  power_budget_per_area = 0.4 #mW/mm^2 - from literature
  sensing_area = 6.8 * 7.4 #mm^2 - BISC
  total_area = 144 #mm^2 - BISC
  non_sensing_area = total_area - sensing_area
  non_sensing_area_per_channel = non_sensing_area / max_comm_channels
  sensing_area_per_channel = sensing_area / max_orig_channels #0.000768 #mm^2

  print("Non sensing area per channel:", non_sensing_area_per_channel)
  print("Sensing area per channel:", sensing_area_per_channel)
  print("Max communication channels:", max_comm_channels)

  area_per_channel = non_sensing_area_per_channel + sensing_area_per_channel
  # total_area = area_per_channel * channels
  # total_power_budget = power_budget_per_area * total_area

  #Check what happens when number of channels grows
  channel_numbers = list(range(min_channels, max_channels, step))
  x_axis = list()
  linear_comm = list()
  not_supported_linear_comm = list()
  qam_communication = list()
  densenet_comp = list()
  mlp_comp = list()
  power_budget = list()
  tick_ratio = math.ceil(max_channels / 128 * 16 / step)

  #parameters
  reduction = True
  aggressive = True
  budget=True
  check_data_rate = True
  network_input_dependency_densenet = 0.04
  network_input_dependency_mlp = 0.2
  found = False

  for channels in channel_numbers:
    print("Num channels:", channels)

    # new_power_budget = channels * (power_per_channel - power_per_channel_recording)
    total_area = area_per_channel * channels
    total_power_budget = power_budget_per_area * total_area
    new_power_budget = total_power_budget
    total_sensing_area = sensing_area_per_channel * channels
    total_non_sensing_area = non_sensing_area_per_channel * channels

    print("\nPOWER BUDGET:", new_power_budget, "mW")
    print("TOTAL AREA:", total_area, "mm^2")
    print("SENSING AREA:", total_sensing_area, "mm^2")
    print("NON SENSING AREA:", total_non_sensing_area, "mm^2\n")

    physical["power_budget"] = new_power_budget
    power_budget.append(new_power_budget)

    comm["energy_per_bit"] = 50 * 10**-12 #J
    comm["max_data_rate"] = max_data_rate #300MHz
    physical["network_time"] = network_time
    physical["data_type"] = data_type

    #calculate the communication - no QAM
    communication_power, data_rate = calc_communication(channels, 0, physical, comm, enable_qam=False)
    if(data_rate/10**6 < max_data_rate):
      linear_comm.append(communication_power*10**3)
      not_supported_linear_comm.append(None)
      qam_communication.append(None)
      densenet_comp.append(None)
      mlp_comp.append(None)
    else:
      linear_comm.append(None)
      not_supported_linear_comm.append(communication_power*10**3)
      communication_power, data_rate = calc_communication(channels, 0, physical, comm, enable_qam=True)
      qam_communication.append(communication_power*10**3)

      #Also try pipelined computation - densenet

      found = False
      densenet_res = 0
      high = 1.0
      low = 0.0
      epsilon = 1e-4
      physical["power_budget"] = new_power_budget * 0.9
      while found == False:
        mid = (high + low) / 2
        if high - low < epsilon: #finished
          found = True
          print("Dropout value for DenseNet:", low)
          network_input_dependency_densenet = low
        else:
          network_input_dependency_densenet = mid

        dense_arch = densenet_architecture.copy()
        original_channels = dense_arch["input_channels"]
        physical["num_channels"] = channels
        dense_arch["input_channels"] = channels #dense

        ratio = (channels / original_channels * network_input_dependency_densenet)
        dense_arch["growth_factor"] = math.ceil(ratio * dense_arch["growth_factor"])

        print("\nDENSENET INPUT CHANNELS:", dense_arch["input_channels"], "\n")

        _, _, densenet_res = \
          run_pipeline_network(densenet_pipe, dense_arch, physical, comm, allow_reduction=reduction, aggressive=aggressive, budget=budget, check_data_rate=check_data_rate)

        if densenet_res == None: #failed - need lower
          high = mid
        else:
          low = mid

      densenet_comp.append(densenet_res)

      #MLP

      found = False
      mlp_res = 0
      high = 1.0
      low = 0.0
      epsilon = 1e-4
      physical["power_budget"] = new_power_budget * 0.8
      while found == False:
        mid = (high + low) / 2
        if high - low < epsilon: #finished
          found = True
          print("Dropout value for MLP:", low)
          network_input_dependency_mlp = low
        else:
          network_input_dependency_mlp = mid

        mlp_arch = mlp_architecture.copy()
        original_channels = mlp_arch["input_channels"]
        dropout_channels = math.ceil(channels * network_input_dependency_mlp)
        physical["num_channels"] = channels
        mlp_arch["input_channels"] = dropout_channels

        mlp_arch["input_size"] = dropout_channels * mlp_arch["timestamps"] #mlp
        ratio = dropout_channels / original_channels
        # if ratio < 10:
        update_hidden = [round(layer * ratio) for layer in mlp_arch["hidden_size"]]
        mlp_arch["hidden_size"] = update_hidden
        # else:
        #   int_ratio = int(round(ratio))
        #   ratio = ratio - int_ratio + 1
        #   orig_hidden = mlp_arch["hidden_size"][0]
        #   #enlarge hidden layers
        #   update_hidden = [round(layer * ratio) for layer in mlp_arch["hidden_size"]]
        #   #add hidden layers
        #   int_ratio = round(int_ratio/10)
        #   update_hidden = update_hidden + [round(orig_hidden * ratio)] * int_ratio
        #   mlp_arch["hidden_size"] = update_hidden

        print("\nMLP INPUT SIZE:", mlp_arch["input_size"])
        print("\nMLP HIDDEN SIZE:", mlp_arch["hidden_size"], "\n")

        _, _, mlp_res = \
          run_pipeline_network(mlp_pipe, mlp_arch, physical, comm, allow_reduction=reduction, aggressive=aggressive, budget=budget, check_data_rate=check_data_rate)

        # _, _, mlp_res = \
        #   run_network(mlp, mlp_arch, physical, comm, allow_reduction=reduction, aggressive=aggressive, budget=budget, check_data_rate=check_data_rate)

        if mlp_res == None: #failed - need lower
          high = mid
        else:
          low = mid

      mlp_comp.append(mlp_res)
      # if found == True and mlp_res == None:
      #   x_val = channels
      #   x_axis.append(x_val)
      #   break
      # if mlp_res != None:
      #   found = True

    #Prepare x axis - channel number
    x_val = channels
    x_axis.append(x_val)


  plt.figure()
  plt.plot(x_axis, linear_comm, label='Linear communication scaling')
  plt.plot(x_axis, not_supported_linear_comm, label='Unsupported linear scaling')
  plt.plot(x_axis, qam_communication, label='QAM communication scaling')
  plt.plot(x_axis, densenet_comp, label='DenseNet speech computation')
  plt.plot(x_axis, mlp_comp, label='MLP speech computation')
  plt.plot(x_axis, power_budget, label='power budget')
  plt.xticks(x_axis[::tick_ratio])
  plt.xlabel('Number of channels')
  plt.ylabel('Power [mW]')
  plt.title('Number of Channels VS. Power')
  plt.legend()


  plt.show()


def bisc_power_optimized_dropout_test():

  physical = physical_constraints.copy()
  comm = communication.copy()

  max_channels = 4096*2
  min_channels = 256
  step = 256

  #power_per_channel_recording = 0.01325 #mW
  #power_per_channel = 0.03789 #mW

  #BISC parameters
  orig_channels = 1024
  max_orig_channels = 65536
  max_data_rate = 300 #Mb/sec
  data_type = 12.5
  network_time = 0.5*10**6 / 4 #8KHz sampling
  max_comm_channels = 300*10**6 / data_type * network_time * 10**-9
  power_budget_per_area = 0.4 #mW/mm^2 - from literature
  sensing_area = 6.8 * 7.4 #mm^2 - BISC
  total_area = 144 #mm^2 - BISC
  non_sensing_area = total_area - sensing_area
  non_sensing_area_per_channel = non_sensing_area / max_comm_channels
  sensing_area_per_channel = sensing_area / max_orig_channels #0.000768 #mm^2

  print("Non sensing area per channel:", non_sensing_area_per_channel)
  print("Sensing area per channel:", sensing_area_per_channel)
  print("Max communication channels:", max_comm_channels)

  area_per_channel = non_sensing_area_per_channel + sensing_area_per_channel
  # total_area = area_per_channel * channels
  # total_power_budget = power_budget_per_area * total_area

  #Check what happens when number of channels grows
  channel_numbers = list(range(min_channels, max_channels, step))
  x_axis = list()
  linear_comm = list()
  not_supported_linear_comm = list()
  qam_communication = list()
  densenet_comp = list()
  mlp_comp = list()
  power_budget = list()
  real_power_budget = list()
  tick_ratio = math.ceil(max_channels / 128 * 16 / step)

  #parameters
  reduction = True
  aggressive = True
  budget=True
  check_data_rate = True
  network_input_dependency_densenet = 0.04
  network_input_dependency_mlp = 0.2
  found = False
  fake_channels = 20 #times

  for channels in channel_numbers:
    print("Num channels:", channels)

    # new_power_budget = channels * (power_per_channel - power_per_channel_recording)
    area_per_channel = non_sensing_area_per_channel + sensing_area_per_channel
    total_area = area_per_channel * channels
    total_power_budget = power_budget_per_area * total_area
    new_power_budget = total_power_budget
    total_sensing_area = sensing_area_per_channel * channels
    total_non_sensing_area = non_sensing_area_per_channel * channels

    print("\nPOWER BUDGET:", new_power_budget, "mW")
    print("TOTAL AREA:", total_area, "mm^2")
    print("SENSING AREA:", total_sensing_area, "mm^2")
    print("NON SENSING AREA:", total_non_sensing_area, "mm^2\n")

    physical["power_budget"] = new_power_budget
    power_budget.append(new_power_budget)
    # real_power_budget.append(None)

    comm["energy_per_bit"] = 50 * 10**-12 #J
    comm["max_data_rate"] = max_data_rate #300MHz
    physical["network_time"] = network_time
    physical["data_type"] = data_type

    #calculate the communication - no QAM
    communication_power, data_rate = calc_communication(channels, 0, physical, comm, enable_qam=False)
    if(data_rate/10**6 < max_data_rate):
      linear_comm.append(communication_power*10**3)
      not_supported_linear_comm.append(None)
      qam_communication.append(None)
      densenet_comp.append(None)
      mlp_comp.append(None)
      real_power_budget.append(None)
    else:
      linear_comm.append(None)
      not_supported_linear_comm.append(communication_power*10**3)
      communication_power, data_rate = calc_communication(channels, 0, physical, comm, enable_qam=True)
      qam_communication.append(communication_power*10**3)

      #Update power budget
      #non_sensing_area_per_channel = non_sensing_area / max_comm_channels
      #sensing_area_per_channel = sensing_area / max_orig_channels #0.000768 #mm^2
      #area_per_channel = non_sensing_area_per_channel + sensing_area_per_channel

      total_channels = channels + fake_channels*channels

      total_area = sensing_area_per_channel * total_channels + non_sensing_area_per_channel * max_comm_channels
      total_power_budget = power_budget_per_area * total_area
      new_power_budget = total_power_budget
      #total_sensing_area = sensing_area_per_channel * channels
      #total_non_sensing_area = non_sensing_area_per_channel * channels
      real_power_budget.append(new_power_budget)

      print("\nREAL POWER BUDGET:", new_power_budget, "mW\n")
      #power_budget.append(None)

      #Also try pipelined computation - densenet

      found = False
      densenet_res = 0
      high = 1.0
      low = 0.0
      epsilon = 1e-4
      physical["power_budget"] = new_power_budget * 0.9
      while found == False:
        mid = (high + low) / 2
        if high - low < epsilon: #finished
          found = True
          print("Dropout value for DenseNet:", low)
          network_input_dependency_densenet = low
        else:
          network_input_dependency_densenet = mid

        dense_arch = densenet_architecture.copy()
        original_channels = dense_arch["input_channels"]
        physical["num_channels"] = channels
        dense_arch["input_channels"] = channels #dense

        ratio = (channels / original_channels * network_input_dependency_densenet)
        dense_arch["growth_factor"] = math.ceil(ratio * dense_arch["growth_factor"])

        print("\nDENSENET INPUT CHANNELS:", dense_arch["input_channels"], "\n")

        _, _, densenet_res = \
          run_pipeline_network(densenet_pipe, dense_arch, physical, comm, allow_reduction=reduction, aggressive=aggressive, budget=budget, check_data_rate=check_data_rate)

        if densenet_res == None: #failed - need lower
          high = mid
        else:
          low = mid

      densenet_comp.append(densenet_res)

      #MLP

      found = False
      mlp_res = 0
      high = 1.0
      low = 0.0
      epsilon = 1e-4
      physical["power_budget"] = new_power_budget * 0.9

      # network_input_dependency_mlp = find_maximum_value(mlp_function, channels, physical, comm, reduction, aggressive, budget, check_data_rate)
      # mlp_res = mlp_function(network_input_dependency_mlp, channels, physical, comm, reduction, aggressive, budget, check_data_rate)

      while found == False:
        mid = (high + low) / 2
        if high - low < epsilon: #finished
          found = True
          print("Dropout value for MLP:", low)
          network_input_dependency_mlp = low
        else:
          network_input_dependency_mlp = mid

        mlp_arch = mlp_architecture.copy()
        original_channels = mlp_arch["input_channels"]
        dropout_channels = math.ceil(channels * network_input_dependency_mlp)
        physical["num_channels"] = channels
        mlp_arch["input_channels"] = dropout_channels

        mlp_arch["input_size"] = dropout_channels * mlp_arch["timestamps"] #mlp
        ratio = dropout_channels / original_channels
        # if ratio < 10:
        update_hidden = [round(layer * ratio) for layer in mlp_arch["hidden_size"]]
        mlp_arch["hidden_size"] = update_hidden
        # else:
        #   int_ratio = int(round(ratio))
        #   ratio = ratio - int_ratio + 1
        #   orig_hidden = mlp_arch["hidden_size"][0]
        #   #enlarge hidden layers
        #   update_hidden = [round(layer * ratio) for layer in mlp_arch["hidden_size"]]
        #   #add hidden layers
        #   int_ratio = round(int_ratio/10)
        #   update_hidden = update_hidden + [round(orig_hidden * ratio)] * int_ratio
        #   mlp_arch["hidden_size"] = update_hidden

        print("\nMLP INPUT SIZE:", mlp_arch["input_size"])
        print("\nMLP HIDDEN SIZE:", mlp_arch["hidden_size"], "\n")

        _, _, mlp_res = \
          run_pipeline_network(mlp_pipe, mlp_arch, physical, comm, allow_reduction=reduction, aggressive=aggressive, budget=budget, check_data_rate=check_data_rate)

        # _, _, mlp_res = \
        #   run_network(mlp, mlp_arch, physical, comm, allow_reduction=reduction, aggressive=aggressive, budget=budget, check_data_rate=check_data_rate)

        if mlp_res == None: #failed - need lower
          high = mid
        else:
          low = mid

      mlp_comp.append(mlp_res)
      # if found == True and mlp_res == None:
      #   x_val = channels
      #   x_axis.append(x_val)
      #   break
      # if mlp_res != None:
      #   found = True

    #Prepare x axis - channel number
    x_val = channels
    x_axis.append(x_val)


  plt.figure()
  plt.plot(x_axis, linear_comm, label='Linear communication scaling')
  plt.plot(x_axis, not_supported_linear_comm, label='Unsupported linear scaling')
  plt.plot(x_axis, qam_communication, label='QAM communication scaling')
  plt.plot(x_axis, densenet_comp, label='DenseNet speech computation')
  plt.plot(x_axis, mlp_comp, label='MLP speech computation')
  plt.plot(x_axis, power_budget, label='power budget')
  plt.plot(x_axis, real_power_budget, label='real power budget')
  plt.xticks(x_axis[::tick_ratio])
  plt.xlabel('Number of channels')
  plt.ylabel('Power [mW]')
  plt.title('Number of Channels VS. Power')
  plt.legend()


  plt.show()


def bisc_neural_iterface_scaling():

  physical_bisc = physical_constraints.copy()

  max_channels = 2048+64+512
  min_channels = 0
  step = 32


  #BISC parameters
  bisc_orig_channels = 1024
  bisc_max_orig_channels = 65536
  bisc_sensing_area = 6.8 * 7.4 #mm^2 - BISC
  bisc_total_area = 144 #mm^2 - BISC
  bisc_non_sensing_area = bisc_total_area - bisc_sensing_area
  bisc_power_consumption = 38.8 #mW
  bisc_max_comm_channels = bisc_orig_channels

  # #IBIS parameters
  # ibis_orig_channels = 192 * 245
  # ibis_max_orig_channels = 192 * 245
  # ibis_sensing_area = 5.76 * 7.68 #mm^2
  # ibis_total_area = 144 #mm^2 - BISC
  # ibis_non_sensing_area = ibis_total_area - ibis_sensing_area
  # ibis_power_consumption = 48 #mW
  # ibis_max_comm_channels = ibis_orig_channels

  #max_data_rate = 300 #Mb/sec
  data_type = 12.5
  network_time = 0.5*10**6 / 4 #8KHz sampling
  #max_comm_channels = 300*10**6 / data_type * network_time * 10**-9
  power_budget_per_area = 0.4 #mW/mm^2 - from literature

  bisc_non_sensing_area_per_channel = bisc_non_sensing_area / bisc_max_comm_channels
  bisc_sensing_area_per_channel = bisc_sensing_area / bisc_max_orig_channels #0.000768 #mm^2
  #bisc_area_per_channel = bisc_non_sensing_area_per_channel + bisc_sensing_area_per_channel
  bisc_area_per_channel = bisc_total_area / bisc_orig_channels

  #bisc_power_per_channel_recording = 0.01325 #mW
  bisc_power_per_channel_recording = bisc_power_consumption * bisc_sensing_area/bisc_total_area / bisc_orig_channels
  bisc_power_per_channel_comm = bisc_power_consumption * bisc_non_sensing_area/bisc_total_area / bisc_max_comm_channels
  #power_per_channel = 0.03789 #mW


  print("BISC: Non sensing area per channel:", bisc_non_sensing_area_per_channel)
  print("BISC: Sensing area per channel:", bisc_sensing_area_per_channel)
  print("BISC: Max communication channels:", bisc_max_comm_channels)


  # total_area = area_per_channel * channels
  # total_power_budget = power_budget_per_area * total_area

  #Check what happens when number of channels grows
  channel_numbers = list(range(min_channels, max_channels, step))
  x_axis = list()
  bisc_total_power_budget_plot = list()
  bisc_sensing_power_budget_plot = list()
  bisc_non_sensing_power_budget_plot = list()
  bisc_sensing_power_consumption_plot = list()
  bisc_non_sensing_power_consumption_plot = list()
  bisc_power_consumption_plot = list()

  #tick_ratio = math.ceil(max_channels / 256 * 16 / step)
  tick_ratio = 8

  for channels in channel_numbers:
    print("Num channels:", channels)

    #area_per_channel = non_sensing_area_per_channel + sensing_area_per_channel
    if channels == 0:
      bisc_total_power_budget_plot.append(0)
      bisc_sensing_power_budget_plot.append(0)
      bisc_non_sensing_power_budget_plot.append(0)
      bisc_sensing_power_consumption_plot.append(0)
      bisc_non_sensing_power_consumption_plot.append(0)
      bisc_power_consumption_plot.append(0)

      #Prepare x axis - channel number
      x_val = channels
      x_axis.append(x_val)
      continue

    bisc_total_area = bisc_area_per_channel * channels
    bisc_total_power_budget = power_budget_per_area * bisc_total_area
    bisc_total_sensing_area = bisc_sensing_area_per_channel * channels
    bisc_total_non_sensing_area = bisc_non_sensing_area_per_channel * channels

    bisc_sensing_power_budget = bisc_total_sensing_area/bisc_total_area * bisc_total_power_budget
    bisc_non_sensing_power_budget = bisc_total_non_sensing_area/bisc_total_area * bisc_total_power_budget

    #bisc_sensing_power_consumption = (bisc_total_sensing_area/bisc_total_area * bisc_power_consumption) / bisc_orig_channels * channels
    bisc_sensing_power_consumption = bisc_power_per_channel_recording * channels
    #bisc_power_per_channel_recording = bisc_power_consumption * bisc_sensing_area/bisc_total_area / bisc_orig_channels
    #bisc_non_sensing_power_consumption = (bisc_total_non_sensing_area/bisc_total_area * bisc_power_consumption) / bisc_orig_channels * channels
    bisc_non_sensing_power_consumption = bisc_power_per_channel_comm * channels
    #bisc_power_consumption_channels = bisc_power_consumption * channels / bisc_orig_channels
    bisc_power_consumption_channels = (bisc_power_per_channel_recording + bisc_power_per_channel_comm) * channels

    # print("\nPOWER BUDGET:", new_power_budget, "mW")
    # print("TOTAL AREA:", total_area, "mm^2")
    # print("SENSING AREA:", total_sensing_area, "mm^2")
    # print("NON SENSING AREA:", total_non_sensing_area, "mm^2\n")

    bisc_total_power_budget_plot.append(bisc_total_power_budget)
    bisc_sensing_power_budget_plot.append(bisc_sensing_power_budget)
    bisc_non_sensing_power_budget_plot.append(bisc_non_sensing_power_budget)
    bisc_sensing_power_consumption_plot.append(bisc_sensing_power_consumption)
    bisc_non_sensing_power_consumption_plot.append(bisc_non_sensing_power_consumption)
    bisc_power_consumption_plot.append(bisc_power_consumption_channels)

    #Prepare x axis - channel number
    x_val = channels
    x_axis.append(x_val)


  fig, ax = plt.subplots(figsize=(8, 6))

  mid_index = len(bisc_total_power_budget_plot) // 2
  label_x, label_y = x_axis[mid_index+3], bisc_total_power_budget_plot[mid_index+10]
  angle = np.arctan2(bisc_total_power_budget_plot[mid_index+1] - bisc_total_power_budget_plot[mid_index-1], x_axis[mid_index+1] - x_axis[mid_index-1]) * 8.7 #/ np.pi
  print(angle)
  angle_degrees = np.degrees(angle)
  print(angle_degrees)

  ax.axvline(x=1024, color='b', linestyle=':', linewidth=2, label="Data rate limit")
  label_xx, label_yy = 1024-40, bisc_total_power_budget_plot[mid_index * 2 - 10]

  plt.plot(x_axis, bisc_total_power_budget_plot, linestyle='--', linewidth=4, color='red', label=None)
  plt.text(label_x, label_y, 'Power Budget', rotation=angle_degrees-360, ha='center', va='center', color='red', weight='bold', fontsize=18)
  #plt.text(label_xx, label_yy, 'Zeng et al.', rotation=90, ha='center', va='center', color='blue', fontsize=12)
  #plt.text(x_axis[int(max_channels/2 / step)] + 0.1, bisc_total_power_budget_plot[int(max_channels/2 / step)], 'Power Budget', color='red')

  #plt.plot(x_axis, bisc_sensing_power_budget_plot, label='sensing power budget')
  plt.plot(x_axis, bisc_sensing_power_consumption_plot, linewidth=3, label='Sensing Power')
  plt.plot(x_axis, bisc_non_sensing_power_consumption_plot, linewidth=3, label='Non-Sensing Power')
  plt.plot(x_axis, bisc_power_consumption_plot, linewidth=3, label='Total Power Consumption')
  #plt.plot(x_axis, bisc_non_sensing_power_budget_plot, label='non sensing power budget')
  plt.xticks(x_axis[::tick_ratio], fontsize=18)
  plt.yticks(fontsize=18)
  plt.xlabel('Number of Active Channels', fontsize=22)
  plt.ylabel('Power [mW]', fontsize=22)
  plt.grid(axis='y')
  #plt.title('Number of Active Channels vs. Power')
  plt.legend(fontsize=15,loc='upper left')

  plt.savefig('neural_interface.pdf')

  plt.show()


def bisc_communication_neural_iterface_scaling():

  physical_bisc = physical_constraints.copy()
  comm_bisc = communication.copy()

  max_channels = (2048+64+512)
  min_channels = 0
  step = 4


  #BISC parameters
  bisc_orig_channels = 1024
  bisc_max_orig_channels = 65536
  bisc_sensing_area = 6.8 * 7.4 #mm^2 - BISC
  bisc_total_area = 144 #mm^2 - BISC
  bisc_non_sensing_area = bisc_total_area - bisc_sensing_area
  bisc_power_consumption = 38.8 #mW
  bisc_max_comm_channels = bisc_orig_channels

  # #IBIS parameters
  # ibis_orig_channels = 192 * 245
  # ibis_max_orig_channels = 192 * 245
  # ibis_sensing_area = 5.76 * 7.68 #mm^2
  # ibis_total_area = 144 #mm^2 - BISC
  # ibis_non_sensing_area = ibis_total_area - ibis_sensing_area
  # ibis_power_consumption = 48 #mW
  # ibis_max_comm_channels = ibis_orig_channels

  #max_data_rate = 300 #Mb/sec
  data_type = 10 #12.5
  network_time = 0.5*10**6 / 4 #8KHz sampling
  #max_comm_channels = 300*10**6 / data_type * network_time * 10**-9
  power_budget_per_area = 0.4 #mW/mm^2 - from literature

  bisc_non_sensing_area_per_channel = bisc_non_sensing_area / bisc_max_comm_channels
  bisc_sensing_area_per_channel = bisc_sensing_area / bisc_orig_channels #0.000768 #mm^2
  #bisc_area_per_channel = bisc_non_sensing_area_per_channel + bisc_sensing_area_per_channel
  bisc_area_per_channel = bisc_total_area / bisc_orig_channels

  #bisc_power_per_channel_recording = 0.01325 #mW
  bisc_power_per_channel_recording = bisc_power_consumption * bisc_sensing_area/bisc_total_area / bisc_orig_channels
  bisc_power_per_channel_comm = bisc_power_consumption * bisc_non_sensing_area/bisc_total_area / bisc_max_comm_channels
  #power_per_channel = 0.03789 #mW


  print("BISC: Non sensing area per channel:", bisc_non_sensing_area_per_channel)
  print("BISC: Sensing area per channel:", bisc_sensing_area_per_channel)
  print("BISC: Max communication channels:", bisc_max_comm_channels)


  # total_area = area_per_channel * channels
  # total_power_budget = power_budget_per_area * total_area

  #Check what happens when number of channels grows
  channel_numbers = list(range(min_channels, max_channels, step))
  x_axis = list()
  bisc_total_power_budget_plot = list()
  bisc_sensing_power_budget_plot = list()
  bisc_non_sensing_power_budget_plot = list()
  bisc_sensing_power_consumption_plot = list()
  bisc_non_sensing_power_consumption_plot = list()
  bisc_power_consumption_plot = list()

  #tick_ratio = math.ceil(max_channels / 256 * 16 / step)
  tick_ratio = 64

  for channels in channel_numbers:
    print("Num channels:", channels)

    if channels <= bisc_orig_channels:
      #area_per_channel = non_sensing_area_per_channel + sensing_area_per_channel
      if channels == 0:
        bisc_total_power_budget_plot.append(0)
        bisc_sensing_power_consumption_plot.append(0)
        bisc_non_sensing_power_consumption_plot.append(0)
        bisc_power_consumption_plot.append(0)

        #Prepare x axis - channel number
        x_val = channels
        x_axis.append(x_val)
        continue

      bisc_total_area = bisc_area_per_channel * channels
      bisc_total_power_budget = power_budget_per_area * bisc_total_area

      bisc_sensing_power_consumption = bisc_power_per_channel_recording * channels
      bisc_non_sensing_power_consumption = bisc_power_per_channel_comm * channels
      bisc_power_consumption_channels = (bisc_power_per_channel_recording + bisc_power_per_channel_comm) * channels

      # print("\nPOWER BUDGET:", new_power_budget, "mW")
      # print("TOTAL AREA:", total_area, "mm^2")
      # print("SENSING AREA:", total_sensing_area, "mm^2")
      # print("NON SENSING AREA:", total_non_sensing_area, "mm^2\n")

      bisc_total_power_budget_plot.append(bisc_total_power_budget)
      bisc_sensing_power_consumption_plot.append(bisc_sensing_power_consumption)
      bisc_non_sensing_power_consumption_plot.append(bisc_non_sensing_power_consumption)
      bisc_power_consumption_plot.append(bisc_power_consumption_channels)

    else: #communication scaling

      #sensing power and area
      bisc_sensing_power_consumption = bisc_power_per_channel_recording * channels
      bisc_sensing_area = bisc_sensing_area_per_channel * channels

      #non sensing area
      #bisc_non_sensing_area - stays constant with 1024 channels
      #bisc_non_sensing_area = bisc_non_sensing_area_per_channel * channels

      #non-sensing power - communication
      #calculate the communication - no QAM
      physical_bisc["network_time"] = network_time
      physical_bisc["data_type"] = data_type + 2.5
      communication_power_orig, data_rate_orig = calc_communication(bisc_orig_channels, 0, physical_bisc, comm_bisc, enable_qam=False)
      comm_bisc["max_data_rate"] = data_rate_orig / 10**6
      #communication_power, data_rate = calc_communication(channels, 0, physical_bisc, comm_bisc, enable_qam=False)
      communication_power_qam, data_rate_qam = calc_communication(channels, 0, physical_bisc, comm_bisc, enable_qam=True)
      # if(data_rate/10**6 < max_data_rate):
      #   linear_comm.append(communication_power*10**3)
      #   not_supported_linear_comm.append(None)
      #   qam_communication.append(None)
      # else:
      #   linear_comm.append(None)
      #   not_supported_linear_comm.append(communication_power*10**3)
      #   communication_power, data_rate = calc_communication(channels, 0, physical, comm, enable_qam=True)
      #   qam_communication.append(communication_power*10**3)
      bisc_non_sensing_power_consumption = bisc_power_per_channel_comm * bisc_orig_channels + communication_power_qam * 10**3
      #bisc_non_sensing_power_consumption = bisc_non_sensing_power_consumption + communication_power * 10**3

      #power budget
      bisc_total_power_budget = power_budget_per_area * (bisc_sensing_area + bisc_non_sensing_area)

      #power consumption
      bisc_power_consumption_channels = bisc_sensing_power_consumption + bisc_non_sensing_power_consumption

      bisc_total_power_budget_plot.append(bisc_total_power_budget) #
      bisc_sensing_power_consumption_plot.append(bisc_sensing_power_consumption) #
      bisc_non_sensing_power_consumption_plot.append(bisc_non_sensing_power_consumption)
      bisc_power_consumption_plot.append(bisc_power_consumption_channels)

    #Prepare x axis - channel number
    x_val = channels
    x_axis.append(x_val)


  fig, ax = plt.subplots(figsize=(12, 6))

  mid_index = len(bisc_total_power_budget_plot) // 2
  label_x, label_y = x_axis[mid_index+10], bisc_total_power_budget_plot[mid_index+70]
  angle = np.arctan2(bisc_total_power_budget_plot[mid_index+1] - bisc_total_power_budget_plot[mid_index-1], x_axis[mid_index+1] - x_axis[mid_index-1]) * 13 #/ np.pi
  print(angle)
  angle_degrees = np.degrees(angle)
  print(angle_degrees)

  ax.axvline(x=1024, color='b', linestyle=':', linewidth=2, label="Data rate limit")
  label_xx, label_yy = 1024-40, bisc_total_power_budget_plot[mid_index * 2 - 10]

  ax.axvline(x=2048,ymin=0, ymax=0.75, color='gray', linestyle='--', linewidth=2, label=None)

  plt.plot(x_axis, bisc_total_power_budget_plot, linestyle='--', linewidth=4, color='red', label=None)
  plt.text(label_x, label_y, 'Power Budget', rotation=angle_degrees-360, ha='center', va='center', color='red', weight='bold', fontsize=18)
  #plt.text(label_xx, label_yy, 'Zeng et al.', rotation=90, ha='center', va='center', color='blue', fontsize=12)
  #plt.text(x_axis[int(max_channels/2 / step)] + 0.1, bisc_total_power_budget_plot[int(max_channels/2 / step)], 'Power Budget', color='red')

  plt.plot(x_axis, bisc_sensing_power_consumption_plot, linewidth=3, label='Sensing Power')
  plt.plot(x_axis, bisc_non_sensing_power_consumption_plot, linewidth=3, label='Non-Sensing Power')
  plt.plot(x_axis, bisc_power_consumption_plot, linewidth=3, label='Total Power Consumption')
  plt.xticks(x_axis[::tick_ratio], fontsize=18)
  plt.yticks(fontsize=18)
  plt.xlabel('Number of Active Channels', fontsize=22)
  plt.ylabel('Power [mW]', fontsize=22)
  plt.grid(axis='y')
  #plt.title('Number of Active Channels vs. Power')
  plt.legend(fontsize=15,loc='upper left')

  plt.savefig('qam_comm.pdf')

  plt.show()


def bisc_computation_neural_iterface_scaling():

  physical_bisc = physical_constraints.copy()
  comm_bisc = communication.copy()

  max_channels = (2048+64+256)
  min_channels = 0
  step = 4


  #BISC parameters
  bisc_orig_channels = 1024
  bisc_max_orig_channels = 65536
  bisc_sensing_area = 6.8 * 7.4 #mm^2 - BISC
  bisc_total_area = 144 #mm^2 - BISC
  bisc_non_sensing_area = bisc_total_area - bisc_sensing_area
  bisc_power_consumption = 38.8 #mW
  bisc_max_comm_channels = bisc_orig_channels

  # #IBIS parameters
  # ibis_orig_channels = 192 * 245
  # ibis_max_orig_channels = 192 * 245
  # ibis_sensing_area = 5.76 * 7.68 #mm^2
  # ibis_total_area = 144 #mm^2 - BISC
  # ibis_non_sensing_area = ibis_total_area - ibis_sensing_area
  # ibis_power_consumption = 48 #mW
  # ibis_max_comm_channels = ibis_orig_channels

  #max_data_rate = 300 #Mb/sec
  data_type = 10 #12.5
  network_time = 0.5*10**6 / 4 #8KHz sampling
  #max_comm_channels = 300*10**6 / data_type * network_time * 10**-9
  power_budget_per_area = 0.4 #mW/mm^2 - from literature

  bisc_non_sensing_area_per_channel = bisc_non_sensing_area / bisc_max_comm_channels
  bisc_sensing_area_per_channel = bisc_sensing_area / bisc_orig_channels #0.000768 #mm^2
  #bisc_area_per_channel = bisc_non_sensing_area_per_channel + bisc_sensing_area_per_channel
  bisc_area_per_channel = bisc_total_area / bisc_orig_channels

  #bisc_power_per_channel_recording = 0.01325 #mW
  bisc_power_per_channel_recording = bisc_power_consumption * bisc_sensing_area/bisc_total_area / bisc_orig_channels
  bisc_power_per_channel_comm = bisc_power_consumption * bisc_non_sensing_area/bisc_total_area / bisc_max_comm_channels
  #power_per_channel = 0.03789 #mW


  print("BISC: Non sensing area per channel:", bisc_non_sensing_area_per_channel)
  print("BISC: Sensing area per channel:", bisc_sensing_area_per_channel)
  print("BISC: Max communication channels:", bisc_max_comm_channels)


  # total_area = area_per_channel * channels
  # total_power_budget = power_budget_per_area * total_area

  #Check what happens when number of channels grows
  channel_numbers = list(range(min_channels, max_channels, step))
  x_axis = list()
  bisc_total_power_budget_plot = list()
  bisc_sensing_power_budget_plot = list()
  bisc_non_sensing_power_budget_plot = list()
  bisc_sensing_power_consumption_plot = list()
  bisc_non_sensing_power_consumption_no_pipe_plot = list()
  bisc_non_sensing_power_consumption_pipe_plot = list()
  bisc_non_sensing_power_consumption_mlp_plot = list()
  bisc_non_sensing_power_consumption_comm_plot = list()
  bisc_power_consumption_plot = list()

  #tick_ratio = math.ceil(max_channels / 256 * 16 / step)
  tick_ratio = 64


  for channels in channel_numbers:
    print("Num channels:", channels)

    if channels <= bisc_orig_channels:
      #area_per_channel = non_sensing_area_per_channel + sensing_area_per_channel
      if channels == 0:
        bisc_total_power_budget_plot.append(0)
        bisc_sensing_power_consumption_plot.append(0)
        #bisc_non_sensing_power_consumption_no_pipe_plot.append(0)
        bisc_non_sensing_power_consumption_pipe_plot.append(None)
        bisc_non_sensing_power_consumption_mlp_plot.append(None)
        bisc_non_sensing_power_consumption_comm_plot.append(0)
        bisc_power_consumption_plot.append(0)

        #Prepare x axis - channel number
        x_val = channels
        x_axis.append(x_val)
        continue

      bisc_total_area = bisc_area_per_channel * channels
      bisc_total_power_budget = power_budget_per_area * bisc_total_area

      bisc_sensing_power_consumption = bisc_power_per_channel_recording * channels
      bisc_non_sensing_power_consumption = bisc_power_per_channel_comm * channels
      bisc_power_consumption_channels = (bisc_power_per_channel_recording + bisc_power_per_channel_comm) * channels

      # print("\nPOWER BUDGET:", new_power_budget, "mW")
      # print("TOTAL AREA:", total_area, "mm^2")
      # print("SENSING AREA:", total_sensing_area, "mm^2")
      # print("NON SENSING AREA:", total_non_sensing_area, "mm^2\n")

      bisc_total_power_budget_plot.append(bisc_total_power_budget)
      bisc_sensing_power_consumption_plot.append(bisc_sensing_power_consumption)
      #bisc_non_sensing_power_consumption_no_pipe_plot.append(bisc_non_sensing_power_consumption)
      bisc_non_sensing_power_consumption_pipe_plot.append(None) #no computation yet
      bisc_non_sensing_power_consumption_mlp_plot.append(None)
      bisc_non_sensing_power_consumption_comm_plot.append(bisc_non_sensing_power_consumption)
      bisc_power_consumption_plot.append(bisc_power_consumption_channels)

    else: #computation scaling

      #parameters
      reduction = False #doesn't need to be set true can just set aggressive to true
      aggressive = False
      budget=True # sets power budget to the maximum
      check_data_rate = True
      network_input_dependency = 1

      #sensing power and area
      bisc_sensing_power_consumption = bisc_power_per_channel_recording * channels
      bisc_sensing_area = bisc_sensing_area_per_channel * channels

      #non sensing area
      #bisc_non_sensing_area - stays constant with 1024 channels
      #bisc_non_sensing_area = bisc_non_sensing_area_per_channel * channels

      #non-sensing power - computation
      #calculate the communication - no QAM
      physical_bisc["network_time"] = network_time * 4 #2KHz
      physical_bisc["data_type"] = data_type #+2.5
      # physical_bisc["mac_power"] = 0.026 #12nm
      # physical_bisc["mac_time"] = 1 #12nm
      physical_bisc["mac_area"] = 0.000783 * 1.5 #45
      communication_power_orig, data_rate_orig = calc_communication(bisc_orig_channels, 0, physical_bisc, comm_bisc, enable_qam=False)
      comm_bisc["max_data_rate"] = data_rate_orig / 10**6
      print("MAX DATA RATE:", comm_bisc["max_data_rate"])
      #communication_power, data_rate = calc_communication(channels, 0, physical_bisc, comm_bisc, enable_qam=False)
      #communication_power_qam, data_rate_qam = calc_communication(channels, 0, physical_bisc, comm_bisc, enable_qam=True)

      computation_power = 0

      arch = densenet_architecture.copy()
      original_channels = arch["input_channels"]
      physical_bisc["num_channels"] = channels
      arch["input_channels"] = channels #dense

      # #power budget
      # bisc_total_power_budget = power_budget_per_area * (bisc_sensing_area + bisc_non_sensing_area)
      #physical_bisc["power_budget"] = bisc_total_power_budget - bisc_sensing_power_consumption #update power budget
      non_sensing_power_budget = np.inf #bisc_total_power_budget - bisc_sensing_power_consumption #update power budget

      ratio = (channels / original_channels)
      arch["growth_factor"] = math.ceil(ratio * network_input_dependency * arch["growth_factor"])

      no_compute_res, full_pipe_res, merge_pipe_res = \
        run_pipeline_network(paper_densenet_pipe, arch, physical_bisc, comm_bisc, allow_reduction=reduction, aggressive=aggressive, budget=budget, check_data_rate=check_data_rate)

      # _, recursive_res, no_recursive_res = \
      #   run_network(densenet, arch, physical_bisc, comm_bisc, allow_reduction=reduction, aggressive=aggressive, budget=budget, check_data_rate=check_data_rate)

      # if(data_rate/10**6 < max_data_rate):
      #   linear_comm.append(communication_power*10**3)
      #   not_supported_linear_comm.append(None)
      #   qam_communication.append(None)
      # else:
      #   linear_comm.append(None)
      #   not_supported_linear_comm.append(communication_power*10**3)
      #   communication_power, data_rate = calc_communication(channels, 0, physical, comm, enable_qam=True)
      #   qam_communication.append(communication_power*10**3)

      #bisc_non_sensing_power_consumption_no_pipe = bisc_power_per_channel_comm * bisc_orig_channels + recursive_res
      if merge_pipe_res != None and merge_pipe_res < non_sensing_power_budget:
        bisc_non_sensing_power_consumption_pipe = merge_pipe_res
        #bisc_non_sensing_power_consumption_pipe = bisc_power_per_channel_comm * bisc_orig_channels + merge_pipe_res
      else:
        break
        #bisc_non_sensing_power_consumption = bisc_non_sensing_power_consumption + communication_power * 10**3

      #power budget
      # bisc_comp_non_sensing_area = physical_bisc["mac_num"] * physical_bisc["mac_area"] #instead of keeping it the same as communication
      # bisc_comm_non_sensing_area = arch["output_channels"] / bisc_orig_channels * bisc_non_sensing_area #instead of keeping it the same as communication
      # bisc_total_power_budget = power_budget_per_area * (bisc_sensing_area + bisc_comm_non_sensing_area + bisc_comp_non_sensing_area)
      bisc_total_power_budget = power_budget_per_area * (bisc_sensing_area + bisc_non_sensing_area)

      #power consumption
      bisc_power_consumption_channels = bisc_sensing_power_consumption + bisc_non_sensing_power_consumption_pipe

      bisc_total_power_budget_plot.append(bisc_total_power_budget) #
      bisc_sensing_power_consumption_plot.append(bisc_sensing_power_consumption) #
      #bisc_non_sensing_power_consumption_no_pipe_plot.append(bisc_non_sensing_power_consumption_no_pipe)
      bisc_non_sensing_power_consumption_pipe_plot.append(bisc_non_sensing_power_consumption_pipe)
      bisc_non_sensing_power_consumption_comm_plot.append(None)
      #bisc_power_consumption_plot.append(bisc_power_consumption_channels)

      #MLP

      mlp_arch = mlp_architecture.copy()
      original_channels = mlp_arch["input_channels"]
      ratio = (channels / original_channels)
      #dropout_channels = math.ceil(channels * network_input_dependency_mlp)
      #physical["num_channels"] = channels
      mlp_arch["input_channels"] = channels
      mlp_arch["input_size"] = channels * mlp_arch["timestamps"] #mlp
      update_hidden = [mlp_arch["hidden_size"][0]] * len(mlp_arch["hidden_size"] * math.ceil(ratio)) #.copy()
      for i in range(len(update_hidden)):
        N = len(update_hidden)
        update_hidden[i] = round(update_hidden[i] * (1 + ((N - i - 1) / (N - 1)) * (ratio - 1)))
        print(update_hidden)
      #update_hidden = [round(layer * ratio) for layer in mlp_arch["hidden_size"]]
      mlp_arch["hidden_size"] = update_hidden

      _, _, mlp_res = \
        run_pipeline_network(paper_mlp_pipe, mlp_arch, physical_bisc, comm_bisc, allow_reduction=reduction, aggressive=aggressive, budget=budget, check_data_rate=check_data_rate)

      bisc_non_sensing_power_consumption_mlp_plot.append(mlp_res)

      #power consumption
      bisc_non_sensing_power_consumption_mlp = mlp_res
      bisc_power_consumption_channels = bisc_sensing_power_consumption + bisc_non_sensing_power_consumption_mlp
      if channels == bisc_orig_channels + step:
        bisc_power_consumption_plot.append(None)
      else:
        bisc_power_consumption_plot.append(bisc_power_consumption_channels)

    #Prepare x axis - channel number
    x_val = channels
    x_axis.append(x_val)


  fig, ax = plt.subplots(figsize=(12, 6))

  mid_index = len(bisc_total_power_budget_plot) // 2
  label_x, label_y = x_axis[mid_index-120], bisc_total_power_budget_plot[mid_index-80]
  #angle = np.arctan2(bisc_total_power_budget_plot[mid_index+1] - bisc_total_power_budget_plot[mid_index-1], x_axis[mid_index+1] - x_axis[mid_index-1]) * 20 #/ np.pi
  angle = np.arctan2(bisc_total_power_budget_plot[mid_index+1] - bisc_total_power_budget_plot[mid_index-1], x_axis[mid_index+1] - x_axis[mid_index-1]) * 25 #/ np.pi
  print(angle)
  angle_degrees = np.degrees(angle)
  print(angle_degrees)

  ax.axvline(x=1024, color='b', linestyle=':', linewidth=2, label="Data rate limit")
  label_xx, label_yy = 1024-40, bisc_power_consumption_plot[mid_index * 2 - 110]
  print(bisc_total_power_budget_plot[-1])

  #ax.axvline(x=1370, ymin=0, ymax=0.4, color='gray', linestyle='--', linewidth=2, label=None)
  ax.axvline(x=1750, ymin=0, ymax=0.53, color='gray', linestyle='--', linewidth=2, label=None)

  plt.plot(x_axis, bisc_total_power_budget_plot, linestyle='--', linewidth=4, color='red', label=None)
  plt.text(label_x, label_y, 'Power Budget', rotation=angle_degrees-360, ha='center', va='center', color='red', weight='bold', fontsize=18)
  #plt.text(label_xx, label_yy, 'Zeng et al.', rotation=90, ha='center', va='center', color='blue', fontsize=12)
  #plt.text(x_axis[int(max_channels/2 / step)] + 0.1, bisc_total_power_budget_plot[int(max_channels/2 / step)], 'Power Budget', color='red')

  plt.plot(x_axis, bisc_sensing_power_consumption_plot, label='Sensing Power Consumption', color='blue', linewidth=3)
  #plt.plot(x_axis, bisc_non_sensing_power_consumption_pipe_plot, label='Non-Sensing Power (DN-CNN)', color='purple', linewidth=3)
  plt.plot(x_axis, bisc_non_sensing_power_consumption_mlp_plot, label='Non-Sensing Power (MLP)', color='purple', linewidth=3)
  plt.plot(x_axis, bisc_non_sensing_power_consumption_comm_plot, label='Non-Sensing Power (Comm.)', color=(0.8, 0.6, 0), linewidth=3)
  #plt.plot(x_axis, bisc_non_sensing_power_consumption_no_pipe_plot, label='Non-Sensing Power Consumption No Pipe')
  plt.plot(x_axis, bisc_power_consumption_plot, label='Total Power Consumption', color='green', linewidth=3)
  plt.xticks(x_axis[::tick_ratio], fontsize=18)
  plt.yticks(fontsize=18)
  plt.xlabel('Number of Active Channels', fontsize=22)
  plt.ylabel('Power [mW]', fontsize=22)
  plt.grid(axis='y')
  #plt.title('Number of Active Channels vs. Power')
  plt.legend(fontsize=15,loc='upper left', bbox_to_anchor=(0, 1), ncol=1)

  #plt.savefig('comp_scale1.pdf')
  plt.savefig('comp_scale2.pdf')

  plt.show()


def bisc_layers_neural_iterface_scaling():

  physical_bisc = physical_constraints.copy()
  comm_bisc = communication.copy()

  max_channels = (2048+64+256)
  min_channels = 0
  step = 4


  #BISC parameters
  bisc_orig_channels = 1024
  bisc_max_orig_channels = 65536
  bisc_sensing_area = 6.8 * 7.4 #mm^2 - BISC
  bisc_total_area = 144 #mm^2 - BISC
  bisc_non_sensing_area = bisc_total_area - bisc_sensing_area
  bisc_power_consumption = 38.8 #mW
  bisc_max_comm_channels = bisc_orig_channels

  # #IBIS parameters
  # ibis_orig_channels = 192 * 245
  # ibis_max_orig_channels = 192 * 245
  # ibis_sensing_area = 5.76 * 7.68 #mm^2
  # ibis_total_area = 144 #mm^2 - BISC
  # ibis_non_sensing_area = ibis_total_area - ibis_sensing_area
  # ibis_power_consumption = 48 #mW
  # ibis_max_comm_channels = ibis_orig_channels

  #max_data_rate = 300 #Mb/sec
  data_type = 10 #12.5
  network_time = 0.5*10**6 / 4 #8KHz sampling
  #max_comm_channels = 300*10**6 / data_type * network_time * 10**-9
  power_budget_per_area = 0.4 #mW/mm^2 - from literature

  bisc_non_sensing_area_per_channel = bisc_non_sensing_area / bisc_max_comm_channels
  bisc_sensing_area_per_channel = bisc_sensing_area / bisc_orig_channels #0.000768 #mm^2
  #bisc_area_per_channel = bisc_non_sensing_area_per_channel + bisc_sensing_area_per_channel
  bisc_area_per_channel = bisc_total_area / bisc_orig_channels

  #bisc_power_per_channel_recording = 0.01325 #mW
  bisc_power_per_channel_recording = bisc_power_consumption * bisc_sensing_area/bisc_total_area / bisc_orig_channels
  bisc_power_per_channel_comm = bisc_power_consumption * bisc_non_sensing_area/bisc_total_area / bisc_max_comm_channels
  #power_per_channel = 0.03789 #mW


  print("BISC: Non sensing area per channel:", bisc_non_sensing_area_per_channel)
  print("BISC: Sensing area per channel:", bisc_sensing_area_per_channel)
  print("BISC: Max communication channels:", bisc_max_comm_channels)


  # total_area = area_per_channel * channels
  # total_power_budget = power_budget_per_area * total_area

  #Check what happens when number of channels grows
  channel_numbers = list(range(min_channels, max_channels, step))
  x_axis = list()
  bisc_total_power_budget_plot = list()
  bisc_sensing_power_budget_plot = list()
  bisc_non_sensing_power_budget_plot = list()
  bisc_sensing_power_consumption_plot = list()
  bisc_non_sensing_power_consumption_no_pipe_plot = list()
  bisc_non_sensing_power_consumption_pipe_plot = list()
  bisc_non_sensing_power_consumption_layers_plot = list()
  bisc_non_sensing_power_consumption_mlp_plot = list()
  bisc_non_sensing_power_consumption_comm_plot = list()
  bisc_power_consumption_plot = list()

  #tick_ratio = math.ceil(max_channels / 256 * 16 / step)
  tick_ratio = 64


  for channels in channel_numbers:
    print("Num channels:", channels)

    if channels <= bisc_orig_channels:
      #area_per_channel = non_sensing_area_per_channel + sensing_area_per_channel
      if channels == 0:
        bisc_total_power_budget_plot.append(0)
        bisc_sensing_power_consumption_plot.append(0)
        #bisc_non_sensing_power_consumption_no_pipe_plot.append(0)
        bisc_non_sensing_power_consumption_pipe_plot.append(None)
        bisc_non_sensing_power_consumption_layers_plot.append(None)
        bisc_non_sensing_power_consumption_mlp_plot.append(None)
        bisc_non_sensing_power_consumption_comm_plot.append(0)
        bisc_power_consumption_plot.append(0)

        #Prepare x axis - channel number
        x_val = channels
        x_axis.append(x_val)
        continue

      bisc_total_area = bisc_area_per_channel * channels
      bisc_total_power_budget = power_budget_per_area * bisc_total_area

      bisc_sensing_power_consumption = bisc_power_per_channel_recording * channels
      bisc_non_sensing_power_consumption = bisc_power_per_channel_comm * channels
      bisc_power_consumption_channels = (bisc_power_per_channel_recording + bisc_power_per_channel_comm) * channels

      # print("\nPOWER BUDGET:", new_power_budget, "mW")
      # print("TOTAL AREA:", total_area, "mm^2")
      # print("SENSING AREA:", total_sensing_area, "mm^2")
      # print("NON SENSING AREA:", total_non_sensing_area, "mm^2\n")

      bisc_total_power_budget_plot.append(bisc_total_power_budget)
      bisc_sensing_power_consumption_plot.append(bisc_sensing_power_consumption)
      #bisc_non_sensing_power_consumption_no_pipe_plot.append(bisc_non_sensing_power_consumption)
      bisc_non_sensing_power_consumption_pipe_plot.append(None) #no computation yet
      bisc_non_sensing_power_consumption_layers_plot.append(None)
      bisc_non_sensing_power_consumption_mlp_plot.append(None)
      bisc_non_sensing_power_consumption_comm_plot.append(bisc_non_sensing_power_consumption)
      bisc_power_consumption_plot.append(bisc_power_consumption_channels)

    else: #computation scaling

      #parameters
      reduction = False #doesn't need to be set true can just set aggressive to true
      aggressive = True
      budget=True # sets power budget to the maximum
      check_data_rate = True
      network_input_dependency = 1

      #sensing power and area
      bisc_sensing_power_consumption = bisc_power_per_channel_recording * channels
      bisc_sensing_area = bisc_sensing_area_per_channel * channels

      #non sensing area
      #bisc_non_sensing_area - stays constant with 1024 channels
      #bisc_non_sensing_area = bisc_non_sensing_area_per_channel * channels

      #non-sensing power - computation
      #calculate the communication - no QAM
      physical_bisc["network_time"] = network_time #8KHz as bisc - just to set the maximum data rate
      physical_bisc["data_type"] = data_type #+2.5
      # physical_bisc["mac_power"] = 0.026 #12nm
      # physical_bisc["mac_time"] = 1 #12nm
      physical_bisc["mac_area"] = 0.000783 * 1.5 #mm^2 , 45nm + an overhead for wiring
      communication_power_orig, data_rate_orig = calc_communication(bisc_orig_channels, 0, physical_bisc, comm_bisc, enable_qam=False)
      comm_bisc["max_data_rate"] = data_rate_orig / 10**6
      print("MAX DATA RATE:", comm_bisc["max_data_rate"])
      #communication_power, data_rate = calc_communication(channels, 0, physical_bisc, comm_bisc, enable_qam=False)
      #communication_power_qam, data_rate_qam = calc_communication(channels, 0, physical_bisc, comm_bisc, enable_qam=True)

      #for the model we can increase the network time
      physical_bisc["network_time"] = network_time * 4 #2KHz

      arch = densenet_architecture.copy()
      original_channels = arch["input_channels"]
      physical_bisc["num_channels"] = channels
      arch["input_channels"] = channels #dense

      # #power budget
      # bisc_total_power_budget = power_budget_per_area * (bisc_sensing_area + bisc_non_sensing_area)
      #physical_bisc["power_budget"] = bisc_total_power_budget - bisc_sensing_power_consumption #update power budget
      non_sensing_power_budget = np.inf #bisc_total_power_budget - bisc_sensing_power_consumption #update power budget

      ratio = (channels / original_channels)
      arch["growth_factor"] = math.ceil(ratio * network_input_dependency * arch["growth_factor"])

      bisc_total_power_budget = power_budget_per_area * (bisc_sensing_area + bisc_non_sensing_area)
      bisc_non_sensing_power_budget = bisc_total_power_budget - bisc_sensing_power_consumption

      #start while loop to find the right DNN subset

      no_compute_res, full_pipe_res, merge_pipe_res = \
        run_paper_network(paper_densenet_pipe, arch, physical_bisc, comm_bisc, total_power_budget=bisc_non_sensing_power_budget, allow_reduction=reduction, aggressive=False, budget=budget, check_data_rate=check_data_rate)

      _, _, merge_pipe_layers_res = \
        run_paper_network(paper_densenet_pipe, arch, physical_bisc, comm_bisc, total_power_budget=bisc_non_sensing_power_budget, allow_reduction=reduction, aggressive=True, budget=budget, check_data_rate=check_data_rate)

      # _, recursive_res, no_recursive_res = \
      #   run_network(densenet, arch, physical_bisc, comm_bisc, allow_reduction=reduction, aggressive=aggressive, budget=budget, check_data_rate=check_data_rate)

      # if(data_rate/10**6 < max_data_rate):
      #   linear_comm.append(communication_power*10**3)
      #   not_supported_linear_comm.append(None)
      #   qam_communication.append(None)
      # else:
      #   linear_comm.append(None)
      #   not_supported_linear_comm.append(communication_power*10**3)
      #   communication_power, data_rate = calc_communication(channels, 0, physical, comm, enable_qam=True)
      #   qam_communication.append(communication_power*10**3)

      #bisc_non_sensing_power_consumption_no_pipe = bisc_power_per_channel_comm * bisc_orig_channels + recursive_res
      if merge_pipe_res != None and merge_pipe_res < non_sensing_power_budget:
        bisc_non_sensing_power_consumption_pipe = merge_pipe_res
        bisc_non_sensing_power_consumption_layers = merge_pipe_layers_res
        #bisc_non_sensing_power_consumption_pipe = bisc_power_per_channel_comm * bisc_orig_channels + merge_pipe_res
      else:
        break
        #bisc_non_sensing_power_consumption = bisc_non_sensing_power_consumption + communication_power * 10**3

      #power budget
      # bisc_comp_non_sensing_area = physical_bisc["mac_num"] * physical_bisc["mac_area"] #instead of keeping it the same as communication
      # bisc_comm_non_sensing_area = arch["output_channels"] / bisc_orig_channels * bisc_non_sensing_area #instead of keeping it the same as communication
      # bisc_total_power_budget = power_budget_per_area * (bisc_sensing_area + bisc_comm_non_sensing_area + bisc_comp_non_sensing_area)
      #bisc_total_power_budget = power_budget_per_area * (bisc_sensing_area + bisc_non_sensing_area)
      print("Main: power budget is:", bisc_total_power_budget, "non-sensing budget is:", bisc_non_sensing_power_budget)

      #power consumption
      #bisc_power_consumption_channels = bisc_sensing_power_consumption + bisc_non_sensing_power_consumption_pipe
      bisc_power_consumption_channels = bisc_sensing_power_consumption + bisc_non_sensing_power_consumption_layers

      #this is where I should end the while loop

      bisc_total_power_budget_plot.append(bisc_total_power_budget) #
      bisc_sensing_power_consumption_plot.append(bisc_sensing_power_consumption) #
      #bisc_non_sensing_power_consumption_no_pipe_plot.append(bisc_non_sensing_power_consumption_no_pipe)
      bisc_non_sensing_power_consumption_pipe_plot.append(bisc_non_sensing_power_consumption_pipe)
      bisc_non_sensing_power_consumption_layers_plot.append(bisc_non_sensing_power_consumption_layers)
      bisc_non_sensing_power_consumption_comm_plot.append(None)
      bisc_power_consumption_plot.append(bisc_power_consumption_channels)

      #MLP

      mlp_arch = mlp_architecture.copy()
      original_channels = mlp_arch["input_channels"]
      ratio = (channels / original_channels)
      #dropout_channels = math.ceil(channels * network_input_dependency_mlp)
      #physical["num_channels"] = channels
      mlp_arch["input_channels"] = channels
      mlp_arch["input_size"] = channels * mlp_arch["timestamps"] #mlp
      update_hidden = [mlp_arch["hidden_size"][0]] * len(mlp_arch["hidden_size"] * math.ceil(ratio)) #.copy()
      for i in range(len(update_hidden)):
        N = len(update_hidden)
        update_hidden[i] = round(update_hidden[i] * (1 + ((N - i - 1) / (N - 1)) * (ratio - 1)))
        print(update_hidden)
      #update_hidden = [round(layer * ratio) for layer in mlp_arch["hidden_size"]]
      mlp_arch["hidden_size"] = update_hidden

      _, _, mlp_res = \
        run_paper_network(paper_mlp_pipe, mlp_arch, physical_bisc, comm_bisc, total_power_budget=bisc_non_sensing_power_budget, allow_reduction=reduction, aggressive=False, budget=budget, check_data_rate=check_data_rate)

      _, _, mlp_res_layers = \
        run_paper_network(paper_mlp_pipe, mlp_arch, physical_bisc, comm_bisc, total_power_budget=bisc_non_sensing_power_budget, allow_reduction=reduction, aggressive=True, budget=budget, check_data_rate=check_data_rate)

      bisc_non_sensing_power_consumption_mlp = mlp_res_layers
      bisc_non_sensing_power_consumption_mlp_plot.append(bisc_non_sensing_power_consumption_mlp)

      # #power consumption
      # bisc_power_consumption_channels = bisc_sensing_power_consumption + bisc_non_sensing_power_consumption_mlp
      # if channels == bisc_orig_channels + step:
      #   bisc_power_consumption_plot.append(None)
      # else:
      #   bisc_power_consumption_plot.append(bisc_power_consumption_channels)

    #Prepare x axis - channel number
    x_val = channels
    x_axis.append(x_val)


  fig, ax = plt.subplots(figsize=(12, 6))

  mid_index = len(bisc_total_power_budget_plot) // 2
  label_x, label_y = x_axis[mid_index-120], bisc_total_power_budget_plot[mid_index-70]
  angle = np.arctan2(bisc_total_power_budget_plot[mid_index+1] - bisc_total_power_budget_plot[mid_index-1], x_axis[mid_index+1] - x_axis[mid_index-1]) * 20 #/ np.pi
  print(angle)
  angle_degrees = np.degrees(angle)
  print(angle_degrees)

  ax.axvline(x=1024, color='b', linestyle=':', linewidth=2, label="Data rate limit")
  label_xx, label_yy = 1024-40, bisc_power_consumption_plot[mid_index * 2 - 40]
  print(bisc_total_power_budget_plot[-1])

  plt.plot(x_axis, bisc_total_power_budget_plot, linestyle='--', linewidth=4, color='red', label=None)
  plt.text(label_x, label_y, 'Power Budget', rotation=angle_degrees-360, ha='center', va='center', color='red', weight='bold', fontsize=18)
  #plt.text(label_xx, label_yy, 'Zeng et al.', rotation=90, ha='center', va='center', color='blue', fontsize=12)
  #plt.text(x_axis[int(max_channels/2 / step)] + 0.1, bisc_total_power_budget_plot[int(max_channels/2 / step)], 'Power Budget', color='red')


  #plt.plot(x_axis, bisc_non_sensing_power_consumption_pipe_plot, label='Non-Sensing Power Consumption (DN-CNN)', color='purple')

  ax.axvline(x=1390 ,ymin=0, ymax=0.4, color='gray', linestyle='--', linewidth=2, label=None)
  #ax.axvline(x=2230 ,ymin=0, ymax=0.8, color='gray', linestyle='--', linewidth=2, label=None)

  plt.plot(x_axis, bisc_non_sensing_power_consumption_layers_plot, label='Non-Sensing Power (DN-CNN)', color='purple')
  plt.plot(x_axis, bisc_sensing_power_consumption_plot, label='Sensing Power Consumption', color='blue', linewidth=3)
  #plt.plot(x_axis, bisc_non_sensing_power_consumption_mlp_plot, label='Non-Sensing Power (MLP)', color='purple', linewidth=3)
  plt.plot(x_axis, bisc_non_sensing_power_consumption_comm_plot, label='Non-Sensing Power (Comm.)', color=(0.8, 0.6, 0), linewidth=3)
  #plt.plot(x_axis, bisc_non_sensing_power_consumption_no_pipe_plot, label='Non-Sensing Power Consumption No Pipe')
  plt.plot(x_axis, bisc_power_consumption_plot, label='Total Power Consumption', color='green', linewidth=3)
  plt.xticks(x_axis[::tick_ratio], fontsize=18)
  plt.yticks(fontsize=18)
  plt.xlabel('Number of Active Channels', fontsize=22)
  plt.ylabel('Power [mW]', fontsize=22)
  plt.grid(axis='y')
  #plt.title('Number of Active Channels vs. Power')
  plt.legend(fontsize=15,loc='upper left', bbox_to_anchor=(0, 1), ncol=1)

  plt.savefig('comp_layers1.pdf')
  #plt.savefig('comp_layers2.pdf')

  plt.show()


def bisc_dropout_neural_iterface_scaling():

  physical_bisc = physical_constraints.copy()
  comm_bisc = communication.copy()

  max_channels = (2048+64+256)*4
  min_channels = 1024
  step = 1024


  #BISC parameters
  bisc_orig_channels = 1024
  bisc_max_orig_channels = 65536
  bisc_sensing_area = 6.8 * 7.4 #mm^2 - BISC
  bisc_total_area = 144 #mm^2 - BISC
  bisc_non_sensing_area = bisc_total_area - bisc_sensing_area
  bisc_power_consumption = 38.8 #mW
  bisc_max_comm_channels = bisc_orig_channels

  # #IBIS parameters
  # ibis_orig_channels = 192 * 245
  # ibis_max_orig_channels = 192 * 245
  # ibis_sensing_area = 5.76 * 7.68 #mm^2
  # ibis_total_area = 144 #mm^2 - BISC
  # ibis_non_sensing_area = ibis_total_area - ibis_sensing_area
  # ibis_power_consumption = 48 #mW
  # ibis_max_comm_channels = ibis_orig_channels

  #max_data_rate = 300 #Mb/sec
  data_type = 10 #12.5
  network_time = 0.5*10**6 / 4 #8KHz sampling
  #max_comm_channels = 300*10**6 / data_type * network_time * 10**-9
  power_budget_per_area = 0.4 #mW/mm^2 - from literature

  bisc_non_sensing_area_per_channel = bisc_non_sensing_area / bisc_max_comm_channels
  bisc_sensing_area_per_channel = bisc_sensing_area / bisc_orig_channels #0.000768 #mm^2
  #bisc_area_per_channel = bisc_non_sensing_area_per_channel + bisc_sensing_area_per_channel
  bisc_area_per_channel = bisc_total_area / bisc_orig_channels
  bisc_area_per_channel_density = bisc_non_sensing_area_per_channel + bisc_sensing_area_per_channel / 2

  #bisc_power_per_channel_recording = 0.01325 #mW
  bisc_power_per_channel_recording = bisc_power_consumption * bisc_sensing_area/bisc_total_area / bisc_orig_channels
  bisc_power_per_channel_comm = bisc_power_consumption * bisc_non_sensing_area/bisc_total_area / bisc_max_comm_channels
  #power_per_channel = 0.03789 #mW


  print("BISC: Non sensing area per channel:", bisc_non_sensing_area_per_channel)
  print("BISC: Sensing area per channel:", bisc_sensing_area_per_channel)
  print("BISC: Max communication channels:", bisc_max_comm_channels)


  # total_area = area_per_channel * channels
  # total_power_budget = power_budget_per_area * total_area

  #Check what happens when number of channels grows
  channel_numbers = list(range(min_channels, max_channels, step))
  x_axis = list()
  bisc_total_power_budget_plot = list()
  bisc_total_power_budget_density_plot = list()
  bisc_sensing_power_budget_plot = list()
  bisc_non_sensing_power_budget_plot = list()
  bisc_sensing_power_consumption_plot = list()
  bisc_non_sensing_power_consumption_no_pipe_plot = list()
  bisc_non_sensing_power_consumption_pipe_plot = list()
  bisc_non_sensing_power_consumption_layers_plot = list()
  bisc_non_sensing_power_consumption_mlp_plot = list()
  bisc_non_sensing_power_consumption_mlp_default_plot = list()
  bisc_non_sensing_power_consumption_mlp_dropout_plot = list()
  bisc_non_sensing_power_consumption_mlp_layers_dropout_plot = list()
  bisc_non_sensing_power_consumption_mlp_tech_plot = list()
  bisc_non_sensing_power_consumption_mlp_density_plot = list()
  bisc_non_sensing_power_consumption_comm_plot = list()
  bisc_power_consumption_plot = list()
  bisc_power_consumption_default_plot = list()
  bisc_power_consumption_dropout_plot = list()
  bisc_power_consumption_layers_dropout_plot = list()
  bisc_power_consumption_tech_plot = list()
  bisc_power_consumption_density_plot = list()

  #tick_ratio = math.ceil(max_channels / 256 * 16 / step)
  tick_ratio = 2

  default_size = [0] * len(range(max_channels))
  dropout_size = [0] * len(range(max_channels))
  layers_dropout_size = [0] * len(range(max_channels))
  tech_size = [0] * len(range(max_channels))
  density_size = [0] * len(range(max_channels))

  for channels in channel_numbers:
    print("Num channels:", channels)

    if channels <= bisc_orig_channels:
      #area_per_channel = non_sensing_area_per_channel + sensing_area_per_channel
      if channels == 0:
        bisc_total_power_budget_plot.append(0)
        bisc_total_power_budget_density_plot.append(0)
        bisc_sensing_power_consumption_plot.append(0)
        #bisc_non_sensing_power_consumption_no_pipe_plot.append(0)
        bisc_non_sensing_power_consumption_pipe_plot.append(None)
        bisc_non_sensing_power_consumption_layers_plot.append(None)
        bisc_non_sensing_power_consumption_mlp_plot.append(None)
        bisc_non_sensing_power_consumption_mlp_default_plot.append(None)
        bisc_non_sensing_power_consumption_mlp_dropout_plot.append(None)
        bisc_non_sensing_power_consumption_mlp_layers_dropout_plot.append(None)
        bisc_non_sensing_power_consumption_mlp_tech_plot.append(None)
        bisc_non_sensing_power_consumption_mlp_density_plot.append(None)
        bisc_non_sensing_power_consumption_comm_plot.append(0)
        bisc_power_consumption_plot.append(0)
        bisc_power_consumption_default_plot.append(0)
        bisc_power_consumption_dropout_plot.append(0)
        bisc_power_consumption_layers_dropout_plot.append(0)
        bisc_power_consumption_tech_plot.append(0)
        bisc_power_consumption_density_plot.append(0)

        #Prepare x axis - channel number
        x_val = channels
        x_axis.append(x_val)
        continue

      bisc_total_area = bisc_area_per_channel * channels
      bisc_total_area_density = bisc_area_per_channel_density * channels
      bisc_total_power_budget = power_budget_per_area * bisc_total_area
      bisc_total_power_budget_density = power_budget_per_area * bisc_total_area_density

      bisc_sensing_power_consumption = bisc_power_per_channel_recording * channels
      bisc_non_sensing_power_consumption = bisc_power_per_channel_comm * channels
      bisc_power_consumption_channels = (bisc_power_per_channel_recording + bisc_power_per_channel_comm) * channels

      # print("\nPOWER BUDGET:", new_power_budget, "mW")
      # print("TOTAL AREA:", total_area, "mm^2")
      # print("SENSING AREA:", total_sensing_area, "mm^2")
      # print("NON SENSING AREA:", total_non_sensing_area, "mm^2\n")

      bisc_total_power_budget_plot.append(bisc_total_power_budget)
      bisc_total_power_budget_density_plot.append(bisc_total_power_budget_density)
      bisc_sensing_power_consumption_plot.append(bisc_sensing_power_consumption)
      #bisc_non_sensing_power_consumption_no_pipe_plot.append(bisc_non_sensing_power_consumption)
      bisc_non_sensing_power_consumption_pipe_plot.append(None) #no computation yet
      bisc_non_sensing_power_consumption_layers_plot.append(None)
      bisc_non_sensing_power_consumption_mlp_plot.append(None)
      bisc_non_sensing_power_consumption_mlp_default_plot.append(None)
      bisc_non_sensing_power_consumption_mlp_dropout_plot.append(None)
      bisc_non_sensing_power_consumption_mlp_layers_dropout_plot.append(None)
      bisc_non_sensing_power_consumption_mlp_tech_plot.append(None)
      bisc_non_sensing_power_consumption_mlp_density_plot.append(None)
      bisc_non_sensing_power_consumption_comm_plot.append(bisc_non_sensing_power_consumption)
      bisc_power_consumption_plot.append(bisc_power_consumption_channels)
      bisc_power_consumption_default_plot.append(bisc_power_consumption_channels)
      bisc_power_consumption_dropout_plot.append(bisc_power_consumption_channels)
      bisc_power_consumption_layers_dropout_plot.append(bisc_power_consumption_channels)
      bisc_power_consumption_tech_plot.append(bisc_power_consumption_channels)
      bisc_power_consumption_density_plot.append(bisc_power_consumption_channels)

    else: #computation scaling

      #parameters
      reduction = False #doesn't need to be set true can just set aggressive to true
      aggressive = True
      budget=True # sets power budget to the maximum
      check_data_rate = True
      network_input_dependency = 1

      #sensing power and area
      bisc_sensing_power_consumption = bisc_power_per_channel_recording * channels
      bisc_sensing_area = bisc_sensing_area_per_channel * channels
      bisc_sensing_area_const_density = bisc_sensing_area_per_channel * channels/2
      bisc_non_sensing_area_density = bisc_non_sensing_area #bisc_non_sensing_area_per_channel * channels

      #non sensing area
      #bisc_non_sensing_area - stays constant with 1024 channels
      #bisc_non_sensing_area = bisc_non_sensing_area_per_channel * channels

      #non-sensing power - computation
      #calculate the communication - no QAM
      physical_bisc["network_time"] = network_time #8KHz as bisc - just to set the maximum data rate
      physical_bisc["data_type"] = data_type #+2.5
      physical_bisc["mac_power"] = 0.05 #45nm
      physical_bisc["mac_time"] = 2 #45nm
      physical_bisc["mac_area"] = 0.000783 * 1.5 #mm^2 , 45nm + an overhead for wiring
      communication_power_orig, data_rate_orig = calc_communication(bisc_orig_channels, 0, physical_bisc, comm_bisc, enable_qam=False)
      comm_bisc["max_data_rate"] = data_rate_orig / 10**6
      print("MAX DATA RATE:", comm_bisc["max_data_rate"])
      #communication_power, data_rate = calc_communication(channels, 0, physical_bisc, comm_bisc, enable_qam=False)
      #communication_power_qam, data_rate_qam = calc_communication(channels, 0, physical_bisc, comm_bisc, enable_qam=True)

      #for the model we can increase the network time
      physical_bisc["network_time"] = network_time * 4 #2KHz

      arch = densenet_architecture.copy()
      original_channels = arch["input_channels"]
      physical_bisc["num_channels"] = channels
      arch["input_channels"] = channels #dense

      # #power budget
      # bisc_total_power_budget = power_budget_per_area * (bisc_sensing_area + bisc_non_sensing_area)
      #physical_bisc["power_budget"] = bisc_total_power_budget - bisc_sensing_power_consumption #update power budget
      non_sensing_power_budget = np.inf #bisc_total_power_budget - bisc_sensing_power_consumption #update power budget

      ratio = (channels / original_channels)
      arch["growth_factor"] = math.ceil(ratio * network_input_dependency * arch["growth_factor"])

      bisc_total_power_budget = power_budget_per_area * (bisc_sensing_area + bisc_non_sensing_area)
      bisc_total_power_budget_const_density = power_budget_per_area * (bisc_sensing_area_const_density + bisc_non_sensing_area_density)
      bisc_non_sensing_power_budget = bisc_total_power_budget - bisc_sensing_power_consumption
      bisc_non_sensing_power_budget_const_density = bisc_total_power_budget_const_density - bisc_sensing_power_consumption

      #start while loop to find the right DNN subset

      no_compute_res, full_pipe_res, merge_pipe_res = \
        run_paper_network(paper_densenet_pipe, arch, physical_bisc, comm_bisc, total_power_budget=bisc_non_sensing_power_budget, allow_reduction=reduction, aggressive=False, budget=budget, check_data_rate=check_data_rate)

      _, _, merge_pipe_layers_res = \
        run_paper_network(paper_densenet_pipe, arch, physical_bisc, comm_bisc, total_power_budget=bisc_non_sensing_power_budget, allow_reduction=reduction, aggressive=True, budget=budget, check_data_rate=check_data_rate)

      # _, recursive_res, no_recursive_res = \
      #   run_network(densenet, arch, physical_bisc, comm_bisc, allow_reduction=reduction, aggressive=aggressive, budget=budget, check_data_rate=check_data_rate)

      # if(data_rate/10**6 < max_data_rate):
      #   linear_comm.append(communication_power*10**3)
      #   not_supported_linear_comm.append(None)
      #   qam_communication.append(None)
      # else:
      #   linear_comm.append(None)
      #   not_supported_linear_comm.append(communication_power*10**3)
      #   communication_power, data_rate = calc_communication(channels, 0, physical, comm, enable_qam=True)
      #   qam_communication.append(communication_power*10**3)

      #bisc_non_sensing_power_consumption_no_pipe = bisc_power_per_channel_comm * bisc_orig_channels + recursive_res
      if merge_pipe_res != None and merge_pipe_res < non_sensing_power_budget:
        bisc_non_sensing_power_consumption_pipe = merge_pipe_res
        bisc_non_sensing_power_consumption_layers = merge_pipe_layers_res
        #bisc_non_sensing_power_consumption_pipe = bisc_power_per_channel_comm * bisc_orig_channels + merge_pipe_res
      else:
        break
        #bisc_non_sensing_power_consumption = bisc_non_sensing_power_consumption + communication_power * 10**3

      #power budget
      # bisc_comp_non_sensing_area = physical_bisc["mac_num"] * physical_bisc["mac_area"] #instead of keeping it the same as communication
      # bisc_comm_non_sensing_area = arch["output_channels"] / bisc_orig_channels * bisc_non_sensing_area #instead of keeping it the same as communication
      # bisc_total_power_budget = power_budget_per_area * (bisc_sensing_area + bisc_comm_non_sensing_area + bisc_comp_non_sensing_area)
      #bisc_total_power_budget = power_budget_per_area * (bisc_sensing_area + bisc_non_sensing_area)
      print("Main: power budget is:", bisc_total_power_budget, "non-sensing budget is:", bisc_non_sensing_power_budget)

      #power consumption
      #bisc_power_consumption_channels = bisc_sensing_power_consumption + bisc_non_sensing_power_consumption_pipe
      #bisc_power_consumption_channels = bisc_sensing_power_consumption + bisc_non_sensing_power_consumption_layers

      #this is where I should end the while loop

      bisc_total_power_budget_plot.append(bisc_total_power_budget) #
      bisc_total_power_budget_density_plot.append(bisc_total_power_budget_const_density) #
      bisc_sensing_power_consumption_plot.append(bisc_sensing_power_consumption) #
      #bisc_non_sensing_power_consumption_no_pipe_plot.append(bisc_non_sensing_power_consumption_no_pipe)
      bisc_non_sensing_power_consumption_pipe_plot.append(bisc_non_sensing_power_consumption_pipe)
      bisc_non_sensing_power_consumption_layers_plot.append(bisc_non_sensing_power_consumption_layers)
      bisc_non_sensing_power_consumption_comm_plot.append(None)
      #bisc_power_consumption_plot.append(bisc_power_consumption_channels)

      #MLP

      mlp_arch = mlp_architecture.copy()
      original_channels = mlp_arch["input_channels"]
      ratio = (channels / original_channels)
      #dropout_channels = math.ceil(channels * network_input_dependency_mlp)
      #physical["num_channels"] = channels
      mlp_arch["input_channels"] = channels
      mlp_arch["input_size"] = channels * mlp_arch["timestamps"] #mlp
      update_hidden = [mlp_arch["hidden_size"][0]] * len(mlp_arch["hidden_size"] * math.ceil(ratio)) #.copy()
      for i in range(len(update_hidden)):
        N = len(update_hidden)
        update_hidden[i] = round(update_hidden[i] * (1 + ((N - i - 1) / (N - 1)) * (ratio - 1)))
        print(update_hidden)
      #update_hidden = [round(layer * ratio) for layer in mlp_arch["hidden_size"]]
      mlp_arch["hidden_size"] = update_hidden

      _, _, mlp_res = \
        run_paper_network(paper_mlp_pipe, mlp_arch, physical_bisc, comm_bisc, total_power_budget=bisc_non_sensing_power_budget, allow_reduction=reduction, aggressive=False, budget=budget, check_data_rate=check_data_rate)

      input_size = mlp_arch["input_size"]
      hidden_size = mlp_arch["hidden_size"]
      output_size = mlp_arch["output_size"]

      layer_operations, layer_sequences, output_per_layer = calc_mlp(input_size, hidden_size, output_size)
      total_accumulations = total_macs(layer_operations, layer_sequences)
      default_size[channels] = total_accumulations #store the default size if the model in MACs

      _, _, mlp_res_layers = \
        run_paper_network(paper_mlp_pipe, mlp_arch, physical_bisc, comm_bisc, total_power_budget=bisc_non_sensing_power_budget, allow_reduction=reduction, aggressive=True, budget=budget, check_data_rate=check_data_rate)


      #Channel Dropout
      found = False
      mlp_res_dropout = 0
      high = 1.0
      low = 0.0
      epsilon = 1e-4
      #physical["power_budget"] = new_power_budget * 0.9

      # network_input_dependency_mlp = find_maximum_value(mlp_function, channels, physical, comm, reduction, aggressive, budget, check_data_rate)
      # mlp_res = mlp_function(network_input_dependency_mlp, channels, physical, comm, reduction, aggressive, budget, check_data_rate)

      while found == False:
        mid = (high + low) / 2
        if high - low < epsilon: #finished
          found = True
          print("Dropout value for MLP:", low)
          network_input_dependency_mlp = low
        else:
          network_input_dependency_mlp = mid

        mlp_arch = mlp_architecture.copy()
        original_channels = mlp_arch["input_channels"]
        ratio = (channels / original_channels)
        #dropout_channels = math.ceil(channels * network_input_dependency_mlp)
        #physical["num_channels"] = channels
        mlp_arch["input_channels"] = channels
        mlp_arch["input_size"] = channels * mlp_arch["timestamps"] #mlp
        #network_input_dependency_mlp = 1
        ratio = 1 + (ratio - 1) * network_input_dependency_mlp
        update_hidden = [mlp_arch["hidden_size"][0]] * len(mlp_arch["hidden_size"] * math.ceil(ratio)) #.copy()
        for i in range(len(update_hidden)):
          N = len(update_hidden)
          update_hidden[i] = round(update_hidden[i] * (1 + ((N - i - 1) / (N - 1)) * (ratio - 1)))
        print(update_hidden)
        #update_hidden = [round(layer * ratio) for layer in mlp_arch["hidden_size"]]
        mlp_arch["hidden_size"] = update_hidden

        _, _, mlp_res_dropout = \
          run_paper_network(paper_mlp_pipe, mlp_arch, physical_bisc, comm_bisc, total_power_budget=bisc_non_sensing_power_budget, allow_reduction=reduction, aggressive=False, budget=budget, check_data_rate=check_data_rate)

        input_size = mlp_arch["input_size"]
        hidden_size = mlp_arch["hidden_size"]
        output_size = mlp_arch["output_size"]

        layer_operations, layer_sequences, output_per_layer = calc_mlp(input_size, hidden_size, output_size)
        total_accumulations = total_macs(layer_operations, layer_sequences)
        dropout_size[channels] = total_accumulations #store the default size if the model in MACs

        if mlp_res_dropout > bisc_non_sensing_power_budget: #failed - need lower
          high = mid
        else:
          low = mid


      #end while

      #Channel Layer+Dropout
      found = False
      mlp_res_layers_dropout = 0
      high = 1.0
      low = 0.0
      epsilon = 1e-4
      #physical["power_budget"] = new_power_budget * 0.9

      # network_input_dependency_mlp = find_maximum_value(mlp_function, channels, physical, comm, reduction, aggressive, budget, check_data_rate)
      # mlp_res = mlp_function(network_input_dependency_mlp, channels, physical, comm, reduction, aggressive, budget, check_data_rate)

      while found == False:
        mid = (high + low) / 2
        if high - low < epsilon: #finished
          found = True
          print("Dropout value for MLP:", low)
          network_input_dependency_mlp = low
        else:
          network_input_dependency_mlp = mid

        mlp_arch = mlp_architecture.copy()
        original_channels = mlp_arch["input_channels"]
        ratio = (channels / original_channels)
        #dropout_channels = math.ceil(channels * network_input_dependency_mlp)
        #physical["num_channels"] = channels
        mlp_arch["input_channels"] = channels
        mlp_arch["input_size"] = channels * mlp_arch["timestamps"] #mlp
        #network_input_dependency_mlp = 1
        ratio = 1 + (ratio - 1) * network_input_dependency_mlp
        update_hidden = [mlp_arch["hidden_size"][0]] * len(mlp_arch["hidden_size"] * math.ceil(ratio)) #.copy()
        for i in range(len(update_hidden)):
          N = len(update_hidden)
          update_hidden[i] = round(update_hidden[i] * (1 + ((N - i - 1) / (N - 1)) * (ratio - 1)))
        print(update_hidden)
        #update_hidden = [round(layer * ratio) for layer in mlp_arch["hidden_size"]]
        mlp_arch["hidden_size"] = update_hidden

        _, _, mlp_res_layers_dropout = \
          run_paper_network(paper_mlp_pipe, mlp_arch, physical_bisc, comm_bisc, total_power_budget=bisc_non_sensing_power_budget, allow_reduction=reduction, aggressive=True, budget=budget, check_data_rate=check_data_rate)

        input_size = mlp_arch["input_size"]
        hidden_size = mlp_arch["hidden_size"]
        output_size = mlp_arch["output_size"]

        layer_operations, layer_sequences, output_per_layer = calc_mlp(input_size, hidden_size, output_size)
        total_accumulations = total_macs(layer_operations, layer_sequences)
        layers_dropout_size[channels] = total_accumulations #store the default size if the model in MACs

        if mlp_res_layers_dropout > bisc_non_sensing_power_budget: #failed - need lower
          high = mid
        else:
          low = mid


      #Channel Layer+Dropout+technology
      found = False
      mlp_res_tech = 0
      high = 1.0
      low = 0.0
      epsilon = 1e-4
      #physical["power_budget"] = new_power_budget * 0.9
      physical_bisc["mac_power"] = 0.026 #12nm
      physical_bisc["mac_time"] = 1 #12nm

      # network_input_dependency_mlp = find_maximum_value(mlp_function, channels, physical, comm, reduction, aggressive, budget, check_data_rate)
      # mlp_res = mlp_function(network_input_dependency_mlp, channels, physical, comm, reduction, aggressive, budget, check_data_rate)

      while found == False:
        mid = (high + low) / 2
        if high - low < epsilon: #finished
          found = True
          print("Dropout value for MLP:", low)
          network_input_dependency_mlp = low
        else:
          network_input_dependency_mlp = mid

        mlp_arch = mlp_architecture.copy()
        original_channels = mlp_arch["input_channels"]
        ratio = (channels / original_channels)
        #dropout_channels = math.ceil(channels * network_input_dependency_mlp)
        #physical["num_channels"] = channels
        mlp_arch["input_channels"] = channels
        mlp_arch["input_size"] = channels * mlp_arch["timestamps"] #mlp
        #network_input_dependency_mlp = 1
        ratio = 1 + (ratio - 1) * network_input_dependency_mlp
        update_hidden = [mlp_arch["hidden_size"][0]] * len(mlp_arch["hidden_size"] * math.ceil(ratio)) #.copy()
        for i in range(len(update_hidden)):
          N = len(update_hidden)
          update_hidden[i] = round(update_hidden[i] * (1 + ((N - i - 1) / (N - 1)) * (ratio - 1)))
        print(update_hidden)
        #update_hidden = [round(layer * ratio) for layer in mlp_arch["hidden_size"]]
        mlp_arch["hidden_size"] = update_hidden

        _, _, mlp_res_tech = \
          run_paper_network(paper_mlp_pipe, mlp_arch, physical_bisc, comm_bisc, total_power_budget=bisc_non_sensing_power_budget, allow_reduction=reduction, aggressive=True, budget=budget, check_data_rate=check_data_rate)

        input_size = mlp_arch["input_size"]
        hidden_size = mlp_arch["hidden_size"]
        output_size = mlp_arch["output_size"]

        layer_operations, layer_sequences, output_per_layer = calc_mlp(input_size, hidden_size, output_size)
        total_accumulations = total_macs(layer_operations, layer_sequences)
        tech_size[channels] = total_accumulations #store the default size if the model in MACs

        if mlp_res_tech > bisc_non_sensing_power_budget: #failed - need lower
          high = mid
        else:
          low = mid



      #Channel Layer+Dropout+technology+density
      found = False
      mlp_res_density = 0
      high = 1.0
      low = 0.0
      epsilon = 1e-4
      #physical["power_budget"] = new_power_budget * 0.9
      physical_bisc["mac_power"] = 0.026 #12nm
      physical_bisc["mac_time"] = 1 #12nm

      # network_input_dependency_mlp = find_maximum_value(mlp_function, channels, physical, comm, reduction, aggressive, budget, check_data_rate)
      # mlp_res = mlp_function(network_input_dependency_mlp, channels, physical, comm, reduction, aggressive, budget, check_data_rate)

      while found == False:
        mid = (high + low) / 2
        if high - low < epsilon: #finished
          found = True
          print("Dropout value for MLP:", low)
          network_input_dependency_mlp = low
        else:
          network_input_dependency_mlp = mid

        mlp_arch = mlp_architecture.copy()
        original_channels = mlp_arch["input_channels"]
        ratio = (channels / original_channels)
        #dropout_channels = math.ceil(channels * network_input_dependency_mlp)
        #physical["num_channels"] = channels
        mlp_arch["input_channels"] = channels
        mlp_arch["input_size"] = channels * mlp_arch["timestamps"] #mlp
        #network_input_dependency_mlp = 1
        ratio = 1 + (ratio - 1) * network_input_dependency_mlp
        update_hidden = [mlp_arch["hidden_size"][0]] * len(mlp_arch["hidden_size"] * math.ceil(ratio)) #.copy()
        for i in range(len(update_hidden)):
          N = len(update_hidden)
          update_hidden[i] = round(update_hidden[i] * (1 + ((N - i - 1) / (N - 1)) * (ratio - 1)))
        print(update_hidden)
        #update_hidden = [round(layer * ratio) for layer in mlp_arch["hidden_size"]]
        mlp_arch["hidden_size"] = update_hidden

        _, _, mlp_res_density = \
          run_paper_network(paper_mlp_pipe, mlp_arch, physical_bisc, comm_bisc, total_power_budget=bisc_non_sensing_power_budget_const_density, allow_reduction=reduction, aggressive=True, budget=budget, check_data_rate=check_data_rate)

        input_size = mlp_arch["input_size"]
        hidden_size = mlp_arch["hidden_size"]
        output_size = mlp_arch["output_size"]

        layer_operations, layer_sequences, output_per_layer = calc_mlp(input_size, hidden_size, output_size)
        total_accumulations = total_macs(layer_operations, layer_sequences)
        density_size[channels] = total_accumulations #store the default size if the model in MACs

        if mlp_res_density > bisc_non_sensing_power_budget_const_density: #failed - need lower
          high = mid
        else:
          low = mid

      print("non sensing power budget dense is:",bisc_non_sensing_power_budget_const_density, "it was:", bisc_non_sensing_power_budget)
      print("non sensing power dense:", mlp_res_density)
      print("sensing power:", bisc_sensing_power_consumption)
      print("total power budget density:", bisc_total_power_budget_const_density)


      bisc_non_sensing_power_consumption_mlp = mlp_res_layers
      bisc_non_sensing_power_consumption_mlp_default = mlp_res
      bisc_non_sensing_power_consumption_mlp_dropout = mlp_res_dropout
      bisc_non_sensing_power_consumption_mlp_layers_dropout = mlp_res_layers_dropout
      bisc_non_sensing_power_consumption_mlp_tech = mlp_res_tech
      bisc_non_sensing_power_consumption_mlp_density = mlp_res_density
      bisc_non_sensing_power_consumption_mlp_plot.append(bisc_non_sensing_power_consumption_mlp)
      bisc_non_sensing_power_consumption_mlp_default_plot.append(bisc_non_sensing_power_consumption_mlp_default)
      bisc_non_sensing_power_consumption_mlp_dropout_plot.append(bisc_non_sensing_power_consumption_mlp_dropout)
      bisc_non_sensing_power_consumption_mlp_layers_dropout_plot.append(bisc_non_sensing_power_consumption_mlp_layers_dropout)
      bisc_non_sensing_power_consumption_mlp_tech_plot.append(bisc_non_sensing_power_consumption_mlp_tech)
      bisc_non_sensing_power_consumption_mlp_density_plot.append(bisc_non_sensing_power_consumption_mlp_density)

      #power consumption
      bisc_power_consumption_channels = bisc_sensing_power_consumption + bisc_non_sensing_power_consumption_mlp
      bisc_power_consumption_channels_default = bisc_sensing_power_consumption + bisc_non_sensing_power_consumption_mlp_default
      bisc_power_consumption_channels_dropout = bisc_sensing_power_consumption + bisc_non_sensing_power_consumption_mlp_dropout
      bisc_power_consumption_channels_layers_dropout = bisc_sensing_power_consumption + bisc_non_sensing_power_consumption_mlp_layers_dropout
      bisc_power_consumption_channels_tech = bisc_sensing_power_consumption + bisc_non_sensing_power_consumption_mlp_tech
      bisc_power_consumption_channels_density = bisc_sensing_power_consumption + bisc_non_sensing_power_consumption_mlp_density
      # if channels == bisc_orig_channels + step:
      #   bisc_power_consumption_plot.append(None)
      # else:
      bisc_power_consumption_plot.append(bisc_power_consumption_channels)
      bisc_power_consumption_default_plot.append(bisc_power_consumption_channels_default)
      bisc_power_consumption_dropout_plot.append(bisc_power_consumption_channels_dropout)
      bisc_power_consumption_layers_dropout_plot.append(bisc_power_consumption_channels_layers_dropout)
      bisc_power_consumption_tech_plot.append(bisc_power_consumption_channels_tech)
      bisc_power_consumption_density_plot.append(bisc_power_consumption_channels_density)

      print("Final powers: default:", bisc_power_consumption_channels_default, "layers:", bisc_power_consumption_channels, "dropout:", bisc_power_consumption_channels_dropout)

    #Prepare x axis - channel number
    x_val = channels
    x_axis.append(x_val)


  fig, ax = plt.subplots(figsize=(10, 5))

  mid_index = len(bisc_total_power_budget_plot) // 2
  label_x, label_y = x_axis[mid_index], bisc_total_power_budget_plot[mid_index]
  angle = np.arctan2(bisc_total_power_budget_plot[mid_index+1] - bisc_total_power_budget_plot[mid_index-1], x_axis[mid_index+1] - x_axis[mid_index-1]) * 35 #/ np.pi
  print(angle)
  angle_degrees = np.degrees(angle)
  print(angle_degrees)

  #ax.axvline(x=1024, color='b', linestyle=':', linewidth=1, label=None)
  #label_xx, label_yy = 2048, bisc_power_consumption_plot[mid_index * 2]
  #print(bisc_total_power_budget_plot[-1])

  # plt.plot(x_axis, bisc_total_power_budget_plot, linestyle='--', color='red', label=None)
  # plt.text(label_x, label_y, 'Power Budget', rotation=angle_degrees-360, ha='center', va='center', color='red', weight='bold', fontsize=12)
  #plt.text(label_xx, label_yy, 'Zeng et al.', rotation=90, ha='center', va='center', color='blue', fontsize=12)
  #plt.text(x_axis[int(max_channels/2 / step)] + 0.1, bisc_total_power_budget_plot[int(max_channels/2 / step)], 'Power Budget', color='red')

  # plt.plot(x_axis, bisc_sensing_power_consumption_plot, label='Sensing Power Consumption', color='blue')
  # #plt.plot(x_axis, bisc_non_sensing_power_consumption_pipe_plot, label='Non-Sensing Power Consumption (DN-CNN)', color='purple')
  # #plt.plot(x_axis, bisc_non_sensing_power_consumption_layers_plot, label='Non-Sensing Power Consumption (DN-CNN)', color='purple')
  # ax.axvline(x=2230 ,ymin=0, ymax=0.8, color='gray', linestyle='--', linewidth=1, label=None)
  # plt.plot(x_axis, bisc_non_sensing_power_consumption_mlp_plot, label='Non-Sensing Power Consumption (MLP)', color='purple')
  # plt.plot(x_axis, bisc_non_sensing_power_consumption_comm_plot, label='Non-Sensing Power Consumption (Communication)', color=(0.8, 0.6, 0))
  # #plt.plot(x_axis, bisc_non_sensing_power_consumption_no_pipe_plot, label='Non-Sensing Power Consumption No Pipe')
  # plt.plot(x_axis, bisc_power_consumption_plot, label='Total Power Consumption', color='green')
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
  layer_power = list()
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
    power_budget = bisc_total_power_budget_plot[x_axis.index(channels)]
    power_budget_density = bisc_total_power_budget_density_plot[x_axis.index(channels)]
    print("power budget:", power_budget)
    print("power consumption", bisc_power_consumption_plot[x_axis.index(channels)])
    print("power budget dense:", power_budget_density)
    print("power consumption dense", bisc_power_consumption_density_plot[x_axis.index(channels)])
    default_power = default_power + [bisc_power_consumption_default_plot[x_axis.index(channels)]/power_budget]
    layer_power = layer_power + [bisc_power_consumption_plot[x_axis.index(channels)]/power_budget]
    dropout_power = dropout_power + [bisc_power_consumption_dropout_plot[x_axis.index(channels)]/power_budget]
    layers_dropout_power = layers_dropout_power + [bisc_power_consumption_layers_dropout_plot[x_axis.index(channels)]/power_budget]
    tech_power = tech_power + [bisc_power_consumption_tech_plot[x_axis.index(channels)]/power_budget]
    density_power = density_power + [bisc_power_consumption_density_plot[x_axis.index(channels)]/power_budget_density]
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

  plt.savefig('comp_dropout1.pdf')

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

  plt.savefig('comp_dropout2.pdf')

  plt.show()

if __name__ == "__main__":

  #mlp_channel_test()
  #mlp_pipeline_channel_test()
  #mlp_compare_all_channel_test()
  #mlp_hidden_channel_test() #hidden layers size changes with respect to input
  #mlp_hidden_reduction_channel_test() #allows to reduce the number of layers internally
  #mlp_hidden_reduction_aggressive_channel_test() #aggressively reduces total power to less than only-communication implementation
  #mlp_per_layer_test()
  #mlp_bisc_scale_channel_test()

  #densenet_channel_test()
  #densenet_pipeline_channel_test()
  #densenet_compare_all_channel_test()
  #densenet_scale_channel_test()
  #densenet_bisc_scale_channel_test()

  #s2s_bisc_scale_channel_test()

  #bisc_communication_test()
  #bisc_communication_and_computation_test()
  #bisc_optimized_dropout_test()
  #bisc_power_optimized_dropout_test()

  #bisc_neural_iterface_scaling()
  #bisc_communication_neural_iterface_scaling()
  #bisc_computation_neural_iterface_scaling()
  #bisc_layers_neural_iterface_scaling()
  bisc_dropout_neural_iterface_scaling()
