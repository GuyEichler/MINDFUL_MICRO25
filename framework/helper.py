#Helper function to help with calculating values
import math
import numpy as np

def layer_time(num_operations, sequence_length, num_macs, mac_time):

  sequence_time = sequence_length * mac_time
  max_time = sequence_time * num_operations
  # if num_macs == 0:
  #   return 0
  optimized_op_per_mac = math.ceil(num_operations / num_macs)
  optimized_time = sequence_time * optimized_op_per_mac

  #print("Optimized time:", optimized_time)
  return optimized_time

def total_macs(layer_operations, layer_sequences):

  total_accumulations = 0
  for i in range(len(layer_operations)):
    total_accumulations = total_accumulations + layer_operations[i] * layer_sequences[i]

  print("Total MAC operations:",total_accumulations)

  return total_accumulations

def intersection_finder(y1, y2, x):

  diff = y1 - y2
  sign_changes = np.where(np.diff(np.sign(diff)))[0]

  #Interpolate
  x_intersections = []
  y_intersections = []

  for idx in sign_changes:
    if idx == 0:
      continue
    x_interp = x[idx] - diff[idx] * (x[idx+1] - x[idx]) / (diff[idx+1] - diff[idx])
    y_interp = y1[idx] + (y1[idx+1] - y1[idx]) * ((x_interp - x[idx]) / (x[idx+1] - x[idx]))
    x_intersections.append(x_interp)
    y_intersections.append(y_interp)

  return x_intersections, y_intersections

def label_position(x , y, offset_angle, offset_v):

  max_index = np.where(x == 1024)[0]
  x_mid = (x[0] + x[max_index]) / 2
  idx1 = np.abs(x - x_mid).argmin()
  idx2 = idx1 + 1

  dx = x[idx2] - x[idx1]
  dy = y[idx2 + offset_angle] - y[idx1]
  angle = np.degrees(np.arctan2(dy, dx))

  print(angle)

  x_label = x[idx1]
  y_label = y[idx1] + offset_v

  return x_label, y_label, angle
