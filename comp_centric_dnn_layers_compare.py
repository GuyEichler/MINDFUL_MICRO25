#from run_networks import *
from scaling.scaling_layers_comp import *
from settings.soc_struct import *
from settings.socs_to_test import *
from socs_intersections import *
from socs_intersections_layers import *

if __name__ == "__main__":

  temp = [1.0, 1.0]

  first_normalized = np.divide(list_layers_1, list_1)
  second_normalized = np.divide(list_layers_2, list_2)

  third_normalized = np.array(temp)

  fourth_normalized = np.array(temp)

  fifth_normalized = np.array(temp)

  sixth_normalized = np.divide(list_layers_6[0], list_6[0])
  sixth_normalized = np.append(sixth_normalized, 1.0)

  seventh_normalized = np.divide(list_layers_7[0], list_7[0])
  seventh_normalized = np.append(seventh_normalized, 1.0)

  eighth_normalized = np.divide(list_layers_8[0], list_8[0])
  eighth_normalized = np.divide(eighth_normalized, 1.0)

  fontscale = 1

  categories = ['MLP', 'DN-CNN']

  all_normalized = list()

  all_normalized.append(first_normalized)
  all_normalized.append(second_normalized)
  all_normalized.append(third_normalized)
  all_normalized.append(fourth_normalized)
  all_normalized.append(fifth_normalized)
  all_normalized.append(sixth_normalized)
  all_normalized.append(seventh_normalized)
  all_normalized.append(eighth_normalized)

  x = np.arange(len(categories)) * 1

  mlp_list = list()
  dense_list = list()


  bar_width = 0.1
  width = 10
  height = 4
  plt.figure(1, figsize=(width,height))
  for i in range(8):

    bars = plt.bar(x - (3.5-i)*bar_width+i*0.005, all_normalized[i], width=bar_width, label=str(i+1), color=colors[i])#, edgecolor='black', linewidth=0.3)


  plt.figure(1)
  plt.axhline(y=1, color='red', linestyle='--', linewidth=3, label='Original')
  plt.xlabel('DNN Architecture', fontsize=18*fontscale)
  plt.ylabel('Normalized #Channels', fontsize=18*fontscale)
  plt.xticks(x, categories, fontsize=18*fontscale)
  plt.yticks(fontsize=18*fontscale)
  plt.ylim(0.5,1.5)
  plt.legend(fontsize=18*fontscale,loc='upper center', bbox_to_anchor=(0.5, 1.2), ncol=9,columnspacing=0.5, frameon=False, handlelength=0.8)
  plt.tight_layout()
  plt.savefig(f"figures/compare_layers_scaling_socs.pdf")

  plt.show()
