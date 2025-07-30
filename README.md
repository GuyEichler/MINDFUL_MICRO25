# MINDFUL_MICRO25
This repo contains the artifact for the MINDFUL paper accepted to MICRO 2025

The artifact includes the analytical framework that was developed to investigate the scaling of implantable brain-computer interfaces.

The framwork is written in Python and requires the installation of the following packages:

`matplotlib`\
`numpy`\
`scipy`\
`shapely`\
`math`\
`sys`\
`os`\
`sympy`


Before running the MINDFUL framework follow the next steps:
1) Install all the packages from above in your Python environment
2) `python create_dirs.py` - creates the directories data, logs and figures

The Python scripts at the top level of the repository generate the results presented in the paper:
1) `comm_centric_ook.py` generates the data for figures 5 and 6 - `comm_centric_ook_show.py` creates the subfigures of figures 5 and 6
   -  The script accepts as argument one of two options: `naive` or `high_margin` \
      `python comm_centric_ook.py naive` generates the subfigures that match the Naive Design in figures 5 and 6 \
      `python comm_centric_ook.py high_margin` generates the subfigures that match the High-Margin Design in figures 5 and 6
2) `comm_centric_qam.py` generates the data for figure 7 - `comm_centric_qam_show.py` creates figure 7
3) `comp_centric_dnn.py` generates the data for figure 10 - `comm_centric_dnn_show.py` creates figure 10
4) `comp_centric_dnn_layers.py` generates the data for figure 11 - `comm_centric_dnn_layers_compare.py` creates figure 11
5) `comp_centric_optimization.py` generates the data for figure 12 - `comp_centric_optimization.py` creates figure 12
