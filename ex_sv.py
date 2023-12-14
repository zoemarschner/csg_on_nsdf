import torch

from sdfpy import star_sdf, spacetime_single_cubic
from sdfsampler import DataGenerator
from sdfpytorch import DeepSDF, fit
from nsdf_csg_losses import swept_volume_combined_loss

#+------------------------------------------------------------------------+#
#|                        ~~~~~~HYPERPARAMS ~~~~~~                        |#
batch_size 			= 50000
epochs 				= 600000
report 				= 1000
save   				= epochs
step_size 			= 1e-4
samp_sigma 			= 1e-2
samp_amb_percent 	= 0.3
reuse_data_epochs 	= 20

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
#+------------------------------------------------------------------------+#

hyper_params_dict = {"epochs": epochs, "step_size": step_size, "report": report, "save": save, "device": device, "reuse_data_epochs": reuse_data_epochs}
samp_hyper_params_dict = {"NUM_PTS": batch_size, "gaussian_sigma": samp_sigma, "percent_ambient": samp_amb_percent}

# 1. Set up input: spacetime function of shape moving along sweep path for SV
star = lambda pts, center: star_sdf(pts, center, 0.3, 2, device=device) 
cub_ctr_pts = torch.tensor([[2,2,-1,2,1,0]]).to(device)
cubic_star = spacetime_single_cubic(star, cub_ctr_pts)

# 2. Set up data generation function, for creating sample points during training
DG = DataGenerator([-2, -2, 0], [2, 2, 1], device)
data_gen = lambda model: DG.and_input_eval(DG.importance_and_stratified(model, cubic_star, **samp_hyper_params_dict), cubic_star)

# 3. Intialize the neural network
model = DeepSDF(8, 128, n_dim=2).to(device)

# 4. Define the loss function
loss_fn = swept_volume_combined_loss(300, 300, (10, 0.5, 0.3, 0.3), dim=2)

# 5. Train! 
test_name = "swept_star"
fit(model, loss_fn, data_gen, test_name, **hyper_params_dict)
 
# 6. After training, visualize the results by running `python3 viz_ex.py swept_star -1.05 2.05 -0.9 2.2` in the command line!