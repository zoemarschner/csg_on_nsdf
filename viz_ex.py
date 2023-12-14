import matplotlib.pyplot as plt
import numpy as np
import torch
import sys, pickle

from sdfplot import make_grid, plot_sdf_from_vals

CSG_LOSSES = ['Total', 'CSG', 'Eik', 'CP']
SV_LOSSES = ['Total', 'SV-', 'SV+', 'Eik', 'CP']

def plot_losses(name, save=False):
	try:
		losses = pickle.load(open(f'./data/{name}_losses.pl', 'rb'))

		np_loss = np.array(losses)
		fig, ax = plt.subplots()

		names = CSG_LOSSES if np_loss.shape[1] == len(CSG_LOSSES) + 1 else SV_LOSSES
		
		for l in range(1, np_loss.shape[1]):
			loss_name = names[l-1]
			ax.plot(np_loss[:,0], np_loss[:,l], label=loss_name)
		
		ax.set_yscale('log')
		ax.legend()
		
		if save:
			plt.savefig(f'./img/{name}_losses.png', bbox_inches='tight', dpi=300, transparent=True)
		plt.show()

	except FileNotFoundError:
		print("Couldn't find loss data")

def plot_neural_sdf(name, save=False, domain=None):
	if domain is None:
		domain = ([-2, 2], [-2, 2])
	model = torch.load(f'./models/{name}.pt', map_location=torch.device('cpu'))

	X, Y = make_grid(*domain, res=500)

	test_pts = torch.from_numpy(np.stack((X.flatten(), Y.flatten()), 1).astype('float32'))
	pred_Z = model(test_pts).detach().numpy()

	viz_fig, viz_ax = plt.subplots()
	plot_sdf_from_vals(X, Y, pred_Z, ax=viz_ax, N=50, alpha=1)
	
	plt.tight_layout()

	if save:
		plt.savefig(f'./img/{name}.png', bbox_inches='tight', dpi=300, transparent=True)

	plt.show()


if __name__ == "__main__":
	name = sys.argv[1]
	save = "-s" in sys.argv

	# parse domain
	domain = [float(d) for d in sys.argv[2:] if d != "-s"]
	if len(domain) == 4:
		domain = (domain[:2], domain[2:])
	else:
		domain = None

	plot_losses(name, save)
	plot_neural_sdf(name, save, domain=domain)
