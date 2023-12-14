import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

def make_sdf_cm():
	sdf_cm_colors =  [(0, '#ffffff'), (0.5 + 1e-5, '#489acc'), (0.5 + 1e-5, '#fffae3'), (1, '#ff7424')]

	cmap = mpl.colors.LinearSegmentedColormap.from_list('SDF', sdf_cm_colors, N=256)
	return SDF_cmap(0.95, 'sdf_cmap', cmap._segmentdata)


"""
This class overrides the Colormap class and does some preprocesseing before the main call to the 
colormap to add alternating darkening to the isolines. It is only designed to work with data plotted
using mpl's countour/countourf methods. (eg. through plot_sdf below)
"""
class SDF_cmap(mpl.colors.LinearSegmentedColormap):
	def __init__(self, alph, *args, **kwargs):
		super().__init__(*args, **kwargs)
		self.alph = alph
		
	def set_contours(self, cont):
		self.cont = cont
		
	def set_min_max(self, minv, maxv):
		self.minv = minv
		self.maxv = maxv
		
	def __call__(self, X, **kwargs):
		X = X.squeeze()
		n_c = X.shape[0]
		Xp = (1/2) *( (-X + 1)*(self.minv/self.maxv) + X + 1)
		colors = super().__call__(Xp, **kwargs)
		i = np.minimum(np.floor(X * (n_c)), n_c - 1);
		mask_is = np.repeat((i % 2 == 1)[:, None], 4, axis=1).squeeze()
		mult_mask = (np.ones(colors.shape)).squeeze()
		mult_mask[mask_is] = self.alph
		mult_mask[:, 3] = 1
		return mult_mask * colors
	
def plot_sdf(sdf, X, Y, **kwargs):
	pts = np.stack((X.flatten(), Y.flatten()), 1)
	return plot_sdf_from_vals(X, Y, sdf(pts), **kwargs)

def plot_sdf_from_vals(X, Y, Z, ax=None, colorbar=True, cmap=None, N=25, alpha=1, levels=None, return_levels=False):
	if cmap is None:
		cmap = make_sdf_cm()
	
	Z = np.reshape(Z, X.shape)
	
	if ax is None:
		fig = plt.figure()
		ax = fig.gca()
	else:
		fig = plt.gcf()

	if levels is None:
		levels = mpl.ticker.MaxNLocator(N).tick_values(np.min(Z), np.max(Z))

	try:
		# If this is an SDF_cmap, want to set its min/max
		cmap.set_min_max(levels[0], levels[-1])  
	except:
		pass

	cp = ax.contourf(X, Y, Z, levels, cmap=cmap, alpha=alpha)

	if colorbar:
		fig.colorbar(cp, ax=ax) 

	ax.set_aspect('equal', 'box')

	if return_levels:
		return cp, levels
	return cp


def make_grid(*dims, res=100):
	return np.meshgrid(*[np.linspace(*dim, res) for dim in dims])

