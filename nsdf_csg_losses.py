import torch

###########################
# Combined Loss Functions #
###########################

"""
In general, loss function will take three commen arguments:
	pts:		the points at which the evaluate the loss
	model: 		the current sdf function that the loss is being evaluated on. In typical use, this would be
				the neural network that is being trained
	inp_vals:	the value of pts in the input to the problem, which is the Pseudo-SDF for the CSG problem and
				is the spacetime function for the swept volume problem
The hyperparameters of the losses are:
	strength:	the three editing losses include sigmoid approximations of the step function, this strength indicates
				the steepness of that sigmoid
	lambdas:	these arguments to the combined losses control the weighting of each loss term
Additional parameters come more from programming necessity. A description of a few
important ones follows.
	dim:			this describes the spatial dimension of the probelm (e.g. 2 or 3 typically) and is used when
					working with parametric neural SDFs in order to ensure the gradient is only taken in the _spatial_
					coordinates.
	pre_eval:		the results of evaluting the points in the model is resused to avoid wasted computation, this argument
					handles passing it around if nessecary
	verbose_return:	this argument of the combined losses causes the function to return the seperated weighted 
					loss values, instead of just the sum, which is useful for tracking the losses during training.
"""

def swept_volume_combined_loss(str_neg, str_pos, lambdas, dim=2):
	def inner(pts, model, inp_vals, verbose_return=False):
		ptsXY = pts[:, :-1]
		ptsXY.requires_grad = True
		pre_eval = model(ptsXY)

		E_SV_neg = swept_volume_negative(pts, model, inp_vals, pre_eval=pre_eval, strength=str_neg)
		E_SV_pos = swept_volume_positive(pts, model, inp_vals, pre_eval=pre_eval, strength=str_pos)

		E_CP, pre_eval_grad = closest_point_loss(ptsXY, model, return_grad=True, pre_eval=pre_eval, dim=dim)
		E_eik = eikonal_loss(ptsXY, model, pre_eval=pre_eval, pre_eval_grad=pre_eval_grad, dim=dim)

		SV_neg_loss = (lambdas[0] * E_SV_neg).mean()
		SV_pos_loss  = (lambdas[1] * E_SV_pos).mean()
		eik_loss = (lambdas[2] * E_CP).mean()
		cp_loss   = (lambdas[3] * E_eik).mean()

		loss = SV_neg_loss + SV_pos_loss + eik_loss + cp_loss

		if verbose_return:
			return loss, (SV_neg_loss, SV_pos_loss, eik_loss, cp_loss)
		else:
			return loss

	return inner

def csg_combined_loss(strength, lambdas, dim=2):
	def inner(pts, model, inp_vals, verbose_return=False):
		pts.requires_grad = True
		pre_eval = model(pts)
		
		E_csg = csg(pts, model, inp_vals, pre_eval=None, strength=strength)

		E_CP, pre_eval_grad = closest_point_loss(pts, model, return_grad=True, pre_eval=pre_eval, dim=dim)
		E_eik = eikonal_loss(pts, model, pre_eval=pre_eval, pre_eval_grad=pre_eval_grad, dim=dim)

		csg_loss = (lambdas[0] * E_csg).mean()
		eik_loss = (lambdas[1] * E_CP).mean()
		cp_loss   = (lambdas[2] * E_eik).mean()

		loss = csg_loss + eik_loss + cp_loss

		if verbose_return:
			return loss, (csg_loss, eik_loss, cp_loss)
		else:
			return loss

	return inner

###########################
#      Editing Losses     #
###########################

def swept_volume_negative(pts, model, inp_vals, pre_eval=None, strength=300):
	if pre_eval is None:
		pre_eval = model(pts[:, :-1])

	neg_sign = torch.logical_not(torch.sign(inp_vals)+1)
	
	S = torch.nn.Sigmoid()
	return neg_sign*S(strength*(pre_eval - inp_vals))

def swept_volume_positive(pts, model, inp_vals, pre_eval=None, strength=300):
	if pre_eval is None:
		pre_eval = model(pts[:, :-1])

	S = torch.nn.Sigmoid()
	return S(-strength*pre_eval)

def csg(pts, model, inp_vals, pre_eval=None, strength=300):
	if pre_eval is None:
		pre_eval = model(pts)

	bound_val = inp_vals
	sgn = torch.sign(bound_val)
	   
	S = torch.nn.Sigmoid()
	return S(-strength*sgn*(pre_eval - bound_val))


###########################
#  Regularization Losses  #
###########################

def closest_point_loss(pts, model, return_grad=False, pre_eval=None, dim=2):
	if pre_eval is None:
		pts.requires_grad = True
		pre_eval = model(pts)

	# compute gradient 
	grad_total, = torch.autograd.grad(pre_eval, pts, grad_outputs=torch.ones(pre_eval.shape, device=pts.device), create_graph=True, retain_graph=True)
	graddx = grad_total[:, :dim]

	# calculate mapped points
	inp_mapped = torch.cat((pts[:, :dim] - pre_eval[:, :dim] * graddx, pts[:, dim:]), axis=1)
	
	# compute value of field at mapped points
	mapped_vals = model(inp_mapped)
	ans = (mapped_vals**2)

	if return_grad:
		return ans, graddx
	return ans

def eikonal_loss(pts, model, pre_eval=None, pre_eval_grad=None, dim=2):
	if pre_eval is None or pre_eval_grad is None:
		pts.requires_grad = True
		pre_eval = model(pts)

		grad_total, = torch.autograd.grad(pre_eval, pts, grad_outputs=torch.ones(pre_eval.shape, device=pts.device), create_graph=True, retain_graph=True)
		pre_eval_grad = grad_total[:, :dim]

	return ((1 - (pre_eval_grad**2).sum(1))**2)[:, None]


