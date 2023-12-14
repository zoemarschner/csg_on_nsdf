import torch

#####################################
# Simple Analytic SDFs              #
# (Formulations from Inigo Quillez) #
#####################################

def circle_sdf(pts, center, r, device='cpu'):
	dist = pts - center
	return torch.sqrt(torch.sum(torch.pow(dist, 2), dim=1)[:, None]) - r


def square_sdf(pts, center, dims, device='cpu'):
	shift_pt = torch.abs(pts - center)
	pt_dist = shift_pt - dims/2

	main_res = torch.max(pt_dist, dim=1).values

	corners = torch.min(pt_dist, dim=1).values >= 0
	corner_res = torch.sqrt(torch.sum(pt_dist[corners]**2, dim=1))

	main_res[corners] = corner_res
	return main_res[:, None]


def star_sdf(pts, center, r, angle, device='cpu'):
	pts = pts - center
	k1 = torch.tensor([0.809016994375, -0.587785252292], device=device)
	k2 = torch.tensor([-k1[0], k1[1]], device=device)
	xpts = torch.abs(pts[:, 0:1])
	pts = torch.cat((xpts, pts[:, 1:]), dim=1)

	pts -= 2*torch.clamp(torch.sum(pts*k1, dim=1), min=0).repeat(2,1).T * k1
	pts -= 2*torch.clamp(torch.sum(pts*k2, dim=1), min=0).repeat(2,1).T * k2

	xpts = torch.abs(pts[:, 0:1])
	ypts = pts[:, 1:] - r
	pts = torch.cat((xpts, ypts), dim=1)

	ba = angle * torch.tensor([-k1[1], k1[0]], device=device) - torch.tensor([0,1], device=device)
	innerh = torch.sum(pts*ba, dim=1)/torch.sum(ba*ba)
	h = torch.clamp(torch.clamp(innerh, max=r), min=0)

	final_dot = torch.sqrt(torch.sum(torch.pow(pts-ba*h.repeat(2,1).T, 2), axis=1))
	res = final_dot * torch.sign(pts[:,1] * ba[0] - pts[:,0]*ba[1])
	return res[:, None]


#####################################
# SDF Transformations               #
#####################################

def spacetime_single_cubic(shape_SDF, control_point):
	def inner_spacetime_single_cubic(inp):
		t = inp[:, -1:]
		center = 3 * (1-t)**2 * t * control_point[:, 0:2] + 3 * (1-t) * t ** 2 * control_point[:, 2:4] + t**3 * control_point[:, 4:6]
		return shape_SDF(inp[:, :2], center)
	return inner_spacetime_single_cubic

def union(sdf1, sdf2):
	return lambda pts: torch.minimum(sdf1(pts), sdf2(pts))

def intersection(sdf1, sdf2):
	return lambda pts: torch.maximum(sdf1(pts), sdf2(pts))

def negate(sdf):
	return lambda pts: -sdf(pts)

