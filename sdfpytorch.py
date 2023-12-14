import pickle, sys, time, datetime
import torch
from torch import nn

class DeepSDF(nn.Module):
	def __init__(self, num_layers, hidden_size, n_dim=2, activation=nn.ReLU):
		super().__init__()
		self.layers = nn.ModuleList([nn.Linear(n_dim, hidden_size), activation()])
		
		for _ in range(num_layers-1):
			self.layers.append(nn.Linear(hidden_size, hidden_size))
			self.layers.append(activation())
		
		self.layers.append(nn.Linear(hidden_size, 1))
				
	def forward(self, features):
		out = features
		
		for layer in self.layers:
			out = layer(out)        
		return out


def fit(model, loss_fn, data_generator, name, device='cpu', reuse_data_epochs=10, step_size=1e-4, epochs=10000, report=1000, save=1000):
	print(f'Training {name} for {epochs} epochs on {device}')

	optim = torch.optim.Adam(model.parameters(), lr=step_size)
	losses = []
	start = time.time()

	# -------------------- training loop -----------------------
	for i in range(epochs):
		if i % reuse_data_epochs == 0:
			with torch.no_grad():
				pts, inp_vals = data_generator(model)

		loss, loss_items = loss_fn(pts, model, inp_vals, verbose_return=True)

		loss.backward()
		optim.step()
		optim.zero_grad()

		if i % report == 0:
			losses.append((i, loss.item(), *[l.item() for l in loss_items]))

			col_width = 15
			print(f'Epoch {i}:'.ljust(col_width), 'loss'.ljust(col_width), {losses[-1][1]})
			print(''.ljust(col_width), 'itemized loss'.ljust(col_width), [l.item() for l in loss_items])

			elapsed = time.time() - start 
			est_time = (elapsed * epochs)/(i+1) - elapsed
			print(''.ljust(col_width), 'pred time left'.ljust(col_width), str(datetime.timedelta(seconds=round(est_time))))

		if i % save == 0 and i != 0:
			torch.save(model, f'./models/{name}_{i}.pt')
			
			with open(f'./data/{name}_losses.pl', 'wb') as loss_file:
				pickle.dump(losses, loss_file)

	print(f'Finished {name} in {time.time() - start}')

	# -------------------- output -----------------------
	with open(f'./data/{name}_losses.pl', 'wb') as loss_file:
		pickle.dump(losses, loss_file)

	torch.save(model, f'./models/{name}.pt')

	print("Done!")

