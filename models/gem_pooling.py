import torch.nn as nn


class GeneralizedMeanPooling(nn.Module):
	def __init__(self, p):
		super().__init__()
		self.p = p

	def forward(self, x):
		if self.p != 1.:
			mean = x.clamp(min=1e-6).pow(self.p).mean(dim=(2,3))
			return mean.pow(1./self.p)
		else:
			return x.mean(dim=(2,3))