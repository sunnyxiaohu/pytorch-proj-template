import data
from opts import args
import torch
import model
import loss
import utils.utility as utility
from trainer import Trainer


ckpt = utility.checkpoint(args)
loss = loss.Loss(args, ckpt) if not (args.test_only or args.evaluate_only) else None
model = model.Model(args, ckpt)
loader = data.Data(args)

trainer = Trainer(args, model, loss, loader, ckpt)

n = 0
while not trainer.terminate():
	n += 1
	trainer.train()
	if args.test_every!=0 and n%args.test_every==0:
		trainer.test()	
