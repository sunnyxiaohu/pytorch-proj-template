#distributed=False
srun -p Platform -n1 --gres=gpu:1 python finetuning_torchvision_models_tutorial.py
srun -p Platform -n1 --gres=gpu:1 python finetuning_torchvision_models_tutorial_org.py # set distributed=False

#distributed=True
srun -p Platform -n8 --gres=gpu:8 python finetuning_torchvision_models_tutorial.py # set distributed=True

