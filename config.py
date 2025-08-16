import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# MoCoV2 Parameters
BASE_LR = 1e-4
HEAD_LR = 1e-3
MOCO_N_EPOCHS = 25
MOCO_BATCH_SIZE = 4
MOCO_MODEL_LOC = "moco_v2.pt" # Change with your model path

# Finetuning Parameters
FINETUNING_BATCH_SIZE = 4
FINETUNING_N_EPOCHS = 400
G_LEARNING_RATE = 3e-4
D_LEARNING_RATE = 5e-5
UNFREEZE_EPOCHS = 150
UNFREEZE_EPOCHS2 = 250
GENERATOR_LOC = "generator_model.pt"
DISCRIMINATOR_LOC = "discriminator_model.pt"