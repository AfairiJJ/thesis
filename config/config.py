from datetime import date
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import torch.nn as nn

train_common = './data/common_dataprep/train.pickle'
test_common = './data/common_dataprep/test.pickle'

train_ganprep_noei = "./data/gan_dataprep/train_gan.pickle"
scaler_noei = './data/gan_dataprep/scaler.pickle'

train_ganinput_ei = 'data/gan_dataprep/train_ganinput_ei.pickle'
train_ganinput_noei = 'data/gan_dataprep/train_gan_noei.pickle'

metadata = './config/metadata.json'

dtnow = date.today().strftime('%Y%m%d')
output_generator = f'./data/generators/generator_{dtnow}.pt'
output_discriminator = f'./data/discriminators/discriminator_{dtnow}.pt'
output_loss = f'./data/losses/loss_{dtnow}.csv'



vars = [
    "VehPower", "VehAge", "DrivAge", "Density", "BonusMalus", "Exposure",
    "EI_Density", "EI_DrivAge", "EI_BonusMalus1", "EI_BonusMalus2", "EI_VehAge",
    "GDV_Area", "GDV_VehAge", "GDV_DrivAge",
    "VehBrand", "VehGas", "Region", "Area",
"ClaimNb"]

cats_vars_xgb = [
             'VehBrand',
             'VehGas',
             'Region',
             'Area'
             ]

scaler_ei = MinMaxScaler # Cannot do different scaler since we need 0,1 (or we improve standard scaler to work!
continuous_activation = nn.LeakyReLU

num_samples = 500000
num_features = 60

seed = 1
testsize = 0.2
valsize = 0.2

# Parameters
epochs = 15000
batch_size = 128 # Article: 128, Code: 1500 (not good)
disc_epochs = 2 # Both
gen_epochs = 1 # Both
loss_penalty = 1 # Article: 10, Code: 1
gen_bn_decay = .9 # Article: 0.9, Code: 0.25
disc_leaky_param = 0.2 # Both

learning_rate = 0.01 # Article: 0.01, Code: 0.001 (not good)
hiddens_gen = [100,100,100]
hiddens_disc = [55, 55]
round_with_beginning_set = -1

l2_regularization = 0.00001 # Article: 0, # Code: 0.1
gen_l2_regularization = l2_regularization
noise_size = 60 # z-size, Code: 100
critic = True
disc_bn_decay = 0 if critic else 0.01 # Aritlce: 0.01, Code: 0.2

disc_l2_regularization = l2_regularization
disc_learning_rate = learning_rate
gen_learning_rate = learning_rate




# Generator tuning
cats_vars_gan = cats_vars_xgb + ['ClaimNb']



