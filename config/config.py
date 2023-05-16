from datetime import date

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

num_samples = 500000
num_features = 60
batch_size = 100
noise_size = 128

cont_vars_noei = ['VehPower',
                 'VehAge',
                 'DrivAge',
                 'DensityGLM',
                 'BonusMalus',
                 'Exposure']

cont_vars_ei = cont_vars_noei + [
    'EI_Density',
    'EI_DrivAge',
    'EI_BonusMalus1',
    'EI_BonusMalus2',
    'EI_VehAge',
    'GDV_Area',
    'GDV_VehAge',
    'GDV_DrivAge'
]

seed = 1

testsize = 0.2
valsize = None