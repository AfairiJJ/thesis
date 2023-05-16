from _1_DataPrep import *
from config.config import *
from trainer import trainnn

# GAN without expert input
if False:
    train, test = prepare_alldata()
    train, ss = prepare_gandata(train, cont_vars_noei)
    trainnn()


# GAN with expert input
if True:
    train, test = prepare_alldata()
    train, test = add_expertinput(train, test)
    train, ss = prepare_gandata(train, cont_vars_ei)
    trainnn(train, ss)