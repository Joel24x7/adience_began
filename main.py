import tensorflow as tf
from train import test, train
from model import Began

if __name__=='__main__':
    model = Began()
    train(model, 1)
    test(model)