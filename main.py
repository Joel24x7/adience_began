import tensorflow as tf
import sys
from train import test, train
from model import Began

if __name__=='__main__':

    args = sys.argv
    model = Began()

    if len(args) == 2:
        if args[1] == 'test':
            test(model, samples=15)
    else:
        train(model, epochs=100)
