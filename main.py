import tensorflow as tf
import sys
from train import test, train
from model import Began

'''
main.py
Default run trains model
Add 'test' command line argument to test instead
'''

if __name__=='__main__':

    args = sys.argv
    model = Began()

    if len(args) == 2:
        if args[1] == 'test':
            print('Testing Model')
            test(model, samples=15)
    else:
        print('Training Model')
        train(model, epochs=12)
