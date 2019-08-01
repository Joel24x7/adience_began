import tensorflow as tf
from train import test, train
from model import Began

if __name__=='__main__':
    #100 epochs
    #Stopping and starting training seems to help improve sample quality
    # for i in range(10):
    model = Began()
    train(model, epochs=100)
    test(model, samples=15)