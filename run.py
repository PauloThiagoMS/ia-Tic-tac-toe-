from pybrain.tools.shortcuts import buildNetwork
from pybrain.datasets import SupervisedDataSet
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.structure.modules import SoftmaxLayer
from pybrain.structure.modules import SigmoidLayer
rede = buildNetwork(2,3,1)
base = SupervisedDataSet(2,1)
base.addSample((0,0), (0,))
base.addSample((0,1), (1,))
base.addSample((1,0), (1,))
base.addSample((1,1), (0,))
treino = BackpropTrainer(rede, dataset= base, learningrate = 0.01, momentum = 0.1)
for i in range(1, 60000):
    erro = treino.train()
    if i% 1000 == 0:
        print("ER: %s" % erro)
print(rede.activate([0,0]))
print(rede.activate([1,0]))
print(rede.activate([0,1]))
print(rede.activate([1,1]))

