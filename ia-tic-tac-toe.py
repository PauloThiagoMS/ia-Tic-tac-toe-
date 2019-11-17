from pybrain.tools.shortcuts import buildNetwork
from pybrain.datasets import SupervisedDataSet
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.structure.modules import SoftmaxLayer, SigmoidLayer
from pybrain.tools.customxml import NetworkWriter, NetworkReader


neural_network = buildNetwork(10,11,11,2)

data_base = SupervisedDataSet(10, 2)

# Start cases
# (começar em um canto)
# ia is fisth
data_base.addSample((3, 
0,0,0, 
0,0,0,
0,0,0), (0, 1))
data_base.addSample((3,
0,0,0,
0,0,0,
0,0,0), (0, 3))
data_base.addSample((3,
0,0,0,
0,0,0,
0,0,0), (0, 7))
data_base.addSample((3,
0,0,0,
0,0,0,
0,0,0), (0, 9))
# (começar em um canto se o inimigo marcou no meio)
# ia not fisth
data_base.addSample((3,
0,0,0,
0,2,0,
0,0,0), (0, 1))
data_base.addSample((3,
0,0,0,
0,2,0,
0,0,0), (0, 3))
data_base.addSample((3,
0,0,0,
0,2,0,
0,0,0), (0, 7))
data_base.addSample((3,
0,0,0,
0,2,0,
0,0,0), (0, 9))

# 2 cases
# (marcar no meio se o inimigo  nao iniciou no meio)
data_base.addSample((3,
2,0,0,
0,0,0,
0,0,0), (0, 5))
data_base.addSample((3,
0,2,0,
0,0,0,
0,0,0), (0, 5))
data_base.addSample((3,
0,0,2,
0,0,0,
0,0,0), (0, 5))
data_base.addSample((3,
0,0,0,
2,0,0,
0,0,0), (0, 5))
data_base.addSample((3,
0,0,0,
0,0,2,
0,0,0), (0, 5))
data_base.addSample((3,
0,0,0,
0,0,0,
2,0,0), (0, 5))
data_base.addSample((3,
0,0,0,
0,0,0,
0,2,0), (0, 5))
data_base.addSample((3,
0,0,0,
0,0,0,
0,0,2), (0, 5))

# 3 cases
# (marcar do outro lado caso inimigo 2 marque no meio)
data_base.addSample((2,
1,0,0,
0,2,0,
0,0,0), (0, 9))
data_base.addSample((2,
0,0,1,
0,2,0,
0,0,0), (0, 7))
data_base.addSample((2,
0,0,0,
0,2,0,
1,0,0), (0, 3))
data_base.addSample((2,
0,0,0,
0,2,0,
0,0,1), (0, 1))

# (nao marcar em um canto caso inimigo marke do outro lado)
data_base.addSample((2,
2,0,0,
0,1,0,
0,0,2), (0, 2))
data_base.addSample((2,
2,0,0,
0,1,0,
0,0,2), (0, 4))
data_base.addSample((2,
2,0,0,
0,1,0,
0,0,2), (0, 6))
data_base.addSample((2,
2,0,0,
0,1,0,
0,0,2), (0, 8))

data_base.addSample((2,
0,0,2,
0,1,0,
2,0,0), (0, 2))
data_base.addSample((2,
0,0,2,
0,1,0,
2,0,0), (0, 4))
data_base.addSample((2,
0,0,2,
0,1,0,
2,0,0), (0, 6))
data_base.addSample((2,
0,0,2,
0,1,0,
2,0,0), (0, 8))

# 4 cases
# (preparar armadilha)
data_base.addSample((2,
1,2,0,
0,0,0,
0,0,0), (0, 7))
data_base.addSample((2,
1,0,2,
0,0,0,
0,0,0), (0, 7))
data_base.addSample((2,
1,0,0,
2,0,0,
0,0,0), (0, 3))
data_base.addSample((2,
1,0,0,
0,0,2,
0,0,0), (0, 7))
data_base.addSample((2,
1,0,0,
0,0,0,
2,0,0), (0, 3))
data_base.addSample((2,
1,0,0,
0,0,0,
0,2,0), (0, 3))
data_base.addSample((2,
1,0,0,
0,0,0,
0,0,2), (0, 3))
data_base.addSample((2,
1,0,0,
0,0,0,
0,0,2), (0, 7))
data_base.addSample((2,
2,0,1,
0,0,0,
0,0,0), (0, 9))
data_base.addSample((2,
0,2,1,
0,0,0,
0,0,0), (0, 9))
data_base.addSample((2,
0,0,1,
2,0,0,
0,0,0), (0, 9))
data_base.addSample((2,
0,0,1,
0,0,2,
0,0,0), (0, 1))
data_base.addSample((2,
0,0,1,
0,0,0,
2,0,0), (0, 1))
data_base.addSample((2,
0,0,1,
0,0,0,
2,0,0), (0, 9))
data_base.addSample((2,
0,0,1,
0,0,0,
0,2,0), (0, 1))
data_base.addSample((2,
0,0,1,
0,0,0,
0,0,2), (0, 1))
data_base.addSample((2,
2,0,0,
0,0,0,
1,0,0), (0, 9))
data_base.addSample((2,
0,2,0,
0,0,0,
1,0,0), (0, 9))
data_base.addSample((2,
0,0,2,
0,0,0,
1,0,0), (0, 9))
data_base.addSample((2,
0,0,2,
0,0,0,
1,0,0), (0, 1))
data_base.addSample((2,
0,0,0,
2,0,0,
1,0,0), (0, 9))
data_base.addSample((2,
0,0,0,
0,0,2,
1,0,0), (0, 1))
data_base.addSample((2,
0,0,0,
0,0,0,
1,2,0), (0, 1))
data_base.addSample((2,
0,0,0,
0,0,0,
1,0,2), (0, 1))
data_base.addSample((2,
2,0,0,
0,0,0,
0,0,1), (0, 3))
data_base.addSample((2,
2,0,0,
0,0,0,
0,0,1), (0, 7))
data_base.addSample((2,
0,2,0,
0,0,0,
0,0,1), (0, 7))
data_base.addSample((2,
0,0,2,
0,0,0,
0,0,1), (0, 7))
data_base.addSample((2,
0,0,0,
2,0,0,
0,0,1), (0, 3))
data_base.addSample((2,
0,0,0,
0,0,2,
0,0,1), (0, 7))
data_base.addSample((2,
0,0,0,
0,0,0,
2,0,1), (0, 3))
data_base.addSample((2,
0,0,0,
0,0,0,
0,2,1), (0, 3))


# 5 cases
# (fexando armadilha armadilha)
data_base.addSample((1,
1,2,0,
2,0,0,
1,0,0), (0, 9))
data_base.addSample((1,
1,0,2,
2,0,0,
1,0,0), (0, 9))
data_base.addSample((1,
1,2,1,
2,0,0,
0,0,0), (0, 9))
data_base.addSample((1,
1,0,0,
2,0,2,
1,0,0), (0, 5))
data_base.addSample((1,
1,2,1,
0,0,0,
2,0,0), (0, 9))
data_base.addSample((1,
1,2,1,
0,0,0,
0,2,0), (0, 5))
data_base.addSample((1,
1,2,1,
0,0,0,
0,0,2), (0, 7))
data_base.addSample((1,
1,0,0,
2,0,0,
1,0,2), (0, 3))
data_base.addSample((1,
2,0,1,
0,0,2,
0,0,1), (0, 7))
data_base.addSample((1,
0,2,1,
0,0,2,
0,0,1), (0, 7))
data_base.addSample((1,
0,0,1,
2,0,2,
0,0,1), (0, 5))
data_base.addSample((1,
1,2,1,
0,0,2,
0,0,0), (0, 7))
data_base.addSample((1,
1,2,1,
0,0,0,
2,0,0), (0, 9))
data_base.addSample((1,
0,0,1,
0,0,2,
2,0,1), (0, 1))
data_base.addSample((1,
1,2,1,
0,0,0,
0,2,0), (0, 5))
data_base.addSample((1,
1,2,1,
0,0,0,
0,0,2), (0, 7))
data_base.addSample((1,
2,0,0,
0,0,0,
1,2,1), (0, 3))
data_base.addSample((1,
0,2,0,
0,0,0,
1,2,1), (0, 5))
data_base.addSample((1,
0,0,2,
0,0,0,
1,2,1), (0, 1))
data_base.addSample((1,
1,0,2,
2,0,0,
1,0,0), (0, 9))
data_base.addSample((1,
0,0,0,
2,0,0,
1,2,1), (0, 3))
data_base.addSample((1,
1,0,0,
2,0,2,
1,0,0), (0, 5))
data_base.addSample((1,
1,0,0,
2,0,0,
1,2,0), (0, 3))
data_base.addSample((1,
1,0,0,
2,0,0,
1,0,2), (0, 3))
data_base.addSample((1,
2,0,1,
0,0,2,
0,0,1), (0, 7))
data_base.addSample((1,
2,0,0,
0,0,0,
1,2,1), (0, 3))
data_base.addSample((1,
0,2,0,
0,0,0,
1,2,1), (0, 5))
data_base.addSample((1,
0,0,2,
0,0,0,
1,2,1), (0, 1))
data_base.addSample((1,
0,0,1,
2,0,2,
0,0,1), (0, 5))
data_base.addSample((1,
0,0,0,
0,0,2,
1,2,1), (0, 1))
data_base.addSample((1,
0,0,1,
0,0,2,
2,0,1), (0, 1))
data_base.addSample((1,
0,0,1,
0,0,2,
0,2,1), (0, 1))

neural_network = NetworkReader.readFrom('filename.xml') 
treino = BackpropTrainer(neural_network, dataset= data_base, learningrate = 0.01, momentum = 0.5)
for i in range(1, 50000):
    erro = treino.train()
    if erro < 0.56:
        print("ER: %s" % erro)
        break

NetworkWriter.writeToFile(neural_network, 'filename.xml')
print(neural_network.activate([3,0,0,0,0,0,0,0,0,0]))