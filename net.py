import numpy as np
import matplotlib.pyplot as plt


class network:

    def __init__(self, x , y):
        self.X = x
        self.Y = y
        self.Yh = np.zeros(self.Y.size)


        self.L = 2
        self.dims = [1, 15, 1]
        self.param = {}
        self.ch = {}
        self.grad = {}
        self.loss = []


        self.lr =0.005
        self.sam = self.X.size


    def initialize(self):

        np.random.seed(1)

        self.param['w1'] = np.random.randn(self.dims[1],self.dims[0])/np.sqrt(self.dims[0])

        self.param['b1'] = np.zeros((self.dims[1],1))

        self.param['w2'] = np.random.randn(self.dims[2],self.dims[1])/np.sqrt(self.dims[1])

        self.param['b2'] = np.zeros((self.dims[2], 1))

        return

    def forward(self, e):

        Z1 = self.param['w1'].dot(e) + self.param['b1']
        A1 = Relu(Z1)
        self.ch['z1'], self.ch['a1'] = Z1, A1



        Z2 = self.param['w2'].dot(A1) + self.param['b2']
        A2 = Sigmoid(Z2)
        self.ch['z2'], self.ch['a2'] = Z2, A2



        self.Yh = A2




        return self.Yh


    def nloss(self, Yh, ex):
        #loss = (1. / self.sam) * (-np.dot(ex, np.log(Yh)) - np.dot(1 - ex, np.log(1 - Yh)))

        loss = 0.5*(Yh - ex)**2
        return loss


    def backpass(self, e, ex):

        dLoss_Yh = -(ex-self.Yh)

        dLoss_z2 = dLoss_Yh * dSigmoid(self.ch['z2'])

        dLoss_a1 = np.dot(self.param['w2'].T, dLoss_z2)

        dLoss_w2 = 1./self.ch['a1'].shape[1]*np.dot(dLoss_z2, self.ch['a1'].T)

        dLoss_b2 = 1./self.ch['a1'].shape[1]*np.dot(dLoss_z2, np.ones([dLoss_z2.shape[1],1]))


        dLoss_z1 = dLoss_a1 * dRelu(self.ch['z1'])

        dLoss_a0 = np.dot(self.param['w1'].T,dLoss_z1)

        dLoss_w1 = np.dot(dLoss_z1,e)

        dLoss_b1 = np.dot(dLoss_z1, np.ones([dLoss_z1.shape[1],1]))


        self.param['w1'] = self.param['w1']-self.lr*dLoss_w1
        self.param['b1'] = self.param['b1']-self.lr*dLoss_b1
        self.param["w2"] = self.param["w2"] - self.lr * dLoss_w2
        self.param["b2"] = self.param["b2"] - self.lr * dLoss_b2

        print(self.param['w2'])












def Sigmoid(Z):
    return 1 / (1 + np.exp(-Z))

def Relu(Z):
    return np.maximum(0, Z)

def dRelu(x):
        x[x <= 0] = 0
        x[x > 0] = 1
        return x


def dSigmoid(Z):
        s = 1 / (1 + np.exp(-Z))
        dZ = s * (1 - s)
        return dZ




x = np.arange(1,11,0.1)
y = 2**-x


nn = network(x, y)
nn.initialize()




for i in range(100):

    Yh = nn.forward(x[i])
    nn.backpass(x[i], y[i])

newboi = np.arange(10,12,0.1)
newy = []

for i in range(20):
    Yh= nn.forward(newboi[i])
    newy.append(Yh)


print(newboi)
print("newy: " +str(newy))



plt.figure()
plt.subplot(211)
plt.plot(newboi, np.reshape(np.array(newy), (20,)))

plt.show()


















