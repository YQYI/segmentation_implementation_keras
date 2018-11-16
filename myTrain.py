#!/usr/bin/env python
# coding=utf-8
from myBiSeNet import *
import numpy as np

model = create_BiSeNet(2)
x = np.asarray([np.random.rand(321, 321, 3)])
y = np.asarray([np.ones((321, 321, 3))])
print(x.shape, y.shape)
model.fit(x, y, epochs = 40, batch_size = 1)

