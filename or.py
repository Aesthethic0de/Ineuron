from utils.model import Perceptron
from utils.all_utils import prepare_data
import pandas as pd
import numpy as np


OR = {
    "X1" : [0,0,1,1],
    "X2" : [0,1,0,1],
    "X3" : [0,1,1,1],
}
df = pd.DataFrame(OR)
df
X,y = prepare_data(df)
ETA = 0.3
EPOCHS = 10
model = Perceptron(eta = ETA, epochs = EPOCHS)
model.fit(X,y)
_ = model.total_loss()

