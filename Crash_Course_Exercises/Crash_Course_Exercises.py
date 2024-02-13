
#task 1

from random import randint
import matplotlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split   


#task 2
np.random.seed(101) #doesnt matter what kind of param you pass (literally!)
                    #but if u want to keep the same sequence of rand numbers
                    #keep the param THE SAME
random_array = np.random.randint(0,10)
print(random_array)

#task 3

pass_ =np.array(np.random.randint(1,101, (100,5)))
print(pass_)
plt.imshow(pass_, cmap='grey', aspect= 'auto')
plt.colorbar()
plt.title('Image Title')

#task 4
data_frame = pd.DataFrame(pass_)
print(data_frame)
#task 5
data_frame.plot.scatter(0, 1, color = 'green', marker = 'o', s = 25)
plt.xlabel("x label")
plt.ylabel("y label")
plt.title("Array")


#task  6
scaler_model = MinMaxScaler()
type(scaler_model)
scaler_model.fit(pass_)
print(scaler_model.transform(pass_))


#task 7
data_frame.columns = ['f1', 'f2', 'f3', 'f4', 'label']
print(data_frame)

Top_x = data_frame[['f1', 'f2', 'f3', 'f4']]
y_label = data_frame['label']
x_train, x_test, y_train, y_test = train_test_split(Top_x, y_label, test_size=0.4, random_state=32)
print(x_train.shape)
print(x_test.shape)

plt.show()


