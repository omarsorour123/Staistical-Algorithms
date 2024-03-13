import numpy as np
from LogisticRegression import LogisticRegression
# Generate some random data
np.random.seed(0)
X_train = np.random.randn(2, 500)
Y_train = np.random.randint(0, 2, (1, 500))

# Split the data into train and test sets
X_test = X_train[:, :100]
Y_test = Y_train[:, :100]
X_train = X_train[:, 100:]
Y_train = Y_train[:, 100:]

# Reshape Y_train and Y_test
Y_train = Y_train.reshape(1, -1)
Y_test = Y_test.reshape(1, -1)


print("X_train shape:", X_train.shape)
print("Y_train shape:", Y_train.shape)
print("X_test shape:", X_test.shape)
print("Y_test shape:", Y_test.shape)
lr = LogisticRegression()
lr.fit(X_train,Y_train)