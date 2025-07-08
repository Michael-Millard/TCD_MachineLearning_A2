# ------------------------------------------------------------------------ #

# CS7CS4 Machine Learning
# Week 2 Assignment
#
# Name:         Michael Millard
# Student ID:   24364218
# Due date:     05/10/2024

# ------------------------------------------------------------------------ #

# Imports

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score

# ------------------------------------------------------------------------ #

# Read in data and set labels

# NB: dataset csv file must be in same directory as this solution
labels = ["X1", "X2", "y"]
df = pd.read_csv("ML_W2_Dataset.csv", names=labels)
print("Dataframe head:")
print(df.head())

# Split data frame up into X and y 
X1 = df["X1"]
X2 = df["X2"]
X = np.column_stack((X1, X2))
y = df["y"]

# ------------------------------------------------------------------------ #

# (a)(i)

# Create scatter plot ('+' for +1 labels, '-' for -1 labels)
plt.figure(figsize=(8, 6))
plt.scatter(X1[y == 1], X2[y == 1], marker='+', color='lime', label='y = +1')
plt.scatter(X1[y == -1], X2[y == -1], marker='o', color='blue', label='y = -1')

# Label axes and add legend
plt.xlabel('X1')
plt.ylabel('X2')
plt.legend(loc=1) # TR corner

# Save and show the plot
plt.savefig("scatter_lin_reg_a_i.png")
plt.show()

# ------------------------------------------------------------------------ #

# (a)(ii)

# Logistic regression model (no penalty)
model = LogisticRegression(penalty=None, solver='lbfgs')
model.fit(X, y)
theta0 = model.intercept_.item()
theta1, theta2 = model.coef_.T
theta1, theta2 = theta1.item(), theta2.item() # Convert from np.array to float
print("\nLogistic regression params:")
print("theta0, theta1, theta2 = %8f, %8f, %8f"%(theta0, theta1, theta2))

# ------------------------------------------------------------------------ #

# (a)(iii)

# Create scatter plot ('+' for +1 labels, '-' for -1 labels)
plt.figure(figsize=(8, 6))

# Actual target values
plt.scatter(X1[y == 1], X2[y == 1], marker='+', color='lime', label='y_targ = +1')
plt.scatter(X1[y == -1], X2[y == -1], marker='o', color='blue', label='y_targ = -1')

# Add the trained logistic regression model's predictions on the training data to the plot
y_pred = model.predict(X)

# Predicted values
plt.scatter(X1[y_pred == 1], X2[y_pred == 1], marker='+', color='red', label='y_pred = +1')
plt.scatter(X1[y_pred == -1], X2[y_pred == -1], marker='o', color='purple', label='y_pred = -1')

# Decision boundary x2 = m * x1 + c
m = -theta1 / theta2
c = -theta0 / theta2
y_db = m * X1 + c
plt.plot(X1, y_db, color='green', linewidth=2)
print("Decision boundary params: m, c = %8f, %8f"%(m, c))

# Label axes and add legend
plt.xlabel('X1')
plt.ylabel('X2')
plt.legend(loc=1)

# Save and show the plot
plt.savefig("scatter_db_lin_reg_a_iii.png")
plt.show()

# ------------------------------------------------------------------------ #

# (a)(iv)

print("\nLogistic regression accuracy = %4.2f"%(accuracy_score(y, y_pred)))

# ------------------------------------------------------------------------ #

# (b)(i) and (ii)

print("\nSVM params for various penalty param (C) values:")

# Create subplots grid
num_rows, num_cols = 2, 3
fig, axes = plt.subplots(nrows=num_rows, ncols=num_cols, figsize=(12, 8), sharex=True, sharey=True) # Share x- and -axes between plots
axes = axes.ravel() # Convert axes to 1D array

# Sweep through range of C values [0.001, 100], increasing C by 10x each iteration (6 iterations)
# At each step, train the model with the new penalty parameter using the same training data
# Add subplot at each step
for i in range(0, 6):
    C_ = 0.001 * (10**i) # Underscore to differentiate between this var and LinearSVC param 'C'
    model = LinearSVC(C=C_).fit(X, y)
    theta0 = model.intercept_.item()
    theta1, theta2 = model.coef_.T
    theta1, theta2 = theta1.item(), theta2.item() 
    thetaVec = np.array((theta0, theta1, theta2))

    axes[i].scatter(X1[y == 1], X2[y == 1], marker='+', color='lime', label='y_targ = +1')
    axes[i].scatter(X1[y == -1], X2[y == -1], marker='o', color='blue', label='y_targ = -1')
    
    # Add the trained logistic regression model's predictions on the training data to the plot
    y_pred = model.predict(X)
    axes[i].scatter(X1[y_pred == 1], X2[y_pred == 1], marker='+', color='red', label='y_pred = +1')
    axes[i].scatter(X1[y_pred == -1], X2[y_pred == -1], marker='o', color='purple', label='y_pred = -1')
    
    # Decision boundary x2 = m * x1 + c
    m = -theta1 / theta2
    c = -theta0 / theta2
    y_db = m * X1 + c
    axes[i].plot(X1, y_db, color='green', linewidth=2)

    # Accuracy
    accuracy = accuracy_score(y, y_pred)
    print("%1d. C, theta0, theta1, theta2, penalty, accuracy = %7.3f, %8f, %8f, %8f, %8f, %8f"%(i, C_, theta0, theta1, theta2, (np.dot(thetaVec, thetaVec))/C_, accuracy))
    
    # Label axes and add legend
    axes[i].title.set_text('C = %7.3f'%(C_))
    # Only set labels on bottom row for x-axis and left col for y-axis
    if (i / num_cols >= (num_rows - 1)):
        axes[i].set_xlabel('X1')
    if (i % num_cols == 0):
        axes[i].set_ylabel('X2')
    axes[i].legend(loc=1)
   
# Save and show the plot
plt.savefig("scatter_plots_svm_b_ii.png")
plt.show()

# ------------------------------------------------------------------------ #

# (c)(i)

print("\nNew logistic regression params (4 input features):")

# Create new input features and stack them
X3 = X1**2
X4 = X2**2
X = np.column_stack((X1, X2, X3, X4))

# Logistic Regression
model = LogisticRegression(penalty=None, solver='lbfgs')
model.fit(X, y)
theta0 = model.intercept_.item()
theta1, theta2, theta3, theta4 = model.coef_.T
theta1, theta2, theta3, theta4 = theta1.item(), theta2.item(), theta3.item(), theta4.item()
print("theta0, theta1, theta2, theta3, theta4 = %8f, %8f, %8f, %8f, %8f"%(theta0, theta1, theta2, theta3, theta4))

# ------------------------------------------------------------------------ #

# (c)(ii)

# Create scatter plot ('+' for +1 labels, '-' for -1 labels)
plt.figure(figsize=(8, 6))
plt.scatter(X1[y == 1], X2[y == 1], marker='+', color='lime', label='y_targ = +1')
plt.scatter(X1[y == -1], X2[y == -1], marker='o', color='blue', label='y_targ = -1')

# Add the trained logistic regression model's predictions on the training data to the plot
y_pred = model.predict(X)
plt.scatter(X1[y_pred == 1], X2[y_pred == 1], marker='+', color='red', label='y_pred = +1')
plt.scatter(X1[y_pred == -1], X2[y_pred == -1], marker='o', color='purple', label='y_pred = -1')

# Accuracy
accuracy = accuracy_score(y, y_pred)
print("Accuracy = %8f"%(accuracy))

# Label axes and add legend
plt.xlabel('X1')
plt.ylabel('X2')
plt.legend(loc=1) # Upper right

# Save and show the plot
plt.savefig("scatter_new_lin_reg_c_ii.png")
plt.show()

# ------------------------------------------------------------------------ #

# (c)(iii)

# Performance of most common class predictor
most_common_classifier = DummyClassifier(strategy="most_frequent")
most_common_classifier.fit(X, y)
y_base = most_common_classifier.predict(X)
print("\nMost common accuracy = %4.2f"%(accuracy_score(y, y_base)))

# Performance of logistic regression model
print("Logistic regression accuracy = %4.2f"%(accuracy_score(y, y_pred)))

# ------------------------------------------------------------------------ #

# (c)(iv)

# Create scatter plot ('+' for +1 labels, '-' for -1 labels)
plt.figure(figsize=(8, 6))
plt.scatter(X1[y == 1], X2[y == 1], marker='+', color='lime', label='y_targ = +1')
plt.scatter(X1[y == -1], X2[y == -1], marker='o', color='blue', label='y_targ = -1')

# Add the trained logistic regression model's predictions on the training data to the plot
plt.scatter(X1[y_pred == 1], X2[y_pred == 1], marker='+', color='red', label='y_pred = +1')
plt.scatter(X1[y_pred == -1], X2[y_pred == -1], marker='o', color='purple', label='y_pred = -1')

# Decision boundary 2: x2 = a*x1^2 + b*x1 + c
a = -theta3 / theta2
b = -theta1 / theta2
c = -theta0 / theta2
print("a = %8f, b = %6f, c = %8f"%(a, b, c))

# Find x_range
d = c + 1
x_lower = (-b + np.sqrt(b**2 - 4*a*d))/(2*a)
x_upper = (-b - np.sqrt(b**2 - 4*a*d))/(2*a)
x_lower, x_upper = x_lower.item(), x_upper.item()
print("x_lower = %8f, x_upper = %8f"%(x_lower, x_upper))

# Create linspaced points to plot decision boundary
X_range = np.linspace(x_lower, x_upper, 100)
y_db = a * X_range ** 2 + b * X_range + c
plt.plot(X_range, y_db, color='green', linewidth=2)

# Label axes and add legend
plt.xlabel('X1')
plt.ylabel('X2')
plt.legend(loc=1) # Upper right

# Save and show the plot
plt.savefig("scatter_db_new_lin_reg_c_iii.png")
plt.show()

# ------------------------------------------------------------------------ #