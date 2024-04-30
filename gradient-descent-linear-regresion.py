import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_regression
from matplotlib.animation import FuncAnimation
import matplotlib

# Use the 'TkAgg' backend if not set. Adjust based on your environment.
matplotlib.use('TkAgg')

# Generate a sample dataset
X, y, coef = make_regression(n_samples=100, n_features=1, noise=20.0, coef=True, random_state=42)
X = X.flatten()

# Parameters
learning_rate = 0.01
n_iterations = 500
update_speed = 100 # Update speed (milli seconds per frame)

# Initial coefficients
m = np.random.randn()
b = np.random.randn()

# Gradient descent function
def update_line(num, line):
    global m, b
    if num >= n_iterations:
        return line,
    
    # Calculate predictions
    y_pred = m * X + b
    
    # Calculate the gradients
    gradient_m = -2 * np.sum((y - y_pred) * X) / len(X)
    gradient_b = -2 * np.sum(y - y_pred) / len(X)
    
    # Update the parameters
    m -= learning_rate * gradient_m
    b -= learning_rate * gradient_b
    
    # Update the line
    line.set_ydata(m * X + b)
    plt.title(f'Iteration: {num+1}, Slope: {m:.3f}, Intercept: {b:.3f}')
    
    # Redraw the figure
    fig.canvas.draw()
    fig.canvas.flush_events()
    
    return line,

# Set up the figure, the axis, and the plot element
fig, ax = plt.subplots()
ax.scatter(X, y, color='blue', alpha=0.5, label='Data points')
line, = ax.plot(X, m * X + b, 'r-', linewidth=2, label='Fit line')
ax.set_xlabel("Feature X")
ax.set_ylabel("Target y")
ax.legend()

# Create the animation
ani = FuncAnimation(fig, update_line, frames=n_iterations, fargs=(line,),
                    interval=update_speed , blit=False, repeat=False)

plt.show()
