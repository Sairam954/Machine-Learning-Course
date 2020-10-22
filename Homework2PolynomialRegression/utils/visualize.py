import matplotlib.pyplot as plt
import numpy as np

def plot_predicted(x,y,theta_list,order,order_color,plot_title):

    plt.figure()

    X_grid = np.arange(min(x), max(x), 0.1)
    X_grid = X_grid.reshape(len(X_grid),1)
    label_holder = []
    modelno = 0
    for theta in theta_list:
        line = theta[0]
        label_holder.append('%.*f' % (2, theta[0]))
        for i in np.arange(1, len(theta)):
            line += theta[i] * X_grid ** i
            # label_holder.append(' + ' + '%.*f' % (2, theta[i]) + r'$x^' + str(i) + '$')
        plt.plot(X_grid, line, color=order_color[str(order[modelno])], label='Order'+str(order[modelno]))

        modelno+=1

    plt.scatter(x, y, s=30, c='blue')
    plt.title(plot_title)
    plt.xlabel('Input X')
    plt.ylabel('Output Y')
    plt.legend()
    plt.show()

