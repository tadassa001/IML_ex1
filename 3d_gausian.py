import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.linalg import qr

mean = [0, 0, 0]
cov = np.eye(3) # identity matrix form size 3X3
x_y_z = np.random.multivariate_normal(mean, cov, 50000).T




def get_orthogonal_matrix(dim):
    H = np.random.randn(dim, dim)
    Q, R = qr(H)
    return Q


def plot_3d(x_y_z):
    '''
    plot points in 3D
    :param x_y_z: the points. numpy array with shape: 3 X num_samples (first dimension for x, y, z
    coordinate)
    '''
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x_y_z[0], x_y_z[1], x_y_z[2], s=1, marker='.', depthshade=False)
    ax.set_xlim(-5, 5)
    ax.set_ylim(-5, 5)
    ax.set_zlim(-5, 5)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')

    plt.show()



def plot_2d(x_y):
    '''
    plot points in 2D
    :param x_y_z: the points. numpy array with shape: 2 X num_samples (first dimension for x, y
    coordinate)
    '''
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(x_y[0], x_y[1], s=1, marker='.')
    ax.set_xlim(-5, 5)
    ax.set_ylim(-5, 5)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    plt.show()

# #11
# plot_3d(x_y_z)

# #12
# s = np.diag([0.1, 0.5, 2])
# scaled_matrix = np.dot(s,x_y_z)
# # how to answer: "how do the covariance matrix looks like now"??
# plot_3d(scaled_matrix)
#
# #13
# rnd_orth_mat = get_orthogonal_matrix(3) # orthogonal mat 3X3
# new_mat = np.dot(rnd_orth_mat, scaled_matrix)
# # how to answer: "how do the covariance matrix looks like now"??
# plot_3d(new_mat)
#
# #14
# # projection of data on x_y axes
# x_y = np.array([x_y_z[0], x_y_z[1]]) #todo: check if fine
# plot_2d(x_y)
#
# #15
# x_in_range = []
# y_in_range = []
# for i in range(len(x_y_z[2])):
#     if 0.1 > x_y_z[2][i] > -0.4:
#         # take the relevant x;y coordinates
#         x_in_range.append(x_y_z[0][i])
#         y_in_range.append(x_y_z[1][i])
#
# x_y_in_range = np.array([x_in_range, y_in_range])
# plot_2d(x_y_in_range)
# # x_y_in_range = np.array([[x_y_z[0][i]], [x_y_z[1][i]], [x_y_z[2][i]]
# #                                   for i in range(len(x_y_z[2]))
# #                                   if 0.1 > x_y_z[2][i] > -0.4])
#
#
#16 helper:

def running_ave(data, axis):
    cumsum = np.cumsum(data, axis=axis)
    return cumsum/np.arange(1, 1001)

# 16
#16.1
tosses_seqs = np.random.binomial(1, 0.25, (100000, 1000))

mat = np.array([tosses_seqs[0],
                tosses_seqs[1],
                tosses_seqs[2],
                tosses_seqs[3],
                tosses_seqs[4]])

# now each cell in last col contains the ave
indexes = np.arange(1,1001) # 1<=m<=1000
# est_by_m = running_ave(mat, 1) # running average
#
# fig = plt.figure()
# ax = fig.add_subplot(111)
# colors = ['r', 'b', 'g', 'k', 'y']
# for i in range(len(est_by_m)):
#     ax.scatter(indexes, est_by_m[i], s=1, marker='.', c=colors[i])
# ax.set_xlim(1, 1000)
# ax.set_ylim(-0.1, 1)
# ax.set_xlabel('m')
# plt.show()

#16.2
epsilons = [0.5, 0.25, 0.1, 0.01, 0.001]

for epsilon in epsilons:

    chebishev = []
    hoffding = []
    # we dont know the p parameter X~Ber(p), but we know that Var(X)<=1/4
    for i in range(1, len(tosses_seqs[0])+1):
        cheb = (1/(4*i*epsilon*epsilon))
        hoff = (-2*i*epsilon*epsilon) # inside the exp
        if cheb > 1:
            cheb = 1
        chebishev.append(cheb)
        hoffding.append(hoff)

    chebishev = np.array(chebishev)
    hoffding = np.exp(np.array(hoffding)) * 2
    hoffding[hoffding > 1] = 1

    fig = plt.figure()
    ax = fig.add_subplot(111)
    for i in range(len(chebishev)):
        ax.scatter(indexes, chebishev, s=1, marker='.', c='r')
        ax.scatter(indexes, hoffding, s=1, marker='.', c='b')
    ax.set_xlim(1, 1000)
    ax.set_ylim(-0.01, 1.1)
    ax.set_xlabel('m')
    plt.title(str(epsilon))#, str='center'
    plt.show()

#16.3


# mat = np.array([[1,2,3,4], [5,6,7,8]])
# arr = [1,2,3,4,5]
# nArray = np.array(arr)
# print(nArray)
# print(np.cumsum(nArray))
