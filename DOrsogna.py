#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
from numpy.linalg import *
from math import *
import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
from cycler import cycler
import time
import pylab
from bokeh.plotting import figure, output_file, show 
from bokeh.palettes import magma

# from moviepy.editor import VideoClip
# from moviepy.video.io.bindings import mplfig_to_npimage

mpl.use('TkAgg')
visualization = 'on'   # on/off
dimension = '3d'        # 2d/3d
max_steps = 400

area_width = 50   # x_max[m]
area_height = 50  # y_max[m]
area_depth = 50   # z_max[m]
sea_surface = 40
sea_floor = -20
beta = 0.3
a = 1
b = 0.2


constant_speed = 2          # translational velocity of the individuals
shark_speed = 3            # translational velocity of the sharks
food_speed = 0             # translational velocity of the resources
field_of_view = 3*pi/2      # individuals'field of vision


phi = 0.5
dt = 0.5
n = 50
free_offset = 10        # free space margin (just for visualization)
number_of_alives = n    # initial number of alive individuals
number_of_sharks = 0    # number of sharks
number_of_foods = 0

number_of_alives_list = []

mean_dist = np.zeros([max_steps, n]) #plots to show
shark_dist = np.zeros([max_steps, n])
mean_vel = np.zeros([max_steps, n])
mean_height = []
shark_height = []

cs = 5
ls = 3

cthre = 12
lthre = 25
cr = 10
lr = 2
ca = 20
la = 30
gma = 0.25
sigma = 0.1
# np.seterr(divide='ignore', invalid='ignore')

np.random.seed(2)

class Agent:
    def __init__(self, agent_id, speed):
        self.id = agent_id
        self.pos = np.array([0, 0, 0])
        self.pos[0] = np.random.uniform(0, area_width)
        self.pos[1] = np.random.uniform(0, area_height)
        self.pos[2] = np.random.uniform(sea_floor+15, sea_surface - 15)
        self.vel = np.random.uniform(-1, 1, 3)
        self.is_alive = 1
        self.cir = 0
        if dimension == '2d':
            self.pos[2] = 0
            self.vel[2] = 0
        self.vel = self.vel / norm(self.vel) * speed
if __name__ == '__main__':
    start = time.time()
    swarm = []
    sharks = []
    foods = []
    swarm_pos = np.zeros([n, 3])
    swarm_vel = np.zeros([n, 3])
    swarm_color = np.zeros(n)
    sharks_pos = np.zeros([number_of_sharks, 3])
    sharks_vel = np.zeros([number_of_sharks, 3])
    d = np.array([0, 0, 0])
    d_social = np.array([0, 0, 0])
    d_r = np.array([0, 0, 0])
    d_o = np.array([0, 0, 0])
    d_a = np.array([0, 0, 0])
    d_thr = np.array([0, 0, 0])
    d_res = np.array([0, 0, 0])
    t = 0

    # create 'swarm' and 'shark' instances based on the Agent class
    [swarm.append(Agent(i, constant_speed)) for i in range(n)]
    [sharks.append(Agent(i, shark_speed)) for i in range(number_of_sharks)]

    # initialize the figure
    if visualization == 'on':
        fig = plt.figure()
    if dimension == '3d':
        ax = fig.add_subplot(projection='3d')
    else:
        ax = fig.gca()

    # begin the simulation loop
    number_of_alives = n

    while t < max_steps:

        for i in range(len(swarm)):
            swarm_pos[i, :] = swarm[i].pos
            swarm_vel[i, :] = swarm[i].vel
            swarm_color[i] = 2 - swarm[i].is_alive
        for i in range(number_of_sharks):
            sharks_pos[i, :] = sharks[i].pos
            sharks_vel[i, :] = sharks[i].vel

        t = t + 1
        mean_height.append(sum(swarm_pos[:,2])/n)
        if number_of_sharks >0: shark_height.append(sum(sharks_pos[:,2])/number_of_sharks)

        if visualization == 'on':
            ax.clear()
            q = ax.quiver(swarm_pos[:, 0], swarm_pos[:, 1], swarm_pos[:, 2],
                          swarm_vel[:, 0], swarm_vel[:, 1], swarm_vel[:, 2])

            q.set_array(swarm_color)
            ax.plot(sharks_pos[:, 0], sharks_pos[:, 1], sharks_pos[:, 2], 'o', color='#FF0000')

            ax.set_xlim(-area_width -free_offset , area_width * 2 + free_offset)
            ax.set_ylim(-area_height-free_offset , area_height * 2 + free_offset)
            ax.set_zlim(-20-free_offset, area_depth-20 + free_offset)

            flist.append(fig)

            plt.pause(0.00000001)
            plt.show(block=False)
    #         if (t == 1):
    #             plt.savefig('t0.png')
        dist_list = []
#         shark_dist_list = []
        vel_list = []
        for agent in swarm:
            u = np.array([0, 0, 0])
            w = np.array([0, 0, 0])
            s = np.array([0, 0, 0])
            total_dist = 0
            total_vel = 0
            if agent.is_alive:
                for neighbor in swarm:
                    if agent.id != neighbor.id and neighbor.is_alive:
                        xij = agent.pos - neighbor.pos
                        xij_norm = norm(xij)
                        total_dist = total_dist + xij_norm
                        xij_normalized = xij/xij_norm
                        vij = neighbor.vel - agent.vel
                        agent_vel_normalized = agent.vel/norm(agent.vel)
                        total_vel = total_vel+norm(vij)
                        if acos(np.dot(xij_normalized, agent_vel_normalized)) < field_of_view / 2:
                            u_prime = ca/la*exp(-xij_norm/la) - cr/lr * exp(-xij_norm/lr)
                            wij = 1/((1+xij_norm**2)**gma)
                            w = w + wij * vij
                            u = u + u_prime * xij / xij_norm
                for shark in sharks:
                    xij = agent.pos - shark.pos
                    shark_dist_list.append(norm(xij))
                    s = s + (cthre/lthre * exp(-xij_norm/lthre))* xij / norm(xij)

                dist_list.append(total_dist/(n-1))
                vel_list.append(total_vel/(n-1))

                sea_r = np.array([0,0,agent.pos[2] - sea_surface])
                surf = (cs/ls) * exp(-norm(sea_r)/ls) * abs(sea_r)/norm(sea_r)
                sea_f = np.array([0,0,agent.pos[2] - sea_floor])
                flor = (cs/ls) * exp(-norm(sea_f)/ls) * abs(sea_f)/norm(sea_f)
                dvdt = 0
                if number_of_sharks > 0:
                    dvdt = (w-u)/n + s/number_of_sharks - (surf - flor) + (0.45 - 0.2 * (norm(agent.vel)**2))*agent.vel
                else:
                    dvdt= (w-u)/n - (surf - flor) + (0.45 - 0.2 * (norm(agent.vel)**2))*agent.vel
                dxdt = agent.vel
                agent.pos = agent.pos + dxdt*dt
                agent.vel = agent.vel + dvdt*dt + sigma * np.random.randn()
        mean_dist[t-1,:] = dist_list
#         shark_dist[t-1,:] = shark_dist_list
        mean_vel[t-1, :] = vel_list

        for shark in sharks:
            gx = np.array([0,0,0])
            gv = np.array([0,0,0])
            center = np.array([0,0,0])
            under_count = 0
            for agent in swarm:
                xij = shark.pos - agent.pos
                xij_norm = norm(xij)
                if xij_norm == 0: break
                xij_normalized = xij / xij_norm
                vij = agent.vel - shark.vel
                shark_vel_normalized = shark.vel/norm(shark.vel)
                if acos(np.dot(xij_normalized, shark_vel_normalized)) < field_of_view / 2:
                    u_prime = 40/50*exp(-xij_norm/50) - 10/7*exp(-xij_norm/7)

                    gx = gx + u_prime * xij_normalized
                    if xij[2] > -2 :
                        under_weight = 40/30*exp(-xij_norm/30)
                        under_count = under_count + 1
                        gv = gv + under_weight * abs(np.array([0,0,xij_normalized[2]]))

            sea_r = np.array([0,0,shark.pos[2] - sea_surface])
            surf = (cs/ls) * exp(-norm(sea_r)/ls) * abs(sea_r)/norm(sea_r)
            sea_f = np.array([0,0,shark.pos[2] - sea_floor])
            flor = (cs/ls) * exp(-norm(sea_f)/ls) * abs(sea_f)/norm(sea_f)
            if under_count > 0:
                dvdt = - gx/n -gv/under_count + (a - b * (norm(shark.vel)**2))*shark.vel - (surf - flor)
            else:
                dvdt = - gx/n + (a - b * (norm(shark.vel)**2))*shark.vel - (surf - flor)

            dxdt = shark.vel
            shark.pos = shark.pos + dxdt*dt
            shark.vel = shark.vel + dvdt*dt + sigma * np.random.randn()
    
    graph = figure(title = "Mean Distance Between Individuals") 
     
    graph.xaxis.axis_label = "Time"
    graph.yaxis.axis_label = "Mean Distance"

    tspan = np.linspace(0, max_steps/2 - dt, max_steps)
    xs = [tspan]*n
    ys = [mean_dist[:,i] for i in range(n)]
    # color of the lines
    line_color = magma(50)

    # plotting the graph 
    graph.multi_line(xs, ys, line_color = line_color) 

    # displaying the model 
    show(graph)
    
    graph = figure(title = "Distance Between Individuals and Shark") 
     
    # name of the x-axis 
    graph.xaxis.axis_label = "Time"

    # name of the y-axis 
    graph.yaxis.axis_label = "Distance"

    tspan = np.linspace(0, max_steps/2 - dt, max_steps)
    xs = [tspan]*n
    ys = [shark_dist[:,i] for i in range(n)]
    # color of the lines
    line_color = magma(50)

    # plotting the graph 
    graph.multi_line(xs, ys, line_color = line_color) 

    # displaying the model 
    show(graph)
    
    graph = figure(title = "Mean Difference Between Individuals' Velocities") 
     
    
    # name of the x-axis 
    graph.xaxis.axis_label = "Time"

    # name of the y-axis 
    graph.yaxis.axis_label = "Norm Velocity Difference"

    tspan = np.linspace(0, max_steps/2 - dt, max_steps)
    xs = [tspan]*n
    ys = [mean_vel[:,i] for i in range(n)]
    # color of the lines
    line_color = magma(50)

    # plotting the graph 
    graph.multi_line(xs, ys, line_color = line_color) 

    # displaying the model 
    show(graph)
    
    graph = figure(title = "Mean z-coordinates of Preys") 
     
    # name of the x-axis 
    graph.xaxis.axis_label = "Time"

    # name of the y-axis 
    graph.yaxis.axis_label = "z-coordinates"

    if number_of_sharks != 0:
        tspan = np.linspace(0, max_steps/2 - dt, max_steps)
        xs = [tspan]*2
        ys = [mean_height,shark_height]
        # color of the lines
        line_color = ['black', 'red']

        # plotting the graph 
        graph.multi_line(xs, ys, line_color = line_color) 
    else:
        tspan = np.linspace(0, max_steps/2 - dt, max_steps)
        xs = [tspan]
        ys = [mean_height]
        # color of the lines
        line_color = ['black']

        # plotting the graph 
        graph.multi_line(xs, ys, line_color = line_color) 
    # displaying the model 
    show(graph)

