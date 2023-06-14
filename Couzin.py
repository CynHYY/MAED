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

mpl.use('TkAgg')
visualization = 'on'   # on/off
dimension = '3d'        # 2d/3d
n = 50               # number of agents
max_steps = 400         # maximum simulation steps
dt = 0.5                # simulation time step size
free_offset = 10        # free space margin (just for visualization)
number_of_alives = n    # initial number of alive individuals
number_of_sharks = 1    # number of sharks
number_of_foods = 0

# set the social behavior vs. threat-escaping behavior ratio
alpha = .5          # alpha = 1 means it only has social behavior and
                    # alpha = 0 means it only has escape behavior


#alpha_list = [0.5]
number_of_alives_list = []

# Couzin's repulsion/orientation/attraction radii (agent-agent interaction parameters)
r_r = 2
r_o = 10
r_a = 30

r_shark = 50

r_avoid_sea = 4
# agent-environment parameters
r_thr = 25      # zone of threat (individuals see threats closer than r_thr)
r_res = 100      # zone of resource (individuals see resources closer than r_thr)
r_lethal = 1    # individuals die if they are closer than r_lethal to any threat

r_chase = 20

field_of_view = 3*pi/2      # individuals'field of vision
field_of_view_shark = 3*pi/2   # sharks'field of vision

theta_dot_max = 0.5           # maximum angular velocity of the individuals
theta_dot_max_shark = .3    # maximum angular velocity of the sharks

constant_speed = 2          # translational velocity of the individuals
shark_speed = 3            # translational velocity of the sharks

area_width = 50   # x_max[m]
area_height = 50  # y_max[m]
area_depth = 50   # z_max[m]
sea_surface = 40
sea_floor = -20
sigma = 0.1
# np.seterr(divide='ignore', invalid='ignore')

mean_dist = np.zeros([max_steps, n]) #plots to show
shark_dist = np.zeros([max_steps, n])
mean_vel = np.zeros([max_steps, n])
mean_height = []
shark_height = []

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

    def update_position(self, delta_t):
        if self.is_alive == 1:
            self.pos = self.pos + self.vel * delta_t



def rotation_matrix_about(axis, theta):
    axis = np.asarray(axis)
    axis = axis / sqrt(np.dot(axis, axis))
    a = cos(theta / 2.0)
    b, c, d = -axis * sin(theta / 2.0)
    aa, bb, cc, dd = a * a, b * b, c * c, d * d
    bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
    return np.array([[aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)],
                     [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)],
                     [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc]])


if __name__ == '__main__':

    start = time.time()


    # initialize the variables
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

    # create 'swarm' and 'shark' and 'food' instances based on the Agent class
    [swarm.append(Agent(i, constant_speed)) for i in range(n)]
    [sharks.append(Agent(i, shark_speed)) for i in range(number_of_sharks)]
    [foods.append(Agent(i, food_speed)) for i in range(number_of_foods)]

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
        if t == 0 :
            for i in range(number_of_sharks):
                if sharks[i].pos[2] == min(sharks_pos[:,2]):
                    sharks[i].cir = 1
                    break

        t = t + 1
        mean_height.append(sum(swarm_pos[:,2])/n)
        if number_of_sharks >0: shark_height.append(sum(sharks_pos[:,2])/number_of_sharks)

        if visualization == 'on':
            ax.clear()
            q = ax.quiver(swarm_pos[:, 0], swarm_pos[:, 1], swarm_pos[:, 2],
                          swarm_vel[:, 0], swarm_vel[:, 1], swarm_vel[:, 2])

            q.set_array(swarm_color)
            ax.plot(sharks_pos[:, 0], sharks_pos[:, 1], sharks_pos[:, 2], 'o', color='#FF0000')
            for each_food in foods:
                ax.scatter(each_food.pos[0], each_food.pos[1], each_food.pos[2], color='#228B22')

            ax.set_xlim(-area_width - free_offset, area_width + free_offset)
            ax.set_ylim(-area_height - free_offset, area_height + free_offset)
            ax.set_zlim(-20-free_offset, area_depth-20 + free_offset)
            
            fig.canvas.draw()
            plt.pause(0.00000001)
            if (t == 0):
                plt.savefig('dis_t0.png')
        dist_list = []
        shark_dist_list = []
        vel_list = []
        for agent in swarm:
            d = np.array([0, 0, 0])
            d_social = [0, 0, 0]
            d_r = [0, 0, 0]
            d_o = [0, 0, 0]
            d_a = [0, 0, 0]
            d_thr = [0, 0, 0]
            d_res = [0, 0, 0]
            d_seasurf = np.array([0, 0, 0])
            d_left = np.array([0, 0, 0])
            d_right = np.array([0, 0, 0])
            d_front = np.array([0, 0, 0])
            d_back = np.array([0, 0, 0])
            
            total_dist = 0
            total_vel = 0

            for neighbor in swarm:
                if agent.id != neighbor.id:
                    r = neighbor.pos - agent.pos
                    r_normalized = r/norm(r)
                    norm_r = norm(r)
                    total_dist = total_dist + norm(r)
                    agent_vel_normalized = agent.vel/norm(agent.vel)
                    neighbor_vel_normalized = neighbor.vel / norm(neighbor.vel)
                    total_vel = total_vel + norm(agent.vel - neighbor.vel)
                    if norm(neighbor.pos - agent.pos) < r_a and acos(np.dot(r_normalized, agent_vel_normalized)) < field_of_view / 2:
                        if norm_r < r_r:
                            d_r = d_r - r_normalized
                        elif norm_r < r_o:
                            d_o = d_o + neighbor.vel/norm(neighbor.vel)
                        elif norm_r < r_a:
                            d_a = d_a + r_normalized
            if sea_surface - agent.pos[2] <= r_avoid_sea or sea_surface - agent.pos[2] <= 0:
                d_seasurf = np.array([0,0,1])
            if agent.pos[2]- sea_floor  <= r_avoid_sea or agent.pos[2]- sea_floor <= 0:
                d_seasurf = np.array([0,0,-1])

            if norm(d_r) != 0:
                d_social = d_r
            elif norm(d_a) != 0 and norm(d_o) != 0:
                d_social = (d_o + d_a)/2
            elif norm(d_a) != 0:
                d_social = d_o
            elif norm(d_o) != 0:
                d_social = d_a

            for each_shark in sharks:
                shark_dist_list.append(norm(agent.pos - each_shark.pos))
                if norm(agent.pos - each_shark.pos) <= r_thr:
                    d_thr = d_thr - (each_shark.pos - agent.pos)/norm(each_shark.pos - agent.pos) ** 2

            if norm(d_social) != 0:
                d = alpha * d_social / norm(d_social)
            if norm(d_thr) != 0:
                d = d + (1 - alpha) * d_thr / norm(d_thr)
            if norm(d_seasurf) != 0:
                d = d - d_seasurf/norm(d_seasurf)

                d = d + sigma * np.random.randn()
            if norm(d) != 0:
                z = np.cross(d/norm(d), agent.vel/norm(agent.vel))
                angle_between = asin(norm(z))
                if angle_between >= theta_dot_max*dt:
                    rot = rotation_matrix_about(z, theta_dot_max*dt)
                    agent.vel = np.asmatrix(agent.vel) * rot
                    agent.vel = np.asarray(agent.vel)[0]
                else:
                    rot = rotation_matrix_about(z, angle_between)
                    agent.vel = np.asmatrix(agent.vel) * rot
                    agent.vel = np.asarray(agent.vel)[0]
            dist_list.append(total_dist/(n-1))
            vel_list.append(total_vel/(n-1))
            
        mean_dist[t-1,:] = dist_list
        shark_dist[t-1,:] = shark_dist_list
        mean_vel[t-1, :] = vel_list
        
        for each_shark in sharks:
            if each_shark.cir == 1:
                d = np.array([0,0,0])
                d_s = [0,0,0]
                d_seasurf = [0,0,0]
                for prey in swarm:
                    r = prey.pos - each_shark.pos
                    r_norm = norm(r)
                    prey_vel_normalized = prey.vel / norm(prey.vel)
                    if r_norm < r_shark and acos(np.dot(r/r_norm, each_shark.vel/norm(each_shark.vel))) < field_of_view_shark / 2:
                        d = d + r/r_norm
                        d_s = d_s + prey_vel_normalized
                if sea_surface - each_shark.pos[2] <= r_avoid_sea * 2 or sea_surface - each_shark.pos[2] <= 0:
                    d_seasurf = np.array([0,0,1])
                if each_shark.pos[2]- sea_floor  <= r_avoid_sea or each_shark.pos[2]- sea_floor <= 0:
                    d_seasurf = np.array([0,0,-1])
                
                if norm(d_seasurf) != 0:
                    d = d-d_seasurf/norm(d_seasurf)
                
                d = d + sigma * np.random.randn()
                
                if d[2] <= 0 and norm(d) != 0:
                    dd = d + [0,0,2*d[2]]
                    z = np.cross(dd/norm(dd), each_shark.vel/norm(each_shark.vel))
                    angle_between = asin(norm(z))
                    if angle_between >= theta_dot_max_shark*dt:
                        rot = rotation_matrix_about(z, theta_dot_max_shark*dt)
                        each_shark.vel = np.asmatrix(each_shark.vel) * rot
                        each_shark.vel = np.asarray(each_shark.vel)[0]
                    else:
                        rot = rotation_matrix_about(z, angle_between)
                        each_shark.vel = np.asmatrix(each_shark.vel) * rot
                        each_shark.vel = np.asarray(each_shark.vel)[0]
                
                elif norm(d) != 0:
                    z = np.cross(each_shark.vel/norm(each_shark.vel), d/norm(d))
                    rot = rotation_matrix_about(z, pi/2)
                    dd = np.asmatrix(d) * rot
                    dd = np.asarray(dd)[0]
                    dd = np.asarray([0.7*dd[0] + 0.3*d[0], 0.7*dd[1] + 0.3 *d[1], 0.7*dd[2] + 0.3*d[2]])
                    z = np.cross(dd/norm(dd), each_shark.vel/norm(each_shark.vel))
                    angle_between = asin(norm(z))
                    if angle_between >= theta_dot_max_shark*dt:
                        rot = rotation_matrix_about(z, theta_dot_max_shark*dt)
                        each_shark.vel = np.asmatrix(each_shark.vel) * rot
                        each_shark.vel = np.asarray(each_shark.vel)[0]
                    else:
                        rot = rotation_matrix_about(z, angle_between)
                        each_shark.vel = np.asmatrix(each_shark.vel) * rot
                        each_shark.vel = np.asarray(each_shark.vel)[0]

        [agent.update_position(dt) for agent in swarm]
        [agent.update_position(dt) for agent in sharks]

        number_of_alives_list.append(number_of_alives)
        print('number_of_alives:', number_of_alives, '      time: ', time.time()-start)
        print('maxpos:', max(swarm_pos[:,2]), '      time: ', time.time()-start)
#         print(norm(sharks_vel[0,:]))
    # plotting indicator graphs
    graph = figure(title = "Mean Distance Between Individuals") 
     
    # name of the x-axis 
    graph.xaxis.axis_label = "Time"

    # name of the y-axis 
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


    graph = figure(title = "Distance Between Individuals and Predator") 

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

