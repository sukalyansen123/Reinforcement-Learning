import numpy as np
import pylab as pl
import networkx as nx
import matplotlib.pyplot as plt 

nodes = input("Enter the number of nodes\n")
source = input("Enter the source node\n") 
goal = input("Enter the destination node\n")

color_map = []

#generating graph using erdos renyi model
G= nx.erdos_renyi_graph(nodes,0.3)

#adding different color for source and destination
for n in G:
		if n == source or n == goal:
				color_map.append('blue')
		else:
				color_map.append('red');

nx.draw(G,node_color = color_map, with_labels=True)

edge_list = G.edges()

for e in edge_list:
		print e

plt.figure()
plt.draw()
plt.show(block = False)

#size of q table
size = nodes

#initialize whole matrix with -1
R = np.matrix(np.ones(shape=(size,size)))
R *= -1

for edge in edge_list:
		print(edge)
		if edge[1] == goal:
				R[edge] = 100
		else:
				R[edge] = 0
		
		if edge[0] == goal:
				R[edge[::-1]] = 100
		else:
				R[edge[::-1]] = 0

R[goal,goal]=100

print R

Q = np.matrix(np.zeros([size,size]))

gamma = 0.8
learning_rate = 0.7

initial_state =1

def available_actions(state):
		cur_state = R[state,]
		available_act = np.where(cur_state >= 0)[1]
		return available_act

def sample_next_action(available_act):
		next_action = int(np.random.choice(available_act,1))
		return next_action

available_action = available_actions(initial_state)
action = sample_next_action(available_action)

def update(cur_state,action,gamma):
		max_index = np.where(Q[action,] == np.max(Q[action,]))[1]
  	
		if max_index.shape[0] > 1:
				max_index = int(np.random.choice(max_index, size = 1))
		else:
				max_index = int(max_index)
  	
		max_value = Q[action, max_index]
  	
		Q[cur_state, action] = Q[cur_state,action] + learning_rate*(R[cur_state, action] + gamma * max_value - Q[cur_state,action])
		print('max_value', R[cur_state, action] + gamma * max_value)
  
		if (np.max(Q) > 0):
				return(np.sum(Q/np.max(Q)*100))
		else:
				return (0)
    
update(initial_state, action, gamma)

scores =[]
for i in range(700):
		cur_state = np.random.randint(0,int(Q.shape[0]))
		av_act = available_actions(cur_state)
		action = sample_next_action(av_act)
		score = update(cur_state,action,gamma)
		scores.append(score)
		print ('Score:',str(score))
		
print("Trained Q matrix:")
print(Q/np.max(Q)*100)

current_state = source
steps = [current_state]

while current_state != goal:

		next_step_index = np.where(Q[current_state,] == np.max(Q[current_state,]))[1]
    
		if next_step_index.shape[0] > 1:
				next_step_index = int(np.random.choice(next_step_index, size = 1))
		else:
				next_step_index = int(next_step_index)
    
		steps.append(next_step_index)
		current_state = next_step_index

print("Most efficient path:")
print(steps)

plt.plot(scores)
plt.show()



