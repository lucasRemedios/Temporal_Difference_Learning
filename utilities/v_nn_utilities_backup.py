
import random
import copy
import numpy as np
import sys
sys.path.insert(0, './td_hrr')
import hrr 

"""
global A
global G 
global goal #index that has the reward
A = 1    #learning rate
G = 0.5    #discount
"""

def nn( idx, w, cur_states ):
	dot_prod = np.dot(cur_states[idx], w)
	bias = 1
	return ( dot_prod + bias )

def input_layer_activation( inp_layer_list ):
	return inp_layer_list	

def vals_at_states(w, cur_states):
	for i in range(len(cur_states)):
		print( nn(i, w, cur_states)  )
	print("\n\n")

#====================================================================
# Update And Return New Value For v[s] 
def TD_Delta ( w, r, position, g_state, ep_num, next_position, cur_s, G, A, goal_idx ):	

	print("WEIGHTS UPDATING...")	

	#Non-Terminal	
	delta =  r + ( G* (nn(next_position, w, cur_s)) ) - nn(position, w, cur_s) 
	for i in range(len(w)):
		w[i] +=  A * delta * cur_s[position][i] 

	#Terminal
	if position == goal_idx:
		print("************* TERMINAL STATE ********************") 
		g_state = True
		delta = r - nn(position, w, cur_s) 
		for i in range(len(w)):		
			w[i] += A * delta * cur_s[position][i] 

	return w, g_state

#====================================================================
# Decide Whether To Go Left Or Right
# Return Index of Left or Right Position ie. Index Of Next Position
def Policy( cur_pos, weights, ep_num, num_s, cur_s):
	left = (cur_pos-1)	
	if left < 0: left = num_s - 1 #wrap around	
	right = (cur_pos+1) % num_s #also handles wrapping	

	print("IN POLICY 1/2 Current Position:", cur_pos)

	#go left
	if nn(left, weights, cur_s) > nn(right, weights, cur_s): 	
		print("IN POLICY 2/2")
		print("left INDEX" , left,":", nn(left, weights, cur_s), "right INDEX" , right,":", nn(right, weights, cur_s))
		print("Next position is:", left)
		print("-----------> LEFT")
		return left
	
	#go right
	elif nn(right, weights, cur_s) > nn(left, weights, cur_s): 
		print("IN POLICY 2/2")
		print("right INDEX" ,right,":", nn(right, weights, cur_s), "left INDEX" ,left, ":", nn(left, weights, cur_s))
		print("Next position is:", right)
		print("-----------> RIGHT")
		return right

	#pick random direction if both directions have same value	
	else:
		print("----------->RANDOM DIRECTION") 
		return random.choice([ left, right ]) 

#====================================================================
"""
num_states = int(input("Enter the number of states in the maze: "))

global states
# make num_states dimensional list of hrrs, where each hrr has 10 * num_states elements 
states = hrr.hrrs(10*num_states, num_states, True) #normalized is unitary better for convergence
goal = int(input("Enter the index of the goal state: ")) 
r = [0 for x in range(num_states)] #rewards
r[goal] = 1 #init val at index of goal state to be 1
weights = hrr.hrr(10*num_states, True)
num_episodes = int(input("Enter the number of episodes: "))


start_states = copy.deepcopy(states)
start_weights = copy.deepcopy(weights)

print("The weights for the neural net are init to:", weights)
print("States at Start: ", states, "\nReward array: ", r)
t=0
pos_indices = [x for x in range(num_states)]
#print(pos_indices)


for episode_number in range(num_episodes):
	print("****************EPISODE", episode_number, "************************")		
	goal_state = False
	pos = random.choice(pos_indices) #starting pos
	print("Starting position:", pos)


	#update weights with nn that approximates v(s)
	while not goal_state:	
		next_pos = Policy(pos, weights, episode_number, num_states)	
		weights, goal_state = TD_Delta( weights, r[pos], pos, goal_state, episode_number, next_pos )
		pos = next_pos
		t += 1

		print("*******************\nNN VALUES:")
		vals_at_states()




print("Weights at Start:", start_weights)
print("\n")
print("\nWeights at End: ", weights)

print("States at Start:", start_states)
print("\n")
print("\nStates at End:", states)


vals_at_states()
"""
