import random
import copy
import numpy as np
import sys
sys.path.insert(0, './td_hrr')
import hrr 


#==================================================================
#Neural Net approximation for value function
def nn( pos, w, states ):

	dot_prod = np.dot(states[pos], w)
	bias = 1
	return ( dot_prod + bias )

#==================================================================
def input_layer_activation( inp_layer_list ):
	return inp_layer_list	

#====================================================================
# Update weights of neural net 
def TD_Delta ( w, r, position, g_state, next_position, states, G, A, goal_idx ):	

	next_value = nn(next_position, w, states)
	value = nn(position, w, states)


	#Terminal
	if position == goal_idx:
		g_state = True
		delta = r - value 
		w += A * delta * states[position] 
	


	#Non-Terminal	
	else:
		delta =  r + ( G* next_value ) - value  
		w += A * delta * states[position] 


	return w, g_state
#====================================================================
# Decide Whether To Go Left Or Right and return index of the choice
def Policy( cur_pos, weights, num_s, states):
	
	left, right = Get_Left_Right(cur_pos, num_s)
	val_left, val_right = nn(left, weights, states), nn(right, weights, states)

	if val_left > val_right: return left #go left
	
	elif val_right > val_left: return right #go right

	#pick random direction if both directions have same value	
	else: return random.choice([ left, right ]) 


#====================================================================
def Get_Left_Right(posit, length):

    #Get left and right indices
    left = (posit-1)
    if left < 0: left = length - 1 #wrap around    
    
    right = (posit+1) % length #also handles wrapping

    return left, right  
 
#==================================================================
def vals_at_states(w, states):
	values = []
	for pos in range(len(states)):
		values.append( nn(pos, w, states)  )
	return values





"""
def TD_Delta ( w, r, position, g_state, next_position, states, G, A, goal_idx ):	

	#WEIGHTS UPDATING...	

	next_value = nn(next_position, w, states)
	value = nn(position, w, states)

	#Non-Terminal	
	delta =  r + ( G* next_value ) - value  

	tmp = copy.deepcopy(w)
	
	for i in range(len(w)):
#		w[i] +=  A * delta * states[position][i] 

		tmp[i] +=  A * delta * states[position][i] 

	w += A * delta * states[position] 



	#Terminal
	if position == goal_idx:
		g_state = True
		delta = r - value 
		
		for i in range(len(w)):		
			w[i] += A * delta * states[position][i] 
		
#		w += A * delta * states[position] 
	
	return w, g_state
"""



"""
	#Non-Terminal	
	delta =  r + ( G* next_value ) - value  
	w += A * delta * states[position] 



	#Terminal
	if position == goal_idx:
		g_state = True
		delta = r - value 
		w += A * delta * states[position] 
	
	return w, g_state
"""
