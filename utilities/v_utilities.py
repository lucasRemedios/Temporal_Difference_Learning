import random
import copy

#====================================================================
# Update And Return New Value For v[s] 
def TD_Delta ( v, v_next, r, position, g_state, A, G, goal ):

    #Terminal State
    if position == goal:
        g_state = True
        return (v + A * ( r - v )), g_state

    #Non-Terminal State
    else: return (v + A * ( r + ( G*v_next ) - v )), g_state


#====================================================================
# Decide Whether To Go Left Or Right and return index of the choice
def Policy( cur_pos, v_list):
		
    left, right = Get_Left_Right(cur_pos, len(v_list))
    val_left, val_right = v_list[left], v_list[right] 

    if val_left > val_right: return left   #go left

    elif val_right > val_left: return right   #go right

    else: return random.choice([ left, right ]) #random dir if both dir same val   


#====================================================================
def Get_Left_Right(posit, length):

	#Get left and right indices
    left = (posit-1)
    if left < 0: left = length - 1 #wrap around    
    
    right = (posit+1) % length #also handles wrapping

    return left, right    
