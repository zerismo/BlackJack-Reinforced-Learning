import math
import random
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm


def randomCard():
    card = random.randint(1,13)
    if card > 10:
        card = 10
    return card

def useableAce(hand):
    val, ace = hand
    return ((ace) and ((val + 10) <= 21))

def totalValue(hand):
    val, ace = hand
    if (useableAce(hand)):
        return (val + 10)
    else:
        return val

def add_card(hand, card):
    val, ace = hand
    if (card == 1):
        ace = True
    return (val + card, ace)

def eval_dealer(dealerhand):
    while (totalValue(dealerhand) < 17):
        dealerhand = add_card(dealerhand, randomCard())
    return dealerhand

def play(state, dec):
    
    playerhand = state[0] 
    dealerhand = state[1]
    if dec == 0: 
        dealerhand = eval_dealer(dealerhand)

        player_tot = totalValue(playerhand)
        dealer_tot = totalValue(dealerhand)
        status = 1
        if (dealer_tot > 21):
            status = 2 
        elif (dealer_tot == player_tot):
            status = 3 
        elif (dealer_tot < player_tot):
            status = 2 
        elif (dealer_tot > player_tot):
            status = 4 

    elif dec == 1: 
        
        playerhand = add_card(playerhand, randomCard())
        d_hand = eval_dealer(dealerhand)
        player_tot = totalValue(playerhand)
        status = 1
        if (player_tot == 21):
            if (totalValue(d_hand) == 21):
                status = 3 
            else:
                status = 2 
        elif (player_tot > 21):
            status = 4 
        elif (player_tot < 21):
            
            status = 1
    state = (playerhand, dealerhand, status)

    return state


def initGame():
    status = 1 
    playerhand = add_card((0, False), randomCard())
    playerhand = add_card(playerhand, randomCard())
    dealerhand = add_card((0, False), randomCard())
    
    if totalValue(playerhand) == 21:
        if totalValue(dealerhand) != 21:
            status = 2 
        else:
            status = 3 

    state = (playerhand, dealerhand, status)
    return state


##Not using RL concepts till now


def initStateSpace():
    states = []
    for card in range(1,11):
        for val in range(11,22):
            states.append((val, False, card))
            states.append((val, True, card))
    return states



def initStateActions(states):
    av = {}
    for state in states:
        av[(state, 0)] = 0.0
        av[(state, 1)] = 0.0
    return av


def initSAcount(stateActions):
    counts = {}
    for sa in stateActions:
        counts[sa] = 0
    return counts



def calcReward(outcome):
    return 3-outcome


def updateQtable(av_table, av_count, returns):
    for key in returns:
        av_table[key] = av_table[key] + (1 / av_count[key]) * (returns[key]- av_table[key])
    return av_table


def qsv(state, av_table):
    stay = av_table[(state,0)]
    hit = av_table[(state,1)]
    return np.array([stay, hit])



def getRLstate(state):
    playerhand, dealerhand, status = state
    player_val, player_ace = playerhand
    return (player_val, player_ace, dealerhand[0])


epochs = 1000000 
epsilon = 0.1

state_space = initStateSpace()
av_table = initStateActions(state_space)
av_count = initSAcount(av_table)






for i in range(epochs):
    
    state = initGame()
    playerhand, dealerhand, status = state
    
    
    
    while playerhand[0] < 11:
        playerhand = add_card(playerhand, randomCard())
        state = (playerhand, dealerhand, status)
    rl_state = getRLstate(state) 

    
    returns = {} 
    while(state[2] == 1): 
        
        act_probs = qsv(rl_state, av_table)
        if (random.random() < epsilon):
            action = random.randint(0,1)
        else:
            action = np.argmax(act_probs)
        sa = ((rl_state, action))
        returns[sa] = 0 
        av_count[sa] += 1 
        state = play(state, action) 
        rl_state = getRLstate(state)
    
    for key in returns:
        returns[key] = calcReward(state[2])
    av_table = updateQtable(av_table, av_count, returns)
