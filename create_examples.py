#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 20 12:57:57 2020

@author: ajnatarajan
"""
import time
import random
alpha = ['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q', \
         'R','S','T','U','V','W','X','Y','Z']

        
    
def create_airports(maxlen):
    airports = []
    
    def helper(temp, maxlen):
        if len(temp) == maxlen:
            airports.append(''.join(temp))
            return
        for i in range(len(alpha)):
            temp.append(alpha[i])
            helper(temp, maxlen)
            temp.pop()
            
    helper([], maxlen)

    return airports

def create_connections(maxlen, airports, num_connec, prob_intra):
    connections = set()
    for _ in range(num_connec):
        prob = random.uniform(0,1)
        within = False
        if prob < prob_intra:
            within = True
        
        start = []
        end = []
        if within:
            for _ in range(maxlen//2):
                add = random.choice(alpha)
                start.append(add)
                end.append(add)
            for _ in range(maxlen - maxlen//2):   
                start.append(random.choice(alpha))
                end.append(random.choice(alpha))
        else:
            for _ in range(maxlen):
                start.append(random.choice(alpha))
                end.append(random.choice(alpha))
        
        connections.add((''.join(start), ''.join(end)))
    return connections

def create_example_CPU(filename, airports, connections, starting_airport):
    f = open(filename, 'w')
    
    f.write(str(len(airports)) + '\n')
    for a in airports:
        f.write(a + '\n')
    f.write(starting_airport + '\n')
    f.write(str(len(connections)) + '\n')
    for c in connections:
        f.write(c[0] + '\n')
        f.write(c[1] + '\n')
    
    f.close()
    

def create_example_GPU(filename, airports, connections, starting_airport):
    f = open(filename, 'w')

    f.write(str(len(airports)) + ' ' + str(len(connections)) + '\n')
    for a in airports:
        f.write(a + '\n')
    for pair in connections:
        f.write(pair[0] + ' ' + pair[1] + '\n')
    f.write(starting_airport)
    
    f.close()

def create_example(name_len, num_connec, prob_intra):
    ports = create_airports(name_len)
    connects = create_connections(name_len, ports, num_connec, prob_intra)
    start_port = random.choice(ports)
    
    precursor = str(name_len)+','+str(num_connec)+','+str(prob_intra)+'_'
    
    create_example_CPU(precursor + "CPU.txt", ports, connects, start_port)
    create_example_GPU(precursor + "GPU.txt", ports, connects, start_port)

if __name__ == "__main__": 
    maxlen = input("Enter desired length of airport names. The number of \
                   airports will be 26**x, where x is what you enter: ")
    num_connec = input("Enter desired number of connections between airports: ")
    prob_intra = input("Enter desired probability (between 0 and 1) for each \
                       connection to be intracontinental (recommended > 0.75): ")
    print("Creating CPU and GPU sample files...")
    create_example(int(maxlen), int(num_connec), float(prob_intra))
    
    
    
    
    

    
        
    
        
                
    
    
    
    
    
    
    
        
    
    