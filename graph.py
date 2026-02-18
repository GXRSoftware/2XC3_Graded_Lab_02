# For testing
graph = False
testW1 = False
testW2 = True

from collections import deque

##########
# Week 1 #
##########
#Undirected graph using an adjacency list
class Graph:
    def __init__(self, n):
        self.adj = {}
        for i in range(n):
            self.adj[i] = []

    def are_connected(self, node1, node2):
        return node2 in self.adj[node1]

    def adjacent_nodes(self, node):
        return self.adj[node]

    def add_node(self):
        self.adj[len(self.adj)] = []

    def add_edge(self, node1, node2):
        if node1 not in self.adj[node2]:
            self.adj[node1].append(node2)
            self.adj[node2].append(node1)

    def number_of_nodes(self):
        return len(self.adj.keys())

if testW1:
    g = Graph(7)
    g.add_edge(1, 2)
    g.add_edge(1, 3)
    g.add_edge(2, 4)
    g.add_edge(3, 4)
    g.add_edge(3, 5)
    g.add_edge(4, 5)
    g.add_edge(4, 6)
    print(g.adj)

#Breadth First Search
def BFS(G, node1, node2):
    Q = deque([node1])
    marked = {node1 : True}
    for node in G.adj:
        if node != node1:
            marked[node] = False
    while len(Q) != 0:
        current_node = Q.popleft()
        for node in G.adj[current_node]:
            if node == node2:
                return True
            if not marked[node]:
                Q.append(node)
                marked[node] = True
    return False

#Breadth First Search Path
def BFS2(G, node1, node2):
    Q = deque([(node1, [])])
    marked = {node1 : True}

    for node in G.adj:
        if node != node1:
            marked[node] = False
    
    
    while len(Q) != 0:
        current_node = Q.popleft()
        path = current_node[1].copy()
        path.append(current_node[0])
        for node in G.adj[current_node[0]]:
            if node == node2:
                path.append(node)
                return path
            if not marked[node]:
                Q.append((node, path))
                marked[node] = True
    return []

if testW1:
    print(BFS2(g, 1, 6))
    print(BFS2(g, 1, 5))
    print(BFS2(g, 1, 4))
    print(BFS2(g, 0, 4))
    print(g.number_of_nodes())

#Breadth First Search Predecessor Dictionary
def BFS3(G, node1):
    P = {}
    Q = deque([node1])
    marked = {node1 : True}

    for node in G.adj:
        if node != node1:
            marked[node] = False
    
    
    while len(Q) != 0:
        current_node = Q.popleft()
        for node in G.adj[current_node]:
            if not marked[node]:
                Q.append(node)
                marked[node] = True
                P[node] = current_node
    return P

if testW1:
    print(BFS3(g, 0))
    print(BFS3(g, 1))
    print(BFS3(g, 2))
    print(BFS3(g, 3))
    print(BFS3(g, 4))
    print(BFS3(g, 5))
    print(BFS3(g, 6))


#Depth First Search
def DFS(G, node1, node2):
    S = [node1]
    marked = {}
    for node in G.adj:
        marked[node] = False
    while len(S) != 0:
        current_node = S.pop()
        if not marked[current_node]:
            marked[current_node] = True
            for node in G.adj[current_node]:
                if node == node2:
                    return True
                S.append(node)
    return False

import random
def create_random_graph(i, j):
    G = Graph(i)
    for e in range(j):
        n1, n2 = random.randint(0, i - 1), random.randint(0, i - 1)
        while G.are_connected(n1, n2) or n1 == n2:
            n1, n2 = random.randint(0, i - 1), random.randint(0, i - 1)
        G.add_edge(n1, n2)
    return G

if testW1:
    RG = create_random_graph(100, 50)

    print(BFS(RG, 0, 99))
    print(BFS2(RG, 0, 99))
    print(BFS3(RG, 0))

def is_connected(G):
    Q = deque([0])
    visited = {0 : True}

    for node in G.adj:
        if node != 0:
            visited[node] = False
    
    while len(Q) != 0:
        current_node = Q.popleft()
        for node in G.adj[current_node]:
            if not visited[node]:
                Q.append(node)
                visited[node] = True
    return all(visited.values())

if testW1:
    print(is_connected(g))
    g.add_edge(0, 1)
    print(is_connected(g))
    RG = create_random_graph(10, 45)
    print(is_connected(RG))
    RG = create_random_graph(10, 1)
    print(is_connected(RG))

# Experiments
import matplotlib
import random
import timeit
import matplotlib.pyplot as plt
import numpy as np
import math

################
# Experiment 1 #
################



################
# Experiment 2 #
################
if(graph):
    nodes = 100
    max_edges = (nodes * (nodes - 1)) // 2
    run_edges = list((x for x in range(max_edges // 10)))
    RUNS = 1000000

    data = []
    for e in run_edges:
        print(e)
        c = []
        for i in range(RUNS):
            G = create_random_graph(nodes, e)
            c.append(is_connected(G))
        data.append((sum(c) / RUNS) * 100)

    x_percent = list((e / max_edges) * 100 for e in run_edges)
    plt.plot(x_percent, data, color='red', label='Number of Edges')

    plt.title('Probability of Graph being Connected by Number of Edges')
    plt.xlabel('Percentage of Max Edges')
    plt.ylabel('Percentage of Connected')
    plt.legend()
    plt.show()

##########
# Week 2 #
##########

#Use the methods below to determine minimum vertex covers

def add_to_each(sets, element):
    copy = sets.copy()
    for set in copy:
        set.append(element)
    return copy

def power_set(set):
    if set == []:
        return [[]]
    return power_set(set[1:]) + add_to_each(power_set(set[1:]), set[0])

def is_vertex_cover(G, C):
    for start in G.adj:
        for end in G.adj[start]:
            if not(start in C or end in C):
                return False
    return True

def MVC(G):
    nodes = [i for i in range(G.number_of_nodes())]
    subsets = power_set(nodes)
    min_cover = nodes
    for subset in subsets:
        if is_vertex_cover(G, subset):
            if len(subset) < len(min_cover):
                min_cover = subset
    return min_cover


##################
# Approximations #
##################
def approx1(G):
    lG = G.adj.copy()
    C = set()
    cover = False

    while (not cover):
        # Finding the highest degree
        v = (0, 0)
        for node in lG.keys():
            d = len(lG[node])
            if v[1] < d:
                v = (node, d)
        v = v[0]

        # Add v to C
        C.add(v)

        # Remove all edges incident to node v from G
        lG[v] = []
        for node in lG.keys():
            if v in lG[node]:
                lG[node].remove(v)
        
        # Check for cover
        cover = is_vertex_cover(G, C)
    
    # Return the vertex cover
    return C

def approx2(G):
    lG = G.adj.copy()
    C = set()
    cover = False
    
    while (not cover):
        # Select a vertex randomly
        v = random.choice(list(lG.keys()))

        # Add v to C
        C.add(v)

        # Remove v from lG
        del lG[v]

        # Check for cover
        cover = is_vertex_cover(G, C)

    return C

def approx3(G):
    lG = G.adj.copy()
    C = set()
    cover = False

    while (not cover):
        # Select an edge randomly
        u = random.choice([u for u in lG if lG[u]]) 
        v = random.choice(lG[u])


        # Add u and v to C
        C.add(u)
        C.add(v)

        # Remove all edges incident to node u or v from G
        lG[u] = []
        for node in lG.keys():
            if u in lG[node]:
                lG[node].remove(u)

        lG[v] = []
        for node in lG.keys():
            if v in lG[node]:
                lG[node].remove(v)
        
        # Check for cover
        cover = is_vertex_cover(G, C)

    return C

if testW2:
    nodes = 15
    edges = int(((nodes * (nodes + 1)) / 2) * 0.7)
    RG = create_random_graph(nodes, edges)
    while (not is_connected(RG)):
        RG = create_random_graph(nodes, edges)
    a1 = approx1(RG)
    print(a1)
    a2 = approx2(RG)
    print(a2)
    a3 = approx3(RG)
    print(a3)
    mvc = set(MVC(RG))
    print(mvc)
    print("a1 match: " + str(a1 == mvc))
    print("a2 match: " + str(a2 == mvc))
    print("a3 match: " + str(a3 == mvc))



