# GNN_Experiments
Experiments with GNNS

## Introduction
This respository is for early experiments related to real-time rescheduling research project. Here we want to get an early feeling on the potentail of GNNs for the tasks of real time rescheduling.

## Experiments
### Train position propagation
The first experiment is just a simple location propagation within a network. We want to train a GNN to correctly predict the next position of a train within a graph.
[Here](https://github.com/MLerik/GNN_Experiments/blob/master/train_propagation.py) is the corresponding experiment.

How the setup works:
 1. Given the current state of the graph, the hidden state after message passing should predict the next state of the graph.
 2. The hidden state of the graph is taken as new current state of the graph and step one is repeated.
 
Data structure:

We start by a simple graph model with the following graph consisting of 5 nodes and 4 edges (symmetrical, thus 8 if modeled in Pytorch geometric).
```
         O
        /
O--O--O--O

```

We encode the current position of the train by setting the node feature to `1`:

```
         O
        /
1--O--O--O

```

And want tot rain a GNN to predict the next step. For the early experiment the schedule for the train is simply to move from the left most node to the right most node. Thus the sequence of graph time steps are:

`t = 0`

```
         O
        /
1--O--O--O

```

`t = 1`

```
         O
        /
O--1--O--O

```

`t = 2`

```
         O
        /
O--O--1--O

```

`t = 3`

```
         O
        /
O--O--O--1

```
 This is a very simple dataset ;) and its only intention is to get a basic understanding of how pytoch geometric works.
 
 Because future problems will involve much more complex schedules and non-local effects we want to do multiple message passing steps for each time step of the graph.
 In future models we will also introduce some recurrent network structure to have a better representation of time.
 
 ### Train position propagation multi path
 
 We look at the following infrastructure graph
 TODO add node numbers
 ```
           O--O--O
          /       \
A: O--O--O--O--O--O--O--O--O  :B

```

and introduce a second train. We thus have two train:

- One travelling A --> B
- One travelling B --> A

We set up the experiment in multiple phases:

1. Teach the network to propagate the two trains independantly to their goals
2. Teach network that there are multiple paths in the system
3. Teach network to adapt the probabilities when the other train is present.
