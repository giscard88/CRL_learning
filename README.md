### Complementary reinforcement learning system

## Background
Reinforcement learning (RL) is thought to be a key to general artificial intelligence because like the human brain it allows agents to learn a wide range of tasks based on environmental interactions and rewards. One of the most influential RL examples is the deep reinforcement learning (DRL) that trained agents to outperform human players in a subset of Atari 2600 games. Despite their success, the decision-making process of RL/DRL agents differs from ours, and their decisions can be incomprehensible, which reduces RL/DRL agents’ utilities. 

Some recent studies do explain RL agents’ decision-making by estimating expected outcomes and providing user-interpretable descriptions, but it remains unclear whether the proposed algorithms can explain the exact reasons behind RL agents’ actions. In our study, we propose an alternative approach to the accountability of RL agents-- an quasi-symbolic (QS) agent that makes decisions based on a set of understandable symbolic rules. To be precise, RL agents suggest probable actions, and QS agents utilize a few symbolic rules to select the best action among them.   

Here, 1) we propose a generic QS agent and its learning method and 2) compare its performance to that of the original RL agent. The proposed algorithm is designed to be generic as much as possible to increase the generalization of the results, and the comparison between RL and QA agents’ performance was conducted to test QS agents’ utility. 
     
## Methods
Structure of RL agent
We used an actor-critic model as an reference RL agent. The implementation was adopted from an official pytorch github repository. The actor consists of 8 input, 100 hidden and 4 output nodes, and the critic, 8 input, 100 hidden and 1 output nodes. The initial running rate is 0.03 and is decreased by 10 times at every 1000 episode. 

# Structure of quasi-symbolic (QS) agent

QS agent consists of Env and Act networks. The Env network, which is a single-hidden layer perceptron, models transitions between states induced by agents’ actions. Using Env network, a QS agent devises a plan by simulating potential consequences of their actions. We used the ‘lunar-lander’ benchmark task from openAI gym environment to estimate the agents’ performance. As a lunar lander interacts with 8 dimensional environments via 4 possible actions, the input layer of Env network consists of 12 nodes, which is the sum of dimensions of state and action spaces, and the output layer size is 8. The number of hidden layers is 300. 

The Act network provides QS agents’ next potential target states to be reached and is also a single-hidden layer network. The first layer receives state vector as an input. The first synaptic layer connecting the input and hidden layers stores the new states observed during RL agents’ learning episodes by imprinting the normalized states into synaptic weights. With these weights imprinted, the inputs to hidden nodes represent the cosine similarity between the current and stored states. When the previously stored state is presented again to Act network, the hidden node, which is added during the first presentation of the state, will have the maximal synaptic input and the maximal output. That is, each hidden node represents a distinct state of environment, and during the hidden nodes’ activation, we can examine if the current state is distinct from the earlier states. In this study, we consider the current state to be identical to the stored one when the synaptic input (i.e., the cosine similarity) is higher than 0.97. By definition, the Act network size is not fixed. Instead, the hidden layer size grows during RL agents’ learning. The second synaptic layer represents transitions between states (i.e., the hidden nodes), and the output layer size is identical to the hidden layer; that is, whenever a hidden unit is added, a new output unit is also added. Synaptic weights between hidden and output layers are initialized to 0, and when a RL agent moves from one state si to another sj, the weight wji is increased by 1. In this way, each synaptic connection wij, between hidden node i and output unit j, represents the frequency of transitions between the two observed states. The transition between the same states is ignored for the Act network.  

Once the Act network is constructed, its layers are used separately. As it determines whether the current state has been previously presented and needs to be stored, the state vector is introduced to the input layer, and synaptic inputs of hidden layers are examined. When it identifies potential target states, the hidden node corresponding to the current state is activated, and the maximally innervated output node is selected to the most probable target. In this study, we selected X potential target states using the strength of inputs to output nodes. Similarly, hidden and output layers can recursively activated to calculate targets to be reached in multiple steps, not in a single step. 

# Quasi-Symbolic (QS) agents’ decision-making
QS agents’ decision-making process can be summarized in two steps. 

Identifying target states: Given the current state, the Act network identifies potential candidates for next states. If the current state is stored in the Act network, the Act network finds a set of potential target states by recursively innervating hidden nodes and monitoring synaptic inputs to output layers. If the current state is not stored in the Act network, the Act networks return the most valuable states as target states. The states’ values are estimated by summing the synaptic weights converging to the corresponding output node, since synaptic connections to a output node show how often the state is the consequence of RL agent’s action during learning. 

Identifying a plan for QS agent to reach one of the target states: QS agent simulates a sequence of states induced by its actions. For a given state, QS agent uses RL agent to select a probable action and feeds it to the Env network to estimate the next state. It repeats this process 10 times to have a sequence of states. If one of the target states is included in the sequence, QS agent selects it as an action plan and executes the plan. The length of time step varies, as it depends on how long it takes to reach the target state. In this study, we set a plan’s maximal length to be 10. That is, while QS agents can have a 10 time-step long plan, it executes only a half of them. This restriction is introduced because the sequence of actions are simulated using the Env network, an approximation of the actual environment, and the error can be accumulated in a long sequence of actions. If a sequence of states does not reach any target states, it searches for a new sequence. In the experiment, QS agent tested 10 sequences of actions. When QS agents fail to reach one of the target states after 10 sequences, it uses the value function of RL agent’s critic to find the best action. Specifically, QS agent compares the values of the next states from 10 sequences which it generated to find the best one with the maximal value function. Then, QS agent selects the first action to take from the identified sequence.  

## Results
We addressed QS agent’s performance compared to RL agents by using the “lunar-lander” benchmark task available in openAI gym environment. In this study, RL is a actor-critic model implemented with the pytorch, an open-source machine learning library; the reference implementation is part of the official pytorch release. Figures below show the total reward and the environment model error during 5000 episodes. 

[figure1](figures/env.png?raw=true)
[figure2](figures/reward.png?raw=true)

As shown in the figures, both reward and error improved rapidly only until 1000 episodes; afterwards they showed no significant improvements. Thus, we constructed QS agents after 1000 episodes and tested their performance to be compared to the original RL agent. For comparison, we frozen RL agents’ learning and measured the total rewards that both RL and QS agents obtained during 300 episodes. Before the first episode, the gym is initialized with the same random seed, and then the random seed was not reset afterwards. We also tested 10 different instantiations of QS and RL agents by feeding different random seeds into the pytorch. 

[figure3](figures/test.png?raw=true)

As seen in Figure above, QS agents’ performance shown in red is slightly better than that of RL agents shown in blue, suggesting that QS agents perform as good as or as bad as RL agents. In other words, based on RL agents’ learning, our experiments raised the possibility that a secondary agent, with transparent reasons behind actions, can be constructed.

## Future questions

Our focus is to gain a better understanding of QS agent’s properties. 
1. What are the other factors that can influence QS agents’ performance? 
2. How do we decide the valuable states to be used to train QS agents?

