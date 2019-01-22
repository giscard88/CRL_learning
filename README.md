# complementary reinforcement learning for explainable agents

## Backgrounds

Reinforcement learning (RL) algorithms allow agents to learn skills and strategies to perform complex tasks without detailed instructions or expensive labelled training examples. That is, RL agents can learn, as we learn. Given the importance of learning in our intelligence, RL has been thought to be one of key components to general artificial intelligence, and recent breakthroughs in deep reinforcement learning suggest that neural networks (NN) are natural platforms for RL agents. However, despite the efficiency and versatility of NN-based RL agents, their decision-making remains incomprehensible, reducing their utilities. To deploy RL into a wider range of applications, it is imperative to develop explainable NN-based RL agents. Here, we propose a method to derive a secondary comprehensible agent from a NN-based RL agent, whose decision-makings are based on simple rules. Our empirical evaluation of this secondary agent's performance supports the possibility of building a comprehensible and transparent agent using a NN-based RL agent.

For detailed description, plase read a [preprint](https://arxiv.org/abs/1901.00188). 

## Codes

1. train_rl.py: it traines the RL agent (as well as QS agent) and saves it. 
2. run_qs.py: it loads the RL agent saved and utilizes it to test QS agent's action.
3. run_rl.py: it loads the RL agent saved and utlizes it to test the RL agent's performance.
4. batch.sh: it produces results in a simulation condition.
5. function.py: a collection of functions shared by RL and QS agents
6. plot_result.py: an example script to analyze simulation results.

## Dependency

1. Pytorch>0.4
2. OpenAI gym
3. numpy

## contact

please send comments and feedbacks to giscard88@gmail.com

