import argparse
import gym
import numpy as np
from itertools import count
from collections import Counter, defaultdict, namedtuple
from function import *
import pylab
import json
import pickle

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os


from torch.distributions import Categorical

'''

This script is derived from an example included in the pytorch distribution. 

'''


parser = argparse.ArgumentParser(description='Complementary RL model')

parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                    help='discount factor (default: 0.99)')

parser.add_argument('--seed', type=int, default=1, metavar='N',
                    help='random seed (default: 901)')

parser.add_argument('--simlength', type=int, default=1000, metavar='N',
                    help='number of simulation step (default: 1000)')

args = parser.parse_args()


env = gym.make('LunarLander-v2')
env.seed(args.seed)
torch.manual_seed(args.seed)

D_agent=eval_states2(8)

SavedAction = namedtuple('SavedAction', ['log_prob', 'value'])


class Policy(nn.Module):
    def __init__(self):
        super(Policy, self).__init__()
        self.affine1 = nn.Linear(8, 100)
        self.action_head = nn.Linear(100, 4)
        self.value_head = nn.Linear(100, 1)

        self.saved_actions = []
        self.rewards = []

    def forward(self, x):
        x = F.relu(self.affine1(x))
        action_scores = self.action_head(x)
        state_values = self.value_head(x)
        return F.softmax(action_scores, dim=-1), state_values

class Mod_env(nn.Module):
    def __init__(self):
        super(Mod_env, self).__init__()
        self.affine1 = nn.Linear(12, 300)
        self.state_new = nn.Linear(300, 8)

    def forward(self, x):
        x = F.relu(self.affine1(x))
        state_scores = self.state_new(x)    
        return state_scores

class Single(nn.Module):
    def __init__(self,n_out):
        super(Single, self).__init__()
        self.affine1 = nn.Linear(8, n_out,bias=False)

    def forward(self, x):
        x = self.affine1(x)
 
        return x

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = Policy().to(device)
#model=model.cuda()
#model.to(device)
optimizer = optim.Adam(model.parameters(), lr=3e-3)
eps = np.finfo(np.float32).eps.item()
fake_env=Mod_env().to(device)
#fake_env=fake_env.cuda()
#fake_env.to(device)
loss_explicit= nn.MSELoss()
optimizer_explicit = optim.SGD(fake_env.parameters(), lr=5e-2)

def select_action(state,store=True):
    state = torch.from_numpy(state).float()
    #state=state.cuda()
    state=state.to(device)
    probs, state_value = model(state)
    probs=probs.cpu()
    state_value=state_value.cpu()
    m = Categorical(probs)
    action = m.sample()
    if store:
        model.saved_actions.append(SavedAction(m.log_prob(action), state_value))
    return action.item()


def finish_episode():
    R = 0
    saved_actions = model.saved_actions
    policy_losses = []
    value_losses = []
    rewards = []
    for r in model.rewards[::-1]:
        R = r + args.gamma * R
        rewards.insert(0, R)
    rewards = torch.tensor(rewards)
    rewards = (rewards - rewards.mean()) / (rewards.std() + eps)
    for (log_prob, value), r in zip(saved_actions, rewards):
        reward = r - value.item()
        policy_losses.append(-log_prob * reward)
        value_losses.append(F.smooth_l1_loss(value, torch.tensor([r])))
    optimizer.zero_grad()
    loss = torch.stack(policy_losses).sum() + torch.stack(value_losses).sum()
    loss.backward()
    optimizer.step()
    del model.rewards[:]
    del model.saved_actions[:]

def train_mod_env(local_state,local_action):
    length=len(local_state)
    patterns=[]
    outputs_d=[]
    
    for i in range(0,length-1):
        temp=np.zeros(4)
        temp[local_action[i]]=1
        input_i=np.hstack((local_state[i],temp))
        output_i=local_state[i+1]
        patterns.append(input_i)
        outputs_d.append(output_i)
    
    patterns=np.array(patterns)
    outputs_d=np.array(outputs_d)

    patterns=torch.Tensor(patterns)
    outputs_d=torch.Tensor(outputs_d)
    #patterns=patterns.cuda()
    #outputs_d=outputs_d.cuda()
    patterns=patterns.to(device)
    outputs_d=outputs_d.to(device)
    optimizer_explicit.zero_grad()
    outputs=fake_env(patterns)
    le = loss_explicit(outputs, outputs_d)
    le.backward()
    #print ('error', le.item())
    optimizer_explicit.step()
    return le.item()




def main():
    all_reward=[]
    env_errors=[]
    crash=[]
    simlength=args.simlength
    if os.path.exists('data'):
        pass
    else:
        os.mkdir('data')
    
    length_act=20
    for i_episode in range(simlength):
        state = env.reset()
        running_reward=[]
        local_state=[]
        local_action=[]
        prev_t=-999
        for t in range(1000):  # Don't infinite loop while learning
            action = select_action(state)
            local_state.append(state)
            local_action.append(action)
            state, reward, done, _ = env.step(action)
            model.rewards.append(reward)
            running_reward.append(reward)

            if done:
                all_reward.append(np.sum(np.array(running_reward)))
                crash.append(reward)
                print (i_episode, 'reward',all_reward[-1])

                break
        finish_episode()
        rv=np.array(running_reward)
        
        for s, st in enumerate(local_state):
            rt=np.sum(rv[s:])
            if s==0:
                D_agent.set(st)
            else:
                D_agent.update_value(st,rt,rv[s]) 
        D_agent.reset()

        error_i=train_mod_env(local_state,local_action)
        env_errors.append(error_i)
        if (i_episode+1)%1000==0:
            ext=str(i_episode+1)

            fp=open('data/path_'+ext+'.json','w')
            json.dump(D_agent.pw3,fp)
            fp.close()


            fp=open('data/freq_'+ext+'.json','w')
            json.dump(D_agent.pw2,fp)
            fp.close() 
            
            np.save('data/memory_'+ext,D_agent.pw1)
            
            for g in optimizer_explicit.param_groups:
                g['lr'] = g['lr']*0.1

            for g in optimizer.param_groups:
                g['lr'] = g['lr']*0.1

    

            model_dict=model.state_dict()
            fake_env_dict=fake_env.state_dict()
            torch.save(model.state_dict(), 'data/model_'+ext)
            torch.save(fake_env.state_dict(), 'data/fake_env_'+ext)
           
    data={'reward':all_reward,'crash':crash,'env_error':env_errors}
    fp=open('results_'+str(simlength)+'.json','w')
    json.dump(data,fp)
    fp.close()
    
        

if __name__ == '__main__':

    main()
