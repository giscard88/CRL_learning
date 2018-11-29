import argparse
import gym
import numpy as np
from itertools import count
from collections import Counter, defaultdict, namedtuple
from func_export import *
import pylab
import json

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

'''

This script is derived from an example included in the pytorch distribution. 

'''


parser = argparse.ArgumentParser(description='PyTorch actor-critic example')
parser.add_argument('--rejection', type=int, default=10, metavar='N',
                    help='maximal rejection (default: 10)')
parser.add_argument('--seed', type=int, default=543, metavar='N',
                    help='random seed (default: 543)')
parser.add_argument('--search', type=int, default=10, metavar='N',
                    help='length of plan (default: 10)')

parser.add_argument('--start', type=int, default=1000, metavar='N',
                    help='load the model (default: 10)')

parser.add_argument('--test', type=int, default=10, metavar='N',
                    help='number of trials (default: 10)')

parser.add_argument('--torch_seed', type=int, default=500, metavar='N',
                    help='number of trials (default: 500)')

args = parser.parse_args()


env = gym.make('LunarLander-v2')
env.seed(args.seed)
torch.manual_seed(args.torch_seed)

rejection=args.rejection
search=args.search
ext=str(args.start)
test_length=args.test

D_agent=eval_states(8)

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
fake_env=Mod_env().to(device)

model.load_state_dict(torch.load('data/model_'+ext))
fake_env.load_state_dict(torch.load('data/fake_env_'+ext))

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
    return action.item(), state_value


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


def speculate(state, targets, prev_node,test_net, length=10):
    plan=[]
    plan_flag=False
    plan_blind=[]
    #print ('target',targets)
    #state=D_agent.mem[state]

    with torch.no_grad():
        for l in range(length):  # Don't infinite loop while learning
            #print (state)
            action, st = select_action(state,store=False)
            plan_blind.append(action)
            temp=np.zeros(4)
            temp[action]=1
            input_i=np.hstack((state,temp))
            outputs=fake_env(torch.Tensor(input_i).to(device))

            state_f=outputs.cpu().detach().numpy()
            outputs=outputs/torch.norm(outputs)
            outputs=test_net(outputs).cpu().detach()
            values, indices = torch.max(outputs,0)
            values=values.numpy()
            indices=indices.numpy()

            if values>=D_agent.thres:
                state_n=indices
            else:
                state_n=-99
            

            if state_n!=prev_node:

                plan.append(action)
                if state_n in targets:
                    plan_flag=True
                    #print ('optimistic')
                    break

            state=state_f

        del outputs,values,indices, input_i,state_f,state,temp
    #print ('plan in func',plan)
    return (plan_flag, plan,st,plan_blind)
       

def main():
    all_reward=[]
    env_errors=[]
    crash=[]
    testlength=test_length
    
    
    
   

    fp=open('data/path_'+ext+'.json','r')
    pw3=json.load(fp)
    fp.close()

    fp=open('data/freq_'+ext+'.json','r')
    pw2=json.load(fp)
    fp.close() 
    
    pw1=np.load('data/memory_'+ext+'.npy')

    D_agent=eval_states(8)
    D_agent.import_net(pw1,pw2,pw3)
    del pw1,pw2,pw3





    paths=path_net(D_agent)
    paths.make_net()
    n_nodes=D_agent.pw1.shape[0]
    pw1=torch.Tensor(D_agent.pw1)
    test_net=Single(n_nodes)
    model_dict=test_net.state_dict()
    model_dict['affine1.weight']=pw1
    test_net.load_state_dict(model_dict)
    test_net.to(device)
   


    all_reward=list(all_reward)
    for i_episode in range(testlength):
        state = env.reset()
        running_reward=[]
        prev_node=-9999
        for t in range(200):  # Don't infinite loop while learning
            
            target_states,flag=find_next_states4(D_agent,paths, state)
            prev_node=flag

            # add a routine to find the targets here.
            plan_all=[]
            value_all=[] 
            for s in range(rejection): # maxium number of rejection. If it takes too long, let's move on. 
                
                ans=speculate(state, target_states,prev_node,test_net,length=search)
                plan_all.append(ans[3])
                value_all.append(ans[2].detach())

                if ans[0]==True:
                    break


            #ans=[False,False]
            if ans[0]==True:
                #print ('plan proceed',len(ans[1]))
                #print (ans[1])
                for xin in ans[1][:5]: # the actions in the plan
                    state, reward, done, _ = env.step(xin)
                    running_reward.append(reward)
                    #env.render()
                    #print (xin, reward, done)
                    if done:
                        #print ('rw',reward)
                        break

            else:
                value_all=np.array(value_all)
                idx=np.argmax(value_all)
                #print ('suppose to be the best')
                for xin in plan_all[idx][:1]: # the actions in the plan
                    state, reward, done, _ = env.step(xin)
                    running_reward.append(reward)
                    #env.render()
                    #print (xin, reward, done)
                    if done:
                        #print ('rw',reward)
                        break

         
           
            if done:
                rw=np.array(running_reward)
                all_reward.append(np.sum(rw))
                crash.append(reward)
                print (i_episode, np.sum(rw),state)
                break
        else: # else for for-loop.
            rw=np.array(running_reward)
            all_reward.append(np.sum(rw))
            crash.append(reward)
            print (i_episode, np.sum(rw),state)

    data={'reward':all_reward,'crash':crash}
    fp=open('results_'+ext+'_'+str(testlength)+'_'+str(rejection)+'_'+str(search)+str(args.torch_seed)+'.json','w')
    json.dump(data,fp)
    fp.close()       

        

if __name__ == '__main__':

    main()
