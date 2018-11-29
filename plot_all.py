import pylab
import json
import numpy as np
import argparse

parser = argparse.ArgumentParser(description='PyTorch actor-critic example')
parser.add_argument('--rejection', type=int, default=10, metavar='N',
                    help='maximal rejection (default: 10)')
parser.add_argument('--seed', type=int, default=543, metavar='N',
                    help='random seed (default: 543)')
parser.add_argument('--search', type=int, default=10, metavar='N',
                    help='length of plan (default: 10)')

parser.add_argument('--start', type=int, default=1000, metavar='N',
                    help='load the model (default: 10)')

parser.add_argument('--test', type=int, default=100, metavar='N',
                    help='length of test (default: 100)')


args = parser.parse_args()


duration=args.test
rejection=args.rejection
search=args.search
start=args.start

exp=[]
cr_exp=[]
for xin in range(1,11):

    fp=open('results_'+str(start)+'_'+str(duration)+'_'+str(rejection)+'_'+str(search)+str(xin)+'.json','r')
    data=json.load(fp)
    reward=data['reward']
    exp.append(reward)
    cr_exp.append(data['crash'])
exp_r=np.mean(np.array(exp),0)

exp_ref=[]
cr_ref=[]
for xin in range(1,11):

    fp=open('results_ref'+str(start)+'_'+str(duration)+'_'+str(xin)+'.json','r')
    data=json.load(fp)
    reward=data['reward']
    exp_ref.append(reward)
    cr_ref.append(data['crash'])
exp_ref_r=np.mean(np.array(exp_ref),0)

print ('exp reward',np.mean(exp_r))
print ('exp crash',np.mean(np.array(cr_exp)))

print ('ref reward',np.mean(exp_ref_r))
print ('ref crash',np.mean(np.array(cr_ref)))

pylab.plot(exp_r,'r',label='exp')
pylab.plot(exp_ref_r,'b',label='control')


pylab.legend()
pylab.show()

