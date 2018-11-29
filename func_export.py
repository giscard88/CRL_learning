import json
import numpy as np
from collections import Counter, defaultdict, namedtuple
import operator

def matching(Wm, vec, thres):
    vec_n=vec/np.linalg.norm(vec)


    flag_new=False
    if isinstance(Wm,np.ndarray):
        tmp=np.matmul(Wm,vec_n.T)
        num=Wm.shape[0]
    else:
        Wm.append(vec_n)
        Wm=np.array(Wm) # from now on, self.W is array
        tmp=np.ones(1)
        flag_new=True
    #print vec_n, Wm, tmp,np.amax(tmp)
    if np.amax(tmp)<thres:
        r,c=Wm.shape
        vec_n=vec_n.reshape(1,c)

        Wm=np.concatenate((Wm,vec_n))
        tmp_max=r  # index for the maximal activation, which is calculated by definition.
        flag_new=True 
  
    else:
        tmp_max=np.argmax(tmp)
    return Wm,tmp_max, tmp, flag_new


       
class path_net(): 
    
    def __init__(self, obj):
        self.obj=obj
        #values=np.array(obj.pw2)
        #thres_v=np.mean(values)+np.std(values)*0.1
        #idx=np.where(values>thres_v)[0]
        #self.hub=idx
        #print ('hub',type(self.hub),len(self.hub)) 

    def make_net(self):
 
        size=len(self.obj.pw2)
        self.wts=np.zeros((size,size))
        for xin in self.obj.pw3:
            temp=self.obj.pw3[xin]
            for yin in temp:
                #self.wts[int(xin),yin]=self.wts[int(xin),yin]+1
                self.wts[yin, int(xin)]=self.wts[yin, int(xin)]+1
        #for xi,xin in enumerate(self.obj.pw2):  
        #    self.wts[xi,xi]=xin

        values=np.sum(self.wts,axis=1)
        thres_v=np.mean(values)+np.std(values)*1.0
        idx=np.where(values>thres_v)[0]
        self.hub=idx
        print ('hub',type(self.hub),len(self.hub))
        



    def get_mp(self,node,length=5):
        idx_all=[]
        temp=self.wts[:,node]
        idx=np.argsort(temp)
        idx=np.flip(idx,0)
        ones=np.zeros(len(temp))
        ones[idx[:length]]=1.0
        temp=np.multiply(ones,temp)
        idx_all.append(idx)
        
        for xin in range(2):
            out=np.matmul(self.wts,temp.T)
            idx=np.argsort(out)
            idx=np.flip(idx,0)
            idx_all.append(idx)
            ones=np.zeros(len(out))
            ones[idx[:length]]=1.0
            temp=np.multiply(ones,out)
            temp=out
        rv=[]
        for xin in idx_all:
            rv.extend(xin[:length])
        return np.hstack((self.hub,np.array(rv)))
        #return np.array(rv)


def find_next_states4(obj1, obj2, state):
    flag=obj1.test(state)
    values=np.array(obj1.pw2)
    if flag==-99:
        #idx=np.argsort(values)
        #idx=np.flip(idx,0)[:]
        
        target_states=obj2.hub
    else:
        #candidates=obj2.get_path(flag)
        #target_states=candidates[:5]
        target_states=obj2.get_mp(flag,length=20)
    del obj1, obj2
    return target_states,flag 



class eval_states():
 
    def __init__(self, state_n, thres=0.97): 
        self.NI=state_n 
        #self.action_n=action_n
        self.thres=thres
        self.pw1=[]
        self.pn=0
        self.pw2=[]
        self.pw3=defaultdict(list)
        self.prev_state=-999
        self.mem={}
  
    def update(self, state): #policy network is a hidden-layer MLP. 
        self.pw1,tmp_max,activation, flag_new=matching(self.pw1,state, self.thres)
        
         
        if flag_new==True:
            self.pw2.append(1)
            index=int(len(self.pw2)-1)
            self.mem[index]=list(state.astype(float))
        else:
            self.pw2[tmp_max]=self.pw2[tmp_max]+1
            index=int(tmp_max)
        if self.prev_state>-10 and self.prev_state!=index:
            self.pw3[str(self.prev_state)].append(index)
        self.prev_state=index


    def test(self, state,strict=True):
        vec_n=state/np.linalg.norm(state)
        tmp=np.matmul(self.pw1,vec_n.T)
        max_p=np.argmax(tmp)
        max_v=np.amax(tmp)
        #print (max_v,max_p)
        if strict:
            if max_v>=self.thres:
                return max_p
            else:
            
                return -99
        else:
            return max_p


    def import_net(self, pw1,pw2,pw3):
        self.pw1=pw1
        self.pw2=pw2
        self.pw3=pw3

    def reset(self):
        self.prev_state=-999
 
