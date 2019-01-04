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
        self.pw2=obj.pw2
        #values=np.array(obj.pw2)
        #thres_v=np.mean(values)+np.std(values)*0.1
        #idx=np.where(values>thres_v)[0]
        #self.hub=idx
        #print ('hub',type(self.hub),len(self.hub)) 

    def make_net(self):
 
        #size=len(self.obj.pw2)
        #self.wts=np.zeros((size,size))
        
        #for xin in self.obj.pw3:
        #    temp=self.obj.pw3[xin]
             
        #    for yin in temp:
                #print (yin[0],yin[1])
                #self.wts[int(xin),yin]=self.wts[int(xin),yin]+1
        #        self.wts[yin[0], int(xin)]=self.wts[yin[0], int(xin)]+yin[1]
        #for xi,xin in enumerate(self.obj.pw2):  
        #    self.wts[xi,xi]=xin

        #values=np.sum(self.wts,axis=1)
        #thres_v=np.mean(values)+np.std(values)*0.8
        #print (values)
        values=np.array(self.obj.pw2)
        thres_v=np.mean(values)+np.std(values)*0.1
       
        idx=np.where(values>thres_v)[0]
        self.hub=idx
        print ('hub',type(self.hub),len(self.hub))
        



    def get_mp(self,node,length=5):
        idx_all=[]
        #print (node)
        temp=self.pw2[node]
        idx=np.argsort(temp)
        idx=np.flip(idx,0)
        idx_all.append(idx)
        
        
        rv=[]
        for xin in idx_all:
            rv.extend(xin[:length])
        #print ('node', node,'rv',temp)
        #return np.hstack((self.hub,np.array(rv)))
        #return np.array(rv)
        #return self.hub


def find_next_states4(obj1, obj2, state):
    #print (state)
    flag=obj1.test_del(state,False)
    #values=np.array(obj1.pw2)
    
    target_states=obj2.hub
       
    del obj1, obj2
    return target_states,flag 




 

class eval_states2():
 
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
  


    def update_value(self, state,rew,rw):

        if isinstance(self.prev_state,np.ndarray):
        
            del_state=state-self.prev_state
            self.pw1,tmp_max,activation, flag_new=matching(self.pw1,del_state, self.thres)
        
         
            if flag_new==True:
                self.pw2.append(rew)
                index=int(len(self.pw2)-1)
                self.mem[index]=list(del_state.astype(float))
            else:
                self.pw2[tmp_max]=self.pw2[tmp_max]+rw
                 
                index=int(tmp_max)


            self.prev_state=state


    def test(self, test_state,strict=True):
        if isinstance(self.prev_state,np.ndarray): 
            vec_n=test_state/np.linalg.norm(test_state)
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

    def test_del(self, test_state,strict=True):
        norm=np.linalg.norm(test_state)
        if norm==0:
            vec_n=test_state
        else:   
            vec_n=test_state/np.linalg.norm(test_state)
        tmp=np.matmul(self.pw1,vec_n.T)
        max_p=np.argmax(tmp)
        max_v=np.amax(tmp)
        #print (vec_n,tmp)
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

    def set(self,state):
        self.prev_state=state
