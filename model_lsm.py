
import torch
import torch.nn as nn
import random

def init_w(inp_size, n_neurons, n_e_neurons, n_i_neurons, w_ae, w_ai, w, k):
    # Initialize weight matrix for input group -> liquid.
    #   w_pe represents connections from input neurons -> excitatory neurons. It has a size of (inp_size, n_e_neurons). 
    #   We initialize weight of connection matrix one column at a time. Each column will have approximately k['pe'] non-zero elements.
    #   This is to make sure that there are k['pe'] connections from input neurons to excitatory neurons.
    w_pe = torch.zeros(inp_size, n_e_neurons)
    for col in range(n_e_neurons):
        # We first create a column of 0 and 1 with probability of finding 1 = k['pe']/inp_size.
        w_pe[:,col] = (torch.le(torch.rand(inp_size),k['pe']/inp_size)).float()
        # We then multiply column with random weight between 0 and w['pe'].
        w_pe[:,col] *= torch.rand(inp_size)*w['pe']
    #   w_pi represents connections from input neurons -> inhibitory neurons. It is initialized similar to w_pe.
    w_pi = torch.zeros(inp_size, n_i_neurons)
    for col in range(n_i_neurons):
        w_pi[:,col] = (torch.le(torch.rand(inp_size),k['pi']/inp_size)).float()*torch.rand(inp_size)*w['pi']

    # Initialize weight matrix liquid -> liquid
    #   We would like to exploit dynamic from random connections; however, we also want to avoid chaotic activity that arises when 
    #   (1) excitatory neurons connects to each other and form a loop with always leads positive drift in membrane potential
    #   (2) excitatory neurons connects to itself and repeatedly get excited from its activity
    #   We avoid (1) situation by creating w_ee based on a product of w_ei and w_ie
    #   Product of w_ei and w_ie can be thought as inhibitory connection from one excitatory nueron to other excitatory neurons
    #   If w_ei x w_ie is made statistically greater or equivalent to w_ee, large positive drift in membrane potential is not likely to happen.
    #   We avoid (2) situation by creating w_ei and w_ie together. 
    #   Row of w_ei and column w_ie which corresponds to the same excitatory neurons are created such that they have non-zero elements at different indices.
    #   As a result, when w_ei x w_ie and w_ee always have zero diagonal value
    w_ei = torch.zeros(n_e_neurons, n_i_neurons)
    w_ie = torch.zeros(n_i_neurons, n_e_neurons)
    #   Create row of w_ei and column of w_ie which corresponds to the same excitatory neurons together
    for i in range(n_e_neurons):
        conn_all = torch.rand(n_i_neurons)<(2*k['ei']/n_i_neurons)
        mask_1 = torch.rand(n_i_neurons)<0.5
        mask_2 = ~mask_1
        w_ei[i,:] = conn_all&mask_1
        w_ie[:,i] = conn_all&mask_2
    #   Create w_ee based on product of w_ei and ie
    w_ee = torch.mm(w_ei, w_ie)
    w_ii = torch.mm(w_ie, w_ei)
    w_ee *= torch.rand(n_e_neurons, n_e_neurons)*w['ee']
    w_ei *= torch.rand(n_e_neurons, n_i_neurons)*w['ei']
    w_ie *= torch.rand(n_i_neurons, n_e_neurons)*w['ie']
    w_ii *= torch.rand(n_i_neurons, n_i_neurons)*w['ii']

    # Copy weight to model parameters 
    w_ae.copy_(torch.cat((w_pe,w_ee,w_ie), dim=0))
    w_ai.copy_(torch.cat((w_pi,w_ei,w_ii), dim=0))

    # Uncomment below to visualize spectrum of connection matrix
    # import matplotlib.pyplot as plt
    # import numpy as np
    # import math
    # eig = np.linalg.eigvals(torch.cat((torch.cat((w_ee, w_ie), dim=0), torch.cat((w_ei, w_ii), dim=0)), dim=1).numpy())
    # plt.scatter(np.real(eig), np.imag(eig))
    # plt.axes().set_aspect(1)
    # plt.show()

# LSM module
class LSM(nn.Module):
    def __init__(self,f,
            t_sim, t_prb,
            inp_size, rate_scale,
            n_neurons, 
            w={},
            k={},
                ):
        super(LSM, self).__init__()

        # Initialize LSM parameters
        self.inp_size = inp_size
        self.rate_scale = rate_scale
        
        self.n_neurons = n_neurons
        self.n_e_neurons = int(0.8*n_neurons)
        self.n_i_neurons = int(0.2*n_neurons)
        assert self.n_neurons == (self.n_e_neurons+self.n_i_neurons), 'number of neurons is not divisible by 5'
        
        # Compute total timesteps to simulate and timestep to start and stop probing result
        self.dt = 0.001
        self.tau_mem = 0.020
        self.ts_sim = int(t_sim/self.dt)
        self.ts_prb_start = int(t_prb[0]/self.dt) 
        self.ts_prb_stop = int(t_prb[1]/self.dt) 
        self.ts_prb = self.ts_prb_stop-self.ts_prb_start

        f.write('inp_size : {}\n'.format(self.inp_size))
        f.write('n_neurons : {}\n'.format(self.n_neurons))
        # Initilize connection matrices
        #   Connection from all neurons (input + excitatory + inhibitory) -> excitatory neurons
        self.register_buffer('w_ae', torch.ones(self.inp_size+self.n_neurons, self.n_e_neurons))
        #   Connection from all neurons (input + excitatory + inhibitory) -> inhibitory neurons
        self.register_buffer('w_ai', torch.ones(self.inp_size+self.n_neurons, self.n_i_neurons))
        # Note about variable k and w
        # 
        #   k is a dictionary which contains mapping of two letter keys and one integer value (for example, k = {'ei':3})
        #   k specifies number of connection that one neurons receives from other neurons 
        #   Specifically, k['ei'] = 3 means that each neurons in i group will receive approximately 3 connections from neurons in e group
        #  
        #   Similarly, w is a dictionary which contains mapping of two letter keys and one floating value (for example, w = {'ei':0.6})
        #   w specifies weight of connection that one neurons receives from other neurons 
        init_w(self.inp_size, self.n_neurons, self.n_e_neurons, self.n_i_neurons, self.w_ae, self.w_ai, w, k)

        # Initialize membrane potential and threshold voltage
        self.register_buffer('vmem', torch.zeros(self.n_neurons))
        self.register_buffer('vth', torch.empty(self.n_neurons))
        self.vth.fill_(0.50)
        self.rate_scale = rate_scale
        # Initialize spikes from liquid at the current and previous timestep 
        self.register_buffer('spike_l', torch.zeros(self.n_neurons, dtype=torch.uint8))
        self.register_buffer('spike_l_prev', torch.zeros(self.n_neurons, dtype=torch.uint8))
        
        # Initialize variables for collecting out 
        self.register_buffer('sumspike_e', torch.empty(self.n_e_neurons))
        self.spike_p_norm = (t_prb[1]-t_prb[0])/self.dt
        self.spike_e_norm = (t_prb[1]-t_prb[0])/self.dt
        self.spike_i_norm = (t_prb[1]-t_prb[0])/self.dt

    def forward(self, inp):
    
        # Scale input 
        inp = inp*self.rate_scale

        # Clear variables used for storing spiking activity
        self.sumspike_e.fill_(0.)

        # Simulation over time
        for ts in range(0, self.ts_sim):
            # Generate input spikes based on Poisson distribution
            spike_p = torch.le(torch.rand_like(inp), inp)

            # Evaluate 0st-order synaptic
            #   dv/vt = -v/tau + W_E*spike_E + W_I*spike_I
            #   spike_p represent input spike
            #   spike_l[:self.n_e_neurons] represents spike for excitatory neurons
            #   spike_l[self.n_e_neurons:] represents spike for inhibitory neurons
            self.vmem += (-self.vmem)/self.tau_mem*self.dt 
            self.vmem += torch.cat( (torch.matmul(torch.cat((spike_p, self.spike_l_prev[:self.n_e_neurons], self.spike_l[self.n_e_neurons:]), dim=0).float(), self.w_ae) \
                                    ,torch.matmul(torch.cat((spike_p, self.spike_l[:self.n_e_neurons], self.spike_l_prev[self.n_e_neurons:]), dim=0).float(), self.w_ai)), dim=0 )
            
            # Copy past spike to create a delay of recurrent connection equal to 1 timesteps
            self.spike_l_prev = self.spike_l

            # Check if any membrane potential reaches threshold and spikes
            self.spike_l = torch.ge(self.vmem, self.vth)
            
            # Reset membrane potential if there is postsynpatic spike
            self.vmem[self.spike_l] = 0.0

            # Collection spiking activity
            if ts >= self.ts_prb_start and ts < self.ts_prb_stop:
                self.sumspike_e += self.spike_l[:self.n_e_neurons].float()

        # Return normalized spiking activitiy
        return self.sumspike_e/self.spike_p_norm

    def extra_repr(self):
        return 'inp_size={}, n_neurons={}'.format(self.inp_size, self.n_neurons)

if __name__ == '__main__':
    # Function for testing model
    import sys
    model_spike = LSM(f=sys.stdout,
            t_sim=0.1, t_prb=(0.0,0.1),
            inp_size=10, rate_scale=0.1, 
            n_neurons=100,
            k={'pe':3,'pi':0,'ei':4},
            w={'pe':0.6,'pi':0.0,
               'ee':0.2,'ei':0.2,
               'ie':-0.5,'ii':-0.2}) 
    # Without cuda()
    outp = model_spike(torch.rand(10))
    # With cuda()
    model_spike.cuda()
    outp = model_spike(torch.rand(10).cuda())
    input('...')
