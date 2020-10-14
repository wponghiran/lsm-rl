
import argparse
import sys
import os
import pathlib
import tempfile
import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim

from atari_wrappers import make_atari_ram, wrap_atari_ram
from functools import reduce
from tqdm import tqdm

env_list = ['Boxing', 'Gopher', 'Freeway', 'Krull']

# Parse argument
parser = argparse.ArgumentParser(description='Training LSM with Q-Learning for playing Atari games', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--env_name', default='Boxing', choices=env_list, type=str, help='environment name')
parser.add_argument('--seed', default=1993, type=int, help='random seed')
parser.add_argument('--checkpoint_dir', default=None, help='name of checkpoint directory')
parser.add_argument('--overwrite', dest='overwrite', action='store_true', help='overwrite the existing log file')
# Additional arguments for q-learning
parser.add_argument('--lr', default=2e-4, type=float, help='learning rate')
parser.add_argument('--train_epoches', default=100, type=int, help='number of training epoches')
parser.add_argument('--train_steps', default=5e3, type=float, help='number of training steps per each training epoch')
parser.add_argument('--start_steps', default=1e2, type=float, help='number of training steps to start training')
parser.add_argument('--train_freq', default=1, type=int, help='frequency of training readout layer')
parser.add_argument('--final_eps', default=1e-1, type=float, help='final exploration rate')
parser.add_argument('--gamma', default=0.95, type=float, help='discount factor')
parser.add_argument('--log_freq', default=100, type=int, help='number of game episodes to log training progress')
parser.add_argument('--buffer_size', default=1e6, type=float, help='replay buffer size')
parser.add_argument('--batch_size', default=32, type=int, help='batch of experience replay')
parser.add_argument('--target_network_update_freq', default=1e4, type=float, help='frequency of updating target network')
parser.add_argument('--test', dest='test', action='store_true', help='test model without reward clipping')
parser.add_argument('--test_steps', default=1e4, type=float, help='number of testing steps per each training epoch')
parser.add_argument('--test_start', default=0, type=int, help='number of episode which testing begin')
parser.add_argument('--test_end', default=100, type=int, help='number of episode which testing begin')
# Additional arguments for liquid
parser.add_argument('--rate_scale', default=0.1, type=float, help='scaling factor of the input maximum firing rate')
parser.add_argument('--t_sim', default=0.500, type=float, help='single example simulation time')
parser.add_argument('--t_prb_start', default=0.000, type=float, help='time to start collecting spike activity')
parser.add_argument('--t_prb_stop', default=0.500, type=float, help='time to stop collecting spike activity')
parser.add_argument('--n_neurons', default=500, type=int, help='total number of LSM neurons')
parser.add_argument('--hidden_size', default=128, type=int, help='number of hidden neurons in readout for dimensional reduction')
parser.add_argument('--warmup_steps', default=100, type=int, help='number of the warmup samples for LSM before actual training')

def main():

    # Parse argument
    global args
    args = parser.parse_args()
    
    # Add extra arguments 
    args.total_steps = int(args.train_epoches*args.train_steps)
    args.test_steps = int(args.test_steps)
    args.checkpoint_freq = int(args.train_steps)
    args.buffer_size = int(args.buffer_size) if args.buffer_size < args.total_steps else args.total_steps
    args.exploration_steps = int(0.2*args.total_steps)
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Initialize random seed
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.device == 'cuda':    torch.cuda.manual_seed_all(args.seed)

    # Create environment
    env = make_atari_ram('{}-ramNoFrameskip-v4'.format(args.env_name))

    if args.test:
        env = wrap_atari_ram(env, episode_life=False, clip_rewards=False)
        test(env)
    else:
        env = wrap_atari_ram(env, episode_life=True, clip_rewards=True)
        learn(env)
    
    # Close environment
    env.close()

# Linear scheduler for epsilon-greedy policy
#   This class is replicated from https://github.com/openai/baselines/blob/master/baselines/common/schedules.py
class LinearSchedule(object):
    def __init__(self, schedule_steps, final_p, initial_p=1.0):
        """Linear interpolation between initial_p and final_p over
        schedule_steps. After this many steps pass final_p is
        returned.
        Parameters
        ----------
        schedule_steps: int
            Number of steps for which to linearly anneal initial_p
            to final_p
        initial_p: float
            initial output value
        final_p: float
            final output value
        """
        self.schedule_steps = float(schedule_steps)
        self.final_p = final_p
        self.initial_p = initial_p
    def value(self, t):
        """See Schedule.value"""
        fraction = min(float(t) / self.schedule_steps, 1.0)
        return self.initial_p + fraction * (self.final_p - self.initial_p)

# Replay buffer for experience replay
#   This class is modified from https://github.com/openai/baselines/blob/master/baselines/deepq/replay_buffer.py 
class ReplayBuffer(object):
    def __init__(self, size):
        """Create Replay buffer.
        Parameters
        ----------
        size: int
            Max number of transitions to store in the buffer. When the buffer
            overflows the old memories are dropped.
        """
        self._storage = []
        self._maxsize = size
        self._next_idx = 0
    def __len__(self):
        return len(self._storage)
    def add(self, obs_t, action, reward, obs_tp1):
        data = (obs_t, action, reward, obs_tp1)
        if self._next_idx >= len(self._storage):
            self._storage.append(data)
        else:
            self._storage[self._next_idx] = data
        self._next_idx = (self._next_idx + 1) % self._maxsize
    def _encode_sample(self, idxes):
        obses_t, actions, rewards, obses_tp1 = [], [], [], []
        for i in idxes:
            data = self._storage[i]
            obs_t, action, reward, obs_tp1 = data
            obses_t.append(obs_t)
            actions.append(action)
            rewards.append(reward)
            obses_tp1.append(obs_tp1)
        return obses_t, actions, rewards, obses_tp1
    def sample(self, batch_size):
        """Sample a batch of experiences.
        Parameters
        ----------
        batch_size: int
            How many transitions to sample.
        Returns
        -------
        obs_batch: np.array
            batch of observations
        act_batch: np.array
            batch of actions executed given obs_batch
        rew_batch: np.array
            rewards received as results of executing act_batch
        next_obs_batch: np.array
            next set of observations seen after executing act_batch
        done_mask: np.array
            done_mask[i] = 1 if executing act_batch[i] resulted in
            the end of an episode and 0 otherwise.
        """
        idxes = [random.randint(0, len(self._storage) - 1) for _ in range(batch_size)]
        return self._encode_sample(idxes)

# Simple non-spiking model for readout layer
class MLP(nn.Module):
    def __init__(self, inp_size, hidden_size, outp_size):
        super(MLP, self).__init__()
        list_m = [
            nn.Linear(inp_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, outp_size),
                ]
        self.classifier = nn.Sequential(*list_m)
    def forward(self, inp):
        outp = inp.view(inp.size(0), -1)
        outp = self.classifier(outp)
        return outp

# Wrapper for readout layer which is trained with Q-learning
#   This class allows easy load and store for trained readout weights for evaluation
class ReadoutWrapper(object):
    def __init__(self, network, optimizer, f, criterion=nn.SmoothL1Loss, gamma=1.0, grad_clipping=1, obs_size=0, hidden_size=0, act_size=0):
        self.policy_net = network(inp_size=obs_size, hidden_size=hidden_size, outp_size=act_size).to(args.device)
        f.write('{}\n'.format(self.policy_net))
        self.target_net = network(inp_size=obs_size, hidden_size=hidden_size, outp_size=act_size).to(args.device)
        self.target_net.eval() 
        self.optimizer = optimizer(self.policy_net.parameters(), lr=args.lr, eps=1e-6)
        self.criterion = criterion().to(args.device)
        self.gamma = gamma
        self.grad_clipping = grad_clipping
    def load(self, path='./checkpoint.pt'):
        if os.path.isfile(path):
            checkpoint = torch.load(path,map_location='cpu')
            self.policy_net.load_state_dict(checkpoint['policy_net'])
            self.target_net.load_state_dict(checkpoint['target_net'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
        else:
            print('=> No checkpoint found at {}'.format(path))
            sys.exit(0)
    def save(self, f=sys.stdout, path='./checkpoint.pt'):
        state = {
                'policy_net' : self.policy_net.state_dict(),
                'target_net' : self.target_net.state_dict(),
                'optimizer' : self.optimizer.state_dict()
                }
        torch.save(state, path)

# Function to select an action based on epsilon-greedy policy
def act(obs, readout, act_size, update_eps=0):
    if random.random() > update_eps:
        with torch.no_grad():
            return readout.policy_net(obs).max(1)[1].item()
    else:
        return random.randrange(act_size)

# Train function
def train(obses_t, actions, rewards, obses_tp1, readout, batch_size):

    # Compute mask for non-final states and create batch of inptus
    done_obs_mask = torch.tensor(tuple(map(lambda s: s is not None, obses_tp1))).to(args.device)
    done_next_obs_mask = torch.cat([s for s in obses_tp1 if s is not None]).to(args.device)
    obses_t = torch.cat(obses_t).to(args.device)
    actions = torch.cat(actions).to(args.device)
    rewards = torch.cat(rewards).to(args.device)

    # Compute Q(s_t, a)
    state_action_values = readout.policy_net(obses_t).gather(1, actions)

    # Compute V(s_{t+1}) for all next states.
    next_state_values = torch.zeros(batch_size).to(args.device)
    next_state_values[done_obs_mask] = readout.target_net(done_next_obs_mask).max(1)[0].detach()
    # Compute expected Q values based on Bellman equation
    expected_state_action_values = torch.add(torch.mul(next_state_values,readout.gamma),rewards)

    # Compute loss
    loss = readout.criterion(state_action_values, expected_state_action_values.unsqueeze(1))

    # Optimize the readout
    readout.optimizer.zero_grad()
    loss.backward()
    for param in readout.policy_net.parameters():
        param.grad.data.clamp_(-readout.grad_clipping, readout.grad_clipping)
    readout.optimizer.step()

# Update target network function
def update_target(readout):
    readout.target_net.load_state_dict(readout.policy_net.state_dict())

# Pre-processing function 
def pre_process(obs):
    return torch.from_numpy(obs).float()

def learn(env):

    # Open log file to write
    if args.checkpoint_dir is None:
        # Use standard output if checkpoint directory is not specified
        f = sys.stdout
    else:
        # Use checkpoint directory name as training log file
        log_name = '{}_train.log'.format(args.checkpoint_dir)
        # Do not proceed if training log file exists
        if not args.overwrite and os.path.isfile(log_name):
            print('==> File {} exists!'.format(log_name))
            sys.exit(0) 
        f = open(log_name, 'w', buffering=1)
    
    # Print all input arguments to training log file
    f.write(str(args)+'\n')

    # Crate liquid model
    import model_lsm
    model_spike = model_lsm.LSM(f=f,
            t_sim=args.t_sim, t_prb=(args.t_prb_start,args.t_prb_stop),
            inp_size=int(reduce(lambda x, y: x*y, env.observation_space.shape)), rate_scale=args.rate_scale, 
            n_neurons=args.n_neurons, 
            k={'pe':1,'pi':0,'ei':4},
            w={'pe':0.5,'pi':0.0,
               'ee':0.05,'ei':0.25,
               'ie':-0.3,'ii':-0.01}) 
    model_spike.to(args.device)
    model_spike.eval()
    f.write('{}\n'.format(model_spike))

    # Create readout
    readout = ReadoutWrapper(f = f,
            network = MLP,
            optimizer = torch.optim.RMSprop,
            criterion = nn.SmoothL1Loss,
            gamma = args.gamma,
            grad_clipping = 1.0,
            obs_size = model_spike.n_e_neurons,
            hidden_size = args.hidden_size,
            act_size = env.action_space.n
            )
    # Create the replay buffer
    replay_buffer = ReplayBuffer(args.buffer_size)
    # Create the schedule for exploration starting from 1.0
    exploration = LinearSchedule(schedule_steps=args.exploration_steps, initial_p=1.0, final_p=args.final_eps)

    # Initialize the network parameters and copy them to the target network
    update_target(readout)

    # Start training
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create directory for saving result
        if args.checkpoint_dir is None:
            # Use temp directory if checkpoint directory is not specified
            args.checkpoint_dir = temp_dir
        else: 
            pathlib.Path(os.path.abspath(args.checkpoint_dir)).mkdir(parents=True, exist_ok=True)
            f.write('==> Checkpoint path : {}\n'.format(args.checkpoint_dir))
        
        # Reset the environment
        obs = env.reset()
        obs = pre_process(obs)
        with torch.no_grad():
            sumspike_e = model_spike(obs.to(args.device))
            obs = sumspike_e
        inp = obs.unsqueeze(0)

        # Warm-up LSM 
        for t in range(args.warmup_steps):
            new_obs, rew, done, _ = env.step(random.randrange(env.action_space.n))
            new_obs = pre_process(new_obs)
            with torch.no_grad():
                sumspike_e = model_spike(new_obs.to(args.device))
                new_obs = sumspike_e
            new_inp = None if done else new_obs.unsqueeze(0)
            obs = new_obs
            inp = new_inp
            
            # Reset environment
            if done:
                obs = env.reset()
                obs = pre_process(obs)
                with torch.no_grad():
                    sumspike_e = model_spike(obs.to(args.device))
                    obs = sumspike_e
                inp = obs.unsqueeze(0)

        # Declare list for storing cumulative rewards for each game episode and episode counter
        #   Note that game episode is different from training epoch
        #   Game episode is a duration when agent start playing the game until the game concludes
        #   Training epoch is a duration when agent is trained and agent state is saved for evaluation
        episode_rewards = [0.0]
        num_episodes = 1
        
        # Reset the again before start actual training
        obs = env.reset()
        obs = pre_process(obs)
        with torch.no_grad():
            sumspike_e = model_spike(obs.to(args.device))
            obs = sumspike_e
        inp = obs.unsqueeze(0)

        for t in tqdm(range(args.total_steps)):
            # Update exploration to the newest value
            update_eps = exploration.value(t)

            # Take action based on epsilon-policy
            action = act(inp, readout, env.action_space.n, update_eps)
            new_obs, rew, done, _ = env.step(action)
            new_obs = pre_process(new_obs)
            with torch.no_grad():
                sumspike_e = model_spike(new_obs.to(args.device))
                new_obs = sumspike_e
            new_inp = None if done else new_obs.unsqueeze(0)

            # Track episode reward
            episode_rewards[-1] += rew

            # Store state transition into replay buffer and move on to next state
            replay_buffer.add(inp, torch.tensor([[action]]), torch.tensor([rew],dtype=torch.float), new_inp)
            obs = new_obs
            inp = new_inp

            # Minimize the error in Bellman's equation on a batch sampled from replay buffer
            if t > args.start_steps and t % args.train_freq == 0:
                obses_t, actions, rewards, obses_tp1 = replay_buffer.sample(args.batch_size)
                train(obses_t, actions, rewards, obses_tp1, readout, args.batch_size)

            # Update target network periodically
            if t > args.start_steps and t % args.target_network_update_freq == 0:
                update_target(readout)
            
            # Reset environment
            if done:
                obs = env.reset()
                obs = pre_process(obs)
                with torch.no_grad():
                    sumspike_e = model_spike(obs.to(args.device))
                    obs = sumspike_e
                inp = obs.unsqueeze(0)
                # Print progress to log file for tracking performance during training
                if num_episodes % args.log_freq == 0:
                    f.write('step {}   episode {:.3f}   avg_reward {:.3f}   max_reward {:.3f}   percent_explore {}%\n'.format(t, num_episodes, np.mean(episode_rewards[-args.log_freq:]), np.max(episode_rewards[-args.log_freq:]), int(100*exploration.value(t))))
                # Update cumulative reward list and episode counter
                episode_rewards.append(0.0)
                num_episodes += 1 
            
            # Save model for evaluation
            if t % args.checkpoint_freq == 0:
                readout.save(f, os.path.join(args.checkpoint_dir, '{}_checkpoint_{}.pt'.format(args.env_name, t)))
                f.write('=> Save checkpoint at timestep {}\n'.format(t))
        
    # Close training log file
    if not args.checkpoint_dir is None:
        f.close()

def test(env):

    # Read train log file to get saved model path
    checkpoint_steps = []
    with open('{}_train.log'.format(args.checkpoint_dir), 'r') as f:
        for line in f.readlines():
            chunks = line.strip().split()
            if chunks[0] != '=>':    continue
            checkpoint_steps.append(chunks[-1])

    # Open test log file to write
    # log_name='{}_test.log'.format(args.checkpoint_dir)
    log_name='{}_test{}to{}.log'.format(args.checkpoint_dir, args.test_start, args.test_end)
    if not args.overwrite and os.path.isfile(log_name):
        print('==> File {} exists!'.format(log_name))
        sys.exit(0) 
    f = open(log_name, 'w', buffering=1)

    # Crate liquid
    import model_lsm
    model_spike = model_lsm.LSM(f=f,
            t_sim=args.t_sim, t_prb=(args.t_prb_start,args.t_prb_stop),
            inp_size=int(reduce(lambda x, y: x*y, env.observation_space.shape)), rate_scale=args.rate_scale, 
            n_neurons=args.n_neurons, 
            k={'pe':1,'pi':0,'ei':4},
            w={'pe':0.5,'pi':0.0,
               'ee':0.05,'ei':0.25,
               'ie':-0.3,'ii':-0.01}) 
    model_spike.to(args.device)
    model_spike.eval()
    f.write('{}\n'.format(model_spike))

    # Create readout
    readout = ReadoutWrapper(f = f,
            network = MLP,
            optimizer = torch.optim.RMSprop,
            criterion = nn.SmoothL1Loss,
            gamma = 0,
            grad_clipping = 0,
            obs_size = model_spike.n_e_neurons,
            hidden_size = args.hidden_size,
            act_size = env.action_space.n
            )

    # Reset the environment
    obs = env.reset()
    obs = pre_process(obs)
    with torch.no_grad():
        sumspike_e = model_spike(obs.to(args.device))
        obs = sumspike_e
    inp = obs.unsqueeze(0)
        
    # Warm-up LSM 
    for t in range(args.warmup_steps):
        new_obs, rew, done, _ = env.step(random.randrange(env.action_space.n))
        new_obs = pre_process(new_obs)
        with torch.no_grad():
            sumspike_e = model_spike(new_obs.to(args.device))
            new_obs = sumspike_e
        new_inp = None if done else new_obs.unsqueeze(0)
        obs = new_obs
        inp = new_inp
        
        # Reset environment
        if done:
            obs = env.reset()
            obs = pre_process(obs)
            with torch.no_grad():
                sumspike_e = model_spike(obs.to(args.device))
                obs = sumspike_e
            inp = obs.unsqueeze(0)

    # Run actual testing for every checkpoint 
    # for checkpoint_step in tqdm(checkpoint_steps):
    for checkpoint_step in tqdm(checkpoint_steps[args.test_start:args.test_end]):
        # Load readlayer for checkpoint
        # readout.load(os.path.join(args.checkpoint_dir, '{}_checkpoint_{}.pt'.format(args.env_name, checkpoint_step)))
        readout.load(os.path.join(args.checkpoint_dir, '{}_checkpoint_{}.pt'.format(args.env_name, checkpoint_step)))
        
        # Declare list for storing cumulative rewards for each game episode and episode counter
        episode_rewards = [0.0]
        num_episodes = 1

        # Reset environment
        obs = env.reset()
        obs = pre_process(obs)
        with torch.no_grad():
            sumspike_e = model_spike(obs.to(args.device))
            obs = sumspike_e
        inp = obs.unsqueeze(0)

        for t in range(args.test_steps):
            # Take action based on epsilon-policy with probability of perform random action = 0.05% 
            action = act(inp, readout, env.action_space.n, 0.05)
            new_obs, rew, done, _ = env.step(action)
            new_obs = pre_process(new_obs)
            with torch.no_grad():
                sumspike_e = model_spike(new_obs.to(args.device))
                new_obs = sumspike_e
            new_inp = None if done else new_obs.unsqueeze(0)
            
            # Track episode reward
            episode_rewards[-1] += rew
            
            # Move on to the next state 
            obs = new_obs
            inp = new_inp
            
            # Reset environment
            if done:
                obs = env.reset()
                obs = pre_process(obs)
                with torch.no_grad():
                    sumspike_e = model_spike(obs.to(args.device))
                    obs = sumspike_e
                inp = obs.unsqueeze(0)
                # Update cumulative reward list and episode counter
                episode_rewards.append(0.0)
                num_episodes += 1 
               
        # Print average cumulative reward per episodes to log file for tracking performance
        # Drop last value in list of episode_rewards as the episode may not yet complete
        if num_episodes == 1:
            f.write('step {}   num_episode {}   avg_reward {:.3f}   max_reward {:.3f}\n'.format(checkpoint_step, num_episodes-1, np.mean(episode_rewards), np.max(episode_rewards)))
        else:
            f.write('step {}   num_episode {}   avg_reward {:.3f}   max_reward {:.3f}\n'.format(checkpoint_step, num_episodes-1, np.mean(episode_rewards[:-1]), np.max(episode_rewards[:-1])))
            
    # Close testing log file
    if not args.checkpoint_dir is None:
        f.close()

if __name__ == '__main__':
    main()
