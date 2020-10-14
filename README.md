

# Reinforcement Learning With Low-Complexity Liquid State Machines

This repository contains code to "Reinforcement Learning With Low-Complexity Liquid State Machines" paper published in [Frontier Neuroscience](https://www.frontiersin.org/articles/10.3389/fnins.2019.00883/full).


# Instruction to run experiments


1. Create conda environment & activate it
```
conda env create -p <path> python=3.5
conda activate <path>
```

2. Install required packages
```
conda env update --file requirements.yml
```
alternatively
```
conda install cython=0.28.5 cloudpickle=0.5.5 tqdm=4.29.1 networkx=2.1 matplotlib=3.0.0 pytorch=0.4.1 -c pytorch
pip install gym[classic]==0.10.11 gym[atari]==0.10.11
```

4. Train model to balance cartpole
```
SEED=001993 bash -c 'python run_cartpole.py --checkpoint_dir CartPole_seed${SEED}_s1e3_n150_h32 --seed ${SEED} --train_steps 1e3 --n_neurons 150 --hidden_size 32'
```
Training takes around 1 1/4hr and requires 500MB for model storage. 
Model is trained for 100,000 steps in total which is equally divided into 100 sets (or training epochs). After each set, readout layer is saved for evaluation.
`run_cartpole.py` will generate log file ending with \*\_train.log. Refer to set of initial random seeds at the end of file to reproduce the same result in the paper.

5. Test model to balance cartpole
```
SEED=001993 bash -c 'python run_cartpole.py --checkpoint_dir ./checkpoints/CartPole_seed${SEED}_s1e3_n150_h32 --seed ${SEED} --train_steps 1e3 --n_neurons 150 --hidden_size 32 --test'
```
Above command runs evluation on saved models. Testing take 1/2hr and will generate log file ending with \*\_test.log
If you don't train model from scratch, checkpoints are available for download at 

6. Plot results from testing over multiple random seeds 
```
python plot_results.py -f \
./checkpoint/CartPole_seed001993_s1e3_n150_h32_test.log \
./checkpoint/CartPole_seed071198_s1e3_n150_h32_test.log \
./checkpoint/CartPole_seed109603_s1e3_n150_h32_test.log \
./checkpoint/CartPole_seed213556_s1e3_n150_h32_test.log \
./checkpoint/CartPole_seed383163_s1e3_n150_h32_test.log \
./checkpoint/CartPole_seed410290_s1e3_n150_h32_test.log \
./checkpoint/CartPole_seed619089_s1e3_n150_h32_test.log \
./checkpoint/CartPole_seed908818_s1e3_n150_h32_test.log
```
Above command calls a script which plots median cumulative reward from multiple testing logs over 100 training epoch.

7. Train model to play Pacman/Atari games 

The rest of the step is similar scripts to train model to balance the cartpole.

For Pacman, there are 3 mazes that are included in `./pacman/mazes/` that are `smallF3G1C0.maze`, `mediumF6G2C2.maze`, and `largeF6G1C0.maze`.
Train model to play Pacman in `smallF3G1C0.maze` using the following command:
```
SEED=001993 bash -c 'python run_pacman.py --checkpoint_dir Pacman_sF3G1C0_seed${SEED}_s5e3_n500_h128 --seed ${SEED} --maze_path ./pacman/mazes/smallF3G1C0.maze --train_steps 5e3 --n_neurons 500 --hidden_size 128'
```
Test model to play Pacman in `smallF3G1C0.maze` using the following command:
```
SEED=001993 bash -c 'python run_pacman.py --checkpoint_dir Pacman_sF3G1C0_seed${SEED}_s5e3_n500_h128 --seed ${SEED} --maze_path ./pacman/mazes/smallF3G1C0.maze --train_steps 5e3 --n_neurons 500 --hidden_size 128 --test'
```

Train model to play Pacman in `mediumF6G2C2.maze` using the following command:
```
SEED=001993 bash -c 'python run_pacman.py --checkpoint_dir Pacman_mF6G2C2_seed${SEED}_s5e3_n2000_h512 --seed ${SEED} --maze_path ./pacman/mazes/mediumF6G2C2.maze --train_steps 5e3 --n_neurons 2000 --hidden_size 512'
```
Test model to play Pacman in `mediumF6G2C2.maze` using the following command:
```
SEED=001993 bash -c 'python run_pacman.py --checkpoint_dir Pacman_mF6G2C2_seed${SEED}_s5e3_n2000_h512 --seed ${SEED} --maze_path ./pacman/mazes/mediumF6G2C2.maze --train_steps 5e3 --n_neurons 2000 --hidden_size 512 --test --test_step 1e4'
```

Train model to play in `largeF6G1C0.maze` using the following command:
```
SEED=001993 bash -c 'python run_pacman.py --checkpoint_dir Pacman_lF6G1C0_seed${SEED}_s3e4_n3000_h512 --seed ${SEED} --maze_path ./pacman/mazes/largeF6G1C0.maze --train_steps 3e4 --n_neurons 3000 --hidden_size 512'
```
Test model to play in `largeF6G1C0.maze` using the following command:
```
SEED=001993 bash -c 'python run_pacman.py --checkpoint_dir Pacman_lF6G1C0_seed${SEED}_s3e4_n3000_h512 --seed ${SEED} --maze_path ./pacman/mazes/largeF6G1C0.maze --train_steps 3e4 --n_neurons 3000 --hidden_size 512 --test --test_setp 1e4'
```

Train model to play 4 selected games using the following command:
```
SEED=001993 bash -c 'python run_atariram.py --env Boxing --checkpoint_dir Boxing_seed${SEED}_s5e3_n500_h128_t0p5 --seed ${SEED} --train_steps 5e3 --n_neurons 500 --hidden_size 128 --log_freq 10'
SEED=001993 bash -c 'python run_atariram.py --env Gopher --checkpoint_dir Gopher_seed${SEED}_s5e3_n500_h128_t0p5 --seed ${SEED} --train_steps 5e3 --n_neurons 500 --hidden_size 128'
SEED=001993 bash -c 'python run_atariram.py --env Freeway --checkpoint_dir Freeway_seed${SEED}_s5e3_n500_h128_t0p5 --seed ${SEED} --train_steps 5e3 --n_neurons 500 --hidden_size 128 --log_freq 10'
SEED=001993 bash -c 'python run_atariram.py --env Krull --checkpoint_dir Krull_seed${SEED}_s5e3_n500_h128_t0p5 --seed ${SEED} --train_steps 5e3 --n_neurons 500 --hidden_size 128'
```
Test model to play 4 selected games using the following command:
```
SEED=001993 bash -c 'python run_atariram.py --env Boxing --checkpoint_dir Boxing_seed${SEED}_s5e3_n500_h128_t0p5 --seed ${SEED} --train_steps 5e3 --n_neurons 500 --hidden_size 128 --log_freq 10 --test'
SEED=001993 bash -c 'python run_atariram.py --env Gopher --checkpoint_dir Gopher_seed${SEED}_s5e3_n500_h128_t0p5 --seed ${SEED} --train_steps 5e3 --n_neurons 500 --hidden_size 128 --test'
SEED=001993 bash -c 'python run_atariram.py --env Freeway --checkpoint_dir Freeway_seed${SEED}_s5e3_n500_h128_t0p5 --seed ${SEED} --train_steps 5e3 --n_neurons 500 --hidden_size 128 --log_freq 10 --test'
SEED=001993 bash -c 'python run_atariram.py --env Krull --checkpoint_dir Krull_seed${SEED}_s5e3_n500_h128_t0p5 --seed ${SEED} --train_steps 5e3 --n_neurons 500 --hidden_size 128 --test'
```

# Set of initial random seeds
001993
109603
619089
071198
383163
213556
410290
908818

# Citation
If you use this code in your work, please cite the following [paper](https://www.frontiersin.org/articles/10.3389/fnins.2019.00883/full)
```
@article{ponghiran2019reinforcement,
  title={Reinforcement learning with low-complexity liquid state machines},
  author={Ponghiran, Wachirawit and Srinivasan, Gopalakrishnan and Roy, Kaushik},
  journal={Frontiers in Neuroscience},
  volume={13},
  pages={883},
  year={2019},
  publisher={Frontiers}
}
```

