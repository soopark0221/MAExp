import argparse
from pickle import TRUE
import torch
import time
import os
import numpy as np
from gym.spaces import Box, Discrete
from pathlib import Path
from torch.autograd import Variable
from tensorboardX import SummaryWriter
from utils.make_env import make_env
from utils.buffer import ReplayBuffer, BootaBuffer
from utils.env_wrappers import SubprocVecEnv, DummyVecEnv
from algorithms.maddpg import MADDPG
from algorithms.swagmaddpg import SWAGMADDPG
from algorithms.maddpg_share import MADDPG_Share

USE_CUDA = True if torch.cuda.is_available() else False  # torch.cuda.is_available()

def make_parallel_env(env_id, n_rollout_threads, seed, discrete_action):
    def get_env_fn(rank):
        def init_env():
            discrete_action=True
            env = make_env(env_id, discrete_action=discrete_action)
            env.seed(seed + rank * 1000)
            np.random.seed(seed + rank * 1000)
            return env
        return init_env
    if n_rollout_threads == 1:
        return DummyVecEnv([get_env_fn(0)])
    else:
        return SubprocVecEnv([get_env_fn(i) for i in range(n_rollout_threads)])

def run(config):
    #os.environ["CUDA_VISIBLE_DEVICES"] = config.gpu_no
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    model_dir = Path('./models') / config.env_id / config.model_name

    if not model_dir.exists():
        curr_run = 'run1'
    else:
        exst_run_nums = [int(str(folder.name).split('run')[1]) for folder in
                         model_dir.iterdir() if
                         str(folder.name).startswith('run')]
        if len(exst_run_nums) == 0:
            curr_run = 'run1'
        else:
            curr_run = 'run%i' % (max(exst_run_nums) + 1)
    run_dir = model_dir / curr_run
    log_dir = run_dir / 'logs'
    os.makedirs(log_dir)
    logger = SummaryWriter(str(log_dir))
    with open(run_dir / 'config.txt','a') as f:
        f.write(str(config))

    if not USE_CUDA:
        torch.set_num_threads(config.n_training_threads)
    env = make_parallel_env(config.env_id, config.n_rollout_threads, config.seed,
                            config.discrete_action)

    t = 0
    epi_reward=0
    adv_reward=0
    
    if config.alg == 'maddpg':
        maddpg = MADDPG.init_from_env(env, agent_alg=config.agent_alg,
                                  adversary_alg=config.adversary_alg,
                                  tau=config.tau,
                                  lr=config.lr,
                                  swag_lr=config.swag_lr,
                                  swag_start=config.swag_start,
                                  lr_cycle=config.lr_cycle,
                                  hidden_dim=config.hidden_dim)
    elif config.alg == 'share':
        maddpg=MADDPG_Share.init_from_env(env, agent_alg=config.agent_alg,
                                  adversary_alg=config.adversary_alg,
                                  tau=config.tau,
                                  lr=config.lr,
                                  swag_lr=config.swag_lr,
                                  hidden_dim=config.hidden_dim)

    alg_types = [config.adversary_alg if atype == 'adversary' else config.agent_alg for
                     atype in env.agent_types]
    if 'Boota' in alg_types:
        replay_buffer = BootaBuffer(config.buffer_length, maddpg.nagents,
                                 [obsp.shape[0] for obsp in env.observation_space],
                                 [acsp.shape[0] if isinstance(acsp, Box) else acsp.n
                                  for acsp in env.action_space])
    else:
        replay_buffer = ReplayBuffer(config.buffer_length, maddpg.nagents,
                                 [obsp.shape[0] for obsp in env.observation_space],
                                 [acsp.shape[0] if isinstance(acsp, Box) else acsp.n
                                  for acsp in env.action_space])

    for ep_i in range(0, config.n_episodes, config.n_rollout_threads):
        
        epi_reward=0
        adv_reward=0
        ag_reward=0
        c_loss, a_loss= None, None
        total_closs, total_aloss = 0, 0
        if 'Boota' in alg_types:
            actor_ids=[np.random.randint(5) if alg=='Boota' else None for alg in alg_types]
            maddpg.actor_ids=actor_ids

        obs = env.reset()
        # obs.shape = (n_rollout_threads, nagent)(nobs), nobs differs per agent so not tensor
        maddpg.prep_rollouts(device='cpu')

        explr_pct_remaining = max(0, config.n_exploration_eps - ep_i) / config.n_exploration_eps
        maddpg.scale_noise(config.final_noise_scale + (config.init_noise_scale - config.final_noise_scale) * explr_pct_remaining)
        maddpg.reset_noise()

        if 'SWAG' in alg_types:
            if ep_i >= config.swag_start and (ep_i-config.swag_start) % config.collect_freq == 0:
                maddpg.collect_params()  # collect actor network params 
            if ep_i >= config.swag_start and (ep_i-config.swag_start) % config.sample_freq == 0:
                maddpg.sample_params(scale = config.scale)  # update path collecting model
        for et_i in range(config.episode_length):
            # rearrange observations to be per agent, and convert to torch Variable
            torch_obs = [Variable(torch.Tensor(np.vstack(obs[:, i])),
                                requires_grad=False)
                        for i in range(maddpg.nagents)]
            # get actions as torch Variables
            if 'SWAG' in alg_types:
                if ep_i < (config.swag_start+config.sample_freq):
                    torch_agent_actions = maddpg.step_maddpg(torch_obs, explore=False)
                elif ep_i >= (config.swag_start+config.sample_freq):
                    torch_agent_actions = maddpg.step(torch_obs, explore=False)
            else:
                torch_agent_actions = maddpg.step(torch_obs, explore=True)

            # convert actions to numpy arrays
            agent_actions = [ac.data.detach().cpu().numpy() for ac in torch_agent_actions]
            # rearrange actions to be per environment
            actions = [[ac[i] for ac in agent_actions] for i in range(config.n_rollout_threads)]
            next_obs, rewards, dones, infos = env.step(actions)
            if config.display and (ep_i+1)%500==0:
                import time
                time.sleep(0.05)
                env.render()
            if 'Boota' in alg_types:
                replay_buffer.push(obs, agent_actions, rewards, next_obs, dones, actor_ids)
            else:
                replay_buffer.push(obs, agent_actions, rewards, next_obs, dones)
            
            obs = next_obs
            epi_reward+=np.sum(np.array(rewards))
            adv_reward+=np.sum(np.array(rewards[0][:3]))
            if maddpg.nagents>3:
                ag_reward+=np.sum(np.array(rewards[0][3:]))

            t += config.n_rollout_threads
            

            if (ep_i>=config.n_epi_before_train and
                (t % config.steps_per_update) < config.n_rollout_threads):
                if USE_CUDA:
                    maddpg.prep_training(device='gpu')
                else:
                    maddpg.prep_training(device='cpu')
                for u_i in range(config.n_rollout_threads):
                    total_closs, total_aloss=0,0
                    for a_i in range(maddpg.nagents):
                        if 'Boota' in alg_types:
                            sample = replay_buffer.sample_boot(config.batch_size,
                                                    to_gpu=USE_CUDA)
                        else:
                            sample = replay_buffer.sample(config.batch_size,
                                                    to_gpu=USE_CUDA)

                        c_loss, a_loss, lr =maddpg.update(sample, a_i, logger=logger)
                        total_closs+=float(c_loss)
                        total_aloss+=float(a_loss)
                    if (ep_i+1)%100==0:
                        if env.envs[0].shared_reward:
                            print("Episode %i, epi_reward= %6.4f" % (ep_i, epi_reward/maddpg.nagents))

                        else:  
                            print("Episode %i, adv_reward= %6.4f & agent_rew= %6.4f" % (ep_i, adv_reward, ag_reward))
                        
                        print(f"c_loss: {round(total_closs/maddpg.nagents,4)} a_loss: {round(total_aloss/maddpg.nagents,4)}")

                    maddpg.update_all_targets()
                maddpg.prep_rollouts(device='cpu')

        if logger is not None:
            logger.add_scalar('rewards/epi_rew', epi_reward,ep_i)
            logger.add_scalar('rewards/adv_rew', adv_reward,ep_i)
            logger.add_scalar('rewards/agent_rew', ag_reward,ep_i)

            if (total_closs, total_aloss) != (0, 0):
                logger.add_scalar('losses/c_loss', total_closs/maddpg.nagents,ep_i)
                logger.add_scalar('losses/a_loss', total_aloss/maddpg.nagents,ep_i)
                logger.add_scalar('lrs/lr', lr,ep_i)

            

        # ep_rews = replay_buffer.get_average_rewards(
        #     config.episode_length * config.n_rollout_threads)
        # for a_i, a_ep_rew in enumerate(ep_rews):
        #     logger.add_scalar('agent%i/mean_episode_rewards' % a_i, a_ep_rew, ep_i)

        if ep_i % config.save_interval < config.n_rollout_threads:
            os.makedirs(run_dir / 'incremental', exist_ok=True)
            maddpg.save(run_dir / 'model.pt')

    maddpg.save(run_dir / 'model.pt')
    env.close()
    logger.export_scalars_to_json(str(log_dir / 'summary.json'))
    logger.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("env_id", help="Name of environment")
    parser.add_argument("model_name",
                        help="Name of directory to store " +
                             "model/training contents")
    parser.add_argument("--seed",
                        default=1, type=int,
                        help="Random seed")
    parser.add_argument("--n_rollout_threads", default=1, type=int)
    parser.add_argument("--n_training_threads", default=6, type=int)
    parser.add_argument("--buffer_length", default=int(5*1e5), type=int)
    parser.add_argument("--n_episodes", default=30000, type=int)
    parser.add_argument("--n_epi_before_train", default=100, type=int)
    parser.add_argument("--episode_length", default=25, type=int)
    parser.add_argument("--steps_per_update", default=100, type=int)
    parser.add_argument("--batch_size",
                        default=1024, type=int,
                        help="Batch size for model training")
    parser.add_argument("--n_exploration_eps", default=25000, type=int)
    parser.add_argument("--init_noise_scale", default=0.3, type=float)
    parser.add_argument("--final_noise_scale", default=0.0, type=float)
    parser.add_argument("--save_interval", default=1000, type=int)
    parser.add_argument("--hidden_dim", default=64, type=int)
    parser.add_argument("--lr", default=7e-4, type=float)
    parser.add_argument("--swag_lr", default=1e-2, type=float)
    parser.add_argument("--tau", default=0.005, type=float)
    parser.add_argument("--agent_alg",
                        default="MADDPG", type=str,
                        choices=['MADDPG', 'DDPG', 'Bootc', 'Boota', 'SWAG'])
    parser.add_argument("--adversary_alg",
                        default="MADDPG", type=str,
                        choices=['MADDPG', 'DDPG', 'Bootc','Boota', 'SWAG'])
    parser.add_argument("--discrete_action",
                        action='store_true')
    parser.add_argument("--alg", default='maddpg', type=str, choices=['maddpg', 'share', 'swag(temporaily)'])
    parser.add_argument("--sample_freq", default=100, type=int)
    parser.add_argument("--collect_freq", default=4, type=int)
    parser.add_argument("--swag_start", default=20000, type=int)
    parser.add_argument("--lr_cycle", action='store_true')
    parser.add_argument("--scale", default=0.5, type=float)
    parser.add_argument("--display",
                        action='store_true')
    parser.add_argument("--gpu_no", default='0', type=str)
    config = parser.parse_args()

    run(config)