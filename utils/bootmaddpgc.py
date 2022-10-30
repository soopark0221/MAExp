from torch import Tensor
from torch.autograd import Variable
from torch.optim import Adam
from .networks import MLPNetwork
from .misc import hard_update, gumbel_softmax, onehot_from_logits
from .noise import OUNoise
import torch
import numpy as np



class BootcAgent(object):

    def __init__(self, num_in_pol, num_out_pol, num_in_critic, hidden_dim=64,
                 lr=0.01, discrete_action=True, gamma=0.99,n_ensemble=5):
        """
        Inputs:
            num_in_pol (int): number of dimensions for policy input
            num_out_pol (int): number of dimensions for policy output
            num_in_critic (int): number of dimensions for critic input
        """
        self.n_ensemble=n_ensemble
        self.gamma=gamma
        self.policy = MLPNetwork(num_in_pol, num_out_pol,
                                 hidden_dim=hidden_dim,
                                 constrain_out=True,
                                 discrete_action=discrete_action)
        self.critic =torch.nn.ModuleList([ MLPNetwork(num_in_critic, 1,
                                 hidden_dim=hidden_dim,
                                 constrain_out=False) for _ in range(n_ensemble)])
        self.target_policy =MLPNetwork(num_in_pol, num_out_pol,
                                        hidden_dim=hidden_dim,
                                        constrain_out=True,
                                        discrete_action=discrete_action)
        self.target_critic = torch.nn.ModuleList([MLPNetwork(num_in_critic, 1,
                                        hidden_dim=hidden_dim,
                                        constrain_out=False) for _ in range(n_ensemble)])

        hard_update(self.target_policy, self.policy)
        hard_update(self.target_critic, self.critic)
        self.policy_optimizer = Adam(self.policy.parameters(), lr=lr)
        self.critic_optimizer = Adam(self.critic.parameters(), lr=lr)
        if not discrete_action:
            self.exploration = OUNoise(num_out_pol)
        else:
            self.exploration = 0.3  # epsilon for eps-greedy
        self.discrete_action = discrete_action

    def reset_noise(self):
        if not self.discrete_action:
            self.exploration.reset()

    def scale_noise(self, scale):
        if self.discrete_action:
            self.exploration = scale
        else:
            self.exploration.scale = scale

    def step(self, obs, explore=False):
        """
        Take a step forward in environment for a minibatch of observations
        Inputs:
            obs (PyTorch Variable): Observations for this agent
            explore (boolean): Whether or not to add exploration noise
        Outputs:
            action (PyTorch Variable): Actions for this agent
        """

        action = self.policy(obs)
        if self.discrete_action:
            if explore:
                action = gumbel_softmax(action, hard=True)
            else:
                action = onehot_from_logits(action)
        else:  # continuous action
            if explore:
                action += Variable(Tensor(self.exploration.noise()),
                                   requires_grad=False)
            action = action.clamp(-1, 1)
        return action

    def update(self, sample,agent_i, agents):
        obs, acs, rews, next_obs, dones = sample

        self.critic_optimizer.zero_grad()
        trgt_pol=[a.target_policy for a in agents]

        if self.discrete_action: # one-hot encode action
            all_trgt_acs = [onehot_from_logits(pi(nobs)) for pi, nobs in
                            zip(trgt_pol, next_obs)]
        else:
            all_trgt_acs = [pi(nobs) for pi, nobs in zip(trgt_pol,
                                                            next_obs)]
        trgt_vf_in = torch.cat((*next_obs, *all_trgt_acs), dim=1)
        target_value = torch.stack([ rews[agent_i].view(-1, 1) + self.gamma *
                        self.target_critic[i](trgt_vf_in) *
                        (1 - dones[agent_i].view(-1, 1)) for i in range(self.n_ensemble)])

        vf_in = torch.cat((*obs, *acs), dim=1)

        actual_value = torch.stack([ self.critic[i](vf_in) for i in range(self.n_ensemble)])
        vf_loss=[]
        
        for head in range(self.n_ensemble):
            mask=torch.tensor(np.random.binomial(self.n_ensemble,0.9,len(obs[0])))
            used=torch.sum(mask)
            closs = torch.nn.MSELoss(reduction='none')(actual_value[head], target_value[head].detach())
            closs*=mask.unsqueeze(1)
            closs=torch.sum(closs/used)
            vf_loss.append(closs)
        if len(vf_loss)>0:
            vf_loss=sum(vf_loss)/float(self.n_ensemble)
            vf_loss.backward()
            torch.nn.utils.clip_grad_norm(self.critic.parameters(), 0.5)
            self.critic_optimizer.step()

        self.policy_optimizer.zero_grad()

        if self.discrete_action:
            # Forward pass as if onehot (hard=True) but backprop through a differentiable
            # Gumbel-Softmax sample. The MADDPG paper uses the Gumbel-Softmax trick to backprop
            # through discrete categorical samples, but I'm not sure if that is
            # correct since it removes the assumption of a deterministic policy for
            # DDPG. Regardless, discrete policies don't seem to learn properly without it.
            curr_pol_out = self.policy(obs[agent_i])
            curr_pol_vf_in = gumbel_softmax(curr_pol_out, hard=True)
        else:
            curr_pol_out = self.policy(obs[agent_i])
            curr_pol_vf_in = curr_pol_out
        
        all_pol_acs = []
        for i in range(len(agents)):
            if i == agent_i:
                all_pol_acs.append(curr_pol_vf_in)
            elif self.discrete_action:
                all_pol_acs.append(acs[i].detach()) # XXX: Why resample? -> use acs
            else:
                all_pol_acs.append(acs[i].detach())
        vf_in = torch.cat((*obs, *all_pol_acs), dim=1)

        random_head=np.random.randint(self.n_ensemble)
        pol_loss = -self.critic[random_head](vf_in).mean()
        pol_loss += (curr_pol_out**2).mean() * 1e-3
        pol_loss.backward()

        torch.nn.utils.clip_grad_norm(self.policy.parameters(), 0.5)
        self.policy_optimizer.step()

        return vf_loss.detach(), pol_loss.detach()

    def get_params(self):
        return {'policy': self.policy.state_dict(),
                'critic': self.critic.state_dict(),
                'target_policy': self.target_policy.state_dict(),
                'target_critic': self.target_critic.state_dict(),
                'policy_optimizer': self.policy_optimizer.state_dict(),
                'critic_optimizer': self.critic_optimizer.state_dict()}

    def load_params(self, params):
        self.policy.load_state_dict(params['policy'])
        self.critic.load_state_dict(params['critic'])
        self.target_policy.load_state_dict(params['target_policy'])
        self.target_critic.load_state_dict(params['target_critic'])
        self.policy_optimizer.load_state_dict(params['policy_optimizer'])
        self.critic_optimizer.load_state_dict(params['critic_optimizer'])
