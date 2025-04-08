import torch
import torch.nn.functional as F
import torch.optim as optim

class ProximalPolicyOptimization:
  def __init__(self, actor, critic, learning_rate, gamma, epsilon, lmbda):
    super(ProximalPolicyOptimization,  self).__init__()
    self.actor = actor
    self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=learning_rate)

    self.critic = critic
    self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=learning_rate)

    self.gamma = gamma
    self.epsilon = epsilon
    self.lmbda = lmbda

  def get_action(self, state):
    logits = self.actor(state)
    probs = F.softmax(logits, dim=-1)
    distribution = torch.distributions.Categorical(probs)
    action = distribution.sample()
    log_prob = distribution.log_prob(action)
    return log_prob
  
  def compute_advantages(self, rewards, values, next_value):
    advantages = []
    gae = 0
    for idx in range(len(rewards) - 1, -1, -1):
      delta = rewards[idx] + self.gamma * next_value - values[idx]
      gae = delta + self.gamma * self.lmbda * gae
      advantages.insert(0, gae)
      next_value = values[idx]
    return torch.tensor(advantages)

  def compute_actor_loss(self, log_probs, prev_log_probs, advantages):
    ratio = torch.exp(log_probs - prev_log_probs)
    advantages = torch.clamp(ratio, 1 - self.epsilon, 1 + self.epsilon) * advantages
    return -torch.min(ratio, advantages).mean()

  def compute_critic_loss(self, rewards, values, advantages):
    return F.mse_loss(rewards + advantages, values)

  def compute_total_loss(self, actor_loss, critic_loss):
    return actor_loss + 0.5 * critic_loss
  
  def update(self, state, prev_log_probs, rewards, values, next_value):
    advantages = self.compute_advantages(self, rewards, values, next_value)
    log_probs = self.get_action(state)
    actor_loss = self.compute_actor_loss(log_probs, prev_log_probs, advantages)
    critic_loss = self.compute_critic_loss(rewards, values, advantages)
    total_loss = self.compute_total_loss(actor_loss, critic_loss)

    self.actor_optimizer.zero_grad()
    actor_loss.backward()
    self.actor_optimizer.step()

    self.critic_optimizer.zero_grad()
    critic_loss.backward()
    self.critic_optimizer.step()
    return actor_loss.item(), critic_loss.item(), total_loss.item()
