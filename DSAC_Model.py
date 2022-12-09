from datetime import datetime
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow_probability as tfp
import matplotlib.pyplot as plt
import tracks 


class DSAC_Model:
    def __init__(self) -> None:
        self.num_states = 5  
        self.num_actions = 2 
        self.upper_bound = 1
        self.lower_bound = -1

        self.racer = tracks.Racer()

        self.actor_model = Get_actor(num_actions = self.num_actions)
        self.target_actor_model = Get_actor(num_actions=self.num_actions)
        self.target_actor_model.trainable = False
            

    def train(self, total_iterations = 50000, gamma = .99, tau = .005, update_freq = 2, sigma_min = 1, clip_boundary = 10, learning_rate = 0.001, buffer_dim = 50000, batch_size = 64):
        
        self.target_entropy = -tf.constant(self.num_actions, dtype=tf.float32)
        log_alpha = tf.Variable(0.0, dtype=tf.float32)
        self.alpha = tfp.util.DeferredTensor(log_alpha, tf.exp)

        self.critic_model = Get_critic(sigma_min)
        self.target_critic_model = Get_critic(sigma_min)
        self.target_critic_model.trainable = False

        target_critic_weights = self.critic_model.get_weights()
        self.target_critic_model.set_weights(target_critic_weights)

        target_actor_weights = self.actor_model.get_weights()
        self.target_actor_model.set_weights(target_actor_weights)

        actor_optimizer = tf.keras.optimizers.Adam(learning_rate)
        critic_optimizer = tf.keras.optimizers.Adam(learning_rate)
        alpha_optimizer = tf.keras.optimizers.Adam(learning_rate)

        self.critic_model.compile(optimizer=critic_optimizer)
        self.actor_model.compile(optimizer=actor_optimizer)

        buffer = Buffer(self.num_states, self.num_actions, buffer_dim, batch_size)

        # History of rewards per episode
        ep_reward_list = []
        # Average reward history of last few episodes
        avg_reward_list = []

        i = 0
        mean_speed = 0
        ep = 0
        avg_reward = 0

        while i < total_iterations:

            prev_state = self.racer.reset()
            episodic_reward = 0
            mean_speed += prev_state[4]
            done = False

            while not(done):
                i = i+1

                tf_prev_state = tf.expand_dims(tf.convert_to_tensor(prev_state), 0)
                _, action, _ = self.actor_model(tf_prev_state)
                state, reward, done = self.step(action)

                #we distinguish between termination with failure (state = None) and succesfull termination on track completion
                #succesfull termination is stored as a normal tuple
                fail = done and len(state) < 5
                buffer.record((prev_state, action, reward, fail, state))
                if not(done):
                    mean_speed += state[4]

                episodic_reward += reward

                if buffer.buffer_counter > batch_size:
                    states, actions, rewards, dones, newstates = buffer.sample_batch()
                    states = tf.stack(tf.convert_to_tensor(
                        states, dtype=tf.float32))
                    actions = tf.stack(tf.convert_to_tensor(
                        actions, dtype=tf.float32))
                    rewards = tf.stack(tf.convert_to_tensor(
                        rewards, dtype=tf.float32))
                    dones = tf.stack(tf.convert_to_tensor(dones, dtype=tf.float32))
                    newstates = tf.stack(tf.convert_to_tensor(
                        newstates, dtype=tf.float32))

                    self.update_critics(states, actions, rewards, dones, newstates, gamma, clip_boundary)
                    if i % update_freq == 0:
                        self.update_actor(states)
                        self.update_entropy(states, log_alpha, alpha_optimizer)
                        self.update_target(self.target_critic_model.variables, self.critic_model.variables, tau)
                        self.update_target(self.target_actor_model.variables, self.actor_model.variables, tau)

                prev_state = state

                if i % 100 == 0:
                    avg_reward_list.append(avg_reward)

            ep_reward_list.append(episodic_reward)

            # Mean of last 40 episodes
            avg_reward = np.mean(ep_reward_list[-40:])
            print("Episode {}: Iterations {}, Avg. Reward = {}, Last reward = {}. Avg. speed = {}".format(
                ep, i, avg_reward, episodic_reward, mean_speed/i))
            print("\n")

            if ep > 0 and ep % 40 == 0:
                print("## Evaluating policy ##")
                tracks.metrics_run(self.actor_model, 10)
            ep += 1

        # if total_iterations > 0:
        #     # Plotting Episodes versus Avg. Rewards
        #     plt.plot(avg_reward_list)
        #     plt.xlabel("Training steps x100")
        #     plt.ylabel("Avg. Episodic Reward")
        #     plt.ylim(-3.5, 7)
        #     plt.show(block=False)
        #     plt.pause(0.001)
        #     print("### DSAC Training ended ###")
        #     print("Trained over {} steps".format(i))

    def step(self, action):
        n = 1
        t = np.random.randint(0, n)
        action = tf.squeeze(action)
        state, reward, done = self.racer.step(action)
        for i in range(t):
            if not done:
                state, t_r, done = self.racer.step([0, 0])
                #state ,t_r, done =racer.step(action)
                reward += t_r
        return (state, reward, done)

    def save_weights(self, weights_file_actor="weights/dsac_actor_model_car", weights_file_critic="weights/dsac_critic_model_car"):
        self.actor_model.save(weights_file_actor)
        self.critic_model.save(weights_file_critic)

    def load_weights(self, weights_file_actor="weights/dsac_actor_model_car", weights_file_critic="weights/dsac_critic_model_car"):
        self.actor_model = tf.keras.models.load_model(weights_file_actor)
        self.critic_model = tf.keras.models.load_model(weights_file_critic)

    @tf.function
    def update_target(self, target_weights, weights, tau):
        for (a, b) in zip(target_weights, weights):
            a.assign(b * tau + a * (1 - tau))

    def update_weights(self, target_weights, weights, tau):
        return(target_weights * (1 - tau) + weights * tau)


    def value_and_logp(self, critic, states, actions, given_value=None):
        mu, sigma = critic([states, actions])
        dist = tfp.distributions.Normal(mu, sigma)
        log_p = None
        if given_value is None:
            given_value = mu + sigma * tfp.distributions.Normal(0, 1).sample()
        log_p = dist.log_prob(given_value)
        return given_value, log_p


    @tf.function
    def update_critics(self, states, actions, rewards, dones, newstates, gamma, clip_boundary):
        entropy_scale = tf.convert_to_tensor(self.alpha)
        _, new_policy_actions, log_probs = self.target_actor_model(newstates)
        tcritic_v, _ = self.value_and_logp(self.critic_model, newstates, new_policy_actions)
        q, _ = self.value_and_logp(self.critic_model, states, actions)
        newvalue = tcritic_v-entropy_scale*log_probs
        q_hat = rewards + gamma*newvalue*(1-dones)
        q_hat = tf.clip_by_value(q_hat, clip_value_min=q -clip_boundary, clip_value_max=q + clip_boundary)
        with tf.GradientTape() as tape1:
            _, log_p_c = self.value_and_logp(self.critic_model, states, actions, q_hat)
            loss_c1 = - tf.reduce_mean(log_p_c)
        critic1_gradient = tape1.gradient(loss_c1, self.critic_model.trainable_variables)
        self.critic_model.optimizer.apply_gradients(zip(critic1_gradient, self.critic_model.trainable_variables))


    @tf.function
    def update_actor(self, states):
        entropy_scale = tf.convert_to_tensor(self.alpha)
        with tf.GradientTape(watch_accessed_variables=False) as tape:
            tape.watch(self.actor_model.trainable_variables)
            _, new_policy_actions, log_probs = self.actor_model(states)
            critic_v, _ = self.value_and_logp(self.critic_model, states, new_policy_actions)
            actor_loss = critic_v - entropy_scale*log_probs
            actor_loss = -tf.reduce_mean(actor_loss)
        actor_gradient = tape.gradient(actor_loss, self.actor_model.trainable_variables)
        self.actor_model.optimizer.apply_gradients(zip(actor_gradient, self.actor_model.trainable_variables))

    @tf.function
    def update_entropy(self, states, log_alpha, alpha_optimizer):
        _, _, log_probs = self.actor_model(states)

        with tf.GradientTape() as tape:
            alpha_loss = tf.reduce_mean(- self.alpha *
                                        tf.stop_gradient(log_probs + self.target_entropy))

        alpha_grad = tape.gradient(alpha_loss, [log_alpha])
        alpha_optimizer.apply_gradients(zip(alpha_grad, [log_alpha]))

    def get_actor_model(self):
        if self.actor_model == None:
            raise Exception("Model is equal to None and can't be exported")
        return self.actor_model

#The actor choose the move, given the state
class Get_actor(tf.keras.Model):
    def __init__(self, num_actions = 2):
        super().__init__()
        self.num_actions = num_actions
        self.d1 = layers.Dense(64, activation="relu")
        self.d2 = layers.Dense(64, activation="relu")
        self.m = layers.Dense(self.num_actions)
        self.s = layers.Dense(self.num_actions)
        
    def call(self, inputs):
        out = self.d1(inputs)
        out = self.d2(out)
        mu = self.m(out)
        log_sigma = self.s(out)
        sigma = tf.exp(log_sigma)
        
        dist = tfp.distributions.Normal(mu, sigma)
        #action = dist.sample()
        action = mu + sigma * tfp.distributions.Normal(0,1).sample(self.num_actions)            
        valid_action = tf.tanh(action)
        
        log_p = dist.log_prob(action)
        log_p = log_p - tf.reduce_sum(tf.math.log(1 - valid_action**2 + 1e-16), axis=1, keepdims=True)
        
        if len(log_p.shape)>1:
            log_p = tf.reduce_sum(log_p,1)
        else:
            log_p = tf.reduce_sum(log_p)
        log_p = tf.reshape(log_p,(-1,1))     
        
        eval_action = tf.tanh(mu)
        
        return eval_action, valid_action, log_p
    
    @property  
    def trainable_variables(self):
        return self.d1.trainable_variables + \
                self.d2.trainable_variables + \
                self.m.trainable_variables + \
                self.s.trainable_variables


#the critic compute the q-value, given the state and the action
class Get_critic(tf.keras.Model):
    def __init__(self, sigma_min):
        super().__init__()
        self.sigma_min = sigma_min
        self.d1 = layers.Dense(64, activation="relu")
        self.d2 = layers.Dense(64, activation="relu")
        self.m = layers.Dense(1)
        self.s = layers.Dense(1)
        
    def call(self, inputs):
        state, action = inputs
        state_action = tf.concat([state, action], axis=1)
        out = self.d1(state_action)
        out = self.d2(out)
        mu = self.m(out)
        log_sigma = self.s(out)
        sigma = tf.exp(log_sigma)
        sigma = tf.maximum(sigma, self.sigma_min)
        
        return mu, sigma
    @property
    def trainable_variables(self):
        return self.d1.trainable_variables + \
                self.d2.trainable_variables + \
                self.m.trainable_variables + \
                self.s.trainable_variables


#Replay buffer
class Buffer:
    def __init__(self, num_states, num_actions, buffer_capacity=100000, batch_size=64):
        # Max Number of tuples that can be stored
        self.buffer_capacity = buffer_capacity
        # Num of tuples used for training
        self.batch_size = batch_size

        # Current number of tuples in buffer
        self.buffer_counter = 0

        # We have a different array for each tuple element
        self.state_buffer = np.zeros((self.buffer_capacity, num_states))
        self.action_buffer = np.zeros((self.buffer_capacity, num_actions))
        self.reward_buffer = np.zeros((self.buffer_capacity, 1))
        self.done_buffer = np.zeros((self.buffer_capacity, 1))
        self.next_state_buffer = np.zeros((self.buffer_capacity, num_states))

    # Stores a transition (s,a,r,s') in the buffer
    def record(self, obs_tuple):
        s,a,r,T,sn = obs_tuple
        # restart form zero if buffer_capacity is exceeded, replacing old records
        index = self.buffer_counter % self.buffer_capacity

        self.state_buffer[index] = tf.squeeze(s)
        self.action_buffer[index] = a
        self.reward_buffer[index] = r
        self.done_buffer[index] = T
        self.next_state_buffer[index] = tf.squeeze(sn)

        self.buffer_counter += 1

    def sample_batch(self):
        # Get sampling range
        record_range = min(self.buffer_counter, self.buffer_capacity)
        # Randomly sample indices
        batch_indices = np.random.choice(record_range, self.batch_size)

        s = self.state_buffer[batch_indices]
        a = self.action_buffer[batch_indices]
        r = self.reward_buffer[batch_indices]
        T = self.done_buffer[batch_indices]
        sn = self.next_state_buffer[batch_indices]
        return ((s,a,r,T,sn))


while 1:
    model = DSAC_Model()

    print("### Start training ###")
    start_t = datetime.now()
    model.train()
    end_t = datetime.now()
    print("### Time elapsed: {} ###".format(end_t-start_t))

    best_model = DSAC_Model()
    best_model.load_weights(weights_file_actor="weights/best_dsac_actor")

    models = [(best_model.get_actor_model(), "best"), (model.get_actor_model(), "trained")]

    print("### Evaluating model against best ###")
    winner1 = tracks.newrun(models)
    winner2 = tracks.newrun(models)

    if winner1 == "trained" and winner2 == "trained":
        model.save_weights(weights_file_actor="weights/best_dsac_actor")
        print("### New best model saved")

