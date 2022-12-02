from datetime import datetime
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow import keras
from tensorflow.keras import regularizers
import numpy as np
import matplotlib.pyplot as plt
import tracks


class DDPG_Model():
    def __init__(self, tot_it = 50000, gamma = 0.99, tau = 0.005, clr = 0.001, alr = 0.001) -> None:
        self.racer = tracks.Racer()
        self.total_iterations = tot_it
        self.gamma = gamma
        self.tau = tau
        self.critic_lr = clr
        self.aux_lr = alr
        self.num_states = 5
        self.num_actions = 2
        self.upper_bound = 1
        self.lower_bound = -1
        self.buffer_dim = 50000
        self.batch_size = 64
        self.actor_model = None
        self.critic_model = None

    def load_weight(self, weights_file_critic="weights/ddpg_critic_model_car",  weights_file_actor="weights/ddpg_actor_model_car"):
        self.critic_model = keras.models.load_model(weights_file_critic)
        self.actor_model = keras.models.load_model(weights_file_actor)

    def save_weight(self, weights_file_critic= "weights/ddpg_critic_model_car",  weights_file_actor = "weights/ddpg_actor_model_car"):
        if (self.actor_model == None or self.critic_model == None):
            print("  - Actor or Critis is none, models can't be saved! ")
            return
        self.critic_model.save(weights_file_critic)
        self.actor_model.save(weights_file_actor)

    def get_actor(self, train_acceleration = True, train_direction = True):
        # the actor has separate towers for action and speed
        # in this way we can train them separately

        inputs = layers.Input(shape=(self.num_states,))
        out1 = layers.Dense(32, activation="relu",
                            trainable=train_acceleration)(inputs)
        out1 = layers.Dense(32, activation="relu",
                            trainable=train_acceleration)(out1)
        out1 = layers.Dense(1, activation='tanh',
                            trainable=train_acceleration)(out1)

        out2 = layers.Dense(32, activation="relu",
                            trainable=train_direction)(inputs)
        out2 = layers.Dense(32, activation="relu", trainable=train_direction)(out2)
        out2 = layers.Dense(1, activation='tanh', trainable=train_direction)(out2)

        outputs = layers.concatenate([out1, out2])

        #outputs = outputs * upper_bound #resize the range, if required
        model = tf.keras.Model(inputs, outputs, name="actor")
        return model

    def get_critic(self):
        # State as input
        state_input = layers.Input(shape=(self.num_states))
        state_out = layers.Dense(16, activation="relu")(state_input)
        state_out = layers.Dense(32, activation="relu")(state_out)

        # Action as input
        action_input = layers.Input(shape=(self.num_actions))
        action_out = layers.Dense(32, activation="relu")(action_input)

        concat = layers.Concatenate()([state_out, action_out])

        out = layers.Dense(64, activation="relu")(concat)
        out = layers.Dense(64, activation="relu")(out)
        outputs = layers.Dense(1)(out)  # Outputs single value

        model = tf.keras.Model([state_input, action_input], outputs, name="critic")

        return model

    def compose(self, actor, critic):
        state_input = layers.Input(shape=(self.num_states))
        a = actor(state_input)
        q = critic([state_input, a])
        #reg_weights = actor.get_layer('out').get_weights()[0]
        #print(tf.reduce_sum(0.01 * tf.square(reg_weights)))

        m = tf.keras.Model(state_input, q)
        #the loss function of the compound model is just the opposite of the critic output
        m.add_loss(-q)
        return(m)


    def step(self, action):
            n = 1
            t = np.random.randint(0, n)
            state, reward, done = self.racer.step(action)
            for i in range(t):
                if not done:
                    state, t_r, done = self.racer.step([0, 0])
                    reward += t_r
            return (state, reward, done)

    def policy(self, actor_model, state, verbose=False):
        #the policy used for training just add noise to the action
        #the amount of noise is kept constant during training
        sampled_action = tf.squeeze(actor_model(state))
        noise = np.random.normal(scale=0.1, size=2)
        #we may change the amount of noise for actions during training
        noise[0] *= 2
        noise[1] *= .5
        # Adding noise to action
        sampled_action = sampled_action.numpy()
        sampled_action += noise
        #in verbose mode, we may print information about selected actions
        if verbose and sampled_action[0] < 0:
            print("decelerating")

        #Finally, we ensure actions are within bounds
        legal_action = np.clip(sampled_action, self.lower_bound, self.upper_bound)

        return [np.squeeze(legal_action)]

    def train(self):

        actor_model = self.get_actor()
        critic_model = self.get_critic()

        target_actor = self.get_actor()
        target_critic = self.get_critic()
        target_actor.trainable = False
        target_critic.trainable = False

        #We compose actor and critic in a single model.
        #The actor is trained by maximizing the future expected reward, estimated
        #by the critic. The critic should be freezed while training the actor.
        #For simplicitly, we just use the target critic, that is not trainable.

        aux_model = self.compose(actor_model, target_critic)

        ## TRAINING ##
        # if load_weights:
        #     critic_model = keras.models.load_model(weights_file_critic)
        #     actor_model = keras.models.load_model(weights_file_actor)

        # Making the weights equal initially
        target_actor_weights = actor_model.get_weights()
        target_critic_weights = critic_model.get_weights()
        target_actor.set_weights(target_actor_weights)
        target_critic.set_weights(target_critic_weights)


        critic_optimizer = tf.keras.optimizers.Adam(self.critic_lr)
        aux_optimizer = tf.keras.optimizers.Adam(self.aux_lr)

        critic_model.compile(loss='mse', optimizer=critic_optimizer)
        aux_model.compile(optimizer=aux_optimizer)


        buffer = Buffer(self.num_states, self.num_actions, self.buffer_dim, self.batch_size)

        # History of rewards per episode
        ep_reward_list = []
        # Average reward history of last few episodes
        avg_reward_list = []

        # We introduce a probability of doing n empty actions to separate the environment time-step from the agent

        i = 0
        mean_speed = 0
        ep = 0
        avg_reward = 0

        start_t = datetime.now()
        while i < self.total_iterations:

            prev_state = self.racer.reset()
            episodic_reward = 0
            mean_speed += prev_state[self.num_states-1]
            done = False

            while not(done):
                i = i+1
                tf_prev_state = tf.expand_dims(tf.convert_to_tensor(prev_state), 0)
                #our policy is always noisy
                action = self.policy(actor_model, tf_prev_state)[0]
                # Get state and reward from the environment
                state, reward, done = self.step(action)

                #we distinguish between termination with failure (state = None) and succesfull termination on track completion
                #succesfull termination is stored as a normal tuple
                fail = done and len(state) < self.num_states

                buffer.record((prev_state, action, reward, fail, state))

                if not(done):
                    mean_speed += state[self.num_states-1]

                episodic_reward += reward

                if buffer.buffer_counter >self.batch_size:
                    states, actions, rewards, dones, newstates = buffer.sample_batch()
                    targetQ = rewards + \
                        (1-dones)*self.gamma * \
                        (target_critic([newstates, target_actor(newstates)]))
                    loss1 = critic_model.train_on_batch([states, actions], targetQ)
                    loss2 = aux_model.train_on_batch(states)

                    update_target(target_actor.variables,
                                actor_model.variables, self.tau)
                    update_target(target_critic.variables,
                                critic_model.variables, self.tau)
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
                tracks.metrics_run(actor_model, 10)
            ep += 1

        if self.total_iterations > 0:
            # if save_weights:
            #     critic_model.save(weights_file_critic)
            #     actor_model.save(weights_file_actor)
           
            self.plot_rewards(avg_reward_list)
        
        end_t = datetime.now()
        print("### DDPG Training ended ###")
        print("Trained over {} steps".format(self.total_iterations))
        print("Time elapsed: {}".format(end_t-start_t))
        self.actor_model = actor_model
        self.critic_model = critic_model


    def plot_rewards(self, avg_reward_list):
        plt.plot(avg_reward_list)
        plt.xlabel("Training steps x100")
        plt.ylabel("Avg. Episodic Reward")
        plt.ylim(-3.5, 7)
        plt.show(block=False)

    def get_actor_model(self):
        if self.actor_model == None:
            raise Exception("Model is equal to None and can't be exported")
        return self.actor_model

        



#Replay buffer 
class Buffer:
    def __init__(self, num_states, num_actions, buffer_capacity=100000, batch_size=64, ):
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
        s, a, r, T, sn = obs_tuple
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
        return ((s, a, r, T, sn))


@tf.function
def update_target(target_weights, weights, tau):
    for (a, b) in zip(target_weights, weights):
        a.assign(b * tau + a * (1 - tau))


def update_weights(target_weights, weights, tau):
    return(target_weights * (1 - tau) + weights * tau)




model = DDPG_Model()
# model.load_weight()
model.train()
# model.save_weight()

tracks.newrun([model.get_actor_model()])
