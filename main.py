from __future__ import absolute_import, division, print_function


import reverb

import tensorflow as tf

from tf_agents.agents.dqn import dqn_agent
from tf_agents.drivers import py_driver
from tf_agents.environments import tf_py_environment

from tf_agents.networks import sequential
from tf_agents.policies import py_tf_eager_policy
from tf_agents.policies import random_tf_policy
from tf_agents.replay_buffers import reverb_replay_buffer
from tf_agents.replay_buffers import reverb_utils
from tf_agents.specs import tensor_spec
from tf_agents.utils import common
# ---------------------------------------------------------------------------------------

VERSION = 2
num_iterations = 20000

initial_collect_steps = 100
collect_steps_per_iteration =  1
replay_buffer_max_length = 100000

batch_size = 64
learning_rate = 0.001
log_interval = 200

num_eval_episodes = 1
eval_interval = 1000
# ------------------------------------------------------------------------------------------
 # We will insitanitae the game logic here
import coordinates
import player
import render
import logic
widht = 5
height = 5
co = coordinates.coordinates(height=height,width=widht)
pl = player.player(co)
rendr = render.render()
game_env = logic.logic(pl,co,rendr)

co_train = coordinates.coordinates(height=height,width=widht)
pl_train = player.player(co_train)
rendr_train = render.render()

co_eval = coordinates.coordinates(height=height,width=widht)
pl_eval = player.player(co_eval)
rendr_eval = render.render()

train_py_env = logic.logic(pl_train,co_train,rendr_train)
eval_py_env = logic.logic(pl_eval,co_eval,rendr_eval,generate_video=True)


train_env = tf_py_environment.TFPyEnvironment(train_py_env)
eval_env = tf_py_environment.TFPyEnvironment(eval_py_env)



fc_layer_params = (100,50)
#input_shape = tensor_spec.from_spec(game_env.time_step_spec())
action_tensor_spec = tensor_spec.from_spec(game_env.action_spec())
num_actions = action_tensor_spec.maximum - action_tensor_spec.minimum +1

# Define a helper function to create Dense layers configured with the right
# activation and kernel initializer.
def dense_layer(num_units):
  return tf.keras.layers.Dense(
      num_units,
      activation=tf.keras.activations.relu,
      kernel_initializer=tf.keras.initializers.VarianceScaling(
          scale=2.0, mode='fan_in', distribution='truncated_normal'))
#-------------------------------------------------------------------------------------------
dense_layers = [dense_layer(num_units) for num_units in fc_layer_params]
q_values_layer = tf.keras.layers.Dense(
    num_actions,
    activation = None,
    kernel_initializer=tf.keras.initializers.RandomUniform(
        minval=-0.03, maxval=0.03),
    bias_initializer=tf.keras.initializers.Constant(-0.2))
flatten_layer = tf.keras.layers.Flatten()
#---


q_net = sequential.Sequential(dense_layers +[flatten_layer] + [q_values_layer])


optimizer = tf.keras.optimizers.Nadam(learning_rate=learning_rate)
loss = common.element_wise_squared_loss
train_step_counter = tf.Variable(0)

agent = dqn_agent.DqnAgent(
    train_env.time_step_spec(),
    train_env.action_spec(),
    q_network=q_net,
    optimizer=optimizer,
    td_errors_loss_fn=loss,
    train_step_counter=train_step_counter)

agent.initialize()

eval_policy = agent.policy
collect_policy = agent.collect_policy

random_policy = random_tf_policy.RandomTFPolicy(train_env.time_step_spec(),
                                                train_env.action_spec())


#TODO try to make a tensorflow function to get a better result
def compute_avg_return(environment, policy, num_episodes=10):

  total_return = 0.0
  for _ in range(num_episodes):

    time_step = environment.reset()
    episode_return = 0.0
    while not time_step.is_last():
      action_step = policy.action(time_step)
      time_step = environment.step(action_step.action)
      episode_return += time_step.reward
    total_return += episode_return

  avg_return = total_return / num_episodes
  return avg_return.numpy()[0]


#print(f'base average score: {compute_avg_return(eval_env, random_policy, num_eval_episodes)}')


table_name = 'uniform_table'
replay_buffer_signature = tensor_spec.from_spec(
      agent.collect_data_spec)
replay_buffer_signature = tensor_spec.add_outer_dim(
    replay_buffer_signature)

table = reverb.Table(
    table_name,
    max_size=replay_buffer_max_length,
    sampler=reverb.selectors.Uniform(),
    remover=reverb.selectors.Fifo(),
    rate_limiter=reverb.rate_limiters.MinSize(1),
    signature=replay_buffer_signature)

reverb_server = reverb.Server([table])

replay_buffer = reverb_replay_buffer.ReverbReplayBuffer(
    agent.collect_data_spec,
    table_name=table_name,
    sequence_length=2,
    local_server=reverb_server)

rb_observer = reverb_utils.ReverbAddTrajectoryObserver(
  replay_buffer.py_client,
  table_name,
  sequence_length=2)

# Important not to get an error from the pydriver
game_env.reset()


py_driver.PyDriver(
    game_env,
    py_tf_eager_policy.PyTFEagerPolicy(
      random_policy, use_tf_function=True),
    [rb_observer],
    max_steps=initial_collect_steps).run(game_env.reset())


dataset = replay_buffer.as_dataset(
    num_parallel_calls=3,
    sample_batch_size=batch_size,
    num_steps=2).prefetch(tf.data.AUTOTUNE)

iterator = iter(dataset)


# (Optional) Optimize by wrapping some of the code in a graph using TF function.
agent.train = common.function(agent.train)

# Reset the train step.
agent.train_step_counter.assign(0)

# Evaluate the agent's policy once before training.
print("Started computing average score")
avg_return = compute_avg_return(eval_env, agent.policy, num_eval_episodes)
print(f"Evaluation average return: {avg_return}")
returns = [avg_return]


# Reset the environment.
time_step = train_py_env.reset()

# Create a driver to collect experience.
collect_driver = py_driver.PyDriver(
    game_env,
    py_tf_eager_policy.PyTFEagerPolicy(
      agent.collect_policy, use_tf_function=True),
    [rb_observer],
    max_steps=collect_steps_per_iteration)


path = f'Saved Model/{VERSION}/Check Pointer'
train_checkpointer = common.Checkpointer(ckpt_dir=path,max_to_keep=1,agent=agent,policy=agent.policy,replay_buffer=replay_buffer,global_step=train_step_counter)
def load_check_pointer():
    train_checkpointer.initialize_or_restore()
# ----------------------------------
def save_check_point(path=path,agent=agent,policy=agent.policy,replay_buffer=replay_buffer,global_step=train_step_counter):
    train_checkpointer.save(global_step)
    print('check point saved....')
# ----------------------------------
from tf_agents.policies import PolicySaver
policy_path = f'Saved Model/{VERSION}/last_policy'
def save_policy(agent=agent,path=policy_path,batch_size=batch_size):
    my_policy = agent.collect_policy
    policy_saver = PolicySaver(my_policy)#You can define the batch if needed to train only
    policy_saver.save(path)
    print('policy saved...')

load_check_pointer()
def train_function(epochs=10,time_step=time_step,num_iterations=num_iterations):
    for epoch in range(epochs):
        print(f"------------------------------Epoch: {epoch}")
        for itr_num in range(num_iterations):
            # Collect a few steps and save to the replay buffer.
            time_step, _ = collect_driver.run(time_step)

            # Sample a batch of data from the buffer and update the agent's network.
            experience, unused_info = next(iterator)
            train_loss = agent.train(experience).loss

            step = agent.train_step_counter.numpy()

            if step % log_interval == 0:
                print('step = {0}: loss = {1}'.format(step, train_loss))
                save_check_point()

            if step % eval_interval == 0:
                #save_policy()
                avg_return = compute_avg_return(eval_env, agent.policy, num_eval_episodes)
                print('step = {0}: Average Return = {1}'.format(step, avg_return))
                returns.append(avg_return)


train_function(epochs=15)

print(f"Max avg value: {max(avg_return)}")

# To make an evaluation video
policy = agent.policy
time_step = eval_env.reset()
while not time_step.is_last():
    action_step = policy.action(time_step)
    time_step = eval_env.step(action_step.action)



