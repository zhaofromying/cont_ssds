import gym
import random
import unittest

import ray
from ray.rllib.agents.pg import PGTrainer
from ray.rllib.agents.pg.pg_tf_policy import PGTFPolicy
from ray.rllib.agents.dqn.dqn_tf_policy import DQNTFPolicy
from ray.rllib.optimizers import (SyncSamplesOptimizer, SyncReplayOptimizer,
                                  AsyncGradientsOptimizer)
from ray.rllib.tests.test_rollout_worker import (MockEnv, MockEnv2, MockPolicy)
from ray.rllib.evaluation.rollout_worker import RolloutWorker
from ray.rllib.policy.tests.test_policy import TestPolicy
from ray.rllib.evaluation.metrics import collect_metrics
from ray.rllib.evaluation.worker_set import WorkerSet
from ray.rllib.env.base_env import _MultiAgentEnvToBaseEnv
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from ray.tune.registry import register_env

from social_dilemmas.envs.cleanup import CleanupEnv
from ray.rllib.models import ModelCatalog
from algorithms.a3c_baseline import build_a3c_baseline_trainer, build_a3c_policy
from models.baseline_model import BaselineModel
import copy
from utility_funcs import update_nested_dict
from ray.rllib.agents.registry import get_agent_class


def one_hot(i, n):
    out = [0.0] * n
    out[i] = 1.0
    return out


class BasicMultiAgent(MultiAgentEnv):
    """Env of N independent agents, each of which exits after 25 steps."""

    def __init__(self, num):
        self.agents = [MockEnv(25) for _ in range(num)]
        self.dones = set()
        self.observation_space = gym.spaces.Discrete(2)
        self.action_space = gym.spaces.Discrete(2)
        self.resetted = False

    def reset(self):
        self.resetted = True
        self.dones = set()
        return {i: a.reset() for i, a in enumerate(self.agents)}

    def step(self, action_dict):
        obs, rew, done, info = {}, {}, {}, {}
        for i, action in action_dict.items():
            obs[i], rew[i], done[i], info[i] = self.agents[i].step(action)
            if done[i]:
                self.dones.add(i)
        done["__all__"] = len(self.dones) == len(self.agents)
        return obs, rew, done, info


class EarlyDoneMultiAgent(MultiAgentEnv):
    """Env for testing when the env terminates (after agent 0 does)."""

    def __init__(self):
        self.agents = [MockEnv(3), MockEnv(5)]
        self.dones = set()
        self.last_obs = {}
        self.last_rew = {}
        self.last_done = {}
        self.last_info = {}
        self.i = 0
        self.observation_space = gym.spaces.Discrete(10)
        self.action_space = gym.spaces.Discrete(2)

    def reset(self):
        self.dones = set()
        self.last_obs = {}
        self.last_rew = {}
        self.last_done = {}
        self.last_info = {}
        self.i = 0
        for i, a in enumerate(self.agents):
            self.last_obs[i] = a.reset()
            self.last_rew[i] = None
            self.last_done[i] = False
            self.last_info[i] = {}
        obs_dict = {self.i: self.last_obs[self.i]}
        self.i = (self.i + 1) % len(self.agents)
        return obs_dict

    def step(self, action_dict):
        assert len(self.dones) != len(self.agents)
        for i, action in action_dict.items():
            (self.last_obs[i], self.last_rew[i], self.last_done[i],
             self.last_info[i]) = self.agents[i].step(action)
        obs = {self.i: self.last_obs[self.i]}
        rew = {self.i: self.last_rew[self.i]}
        done = {self.i: self.last_done[self.i]}
        info = {self.i: self.last_info[self.i]}
        if done[self.i]:
            rew[self.i] = 0
            self.dones.add(self.i)
        self.i = (self.i + 1) % len(self.agents)
        done["__all__"] = len(self.dones) == len(self.agents) - 1
        return obs, rew, done, info


class RoundRobinMultiAgent(MultiAgentEnv):
    """Env of N independent agents, each of which exits after 5 steps.

    On each step() of the env, only one agent takes an action."""

    def __init__(self, num, increment_obs=False):
        if increment_obs:
            # Observations are 0, 1, 2, 3... etc. as time advances
            self.agents = [MockEnv2(5) for _ in range(num)]
        else:
            # Observations are all zeros
            self.agents = [MockEnv(5) for _ in range(num)]
        self.dones = set()
        self.last_obs = {}
        self.last_rew = {}
        self.last_done = {}
        self.last_info = {}
        self.i = 0
        self.num = num
        self.observation_space = gym.spaces.Discrete(10)
        self.action_space = gym.spaces.Discrete(2)

    def reset(self):
        self.dones = set()
        self.last_obs = {}
        self.last_rew = {}
        self.last_done = {}
        self.last_info = {}
        self.i = 0
        for i, a in enumerate(self.agents):
            self.last_obs[i] = a.reset()
            self.last_rew[i] = None
            self.last_done[i] = False
            self.last_info[i] = {}
        obs_dict = {self.i: self.last_obs[self.i]}
        self.i = (self.i + 1) % self.num
        return obs_dict

    def step(self, action_dict):
        assert len(self.dones) != len(self.agents)
        for i, action in action_dict.items():
            (self.last_obs[i], self.last_rew[i], self.last_done[i],
             self.last_info[i]) = self.agents[i].step(action)
        obs = {self.i: self.last_obs[self.i]}
        rew = {self.i: self.last_rew[self.i]}
        done = {self.i: self.last_done[self.i]}
        info = {self.i: self.last_info[self.i]}
        if done[self.i]:
            rew[self.i] = 0
            self.dones.add(self.i)
        self.i = (self.i + 1) % self.num
        done["__all__"] = len(self.dones) == len(self.agents)
        return obs, rew, done, info


def make_multiagent(env_name):
    class MultiEnv(MultiAgentEnv):
        def __init__(self, num):
            self.agents = [gym.make(env_name) for _ in range(num)]
            self.dones = set()
            self.observation_space = self.agents[0].observation_space
            self.action_space = self.agents[0].action_space

        def reset(self):
            self.dones = set()
            return {i: a.reset() for i, a in enumerate(self.agents)}

        def step(self, action_dict):
            obs, rew, done, info = {}, {}, {}, {}
            for i, action in action_dict.items():
                obs[i], rew[i], done[i], info[i] = self.agents[i].step(action)
                if done[i]:
                    self.dones.add(i)
            done["__all__"] = len(self.dones) == len(self.agents)
            return obs, rew, done, info

    return MultiEnv


MultiCartpole = make_multiagent("CartPole-v0")
MultiMountainCar = make_multiagent("MountainCarContinuous-v0")


class TestMultiAgentEnv(unittest.TestCase):
    def setUp(self) -> None:
        ray.init(num_cpus=4)

    def tearDown(self) -> None:
        ray.shutdown()

    def testCleanupEnv(self):
        def env_creator(_):
            return CleanupEnv(
                num_agents=5,
                return_agent_actions=True,
                use_collective_reward=False,
            )

        env_name = 'cleanup_env'
        register_env(env_name, env_creator)

        single_env = env_creator(5)
        obs_space = single_env.observation_space
        act_space = single_env.action_space
        model_name = 'baseline_lstm'
        ModelCatalog.register_custom_model(model_name, BaselineModel)

        def gen_policy():
            return None, obs_space, act_space, {"custom_model": model_name}

        # Create 1 distinct policy per agent
        policy_graphs = {}
        for i in range(5):
            policy_graphs["agent-" + str(i)] = gen_policy()

        def policy_mapping_fn(agent_id):
            return agent_id

        agent_cls = get_agent_class('A3C')
        config = copy.deepcopy(agent_cls._default_config)

        config["env"] = env_name
        config["eager"] = False

        # information for replay
        config["env_config"]["func_create"] = env_creator
        config["env_config"]["env_name"] = env_name

        conv_filters = [[6, [3, 3], 1]]
        fcnet_hiddens = [32, 32]
        lstm_cell_size = 128

        update_nested_dict(
            config,
            {
                'seed': 132455,
                "horizon": 200,
                "gamma": 0.99,
                "lr": 0.0001,
                "lr_schedule": None,
                "vf_loss_coeff": 0.1,
                "rollout_fragment_length": 200,
                "train_batch_size": 2000,
                "multiagent": {"policies": policy_graphs, "policy_mapping_fn": policy_mapping_fn},
                "callbacks": single_env.get_environment_callbacks(),
                "model": {
                    "custom_model": model_name,
                    "use_lstm": False,
                    "conv_filters": conv_filters,
                    "fcnet_hiddens": fcnet_hiddens,
                    "custom_options": {
                        "cell_size": lstm_cell_size,
                        "num_other_agents": 4,
                        'predict_steps': 15,
                        "contribute_reward_clip": 10,
                        'contribute_reward_weight': 0.05,
                        'contribute_reward_schedule_steps': 1e8,
                        'contribute_reward_schedule_weights': 1.0
                    },
                },
            },
        )

        act = build_a3c_baseline_trainer(config)
        act = act(env='cleanup_env')
        act.train()

'''
    def test_basic_mock(self):
        env = BasicMultiAgent(4)
        obs = env.reset()
        self.assertEqual(obs, {0: 0, 1: 0, 2: 0, 3: 0})
        for _ in range(24):
            obs, rew, done, info = env.step({0: 0, 1: 0, 2: 0, 3: 0})
            self.assertEqual(obs, {0: 0, 1: 0, 2: 0, 3: 0})
            self.assertEqual(rew, {0: 1, 1: 1, 2: 1, 3: 1})
            self.assertEqual(done, {
                0: False,
                1: False,
                2: False,
                3: False,
                "__all__": False
            })
        obs, rew, done, info = env.step({0: 0, 1: 0, 2: 0, 3: 0})
        self.assertEqual(done, {
            0: True,
            1: True,
            2: True,
            3: True,
            "__all__": True
        })

    def test_round_robin_mock(self):
        env = RoundRobinMultiAgent(2)
        obs = env.reset()
        self.assertEqual(obs, {0: 0})
        for _ in range(5):
            obs, rew, done, info = env.step({0: 0})
            self.assertEqual(obs, {1: 0})
            self.assertEqual(done["__all__"], False)
            obs, rew, done, info = env.step({1: 0})
            self.assertEqual(obs, {0: 0})
            self.assertEqual(done["__all__"], False)
        obs, rew, done, info = env.step({0: 0})
        self.assertEqual(done["__all__"], True)

    def test_no_reset_until_poll(self):
        env = _MultiAgentEnvToBaseEnv(lambda v: BasicMultiAgent(2), [], 1)
        self.assertFalse(env.get_unwrapped()[0].resetted)
        env.poll()
        self.assertTrue(env.get_unwrapped()[0].resetted)

    def test_vectorize_basic(self):
        env = _MultiAgentEnvToBaseEnv(lambda v: BasicMultiAgent(2), [], 2)
        obs, rew, dones, _, _ = env.poll()
        self.assertEqual(obs, {0: {0: 0, 1: 0}, 1: {0: 0, 1: 0}})
        self.assertEqual(rew, {0: {0: None, 1: None}, 1: {0: None, 1: None}})
        self.assertEqual(
            dones, {
                0: {
                    0: False,
                    1: False,
                    "__all__": False
                },
                1: {
                    0: False,
                    1: False,
                    "__all__": False
                }
            })
        for _ in range(24):
            env.send_actions({0: {0: 0, 1: 0}, 1: {0: 0, 1: 0}})
            obs, rew, dones, _, _ = env.poll()
            self.assertEqual(obs, {0: {0: 0, 1: 0}, 1: {0: 0, 1: 0}})
            self.assertEqual(rew, {0: {0: 1, 1: 1}, 1: {0: 1, 1: 1}})
            self.assertEqual(
                dones, {
                    0: {
                        0: False,
                        1: False,
                        "__all__": False
                    },
                    1: {
                        0: False,
                        1: False,
                        "__all__": False
                    }
                })
        env.send_actions({0: {0: 0, 1: 0}, 1: {0: 0, 1: 0}})
        obs, rew, dones, _, _ = env.poll()
        self.assertEqual(
            dones, {
                0: {
                    0: True,
                    1: True,
                    "__all__": True
                },
                1: {
                    0: True,
                    1: True,
                    "__all__": True
                }
            })

        # Reset processing
        self.assertRaises(
            ValueError, lambda: env.send_actions({
                0: {
                    0: 0,
                    1: 0
                },
                1: {
                    0: 0,
                    1: 0
                }
            }))
        self.assertEqual(env.try_reset(0), {0: 0, 1: 0})
        self.assertEqual(env.try_reset(1), {0: 0, 1: 0})
        env.send_actions({0: {0: 0, 1: 0}, 1: {0: 0, 1: 0}})
        obs, rew, dones, _, _ = env.poll()
        self.assertEqual(obs, {0: {0: 0, 1: 0}, 1: {0: 0, 1: 0}})
        self.assertEqual(rew, {0: {0: 1, 1: 1}, 1: {0: 1, 1: 1}})
        self.assertEqual(
            dones, {
                0: {
                    0: False,
                    1: False,
                    "__all__": False
                },
                1: {
                    0: False,
                    1: False,
                    "__all__": False
                }
            })

    def test_vectorize_round_robin(self):
        env = _MultiAgentEnvToBaseEnv(lambda v: RoundRobinMultiAgent(2), [], 2)
        obs, rew, dones, _, _ = env.poll()
        self.assertEqual(obs, {0: {0: 0}, 1: {0: 0}})
        self.assertEqual(rew, {0: {0: None}, 1: {0: None}})
        env.send_actions({0: {0: 0}, 1: {0: 0}})
        obs, rew, dones, _, _ = env.poll()
        self.assertEqual(obs, {0: {1: 0}, 1: {1: 0}})
        env.send_actions({0: {1: 0}, 1: {1: 0}})
        obs, rew, dones, _, _ = env.poll()
        self.assertEqual(obs, {0: {0: 0}, 1: {0: 0}})

    def test_multi_agent_sample(self):
        act_space = gym.spaces.Discrete(2)
        obs_space = gym.spaces.Discrete(2)
        ev = RolloutWorker(
            env_creator=lambda _: BasicMultiAgent(5),
            policy={
                "p0": (MockPolicy, obs_space, act_space, {}),
                "p1": (MockPolicy, obs_space, act_space, {}),
            },
            policy_mapping_fn=lambda agent_id: "p{}".format(agent_id % 2),
            rollout_fragment_length=50)
        batch = ev.sample()
        self.assertEqual(batch.count, 50)
        self.assertEqual(batch.policy_batches["p0"].count, 150)
        self.assertEqual(batch.policy_batches["p1"].count, 100)
        self.assertEqual(batch.policy_batches["p0"]["t"].tolist(),
                         list(range(25)) * 6)

    def test_multi_agent_sample_sync_remote(self):
        # Allow to be run via Unittest.
        ray.init(num_cpus=4, ignore_reinit_error=True)
        act_space = gym.spaces.Discrete(2)
        obs_space = gym.spaces.Discrete(2)
        ev = RolloutWorker(
            env_creator=lambda _: BasicMultiAgent(5),
            policy={
                "p0": (MockPolicy, obs_space, act_space, {}),
                "p1": (MockPolicy, obs_space, act_space, {}),
            },
            policy_mapping_fn=lambda agent_id: "p{}".format(agent_id % 2),
            rollout_fragment_length=50,
            num_envs=4,
            remote_worker_envs=True,
            remote_env_batch_wait_ms=99999999)
        batch = ev.sample()
        self.assertEqual(batch.count, 200)

    def test_multi_agent_sample_async_remote(self):
        # Allow to be run via Unittest.
        ray.init(num_cpus=4, ignore_reinit_error=True)
        act_space = gym.spaces.Discrete(2)
        obs_space = gym.spaces.Discrete(2)
        ev = RolloutWorker(
            env_creator=lambda _: BasicMultiAgent(5),
            policy={
                "p0": (MockPolicy, obs_space, act_space, {}),
                "p1": (MockPolicy, obs_space, act_space, {}),
            },
            policy_mapping_fn=lambda agent_id: "p{}".format(agent_id % 2),
            rollout_fragment_length=50,
            num_envs=4,
            remote_worker_envs=True)
        batch = ev.sample()
        self.assertEqual(batch.count, 200)

    def test_multi_agent_sample_with_horizon(self):
        act_space = gym.spaces.Discrete(2)
        obs_space = gym.spaces.Discrete(2)
        ev = RolloutWorker(
            env_creator=lambda _: BasicMultiAgent(5),
            policy={
                "p0": (MockPolicy, obs_space, act_space, {}),
                "p1": (MockPolicy, obs_space, act_space, {}),
            },
            policy_mapping_fn=lambda agent_id: "p{}".format(agent_id % 2),
            episode_horizon=10,  # test with episode horizon set
            rollout_fragment_length=50)
        batch = ev.sample()
        self.assertEqual(batch.count, 50)

    def test_sample_from_early_done_env(self):
        act_space = gym.spaces.Discrete(2)
        obs_space = gym.spaces.Discrete(2)
        ev = RolloutWorker(
            env_creator=lambda _: EarlyDoneMultiAgent(),
            policy={
                "p0": (MockPolicy, obs_space, act_space, {}),
                "p1": (MockPolicy, obs_space, act_space, {}),
            },
            policy_mapping_fn=lambda agent_id: "p{}".format(agent_id % 2),
            batch_mode="complete_episodes",
            rollout_fragment_length=1)
        self.assertRaisesRegexp(ValueError,
                                ".*don't have a last observation.*",
                                lambda: ev.sample())

    def test_multi_agent_sample_round_robin(self):
        act_space = gym.spaces.Discrete(2)
        obs_space = gym.spaces.Discrete(10)
        ev = RolloutWorker(
            env_creator=lambda _: RoundRobinMultiAgent(5, increment_obs=True),
            policy={
                "p0": (MockPolicy, obs_space, act_space, {}),
            },
            policy_mapping_fn=lambda agent_id: "p0",
            rollout_fragment_length=50)
        batch = ev.sample()
        self.assertEqual(batch.count, 50)
        # since we round robin introduce agents into the env, some of the env
        # steps don't count as proper transitions
        self.assertEqual(batch.policy_batches["p0"].count, 42)
        self.assertEqual(batch.policy_batches["p0"]["obs"].tolist()[:10], [
            one_hot(0, 10),
            one_hot(1, 10),
            one_hot(2, 10),
            one_hot(3, 10),
            one_hot(4, 10),
        ] * 2)
        self.assertEqual(batch.policy_batches["p0"]["new_obs"].tolist()[:10], [
            one_hot(1, 10),
            one_hot(2, 10),
            one_hot(3, 10),
            one_hot(4, 10),
            one_hot(5, 10),
        ] * 2)
        self.assertEqual(batch.policy_batches["p0"]["rewards"].tolist()[:10],
                         [100, 100, 100, 100, 0] * 2)
        self.assertEqual(batch.policy_batches["p0"]["dones"].tolist()[:10],
                         [False, False, False, False, True] * 2)
        self.assertEqual(batch.policy_batches["p0"]["t"].tolist()[:10],
                         [4, 9, 14, 19, 24, 5, 10, 15, 20, 25])

    def test_custom_rnn_state_values(self):
        h = {"some": {"arbitrary": "structure", "here": [1, 2, 3]}}

        class StatefulPolicy(TestPolicy):
            def compute_actions(self,
                                obs_batch,
                                state_batches=None,
                                prev_action_batch=None,
                                prev_reward_batch=None,
                                episodes=None,
                                explore=True,
                                timestep=None,
                                **kwargs):
                return [0] * len(obs_batch), [[h] * len(obs_batch)], {}

            def get_initial_state(self):
                return [{}]  # empty dict

        ev = RolloutWorker(
            env_creator=lambda _: gym.make("CartPole-v0"),
            policy=StatefulPolicy,
            rollout_fragment_length=5)
        batch = ev.sample()
        self.assertEqual(batch.count, 5)
        self.assertEqual(batch["state_in_0"][0], {})
        self.assertEqual(batch["state_out_0"][0], h)
        self.assertEqual(batch["state_in_0"][1], h)
        self.assertEqual(batch["state_out_0"][1], h)

    def test_returning_model_based_rollouts_data(self):
        class ModelBasedPolicy(PGTFPolicy):
            def compute_actions(self,
                                obs_batch,
                                state_batches,
                                prev_action_batch=None,
                                prev_reward_batch=None,
                                episodes=None,
                                **kwargs):
                # Pretend we did a model-based rollout and want to return
                # the extra trajectory.
                builder = episodes[0].new_batch_builder()
                rollout_id = random.randint(0, 10000)
                for t in range(5):
                    builder.add_values(
                        agent_id="extra_0",
                        policy_id="p1",  # use p1 so we can easily check it
                        t=t,
                        eps_id=rollout_id,  # new id for each rollout
                        obs=obs_batch[0],
                        actions=0,
                        rewards=0,
                        dones=t == 4,
                        infos={},
                        new_obs=obs_batch[0])
                batch = builder.build_and_reset(episode=None)
                episodes[0].add_extra_batch(batch)

                # Just return zeros for actions
                return [0] * len(obs_batch), [], {}

        single_env = gym.make("CartPole-v0")
        obs_space = single_env.observation_space
        act_space = single_env.action_space
        ev = RolloutWorker(
            env_creator=lambda _: MultiCartpole(2),
            policy={
                "p0": (ModelBasedPolicy, obs_space, act_space, {}),
                "p1": (ModelBasedPolicy, obs_space, act_space, {}),
            },
            policy_mapping_fn=lambda agent_id: "p0",
            rollout_fragment_length=5)
        batch = ev.sample()
        self.assertEqual(batch.count, 5)
        self.assertEqual(batch.policy_batches["p0"].count, 10)
        self.assertEqual(batch.policy_batches["p1"].count, 25)

    def test_train_multi_cartpole_single_policy(self):
        n = 10
        register_env("multi_cartpole", lambda _: MultiCartpole(n))
        pg = PGTrainer(env="multi_cartpole", config={"num_workers": 0})
        for i in range(100):
            result = pg.train()
            print("Iteration {}, reward {}, timesteps {}".format(
                i, result["episode_reward_mean"], result["timesteps_total"]))
            if result["episode_reward_mean"] >= 50 * n:
                return
        raise Exception("failed to improve reward")

    def test_train_multi_cartpole_multi_policy(self):
        n = 10
        register_env("multi_cartpole", lambda _: MultiCartpole(n))
        single_env = gym.make("CartPole-v0")

        def gen_policy():
            config = {
                "gamma": random.choice([0.5, 0.8, 0.9, 0.95, 0.99]),
                "n_step": random.choice([1, 2, 3, 4, 5]),
            }
            obs_space = single_env.observation_space
            act_space = single_env.action_space
            return (None, obs_space, act_space, config)

        pg = PGTrainer(
            env="multi_cartpole",
            config={
                "num_workers": 0,
                "multiagent": {
                    "policies": {
                        "policy_1": gen_policy(),
                        "policy_2": gen_policy(),
                    },
                    "policy_mapping_fn": lambda agent_id: "policy_1",
                },
            })

        # Just check that it runs without crashing
        for i in range(10):
            result = pg.train()
            print("Iteration {}, reward {}, timesteps {}".format(
                i, result["episode_reward_mean"], result["timesteps_total"]))
        self.assertTrue(
            pg.compute_action([0, 0, 0, 0], policy_id="policy_1") in [0, 1])
        self.assertTrue(
            pg.compute_action([0, 0, 0, 0], policy_id="policy_2") in [0, 1])
        self.assertRaises(
            KeyError,
            lambda: pg.compute_action([0, 0, 0, 0], policy_id="policy_3"))

    def _test_with_optimizer(self, optimizer_cls):
        n = 3
        env = gym.make("CartPole-v0")
        act_space = env.action_space
        obs_space = env.observation_space
        dqn_config = {"gamma": 0.95, "n_step": 3}
        if optimizer_cls == SyncReplayOptimizer:
            # TODO: support replay with non-DQN graphs. Currently this can't
            # happen since the replay buffer doesn't encode extra fields like
            # "advantages" that PG uses.
            policies = {
                "p1": (DQNTFPolicy, obs_space, act_space, dqn_config),
                "p2": (DQNTFPolicy, obs_space, act_space, dqn_config),
            }
        else:
            policies = {
                "p1": (PGTFPolicy, obs_space, act_space, {}),
                "p2": (DQNTFPolicy, obs_space, act_space, dqn_config),
            }
        worker = RolloutWorker(
            env_creator=lambda _: MultiCartpole(n),
            policy=policies,
            policy_mapping_fn=lambda agent_id: ["p1", "p2"][agent_id % 2],
            rollout_fragment_length=50)
        if optimizer_cls == AsyncGradientsOptimizer:

            def policy_mapper(agent_id):
                return ["p1", "p2"][agent_id % 2]

            remote_workers = [
                RolloutWorker.as_remote().remote(
                    env_creator=lambda _: MultiCartpole(n),
                    policy=policies,
                    policy_mapping_fn=policy_mapper,
                    rollout_fragment_length=50)
            ]
        else:
            remote_workers = []
        workers = WorkerSet._from_existing(worker, remote_workers)
        optimizer = optimizer_cls(workers)
        for i in range(200):
            optimizer.step()
            result = collect_metrics(worker, remote_workers)
            if i % 20 == 0:

                def do_update(p):
                    if isinstance(p, DQNTFPolicy):
                        p.update_target()

                worker.foreach_policy(lambda p, _: do_update(p))
                print("Iter {}, rew {}".format(i,
                                               result["policy_reward_mean"]))
                print("Total reward", result["episode_reward_mean"])
            if result["episode_reward_mean"] >= 25 * n:
                return
        print(result)
        raise Exception("failed to improve reward")

    def test_multi_agent_sync_optimizer(self):
        self._test_with_optimizer(SyncSamplesOptimizer)

    def test_multi_agent_async_gradients_optimizer(self):
        # Allow to be run via Unittest.
        ray.init(num_cpus=4, ignore_reinit_error=True)
        self._test_with_optimizer(AsyncGradientsOptimizer)

    def test_multi_agent_replay_optimizer(self):
        self._test_with_optimizer(SyncReplayOptimizer)

    def test_train_multi_cartpole_many_policies(self):
        n = 20
        env = gym.make("CartPole-v0")
        act_space = env.action_space
        obs_space = env.observation_space
        policies = {}
        for i in range(20):
            policies["pg_{}".format(i)] = (PGTFPolicy, obs_space, act_space,
                                           {})
        policy_ids = list(policies.keys())
        worker = RolloutWorker(
            env_creator=lambda _: MultiCartpole(n),
            policy=policies,
            policy_mapping_fn=lambda agent_id: random.choice(policy_ids),
            rollout_fragment_length=100)
        workers = WorkerSet._from_existing(worker, [])
        optimizer = SyncSamplesOptimizer(workers)
        for i in range(100):
            optimizer.step()
            result = collect_metrics(worker)
            print("Iteration {}, rew {}".format(i,
                                                result["policy_reward_mean"]))
            print("Total reward", result["episode_reward_mean"])
            if result["episode_reward_mean"] >= 25 * n:
                return
        raise Exception("failed to improve reward")
'''

if __name__ == "__main__":
    import pytest
    import sys
    sys.exit(pytest.main(["-v", __file__]))
