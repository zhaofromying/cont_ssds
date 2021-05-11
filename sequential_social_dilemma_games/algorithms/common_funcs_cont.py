import numpy as np
from ray.rllib.models import ModelCatalog
from ray.rllib.policy.policy import Policy
from ray.rllib.utils import try_import_tf
from ray.rllib.utils.annotations import override

from algorithms.common_funcs_baseline import BaselineResetConfigMixin

tf = try_import_tf()

VISIBILITY = "others_visibility"

ACTION_LOGITS = "action_logits"
OTHERS_ACTIONS = "others_actions"
PREDICTED_ACTIONS = "predicted_actions"
CONTRIBUTE_REWARD = "contribute_reward"
POLICY_SCOPE = "func"


class ContributeScheduleMixIn(object):
    def __init__(self, config):
        config = config["model"]["custom_options"]
        self.baseline_contribute_reward_weight = config["contirbute_weight"]
        if any(
            config[key] is None
            for key in ["contribute_reward_schedule_steps", "contribute_reward_schedule_weights"]
        ):
            self.compute_contribute_reward_weight = lambda: self.baseline_contribute_reward_weight
        self.contribute_reward_schedule_steps = config["contribute_reward_schedule_steps"]
        self.contribute_reward_schedule_weights = config["contribute_reward_schedule_weights"]
        self.timestep = 0
        self.cur_contribute_reward_weight = np.float32(self.compute_contribute_reward_weight())
        # This tensor is for logging the weight to progress.csv
        self.cur_contribute_reward_weight_tensor = tf.get_variable(
            "cur_contribute_reward_weight",
            initializer=self.cur_contribute_reward_weight,
            trainable=False,
        )

    @override(Policy)
    def on_global_var_update(self, global_vars):
        super(ContributeScheduleMixIn, self).on_global_var_update(global_vars)
        self.timestep = global_vars["timestep"]
        self.cur_contribute_reward_weight = self.compute_contribute_reward_weight()
        self.cur_contribute_reward_weight_tensor.load(
            self.cur_contribute_reward_weight, session=self._sess
        )

    def compute_contribute_reward_weight(self):
        """ Computes multiplier for influence reward based on training steps
        taken and schedule parameters.
        """
        weight = np.interp(
            self.timestep,
            self.contribute_reward_schedule_steps,
            self.contribute_reward_schedule_weights,
        )
        return weight * self.baseline_contribute_reward_weight


def cont_postprocess_trajectory(policy, sample_batch, other_agent_batches=None, episode=None):
    # Weigh social influence reward and add to batch.
    sample_batch = weigh_and_add_contribute_reward(policy, sample_batch)
    return sample_batch


# 计算cont_reward 并将其添加在sampleBatch中
def weigh_and_add_contribute_reward(policy, sample_batch):
    cur_contribute_reward_weight = policy.compute_contribute_reward_weight()
    # Since the reward calculation is delayed by 1 step, sample_batch[SOCIAL_INFLUENCE_REWARD][0]
    # contains the reward for timestep -1, which does not exist. Hence we shift the array.
    # Then, pad with a 0-value at the end to make the influence rewards align with sample_batch.
    # This leaks some information about the episode end though.
    contribute = np.concatenate((sample_batch['cont_reward'][1:], [0]))

    # Clip and weigh influence reward
    influence = np.clip(contribute, -policy.contribute_reward_clip, policy.contribute_reward_clip)
    influence = influence * cur_contribute_reward_weight

    # Add to trajectory
    sample_batch['cont_reward'] = influence
    sample_batch["extrinsic_reward"] = sample_batch["rewards"]
    sample_batch["rewards"] = sample_batch["rewards"] + contribute

    return sample_batch


def agent_name_to_idx(agent_num, self_id):
    """split agent id around the index and return its appropriate position in terms
    of the other agents"""
    agent_num = int(agent_num)
    if agent_num > self_id:
        return agent_num - 1
    else:
        return agent_num


def get_agent_visibility_multiplier(trajectory, num_other_agents, agent_ids):
    traj_len = len(trajectory["obs"])
    visibility = np.zeros((traj_len, num_other_agents))
    for i, v in enumerate(trajectory[VISIBILITY]):
        vis_agents = [agent_name_to_idx(a, agent_ids[i]) for a in v]
        visibility[i, vis_agents] = 1
    return visibility


def extract_last_actions_from_episodes(episodes, batch_type=False, own_actions=None):
    """Pulls every other agent's previous actions out of structured data.
    Args:
        episodes: the structured data type. Typically a dict of episode
            objects.
        batch_type: if True, the structured data is a dict of tuples,
            where the second tuple element is the relevant dict containing
            previous actions.
        own_actions: an array of the agents own actions. If provided, will
            be the first column of the created action matrix.
    Returns: a real valued array of size [batch, num_other_agents] (meaning
        each agents' actions goes down one column, each row is a timestep)
    """
    if episodes is None:
        print("Why are there no episodes?")
        import ipdb

        ipdb.set_trace()

    # Need to sort agent IDs so same agent is consistently in
    # same part of input space.
    agent_ids = sorted(episodes.keys())
    prev_actions = []

    for agent_id in agent_ids:
        if batch_type:
            prev_actions.append(episodes[agent_id][1]["actions"])
        else:
            prev_actions.append([e.prev_action for e in episodes[agent_id]])

    all_actions = np.transpose(np.array(prev_actions))

    # Attach agents own actions as column 1
    if own_actions is not None:
        all_actions = np.hstack((own_actions, all_actions))

    return all_actions


def cont_fetches(policy):
    """Adds logits, moa predictions of counterfactual actions to experience train_batches."""
    return {
        # Be aware that this is frozen here so that we don't
        # propagate agent actions through the reward
        ACTION_LOGITS: policy.model.action_logits(),
        # TODO(@evinitsky) remove this once we figure out how to split the obs
        OTHERS_ACTIONS: policy.model.other_agent_actions(),
        VISIBILITY: policy.model.visibility(),
        CONTRIBUTE_REWARD: policy.model.contribute_reward(),
        PREDICTED_ACTIONS: policy.model.predicted_actions(),
    }


class CONTConfigInitializerMixIn(object):
    def __init__(self, config):
        config = config["model"]["custom_options"]
        self.num_other_agents = config["num_other_agents"]
        self.contribute_reward_clip = config["contribute_reward_clip"]
        self.train_moa_only_when_visible = config["train_cont_only_when_visible"]
        self.contribute_divergence_measure = config["contribute_divergence_measure"]
        self.contribute_only_when_visible = config["contribute_only_when_visible"]


class CONTResetConfigMixin(object):
    @staticmethod
    def reset_policies(policies, new_config, session):
        custom_options = new_config["model"]["custom_options"]
        for policy in policies:
            policy.compute_contribute_reward_weight = lambda: custom_options[
                "contribute_reward_weight"
            ]

    def reset_config(self, new_config):
        policies = self.optimizer.policies.values()
        BaselineResetConfigMixin.reset_policies(policies, new_config)
        self.reset_policies(policies, new_config, self.optimizer.sess)
        self.config = new_config
        return True


def build_model(policy, obs_space, action_space, config):
    _, logit_dim = ModelCatalog.get_action_dist(action_space, config["model"])

    policy.model = ModelCatalog.get_model_v2(
        obs_space, action_space, logit_dim, config["model"], name=POLICY_SCOPE, framework="tf",
    )

    return policy.model


def setup_moa_mixins(policy, obs_space, action_space, config):
    ContributeScheduleMixIn.__init__(policy, config)
    CONTConfigInitializerMixIn.__init__(policy, config)


def get_moa_mixins():
    return [
        CONTConfigInitializerMixIn,
        ContributeScheduleMixIn,
    ]


def validate_moa_config(config):
    config = config["model"]["custom_options"]
    if config["contribute_reward_weight"] < 0:
        raise ValueError("Contribute reward weight must be >= 0.")