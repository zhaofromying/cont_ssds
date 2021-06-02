import numpy as np
from ray.rllib.utils import try_import_tf
from ray.rllib.utils.annotations import override
from ray.rllib.policy.policy import Policy

tf = try_import_tf()


class BaselineResetConfigMixin(object):
    @staticmethod
    def reset_policies(policies, new_config):
        for policy in policies:
            policy.entropy_coeff_schedule.value = lambda _: new_config["entropy_coeff"]
            policy.config["entropy_coeff"] = new_config["entropy_coeff"]
            policy.lr_schedule.value = lambda _: new_config["lr"]
            policy.config["lr"] = new_config["lr"]

    def reset_config(self, new_config):
        self.reset_policies(self.optimizer.policies.values(), new_config)
        self.config = new_config
        return True


class ContributeScheduleMixIn(object):
    def __init__(self, config):
        config = config["model"]["custom_options"]
        self.baseline_contribute_reward_weight = config["contribute_reward_weight"]
        if any(
            config[key] is None
            for key in ["contribute_reward_schedule_steps", "contribute_reward_schedule_weights"]
        ):
            self.compute_contribute_reward_weight = lambda: self.baseline_contribute_reward_weight
        # self.contribute_reward_schedule_steps = config["contribute_reward_schedule_steps"]
        self.contribute_reward_clip = config['contribute_reward_clip']
        self.contribute_reward_schedule_steps = [0, 10000000, 1e8, 3e8]
        self.contribute_reward_schedule_weights = [0.0, 0.0, 1.0, 0.5]
        # self.contribute_reward_schedule_weights = config["contribute_reward_schedule_weights"]
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
    sample_batch = weight_and_add_contribute_reward(policy, sample_batch)

    return sample_batch


def weight_and_add_contribute_reward(policy, sample_batch):
    # cur_contribute_reward_weight = policy.compute_contribute_reward_weight()
    # Since the reward calculation is delayed by 1 step, sample_batch[SOCIAL_INFLUENCE_REWARD][0]
    # contains the reward for timestep -1, which does not exist. Hence we shift the array.
    # Then, pad with a 0-value at the end to make the influence rewards align with sample_batch.
    # This leaks some information about the episode end though.
    # contribute = np.concatenate((sample_batch['contribute_reward'][1:], [0]))
    contribute = sample_batch['cont_rewards']
    # Clip and weigh influence reward
    contribute = np.clip(contribute, -policy.contribute_reward_clip, policy.contribute_reward_clip)
    contribute = contribute * policy.baseline_contribute_reward_weight
    # Add to trajectory
    sample_batch['cont_rewards'] = contribute
    sample_batch["extrinsic_reward"] = sample_batch["rewards"]
    sample_batch["rewards"] = sample_batch["rewards"] + contribute

    return sample_batch