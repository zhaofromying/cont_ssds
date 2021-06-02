from __future__ import absolute_import, division, print_function

from ray.rllib.agents.ppo.ppo import (
    choose_policy_optimizer,
    update_kl,
    validate_config,
    warn_about_bad_reward_scales,
)
from ray.rllib.agents.ppo.ppo_tf_policy import (
    KLCoeffMixin,
    PPOLoss,
    ValueNetworkMixin,
    clip_gradients,
    kl_and_loss_stats,
    postprocess_ppo_gae,
    ppo_surrogate_loss,
    setup_config,
    # setup_mixins,
    vf_preds_fetches,
)
from ray.rllib.models import ModelCatalog
from ray.rllib.agents.trainer_template import build_trainer
from ray.rllib.evaluation.postprocessing import Postprocessing
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.policy import build_tf_policy
from ray.rllib.policy.tf_policy import EntropyCoeffSchedule, LearningRateSchedule
from ray.rllib.utils import try_import_tf

from algorithms.common_funcs_baseline import BaselineResetConfigMixin, cont_postprocess_trajectory, ContributeScheduleMixIn
tf = try_import_tf()

POLICY_SCOPE = "func"


def extra_cont_stats(policy, train_batch):
    """
    Add stats that are logged in progress.csv
    :return: Combined PPO+MOA stats
    """
    base_stats = kl_and_loss_stats(policy, train_batch)
    base_stats = {
        **base_stats,
        "var_gnorm": tf.global_norm([x for x in policy.model.trainable_variables()]),
        "cur_contribute_reward_weight": tf.cast(
            policy.cur_contribute_reward_weight_tensor, tf.float32
        ),
        'cont_rewards': train_batch['cont_rewards'],
        "extrinsic_rewards": train_batch['extrinsic_rewards'],
    }

    return base_stats


def postprocess_ppo_cont(policy, sample_batch, other_agent_batches=None, episode=None):
    """
    Add the influence reward to the trajectory.
    Then, add the policy logits, VF preds, and advantages to the trajectory.
    :return: Updated trajectory (batch)
    """
    batch = cont_postprocess_trajectory(policy, sample_batch)
    batch = postprocess_ppo_gae(policy, batch)
    return batch


def setup_ppo_cont_mixins(policy, obs_space, action_space, config):
    """
    Calls init on all PPO+MOA mixins in the policy
    """
    ValueNetworkMixin.__init__(policy, obs_space, action_space, config)
    KLCoeffMixin.__init__(policy, config)
    EntropyCoeffSchedule.__init__(policy, config["entropy_coeff"], config["entropy_coeff_schedule"])
    LearningRateSchedule.__init__(policy, config["lr"], config["lr_schedule"])
    ContributeScheduleMixIn.__init__(policy, config)


def build_model(policy, obs_space, action_space, config):
    _, logit_dim = ModelCatalog.get_action_dist(action_space, config["model"])

    policy.model = ModelCatalog.get_model_v2(
        obs_space, action_space, logit_dim, config["model"], name=POLICY_SCOPE, framework="tf",
    )

    return policy.model


def build_ppo_baseline_trainer(config):
    """
    Creates a PPO policy class, then creates a trainer with this policy.
    :param config: The configuration dictionary.
    :return: A new PPO trainer.
    """
    tf.keras.backend.set_floatx("float32")

    policy = build_tf_policy(
        name="ContPPOTFPolicy",
        get_default_config=lambda: config,
        loss_fn=ppo_surrogate_loss,
        make_model=build_model,
        stats_fn=kl_and_loss_stats,
        extra_action_fetches_fn=vf_preds_fetches,
        postprocess_fn=postprocess_ppo_cont,
        gradients_fn=clip_gradients,
        before_init=setup_config,
        before_loss_init=setup_ppo_cont_mixins,
        mixins=[LearningRateSchedule, EntropyCoeffSchedule, KLCoeffMixin, ValueNetworkMixin, ContributeScheduleMixIn],
    )

    ppo_trainer = build_trainer(
        name="BaselinePPOTrainer",
        make_policy_optimizer=choose_policy_optimizer,
        default_policy=policy,
        default_config=config,
        validate_config=validate_config,
        after_optimizer_step=update_kl,
        after_train_result=warn_about_bad_reward_scales,
        mixins=[BaselineResetConfigMixin],
    )
    return ppo_trainer
