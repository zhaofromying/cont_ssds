from __future__ import absolute_import, division, print_function

from ray.rllib.agents.a3c.a3c import get_policy_class, make_async_optimizer, validate_config
from ray.rllib.agents.a3c.a3c_tf_policy import A3CTFPolicy
from ray.rllib.agents.trainer_template import build_trainer

from ray.rllib.policy.tf_policy_template import build_tf_policy
from ray.rllib.utils import try_import_tf
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.evaluation.postprocessing import Postprocessing
from ray.rllib.utils.explained_variance import explained_variance
from ray.rllib.agents.a3c.a3c_tf_policy import postprocess_advantages
from ray.rllib.utils.tf_ops import make_tf_callable
from ray.rllib.policy.tf_policy import LearningRateSchedule

from algorithms.common_funcs_baseline import (cont_postprocess_trajectory, ContributeScheduleMixIn)

tf = try_import_tf()


class A3CLoss(object):
    def __init__(
        self, action_dist, actions, advantages, v_target, vf, vf_loss_coeff=0.5, entropy_coeff=0.01,
    ):
        log_prob = action_dist.logp(actions)

        # The "policy gradients" loss
        self.pi_loss = -tf.reduce_sum(log_prob * advantages)

        delta = vf - v_target
        self.vf_loss = 0.5 * tf.reduce_sum(tf.square(delta))
        self.entropy = tf.reduce_sum(action_dist.entropy())
        self.total_loss = self.pi_loss + self.vf_loss * vf_loss_coeff - self.entropy * entropy_coeff


def postprocess_a3c_cont(policy, sample_batch, other_agent_batches=None, episode=None):
    """Adds the policy logits, VF preds, and advantages to the trajectory."""

    batch = cont_postprocess_trajectory(policy, sample_batch)
    batch = postprocess_advantages(policy, batch)
    return batch


def actor_critic_loss(policy, model, dist_class, train_batch):
    logits, _ = model.from_batch(train_batch)
    action_dist = dist_class(logits, model)
    policy.loss = A3CLoss(
        action_dist,
        train_batch[SampleBatch.ACTIONS],
        train_batch[Postprocessing.ADVANTAGES],
        train_batch[Postprocessing.VALUE_TARGETS],
        model.value_function(),
        policy.config["vf_loss_coeff"],
        policy.config["entropy_coeff"],
    )
    return policy.loss.total_loss


class ValueNetworkMixin(object):
    def __init__(self):
        @make_tf_callable(self.get_session())
        def value(ob, prev_action, prev_reward, *state):
            model_out, _ = self.model(
                {
                    SampleBatch.CUR_OBS: tf.convert_to_tensor([ob]),
                    SampleBatch.PREV_ACTIONS: tf.convert_to_tensor([prev_action]),
                    SampleBatch.PREV_REWARDS: tf.convert_to_tensor([prev_reward]),
                    "is_training": tf.convert_to_tensor(False),
                },
                [tf.convert_to_tensor([s]) for s in state],
                tf.convert_to_tensor([1]),
            )
            return self.model.value_function()[0]

        self._value = value


def stats(policy, train_batch):
    base_stats = {
        "cur_lr": policy.cur_lr,
        "policy_loss": policy.loss.pi_loss,
        "policy_entropy": policy.loss.entropy,
        "var_gnorm": tf.global_norm([x for x in policy.model.trainable_variables()]),
        "vf_loss": policy.loss.vf_loss,
        "cur_contribute_reward_weight": tf.cast(
            policy.cur_influence_reward_weight_tensor, tf.float32
        ),
        'contribute_reward': train_batch['contribute_reward'],
        'extrinsic_reward': train_batch['extrinsic_reward'],
    }
    return base_stats


def grad_stats(policy, train_batch, grads):
    return {
        "grad_gnorm": tf.global_norm(grads),
        "vf_explained_var": explained_variance(
            train_batch[Postprocessing.VALUE_TARGETS], policy.model.value_function()
        ),
    }


def clip_gradients(policy, optimizer, loss):
    grads_and_vars = optimizer.compute_gradients(loss, policy.model.trainable_variables())
    grads = [g for (g, v) in grads_and_vars]
    grads, _ = tf.clip_by_global_norm(grads, policy.config["grad_clip"])
    clipped_grads = list(zip(grads, policy.model.trainable_variables()))
    return clipped_grads


def setup_mixins(policy, obs_space, action_space, config):
    ValueNetworkMixin.__init__(policy)
    LearningRateSchedule.__init__(policy, config["lr"], config["lr_schedule"])
    ContributeScheduleMixIn.__init__(policy, config)


def build_a3c_policy(config):
    a3c_tf_cont_policy = build_tf_policy(
        name='A3CContTFPolicy',
        get_default_config=lambda: config,
        loss_fn=actor_critic_loss,
        stats_fn=stats,
        grad_stats_fn=grad_stats,
        gradients_fn=clip_gradients,
        postprocess_fn=postprocess_a3c_cont,
        before_loss_init=setup_mixins,
        mixins=[ValueNetworkMixin, LearningRateSchedule, ContributeScheduleMixIn]
    )
    return a3c_tf_cont_policy


def build_a3c_baseline_trainer(config):

    a3c_tf_cont_policy = build_tf_policy(
        name='A3CContTFPolicy',
        get_default_config=lambda: config,
        loss_fn=actor_critic_loss,
        stats_fn=stats,
        grad_stats_fn=grad_stats,
        gradients_fn=clip_gradients,
        postprocess_fn=postprocess_a3c_cont,
        before_loss_init=setup_mixins,
        mixins=[ValueNetworkMixin, LearningRateSchedule, ContributeScheduleMixIn]
    )
    # config['multiagent']['policies_to_train'] = a3c_tf_cont_policy
    a3c_trainer = build_trainer(
        name="ContA3C",
        default_config=config,
        default_policy=a3c_tf_cont_policy,
        get_policy_class=get_policy_class,
        validate_config=validate_config,
        make_policy_optimizer=make_async_optimizer,
    )
    return a3c_trainer

