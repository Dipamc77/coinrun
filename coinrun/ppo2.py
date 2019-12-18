"""
This is a copy of PPO from openai/baselines (https://github.com/openai/baselines/blob/52255beda5f5c8760b0ae1f676aa656bb1a61f80/baselines/ppo2/ppo2.py) with some minor changes.
"""

import time
import joblib
import numpy as np
import tensorflow as tf
from collections import deque

from mpi4py import MPI

from coinrun.tb_utils import TB_Writer
import coinrun.main_utils as utils

from coinrun.config import Config

mpi_print = utils.mpi_print

from baselines.common.runners import AbstractEnvRunner
from baselines.common.tf_util import initialize
from baselines.common.mpi_util import sync_from_root

class MpiAdamOptimizer(tf.train.AdamOptimizer):
    """Adam optimizer that averages gradients across mpi processes."""
    def __init__(self, comm, **kwargs):
        self.comm = comm
        self.train_frac = 1.0 - Config.get_test_frac()
        tf.train.AdamOptimizer.__init__(self, **kwargs)
    def compute_gradients(self, loss, var_list, **kwargs):
        grads_and_vars = tf.train.AdamOptimizer.compute_gradients(self, loss, var_list, **kwargs)
        grads_and_vars = [(g, v) for g, v in grads_and_vars if g is not None]

        flat_grad = tf.concat([tf.reshape(g, (-1,)) for g, v in grads_and_vars], axis=0)

        if Config.is_test_rank():
            flat_grad = tf.zeros_like(flat_grad)

        shapes = [v.shape.as_list() for g, v in grads_and_vars]
        sizes = [int(np.prod(s)) for s in shapes]

        num_tasks = self.comm.Get_size()
        buf = np.zeros(sum(sizes), np.float32)

        def _collect_grads(flat_grad):
            self.comm.Allreduce(flat_grad, buf, op=MPI.SUM)
            np.divide(buf, float(num_tasks) * self.train_frac, out=buf)
            return buf

        avg_flat_grad = tf.py_func(_collect_grads, [flat_grad], tf.float32)
        avg_flat_grad.set_shape(flat_grad.shape)
        avg_grads = tf.split(avg_flat_grad, sizes, axis=0)
        avg_grads_and_vars = [(tf.reshape(g, v.shape), v)
                    for g, (_, v) in zip(avg_grads, grads_and_vars)]

        return avg_grads_and_vars

class Model(object):
    def __init__(self, *, policy, ob_space, ac_space, nbatch_act, nbatch_train,
                nsteps, ent_coef, vf_coef, max_grad_norm):
        sess = tf.get_default_session()

        train_model = policy(sess, ob_space, ac_space, nbatch_train, nsteps)
        norm_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        act_model = policy(sess, ob_space, ac_space, nbatch_act, 1)

        A = train_model.pdtype.sample_placeholder([None])
        ADV = tf.placeholder(tf.float32, [None])
        R = tf.placeholder(tf.float32, [None])
        OLDNEGLOGPAC = tf.placeholder(tf.float32, [None])
        OLDVPRED = tf.placeholder(tf.float32, [None])
        LR = tf.placeholder(tf.float32, [])
        CLIPRANGE = tf.placeholder(tf.float32, [])

        # Dipam:  Add placeholder for discriminator labels and hyperparameters
        #DISC_LR = tf.placeholder(tf.float32, [])
        DISC_LAM = tf.placeholder(tf.float32, [])
        DISC_LABELS = tf.placeholder(tf.int64, [None])

        neglogpac = train_model.pd.neglogp(A)
        entropy = tf.reduce_mean(train_model.pd.entropy())

        #Dipam: Add loss for domain discriminator here       
        disc_logits = train_model.disc_logits
        
        domain_onehot = tf.one_hot(DISC_LABELS, 2)
        disc_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=disc_logits, labels = domain_onehot))
        #disc_trainer = tf.train.AdamOptimizer(learning_rate = DISC_LR, epsilon=1e-5)

        vpred = train_model.vf
        vpredclipped = OLDVPRED + tf.clip_by_value(train_model.vf - OLDVPRED, - CLIPRANGE, CLIPRANGE)
        vf_losses1 = tf.square(vpred - R)
        vf_losses2 = tf.square(vpredclipped - R)
        vf_loss = .5 * tf.reduce_mean(tf.maximum(vf_losses1, vf_losses2))
        ratio = tf.exp(OLDNEGLOGPAC - neglogpac)
        pg_losses = -ADV * ratio
        pg_losses2 = -ADV * tf.clip_by_value(ratio, 1.0 - CLIPRANGE, 1.0 + CLIPRANGE)
        pg_loss = tf.reduce_mean(tf.maximum(pg_losses, pg_losses2))
        approxkl = .5 * tf.reduce_mean(tf.square(neglogpac - OLDNEGLOGPAC))
        clipfrac = tf.reduce_mean(tf.to_float(tf.greater(tf.abs(ratio - 1.0), CLIPRANGE)))

        params = tf.trainable_variables()
        weight_params = [v for v in params if '/b' not in v.name]

        total_num_params = 0

        for p in params:
            shape = p.get_shape().as_list()
            num_params = np.prod(shape)
            mpi_print('param', p, num_params)
            total_num_params += num_params

        mpi_print('total num params:', total_num_params)

        l2_loss = tf.reduce_sum([tf.nn.l2_loss(v) for v in weight_params])

        loss = pg_loss - entropy * ent_coef + vf_loss * vf_coef + l2_loss * Config.L2_WEIGHT

        #if Config.SYNC_FROM_ROOT:
        #    trainer = MpiAdamOptimizer(MPI.COMM_WORLD, learning_rate=LR, epsilon=1e-5)
        #else:
        orig_trainer = tf.train.AdamOptimizer(learning_rate=LR, epsilon=1e-5)
        feat_trainer = tf.train.AdamOptimizer(learning_rate=LR, epsilon=1e-5)
        disc_trainer = tf.train.AdamOptimizer(learning_rate=LR, epsilon=1e-5)
        polc_trainer = tf.train.AdamOptimizer(learning_rate=LR, epsilon=1e-5)

        feat_params = tf.trainable_variables("model/features")
        disc_params = tf.trainable_variables("model/discriminator")
        polc_params = tf.trainable_variables("model/policy")

        feat_loss = loss - tf.multiply(DISC_LAM,disc_loss) # Flip gradients from discriminator

        feat_grad_var = feat_trainer.compute_gradients(feat_loss, feat_params)
        polc_grad_var = polc_trainer.compute_gradients(loss, polc_params)
        disc_grad_var = disc_trainer.compute_gradients(disc_loss, disc_params) 

        grads_and_var = orig_trainer.compute_gradients(loss, params)
        # Dipam: Compute discriminator gradients and apply here along with policy gradients
 
        grads, var = zip(*grads_and_var)
        # Dipam: Add discriminator gradients to policy gradients        
        
        if max_grad_norm is not None:
            grads, _grad_norm = tf.clip_by_global_norm(grads, max_grad_norm)
        grads_and_var = list(zip(grads, var))

       # def apply_max_grad_norm(grads_and_var):
       #     grads, var = zip(*grads_and_var)
       # 
       #     if max_grad_norm is not None:
       #         grads, _grad_norm = tf.clip_by_global_norm(grads, max_grad_norm)
       #     return list(zip(grads, var))
# Dipam : TODO: This separate grad norm clipping is not correct, 
# correct method: ppend all the grads and vars -> clip by global norm-> separate-> apply individually
       # feat_grad_var = apply_max_grad_norm(feat_grad_var) 
       # polc_grad_var = apply_max_grad_norm(polc_grad_var)
       # disc_grad_var = apply_max_grad_norm(disc_grad_var)
        
        _train = orig_trainer.apply_gradients(grads_and_var)
        _train_feat = feat_trainer.apply_gradients(feat_grad_var)
        _train_polc = polc_trainer.apply_gradients(polc_grad_var)
        _train_disc = disc_trainer.apply_gradients(disc_grad_var)

        def train(lr, cliprange, disc_lam, obs, returns, masks, actions, values, neglogpacs, levelids, states=None):
            advs = returns - values

            adv_mean = np.mean(advs, axis=0, keepdims=True)
            adv_std = np.std(advs, axis=0, keepdims=True)
            advs = (advs - adv_mean) / (adv_std + 1e-8)
            
            domain_labels = levelids % 2

            td_map = {train_model.X:obs, A:actions, ADV:advs, R:returns, LR:lr,
                    CLIPRANGE:cliprange, OLDNEGLOGPAC:neglogpacs, OLDVPRED:values,
                    DISC_LABELS: domain_labels, DISC_LAM: disc_lam}

            if states is not None:
                td_map[train_model.S] = states
                td_map[train_model.M] = masks

            if disc_lam == 0:
                return sess.run(
                        [pg_loss, vf_loss, entropy, approxkl, clipfrac, l2_loss, loss,_train],
                        td_map)[:-1]
            else:
                return sess.run(
                    [pg_loss, vf_loss, entropy, approxkl, clipfrac, l2_loss, loss , feat_loss, disc_loss, 
                    _train_feat, _train_polc, _train_disc],
                    td_map)[:-3]
        self.loss_names = ['policy_grad_loss', 'value_loss', 'policy_entropy', 'approxkl', 'clipfrac', 'l2_loss', 
                           'total_loss']
        self.disc_loss_names = ['policy_grad_loss', 'value_loss', 'policy_entropy', 'approxkl', 'clipfrac', 'l2_loss', 
                                'total_loss', 'feat_loss', 'disc_loss']

        def save(save_path):
            ps = sess.run(params)
            joblib.dump(ps, save_path)

        def load(load_path):
            loaded_params = joblib.load(load_path)
            restores = []
            for p, loaded_p in zip(params, loaded_params):
                restores.append(p.assign(loaded_p))
            sess.run(restores)

        self.train = train
        self.train_model = train_model
        self.act_model = act_model
        self.step = act_model.step
        self.value = act_model.value
        self.initial_state = act_model.initial_state
        self.save = save
        self.load = load

        if Config.SYNC_FROM_ROOT:
            if MPI.COMM_WORLD.Get_rank() == 0:
                initialize()
            
            global_variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="")
            sync_from_root(sess, global_variables) #pylint: disable=E1101
        else:
            initialize()

class Runner(AbstractEnvRunner):
    def __init__(self, *, env, model, nsteps, gamma, lam):
        super().__init__(env=env, model=model, nsteps=nsteps)
        self.lam = lam
        self.gamma = gamma

    def run(self):
        # Here, we init the lists that will contain the mb of experiences
        mb_obs, mb_rewards, mb_actions, mb_values, mb_dones, mb_neglogpacs = [],[],[],[],[],[]
        mb_levelids = []
        mb_states = self.states
        epinfos = []
        # For n in range number of steps
        for _ in range(self.nsteps):
            # Given observations, get action value and neglopacs
            # We already have self.obs because Runner superclass run self.obs[:] = env.reset() on init
            actions, values, self.states, neglogpacs = self.model.step(self.obs, self.states, self.dones)
            # Dipam : Change output of env.reset to return level id also,
            # Dipam : Slice level id out of infos and append to the return value 
            mb_obs.append(self.obs.copy())
            mb_actions.append(actions)
            mb_values.append(values)
            mb_neglogpacs.append(neglogpacs)
            mb_dones.append(self.dones)

            # Take actions in env and look the results
            # Infos contains a ton of useful informations
            self.obs[:], rewards, self.dones, infos = self.env.step(actions)
            for info in infos:
                maybeepinfo = info.get('episode')
                if maybeepinfo: epinfos.append(maybeepinfo)
            mb_rewards.append(rewards)
            levelids = infos[-1].get('level_id')
            mb_levelids.append(levelids)
        #batch of steps to batch of rollouts
        mb_obs = np.asarray(mb_obs, dtype=self.obs.dtype)
        mb_rewards = np.asarray(mb_rewards, dtype=np.float32)
        mb_actions = np.asarray(mb_actions)
        mb_values = np.asarray(mb_values, dtype=np.float32)
        mb_neglogpacs = np.asarray(mb_neglogpacs, dtype=np.float32)
        mb_dones = np.asarray(mb_dones, dtype=np.bool)
        mb_levelids = np.asarray(mb_levelids, dtype=np.int32)
        last_values = self.model.value(self.obs, self.states, self.dones)

        # discount/bootstrap off value fn
        mb_returns = np.zeros_like(mb_rewards)
        mb_advs = np.zeros_like(mb_rewards)
        lastgaelam = 0
        for t in reversed(range(self.nsteps)):
            if t == self.nsteps - 1:
                nextnonterminal = 1.0 - self.dones
                nextvalues = last_values
            else:
                nextnonterminal = 1.0 - mb_dones[t+1]
                nextvalues = mb_values[t+1]
            delta = mb_rewards[t] + self.gamma * nextvalues * nextnonterminal - mb_values[t]
            mb_advs[t] = lastgaelam = delta + self.gamma * self.lam * nextnonterminal * lastgaelam
        mb_returns = mb_advs + mb_values

        return (*map(sf01, (mb_obs, mb_returns, mb_dones, mb_actions, mb_values, mb_neglogpacs, mb_levelids)),
            mb_states, epinfos)

def sf01(arr):
    """
    swap and then flatten axes 0 and 1
    """
    s = arr.shape
    return arr.swapaxes(0, 1).reshape(s[0] * s[1], *s[2:])


def constfn(val):
    def f(_):
        return val
    return f


def learn(*, policy, env, nsteps, total_timesteps, ent_coef, lr,
            vf_coef=0.5,  max_grad_norm=0.5, gamma=0.99, lam=0.95,
            log_interval=10, nminibatches=4, noptepochs=4, cliprange=0.2,
            save_interval=0, load_path=None):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    mpi_size = comm.Get_size()

    sess = tf.get_default_session()
    tb_writer = TB_Writer(sess)

    if isinstance(lr, float): lr = constfn(lr)
    else: assert callable(lr)
    if isinstance(cliprange, float): cliprange = constfn(cliprange)
    else: assert callable(cliprange)
    total_timesteps = int(total_timesteps)

    nenvs = env.num_envs
    ob_space = env.observation_space
    ac_space = env.action_space
    nbatch = nenvs * nsteps
    
    nbatch_train = nbatch // nminibatches

    model = Model(policy=policy, ob_space=ob_space, ac_space=ac_space, nbatch_act=nenvs, nbatch_train=nbatch_train,
                    nsteps=nsteps, ent_coef=ent_coef, vf_coef=vf_coef,
                    max_grad_norm=max_grad_norm)

    utils.load_all_params(sess)

    runner = Runner(env=env, model=model, nsteps=nsteps, gamma=gamma, lam=lam)

    epinfobuf10 = deque(maxlen=10)
    epinfobuf100 = deque(maxlen=100)
    tfirststart = time.time()
    active_ep_buf = epinfobuf100

    nupdates = total_timesteps//nbatch
    mean_rewards = []
    datapoints = []

    run_t_total = 0
    train_t_total = 0

    can_save = True
    checkpoints = [32, 64]
    saved_key_checkpoints = [False] * len(checkpoints)

    if Config.SYNC_FROM_ROOT and rank != 0:
        can_save = False

    def save_model(base_name=None):
        base_dict = {'datapoints': datapoints}
        utils.save_params_in_scopes(sess, ['model'], Config.get_save_file(base_name=base_name), base_dict)

    for update in range(1, nupdates+1):
        assert nbatch % nminibatches == 0
        nbatch_train = nbatch // nminibatches
        tstart = time.time()
        frac = 1.0 - (update - 1.0) / nupdates
        lrnow = lr(frac)
        cliprangenow = cliprange(frac)
        if update > 1500: # Dipam: Apply discriminator for only 80% of the training 
            disc_lam = 0
        elif update < 400:
            disc_lam = 0
        else:
            disc_p = (update-400)/nupdates*8 
            disc_lam = 2/(1 + np.exp(-10*disc_p)) - 1
            dis_lam = disc_lam*0.3

        mpi_print('collecting rollouts...')
        run_tstart = time.time()

        obs, returns, masks, actions, values, neglogpacs, levelids, states, epinfos = runner.run()
        epinfobuf10.extend(epinfos)
        epinfobuf100.extend(epinfos)

        run_elapsed = time.time() - run_tstart
        run_t_total += run_elapsed
        mpi_print('rollouts complete')

        mblossvals = []

        mpi_print('updating parameters...')
        train_tstart = time.time()

        if states is None: # nonrecurrent version
            inds = np.arange(nbatch)
            for _ in range(noptepochs):
                np.random.shuffle(inds)
                for start in range(0, nbatch, nbatch_train):
                    end = start + nbatch_train
                    mbinds = inds[start:end]
                    slices = (arr[mbinds] for arr in (obs, returns, masks, actions, values, neglogpacs, levelids))
                    mblossvals.append(model.train(lrnow, cliprangenow, disc_lam, *slices))
        else: # recurrent version
            assert nenvs % nminibatches == 0
            envinds = np.arange(nenvs)
            flatinds = np.arange(nenvs * nsteps).reshape(nenvs, nsteps)
            envsperbatch = nbatch_train // nsteps
            for _ in range(noptepochs):
                np.random.shuffle(envinds)
                for start in range(0, nenvs, envsperbatch):
                    end = start + envsperbatch
                    mbenvinds = envinds[start:end]
                    mbflatinds = flatinds[mbenvinds].ravel()
                    slices = (arr[mbflatinds] for arr in (obs, returns, masks, actions, values, neglogpacs))
                    mbstates = states[mbenvinds]
                    mblossvals.append(model.train(lrnow, cliprangenow, *slices, mbstates))

        # update the dropout mask
        sess.run([model.train_model.dropout_assign_ops])

        train_elapsed = time.time() - train_tstart
        train_t_total += train_elapsed
        mpi_print('update complete')

        lossvals = np.mean(mblossvals, axis=0)
        tnow = time.time()
        fps = int(nbatch/ (tnow - tstart))

        if update % log_interval == 0 or update == 1:
            step = update*nbatch
            rew_mean_10 = utils.process_ep_buf(active_ep_buf, tb_writer=tb_writer, suffix='', step=step)
            ep_len_mean = np.nanmean([epinfo['l'] for epinfo in active_ep_buf])
            
            mpi_print('\n----', update)

            mean_rewards.append(rew_mean_10)
            datapoints.append([step, rew_mean_10])

            tb_writer.log_scalar(ep_len_mean, 'ep_len_mean')
            tb_writer.log_scalar(fps, 'fps')

            mpi_print('time_elapsed', tnow - tfirststart, (tnow - tfirststart)/3600., run_t_total, train_t_total)
            mpi_print('timesteps', update*nsteps, total_timesteps)

            mpi_print('eplenmean', ep_len_mean)
            mpi_print('eprew', rew_mean_10)
            mpi_print('fps', fps)
            mpi_print('total_timesteps', update*nbatch)
            mpi_print([epinfo['r'] for epinfo in epinfobuf10])

            if len(mblossvals):
                if disc_lam == 0:
                    for (lossval, lossname) in zip(lossvals, model.loss_names):
                        mpi_print(lossname, lossval)
                        tb_writer.log_scalar(lossval, lossname)
                else:
                    for (lossval, lossname) in zip(lossvals, model.disc_loss_names):
                        mpi_print(lossname, lossval)
                        tb_writer.log_scalar(lossval, lossname)
            mpi_print('----\n')

        if can_save:
            if save_interval and (update % save_interval == 0):
                save_model()

            for j, checkpoint in enumerate(checkpoints):
                if (not saved_key_checkpoints[j]) and (step >= (checkpoint * 1e6)):
                    saved_key_checkpoints[j] = True
                    save_model(str(checkpoint) + 'M')

    save_model()

    env.close()
    return mean_rewards
