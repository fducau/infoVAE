import numpy as np
import tensorflow as tf
import input_data
from tensorflow.contrib.distributions import Normal
import copy
import datetime
import dateutil
from misc.utils import mkdir_p
from misc.datasets import BasicPropDataset, \
                          BasicPropAngleDataset, \
                          BasicPropAngleNoiseDataset, \
                          BasicPropAngleNoiseBGDataset,  \
                          MnistDataset

SAVE_MODEL_TO = './models'

np.random.seed(0)
tf.set_random_seed(0)

network_architecture = dict(n_hidden_recog_1=500,  # 1st layer encoder neurons
                            n_hidden_recog_2=500,  # 2nd layer encoder neurons
                            n_hidden_gener_1=500,  # 1st layer decoder neurons
                            n_hidden_gener_2=500,  # 2nd layer decoder neurons
                            n_input=784,  # MNIST data input (img shape: 28*28)
                            n_z=10,       # dimensionality of latent space no MI
                            n_c=10,        # dimensionality of latent space with MI
                            info=True)



def xavier_init(fan_in, fan_out, constant=1):
    """ Xavier initialization of network weights"""
    # https://stackoverflow.com/questions/33640581/how-to-do-xavier-initialization-on-tensorflow
    low = - constant * np.sqrt(6.0 / (fan_in + fan_out))
    high = constant * np.sqrt(6.0 / (fan_in + fan_out))
    return tf.random_uniform((fan_in, fan_out),
                             minval=low, maxval=high,
                             dtype=tf.float32)


def load_dataset(dataset_name='MNIST'):
    if dataset_name == 'MNIST':
        # dataset = input_data.read_data_sets('MNIST_data', one_hot=True)
        dataset = MnistDataset()
    elif dataset_name == 'BASICPROP':
        dataset = BasicPropDataset()
    elif dataset_name == 'BPAngle':
        dataset = BasicPropAngleDataset()
    elif dataset_name == 'BPAngleNoise':
        dataset = BasicPropAngleNoiseDataset()
    elif dataset_name == 'BPAngleNoiseBG':
        dataset = BasicPropAngleNoiseBGDataset()
    else:
        raise Exception("Please specify a valid dataset.")

    return dataset


class VariationalAutoencoder(object):
    """ Variation Autoencoder (VAE) with an sklearn-like interface implemented using TensorFlow.

    This implementation uses probabilistic encoders and decoders using Gaussian
    distributions and  realized by multi-layer perceptrons. The VAE can be learned
    end-to-end.

    See "Auto-Encoding Variational Bayes" by Kingma and Welling for more details.
    """
    def __init__(self, network_architecture, transfer_fct=tf.nn.softplus,
                 learning_rate=0.001, batch_size=100,
                 dataset_name='unkn'):

        self.network_architecture = network_architecture
        self.transfer_fct = transfer_fct
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.step = 0
        self.info = network_architecture['info']
        self.dataset_name = dataset_name
        self.summary_dir = './summary/'

        now = datetime.datetime.now(dateutil.tz.tzlocal())
        timestamp = now.strftime('%H_%M_%S_%Y%m%d')
        n_z = self.network_architecture["n_z"]
        n_c = self.network_architecture["n_z"]
        self.summary_dir = './summary/DS-{}_nz{}_nc{}_info{}_{}'.format(self.dataset_name,
                                                                        n_z, n_c, self.info,
                                                                        timestamp)
        self.sess = tf.InteractiveSession()

        # tf Graph input
        self.x = tf.placeholder(tf.float32, [None, network_architecture["n_input"]])

        # Create autoencoder network
        self._create_network()
        # Define loss function based variational upper-bound and
        # corresponding optimizer
        self._create_loss_optimizer()

        self.train_summary_writer = tf.train.SummaryWriter(self.summary_dir, self.sess.graph)
        self.saver = tf.train.Saver(tf.all_variables())
        # Initializing the tensor flow variables
        init = tf.initialize_all_variables()

        # Launch the session
        self.sess.run(init)

    def _create_network(self):
        # Initialize autoencode network weights and biases
        network_weights = self._initialize_weights(**self.network_architecture)

        # Use recognition network to determine mean and
        # (log) variance of Gaussian distribution in latent
        # space
        self.z_mean, self.c_mean, self.z_log_sigma_sq, self.c_log_sigma_sq = \
            self._recognition_network(network_weights["weights_recog"],
                                      network_weights["biases_recog"],
                                      self.x)

        self.z_mean_concat = tf.concat(1, [self.z_mean, self.c_mean])
        self.z_log_sigma_sq_concat = tf.concat(1, [self.z_log_sigma_sq, self.c_log_sigma_sq])


        # Compute I(Z,X) point estimate as H(Z|X)

        self.cond_ent_lat_given_x = tf.reduce_mean(tf.reduce_sum(tf.mul(tf.constant(0.5), tf.add(self.z_log_sigma_sq_concat, tf.constant(2.838))), reduction_indices=1))
        self.cond_ent_z_given_x = tf.reduce_mean(tf.reduce_sum(tf.mul(tf.constant(0.5), tf.add(self.z_log_sigma_sq, tf.constant(2.838))), reduction_indices=1))
        self.cond_ent_c_given_x = tf.reduce_mean(tf.reduce_sum(tf.mul(tf.constant(0.5), tf.add(self.c_log_sigma_sq, tf.constant(2.838))), reduction_indices=1))

        # Draw one sample z from Gaussian distribution
        n_z = self.network_architecture["n_z"]
        n_c = self.network_architecture["n_c"]

        eps = tf.random_normal((self.batch_size, n_z + n_c), 0, 1,
                               dtype=tf.float32)

        # z = mu + sigma*epsilon
        self.z = tf.add(self.z_mean_concat,
                        tf.mul(tf.sqrt(tf.exp(self.z_log_sigma_sq_concat)), eps),
                        name='z')

        # Use generator to determine mean of
        # Bernoulli distribution of reconstructed input
        self.x_reconstr_mean = \
            self._generator_network(network_weights["weights_gener"],
                                    network_weights["biases_gener"],
                                    z=self.z)

        ####
        ####
        ####
        eps = tf.random_normal((self.batch_size, n_z + n_c), 0, 1,
                               dtype=tf.float32)

        self.z_theta_concat = tf.add(0.0, tf.mul(1.0, eps), name='z_theta')
        self.z_theta = self.z_theta_concat[:, :n_z]
        self.c_theta = self.z_theta_concat[:, n_z:]

        self.x_prime = self._generator_network(network_weights["weights_gener"],
                                               network_weights["biases_gener"],
                                               z=self.z_theta_concat)

        self.z_prime_mean, self.c_prime_mean, self.z_prime_log_sigma_sq, self.c_prime_log_sigma_sq = \
            self._recognition_network(network_weights["weights_recog"],
                                      network_weights["biases_recog"],
                                      self.x_prime)

        self.z_prime_mean_concat = tf.concat(1, [self.z_prime_mean, self.c_prime_mean])
        self.z_prime_log_sigma_sq_concat = tf.concat(1, [self.z_prime_log_sigma_sq, self.c_prime_log_sigma_sq])

        # XEntropy for the code C
        dist = Normal(mu=self.c_prime_mean,
                      sigma=tf.sqrt(tf.exp(self.c_prime_log_sigma_sq)))
        logli = tf.reduce_sum(dist.log_pdf(self.c_theta, name='xc_entropy'),
                              reduction_indices=1)
        self.cross_entropy = tf.reduce_mean(- logli)
        self.entropy = tf.constant(1.4185 * n_c)

        # XEntropy for the entire latent code
        dist_all = Normal(mu=self.z_prime_mean_concat,
                          sigma=tf.sqrt(tf.exp(self.z_prime_log_sigma_sq_concat)))
        logli_all = tf.reduce_sum(dist_all.log_pdf(self.z_theta_concat, name='x_entropy_concat'),
                                  reduction_indices=1)
        self.cross_entropy_concat = tf.reduce_mean(- logli_all)
        self.entropy_concat = tf.constant(1.4185 * (n_z + n_c))

        # Entropy for the code Z
        dist_z = Normal(mu=self.z_prime_mean,
                        sigma=tf.sqrt(tf.exp(self.z_prime_log_sigma_sq)))
        logli_z = tf.reduce_sum(dist_z.log_pdf(self.z_theta, name='xz_entropy'),
                                reduction_indices=1)
        self.cross_entropy_z = tf.reduce_mean(- logli_z)
        self.entropy_z = tf.constant(1.4185 * n_z)

    def _initialize_weights(self, n_hidden_recog_1, n_hidden_recog_2,
                            n_hidden_gener_1, n_hidden_gener_2,
                            n_input, n_z, n_c, info):
        n_lat = n_z + n_c
        all_weights = dict()
        all_weights['weights_recog'] = {
            'h1': tf.Variable(xavier_init(n_input, n_hidden_recog_1)),
            'h2': tf.Variable(xavier_init(n_hidden_recog_1, n_hidden_recog_2)),
            'out_mean': tf.Variable(xavier_init(n_hidden_recog_2, n_lat)),
            'out_log_sigma': tf.Variable(xavier_init(n_hidden_recog_2, n_lat))}
        all_weights['biases_recog'] = {
            'b1': tf.Variable(tf.zeros([n_hidden_recog_1], dtype=tf.float32)),
            'b2': tf.Variable(tf.zeros([n_hidden_recog_2], dtype=tf.float32)),
            'out_mean': tf.Variable(tf.zeros([n_lat], dtype=tf.float32)),
            'out_log_sigma': tf.Variable(tf.zeros([n_lat], dtype=tf.float32))}
        all_weights['weights_gener'] = {
            'h1': tf.Variable(xavier_init(n_lat, n_hidden_gener_1)),
            'h2': tf.Variable(xavier_init(n_hidden_gener_1, n_hidden_gener_2)),
            'out_mean': tf.Variable(xavier_init(n_hidden_gener_2, n_input)),
            'out_log_sigma': tf.Variable(xavier_init(n_hidden_gener_2, n_input))}
        all_weights['biases_gener'] = {
            'b1': tf.Variable(tf.zeros([n_hidden_gener_1], dtype=tf.float32)),
            'b2': tf.Variable(tf.zeros([n_hidden_gener_2], dtype=tf.float32)),
            'out_mean': tf.Variable(tf.zeros([n_input], dtype=tf.float32)),
            'out_log_sigma': tf.Variable(tf.zeros([n_input], dtype=tf.float32))}
        return all_weights

    def _recognition_network(self, weights, biases, x):
        # Generate probabilistic encoder (recognition network), which
        # maps inputs onto a normal distribution in latent space.
        # The transformation is parametrized and can be learned.
        n_z = self.network_architecture['n_z']

        layer_1 = self.transfer_fct(tf.add(tf.matmul(x, weights['h1']),
                                           biases['b1']))
        layer_2 = self.transfer_fct(tf.add(tf.matmul(layer_1, weights['h2']),
                                           biases['b2']))

        z_mean_concat = tf.add(tf.matmul(layer_2, weights['out_mean']),
                               biases['out_mean'])
        z_log_sigma_sq_concat = \
            tf.add(tf.matmul(layer_2, weights['out_log_sigma']),
                   biases['out_log_sigma'])

        z_mean = z_mean_concat[:, :n_z]
        c_mean = z_mean_concat[:, n_z:]

        z_log_sigma_sq = z_log_sigma_sq_concat[:, :n_z]
        c_log_sigma_sq = z_log_sigma_sq_concat[:, n_z:]

        return (z_mean, c_mean, z_log_sigma_sq, c_log_sigma_sq)

    def _generator_network(self, weights, biases, z):
        # Generate probabilistic decoder (decoder network), which
        # maps points in latent space onto a Bernoulli distribution in data space.
        # The transformation is parametrized and can be learned.
        layer_1 = self.transfer_fct(tf.add(tf.matmul(z, weights['h1']),
                                           biases['b1']))
        layer_2 = self.transfer_fct(tf.add(tf.matmul(layer_1, weights['h2']),
                                           biases['b2']))
        x_reconstr_mean = \
            tf.nn.sigmoid(tf.add(tf.matmul(layer_2, weights['out_mean']),
                                 biases['out_mean']))
        return x_reconstr_mean

    def _create_loss_optimizer(self):
        # The loss is composed of two terms:
        # 1.) The reconstruction loss (the negative log probability
        #     of the input under the reconstructed Bernoulli distribution
        #     induced by the decoder in the data space).
        #     This can be interpreted as the number of "nats" required
        #     for reconstructing the input when the activation in latent
        #     is given.
        # Adding 1e-10 to avoid evaluatio of log(0.0)
        reconstr_loss = -\
            tf.reduce_sum(self.x * tf.log(1e-10 + self.x_reconstr_mean) +
                          (1 - self.x) * tf.log(1e-10 + 1 - self.x_reconstr_mean), 1,
                          name='reconstruction_loss')

        # 2.) The latent loss, which is defined as the Kullback Leibler divergence
        #     between the distribution in latent space induced by the encoder on
        #     the data and some prior. This acts as a kind of regularizer.
        #     This can be interpreted as the number of "nats" required
        #     for transmitting the the latent space distribution given
        #     the prior.

        latent_loss = -0.5 * tf.reduce_sum(1 + self.z_log_sigma_sq_concat -
                                           tf.square(self.z_mean_concat) -
                                           tf.exp(self.z_log_sigma_sq_concat), 1,
                                           name='latent_loss')

        latent_loss_z = -0.5 * tf.reduce_sum(1 + self.z_log_sigma_sq -
                                           tf.square(self.z_mean) -
                                           tf.exp(self.z_log_sigma_sq), 1,
                                           name='latent_loss_z')

        latent_loss_c = -0.5 * tf.reduce_sum(1 + self.c_log_sigma_sq -
                                           tf.square(self.c_mean) -
                                           tf.exp(self.c_log_sigma_sq), 1,
                                           name='latent_loss_c')

        # 3.) Mutual Information loss
        self.lmbda = tf.constant(1.0)
        self.MI = tf.add(self.entropy, - self.cross_entropy, name='MI_loss')

        if self.info:
            self.cost = tf.reduce_mean(reconstr_loss + latent_loss - self.MI)
        else:
            self.cost = tf.reduce_mean(reconstr_loss + latent_loss)
            # self.cost = tf.reduce_mean(reconstr_loss - 10. * self.MI)

        # Use ADAM optimizer
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.cost)

        rec_summary = tf.scalar_summary('reconstruction loss', tf.reduce_mean(reconstr_loss))

        latent_summary = tf.scalar_summary('KLD q(z_concat|x) || p(z)', tf.reduce_mean(latent_loss))
        latent_summary_z = tf.scalar_summary('KLD q(z|x) || p(z)', tf.reduce_mean(latent_loss_z))
        latent_summary_c = tf.scalar_summary('KLD q(c|x) || p(z)', tf.reduce_mean(latent_loss_c))

        cost_summary = tf.scalar_summary('Cost', self.cost)

        MI_summary = tf.scalar_summary('MI c', self.MI)
        MI_summary_concat = tf.scalar_summary('MI_concat', tf.add(self.entropy_concat, - self.cross_entropy_concat, name='MI_losss_concat'))
        MI_summary_z = tf.scalar_summary('MI_z', tf.add(self.entropy_z, - self.cross_entropy_z, name='MI_losss_z'))

        MI_lat_input = tf.scalar_summary('MI_INPUT_LAT', tf.add(self.entropy_concat, - self.cond_ent_lat_given_x))
        MI_z_input = tf.scalar_summary('MI_INPUT_Z', tf.add(self.entropy_z, - self.cond_ent_z_given_x))
        MI_c_input = tf.scalar_summary('MI_INPUT_C', tf.add(self.entropy, - self.cond_ent_c_given_x))

        sigma_summary = tf.scalar_summary('Sigma', tf.reduce_mean(tf.sqrt(tf.exp(self.z_log_sigma_sq_concat))))
        mu_summary = tf.scalar_summary('mu', tf.reduce_mean(self.z_mean_concat))

        summaries = [rec_summary, latent_summary, cost_summary, MI_summary, MI_summary_z,
                     MI_summary_concat, sigma_summary, mu_summary, MI_lat_input,
                     MI_z_input, MI_c_input,
                     latent_summary_z, latent_summary_c]

        self.merged = tf.merge_summary(summaries)

    def partial_fit(self, X, last=False):
        """Train model based on mini-batch of input data.

        Return cost of mini-batch.
        """

        opt, cost, cross_entropy, MI, summary = \
            self.sess.run((self.optimizer, self.cost,
                          self.cross_entropy,
                          self.MI,
                          self.merged),
                          feed_dict={self.x: X})

        self.train_summary_writer.add_summary(summary, self.step)
        if last:
            now = datetime.datetime.now(dateutil.tz.tzlocal())
            timestamp = now.strftime('%H_%M_%S_%Y%m%d')
            n_z = self.network_architecture['n_z']
            n_c = self.network_architecture['n_c']

            savefolder = '{}/DS-{}_nz{}_nc{}_info{}_{}'.format(SAVE_MODEL_TO,
                                                               self.dataset_name,
                                                               n_z, n_c,
                                                               self.info,
                                                               timestamp)
            mkdir_p(savefolder)
            self.saver.save(self.sess, '{}/model'.format(savefolder))

        self.step += 1

        return cost

    def test_cost(self, X):
        cost, mi_loss = self.sess.run((self.cost, self.MI), feed_dict={self.x: X})
        info = self.network_architecture['info']
        if info:
            return cost + mi_loss
        else:
            return cost

    def transform(self, X):
        """Transform data by mapping it into the latent space."""
        # Note: This maps to mean of distribution, we could alternatively
        # sample from Gaussian distribution
        return self.sess.run(self.z_mean_concat, feed_dict={self.x: X})

    def generate(self, z_mu=None):
        """ Generate data by sampling from latent space.

        If z_mu is not None, data for this point in latent space is
        generated. Otherwise, z_mu is drawn from prior in latent
        space.
        """
        if z_mu is None:
            z_mu = np.random.normal(size=self.network_architecture['n_z'] + self.network_architecture['n_c'])
        # Note: This maps to mean of distribution, we could alternatively
        # sample from Gaussian distribution
        return self.sess.run(self.x_reconstr_mean,
                             feed_dict={self.z: z_mu})

    def reconstruct(self, X):
        """ Use VAE to reconstruct given data. """
        return self.sess.run(self.x_reconstr_mean,
                             feed_dict={self.x: X})


def train(network_architecture, learning_rate=0.001,
          batch_size=100, training_epochs=10, display_step=5,
          info=False, dataset='MNIST', n_z=None, n_c=None):


    mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
    n_samples = mnist.train.num_examples

    if isinstance(dataset, str):
        dataset = load_dataset(dataset)

    network_arch = copy.deepcopy(network_architecture)
    network_arch['info'] = info

    vae = VariationalAutoencoder(network_arch,
                                 learning_rate=learning_rate,
                                 batch_size=batch_size,
                                 dataset_name=dataset.dataset_name)

    # Training cycle
    for epoch in range(training_epochs):
        avg_cost = 0.
        total_batch = int(n_samples / batch_size)
        # Loop over all batches
        for i in range(total_batch):
            batch_xs, _ = dataset.train.next_batch(batch_size)
            if dataset.dataset_name == "BASICPROP-angle":
                batch_xs = np.ceil(batch_xs)

            # Fit training using batch data
            if epoch == training_epochs - 1 and i == total_batch - 1:
                    cost = vae.partial_fit(batch_xs, last=True)
            else:
                cost = vae.partial_fit(batch_xs)
            # Compute average loss
            avg_cost += cost / n_samples * batch_size

        # Display logs per epoch step
        if epoch % display_step == 0:
            print "Epoch:", '%04d' % (epoch + 1), "cost=", "{:.9f}".format(avg_cost)

    return vae


def evaluate(vae, dataset='MNIST'):

    batch_size = 100
    mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
    n_samples = mnist.test.num_examples

    if isinstance(dataset, str):
        dataset = load_dataset(dataset)

    network_architecture = vae.network_architecture

    avg_cost = 0.
    total_batch = int(n_samples / batch_size)
    # Loop over all batches
    for i in range(total_batch):
        batch_xs, _ = dataset.test.next_batch(batch_size)
        cost = vae.test_cost(batch_xs)
        # Compute average loss
        avg_cost += cost / n_samples * batch_size


    return avg_cost







def main():
    vae = train(network_architecture, training_epochs=25, info=True)


if __name__ == '__main__':
    main()
