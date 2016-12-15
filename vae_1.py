import numpy as np
import tensorflow as tf
import input_data
from tensorflow.contrib.distributions import Normal
from distributions import Gaussian

np.random.seed(0)
tf.set_random_seed(0)

network_architecture = dict(n_hidden_recog_1=500,  # 1st layer encoder neurons
                            n_hidden_recog_2=500,  # 2nd layer encoder neurons
                            n_hidden_gener_1=500,  # 1st layer decoder neurons
                            n_hidden_gener_2=500,  # 2nd layer decoder neurons
                            n_input=784,  # MNIST data input (img shape: 28*28)
                            n_z=20,       # dimensionality of latent space
                            info=False)

# Load MNIST data in a format suited for tensorflow.
# The script input_data is available under this URL:
# https://raw.githubusercontent.com/tensorflow/tensorflow/master/tensorflow/g3doc/tutorials/mnist/input_data.py
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
n_samples = mnist.train.num_examples


def xavier_init(fan_in, fan_out, constant=1):
    """ Xavier initialization of network weights"""
    # https://stackoverflow.com/questions/33640581/how-to-do-xavier-initialization-on-tensorflow
    low = - constant * np.sqrt(6.0 / (fan_in + fan_out))
    high = constant * np.sqrt(6.0 / (fan_in + fan_out))
    return tf.random_uniform((fan_in, fan_out),
                             minval=low, maxval=high,
                             dtype=tf.float32)


class VariationalAutoencoder(object):
    """ Variation Autoencoder (VAE) with an sklearn-like interface implemented using TensorFlow.

    This implementation uses probabilistic encoders and decoders using Gaussian
    distributions and  realized by multi-layer perceptrons. The VAE can be learned
    end-to-end.

    See "Auto-Encoding Variational Bayes" by Kingma and Welling for more details.
    """
    def __init__(self, network_architecture, transfer_fct=tf.nn.softplus,
                 learning_rate=0.001, batch_size=100):
        self.network_architecture = network_architecture
        self.transfer_fct = transfer_fct
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.step = 0
        self.summary_dir = './summary/'
        self.info = network_architecture['info']

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
        self.z_mean, self.z_log_sigma_sq = \
            self._recognition_network(network_weights["weights_recog"],
                                      network_weights["biases_recog"],
                                      self.x)

        # Draw one sample z from Gaussian distribution
        n_z = self.network_architecture["n_z"]
        eps = tf.random_normal((self.batch_size, n_z), 0, 1,
                               dtype=tf.float32)
        # z = mu + sigma*epsilon
        self.z = tf.add(self.z_mean,
                        tf.mul(tf.sqrt(tf.exp(self.z_log_sigma_sq)), eps),
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
        eps = tf.random_normal((self.batch_size, n_z), 0, 1,
                               dtype=tf.float32)

        self.z_theta = tf.add(0.0, tf.mul(1.0, eps), name='z_theta')

        self.x_prime = self._generator_network(network_weights["weights_gener"],
                                               network_weights["biases_gener"],
                                               z=self.z_theta)

        self.z_prime, self.z_prime_log_sigma_sq = self._recognition_network(
            network_weights["weights_recog"],
            network_weights["biases_recog"],
            self.x_prime)

        dist = Normal(mu=self.z_prime, sigma=tf.sqrt(tf.exp(self.z_prime_log_sigma_sq)))
        logli = tf.reduce_sum(dist.log_pdf(self.z_theta, name='x_entropy'), reduction_indices=1)

        self.cross_entropy = tf.reduce_mean(- logli)

        self.entropy = tf.constant(28.37)

    def _initialize_weights(self, n_hidden_recog_1, n_hidden_recog_2,
                            n_hidden_gener_1, n_hidden_gener_2,
                            n_input, n_z, info):
        all_weights = dict()
        all_weights['weights_recog'] = {
            'h1': tf.Variable(xavier_init(n_input, n_hidden_recog_1)),
            'h2': tf.Variable(xavier_init(n_hidden_recog_1, n_hidden_recog_2)),
            'out_mean': tf.Variable(xavier_init(n_hidden_recog_2, n_z)),
            'out_log_sigma': tf.Variable(xavier_init(n_hidden_recog_2, n_z))}
        all_weights['biases_recog'] = {
            'b1': tf.Variable(tf.zeros([n_hidden_recog_1], dtype=tf.float32)),
            'b2': tf.Variable(tf.zeros([n_hidden_recog_2], dtype=tf.float32)),
            'out_mean': tf.Variable(tf.zeros([n_z], dtype=tf.float32)),
            'out_log_sigma': tf.Variable(tf.zeros([n_z], dtype=tf.float32))}
        all_weights['weights_gener'] = {
            'h1': tf.Variable(xavier_init(n_z, n_hidden_gener_1)),
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
        layer_1 = self.transfer_fct(tf.add(tf.matmul(x, weights['h1']),
                                           biases['b1']))
        layer_2 = self.transfer_fct(tf.add(tf.matmul(layer_1, weights['h2']),
                                           biases['b2']))

        z_mean = tf.add(tf.matmul(layer_2, weights['out_mean']),
                        biases['out_mean'])
        z_log_sigma_sq = \
            tf.add(tf.matmul(layer_2, weights['out_log_sigma']),
                   biases['out_log_sigma'])
        return (z_mean, z_log_sigma_sq)

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

        latent_loss = -0.5 * tf.reduce_sum(1 + self.z_log_sigma_sq -
                                           tf.square(self.z_mean) -
                                           tf.exp(self.z_log_sigma_sq), 1,
                                           name='latent_loss')

        # 3.) Mutual Information loss
        self.lmbda = tf.constant(1.0)
        self.MI = tf.add(self.entropy, - self.cross_entropy, name='MI_loss')

        if self.info:
            self.cost = tf.add(tf.reduce_mean(reconstr_loss + latent_loss),
                               - tf.mul(self.MI, self.lmbda))   # average over batch
        else:
            #self.cost = tf.reduce_mean(reconstr_loss + latent_loss)
            self.cost = tf.reduce_mean(reconstr_loss - 10. * self.MI)


        # Use ADAM optimizer
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.cost)

        rec_summary = tf.scalar_summary('reconstruction loss', tf.reduce_mean(reconstr_loss))
        latent_summary = tf.scalar_summary('KLD q(z|x) || p(z)', tf.reduce_mean(latent_loss))
        cost_summary = tf.scalar_summary('Cost', self.cost)
        sigma_summary = tf.scalar_summary('Sigma', tf.reduce_mean(tf.sqrt(tf.exp(self.z_prime_log_sigma_sq))))
        mu_summary = tf.scalar_summary('mu', tf.reduce_mean(self.z_prime))

        # q_MI_summary = tf.scalar_summary('q_theta(z|x)', tf.reduce_mean(self.q_z_theta_given_x_prime))
        x_entropy_summary = tf.scalar_summary('H(z|x)', self.cross_entropy)
        MI_summary = tf.scalar_summary('MI', self.MI)

        summaries = [rec_summary, latent_summary, cost_summary, x_entropy_summary, MI_summary, sigma_summary, mu_summary]
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
            self.saver.save(self.sess, 'model')
        self.step += 1

        return cost

    def transform(self, X):
        """Transform data by mapping it into the latent space."""
        # Note: This maps to mean of distribution, we could alternatively
        # sample from Gaussian distribution
        return self.sess.run(self.z_mean, feed_dict={self.x: X})

    def generate(self, z_mu=None):
        """ Generate data by sampling from latent space.

        If z_mu is not None, data for this point in latent space is
        generated. Otherwise, z_mu is drawn from prior in latent
        space.
        """
        if z_mu is None:
            z_mu = np.random.normal(size=self.network_architecture["n_z"])
        # Note: This maps to mean of distribution, we could alternatively
        # sample from Gaussian distribution
        return self.sess.run(self.x_reconstr_mean,
                             feed_dict={self.z: z_mu})

    def reconstruct(self, X):
        """ Use VAE to reconstruct given data. """
        return self.sess.run(self.x_reconstr_mean,
                             feed_dict={self.x: X})


def train(network_architecture, learning_rate=0.001,
          batch_size=100, training_epochs=10, display_step=5):

    vae = VariationalAutoencoder(network_architecture,
                                 learning_rate=learning_rate,
                                 batch_size=batch_size)

    # Training cycle
    for epoch in range(training_epochs):
        avg_cost = 0.
        total_batch = int(n_samples / batch_size)
        # Loop over all batches
        for i in range(total_batch):
            batch_xs, _ = mnist.train.next_batch(batch_size)

            # Fit training using batch data
            if i == total_batch:
                cost = vae.partial_fit(batch_xs, last=True)
            else:
                cost = vae.partial_fit(batch_xs)
            # Compute average loss
            avg_cost += cost / n_samples * batch_size

        # Display logs per epoch step
        if epoch % display_step == 0:
            print "Epoch:", '%04d' % (epoch + 1), "cost=", "{:.9f}".format(avg_cost)

    return vae


def main():


    vae = train(network_architecture, training_epochs=75)




if __name__ == '__main__':
    main()