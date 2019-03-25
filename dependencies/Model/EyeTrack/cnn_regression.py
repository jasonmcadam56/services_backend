import tensorflow
import os
import numpy as np
import json


def shuffle(data):
    """
        Taking the input data and shuffling the dataset around so that it is
        better randomised.
        Args:
            data - The input data that we which to shuffle.
        Returns:
            type - explain
    """
    index = np.arange(data[0].shape[0])
    np.random.shuffle(index)
    for i in range(len(data)):
        data[i] = data[i][index]
    return data


class Cnn_regression(object):

    def __init__(self, v=False, training=True):
        """
           Set up the model with placeholders for data that will be needed in
           the train method later on.

           Args:
                v (bool) : Flag for verbose mode.
        """
        self.verbose = v
        if training:
            # [Data ? , X_image_size, Y_image_size, 3 channels RGB]
            self.right_eye = tensorflow.placeholder(tensorflow.float32,
                                                    [None, 64, 64, 3],
                                                    name='right_eye')
            self.left_eye = tensorflow.placeholder(tensorflow.float32,
                                                   [None, 64, 64, 3],
                                                   name='left_eye')

            self.face = tensorflow.placeholder(tensorflow.float32,
                                               [None, 64, 64, 3],
                                               name='face')

            self.mask = tensorflow.placeholder(tensorflow.float32, [None, 25 * 25],
                                               name='mask')

            self.postion = tensorflow.placeholder(tensorflow.float32, [None, 2],
                                                  name='postion')

            self.keep_prob = 0.8
            self.prediction = self.create_nets()


    def tf_conv2d_helper(self, data, weight, biases, strides=1):
        """
            This method is a wrapper for computing a 2-D convolution and then
            applying ReLU as the activation function  and then returning this
            newly formatted data.

            Args:
                 Data - Input data to use.
                 Weight - A 4-D tensor of shape
                 Biases - A 1-D Tensor with size matching the last dimension
                          of the input data.
                 Strides -  Used to build the 4 int long array.
            Returns:
                (Tensor) That has went through conv2d and relu functions.

            Links:
                https://towardsdatascience.com/activation-functions-neural-networks-1cbd9f8d91d6
        """
        # Must have strides[0] = strides[3] = 1. For the most common case of
        # the same horizontal and vertices strides,
        # strides = [1, stride, stride, 1]. www.tensorflow.org
        data = tensorflow.nn.conv2d(
            data, weight, strides=[1, strides, strides, 1], padding='VALID')

        return tensorflow.nn.relu(tensorflow.nn.bias_add(data, biases))

    def create_nets(self):
        """
            This method will create the tensorflow neural network arch, the convolutional layers are:
            conv1 = 11×11/96 , conv2 = 5×5/256, conv3 = 3×3/384, conv4 = 1×1/64
            and the fully-connexcted layers are sized:
            FC-E1: 128, FC-F1: 128, FC-F2: 64, FC-FG1: 256, FC-FG2: 128, FC1: 128, FC2: 2)

            Possible improvement:
                It has been sugguested that adding a fully-connected layer for Mask and Face could
                have improvements on the error rate of the model this should be consider and tested.

            Return:
                Tensor: Fully-connected layers.
        """
        if self.verbose:
            print('Creating network')
        # Load weights
        with tensorflow.variable_scope('', reuse=tensorflow.AUTO_REUSE):
            fc_eye_weight = tensorflow.get_variable('fc_eye_weight',
                                                    shape=((2 * 2 * 2 * 64), 128),
                                                    initializer=tensorflow.initializers.variance_scaling())
            face_w = tensorflow.get_variable('face_w',
                                             shape=((2 * 2 * 64), 128),
                                             initializer=tensorflow.initializers.variance_scaling())
            mask_w = tensorflow.get_variable('mask_w',
                                             shape=(25 * 25, 256),
                                             initializer=tensorflow.initializers.variance_scaling())
            net_w = tensorflow.get_variable('net_w',
                                            shape=(512, 128),
                                            initializer=tensorflow.initializers.variance_scaling())
            net_w2 = tensorflow.get_variable('net_w2',
                                             shape=(128, 2),
                                             initializer=tensorflow.initializers.variance_scaling())
            mask_w1 = tensorflow.get_variable('full_con_mask',
                                              shape=(25 * 25, 256),
                                              initializer=tensorflow.initializers.variance_scaling())

        # Create the right/left eye and face.
        right_eye = self._conv_nets(self.right_eye, max_pool_ksize=2, max_pool_stride=2)
        left_eye = self._conv_nets(self.left_eye, max_pool_ksize=2, max_pool_stride=2)
        l_face = self._conv_nets(self.face, max_pool_ksize=2, max_pool_stride=2)
        # 2 per conv and then finally * by the last conv out size

        fc_eye_biases = tensorflow.Variable(tensorflow.constant(0.1, shape=[128]))
        # Create the fully-connected layers for the eye for the model
        eye = self._fully_connected(right_eye, left_eye, fc_eye_weight, fc_eye_biases)

        # Face conv5
        face_b = tensorflow.Variable(tensorflow.constant(0.1, shape=[128]))

        l_face = tensorflow.reshape(l_face, [-1, int(np.prod(l_face.get_shape()[1:]))])

        l_face = tensorflow.nn.relu(tensorflow.add(tensorflow.matmul(l_face, face_w), face_b))
        # Mask

        mask_b = tensorflow.Variable(tensorflow.constant(0.1, shape=[256]))

        mask = tensorflow.nn.relu(tensorflow.add(tensorflow.matmul(self.mask, mask_w1), mask_b))
        mask = tensorflow.nn.relu(tensorflow.add(tensorflow.matmul(self.mask, mask_w), mask_b))

        # fully-connected layers (with sizes: FC-E1: 128, FC-F1: 128, FC-F2: 64, FC-FG1: 256, FC-FG2: 128, FC1: 128, FC2: 2).
        # Forward Propagation Eyes + Face
        nets = tensorflow.concat([eye, l_face, mask], 1)

        net_b = tensorflow.Variable(tensorflow.constant(0.1, shape=[128]))
        nets = tensorflow.nn.relu(tensorflow.add(tensorflow.matmul(nets, net_w), net_b))

        net_b = tensorflow.Variable(tensorflow.constant(0.1, shape=[2]))
        return tensorflow.add(tensorflow.matmul(nets, net_w2), net_b)

    def _conv_nets(self, data, max_pool_ksize, max_pool_stride):
        """
            Since all our features shares the same first 4 convolutional layers this method will
            help with building these layers for refernces they are:
            conv1 = 11×11/96 , conv2 = 5×5/256, conv3 = 3×3/384, conv4 = 1×1/64

            This method will also applys a drop out to each layer to help with overfitting.

            Args:
                Data : the data that will be used e.g. left_eye, right_eye, mask
                max_pool_ksize (int) : This will be used to set max_pool ksize.
                max_pool_stride (int) : This will be used to set tf_conv2d_helper strides.

        """
        # shape = eye size, eye size, RGN channels, eye out E1 is a 11 x 11 / 96 kernal.
        # https://www.oreilly.com/ideas/building-deep-learning-neural-networks-using-tensorflow-layers
        # We need to scope so that we can reuse these varibles instead of recreating them.
        # TODO: Try replacing xavier with tf.keras.initializers.VarianceScaling since we are using relu activation functions.
        with tensorflow.variable_scope('', reuse=tensorflow.AUTO_REUSE):
            conv1_weight = tensorflow.get_variable('conv1_eye_weight', shape=(11, 11, 3, 96),
                                                   initializer=tensorflow.initializers.variance_scaling())
            conv2_weight = tensorflow.get_variable('conv2_eye_weight', shape=(5, 5, 96, 256),
                                                   initializer=tensorflow.initializers.variance_scaling())
            conv3_weight = tensorflow.get_variable('conv3_eye_weight', shape=(3, 3, 256, 384),
                                                   initializer=tensorflow.initializers.variance_scaling())
            conv4_weight = tensorflow.get_variable('conv4_eye_weight', shape=(1, 1, 384, 64),
                                                   initializer=tensorflow.initializers.variance_scaling())

        conv1_biases = tensorflow.Variable(tensorflow.constant(0.1, shape=[96]))
        conv2_biases = tensorflow.Variable(tensorflow.constant(0.1, shape=[256]))
        conv3_biases = tensorflow.Variable(tensorflow.constant(0.1, shape=[384]))
        conv4_biases = tensorflow.Variable(tensorflow.constant(0.1, shape=[64]))

        # https://www.tensorflow.org/tutorials/estimators/cnn show using max_pool with conv2d
        # Conv1
        l_data = self.tf_conv2d_helper(data, conv1_weight, conv1_biases, )
        l_data = tensorflow.nn.dropout(l_data, self.keep_prob)
        l_data = tensorflow.nn.max_pool(l_data, ksize=[1, max_pool_ksize, max_pool_ksize, 1],
                                        strides=[1, max_pool_stride, max_pool_stride, 1], padding='VALID')
        # Conv2
        l_data = self.tf_conv2d_helper(l_data, conv2_weight, conv2_biases)
        l_data = tensorflow.nn.dropout(l_data, self.keep_prob)
        l_data = tensorflow.nn.max_pool(l_data, ksize=[1, max_pool_ksize, max_pool_ksize, 1],
                                        strides=[1, max_pool_stride, max_pool_stride, 1], padding='VALID')
        # Conv3
        l_data = self.tf_conv2d_helper(l_data, conv3_weight, conv3_biases)
        l_data = tensorflow.nn.dropout(l_data, self.keep_prob)
        l_data = tensorflow.nn.max_pool(l_data, ksize=[1, max_pool_ksize, max_pool_ksize, 1],
                                        strides=[1, max_pool_stride, max_pool_stride, 1], padding='VALID')
        # Conv4
        l_data = self.tf_conv2d_helper(l_data, conv4_weight, conv4_biases)
        l_data = tensorflow.nn.dropout(l_data, self.keep_prob)
        l_data = tensorflow.nn.max_pool(l_data, ksize=[1, max_pool_ksize, max_pool_ksize, 1],
                                        strides=[1, max_pool_stride, max_pool_stride, 1], padding='VALID')
        return l_data

    def _fully_connected(self, right_data, left_data, weight, biases):
        """ This method will try to concat two datasets together to make a fully-connected layers along with
         running a relu function on it to format the output data.

         Args:
             Right_data - The right Tensor that will be used to connect.
             Left_data  - The left tensor that will be used to connect.
             Weight     - A 4-D tensor of shape.
             Biases     - A 1-D Tensor with size matching the last dimension of the input data.

         Returns
             (Tensor) - That has been concated together and ran through a activation function (Relu).

         Link: https://www.oreilly.com/library/view/tensorflow-for-deep/9781491980446/ch04.html
        """
        right_data = tensorflow.reshape(right_data, [-1, int(np.prod(right_data.get_shape()[1:]))])
        left_data = tensorflow.reshape(left_data, [-1, int(np.prod(left_data.get_shape()[1:]))])
        fc_data = tensorflow.concat([right_data, left_data], 1)
        fc_data = tensorflow.nn.relu(tensorflow.add(tensorflow.matmul(fc_data, weight), biases))
        return fc_data

    def _next_batch(self, data, size):
        """
            We need to be able to proccess the data however creating a new list is very compute heavy
            and this method is called for every loop of both training and valation so we need to reduce
            compute time. We return a genator here instead so it can be process when needed.
            Args:
                data (numpy): Data to process.
                size (int): The chuck size to yield.

            Returns:
                generator: to be used for processing

        """
        for i in np.arange(0, data[0].shape[0], size):
            # Explained: we are aranging the data into chunks then taking
            # current chunk to next chunk by using list comprehension and spliting.
            yield [d[i: i + size] for d in data]

    def train(self, training, valation, path='', epoch=1000):
        """
            Taking a training data and valation data perform the correct amount of steps,

            Args:
                 training (numpy.Array) : Used to replace placeholder data in the session for training.
                 valation (numpy.Array) : Used to replace placeholder data in the session for valation.
                 path (string) : Location of to save model.

            Raises:

                RuntimeError: If this Session is in an invalid state (e.g. has been closed).
                TypeError   : If feed_dict keys are of an inappropriate type.
                ValueError  : If feed_dict keys are invalid or refer to a Tensor that doesn't exist.
        """
        if self.verbose:
            print('We are training now!')
        if not path:
            path = os.path.dirname(os.path.realpath(__file__))  # Get the current path.
            if self.verbose:
                print('Save path is {}'.format(path))
        model_save_loc = '{}/eye_q/eyeq_model'.format(path)
        self.saver = tensorflow.train.Saver(max_to_keep=1)

        self.mase = tensorflow.losses.mean_squared_error(self.postion, self.prediction)
        self.optimizer = tensorflow.train.AdamOptimizer(learning_rate=1e-4, beta1=0.9, beta2=0.999, epsilon=1e-08).minimize(self.mase)
        self.err = tensorflow.reduce_mean(
            tensorflow.sqrt(
                tensorflow.reduce_sum(
                    tensorflow.squared_difference(self.prediction, self.postion), axis=1)))
        tensorflow.summary.scalar('Error', self.err)
        tensorflow.summary.scalar('Mean Squared Error', self.mase)
        merged = tensorflow.summary.merge_all()
        # if not os.path.exists(model_save_loc):
        #     os.makedirs(model_save_loc)

        best_loss = float('Inf')  # Will not know best lost until we calc it after the first run of the network.

        if self.verbose:
            print('Training set {}, validation set {}'.format(training[0].shape, valation[0].shape))

        tensorflow.get_collection("v_nodes")

        for item in [self.right_eye, self.left_eye, self.face, self.mask, self.prediction]:
            tensorflow.add_to_collection("v_nodes", item)

        model_init = tensorflow.global_variables_initializer()

        with tensorflow.Session() as session:
            session.run(model_init)
            train_file_writer = tensorflow.summary.FileWriter(model_save_loc + '/logs/train', session.graph)
            val_file_writer = tensorflow.summary.FileWriter(model_save_loc + '/logs/test', session.graph)

            batches = 256  # Becareful depending on your machine higher level of batches will core dump.
            for epoch_number in range(1, epoch):
                out_t_merg_err = bytes()
                training = shuffle(training)  # Need to shuffle outside of _next_batch or it becomes very memory costly.
                for batch_training in self._next_batch(training, batches):
                    l_merg, l_err, l_mass = self._training(session, batch_training, [merged, self.err, self.mase], True)
                    out_t_merg_err += l_merg

                # End of batch training
                loss_mase = 0
                out_v_merge = bytes()
                out_merg = bytes()
                for batched_valation in self._next_batch(valation, batches):
                    l_merg, out_post, out_realp, mase, = self._training(session, batched_valation, [merged, self.postion, self.prediction, self.mase])
                    loss_mase += mase
                    if self.verbose:
                        correct_preds = tensorflow.equal(tensorflow.argmax(out_realp, 1), tensorflow.argmax(out_post, 1))
                        accuracy = tensorflow.reduce_sum(tensorflow.cast(correct_preds, tensorflow.float32))
                        if epoch_number == 1:
                            tensorflow.summary.scalar('accuracy', accuracy)
                            merged_v = tensorflow.summary.merge_all()
                        out_v_merge += session.run(merged_v, feed_dict=self._create_feed_dict(batched_valation))
                        out_merg += l_merg
                # Reduce recording if no progress is made so it will scale down to every 2 epochs (handy for larger runs)
                if (epoch_number % 2 == 0) or (loss_mase < best_loss):
                    val_file_writer.add_summary(out_merg, epoch_number)
                    train_file_writer.add_summary(out_t_merg_err, epoch_number)
                    if self.verbose:
                        val_file_writer.add_summary(out_v_merge, epoch_number)

                if loss_mase < best_loss:
                    if self.verbose:
                        print('New best loss {} old one was {}, saving new model checkpoint.'.format(loss_mase, best_loss))
                        # print('Mase loss for this model {}'.format(loss_mase))
                    self.saver.save(session, model_save_loc, global_step=epoch_number)
                    best_loss = loss_mase

                if self.verbose:
                    print('epoch {} done'.format(epoch_number))
            print('Model run done with {} epochs'.format(epoch))
            tensorflow.saved_model.simple_save(session, model_save_loc, inputs={"left_eye": self.left_eye, "right_eye": self.right_eye, "mask": self.mask, "face": self.face}, outputs={"postion": self.postion})

    def _create_feed_dict(self, data, val=False):
        """
            Creates the feed_dict in the right format of the network

            Args:
                Data (numpy.Array): Preloaded data.

            Returns:
                Dict: Formatted to networks arch.

            Raises:
                ValueError  : If data is missing its data.
        """
        return {self.left_eye: data[0],
                self.right_eye: data[1],
                self.face: data[2],
                self.mask: data[3],
                self.postion: data[4]
                }

    def _training(self, session, data, fetches, training_flag=False):
        """
            Taking a session and numpy data perform a training step, when training flag is truew a
            optimiation is used reduce drop out this is perfibily only done during the training and
            not vailation.

            Args:
                 Session (tf.Session): Has setup with the correct encapsulation enviroement
                                       to run our Tensor objects.
                 Data (numpy.Array)  : Used to replace placeholder data in the session.
                 Fetches (List)      : List of graph elements to feed session.
                 training_flag (bool): if we want to run the optimizer or not.

            Return:
                 Dict: Dict of data that the feed_dict was fed.

            Raises:

                RuntimeError: If this Session is in an invalid state (e.g. has been closed).
                TypeError   : If feed_dict keys are of an inappropriate type.
                ValueError  : If feed_dict keys are invalid or refer to a Tensor that doesn't exist.

        """
        feed_dict_batch = self._create_feed_dict(data)
        if training_flag:
            session.run(self.optimizer, feed_dict=feed_dict_batch)
        return session.run(fetches, feed_dict=feed_dict_batch)

    def testing(self, file_path_mod, data, batches=254):
        """
            Taking a file path to a model location and data location, load that model
            then run the new data against it.

            Verbose mode:
                When this flag is active, the system will record the accuracy of the model along with this
                print out the Predictions vs Real (this should only be done with a limited size of data)
            Args:
                file_path_mod (str): File path to the model.
                data (Tensor): Data to feed the nextwork.
        """
        json_file = os.path.dirname(os.path.realpath(__file__)) + '.\\test_values.json'
        with tensorflow.Session() as sess:
            meta_file = file_path_mod + ".meta"
            if not os.path.exists(meta_file):
                raise ValueError('No metadata file loaded')

            saver = tensorflow.train.import_meta_graph(meta_file)
            saver.restore(sess, os.path.join("./", file_path_mod))
            nodes = tensorflow.get_collection_ref("v_nodes")

            if len(nodes) != 5:
                raise ValueError('Wrong model type was looking for {} nodes got {}'.format('5', len(nodes)))

            self.left_eye, self.right_eye, self.face, self.mask, self.prediction = nodes

            self.postion = tensorflow.placeholder(tensorflow.float32, [None, 2], name='postion')
            self.err = tensorflow.reduce_mean(tensorflow.sqrt(tensorflow.reduce_sum(tensorflow.squared_difference(self.prediction, self.postion), axis=1)))
            self.mase = tensorflow.losses.mean_squared_error(self.prediction, self.postion)
            t_err, t_mase = (0, 0)
            total_correct_preds = 0
            num_of_batches = data[0].shape[0] / batches + (data[0].shape[0] % batches != 0)
            for batch_data in self._next_batch(data, batches):
                b_err, b_mase, l_pred, l_realp = sess.run([self.err, self.mase, self.prediction, self.postion], feed_dict=self._create_feed_dict(batch_data))
                t_err += b_err / num_of_batches
                t_mase += b_mase
                total_correct_preds += self._record_accuracy(l_pred, l_realp, sess)
            jdict = {'pred': {'real': l_realp.tolist(), 'model': l_pred.tolist()},
                     'stats': {'accuracy': total_correct_preds, 'error': t_err, 'mase': t_mase}}
            json.dump(jdict, open(json_file, 'w'))
            if self.verbose:
                print('Total accuracy {}%'.format(total_correct_preds))

            return (t_err, t_mase)

    def _record_accuracy(self, pred, post, sess):
        """
        This method will record the networks accuracy based on pred (network output), and post (label corrected)
        tensor's.
            Args:
                pred (numpy.ndarray): The acutal network predctions.
                post (numpy.ndarray): The real postions.
                sess (Session): A session of tensorflow to run with.

            Returns:
                numpy.float32: accuracy of the network.
        """
        correct_preds = tensorflow.equal(tensorflow.argmax(pred, 1), tensorflow.argmax(post, 1))
        accuracy = tensorflow.reduce_sum(tensorflow.cast(correct_preds, tensorflow.float32))
        return sess.run(accuracy)
