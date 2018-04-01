from Models.TimitReader import *

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import tensorflow as tf

cur_time = time.time()

model_save_file = 'speaker_verification_models'
if not os.path.exists(model_save_file):
    os.mkdir(model_save_file)
model_save_path = os.path.join(model_save_file, 'conv2d_triplet_01.ckpt')

DR = Data_Read_Timit('/data/audio/timit/train')
print("***************************************************************")
DR.ReadDataAndConvertIntoSpectrogram()
print("***************************************************************")
print(time.time() - cur_time)
cur_time = time.time()
print("***************************************************************")

DR_test = Data_Read_Timit('/data/audio/timit/test')
print("***************************************************************")
DR_test.ReadDataAndConvertIntoSpectrogram()
print("***************************************************************")
print(time.time() - cur_time)
cur_time = time.time()
print("***************************************************************")

np.random.seed(int(time.time()))

sample_rate = 16000
frame_length = 511
frame_step = 122
log_offset = 1e-12

W = 127
H = 256
C = 1

def leaky_relu(x, alpha=0.01):
    return tf.maximum(x, alpha*x)

def Conv2dLayer(number, input_data, filters, kernel_size, reuse=False):
    with tf.variable_scope("conv2d_layer_{}".format(number), reuse=reuse):
        conv = tf.layers.conv2d(input_data, filters=filters, kernel_size=kernel_size, padding='same')
        lrl = leaky_relu(conv)
        pool = tf.layers.max_pooling2d(lrl, 2, 2, padding='same')
        return pool

class SeparateAnchorModel(object):

    def __init__(self, alpha=0.5, cls_lr=0.001, tri_lr=0.001, pool=tf.layers.max_pooling2d):
        tf.reset_default_graph()
        self.sess = None
        self.alpha = alpha  # abs(ap-an) > alpha
        self.train = tf.placeholder(tf.bool, shape=[])
        self.batch_size = DR.batch_size
        self.once_labels = DR.labels_oncebatch  # how many inset ones shall be select in one batch
        self.cut_of_utter = DR.cut_of_utter  # how many cuts are one utter having
        self.utter_per_label_tri = DR.utter_once_batch  # how many utters shall be chosen from one's own utters, the one been chosen in one batch
        self.num_inset = DR.InSet_Num  # how many people is in in-set group

        self.pool = pool
        self.cls_lr = cls_lr
        self.tri_lr = tri_lr
        # part of placeholder
        # anchor_placeholder, positive_p,negative_t, negative_l
        self.anchor_batch_placeholders = tf.placeholder(tf.float32, [None, cut_of_each_utter, 16000])
        self.positive_batch_placeholders = tf.placeholder(tf.float32, [None, cut_of_each_utter, 16000])
        self.negative_batch_placeholders = tf.placeholder(tf.float32, [None, cut_of_each_utter, 16000])
        self.negative_batch_labels = tf.placeholder(tf.int32, [None])

        _, self.anchor_vectors = self.VectorGet(self.anchor_batch_placeholders, False)
        _, self.positive_vectors = self.VectorGet(self.positive_batch_placeholders, True)
        self.cls_fv, self.negative_vectors = self.VectorGet(self.negative_batch_placeholders, True)

        # part of softmax classification which is used to pre_train
        self.cls_vector = tf.layers.dense(self.cls_fv, self.num_inset)
        self.cls_ce_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=self.cls_vector, labels=tf.cast(self.negative_batch_labels, tf.int64)
        ))
        self.cls_opt = tf.train.AdamOptimizer(self.cls_lr).minimize(self.cls_ce_loss)
        self.cls_pred = tf.argmax(self.cls_vector, axis=1)

        self.triplet_loss = None
        self.triplet_optimizer = None
        self.TripletLossMaker()

        self.EER = 1

        self.saver = tf.train.Saver()

    def SpectrogramCompute(self, input_layer):
        wave_signal = tf.squeeze(input_layer, axis=1)
        stfts = tf.contrib.signal.stft(wave_signal, frame_length=frame_length,
                                       frame_step=frame_step, fft_length=frame_length)
        magnitude_spectrograms = tf.abs(stfts)
        log_magnitude_spectrogram = tf.log(magnitude_spectrograms + log_offset)
        this_mean, this_std = tf.nn.moments(log_magnitude_spectrogram, axes=[1, 2], keep_dims=True)
        ret = (log_magnitude_spectrogram - this_mean) / this_std
        # bs 127 128
        return ret

    def VectorGet(self, input_layer, reuse=False, fv_mode=None):
        batches = tf.split(input_layer, axis=1, num_or_size_splits=cut_of_each_utter)
        feature_vectors = []
        if not reuse:
            feature_vectors.append(self.Conv2dModel(self.SpectrogramCompute(batches[0])))
        else:
            feature_vectors.append(self.Conv2dModel(self.SpectrogramCompute(batches[0]), reuse=True))
        for i in range(1, cut_of_each_utter):
            feature_vectors.append(self.Conv2dModel(self.SpectrogramCompute(batches[i]), reuse=True))
        # first, the fv mode is None, use the most simple way
        fv = tf.reduce_mean(feature_vectors, axis=0)
        compare_vector = self.LinearProjection(fv, reuse)
        return fv, compare_vector

    def LinearProjection(self, input, reuse=False):
        with tf.variable_scope('linear_projection', reuse=reuse):
            lp = tf.layers.dense(input, 128)
            return lp / tf.sqrt(tf.reduce_sum(tf.square(input), axis=1, keep_dims=True))
    '''
    def Conv2dModel(self, input, reuse=False):
        with tf.variable_scope('conv2d_model', reuse=reuse):
            input_layer = tf.reshape(input, shape=(-1, W, H, C))
            conv1 = Conv2dLayer(1, input_layer, 16, 7, reuse)
            # bs 64 64 8
            conv2 = Conv2dLayer(2, conv1, 32, 5, reuse)
            # bs 32 32 16
            conv3 = Conv2dLayer(3, conv2, 64, 5, reuse)
            # bs 16 16 32
            conv4 = Conv2dLayer(4, conv3, 128, 3, reuse)
            # bs 8 8 64
            conv5 = Conv2dLayer(5, conv4, 256, 3, reuse)
            # bs 4 4 128
            conv6 = Conv2dLayer(6, conv5, 512, 3, reuse)
            # bs 2 2 256
            conv7 = Conv2dLayer(7, conv6, 1024, 3, reuse)
            # bs 1 1 512
            feature_vector = tf.layers.batch_normalization(tf.layers.flatten(conv7), training=self.train)
            return feature_vector
    '''
    def Conv2dModel(self, input, reuse=False):
        with tf.variable_scope('conv2d_model', reuse=reuse):
            input_layer = tf.reshape(input, shape=(-1, W, H, C))
            conv1 = tf.layers.conv2d(input_layer, 16, 5, strides=1, padding='same')
            pool1 = self.pool(conv1, pool_size=2, strides=2, padding='same')
            relu1 = tf.nn.relu(pool1)  # 64 128 16
            conv2 = tf.layers.conv2d(relu1, 32, 5, strides=1, padding='same')
            pool2 = self.pool(conv2, pool_size=2, strides=2, padding='same')
            relu2 = tf.nn.relu(pool2)  # 32 64 32
            conv3 = tf.layers.conv2d(relu2, 64, 3, strides=1, padding='same')
            pool3 = self.pool(conv3, pool_size=2, strides=2, padding='same')
            relu3 = tf.nn.relu(pool3)  # 16 32 64
            conv4 = tf.layers.conv2d(relu3, 128, 3, strides=1, padding='same')
            pool4 = self.pool(conv4, pool_size=2, strides=2, padding='same')
            relu4 = tf.nn.relu(pool4)  # 8 16 128
            conv5 = tf.layers.conv2d(relu4, 256, 3, strides=1, padding='same')
            pool5 = self.pool(conv5, pool_size=2, strides=2, padding='same')
            relu5 = tf.nn.relu(pool5)  # 4 8 256
            conv6 = tf.layers.conv2d(relu5, 512, 3, strides=1, padding='same')
            pool6 = self.pool(conv6, pool_size=2, strides=2, padding='same')
            relu6 = tf.nn.relu(pool6)  # 2 4 512
            conv7 = tf.layers.conv2d(relu6, 1024, 3, strides=1, padding='same')
            pool7 = self.pool(conv7, pool_size=2, strides=2, padding='same')
            relu7 = tf.nn.relu(pool7)  # 1 2 1024
            conv8 = tf.layers.conv2d(relu7, 1024, 3, strides=1, padding='same')
            pool8 = self.pool(conv8, pool_size=[1, 2], strides=[1, 2], padding='same')
            relu8 = tf.nn.relu(pool8)  # 1 1 1024
            output_layer = tf.layers.batch_normalization(tf.reshape(relu8, (-1, 1024)), training=self.train)
            return output_layer

    #
    def Model_Init(self):
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=config)
        tf.global_variables_initializer().run(session=self.sess)
        print("******************* model initialized **********************")

    def Model_Close(self):
        self.sess.close()
        print(" This model has been shut down ")

    def Model_Save(self):
        print('save model at ', model_save_path)
        self.saver.save(self.sess, model_save_path)

    def Model_Load(self):
        print("load this model ", model_save_path)
        self.saver.restore(self.sess, model_save_path)

    def TripletLossMaker(self):
        with tf.variable_scope("triplet_loss"):
            sap = tf.square(self.anchor_vectors - self.positive_vectors)
            san = tf.square(self.anchor_vectors - self.negative_vectors)
            ap = tf.reduce_mean(tf.reduce_sum(sap, axis=1))
            an = tf.reduce_mean(tf.reduce_sum(san, axis=1))
            #tripletloss = tf.reduce_mean(tf.nn.relu(ap + self.alpha - an)) + self.alpha * tf.reduce_mean(tf.nn.relu(ap))
            tripletloss = tf.reduce_mean(tf.nn.relu(ap + self.alpha - an)) + self.alpha * tf.reduce_mean(ap) + tf.reduce_mean(tf.nn.relu(ap-self.alpha))
            #tripletloss = tf.reduce_mean(tf.nn.relu(ap + self.alpha - an)) + self.alpha * tf.reduce_mean(
            #    ap) + tf.reduce_mean(tf.nn.relu(ap - self.alpha)) + tf.reduce_mean(tf.nn.relu(self.alpha - an))
            self.triplet_loss = tripletloss
            self.triplet_optimizer = tf.train.AdamOptimizer(self.tri_lr).minimize(self.triplet_loss)

    def TripletLossTrain(self, epoches=31):
        for i in range(epoches):
            losses_c = []
            losses_t = []
            for j in range(DR.InSet_Num):
                tmp_a, tmp_p, tmp_n, apl, nl = DR.TripletDataGeter(j)
                feeddict = {
                    self.anchor_batch_placeholders : tmp_a,
                    self.positive_batch_placeholders : tmp_p,
                    self.negative_batch_placeholders : tmp_n,
                    self.negative_batch_labels : nl,
                    self.train : True
                }
                _, cl, _, tl = self.sess.run(
                    [self.cls_opt, self.cls_ce_loss, self.triplet_optimizer, self.triplet_loss], feed_dict=feeddict)
                losses_c.append(cl)
                losses_t.append(tl)

                if (j + i) % 100 == 0:
                    print(i, j)
            print(i, np.mean(losses_c), np.mean(losses_t))
            print("**************************************************")
            print("between train data")
            self.EERAndSave(DR.InSet_Train_Data, DR.InSet_Train_Label)
            print("**************************************************")
            print("between test data")
            self.EERAndSave(DR_test.InSet_Train_Data, DR_test.InSet_Train_Label, save_model=True)
            print("**************************************************")
        print(" now test the best model ")
        self.ModelTest(DR_test.InSet_Train_Data, DR_test.InSet_Train_Label, True)

    '''
    def EER_Compute_2(self, l2same, l2diff):
        print("dist same", np.mean(l2same), np.std(l2same), 'dist diff', np.mean(l2diff), np.std(l2diff))
        base = 0
        sl = len(l2same)
        dl = len(l2diff)
        min_rd = 1
        min_id = 0
        for i in range(71):
            tmpt = base + i * 0.01
            fault_reject = np.sum(l2same > tmpt) / sl
            fault_accept = np.sum(l2diff <= tmpt) / dl
            rd = np.abs(fault_accept - fault_reject)
            if rd < min_rd:
                min_rd = rd
                min_id = i
        eer_t = base + min_id * 0.01
        eer_fr = np.sum(l2same > eer_t) / sl
        eer_fa = np.sum(l2diff <= eer_t) / dl
        return eer_t, (eer_fa + eer_fr) / 2
    '''

    def EER_Compute_2(self, l2same, l2diff):
        mean_same = np.mean(l2same)
        std_same = np.std(l2same)
        mean_diff = np.mean(l2diff)
        std_diff = np.std(l2diff)
        print("dist same", mean_same, std_same, 'dist diff', mean_diff, std_diff)
        sl = len(l2same)
        dl = len(l2diff)
        divide = 1
        threshold = mean_same
        last_threshold = threshold
        step_adjust = std_same
        last_eer = 0.5
        best_threshold = 0
        compute_count = 0
        while divide <= 32 and compute_count<10:
            fault_accept = np.sum(l2same > threshold) / sl
            fault_reject = np.sum(l2diff <= threshold) / dl
            cur_eer = np.abs(fault_accept - fault_reject)
            if cur_eer < last_eer:
                last_eer = cur_eer
                best_threshold = threshold
                last_threshold = threshold
                threshold += (step_adjust/divide)
                compute_count += 1
            else:
                threshold = last_threshold
                divide *= 2
                if divide > 32:
                    compute_count += 1
                    step_adjust *= -1
                    divide = 1

        eer_t = best_threshold
        eer_fr = np.sum(l2same > eer_t) / sl
        eer_fa = np.sum(l2diff <= eer_t) / dl
        return eer_t, (eer_fa + eer_fr)/2 

    # divide scores into same pair and diff pair
    def ScoreDivide(self, score, label):
        dist_same = []
        dist_diff = None
        for b in set(label):
            cur_p = score[label == b]
            cur_n = score[label != b]
            lop = len(cur_p)
            # compute the distance between same
            for i in range(lop):
                for j in range(i + 1, lop):
                    dist = np.sum(np.square(cur_p[i] - cur_p[j]))
                    dist_same.append(dist)
            for i in range(lop):
                now_p = cur_p[i]
                dists = np.sum(np.square(now_p - cur_n), axis=1)
                if dist_diff is None:
                    dist_diff = dists
                else:
                    dist_diff = np.concatenate((dist_diff, dists), axis=0)
        dist_same = np.array(dist_same)
        return dist_same, dist_diff

    def FeatureVectorsGeter(self, data, label):
        dl = len(label)
        rounds = dl // self.batch_size
        remain = dl % self.batch_size
        features = None
        for i in range(rounds):
            batch_data, batch_label = DR.DataBatch(data, label, i)
            fvs = self.sess.run(self.negative_vectors, feed_dict={
                self.negative_batch_placeholders : batch_data,
                self.train : False
            })
            if features is None:
                features = fvs
            else:
                features = np.concatenate((features, fvs), axis=0)
        if remain > 0:
            batch_data, batch_label = DR.DataBatch(data, label, rounds, remain)
            fvs = self.sess.run(self.negative_vectors, feed_dict={
                self.negative_batch_placeholders : batch_data,
                self.train : False
            })
            if features is None:
                features = fvs
            else:
                features = np.concatenate((features, fvs), axis=0)
        return features

    def EERAndSave(self, data, label, save_model=False):
        fv = self.FeatureVectorsGeter(data, label)
        dist_same, dist_diff = self.ScoreDivide(fv, label)
        threshold, eer = self.EER_Compute_2(dist_same, dist_diff)
        print("current equal error rate is", eer, ' the pretaining threshold is', threshold)
        right_same = np.sum(dist_same <= threshold)
        right_diff = np.sum(dist_diff > threshold)
        print("right accept:{} in {}, percent:{}".format(right_same, len(dist_same), right_same/len(dist_same)), " ; ", "right reject:{} in {}, percent:{}".format(right_diff, len(dist_diff), right_diff/len(dist_diff)))
        if save_model:
            if eer < self.EER:
                self.EER = eer
                self.Model_Save()
            else:
                print("no promotion so no save")

    def ModelTest(self, testdata, testlabel, loadModel=False):
        if loadModel:
            self.Model_Load()
        testscore = self.FeatureVectorsGeter(testdata, testlabel)
        dist_same, dist_diff = self.ScoreDivide(testscore, testlabel)
        threshold, eer = self.EER_Compute_2(dist_same, dist_diff)
        print("current equal error rate is", eer, ' the pretaining threshold is', threshold)
        right_same = np.sum(dist_same <= threshold)
        right_diff = np.sum(dist_diff > threshold)
        print("right accept:{} in {}, percent:{}".format(right_same, len(dist_same), right_same/len(dist_same)), " ; ", "right reject:{} in {}, percent:{}".format(right_diff, len(dist_diff), right_diff/len(dist_diff)))

