from __future__ import print_function
import nd_aggregation
import mxnet as mx
from mxnet import nd, autograd, gluon
from mxnet.gluon.model_zoo import vision as models
import numpy as np
import random
import argparse
import byzantine
import sys
import os
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import scipy
from collections import namedtuple

os.environ["MXNET_CUDNN_AUTOTUNE_DEFAULT"] = "0"
np.warnings.filterwarnings('ignore')


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", help="dataset", default='cifar10', type=str)
    parser.add_argument("--bias", help="degree of non-IID to assign data to workers", type=float, default=0.1)
    parser.add_argument("--net", help="net", default='resnet', type=str, choices=['mlr', 'cnn', 'fcnn', 'resnet'])
    parser.add_argument("--batch_size", help="batch size", default=32, type=int)
    parser.add_argument("--lr", help="learning rate", default=0.001, type=float)
    parser.add_argument("--nworkers", help="# workers", default=100, type=int)
    parser.add_argument("--nepochs", help="# epochs", default=2000, type=int)
    parser.add_argument("--gpu", help="index of gpu", default=0, type=int)
    parser.add_argument("--seed", help="seed", default=42, type=int)
    parser.add_argument("--nbyz", help="# byzantines", default=28, type=int)
    parser.add_argument("--byz_type", help="type of attack", default='scaling_attack', type=str,
                        choices=['no', 'partial_trim', 'full_trim', 'mean_attack', 'full_mean_attack', 'gaussian',
                                 'dir_partial_krum_lambda', 'dir_full_krum_lambda', 'label_flip', 'backdoor', 'dba',
                                 'edge', 'scaling_attack'])
    parser.add_argument("--aggregation", help="aggregation rule", default='median', type=str,
                        choices=['simple_mean', 'trim', 'krum', 'median'])
    return parser.parse_args()


class VAE(gluon.HybridBlock):
    def __init__(self, ctx):
        super(VAE, self).__init__()
        self.ctx = ctx
        with self.name_scope():
            self.fc1 = gluon.nn.Dense(500)
            self.fc21 = gluon.nn.Dense(20)
            self.fc22 = gluon.nn.Dense(20)
            self.fc3 = gluon.nn.Dense(500)
            self.fc4 = gluon.nn.Dense(640)

    def encode(self, x):
        h1 = nd.Activation(self.fc1(x), 'relu')
        return self.fc21(h1), nd.Activation(self.fc22(h1), 'softrelu')

    def reparametrize(self, mu, logvar):
        '''
        mu is a number and logvar is a ndarray
        '''
        std = nd.exp(0.5 * logvar)
        eps = nd.random_normal(
            loc=0, scale=1, shape=std.shape).as_in_context(self.ctx)
        return mu + eps * std

    def decode(self, z):
        h3 = nd.Activation(self.fc3(z), 'relu')
        return nd.Activation(self.fc4(h3), 'sigmoid')

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparametrize(mu, logvar)
        return self.decode(z), mu, logvar


def lbfgs(args, S_k_list, Y_k_list, v):
    curr_S_k = nd.concat(*S_k_list, dim=1)
    curr_Y_k = nd.concat(*Y_k_list, dim=1)
    S_k_time_Y_k = nd.dot(curr_S_k.T, curr_Y_k)
    S_k_time_S_k = nd.dot(curr_S_k.T, curr_S_k)
    R_k = np.triu(S_k_time_Y_k.asnumpy())
    L_k = S_k_time_Y_k - nd.array(R_k, ctx=mx.gpu(args.gpu))
    sigma_k = nd.dot(Y_k_list[-1].T, S_k_list[-1]) / (nd.dot(S_k_list[-1].T, S_k_list[-1]))
    D_k_diag = nd.diag(S_k_time_Y_k)
    upper_mat = nd.concat(*[sigma_k * S_k_time_S_k, L_k], dim=1)
    lower_mat = nd.concat(*[L_k.T, -nd.diag(D_k_diag)], dim=1)
    mat = nd.concat(*[upper_mat, lower_mat], dim=0)
    mat_inv = nd.linalg.inverse(mat)

    approx_prod = sigma_k * v
    p_mat = nd.concat(*[nd.dot(curr_S_k.T, sigma_k * v), nd.dot(curr_Y_k.T, v)], dim=0)
    approx_prod -= nd.dot(nd.dot(nd.concat(*[sigma_k * curr_S_k, curr_Y_k], dim=1), mat_inv), p_mat)

    return approx_prod


def params_convert(net):
    tmp = []
    for param in net.collect_params().values():
        if param.grad_req == 'null':
            continue
        tmp.append(param.data().copy())
    params = nd.concat(*[x.reshape((-1, 1)) for x in tmp], dim=0)
    return params


def clip(a, b, c):
    tmp = nd.minimum(nd.maximum(a, b), c)
    return tmp


def detection(score, nobyz):
    estimator = KMeans(n_clusters=2)
    estimator.fit(score.reshape(-1, 1))
    label_pred = estimator.labels_
    if np.mean(score[label_pred==0])<np.mean(score[label_pred==1]):
        #0 is the label of malicious clients
        label_pred = 1 - label_pred
    real_label=np.ones(100)
    real_label[:nobyz]=0
    acc=len(label_pred[label_pred==real_label])/100
    recall=1-np.sum(label_pred[:nobyz])/nobyz
    fpr=1-np.sum(label_pred[nobyz:])/(100-nobyz)
    fnr=np.sum(label_pred[:nobyz])/nobyz
    print("acc %0.4f; recall %0.4f; fpr %0.4f; fnr %0.4f;" % (acc, recall, fpr, fnr))
    print(silhouette_score(score.reshape(-1, 1), label_pred))

def detection1(score, nobyz):
    nrefs = 10
    ks = range(1, 8)
    gaps = np.zeros(len(ks))
    gapDiff = np.zeros(len(ks) - 1)
    sdk = np.zeros(len(ks))
    min = np.min(score)
    max = np.max(score)
    score = (score - min)/(max-min)
    for i, k in enumerate(ks):
        estimator = KMeans(n_clusters=k)
        estimator.fit(score.reshape(-1, 1))
        label_pred = estimator.labels_
        center = estimator.cluster_centers_
        Wk = np.sum([np.square(score[m]-center[label_pred[m]]) for m in range(len(score))])
        WkRef = np.zeros(nrefs)
        for j in range(nrefs):
            rand = np.random.uniform(0, 1, len(score))
            estimator = KMeans(n_clusters=k)
            estimator.fit(rand.reshape(-1, 1))
            label_pred = estimator.labels_
            center = estimator.cluster_centers_
            WkRef[j] = np.sum([np.square(rand[m]-center[label_pred[m]]) for m in range(len(rand))])
        gaps[i] = np.log(np.mean(WkRef)) - np.log(Wk)
        sdk[i] = np.sqrt((1.0 + nrefs) / nrefs) * np.std(np.log(WkRef))

        if i > 0:
            gapDiff[i - 1] = gaps[i - 1] - gaps[i] + sdk[i]
    #print(gapDiff)
    for i in range(len(gapDiff)):
        if gapDiff[i] >= 0:
            select_k = i+1
            break
    if select_k == 1:
        print('No attack detected!')
        return 0
    else:
        print('Attack Detected!')
        return 1


def main(args):
    if args.gpu == -1:
        ctx = mx.cpu()
    else:
        ctx = mx.gpu(args.gpu)

    with ctx:
        batch_size = args.batch_size

        if args.dataset == 'cifar10':
            num_inputs = 32 * 32 * 3
            num_outputs = 10
        else:
            raise NotImplementedError

            #################################################
        # Multiclass Logistic Regression
        MLR = gluon.nn.Sequential()
        with MLR.name_scope():
            MLR.add(gluon.nn.Dense(num_outputs))

        #################################################
        # two-layer fully connected nn
        fcnn = gluon.nn.Sequential()
        with fcnn.name_scope():
            fcnn.add(gluon.nn.Dense(256, activation="relu"))
            fcnn.add(gluon.nn.Dense(256, activation="relu"))
            fcnn.add(gluon.nn.Dense(num_outputs))

        #################################################
        # CNN
        cnn = gluon.nn.Sequential()
        with cnn.name_scope():
            cnn.add(gluon.nn.Conv2D(channels=30, kernel_size=5, activation='relu'))
            cnn.add(gluon.nn.MaxPool2D(pool_size=2, strides=2))
            cnn.add(gluon.nn.Conv2D(channels=50, kernel_size=5, activation='relu'))
            cnn.add(gluon.nn.MaxPool2D(pool_size=2, strides=2))
            # The Flatten layer collapses all axis, except the first one, into one axis.
            cnn.add(gluon.nn.Flatten())
            cnn.add(gluon.nn.Dense(512, activation="relu"))
            cnn.add(gluon.nn.Dense(num_outputs))

        ########################################################################################################################
        def evaluate_accuracy(data_iterator, net, trigger=False, target=None):
            acc = mx.metric.Accuracy()
            for i, (data, label) in enumerate(data_iterator):

                data = data.as_in_context(ctx)
                label = label.as_in_context(ctx)
                remaining_idx = list(range(data.shape[0]))
                if trigger:
                    for example_id in range(data.shape[0]):
                        data[example_id][0][1][28] = 1
                        data[example_id][1][1][28] = 1
                        data[example_id][2][1][28] = 1
                        data[example_id][0][1][29] = 1
                        data[example_id][1][1][29] = 1
                        data[example_id][2][1][29] = 1
                        data[example_id][0][1][30] = 1
                        data[example_id][1][1][30] = 1
                        data[example_id][2][1][30] = 1
                        data[example_id][0][2][29] = 1
                        data[example_id][1][2][29] = 1
                        data[example_id][2][2][29] = 1

                        data[example_id][0][3][28] = 1
                        data[example_id][1][3][28] = 1
                        data[example_id][2][3][28] = 1
                        data[example_id][0][4][29] = 1
                        data[example_id][1][4][29] = 1
                        data[example_id][2][4][29] = 1
                        data[example_id][0][5][28] = 1
                        data[example_id][1][5][28] = 1
                        data[example_id][2][5][28] = 1
                        data[example_id][0][5][29] = 1
                        data[example_id][1][5][29] = 1
                        data[example_id][2][5][29] = 1
                        data[example_id][0][5][30] = 1
                        data[example_id][1][5][30] = 1
                        data[example_id][2][5][30] = 1
                    for example_id in range(data.shape[0]):
                        if label[example_id] != target:
                            label[example_id] = target
                        else:
                            remaining_idx.remove(example_id)
                output = net(data)
                predictions = nd.argmax(output, axis=1)

                predictions = predictions[remaining_idx]
                label = label[remaining_idx]

                acc.update(preds=predictions, labels=label)
            return acc.get()[1]

        ########################################################################################################################
        # decide attack type
        if args.byz_type == 'partial_trim':
            # partial knowledge trim attack
            byz = byzantine.partial_trim
        elif args.byz_type == 'full_trim':
            # full knowledge trim attack
            byz = byzantine.full_trim
        elif args.byz_type == 'no':
            byz = byzantine.no_byz
        elif args.byz_type == 'gaussian':
            byz = byzantine.gaussian_attack
        elif args.byz_type == 'mean_attack':
            byz = byzantine.mean_attack
        elif args.byz_type == 'full_mean_attack':
            byz = byzantine.full_mean_attack
        elif args.byz_type == 'dir_partial_krum_lambda':
            byz = byzantine.dir_partial_krum_lambda
        elif args.byz_type == 'dir_full_krum_lambda':
            byz = byzantine.dir_full_krum_lambda
        elif args.byz_type == 'backdoor' or 'dba' or 'scaling_attack':
            byz = byzantine.scaling_attack
        elif args.byz_type == 'label_flip':
            byz = byzantine.no_byz
        else:
            raise NotImplementedError

            # decide model architecture
        if args.net == 'cnn':
            net = cnn
            net.collect_params().initialize(mx.init.Xavier(magnitude=2.24), force_reinit=True, ctx=ctx)
        elif args.net == 'fcnn':
            net = fcnn
            net.collect_params().initialize(mx.init.Xavier(magnitude=2.24), force_reinit=True, ctx=ctx)
        elif args.net == 'mlr':
            net = MLR
            net.collect_params().initialize(mx.init.Xavier(magnitude=1.), force_reinit=True, ctx=ctx)
        elif args.net == 'resnet':
            kwargs = {'classes': 10, 'thumbnail': True}
            res_layers = [3, 3, 3]
            res_channels = [16, 16, 32, 64]
            model = 'resnet20orig'
            resnet_class = models.ResNetV1
            block_class = models.BasicBlockV1
            net = resnet_class(block_class, res_layers, res_channels, **kwargs)
            net.initialize(mx.init.Xavier(magnitude=3.1415926), ctx=ctx)
        else:
            raise NotImplementedError

        # define loss
        softmax_cross_entropy = gluon.loss.SoftmaxCrossEntropyLoss()

        # set upt parameters
        num_workers = args.nworkers
        lr = args.lr
        epochs = args.nepochs
        grad_list = []
        old_grad_list = []
        weight_record = []
        grad_record = []
        train_acc_list = []

        # generate a string indicating the parameters
        paraString = str(args.dataset) + "+bias " + str(args.bias) + "+net " + str(
            args.net) + "+nepochs " + str(args.nepochs) + "+lr " + str(
            args.lr) + "+batch_size " + str(args.batch_size) + "+nworkers " + str(
            args.nworkers) + "+nbyz " + str(args.nbyz) + "+byz_type " + str(
            args.byz_type) + "+aggregation " + str(args.aggregation) + ".txt"

        # set up seed
        seed = args.seed
        mx.random.seed(seed)
        random.seed(seed)
        np.random.seed(seed)

        # load dataset
        if (args.dataset == 'cifar10' and args.net == 'cnn') or (args.dataset == 'cifar10' and args.net == 'resnet'):
            def transform(data, label):
                return nd.transpose(data.astype(np.float32), (2, 0, 1)) / 255., label.astype(np.float32)
            train_data = mx.gluon.data.DataLoader(mx.gluon.data.vision.CIFAR10(train=True, transform=transform), 50000,
                                                    shuffle=True, last_batch='rollover')
            test_data = mx.gluon.data.DataLoader(mx.gluon.data.vision.CIFAR10(train=False, transform=transform), 256,
                                                    shuffle=False, last_batch='rollover')

            # biased assignment
        bias_weight = args.bias
        other_group_size = (1 - bias_weight) / 9.
        worker_per_group = num_workers / 10

        # assign non-IID training data to each worker
        each_worker_data = [[] for _ in range(num_workers)]
        each_worker_label = [[] for _ in range(num_workers)]
        for _, (data, label) in enumerate(train_data):
            for (x, y) in zip(data, label):
                if args.dataset == 'cifar10':
                    x = x.as_in_context(ctx).reshape(1, 3, 32, 32)
                elif args.dataset == 'mnist' and args.net == 'cnn':
                    x = x.as_in_context(ctx).reshape(1, 1, 28, 28)
                else:
                    x = x.as_in_context(ctx).reshape(-1, num_inputs)
                y = y.as_in_context(ctx)

                # assign a data point to a group
                upper_bound = (y.asnumpy()) * (1 - bias_weight) / 9. + bias_weight
                lower_bound = (y.asnumpy()) * (1 - bias_weight) / 9.
                rd = np.random.random_sample()

                if rd > upper_bound:
                    worker_group = int(np.floor((rd - upper_bound) / other_group_size) + y.asnumpy() + 1)
                elif rd < lower_bound:
                    worker_group = int(np.floor(rd / other_group_size))
                else:
                    worker_group = y.asnumpy()

                # assign a data point to a worker
                rd = np.random.random_sample()
                selected_worker = int(worker_group * worker_per_group + int(np.floor(rd * worker_per_group)))
                each_worker_data[selected_worker].append(x)
                each_worker_label[selected_worker].append(y)

        # concatenate the data for each worker
        each_worker_data = [nd.concat(*each_worker, dim=0) for each_worker in each_worker_data]
        each_worker_label = [nd.concat(*each_worker, dim=0) for each_worker in each_worker_label]

        # random shuffle the workers
        random_order = np.random.RandomState(seed=seed).permutation(num_workers)
        each_worker_data = [each_worker_data[i] for i in random_order]
        each_worker_label = [each_worker_label[i] for i in random_order]


        ### begin training
        print('start training!')
        #set malicious scores
        malicious_score = np.zeros((1, args.nworkers))
        for e in range(epochs):
            # for each worker
            for i in range(100):
                batch_x = each_worker_data[i][:]
                batch_y = each_worker_label[i][:]
                if args.byz_type == 'scaling_attack':
                    if i < args.nbyz:
                        for example_id in range(batch_x.shape[0] // 2):
                            # add the trigger to half of the images in the batch
                            # the trigger is the same as that used in "how to backdoor federated learning"
                            batch_x[example_id][0][1][28] = 1
                            batch_x[example_id][1][1][28] = 1
                            batch_x[example_id][2][1][28] = 1
                            batch_x[example_id][0][1][29] = 1
                            batch_x[example_id][1][1][29] = 1
                            batch_x[example_id][2][1][29] = 1
                            batch_x[example_id][0][1][30] = 1
                            batch_x[example_id][1][1][30] = 1
                            batch_x[example_id][2][1][30] = 1
                            batch_x[example_id][0][2][29] = 1
                            batch_x[example_id][1][2][29] = 1
                            batch_x[example_id][2][2][29] = 1

                            batch_x[example_id][0][3][28] = 1
                            batch_x[example_id][1][3][28] = 1
                            batch_x[example_id][2][3][28] = 1
                            batch_x[example_id][0][4][29] = 1
                            batch_x[example_id][1][4][29] = 1
                            batch_x[example_id][2][4][29] = 1
                            batch_x[example_id][0][5][28] = 1
                            batch_x[example_id][1][5][28] = 1
                            batch_x[example_id][2][5][28] = 1
                            batch_x[example_id][0][5][29] = 1
                            batch_x[example_id][1][5][29] = 1
                            batch_x[example_id][2][5][29] = 1
                            batch_x[example_id][0][5][30] = 1
                            batch_x[example_id][1][5][30] = 1
                            batch_x[example_id][2][5][30] = 1

                            batch_y[example_id] = 0

                backdoor_target = 0
                with autograd.record():
                    output = net(batch_x)
                    loss = softmax_cross_entropy(output, batch_y)
                # backward
                loss.backward()
                grad_list.append([param.grad().copy() for param in net.collect_params().values() if param.grad_req != 'null'])

            param_list = [nd.concat(*[xx.reshape((-1, 1)) for xx in x], dim=0) for x in grad_list]

            tmp = []
            for param in net.collect_params().values():
                if param.grad_req != 'null':
                    tmp.append(param.data().copy())
            weight = nd.concat(*[x.reshape((-1, 1)) for x in tmp], dim=0)

            # use lbfgs to calculate hessian vector product
            if e > 20:
                hvp = lbfgs(args, weight_record, grad_record, weight - last_weight)
            else:
                hvp = None

            # perform attack
            if e > 0:
                param_list = byz(param_list, args.nbyz)

            if args.net == 'resnet':
                if e > 500:
                    lr = args.lr / 5.
                elif e > 750:
                    lr = args.lr / 20.
                else:
                    lr = args.lr

            if args.aggregation == 'trim':
                grad, distance = nd_aggregation.trim(old_grad_list, param_list, net, lr, args.nbyz, hvp)
            elif args.aggregation == 'simple_mean':
                grad, distance = nd_aggregation.simple_mean(old_grad_list, param_list, net, lr, args.nbyz, hvp)
            elif args.aggregation == 'median':
                grad, distance = nd_aggregation.median(old_grad_list, param_list, net, lr, args.nbyz, hvp)
            elif args.aggregation == 'krum':
                grad, distance = nd_aggregation.krum(old_grad_list, param_list, net, lr, args.nbyz, hvp)
            else:
                raise NotImplementedError
            # Update malicious distance score
            if distance is not None and e > 20:
                malicious_score = np.row_stack((malicious_score, distance))

            if malicious_score.shape[0] >= 11:
                if detection1(np.sum(malicious_score[-10:], axis=0), args.nbyz):
                    print('Stop at iteration:', e)
                    detection(np.sum(malicious_score[-10:], axis=0), args.nbyz)
                    break

            # update weight record and gradient record
            if e > 0:
                weight_record.append(weight - last_weight)
                grad_record.append(grad - last_grad)

            # free memory & reset the list
            if len(weight_record) > 10:
                del weight_record[0]
                del grad_record[0]

            last_weight = weight
            last_grad = grad
            old_grad_list = param_list
            del grad_list
            grad_list = []

            # compute training accuracy every 10 iterations
            if (e + 1) % 5 == 0:
                if args.byz_type == 'scaling_attack' or 'no':
                    test_accuracy = evaluate_accuracy(test_data, net)
                    backdoor_acc = evaluate_accuracy(test_data, net, trigger=True, target=backdoor_target)
                    train_acc_list.append((test_accuracy, backdoor_acc))
                    print("Epoch %02d. Test_acc %0.4f. Backdoor_acc %0.4f." % (e, test_accuracy, backdoor_acc))

                else:
                    test_accuracy = evaluate_accuracy(test_data, net)
                    train_acc_list.append(test_accuracy)
                    print("Epoch %02d. Test_acc %0.4f" % (e, test_accuracy))

            # save the training accuracy every 100 iterations
            if (e + 1) % 100 == 0:
                if args.dataset == 'cifar10':
                    if not os.path.exists('out_cifar_resnet/'):
                        os.mkdir('out_cifar_resnet/')
                    np.savetxt('out_cifar_resnet/' + paraString, train_acc_list, fmt='%.4f')
                else:
                    raise NotImplementedError

            # compute the final testing accuracy
            if (e + 1) == args.nepochs:
                test_accuracy = evaluate_accuracy(test_data, net)
                print("Epoch %02d. Test_acc %0.4f" % (e, test_accuracy))
                detection(np.sum(malicious_score[-10:], axis=0), args.nbyz)



if __name__ == "__main__":
    args = parse_args()
    main(args)
