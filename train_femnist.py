from __future__ import print_function
import nd_aggregation
import mxnet as mx
from mxnet import nd, autograd, gluon
import numpy as np
import random
import argparse
import byzantine
import sys
import os
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import scipy
import json

os.environ["MXNET_CUDNN_AUTOTUNE_DEFAULT"] = "0"
np.warnings.filterwarnings('ignore')


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", help="dataset", default='femnist', type=str)
    parser.add_argument("--bias", help="degree of non-IID to assign data to workers", type=float, default=0.1)
    parser.add_argument("--net", help="net", default='cnn', type=str, choices=['mlr', 'cnn', 'fcnn'])
    parser.add_argument("--batch_size", help="batch size", default=32, type=int)
    parser.add_argument("--lr", help="learning rate", default=0.002, type=float)
    parser.add_argument("--nworkers", help="# workers", default=300, type=int)
    parser.add_argument("--nepochs", help="# epochs", default=2000, type=int)
    parser.add_argument("--gpu", help="index of gpu", default=0, type=int)
    parser.add_argument("--seed", help="seed", default=42, type=int)
    parser.add_argument("--nbyz", help="# byzantines", default=84, type=int)
    parser.add_argument("--byz_type", help="type of attack", default='scaling_attack', type=str,
                        choices=['no', 'partial_trim', 'full_trim', 'mean_attack', 'full_mean_attack', 'gaussian',
                                 'dir_partial_krum_lambda', 'dir_full_krum_lambda', 'label_flip', 'backdoor', 'dba',
                                 'scaling_attack'])
    parser.add_argument("--aggregation", help="aggregation rule", default='trim', type=str,
                        choices=['simple_mean', 'trim', 'krum', 'median'])
    return parser.parse_args()


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
    real_label=np.ones(300)
    real_label[:nobyz]=0
    acc=len(label_pred[label_pred==real_label])/300
    recall=1-np.sum(label_pred[:nobyz])/nobyz
    fpr=1-np.sum(label_pred[nobyz:])/(300-nobyz)
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

        if args.dataset == 'femnist':
            num_inputs = 28 * 28
            num_outputs = 62
            input_size = (1, 1, 28, 28)
        else:
            raise NotImplementedError

            #################################################
        # Multiclass Logistic Regression
        MLR = gluon.nn.Sequential()
        with MLR.name_scope():
            MLR.add(gluon.nn.Dense(num_outputs))

            #################################################

        vae = gluon.nn.Sequential()
        grad_len = 5120
        with vae.name_scope():
            vae.add(gluon.nn.Dense(500, activation="relu"))
            vae.add(gluon.nn.Dense(100, activation="relu"))
            vae.add(gluon.nn.Dense(500, activation="relu"))
            vae.add(gluon.nn.Dense(grad_len))

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
                        batch_x[example_id][0][26][26] = 1
                        batch_x[example_id][0][24][26] = 1
                        batch_x[example_id][0][26][24] = 1
                        batch_x[example_id][0][25][25] = 1

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
        # assign non-IID training data to each worker
        each_worker_data = []
        each_worker_label = []
        each_worker_num = []
        for i in range(30):
            filestring= "../leaf/data/femnist/data/train/" + "all_data_"+str(i) + "_niid_1_keep_100_train_9.json"
            with open(filestring, 'r') as f:
                load_dict = json.load(f)
                each_worker_num.extend(load_dict['num_samples'])
                for user in load_dict['users']:
                    x = nd.array(load_dict['user_data'][user]['x']).as_in_context(ctx).reshape(-1, 1, 28, 28)
                    y = nd.array(load_dict['user_data'][user]['y']).as_in_context(ctx)

                    each_worker_data.append(x)
                    each_worker_label.append(y)

        # random shuffle the workers
        random_order = np.random.RandomState(seed=seed).permutation(num_workers)
        each_worker_data = [each_worker_data[i] for i in random_order]
        each_worker_label = [each_worker_label[i] for i in random_order]
        each_worker_num = nd.array([each_worker_num[i] for i in random_order]).as_in_context(ctx)

        dataset = mx.gluon.data.dataset.ArrayDataset(nd.concat(*each_worker_data[:2], dim=0), nd.concat(*each_worker_label[:2], dim=0))
        test_data = mx.gluon.data.DataLoader(dataset, 8, shuffle=False)

        # perform attacks
        if args.byz_type == 'label_flip':
            for i in range(args.nbyz):
                each_worker_label[i] = (each_worker_label[i] + 1) % 9
        if args.byz_type == 'backdoor':
            for i in range(args.nbyz):
                each_worker_data[i] = nd.repeat(each_worker_data[i][:300], repeats=2, axis=0)
                each_worker_label[i] = nd.repeat(each_worker_label[i][:300], repeats=2, axis=0)
                for example_id in range(0, each_worker_data[i].shape[0], 2):
                    each_worker_data[i][example_id][0][26][26] = 1
                    each_worker_data[i][example_id][0][24][26] = 1
                    each_worker_data[i][example_id][0][26][24] = 1
                    each_worker_data[i][example_id][0][25][25] = 1
                    each_worker_label[i][example_id] = 0
        if args.byz_type == 'dba':
            for i in range(int(args.nbyz / 4)):
                each_worker_data[i] = nd.repeat(each_worker_data[i][:300], repeats=2, axis=0)
                each_worker_label[i] = nd.repeat(each_worker_label[i][:300], repeats=2, axis=0)
                for example_id in range(0, each_worker_data[i].shape[0], 2):
                    each_worker_data[i][example_id][0][26][26] = 1
                    each_worker_label[i][example_id] = 0
            for i in range(int(args.nbyz / 4), int(args.nbyz / 2)):
                each_worker_data[i] = nd.repeat(each_worker_data[i][:300], repeats=2, axis=0)
                each_worker_label[i] = nd.repeat(each_worker_label[i][:300], repeats=2, axis=0)
                for example_id in range(0, each_worker_data[i].shape[0], 2):
                    each_worker_data[i][example_id][0][24][26] = 1
                    each_worker_label[i][example_id] = 0
            for i in range(int(args.nbyz / 2), int(args.nbyz * 3 / 4)):
                each_worker_data[i] = nd.repeat(each_worker_data[i][:300], repeats=2, axis=0)
                each_worker_label[i] = nd.repeat(each_worker_label[i][:300], repeats=2, axis=0)
                for example_id in range(0, each_worker_data[i].shape[0], 2):
                    each_worker_data[i][example_id][0][26][24] = 1
                    each_worker_label[i][example_id] = 0
            for i in range(int(args.nbyz * 3 / 4), args.nbyz):
                each_worker_data[i] = nd.repeat(each_worker_data[i][:300], repeats=2, axis=0)
                each_worker_label[i] = nd.repeat(each_worker_label[i][:300], repeats=2, axis=0)
                for example_id in range(0, each_worker_data[i].shape[0], 2):
                    each_worker_data[i][example_id][0][25][25] = 1
                    each_worker_label[i][example_id] = 0

        if args.byz_type == 'no':
            byz = byzantine.no_byz

        ### begin training
        #set malicious scores
        malicious_score = np.zeros((1, args.nworkers))
        for e in range(epochs):
            for i in range(300):
                batch_x = each_worker_data[i][:]
                batch_y = each_worker_label[i][:]
                if args.byz_type == 'scaling_attack' and e > 20:
                    if i < args.nbyz:
                        for example_id in range(batch_x.shape[0] // 2):
                            batch_x[example_id][0][26][26] = 1
                            batch_x[example_id][0][24][26] = 1
                            batch_x[example_id][0][26][24] = 1
                            batch_x[example_id][0][25][25] = 1
                            batch_y[example_id] = 0

                backdoor_target = 0
                with autograd.record():
                    output = net(batch_x)
                    loss = softmax_cross_entropy(output, batch_y)*32/each_worker_num[i]
                # backward
                loss.backward()
                grad_list.append([param.grad().copy() for param in net.collect_params().values() if param.grad_req != 'null'])

            param_list = [nd.concat(*[xx.reshape((-1, 1)) for xx in x], dim=0) for x in grad_list]

            tmp = []
            for param in net.collect_params().values():
                tmp.append(param.data().copy())
            weight = nd.concat(*[x.reshape((-1, 1)) for x in tmp], dim=0)

            # use lbfgs to calculate hessian vector product
            if e > 50:
                hvp = lbfgs(args, weight_record, grad_record, weight - last_weight)
            else:
                hvp = None

            # perform attack
            if e > 0:
                param_list = byz(param_list, args.nbyz)

            if args.net == 'cnn':
                if e > 200:
                    lr = args.lr / 5.
                elif e > 500:
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

            if distance is not None and e > 50:
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
                if args.byz_type == 'scaling_attack':
                    test_accuracy = evaluate_accuracy(test_data, net)
                    backdoor_acc = evaluate_accuracy(test_data, net, trigger=True, target=backdoor_target)
                    train_acc_list.append((test_accuracy, backdoor_acc))
                    print("Epoch %02d. Test_acc %0.4f. Backdoor_acc %0.4f." % (e, test_accuracy, backdoor_acc))

                else:
                    test_accuracy = evaluate_accuracy(test_data, net)
                    train_acc_list.append(test_accuracy)
                    print("Epoch %02d. Test_acc %0.4f" % (e, test_accuracy))

            # save the training accuracy every 50 iterations
            if (e + 1) % 50 == 0:
                if (args.dataset == 'femnist' and args.net == 'cnn'):
                    if not os.path.exists('out_femnist_cnn/'):
                        os.mkdir('out_femnist_cnn/')
                    np.savetxt('out_femnist_cnn/' + paraString, train_acc_list, fmt='%.4f')
                else:
                    raise NotImplementedError

            # compute the final testing accuracy
            if (e + 1) == args.nepochs:
                test_accuracy = evaluate_accuracy(test_data, net)
                print("Epoch %02d. Test_acc %0.4f" % (e, test_accuracy))
                #detection(np.sum(malicious_score[-10:], axis=0), args.nbyz)


if __name__ == "__main__":
    args = parse_args()
    main(args)
