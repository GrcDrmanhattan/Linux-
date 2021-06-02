#
# A python3 (re-)implementation of DepthLGP, with some modifications.
# (The original code was lost due to a server crash happened in late 2017.)
# If you find any bug or degrade performance, please contact the author.
# Oh, play fair and do remember to tune the hyper-parameters.
#

from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os
import subprocess
import time

import networkx as nx
import numpy as np
import scipy.io
import scipy.sparse
import sklearn.linear_model
import sklearn.metrics
import sklearn.multiclass
import torch
import torch.nn as nn
import torch.nn.functional as nn_f
import torch.optim as optim
from torch.autograd import Variable

from data.dblp import CoauthorNetwork


class MultiLabelEval(object):
    @staticmethod
    def eval(train_x, train_y, test_x, test_y):
        classifier = sklearn.multiclass.OneVsRestClassifier(
            sklearn.linear_model.LogisticRegression(), n_jobs=-1)
        classifier.fit(train_x, train_y)
        score = classifier.predict_proba(test_x)
        return MultiLabelEval._compute_all_errors(test_y, score)

    @staticmethod
    def _preserve_k_top(score, k, axis):
        assert score.ndim == 2
        assert k.shape == (score.shape[1 - axis],)
        index = np.argsort(-score, axis=axis)
        pred = np.zeros_like(score, np.int)
        for i in range(score.shape[1 - axis]):
            if axis == 0:
                pred[index[:k[i], i], i] = 1
            else:
                pred[i, index[i, :k[i]]] = 1
        return pred

    @staticmethod
    def _compute_tang09_error(score, y, label_wise=False):
        """
        Translated from a MATLAB script provided by Tang & Liu. See:
            Relational learning via latent social dimensions. KDD '09.
        """
        assert score.ndim == 2
        assert score.shape == y.shape
        assert y.dtype in (np.int, np.bool)
        if y.dtype == np.bool:
            y = y.astype(np.int)
        index = (np.sum(y, axis=1) > 0)  # remove samples with no labels
        y = y[index]
        score = score[index]
        if label_wise:
            # Assuming the number of samples per label is known,
            # preserve only top-scored samples for each label.
            pred = MultiLabelEval._preserve_k_top(score, np.sum(y, 0), 0)
        else:
            # Assuming the number of labels per sample is known,
            # preserve only top-scored labels for each sample.
            pred = MultiLabelEval._preserve_k_top(score, np.sum(y, 1), 1)
        acc = np.sum(np.sum(pred ^ y, 1) == 0) / y.shape[0]  # exact match
        num_correct = np.sum(pred & y, 0)
        num_true = np.sum(y, 0)
        num_pred = np.sum(pred, 0)
        nz_cls = (num_correct != 0)  # for preventing divide-by-zero
        precision = np.zeros_like(num_correct, np.float)
        precision[nz_cls] = num_correct[nz_cls] / num_pred[nz_cls]
        recall = np.zeros_like(num_correct, np.float)
        recall[nz_cls] = num_correct[nz_cls] / num_true[nz_cls]
        f1 = np.zeros_like(num_correct, np.float)
        f1[nz_cls] = 2 * num_correct[nz_cls] / (
                num_true[nz_cls] + num_pred[nz_cls])
        macro_precision = np.mean(precision)
        macro_recall = np.mean(recall)
        macro_f1 = np.mean(f1)
        micro_precision = np.sum(num_correct) / np.sum(num_pred)
        micro_recall = np.sum(num_correct) / np.sum(num_true)
        micro_f1 = 2 * np.sum(num_correct) / (
                np.sum(num_true) + np.sum(num_pred))
        return {
            'acc': acc,
            'macro_precision': macro_precision,
            'micro_precision': micro_precision,
            'macro_recall': macro_recall,
            'micro_recall': micro_recall,
            'macro_f1': macro_f1,
            'micro_f1': micro_f1
        }

    @staticmethod
    def _compute_all_errors(y, score):
        perf = {}
        tang09_metrics = MultiLabelEval._compute_tang09_error(score, y)
        perf['macro_f1'] = tang09_metrics['macro_f1']
        perf['micro_f1'] = tang09_metrics['micro_f1']
        perf['coverage_error'] = (sklearn.metrics.coverage_error(y, score))
        perf['label_ranking_average_precision_score'] = (
            sklearn.metrics.label_ranking_average_precision_score(y, score))
        perf['label_ranking_loss'] = (
            sklearn.metrics.label_ranking_loss(y, score))
        return perf


class AlgoBase(object):
    def get_vecs(self, work_dir, train, num_nodes, prefix='train'):
        edge_list = os.path.join(work_dir, prefix + '.elist')
        nx.write_edgelist(train, edge_list, data=False)
        feats_path = self._learn_vecs(edge_list)
        feats = AlgoBase._read_vecs(feats_path, num_nodes)
        return feats

    @staticmethod
    def _read_vecs(vec_path, num_nodes):
        with open(vec_path, 'r') as fin:
            num_lines, d = fin.readline().split()
            num_lines, d = int(num_lines), int(d)
            vecs = np.zeros((num_nodes, d), dtype=np.float32)
            for _ in range(num_lines):
                s = fin.readline().split()
                vecs[int(s[0])] = np.asarray([float(x) for x in s[1:]])
        return vecs

    def _learn_vecs(self, _):
        assert False


class AlgoN2VPY(AlgoBase):
    def __init__(self):
        AlgoBase.__init__(self)
        bin_path = "./ext/node2vec/"
        if not os.path.isdir(bin_path):
            subprocess.check_call([
                "git", "clone", "https://github.com/aditya-grover/node2vec.git",
                bin_path
            ])

    def _learn_vecs(self, edge_list_path):
        vec_path = edge_list_path + '.n2v'
        if not os.path.exists(vec_path):
            subprocess.check_call([
                'python2', './ext/node2vec/src/main.py', '--dimensions',
                '128', '--num-walks', '10', '--workers', '1',
                '--input', edge_list_path, '--output', vec_path])
        return vec_path


def tensor_to_var(x, requires_grad=False, volatile=False):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x, requires_grad=requires_grad, volatile=volatile)


def validate_graph(train, whole):
    n_all = whole.number_of_nodes()
    n_old = train.number_of_nodes()
    n_new = n_all - n_old
    assert n_old > 0 and n_new > 0
    for u in train.nodes():
        assert 0 <= u < n_old
    for u in whole.nodes():
        assert 0 <= u < n_all


class MyModel(nn.Module):
    def __init__(self, n_nodes, n_dim):
        super(MyModel, self).__init__()
        self.embed = nn.Embedding(n_nodes, n_dim)
        self.weigh = nn.Parameter(torch.ones(n_nodes))
        self.fc1 = nn.Linear(n_dim, n_dim // 2)
        self.fc2 = nn.Linear(n_dim // 2, n_dim)
        self.coef = nn.Parameter(torch.ones(2))

    def _net_forward(self, new_xs):
        output = nn_f.leaky_relu(self.fc1(new_xs))
        output = self.fc2(output)
        return new_xs + output

    def init_params(self, feats):
        next(self.embed.parameters()).data.copy_(feats)
        self.weigh.data.zero_().add_(1.00)
        self.fc1.reset_parameters()
        self.fc2.weight.data.zero_()
        self.fc2.bias.data.zero_()
        self.coef.data[0] = 10.0
        self.coef.data[1] = 0.00

    def correct_params(self):
        w = self.weigh.data
        w[w < 0.0] = 0.0
        w[w > 1.0] = 1.0
        c = self.coef.data
        c[c < 0.0] = 0.0

    def forward(self, old_vs, adj_all):
        old_xs = self.embed(old_vs)
        n_dim = old_xs.size(1)
        n_old = old_xs.size(0)
        n_all = adj_all.size(0)
        n_new = n_all - n_old
        w = torch.cat((self.weigh[old_vs], tensor_to_var(torch.ones(n_new))))
        w = w.unsqueeze(1).expand(n_all, n_all)
        adj1 = w * adj_all * w.t()
        lap1 = (adj1.sum(1).diag() - adj1) * self.coef[0]
        adj2 = torch.mm(adj1, adj1)
        lap2 = (adj2.sum(1).diag() - adj2) * self.coef[1]
        adj = tensor_to_var(torch.eye(n_all)) + lap1 + lap2
        adj_no = adj[n_old:n_all, :n_old]
        adj_nn = adj[n_old:n_all, n_old:n_all]
        adj_nn_inv = adj_nn.inverse()
        new_xs = []
        for i in range(n_dim):
            new_x = -adj_nn_inv.mv(adj_no.mv(old_xs[:, i]))
            new_xs.append(new_x.unsqueeze(1))
        new_xs = torch.cat(new_xs, 1)
        return self._net_forward(new_xs)

    def predict_partial(self, old_vs, adj_all):
        assert isinstance(old_vs, list)
        assert isinstance(adj_all, scipy.sparse.csr_matrix)
        old_vs = torch.from_numpy(np.array(old_vs, dtype=np.int64))
        old_vs = tensor_to_var(old_vs, volatile=True)
        adj_all = torch.from_numpy(adj_all.astype(np.float32).toarray())
        adj_all = tensor_to_var(adj_all, volatile=True)
        return self.forward(old_vs, adj_all)

    @staticmethod
    def _compute_lap(graph, n_old, w, alph, beta):
        n_all = graph.number_of_nodes()
        n_new = n_all - n_old
        assert isinstance(w, np.ndarray)
        assert w.shape == (n_all,)
        lap_na = np.zeros((n_new, n_all), dtype=np.float32)
        for u in range(n_old, n_all):
            uu = u - n_old
            lap_na[uu, u] += 1.0
            for v in graph.neighbors(u):
                d = alph * w[u] * w[v]
                lap_na[uu, u] += d
                lap_na[uu, v] -= d
                for r in graph.neighbors(v):
                    d = beta * w[u] * w[v] * w[v] * w[r]
                    lap_na[uu, u] += d
                    lap_na[uu, r] -= d
        lap_no = lap_na[:, :n_old]
        lap_nn = lap_na[:, n_old:]
        return lap_no, lap_nn

    def print_model_stats(self):
        alph = self.coef.data[0]
        beta = self.coef.data[1]
        print('Alpha = %.4f, Beta = %.4f' % (alph, beta))
        w = self.weigh.data.cpu().numpy()
        print('Weights = %.4f(max)/%.4f(avg)/%.4f(min)' % (
            np.max(w), w.sum() / w.size, np.min(w)))
        p_nn = self.fc2.weight.data.cpu().numpy()
        print('NN Last = %.4f(max)/%.4f(avg)/%.4f(min)' % (
            np.max(p_nn), p_nn.sum() / p_nn.size, np.min(p_nn)))

    def predict(self, graph, n_old, feats):
        assert isinstance(feats, torch.cuda.FloatTensor)
        w = self.weigh.data.cpu().numpy()
        w[n_old:] = 1.0
        alph = self.coef.data[0]
        beta = self.coef.data[1]
        lap_no, lap_nn = MyModel._compute_lap(graph, n_old, w, alph, beta)
        lap_no = torch.from_numpy(lap_no).cuda()
        lap_nn = torch.from_numpy(lap_nn).cuda()
        lap_nn_inv = lap_nn.inverse()
        embed = next(self.embed.parameters()).data
        n_dim = embed.size(1)
        for i in range(n_dim):
            feats[n_old:, i] = -lap_nn_inv.mv(lap_no.mv(embed[:n_old, i]))
        new_xs = Variable(feats[n_old:], requires_grad=False, volatile=True)
        feats[n_old:] = self._net_forward(new_xs).data


class Trainer(object):
    def __init__(self, train_feats, n_all_nodes):
        n_dim = train_feats.shape[1]
        self.model = MyModel(n_all_nodes, n_dim)
        if torch.cuda.is_available():
            print('Using CUDA...')
            self.model.cuda()
        self.feats = tensor_to_var(torch.from_numpy(train_feats))
        self.model.init_params(self.feats.data)
        self.optim = optim.Adam(self.model.parameters())

    def save_model(self, path, save_optim=False):
        obj = {"model": self.model.state_dict()}
        if save_optim:
            obj["optim"] = self.optim.state_dict()
        torch.save(obj, path)

    def load_model(self, path):
        obj = torch.load(path)
        self.model.load_state_dict(obj["model"])
        if "optim" in obj:
            self.optim.load_state_dict(obj["optim"])

    def train(self, train_graph, times):
        mse_loss = nn.MSELoss()
        if torch.cuda.is_available():
            mse_loss.cuda()
        avg_loss = None
        for ti in range(times):
            while True:
                s = np.random.choice(train_graph.nodes())
                vs, new_sz = Trainer._sample_a_subgraph(train_graph, s)
                if len(vs) > new_sz:
                    break
            adj = nx.adjacency_matrix(train_graph, vs).toarray()
            adj = torch.from_numpy(adj.astype(np.float32))
            vs = torch.from_numpy(np.array(vs, dtype=np.int64))
            vs = tensor_to_var(vs)
            old_vs = vs[:-new_sz]
            new_vs = vs[-new_sz:]
            adj = tensor_to_var(adj)
            self.optim.zero_grad()
            pred = self.model(old_vs, adj)
            gold = self.feats.index_select(0, new_vs)
            loss = mse_loss(pred, gold)
            loss_val = loss.data[0]
            if loss_val < 1.0:
                avg_loss = avg_loss or loss_val
                avg_loss = avg_loss * 0.95 + loss_val * 0.05
                loss.backward()
                self.optim.step()
                self.model.correct_params()
            if ti > 0 and ti % 32 == 0:
                print('Processed %d/%d: %d -> %d... Loss = %.4f' % (
                    ti, times, len(vs), new_sz, avg_loss))
                avg_loss = None

    def predict(self, train, whole, partial=True):
        self.model.print_model_stats()
        if not partial:
            validate_graph(train, whole)
            self.model.predict(whole, train.number_of_nodes(),
                               self.feats.data)
        else:
            trn_set = set(train.nodes())
            tst_set = set(whole.nodes()) - trn_set
            for s in nx.connected_components(whole.subgraph(tst_set)):
                b = Trainer._sample_neighborhood(whole, s, ignore=tst_set)
                print('Test Case: %d -> %d...' % (len(b), len(s)))
                old_vs = list(b)
                new_vs = list(s)
                all_vs = old_vs + new_vs
                adj = nx.adjacency_matrix(whole, all_vs)
                pred = self.model.predict_partial(old_vs, adj).data
                self.feats.data[new_vs, :] = pred
        return self.feats.data.cpu().numpy()

    @staticmethod
    def _sample_neighborhood(g, core, amp_szs=(128, 0.5), ignore=set()):
        depth = len(amp_szs)
        assert isinstance(core, set)
        assert depth > 0
        curr = core
        s = core.copy()
        last_depth_sz = len(core)
        for di in range(depth):
            succ = set()
            for u in curr:
                for v in g.neighbors(u):
                    if (v not in s) and (v not in ignore):
                        succ.add(v)
            cap_sz = int(last_depth_sz * amp_szs[di])
            if len(succ) > cap_sz:
                succ = set(np.random.choice(
                    list(succ), cap_sz, replace=False))
            last_depth_sz = len(succ)
            s = s.union(succ)
            curr = succ
        return s - core

    @staticmethod
    def _sample_a_subgraph(train, s, new_sz=4):
        y = Trainer._sample_neighborhood(train, {s}, (new_sz,) * new_sz)
        y.add(s)
        y = set(np.random.choice(list(y), min(len(y), new_sz), replace=False))
        x = Trainer._sample_neighborhood(train, y)
        return list(x) + list(y), len(y)


def get_my_pred(num_all_nodes, train, feats, work_dir,
                times=1024, ldsv=False):
    trainer = Trainer(feats, num_all_nodes)
    model_path = os.path.join(work_dir, 'model.sav')
    if ldsv and os.path.exists(model_path):
        trainer.load_model(model_path)
    if times > 0:
        trainer.train(train, times)
        if ldsv:
            trainer.save_model(model_path)
    return lambda whole: trainer.predict(train, whole)


def get_retrain_pred(algocls, num_all_nodes, work_dir, prefix):
    return lambda whole: algocls().get_vecs(work_dir, whole, num_all_nodes,
                                            prefix=prefix)


class EvalAgent(object):
    def __init__(self, dataset, work_dir, algocls):
        retrain_flag = False
        if not os.path.exists(work_dir):
            os.mkdir(work_dir)
        whole = dataset.get_whole_net()
        train = dataset.get_train_net()
        label = dataset.get_label_mat()
        if retrain_flag:
            feats = np.zeros((whole.number_of_nodes(), 128), dtype=np.float32)
        else:
            feats = algocls().get_vecs(work_dir, train, whole.number_of_nodes())
        print('Nodes = %d/%d, Edges = %d/%d, Feats = %dx%d, Labels = %d' % (
            train.number_of_nodes(), whole.number_of_nodes(),
            train.number_of_edges(), whole.number_of_edges(),
            feats.shape[0], feats.shape[1], label.shape[1]))
        if retrain_flag:
            pred = get_retrain_pred(algocls, whole.number_of_nodes(), work_dir,
                                    'whole')
        else:
            pred = get_my_pred(whole.number_of_nodes(), train, feats, work_dir)
        results = EvalAgent._eval(pred, label, train, whole)
        for k in sorted(results.keys()):
            print(k, results[k])

    @staticmethod
    def _eval(preditor, label, train, whole):
        assert train.number_of_nodes() < whole.number_of_nodes()
        time_spent = time.time()
        feats = preditor(whole)
        time_spent = time.time() - time_spent
        print('time spent on predicting: %.2f sec' % time_spent)
        res = {}
        res.update(MultiLabelEval.eval(
            *EvalAgent._split_test(feats, label, train, whole)))
        return res

    @staticmethod
    def _split_test(feats, labels, train_net, whole_net):
        in_train = np.zeros(whole_net.number_of_nodes(), dtype=np.bool)
        for i in train_net.nodes():
            in_train[i] = True
        in_test = ~in_train
        return (
            feats[in_train], labels[in_train],
            feats[in_test], labels[in_test]
        )


def main():
    dname = "coauthor"
    if dname == 'coauthor':
        # from data.dblp import CoauthorNetwork
        dataset = CoauthorNetwork('./data/')
    else:
        print("You need to prepare your own data.")
        assert False
    EvalAgent(dataset, './data/%s/' % dname, AlgoN2VPY)


if __name__ == '__main__':
    main()
