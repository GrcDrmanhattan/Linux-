# coding=utf-8
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os
import pickle
import xml.etree.ElementTree

import networkx as nx
import numpy as np


class DBLP(object):
    @staticmethod
    def get_articles(dir_path='./'):
        pkl_path = os.path.join(dir_path, 'dblp_cs.pkl')
        if not os.path.exists(pkl_path):
            articles = DBLP._extract_cs(dir_path)
            with open(pkl_path, 'wb') as fout:
                pickle.dump(articles, fout)
        else:
            with open(pkl_path, 'rb') as fin:
                articles = pickle.load(fin)
        return articles

    @staticmethod
    def _extract_cs(dir_path):
        xml_path = os.path.join(dir_path, 'dblp.xml')
        root = xml.etree.ElementTree.parse(xml_path).getroot()
        articles = []
        for venue in ['ICDE', 'VLDB', 'SIGMOD Conference', 'KDD', 'ICDM',
                      'SDM', 'CIKM', 'SIGIR', 'IJCAI', 'AAAI', 'ICML', 'NIPS',
                      'CVPR', 'ICCV', 'STOC', 'SODA', 'COLT', 'ACL', 'EMNLP',
                      'COLING', 'SIGCOMM', 'INFOCOM', 'SOSP', 'OSDI', 'POPL']:
            cnt = 0
            for paper in root.iter('inproceedings'):
                book = next(paper.iter('booktitle')).text
                year = int(next(paper.iter('year')).text)
                if book == venue and year <= 2016:
                    articles.append(paper)
                    cnt += 1
            print('Added %d papers from %s.' % (cnt, venue))
        print('Extracted %d papers in total.' % len(articles))
        return articles


class CoauthorNetwork(object):
    def __init__(self, dir_path='./'):
        network, train_net, labels = CoauthorNetwork._load_all(dir_path)
        self.whole_net = network
        self.train_net = train_net
        self.label_mat = labels

    def get_whole_net(self):
        return self.whole_net

    def get_train_net(self):
        return self.train_net

    def get_label_mat(self):
        return self.label_mat

    def print_statistics(self):
        whole_n = self.whole_net.number_of_nodes()  # 返回节点数
        whole_m = self.whole_net.number_of_edges()  # 返回边数
        train_n = self.train_net.number_of_nodes()
        train_m = self.train_net.number_of_edges()
        print('%d nodes, %d edges in the whole network.' % (whole_n, whole_m))
        print('%d nodes, %d edges in the train network.' % (train_n, train_m))  # 训练网络节点，边数
        print('%d new nodes, %d new edges in the whole network.' % (
            whole_n - train_n, whole_m - train_m))  # 新节点，新边数
        is_new = np.ones(whole_n, dtype=np.bool)  # 有值设置为1的矩阵,
        for u in self.train_net.nodes():
            is_new[u] = False  # 如果在训练集出现设置为false
        cnt_between = 0
        cnt_within_new = 0
        for u, v in self.whole_net.edges():
            if is_new[u] != is_new[v]:
                cnt_between += 1    # 新老节点
            elif is_new[u] and is_new[v]:
                cnt_within_new += 1  # 新节点之间
        print('%d edges are between new and old nodes.' % cnt_between)
        print('%d edges are within new nodes.' % cnt_within_new)

    @staticmethod
    def _load_all(dir_path):
        coauthor_pkl = os.path.join(dir_path, 'dblp_cs_coauthor1.pkl')
        if os.path.exists(coauthor_pkl):
            with open(coauthor_pkl, 'rb') as fin:
                whole_train_label = pickle.load(fin)
        else:
            whole_train_label = CoauthorNetwork._prepare_all(dir_path)
            with open(coauthor_pkl, 'wb') as fout:
                pickle.dump(whole_train_label, fout)
        return whole_train_label

    @staticmethod
    def _prepare_all(dir_path):
        articles = DBLP.get_articles(dir_path)
        network, labels = CoauthorNetwork._build_raw_network(articles)
        network = next(nx.connected_component_subgraphs(network))
        train_net = nx.MultiGraph()
        for u, v, data in network.edges(data=True):
            if data['year'] < 2016:
                train_net.add_edge(u, v, year=data['year'])
        network, train_net, labels = CoauthorNetwork._prune_and_rename(
            network, train_net, labels)
        network = CoauthorNetwork._convert_to_simple_graph(network)
        train_net = CoauthorNetwork._convert_to_simple_graph(train_net)
        return network, train_net, labels

    @staticmethod
    def _build_raw_network(articles):
        bt2vn = {
            'ICDE': 0, 'VLDB': 0, 'SIGMOD Conference': 0,
            'KDD': 1, 'ICDM': 1, 'SDM': 1, 'CIKM': 1,
            'SIGIR': 2,
            'IJCAI': 3, 'AAAI': 3, 'ICML': 3, 'NIPS': 3,
            'CVPR': 4, 'ICCV': 4,
            'STOC': 5, 'SODA': 5, 'COLT': 5,
            'ACL': 6, 'EMNLP': 6, 'COLING': 6,
            'SIGCOMM': 7, 'INFOCOM': 7,
            'SOSP': 8, 'OSDI': 8,
            'POPL': 9}
        author_ids = {}
        venue_ids = {}
        network = nx.MultiGraph()
        labels = []
        for paper in articles:
            coauthors = []
            for author in paper.iter('author'):
                if author.text not in author_ids:
                    author_ids[author.text] = len(author_ids)
                coauthors.append(author_ids[author.text])
            year = int(next(paper.iter('year')).text)
            for i in coauthors:
                for j in coauthors:
                    if i < j:
                        network.add_edge(i, j, year=year)
            venue = bt2vn[next(paper.iter('booktitle')).text]
            if venue not in venue_ids:
                venue_ids[venue] = len(venue_ids)
            venue = venue_ids[venue]
            for i in coauthors:
                labels.append((i, venue))
        label_mat = np.zeros((len(author_ids), len(venue_ids)), dtype=np.bool)
        for a, v in labels:
            label_mat[a, v] = True
        return network, label_mat

    @staticmethod
    def _prune_and_rename(network, train_net, labels):
        new_ids = {}
        for i in train_net.nodes():
            if i not in new_ids:
                new_ids[i] = len(new_ids)
        for i in network.nodes():
            if i not in new_ids:
                new_ids[i] = len(new_ids)
        network = nx.relabel_nodes(network, new_ids)
        train_net = nx.relabel_nodes(train_net, new_ids)
        old_ids = -np.ones(network.number_of_nodes(), dtype=np.int32)
        for old_name, new_name in new_ids.items():
            old_ids[new_name] = old_name
        labels = labels[old_ids]
        return network, train_net, labels

    @staticmethod
    def _convert_to_simple_graph(multigraph):
        simple_graph = nx.Graph()
        for u, v in multigraph.edges():
            simple_graph.add_edge(u, v)
        return simple_graph
