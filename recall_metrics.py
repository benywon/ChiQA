# -*- coding: utf-8 -*-
import sys 
import json 
import math 
from sklearn.metrics import accuracy_score, precision_recall_fscore_support


def read_data_f1(data_file, pred_file):
    data_lines = open(data_file, 'r', encoding='utf-8').readlines()
    pred_lines = open(pred_file, 'r', encoding='utf-8').readlines()
    assert len(data_lines) == len(pred_lines)
    labels, preds = [], []
    for line in data_lines:
        items = line.strip().split('\t')
        # query, url, label, img_path
        labels.append(int(float(items[2])))
    for line in pred_lines:
        p = float(line.strip())
        if p <= 0.333:
            preds.append(0)
        elif p < 0.667:
            preds.append(1)
        else:
            preds.append(2)
    return labels, preds 


def compute_precision_recall_f1_acc(preds, labels, average='binary'):
    acc = accuracy_score(labels, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average=average)

    return {
        'accuracy': acc,
        'F1': f1,
        'precision': precision,
        'recall': recall,
    }


def compute_f1(data_file, pred_file):
    labels, preds = read_data_f1(data_file, pred_file) 
    res = compute_precision_recall_f1_acc(preds, labels, average='macro')
    # print(res)
    return res 


def compute_binary_f1_fn(preds, labels):
    assert len(preds) == len(labels)
    accuracy = sum([1 if a==b else 0 for a,b in zip(preds, labels)]) / len(preds)
    if sum(preds) == 0:
        precision = 0
    else:
        precision = sum([1 if a==b and a==1 else 0 for a,b in zip(preds, labels)]) / sum(preds)
    if sum(labels) == 0:
        recall = 0
    else:
        recall = sum([1 if a==b and a==1 else 0 for a,b in zip(preds, labels)]) / sum(labels)
    if precision + recall == 0:
        f1 = 0
    else:
        f1 = 2 * precision * recall / (precision + recall)
    return {
        'accuracy': accuracy,
        'F1': f1,
        'precision': precision,
        'recall': recall
    }


def compute_binary_f1(data_file, pred_file):
    labels = [float(line.strip().split('\t')[2]) for line in open(data_file, 'r', encoding='utf-8').readlines()] # query, url, label, img_path
    labels = [1 if d > 1.5 else 0 for d in labels]
    probs = [float(line.strip()) for line in open(pred_file, 'r', encoding='utf-8').readlines()]
    step = 0.001
    thres = 0.001
    max_res = None 
    while thres < 1:
        preds = [1 if p >= thres else 0 for p in probs]
        res = compute_binary_f1_fn(preds, labels)
        res['threshold'] = thres 
        if max_res is None or res['F1'] > max_res['F1']:
            max_res = res 
        thres += step 
    # print(max_res)
    return max_res 


def read_data(data_file, pred_file):
    data_lines = open(data_file, 'r', encoding='utf-8').readlines()
    pred_lines = open(pred_file, 'r', encoding='utf-8').readlines()
    assert len(data_lines) == len(pred_lines)
    # query, url, label, img_path / score
    query2data = {}
    for d_line, p_line in zip(data_lines, pred_lines):
        d_items = d_line.strip().split('\t')
        query = d_items[0]
        if query not in query2data:
            query2data[query] = []
        query2data[query].append([float(d_items[2]), float(p_line.strip())]) # label, pred
    return query2data 


def compute_dcg(arr):
    dcg = 0
    for i, d in enumerate(arr):
        dcg += (2 ** d - 1) / math.log2(i+2)
    return dcg 


def compute_ndcg_n(data_file, pred_file):
    query2data = read_data(data_file, pred_file)
    # print('size of query2data: ', len(query2data))
    num_q = 0
    ndcg = {1: 0, 3: 0, 5: 0}
    for q in query2data:
        dcg = {n: [0, 0] for n in ndcg}
        data = query2data[q]
        num_q += 1
        sort_data = sorted(data, key=lambda p: p[1], reverse=True) # [[labe, score], ...]
        arr = [d[0] for d in sort_data]
        for n in dcg:
            dcg[n][0] += compute_dcg(arr[:n])
        sort_data = sorted(data, key=lambda p: p[0], reverse=True) # [[labe, score], ...]
        arr = [d[0] for d in sort_data]
        for n in dcg:
            dcg[n][1] += compute_dcg(arr[:n])
        for n in ndcg:
            if dcg[n][1] == 0:
                ndcg[n] += 0
            else:
                ndcg[n] += (dcg[n][0] / dcg[n][1])
    # ndcg_str = json.dumps({'NDCG@{}'.format(n): ndcg[n] / num_q for n in ndcg })
    # print(ndcg_str)
    ndcg_d = {'NDCG@{}'.format(n): ndcg[n] / num_q for n in ndcg }
    ndcg_d['num_query'] = len(query2data)
    return  ndcg_d 


def compute_map_n(data_file, pred_file):
    query2data = read_data(data_file, pred_file)
    # {'xx': [[xx, xx], [xx, xx], ...], ...}
    num_q = 0
    mapn = {1: 0, 3: 0, 5: 0}
    for q in query2data:
        data = query2data[q]
        arr = sorted(data, key=lambda p: p[1], reverse=True)
        for n in mapn:
            mapn[n] += compute_map(arr, n)
        num_q += 1
    map_d = {'MAP@{}'.format(n): mapn[n] / num_q for n in mapn }
    map_d['num_query'] = len(query2data)
    return map_d 


def compute_map(arr, n):
    n_one = 0
    s = 0
    for i, p in enumerate(arr): # [[labe, pred], ..]
        if p[0] > 1.5:
            n_one += 1
            s += n_one / (i+1)
        if (i+1) >= n:
            break 
    if n_one == 0:
        return 0
    s /= n_one
    return s 


def compute_recall_n(data_file, pred_file):
    query2data = read_data(data_file, pred_file)
    # print('size of query2data: ', len(query2data))
    num_q = 0
    recall = {1: 0, 3: 0, 5: 0}
    for q in query2data:
        data = query2data[q]
        arr = sorted(data, key=lambda p: p[1], reverse=True)
        for n in recall:
            recall[n] += compute_recall(arr, n)
        num_q += 1
    # r_str = json.dumps({'R@{}'.format(n): recall[n] / num_q for n in recall })
    # print(r_str)
    r_d = {'R@{}'.format(n): recall[n] / num_q for n in recall }
    r_d['num_query'] = len(query2data)
    return r_d 


def compute_recall(arr, n=1):
    for d in arr[:n]: # arr -> [[label, score], ...]
        if d[0] > 1.5:
            return 1
    return 0

if __name__ == "__main__":

    ndcg = compute_ndcg_n(sys.argv[1], sys.argv[2])
    recall = compute_recall_n(sys.argv[1], sys.argv[2])
    mapn = compute_map_n(sys.argv[1], sys.argv[2])
    macro_f1 = compute_f1(sys.argv[1], sys.argv[2])
    binary_f1 = compute_binary_f1(sys.argv[1], sys.argv[2])
    res = {
        'data_file': sys.argv[1],
        'pred_file': sys.argv[2],
        'ndcg': ndcg,
        'recall': recall,
        'map': mapn,
        'macro_f1': macro_f1,
        'binary_f1': binary_f1 
    }
    print(json.dumps(res, ensure_ascii=False))