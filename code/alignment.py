from data_loader import DataLoader
import os
import time, datetime
import numpy as np
from PIL import Image
from regression_model import *
import pandas as pd
import itertools
import random
from tqdm import tqdm


class Enrichment(object):

    def __init__(self, m, path, x_list, y, t=2, delta=0.1, theta=100, size=1000):
        self.raw_data = None
        self.m = m # time series dimension
        self.size = size
        self.T_list = []
        self.theta = theta  # time constraint parameter
        self.delta = delta  # model constraint parameter
        self.model = None
        self.t = t # optimal parameter
        self.x_list = x_list
        self.y = y
        if m == 3:
            self.preprocessing_3_dim(path)
        else:
            self.preprocessing(path)

    def preprocessing_3_dim(self, path):
        # old preprocessing
        dataloader = DataLoader()
        self.raw_data = None
        X = []
        Y = []
        Z = []
        self.raw_data = dataloader.get_data(path)
        for r in self.raw_data[1:]:
            ts = self.time2ts(r[1])
            if r[2] != "":
                X.append([ts,float(r[2])])
            elif r[3] != "":
                Y.append([ts, float(r[3])])
            elif r[4] != "":
                Z.append([ts, float(r[4])])
        self.T_list = [X, Y, Z]
        if self.size != 0:
            for idx in range(len(self.T_list)):
                if len(self.T_list[idx]) > self.size:
                    self.T_list[idx] = self.T_list[idx][:self.size]


    def preprocessing(self, path):
        data = pd.read_csv(path)
        self.T_list =[]
        for i in range(self.m):
            self.T_list.append([])
        for idx in range(data.shape[0]):
            for col in range(0, self.m):
                self.T_list[col].append([data.iloc[idx,col], float(data.iloc[idx,col+self.m])])
        if self.size != 0:
            for idx in range(len(self.T_list)):
                if len(self.T_list[idx]) > self.size:
                    self.T_list[idx] = self.T_list[idx][:self.size]

    def generate_time_candidate(self):
        candidate_list = []
        candidate_list_encoding = []
        theta = self.theta
        for x in range(self.m):
            # choose x to perform
            pointer_start = [0] * self.m
            pointer_end = [0] * self.m
            for idx in range(0, len(self.T_list[x])):
                ts = int(self.T_list[x][idx][0])
                stop_flag = False
                for y in range(self.m):
                    if y == x:
                        continue
                    start = pointer_start[y]
                    end = pointer_end[y]
                    length = len(self.T_list[y])
                    while start < length and int(self.T_list[y][start][0]) < ts:
                        start += 1
                    while end < length and int(self.T_list[y][end][0]) < ts + theta:
                        end += 1
                    pointer_start[y] = start
                    pointer_end[y] = end
                    if start >= length:
                        stop_flag = True

                candidates = [[0] * (self.m*2)]
                candidates_encoding = [[0] * self.m]
                candidates[0][2*x] = self.T_list[x][idx][0]
                candidates[0][2*x+1] = self.T_list[x][idx][1]
                candidates_encoding[0][x] = idx

                for tmp in range(self.m):
                    if tmp == x:
                        continue
                    start = pointer_start[tmp]
                    end = pointer_end[tmp]
                    candidates_ = []
                    candidates_encoding_ = []
                    for ptr in range(start, end):
                        for c in candidates:
                            c_ = c.copy()
                            c_[2*tmp] = self.T_list[tmp][ptr][0]
                            c_[2*tmp+1] = self.T_list[tmp][ptr][1]
                            candidates_.append(c_)
                        for ce in candidates_encoding:
                            ce_ = ce.copy()
                            ce_[tmp] = ptr
                            candidates_encoding_.append(ce_)
                    candidates = candidates_
                    candidates_encoding = candidates_encoding_

                candidate_list += candidates
                candidate_list_encoding += candidates_encoding

                if stop_flag:
                    break

        ## sort
        candidate_list_combine = [(candidate_list[i], candidate_list_encoding[i]) for i in range(len(candidate_list))]
        candidate_list_combine = sorted(candidate_list_combine, key=lambda x_in: x_in[1])
        candidate_list = [c[0] for c in candidate_list_combine]
        candidate_list_encoding = [c[1] for c in candidate_list_combine]
        return candidate_list, candidate_list_encoding

    def time2ts(self, s):
        timeArray = time.strptime(s, "%Y-%m-%d %H:%M:%S")
        try:
            ts = int(time.mktime(timeArray))
        except:
            print(s)
        return ts

    def enrichment(self, ignore_delta=False):
        candidate_list, candidate_list_encoding = self.generate_time_candidate()
        if self.model and not ignore_delta:
            candidate_list, candidate_list_encoding = self.model_constraint(candidate_list, candidate_list_encoding)

        if VERBOSE:
            print("|R_c|: {}".format(len(candidate_list)))

        idx_dict = [] # for every dimension, idx_dict[dim][encoding] = index list
        intersect_dict = {}
        for dim in range(self.m):
            idx_dict.append({})
        for idx, c in enumerate(candidate_list_encoding):
            for dim in range(self.m):
                self.add_dict_a_with_b(idx_dict[dim], c[dim], idx)
        for dim in range(self.m):
            dim_dict = idx_dict[dim]
            for key in dim_dict:
                if len(dim_dict[key]) >= 1:
                    for i in range(len(dim_dict[key])):
                        for j in range(i+1, len((dim_dict[key]))):
                            c1 = dim_dict[key][i]
                            c2 = dim_dict[key][j]
                            self.add_dict_a_with_b(intersect_dict, c1, c2)
                            self.add_dict_a_with_b(intersect_dict, c2, c1)

        for key in intersect_dict:
            intersect_dict[key] = set(intersect_dict[key])

        # greedy init
        R_sm = set()
        dislike_set = set()
        for idx, c in enumerate(candidate_list_encoding):
            if idx not in dislike_set:
                R_sm.add(idx)
                if idx not in intersect_dict:
                    continue
                inter_points = intersect_dict[idx]
                for p in inter_points:
                    dislike_set.add(p)
        if VERBOSE:
            print("init |R_sm|: {}".format(len(R_sm)))

        # pho optimal local search
        optimal = False

        '''local search'''
        # while not optimal:
        #     optimal = True
        #     for p in range(2, self.t+1):
        #         R_out = [idx for idx in range(0, len(candidate_list_encoding)) if idx not in R_sm]
        #         R_s_list = list(itertools.combinations(R_out, p))  # find all subsets with size p
        #         print("R_s_list")
        #         for idx in tqdm(range(len(R_s_list))):
        #             R_s = R_s_list[idx]
        #             joint_set = intersect_dict[R_s[0]]
        #             for r in R_s[1:]:
        #                joint_set = joint_set.union(intersect_dict[r])
        #             # check overlap
        #             if joint_set.isdisjoint(R_s):
        #                 R_overlap = joint_set.intersection(joint_set)
        #                 if len(R_overlap) < p:
        #                     R_sm = R_sm - R_overlap
        #                     R_sm = R_sm.union(R_s)
        #                     optimal = False
        #                     break

        '''ours optimization and pruning'''
        while not optimal:
            optimal = True
            for q in range(1, self.t):
                R_overlap_list = list(itertools.combinations(R_sm, q))
                for idx in range(len(R_overlap_list)):
                # for idx in tqdm(range(len(R_overlap_list))):
                    R_overlap = R_overlap_list[idx]
                    # pruning
                    if q > 1 and not self.checkMinMaxTime(q, R_overlap, candidate_list):
                        continue
                    if R_overlap[0] in intersect_dict.keys():
                        joint_set = intersect_dict[R_overlap[0]]
                    else:
                        joint_set = set()
                    for r in R_overlap[1:]:
                        if r in intersect_dict.keys():
                            joint_set = joint_set.union(intersect_dict[r])
                    # check
                    if len(joint_set) > q:
                        R_replace = set()
                        for r in joint_set:
                            if intersect_dict[r].intersection(R_sm).issubset(R_overlap) and not intersect_dict[r].intersection(R_replace):
                                R_replace.add(r)
                        if len(R_replace) > q:
                            R_sm = R_sm - set(R_overlap)
                            R_sm = R_sm.union(R_replace)
                            optimal = False
                            break

        return [candidate_list[idx] for idx in R_sm], [candidate_list_encoding[idx] for idx in R_sm]

    def checkMinMaxTime(self, q, R_input, candidate_list):
        # prune
        min_time = 1e20
        max_time = 0
        for r_ in R_input:
            for idx in range(self.m):
                ts = int(candidate_list[r_][idx*2])
                if ts < min_time:
                    min_time = ts
                if ts > max_time:
                    max_time = ts
        return (max_time - min_time) < (2*q+1)*self.theta

    def add_dict_a_with_b(self, tar_dict, a, b):
        if a not in tar_dict:
            tar_dict[a] = [b]
        else:
            if b not in tar_dict[a]:
                tar_dict[a].append(b)

    def model_constraint(self, candidate_list, candidate_list_encoding):
        result_list = []
        result_list_encoding = []
        input_data = []
        input_data = np.asarray(candidate_list).astype(float)
        predict_output = self.model.predict(input_data[:, self.x_list])
        for idx, c in enumerate(candidate_list):
            if abs(predict_output[idx] - input_data[idx,self.y]) <= self.delta:
                result_list.append(candidate_list[idx])
                result_list_encoding.append(candidate_list_encoding[idx])
        return result_list, result_list_encoding

    def training(self, train_X, train_Y, model_name="SVM", seed=1234):
        if model_name == 'Linear':
            model = LinearRegression()
        elif model_name == 'DecisionTree':
            model = DecisionTreeRegressor()
        elif model_name == 'SVM':
            model = SVR(C=1.0, epsilon=0.2, gamma='auto')
        elif model_name == 'KNN':
            model = KNeighborsRegressor(n_neighbors=5)
        elif model_name == 'RandomForest':
            model = RandomForestRegressor(max_depth=2, random_state=seed)
        elif model_name == 'NN':
            model = MLPRegressor(max_iter=500, random_state=seed)
        elif model_name == 'SGD':
            model = SGDRegressor(max_iter=1000, tol=1e-3, random_state=seed)
        elif model_name == 'Ada':
            model = AdaBoostRegressor(n_estimators=100, random_state=seed)
        elif model_name == "XGB":
            model = XGBRegressor(n_estimators=100, random_state=seed)
        else:
            raise ValueError('Wrong model name.')
        model.fit(train_X, train_Y)
        self.model = model

def f1score(enrichment, result_encoding):
    # calculate f1-score
    total = len(enrichment.T_list[0])
    find = len(result_encoding)
    count = 0
    for r in result_encoding:
        if r[0] == r[1] and r[0] == r[2]:
            count += 1

    recall = count/total
    precision = count/find
    f1 = 2 * precision * recall / (recall + precision)
    if VERBOSE:
        print(f"recall: {recall}")
        print(f"precision: {precision}")
    print(f"F1: {f1}")
    return f1

def alignment_test(parameter, dataset, model, size=1000, t=2):

    file_path = os.path.join(parameter['path'], parameter['file'])
    m = parameter['m']
    theta = parameter['theta_time']
    delta = parameter['delta_model'][model]

    # method parameters
    if VERBOSE:
        print("first round")
    enrichment = Enrichment(m=m, path=file_path, t=t, x_list=parameter['x_list'], y=parameter['y'], delta=delta, theta=theta, size=size)
    t1 = time.time()
    result, result_encoding = enrichment.enrichment()
    t2 = time.time()
    if VERBOSE:
        print("Result length: {}".format(len(result)))
        print("time: {}".format(t2-t1))
    f1 = f1score(enrichment, result_encoding)

    # model training
    if VERBOSE:
        print("model training")
    result = np.array(result)
    train_X, train_Y = result[:,parameter['x_list']], result[:, parameter['y']]
    enrichment.training(train_X, train_Y, model_name=model)

    # predict again
    if VERBOSE:
        print("second round")
    result, result_encoding = enrichment.enrichment()
    if VERBOSE:
        print(len(result))
    f1 = f1score(enrichment, result_encoding)

    result = pd.DataFrame(result)
    result.columns = list(range(0, 2*m))
    result = result.sort_values(by=0)
    output_dir = os.path.join(f'../aligned-result/{dataset}',f'samc-{model}-{size}')
    if not os.path.exists(f'../aligned-result/{dataset}'):
        os.mkdir(f'../aligned-result/{dataset}')
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    output_path = os.path.join(output_dir, '0.csv')
    result.to_csv(output_path, index=False, header=None)
    return f1


if __name__ == '__main__':
    VERBOSE = True
    # input files

    dataset = 'household'
    t = 2
    paras = {
        'telemetry': {
            'path': "../telemetry/",
            'file': "telemetry_multi_timestamp.csv",
            'm': 4,
            'theta_time': 80,
            'model_list': ["Linear", "XGB", "RandomForest"],
            'delta_model': {
                "Linear": 0.35,
                "XGB": 0.35,
                "RandomForest": 0.35,
            },
            'x_list': [1, 3, 5],
            'y': 7,
            'size_list': [1000,2000,3000],
        },
        'household':{
            'path': "../household/",
            'file': "household_multi_timestamp.csv",
            'm': 4,
            'theta_time': 80,
            # 'model_list': ["SVM"],
            'model_list':["SVM", "Linear", "NN", "XGB", "RandomForest"],
            'delta_model': {
                "SVM": 8,
                "Linear": 8,
                "NN": 8,
                "XGB": 8,
                "RandomForest": 8,
            },
            'x_list': [1,3,7],
            'y': 5,
            'size_list': [2000],
        },
        'fuel':{
            'path': "../enrichData-30/",
            'file': sorted(os.listdir("../enrichData-30/"))[1],
            'm': 3,
            'theta_time': 80,
            'model_list': ["SVM", "Linear", "XGB", "RandomForest"],
            # 'model_list': ["XGB"],
            'delta_model': {
                "SVM": 25,
                "Linear": 25,
                "XGB": 25,
                "RandomForest": 25,
            },
            'x_list': [1,3],
            'y': 5,
            'size_list': [500, 1000, 5000, 10000],
        },

    }
    parameter = paras[dataset]
    print(parameter['file'])
    size_list = parameter['size_list']
    df_f1 = pd.DataFrame(index=size_list, columns=parameter['model_list'])

    for idx, model in enumerate(parameter['model_list']):
        for size in size_list:
            if VERBOSE:
                print(f"*********{model}-{size}************")
            f1 = alignment_test(parameter, dataset, model=model, size=size, t=t)
            df_f1[f'{model}'][size] = f1

    out_dir = f"../aligned-accuracy/{dataset}"
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    version = ""
    df_f1.to_csv(os.path.join(out_dir, f"samc-{parameter['theta_time']}{version}.csv"))


