import torch
from torch.autograd import Variable
import numpy as np
import math
import heapq

class Helper(object):
    """
        utils class: it can provide any function that we need
    """
    def __init__(self):
        self.timber = True

    def gen_group_member_dict(self, path):
        g_m_d = {}
        with open(path, 'r') as f:
            line = f.readline().strip()
            while line != None and line != "":
                a = line.split('\t')
                g = int(a[0])
                g_m_d[g] = []
                for m in a[1:]:
                    g_m_d[g].append(int(m))
                line = f.readline().strip()
        return g_m_d

    def evaluate_model(self, model, testRatings, testNegatives, K, type_m):
        """
        Evaluate the performance (Hit_Ratio, NDCG) of top-K recommendation
        Return: score of each test rating.
        """
        hits, ndcgs, losses = [], [], []
        
        for idx in range(len(testRatings)):
            loss = self.eval_one_rating(model, testRatings, testNegatives, K, type_m, idx)
            # hits.append(hr)
            # ndcgs.append(ndcg)
            losses.append(loss)
        return losses


    def eval_one_rating(self, model, testRatings, testNegatives, K, type_m, idx):
        rating = testRatings[idx]
        items = testNegatives[idx]
        
        u = rating[0]
        gtItem = rating[1]
        gRating = rating[2]
        
        items.append(gtItem)
        # Get prediction scores
        map_item_score = {}
        users = np.full(len(items), u)

        users_var = torch.from_numpy(users)
        users_var = users_var.long()
        items_var = torch.LongTensor(items)
        if type_m == 'group':
            predictions = model(users_var, None, items_var)
        elif type_m == 'user':
            predictions = model(None, users_var, items_var)
        for i in range(len(items)):
            item = items[i]
            map_item_score[item] = predictions.data.numpy()[i]
        items.pop()
        
        # Evaluate top rank list
        # ranklist = heapq.nlargest(K, map_item_score, key=map_item_score.get)
        
        # hr = self.getHitRatio(ranklist, gtItem)
        # ndcg = self.getNDCG(ranklist, gtItem)
        loss = self.getLoss(map_item_score[item], gRating)
        # print (map_item_score)
        # print (type (map_item_score))
        # print (map_item_score[item], gRating, loss[0])
        # print (stop)
        return loss[0]

    def getHitRatio(self, ranklist, gtItem):
        for item in ranklist:
            if item == gtItem:
                return 1
        return 0

    def getNDCG(self, ranklist, gtItem):
        for i in range(len(ranklist)):
            item = ranklist[i]
            if item == gtItem:
                return math.log(2) / math.log(i+2)
        return 0

    def getLoss(self, score, rating):
        return ((score - rating) ** 2)