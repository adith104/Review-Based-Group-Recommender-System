import math
import time
import random

import itertools
import torch
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from pandas import DataFrame
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader

from model.base_model import BaseModel, BaseConfig
from utils.data_reader import get_review_dict
from utils.data_set import NarreDataset
from utils.log_hepler import logger, add_log_file, remove_log_file
from utils.path_helper import ROOT_DIR


def save_model(model: torch.nn.Module, train_time: time.struct_time):
    path = "model/checkpoints/%s_%s.pt" % (
        model.__class__.__name__, time.strftime("%Y%m%d%H%M%S", train_time)
    )
    path = ROOT_DIR.joinpath(path)
    torch.save(model, path)
    logger.info(f"model saved: {path}")


def load_model(path: str):
    path = ROOT_DIR.joinpath(path)
    # load model to cpu as default.
    model = torch.load(path, map_location=torch.device('cpu'))
    return model


def eval_model(model, data_iter, loss):
    model.eval()
    model_name = 'RGRSModel1'
    config: BaseConfig = model.config
    logger.debug("Evaluating %s..." % model_name)
    
    with torch.no_grad():
        predicts = []
        ratings = []
        for batch_id, iter_i in enumerate(data_iter):
            user_review, user_id, item_id_per_review, item_review, item_id, user_id_per_review, rating = iter_i
            
            user_review = user_review.to(config.device)
            user_id = user_id.to(config.device)
            item_id_per_review = item_id_per_review.to(config.device)

            item_review = item_review.to(config.device)
            item_id = item_id.to(config.device)
            user_id_per_review = user_id_per_review.to(config.device)

            rating = rating.to(config.device)

            predict = model(user_review, user_id, item_id_per_review, item_review, item_id, user_id_per_review)
            
            predicts.append(predict)
            ratings.append(rating)

        predicts = torch.cat(predicts)
        ratings = torch.cat(ratings)
        
        return loss(predicts, ratings).item()
    
    
def create_user_ratings_and_groups (model, data_iter):
    
    model.eval()
    model_name = model.__class__.__name__
    config: BaseConfig = model.config
    logger.debug("Creating dataset %s..." % model_name)
    
    users = {}
    items = {}

    with torch.no_grad():
        predicts = []
        ratings = []
        for batch_id, iter_i in enumerate(data_iter):
            user_review, user_id, item_id_per_review, item_review, item_id, user_id_per_review, rating = iter_i
            
            user_review = user_review.to(config.device)
            user_id = user_id.to(config.device)
            item_id_per_review = item_id_per_review.to(config.device)

            item_review = item_review.to(config.device)
            item_id = item_id.to(config.device)
            user_id_per_review = user_id_per_review.to(config.device)

            rating = rating.to(config.device)
            
            for a, b, c, d, e, f in zip (user_review, user_id, item_id_per_review, item_review, item_id, user_id_per_review):
                
                if b.item() not in users:
                    users[b.item()] = [a, c]
                
                if e.item() not in items:
                    items[e.item()] = [d, f]
        
        with open ('userRatings.txt', 'w') as f1, open ('userRatingNegative.txt', 'w') as f2:
            for user in users.keys():
                
                itemList = list (items.keys ())
                
                random.shuffle (itemList)
                
                selectedItems1 = itemList[:250]
                selectedItems2 = itemList[250:]
                
                a = torch.stack ([users[user][0]] * 250)
                b = torch.tensor (user).repeat (250, 1)
                c = torch.stack ([users[user][1]] * 250)
                d = torch.stack ([items[k][0] for k in selectedItems1])
                e = torch.tensor (selectedItems1)
                e = e.view (-1, 1)
                f = torch.stack ([items[k][1] for k in selectedItems1]) 
                
                predict = model (a, b, c, d, e, f)
                
                for item, res in zip (selectedItems1, predict):
                    # print (user, item, res.item())
                    f1.write (f"{user}\t{item}\t{round (res.item(), 3)}\n")
                    f2.write (f"({user,item})")
                    # print (type (selectedItems2))
                    for i in range (50):
                        f2.write (f"\t{selectedItems2[i]}")
                    f2.write (f"\n")
                    random.shuffle (selectedItems2)
                    # print ('Done')
        
        print (len (users), len (items))
        
        make_groups (list (users.keys()))
        

def train_test_split_user ():
    
    df = pd.read_csv("userRatings.txt", sep='\t', header=None)        
    train_df = df.sample(frac=0.8, random_state=42)
    test_df = df.drop(train_df.index)
    train_df.to_csv('userRatingTrain.txt', sep='\t', header=None, index=False)
    test_df.to_csv('userRatingTest.txt', sep='\t', header=None, index=False)


def train_test_split_group ():
    
    df = pd.read_csv("groupRatings.txt", sep='\t', header=None)        
    train_df = df.sample(frac=0.8, random_state=42)
    test_df = df.drop(train_df.index)
    train_df.to_csv('groupRatingTrain.txt', sep='\t', header=None, index=False)
    test_df.to_csv('groupRatingTest.txt', sep='\t', header=None, index=False)


def create_group_ratings (group_file: str, user_file: str):
    
    df1 = pd.read_csv(group_file, sep='\t', header=None) 
    df2 = pd.read_csv(user_file, sep='\t', header=None) 
    
    with open ('groupRatings.txt', 'w') as f1, open ('groupRatingNegative.txt', 'w') as f2:
    
        for row in df1.iterrows():
            groupID = row[1][0]
            user1 = row[1][1]
            user2 = row[1][2]
            
            # print (groupID, user1, user2)
            user1_items = df2[df2[0] == user1]
            user2_items = df2[df2[0] == user2]
            
            # print (user1_items.head())
            # print (user2_items.head())
            # find the common items between user1_items and user2_items
            common = pd.merge(user1_items, user2_items, on=1)
            
            list1 = user1_items[1].tolist ()
            list2 = user2_items[1].tolist ()
            
            # print (list1)
            # print (list2)
            
            uncommon = list (set (list1).symmetric_difference (set (list2)))
            
            for newRow in common.iterrows():
                
                itemID = int (newRow[1][1])
                user1_rating = newRow[1]["2_x"]
                user2_rating = newRow[1]["2_y"]
                
                group_rating = get_group_rating (user1, user1_rating, user2, user2_rating, itemID)
                f1.write (f"{groupID}\t{itemID}\t{group_rating}\n")
                
                random.shuffle (uncommon)
                
                f2.write (f"({groupID},{itemID})")
                for i in range (50):
                    f2.write (f"\t{uncommon[i]}")
                f2.write ("\n")
                
        
def get_group_rating (user1, user1_rating, user2, user2_rating, itemID):
    
    weight1 = abs (user1 - itemID)
    weight2 = abs (user2 - itemID)
    
    group_rating = (weight1 * user1_rating + weight2 * user2_rating) / (weight1 + weight2)

    num = random.randint (1, 100)
    
    # reduce some ratings as most of them are very high
    if num < 20:
        group_rating = group_rating / 2
    
    return round (group_rating, 3)


def make_groups (users):
    
    group_size = 2
    total_groups = 300
    
    combinations = list (itertools.combinations (users, group_size))
    
    groups = random.sample (combinations, total_groups)

    with open ('groupMember.txt', 'w') as f:
        
        l = list (range (total_groups))
        random.shuffle (l)
        
        for t, num in zip (groups, l):
            f.write (f"{num}")

            for i in t:
                f.write (f"\t{i}")
            f.write (f"\n")
    
    

def train_model(model: BaseModel, train_data: DataFrame, dev_data: DataFrame, is_save_model: bool = True):
    model_name = 'RGRSModel1'
    train_time = time.localtime()
    add_log_file(logger, "log/%s_%s.log" % (model_name, time.strftime("%Y%m%d%H%M%S", train_time)))
    logger.info("Training %s..." % model_name)

    config: BaseConfig = model.config
    logger.info(config.__dict__)
    model.to(config.device)

    opt = torch.optim.Adam(model.parameters(), config.learning_rate, weight_decay=config.l2_regularization)
    lr_s = lr_scheduler.ExponentialLR(opt, gamma=config.learning_rate_decay)
    loss = torch.nn.MSELoss()

    last_progress = 0.
    min_loss = float("inf")

    pin_memory = config.device not in ["cpu", "CPU"]
    review_by_user, review_by_item = get_review_dict("train")
    dataset = NarreDataset(train_data, review_by_user, review_by_item, config)
    train_data_iter = DataLoader(dataset, batch_size=config.batch_size, shuffle=True, pin_memory=pin_memory)
    dataset = NarreDataset(dev_data, review_by_user, review_by_item, config)
    dev_data_iter = DataLoader(dataset, batch_size=config.batch_size, shuffle=True, pin_memory=pin_memory)

    batches_num = math.ceil(len(train_data) / config.batch_size)
    while model.current_epoch < config.num_epochs:

        model.train()

        for batch_id, iter_i in enumerate(train_data_iter):
            user_review, user_id, item_id_per_review, item_review, item_id, user_id_per_review, rating = iter_i

            user_review = user_review.to(config.device)
            user_id = user_id.to(config.device)
            item_id_per_review = item_id_per_review.to(config.device)

            item_review = item_review.to(config.device)
            item_id = item_id.to(config.device)
            user_id_per_review = user_id_per_review.to(config.device)

            rating = rating.to(config.device)

            opt.zero_grad()
            predict = model(user_review, user_id, item_id_per_review, item_review, item_id, user_id_per_review)
            li = loss(predict, rating)
            li.backward()
            opt.step()

            # log progress
            current_batches = model.current_epoch * batches_num + batch_id + 1
            total_batches = config.num_epochs * batches_num
            progress = current_batches / total_batches
            if progress - last_progress > 0.001:
                logger.debug("epoch %d, batch %d, loss: %f (%.2f%%)" %
                             (model.current_epoch, batch_id, li.item(), 100 * progress))
                last_progress = progress

        # complete one epoch
        train_loss = eval_model(model, train_data_iter, loss)
        dev_loss = eval_model(model, dev_data_iter, loss)
        logger.info("Epoch %d complete. Total loss(train/dev)=%f/%f"
                    % (model.current_epoch, train_loss, dev_loss))

        # save best model
        if train_loss < min_loss:
            min_loss = train_loss
            logger.info(f"Get min loss: {train_loss}")
            if is_save_model:
                save_model(model, train_time)

        lr_s.step(model.current_epoch)
        model.current_epoch += 1

    logger.info("%s trained!" % model_name)
    remove_log_file(logger)
    return min_loss
