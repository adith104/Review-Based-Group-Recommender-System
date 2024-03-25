import torch
from torch.utils.data import DataLoader

from utils.data_reader import get_train_dev_test_data, get_review_dict
from utils.data_set import NarreDataset
from utils.log_hepler import logger
from utils.train_helper import load_model, create_user_ratings_and_groups, train_test_split_user, create_group_ratings, train_test_split_group

train_data, dev_data, test_data = get_train_dev_test_data()
model = load_model("model/checkpoints/NarreModel_20230424112221.pt")
model.config.device = "cpu"
model.to(model.config.device)
loss = torch.nn.MSELoss()

review_by_user, review_by_item = get_review_dict("next")
dataset = NarreDataset(test_data, review_by_user, review_by_item, model.config)
data_iter = DataLoader(dataset, batch_size=model.config.batch_size, shuffle=True)
    
create_user_ratings_and_groups (model, data_iter)
train_test_split_user ()
create_group_ratings ("groupMember.txt", "userRatings.txt")
train_test_split_group ()