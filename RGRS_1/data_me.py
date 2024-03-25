from dataclasses import dataclass
from typing import Dict, List

import torch

from pandas import DataFrame
from torch.utils.data import Dataset, DataLoader

from model.narre import NarreConfig
from utils.data_reader import get_review_dict, get_train_dev_test_data, get_max_user_id, get_max_item_id
from utils.log_hepler import logger
from utils.word2vec_helper import PAD_WORD_ID


class NarreDataset(Dataset):
    def __init__(self, data: DataFrame,
                 user_review_dict: Dict[str, DataFrame],
                 item_review_dict: Dict[str, DataFrame],
                 config: NarreConfig):
        """
        Init a NarreDataset.
        :param data: original data. ["userID","itemID","review","rating"]
        :param user_review_dict: the review grouped by userID
        :param item_review_dict: the review grouped by itemID
        :param config: the config of Narre model.
        """

        super().__init__()
        self.data = data
        self.user_review_dict = user_review_dict
        self.item_review_dict = item_review_dict
        self.config = config

        logger.info("Loading dataset...")

        self.user_review, self.user_id, self.item_ids_per_review = self.load_user_review_data()
        self.item_review, self.item_id, self.user_ids_per_review = self.load_item_review_data()

        ratings = self.data["rating"].to_list()
        self.ratings = torch.Tensor(ratings).view(-1, 1)

        logger.info("Dataset loaded.")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx) -> (
            torch.LongTensor,
            torch.LongTensor,
            torch.LongTensor,
            torch.LongTensor,
            torch.LongTensor,
            torch.LongTensor,
            torch.Tensor):

    dataset = NarreDataset(train_data, review_by_user, review_by_item, config)
    loader = DataLoader(dataset, batch_size=128, shuffle=True)
    for user_review, user_id, item_id_per_review, item_review, item_id, user_id_per_review, rating in loader:
        logger.info(
            f"{user_review.shape}, "
            f"{user_id.shape}"
            f"{item_id_per_review.shape}, "
            f"{item_review.shape}, "
            f"{item_id.shape}"
            f"{user_id_per_review.shape}, "
            f"{rating.shape}")
        