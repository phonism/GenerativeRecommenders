"""
Amazon Reviews Dataset
"""
import gzip
import json
import numpy as np
import os
import os.path as osp
import pandas as pd
import gin
import polars as pl
import torch
import random

from collections import defaultdict
from torch_geometric.data import download_google_url
from torch_geometric.data import extract_zip
from torch_geometric.data import HeteroData
from torch_geometric.data import InMemoryDataset
from torch_geometric.io import fs
from typing import Callable
from typing import Optional, List
from einops import rearrange
from sentence_transformers import SentenceTransformer
from generative_recommenders.models.rqvae import RqVae
from generative_recommenders.data.schemas import SeqBatch, FUT_SUFFIX, SeqData
from torch.utils.data import Dataset


class PreprocessingMixin:
    """
    Preprocessing utils
    """
    @staticmethod
    def _process_genres(genres: np.ndarray, one_hot: bool = True) -> np.ndarray:
        """
        Process genres
        """
        if one_hot:
            return genres

        max_genres = genres.sum(axis=1).max()
        idx_list = []
        for i in range(genres.shape[0]):
            idxs = np.where(genres[i, :] == 1)[0] + 1
            missing = max_genres - len(idxs)
            if missing > 0:
                idxs = np.array(list(idxs) + missing * [0])
            idx_list.append(idxs)
        out = np.stack(idx_list)
        return out

    @staticmethod
    def _remove_low_occurrence(
        source_df: pl.DataFrame,
        target_df: pl.DataFrame,
        index_col: str | list[str]
    ) -> pl.DataFrame:
        """
        Remove low occurrence
        """
        if isinstance(index_col, str):
            index_col = [index_col]
        out = target_df.copy()
        for col in index_col:
            count = source_df.groupby(col).agg(ratingCnt=("rating", "count"))
            high_occ = count[count["ratingCnt"] >= 5]
            out = out.merge(high_occ, on=col).drop(columns=["ratingCnt"])
        return out

    @staticmethod
    def _encode_text_feature(
        text_feat: list[str],
        model: Optional[SentenceTransformer] = None
    ) -> torch.Tensor:
        """
        Encode text feature
        """
        embeddings = model.encode(sentences=text_feat, show_progress_bar=True, convert_to_tensor=True).cpu()
        return embeddings
    
    @staticmethod
    def _rolling_window(
        group: pl.DataFrame,
        features: list[str],
        window_size: int = 200,
        stride: int = 1
    ) -> np.ndarray:
        """
        Rolling window
        """
        assert group["userId"].nunique() == 1, "Found data for too many users"
        
        if len(group) < window_size:
            window_size = len(group)
            stride = 1
        n_windows = (len(group) + 1 - window_size) // stride
        feats = group[features].to_numpy().T
        windows = np.lib.stride_tricks.as_strided(
            feats,
            shape=(len(features), n_windows, window_size),
            strides=(feats.strides[0], 8 * stride, 8 * 1)
        )
        feat_seqs = np.split(windows, len(features), axis=0)
        rolling_df = pd.DataFrame({
            name: pd.Series(
                np.split(feat_seqs[i].squeeze(0), n_windows, 0)
            ).map(torch.tensor) for i, name in enumerate(features)
        })
        return rolling_df
    
    @staticmethod
    def _ordered_train_test_split(
        df: pl.DataFrame,
        on: str,
        train_split: float = 0.8
    ) -> pl.DataFrame:
        """
        Ordered train test split
        """
        threshold = df.select(pl.quantile(on, train_split)).item()
        return df.with_columns(is_train=pl.col(on) <= threshold)
    
    @staticmethod
    def _df_to_tensor_dict(
        df: pl.DataFrame,
        features: list[str]
    ) -> dict[str, torch.Tensor]:
        """
        Convert DataFrame to tensor dictionary
        """
        out = {
            feat: torch.from_numpy(
                rearrange(
                    df.select(feat).to_numpy().squeeze().tolist(), "b d -> b d"
                )
            ) if df.select(pl.col(feat).list.len().max() == pl.col(feat).list.len().min()).item()
            else df.get_column("itemId").to_list()
            for feat in features
        }
        fut_out = {
            feat + FUT_SUFFIX: torch.from_numpy(
                df.select(feat + FUT_SUFFIX).to_numpy()
            ) for feat in features
        }
        out.update(fut_out)
        out["userId"] = torch.from_numpy(df.select("userId").to_numpy())
        return out


    @staticmethod
    def _generate_user_history(
        ratings_df: pl.DataFrame,
        features: List[str] = ["movieId", "rating"],
        window_size: int = 200,
        stride: int = 1,
        train_split: float = 0.8,
    ) -> dict[str, torch.Tensor]:
        """
        Generate user history
        """
        if isinstance(ratings_df, pd.DataFrame):
            ratings_df = pl.from_pandas(ratings_df)

        grouped_by_user = (ratings_df
            .sort("userId", "timestamp")
            .group_by_dynamic(
                index_column=pl.int_range(pl.len()),
                every=f"{stride}i",
                period=f"{window_size}i",
                by="userId")
            .agg(
                *(pl.col(feat) for feat in features),
                seq_len=pl.col(features[0]).len(),
                max_timestamp=pl.max("timestamp")
            )
        )
        
        max_seq_len = grouped_by_user.select(pl.col("seq_len").max()).item()
        split_grouped_by_user = PreprocessingMixin._ordered_train_test_split(grouped_by_user, "max_timestamp", 0.8)
        padded_history = (split_grouped_by_user
            .with_columns(pad_len=max_seq_len - pl.col("seq_len"))
            .filter(pl.col("is_train").or_(pl.col("seq_len") > 1))
            .select(
                pl.col("userId"),
                pl.col("max_timestamp"),
                pl.col("is_train"),
                *(pl.when(pl.col("is_train"))
                    .then(
                        pl.col(feat).list.concat(
                            pl.lit(-1, dtype=pl.Int64).repeat_by(pl.col("pad_len"))
                        ).list.to_array(max_seq_len)
                    ).otherwise(
                        pl.col(feat).list.slice(0, pl.col("seq_len") - 1).list.concat(
                            pl.lit(-1, dtype=pl.Int64).repeat_by(pl.col("pad_len") + 1)
                        ).list.to_array(max_seq_len)
                    )
                    for feat in features
                ),
                *(pl.when(pl.col("is_train"))
                    .then(
                        pl.lit(-1, dtype=pl.Int64)
                    )
                    .otherwise(
                        pl.col(feat).list.get(-1)
                    ).alias(feat + FUT_SUFFIX)
                    for feat in features
                )
            )
        )
        
        out = {}
        out["train"] = PreprocessingMixin._df_to_tensor_dict(
            padded_history.filter(pl.col("is_train")),
            features
        )
        out["val"] = PreprocessingMixin._df_to_tensor_dict(
            padded_history.filter(pl.col("is_train").not_()),
            features
        )
        
        return out

def parse(path):
    """
    Parse the data
    """
    g = gzip.open(path, "r")
    for l in g:
        yield eval(l)


class AmazonReviews(InMemoryDataset, PreprocessingMixin):
    """
    Amazon Reviews Dataset
    """
    gdrive_id = "1qGxgmx7G_WB7JE4Cn_bEcZ_o_NAJLE3G"
    gdrive_filename = "P5_data.zip"

    def __init__(
        self,
        root: str,
        split: str = "beauty",
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
        encoder_model_name: str = "sentence-transformers/sentence-t5-xl",
        force_reload: bool = False,
    ) -> None:
        self.split = split
        self.encoder_model_name = encoder_model_name
        super(AmazonReviews, self).__init__(
            root, transform, pre_transform, force_reload
        )
        self.load(self.processed_paths[0], data_cls=HeteroData)
    
    @property
    def raw_file_names(self) -> List[str]:
        """
        Raw file names
        """
        return [self.split]
    
    @property
    def processed_file_names(self) -> str:
        """
        Processed file names
        """
        return f"data_{self.split}.pt"
    
    def download(self) -> None:
        """
        Download the data
        """
        path = download_google_url(self.gdrive_id, self.root, self.gdrive_filename)
        extract_zip(path, self.root)
        os.remove(path)
        folder = osp.join(self.root, "data")
        fs.rm(self.raw_dir)
        os.rename(folder, self.raw_dir)
    
    def _remap_ids(self, x):
        """
        Remap the ids
        """
        return x - 1

    def train_test_split(self, max_seq_len=20):
        """
        Train test split
        """
        splits = ["train", "val", "test"]
        sequences = {sp: defaultdict(list) for sp in splits}
        user_ids = []
        with open(os.path.join(self.raw_dir, self.split, "sequential_data.txt"), "r") as f:
            for line in f:
                parsed_line = list(map(int, line.strip().split()))
                user_ids.append(parsed_line[0])
                items = [self._remap_ids(id) for id in parsed_line[1:]]
                
                # We keep the whole sequence without padding. Allows flexible training-time subsampling.
                train_items = items[:-2]
                sequences["train"]["itemId"].append(train_items)
                sequences["train"]["itemId_fut"].append(items[-2])
                
                eval_items = items[-(max_seq_len + 2): -2]
                sequences["val"]["itemId"].append(eval_items + [-1] * (max_seq_len - len(eval_items)))
                sequences["val"]["itemId_fut"].append(items[-2])
                
                test_items = items[-(max_seq_len + 1): -1]
                sequences["test"]["itemId"].append(test_items + [-1] * (max_seq_len - len(test_items)))
                sequences["test"]["itemId_fut"].append(items[-1])
        
        for sp in splits:
            sequences[sp]["userId"] = user_ids
            sequences[sp] = pl.from_dict(sequences[sp])
        return sequences
    
    def process(self, max_seq_len=20) -> None:
        """
        Process the data
        """
        data = HeteroData()

        with open(os.path.join(self.raw_dir, self.split, "datamaps.json"), 'r') as f:
            data_maps = json.load(f)    

        # Construct user sequences
        sequences = self.train_test_split(max_seq_len=max_seq_len)
        data["user", "rated", "item"].history = {
            k: self._df_to_tensor_dict(v, ["itemId"])
            for k, v in sequences.items() 
        }
        
        # Compute item features
        asin2id = pd.DataFrame([{"asin": k, "id": self._remap_ids(int(v))} for k, v in data_maps["item2id"].items()])
        item_data = (
            pd.DataFrame([
                meta for meta in
                parse(path=os.path.join(self.raw_dir, self.split, "meta.json.gz"))
            ])
            .merge(asin2id, on="asin")
            .sort_values(by="id")
            .fillna({"brand": "Unknown"})
        )

        sentences = item_data.apply(
            lambda row:
                "Title: " +
                str(row["title"]) + "; " +
                "Brand: " +
                str(row["brand"]) + "; " +
                "Categories: " +
                str(row["categories"][0]) + "; " + 
                "Price: " +
                str(row["price"]) + "; ",
            axis=1
        )
        
        model = SentenceTransformer(self.encoder_model_name)
        item_emb = self._encode_text_feature(sentences, model)
        data['item'].x = item_emb
        data['item'].text = np.array(sentences)

        gen = torch.Generator()
        gen.manual_seed(42)
        data['item'].is_train = torch.rand(item_emb.shape[0], generator=gen) > 0.05

        self.save([data], self.processed_paths[0])

@gin.configurable
class P5AmazonReviewsItemDataset(Dataset):
    """
    Old Amazon Reviews Item Dataset
    """
    def __init__(
        self,
        root: str,
        *args,
        train_test_split: str = "all",
        encoder_model_name: str = "sentence-transformers/sentence-t5-xl",
        **kwargs
    ) -> None:
        max_seq_len = 20

        raw_data = AmazonReviews(root=root, *args, encoder_model_name=encoder_model_name, **kwargs)
        
        processed_data_path = raw_data.processed_paths[0]
        if not os.path.exists(processed_data_path):
            raw_data.process(max_seq_len=max_seq_len)
        
        if train_test_split == "train":
            filt = raw_data.data["item"]["is_train"]
        elif train_test_split == "eval":
            filt = ~raw_data.data["item"]["is_train"]
        elif train_test_split == "all":
            filt = torch.ones_like(raw_data.data["item"]["x"][:, 0], dtype=bool)

        self.item_data, self.item_text = raw_data.data["item"]["x"][filt], raw_data.data["item"]["text"][filt]

    def __len__(self):
        return self.item_data.shape[0]

    def __getitem__(self, idx):
        item_ids = torch.tensor(idx).unsqueeze(0) if not isinstance(idx, torch.Tensor) else idx
        x = self.item_data[idx, :768].tolist()
        return x
    
    
@gin.configurable
class P5AmazonReviewsSeqDataset(Dataset):
    """
    Old Amazon Reviews Sequence Dataset
    """
    def __init__(
        self,
        root: str,
        *args,
        train_test_split: str = "train",
        subsample: bool = True,
        force_process: bool = False,
        pretrained_rqvae_path: str = "./out/rqvae/p5_amazon/beauty/checkpoint_299999.pt",
        **kwargs
    ) -> None:
        
        assert (not subsample) or train_test_split == "train", "Can only subsample on training split."

        max_seq_len = 20

        raw_data = AmazonReviews(root=root)

        processed_data_path = raw_data.processed_paths[0]
        if not os.path.exists(processed_data_path) or force_process:
            raw_data.process(max_seq_len=max_seq_len)

        self.subsample = subsample

        rqvae = RqVae(
            input_dim=768,
            embed_dim=32,
            hidden_dims=[512, 256, 128],
            codebook_size=256,
            codebook_kmeans_init=False,
            codebook_normalize=False,
            codebook_sim_vq=False,
            n_layers=3,
            n_cat_features=0,
            commitment_weight=0.25,
        )
        rqvae.load_pretrained(pretrained_rqvae_path)
        rqvae.eval()

        self.sem_ids_list = rqvae.get_semantic_ids(raw_data.data["item"].x).sem_ids.tolist()

        self.sequence_data = raw_data.data[("user", "rated", "item")]["history"][train_test_split]

        self._max_seq_len = max_seq_len
        self.item_data = raw_data.data["item"]["x"]
    
    @property
    def max_seq_len(self):
        """
        Max sequence length
        """
        return self._max_seq_len

    def __len__(self):
        return self.sequence_data["userId"].shape[0]
  
    def __getitem__(self, idx):
        user_ids = self.sequence_data["userId"][idx]
        
        if self.subsample:
            seq = self.sequence_data["itemId"][idx] + self.sequence_data["itemId_fut"][idx].tolist()
            start_idx = random.randint(0, max(0, len(seq) - 3))
            end_idx = random.randint(start_idx + 3, start_idx + self.max_seq_len + 1)
            sample = seq[start_idx:end_idx]
            item_ids = torch.tensor(sample[:-1] + [-1] * (self.max_seq_len - len(sample[:-1])))
            item_ids_fut = torch.tensor([sample[-1]])
        else:
            item_ids = self.sequence_data["itemId"][idx]
            item_ids_fut = self.sequence_data["itemId_fut"][idx]
        
        assert (item_ids >= -1).all(), "Invalid movie id found"
        x = self.item_data[item_ids, :768]
        x[item_ids == -1] = -1

        x_fut = self.item_data[item_ids_fut, :768]
        x_fut[item_ids_fut == -1] = -1

        item_sem_ids = []
        for item_id in item_ids:
            if item_id == -1:
                continue
            item_sem_ids.extend(self.sem_ids_list[item_id])
        target_sem_ids = self.sem_ids_list[item_ids_fut[0]]
        return SeqData(
            user_id=user_ids,
            item_ids=item_sem_ids,
            target_ids=target_sem_ids,
        )

if __name__ == "__main__":
    data = P5AmazonReviewsSeqDataset(root="dataset/amazon", split="beauty")
    print(data[0])
    