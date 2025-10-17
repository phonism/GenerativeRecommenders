"""
Amazon P5 dataset implementation using the new modular architecture
"""
import gzip
import json
import os
import os.path as osp
from typing import Dict, Any, List, Optional, Tuple
import pandas as pd
import torch
import numpy as np
import gin
from torch_geometric.data import download_google_url, extract_zip, HeteroData
from torch_geometric.io import fs
import random

from .base_dataset import BaseRecommenderDataset, ItemDataset, SequenceDataset, DatasetMixin
from .configs import P5AmazonConfig, TextEncodingConfig, SequenceConfig
from .processors import TextProcessor, SequenceProcessor
from .schemas import SeqData


def parse_gzip_json(path: str):
    """Parse gzipped JSON file line by line"""
    with gzip.open(path, "r") as f:
        for line in f:
            yield eval(line)


class P5AmazonDataset(BaseRecommenderDataset, DatasetMixin):
    """
    Amazon P5 dataset with modular architecture
    """
    
    def __init__(self, config: P5AmazonConfig):
        super().__init__(config)
        self.config: P5AmazonConfig = config
        
        # Initialize processors
        self.text_processor = TextProcessor(config.text_config)
        self.sequence_processor = SequenceProcessor(config.sequence_config)
        
    def download(self) -> None:
        """Download and extract P5 dataset"""
        if self._data_exists():
            return
            
        print(f"Downloading P5 dataset from Google Drive...")
        path = download_google_url(
            self.config.gdrive_id, 
            str(self.root_path), 
            self.config.gdrive_filename
        )
        
        print("Extracting dataset...")
        extract_zip(path, str(self.root_path))
        os.remove(path)
        
        # Rename extracted folder
        extracted_folder = self.root_path / "data"
        raw_folder = self.root_path / "raw"
        
        if raw_folder.exists():
            fs.rm(str(raw_folder))
        os.rename(str(extracted_folder), str(raw_folder))
        
        print("Download and extraction completed.")
    
    def _data_exists(self) -> bool:
        """Check if raw data already exists"""
        raw_dir = self.root_path / "raw" / self.config.split
        return raw_dir.exists() and (raw_dir / "sequential_data.txt").exists()
    
    def load_raw_data(self) -> Dict[str, Any]:
        """Load raw data files"""
        raw_dir = self.root_path / "raw" / self.config.split
        
        # Load data mappings
        datamaps_path = raw_dir / "datamaps.json"
        with open(datamaps_path, 'r') as f:
            data_maps = json.load(f)
        
        # Load sequential data
        sequences = self._load_sequential_data(raw_dir / "sequential_data.txt")
        
        # Load item metadata
        items_metadata = self._load_item_metadata(
            raw_dir / "meta.json.gz", 
            data_maps["item2id"]
        )
        
        return {
            "data_maps": data_maps,
            "sequences": sequences,
            "items_metadata": items_metadata
        }
    
    def _load_sequential_data(self, filepath: str) -> List[List[int]]:
        """Load sequential interaction data"""
        sequences = []
        
        with open(filepath, "r") as f:
            for line in f:
                sequence = list(map(int, line.strip().split()))
                # Remap IDs to 0-based indexing
                user_id = sequence[0]
                item_ids = [self._remap_id(id) for id in sequence[1:]]
                sequences.append([user_id] + item_ids)
        
        return sequences
    
    def _load_item_metadata(self, filepath: str, item2id_map: Dict[str, str]) -> pd.DataFrame:
        """Load and process item metadata"""
        # Create ASIN to ID mapping
        asin2id = pd.DataFrame([
            {"asin": k, "id": self._remap_id(int(v))} 
            for k, v in item2id_map.items()
        ])
        
        # Load metadata
        metadata_list = list(parse_gzip_json(filepath))
        items_df = pd.DataFrame(metadata_list)
        
        # Merge with ID mapping and process
        items_df = (
            items_df
            .merge(asin2id, on="asin", how="inner")
            .sort_values("id")
            .reset_index(drop=True)
        )
        
        # Clean and standardize fields
        items_df = self._clean_metadata(items_df)
        
        return items_df
    
    def _clean_metadata(self, items_df: pd.DataFrame) -> pd.DataFrame:
        """Clean and standardize item metadata"""
        # Fill missing values
        items_df["brand"] = items_df["brand"].fillna("Unknown")
        items_df["title"] = items_df["title"].fillna("Unknown Title")
        items_df["price"] = items_df["price"].fillna("Unknown")
        
        # Process categories - take first category if available
        def process_categories(cats):
            if isinstance(cats, list) and len(cats) > 0:
                if isinstance(cats[0], list) and len(cats[0]) > 0:
                    return cats[0][0]
                elif isinstance(cats[0], str):
                    return cats[0]
            return "Unknown"
        
        items_df["categories"] = items_df["categories"].apply(process_categories)
        
        return items_df
    
    def _remap_id(self, original_id: int) -> int:
        """Remap original IDs to 0-based indexing"""
        return original_id - 1
    
    def preprocess_data(self, raw_data: Dict[str, Any]) -> Dict[str, Any]:
        """Preprocess raw data into structured format"""
        # Create interactions DataFrame
        interactions_df = self._create_interactions_dataframe(raw_data["sequences"])
        
        # Filter low-interaction users/items if configured
        if self.config.processing_config.min_user_interactions > 1:
            interactions_df = self.filter_low_interactions(
                interactions_df,
                user_col="user_id",
                item_col="item_id", 
                min_user_interactions=self.config.processing_config.min_user_interactions,
                min_item_interactions=self.config.processing_config.min_item_interactions
            )
        
        # Process item features
        items_df = self._process_item_features(raw_data["items_metadata"])
        
        # Create train/test splits for items
        items_df = self._add_item_splits(items_df)
        
        return {
            "interactions": interactions_df,
            "items": items_df,
            "metadata": raw_data["data_maps"]
        }
    
    def _create_interactions_dataframe(self, sequences: List[List[int]]) -> pd.DataFrame:
        """Convert sequential data to interactions DataFrame"""
        interactions = []
        
        for sequence in sequences:
            user_id = sequence[0]
            item_ids = sequence[1:]
            
            for timestamp, item_id in enumerate(item_ids):
                interactions.append({
                    "user_id": user_id,
                    "item_id": item_id,
                    "timestamp": timestamp,
                    "rating": 1.0  # Implicit feedback
                })
        
        return pd.DataFrame(interactions)
    
    def _process_item_features(self, items_metadata: pd.DataFrame) -> pd.DataFrame:
        """Process item features using text processor"""
        # Generate text embeddings
        cache_key = f"p5_amazon_{self.config.split}"
        embeddings = self.text_processor.encode_item_features(
            items_metadata,
            cache_key=cache_key,
            force_reload=self.config.force_reload
        )
        
        # Add embeddings to dataframe
        items_df = items_metadata.copy()
        items_df["features"] = embeddings.tolist()
        
        # Create text field for reference
        texts = []
        for _, row in items_metadata.iterrows():
            text = self.config.text_config.format_text(
                title=row.get("title", "Unknown"),
                brand=row.get("brand", "Unknown"),
                categories=row.get("categories", "Unknown"),
                price=row.get("price", "Unknown")
            )
            texts.append(text)
        
        items_df["text"] = texts
        
        return items_df
    
    def _add_item_splits(self, items_df: pd.DataFrame) -> pd.DataFrame:
        """Add train/eval splits for items"""
        # Simple random split (can be made more sophisticated)
        np.random.seed(42)
        is_train = np.random.rand(len(items_df)) > 0.05  # 95% train, 5% eval
        items_df["is_train"] = is_train
        
        return items_df
    
    def extract_items(self, processed_data: Dict[str, Any]) -> pd.DataFrame:
        """Extract item information"""
        return processed_data["items"]
    
    def extract_interactions(self, processed_data: Dict[str, Any]) -> pd.DataFrame:
        """Extract user-item interactions"""
        return processed_data["interactions"]


@gin.configurable
class P5AmazonItemDataset(ItemDataset):
    """
    Item dataset for P5 Amazon (for RQVAE training)
    """
    
    def __init__(
        self,
        root: str,
        split: str = "beauty",
        train_test_split: str = "all",
        encoder_model_name: str = "sentence-transformers/sentence-t5-xl",
        force_reload: bool = False,
        **kwargs
    ):
        # Create configuration
        config = P5AmazonConfig(
            root_dir=root,
            split=split,
            force_reload=force_reload,
            text_config=TextEncodingConfig(encoder_model=encoder_model_name)
        )
        
        # Create base dataset
        base_dataset = P5AmazonDataset(config)
        
        # Initialize item dataset
        super().__init__(
            base_dataset=base_dataset,
            split=train_test_split,
            return_text=False
        )
    
    def __getitem__(self, idx: int) -> List[float]:
        """Return item features as list (backward compatibility)"""
        features = super().__getitem__(idx)
        if isinstance(features, torch.Tensor):
            return features[:768].tolist()
        return features


@gin.configurable
class P5AmazonSequenceDataset(SequenceDataset):
    """
    Sequence dataset for P5 Amazon (for TIGER training)
    """
    
    def __init__(
        self,
        root: str,
        split: str = "beauty",
        train_test_split: str = "train",
        subsample: bool = True,
        force_process: bool = False,
        pretrained_rqvae_path: Optional[str] = None,
        max_seq_len: int = 20,
        **kwargs
    ):
        # Create configuration
        config = P5AmazonConfig(
            root_dir=root,
            split=split,
            force_reload=force_process,
            sequence_config=SequenceConfig(max_seq_len=max_seq_len)
        )
        
        # Load semantic encoder if provided
        semantic_encoder = None
        if pretrained_rqvae_path and os.path.exists(pretrained_rqvae_path):
            from ..models.rqvae import RqVae
            semantic_encoder = RqVae(
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
            semantic_encoder.load_pretrained(pretrained_rqvae_path)
            semantic_encoder.eval()
        
        # Create base dataset
        base_dataset = P5AmazonDataset(config)
        
        # Initialize sequence dataset
        super().__init__(
            base_dataset=base_dataset,
            split=train_test_split,
            semantic_encoder=semantic_encoder,
            subsample=subsample
        )


