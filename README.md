# Generative Recommenders

## RQVAE
```
python generative_recommenders/trainers/rqvae_trainer.py config/rqvae/p5_amazon.gin
```

The code is based on the [RQ-VAE-Recommender](https://github.com/EdoardoBotta/RQ-VAE-Recommender). And following the method proposed in [Adapting Large Language Models by Integrating Collaborative Semantics for Recommendation](https://arxiv.org/pdf/2311.09049), we augment the quantize module with a uniform semantic mapping variant.

## TIGER
```
python generative_recommenders/trainers/tiger_trainer.py config/tiger/p5_amazon.gin
```
The codebase largely follows the original RQ-VAE-Recommender implementation, but we refactored some code and do some upgrade. 

Current benchmark:
|Dataset|Metric|Result|
|---|---|---|
P5 Amazon-Beauty|Recall@10|0.42

## Other Models
We provide early implementations for the following large language model recommenders:
- [LCRec: Adapting Large Language Models by Integrating Collaborative Semantics for Recommendation](https://arxiv.org/pdf/2311.09049)
- [NoteLLM: A Retrievable Large Language Model for Note Recommendation](https://arxiv.org/pdf/2403.01744)
- [COBRA: Sparse Meets Dense: Unified Generative Recommendations with Cascaded Sparse-Dense Representations](https://arxiv.org/pdf/2503.02453)

The training scripts for these models are still being prepared, so they are not ready to run yet.

# Planned Roadmap
## TODO
- **Add More Model:** HSTU, OneRec, etc.
- **Test More Dataset:** Test on more datasets.

# References

[RQ-VAE-Recommender](https://github.com/EdoardoBotta/RQ-VAE-Recommender) by Edoardo Botta.
