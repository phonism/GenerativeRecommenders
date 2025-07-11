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

# Planned Roadmap (EN)
## TODO
- **Add More Model:** HSTU, LCRec, Cobra, OneRec, etc.
- **Test More Dataset:** Test on more datasets.

# References

[RQ-VAE-Recommender](https://github.com/EdoardoBotta/RQ-VAE-Recommender) by Edoardo Botta.