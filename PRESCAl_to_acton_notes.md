Ideas for implementing PRESCAL with Thompson sampling into acton package:

| Labeller        | Predictor           | Recommender  |
| ------------- |:-------------:| -----:|
| (i, j, k)  -> {0,1} | $$\mathcal{X}^t, E^{t-1}, R_k^{t-1}$$ -> $$[0,1]^{D \times N \times N}$$  | $$[0,1]^{D \times N \times N}$$ ->  (i, j, k)|

* Labeller
 At iteration t, get the index (i, j, k) from recommender
 query label $$x_{ijk} \in \{0,1\}$$ from true graph $$\mathcal{T}$$ 
 add the  $$x_{ijk}$$ to current known graph $$\mathcal{X}^{t}$$

* Predictor
Get new label $$x_{ijk}$$ from labeller and update posterior $$E^t, R_k^t$$
sample from posteriors
predict label probability distribution $$p(x_{ijk}|E^t,R_k^t)$$
send the predictions to recommender

* Recommender
recieve predictions from predictor
recommend (i,j,k) = argmax $$p(x_{ijk}|E^t,R_k^t)$$ 

Considerations:

1. In Thompson sampling, we usually allow the labeller to query label for the same record multiple times (for the purpose of updating posteriors), while in active learning, we usually assume label from the labeller is the ground truth and each record is only queried once. So if we implement the Thompson sampling into acton, whether each record can be queried more than once?
My thought is to allow repeated queries but for the repeated queries the system records the label and do not need to query that from the expert/true graph.
2. The Predictor class converts sklearn predictor. For PRESCAL, the predictor first updates the posterior and then calculate labels using $$e_i^T R_k e_j$$, instead of using any sklearn predictor. So the overriding from Predictor are being implemented. 
3. Analyse features and labels:
  Labels: True graph $$\mathcal{T} \in \{0,1\}^{D \times N \times N}$$, where each label is recorded by its {known, unknown} states.
  Features: $$E^t \in \mathcal{R}^{N \times D}, R_k^t \in \mathcal{R}^{D \times D}$$ (needs to be updated in terms of each iteration t)
 ```python
    #orginal: 
        def read_features(self, ids: Sequence[int]) -> numpy.ndarray
    #update: 
        def read_features(self, ids: Sequence[int]) -> (numpy.ndarray, numpy.ndarray)
```
