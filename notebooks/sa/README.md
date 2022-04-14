# Sentiment Analysis

## Organization

This subfolder contains 4 notebooks, which are similar in content, but each run
a different experiment.

## Experiments

For this task we performed two experiments, one to compare our model with a
baseline and the other to test the strengths and limits of CANINE, a different
dataset was used for each of them. For the first objective, we opted for SST-2,
part of the GLUE benchmark, because its is widely used by the NLP community and
it is thus a good candidate to evaluate and compare the performance of a model.
We also tested the model robustness to increasing levels of typos in this
dataset. For the second objective we opted for Sentiment140, which contains 1.6
million tweets, this choice was due to the fact that it contains a language
register more prone to abbreviations and colloquialisms, in which CANINE has a
theoretical advantage. In both experiments, DistilBERT was used for the
comparisons.
