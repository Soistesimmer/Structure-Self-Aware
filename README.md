# Structure Self-Aware
 Code for " A Structure Self-Aware Model for Discourse Parsing on Multi-Party Dialogues"

## Dataset

* [Molweni](https://github.com/HIT-SCIR/Molweni). This dataset can be directly preprocessed with our code.
* [STAC](https://www.irit.fr/STAC/corpus.html). This dataset should be formatted first using [data_pre.py](https://github.com/shizhouxing/DialogueDiscourseParsing/blob/master/data_pre.py) provided by Shi and Huang in their work "A Deep Sequential Model for Discourse Parsing on Multi-Party Dialogues".

## Word Vectors

For pre-trained word vectors, this work used [GloVe](https://nlp.stanford.edu/projects/glove/) (200d).

## Run

Run the model without auxiliary losses, please run

```
sh script/train.sh
```

Run our full model with auxiliary losses, please run

```
sh script/full.sh
```

Remember to train the teacher first before adding the ''distill'' command

```
sh script/teacher.sh
```

## Others

* To get the results on STAC reported in our paper, please use the code under Code4STAC. We find that the GNN can't be fully trained due to the small training set. Even the teacher model can't perform well. So, we combine the hidden states from HGRU with learned GNN edges to do prediction. Molweni with more training data won't run into this problem.
* We also provide the example for implementing ELECTRA, which is under the ELECTRA-example dir. 







