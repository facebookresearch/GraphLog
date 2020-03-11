Advanced Usage
==============

GraphLog provides an array of datasets, thus making it a perfect
candidate to test multi-task, continual, and meta-learning in graphs.
Each dataset is derived by its own set of **rules**.

Similarity
----------

Two datasets can have highly overlapping rules to highly non-overlapping
rules. This provides GraphLog a unique way to define the notion of task
**similarity**. Two datasets are highly similar if the underlying rules
are similar.

.. code:: ipython3

    from graphlog import GraphLog
    gl = GraphLog()

First, let's get the available datasets in GraphLog

.. code:: ipython3

    datasets = gl.get_dataset_names_by_split()

.. code:: ipython3

    datasets["train"][0]
    
    >> 'rule_3'



To calculate dataset similarity, we compute the overlap between the
actual rules used in the datasets. GraphLog provides an easy API to do
so.

.. code:: ipython3

    gl.compute_similarity("rule_0","rule_1")
    
    >> 0.95


We see that the datasets ``rule_0`` and ``rule_1`` are 95% similar. To
get top 10 similar datasets as of ``rule_0``, we can call the following
method:

.. code:: ipython3

    gl.get_most_similar_datasets("rule_0",10)

    >> [('rule_0', 1.0),
     ('rule_1', 0.95),
     ('rule_2', 0.9),
     ('rule_3', 0.85),
     ('rule_4', 0.8),
     ('rule_5', 0.75),
     ('rule_6', 0.7),
     ('rule_7', 0.65),
     ('rule_8', 0.6),
     ('rule_9', 0.55)]



MultiTask training
------------------

By providing an easy way to extract datasets and also by grouping them
in terms of similarity, we can easily train and in a multi-task
scenario. Below we provide a dummy snippet to do so.

.. code:: ipython3

    data_ids = gl.get_most_similar_datasets("rule_0",10)
    for epoch in range(100):
        dataset = gl.get_dataset_by_name(random.choice(data_ids))
        train_loader = gl.get_dataloader_by_mode(dataset, "train")
        for batch_id, batch in enumerate(train_loader):
            graphs = batch.graphs
            queries = batch.queries
            labels = batch.targets
            logits = your_model(graphs, queries)

Difficulty
----------

GraphLog also provides an additional option of categorizing each dataset
on their relative *difficulty*. We compute difficulty by the scores of
supervised learning methods as a proxy. For more details how we label
each dataset as per their difficulty, please check out our paper!

We provide additional meta-data to categorize the datasets with respect
to their difficulty. To access it, call the following API. This
will load the datasets directly in memory.

.. code:: ipython3

    easy_datasets = gl.get_easy_datasets()
    moderate_datasets = gl.get_moderate_datasets()
    hard_datasets = gl.get_hard_datasets()

Continual Learning
------------------

Using any of the above categorizations, GraphLog also provides an option
of evaluating models in a continual learning scenario. Here, we provide
a simple example to evaluate continual learning on a rolling window of
similar datasets, based on overlapping rules.
``get_sorted_dataset_ids(mode="train")`` API will return the datasets in
the order they were created in the paper, which follows a
rolling similarity.

.. code:: ipython3

    dataset_names = gl.get_sorted_dataset_ids(mode="train")

    for data_id in dataset_names:
        dataset = gl.get_dataset_by_name(data_id)
        for epoch in range(100):
            train_loader = gl.get_dataloader_by_mode(dataset, "train")
            for batch_id, batch in enumerate(train_loader):
                graphs = batch.graphs
                queries = batch.queries
                labels = batch.targets
                logits = your_model(graphs, queries)

