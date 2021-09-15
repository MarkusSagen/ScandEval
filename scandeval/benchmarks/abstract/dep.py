'''Abstract dependency parsing benchmark'''

from transformers import (DataCollatorForTokenClassification,
                          PreTrainedTokenizerBase)
from datasets import Dataset, load_metric
from functools import partial
from typing import Optional, Dict, List, Tuple
import logging
from abc import ABC
from tqdm.auto import tqdm
import numpy as np
import itertools as it

from .base import BaseBenchmark
from ...utils import InvalidBenchmark
from ...datasets import load_dataset


logger = logging.getLogger(__name__)


class DepBenchmark(BaseBenchmark, ABC):
    '''Abstract dependency parsing benchmark.

    Args:
        name (str):
            The name of the dataset.
        cache_dir (str, optional):
            Where the downloaded models will be stored. Defaults to
            '.benchmark_models'.
        evaluate_train (bool, optional):
            Whether the models should be evaluated on the training scores.
            Defaults to False.
        verbose (bool, optional):
            Whether to print additional output during evaluation. Defaults to
            False.

    Attributes:
        name (str): The name of the dataset.
        task (str): The type of task to be benchmarked.
        metric_names (dict): The names of the metrics.
        id2label (dict or None): A dictionary converting indices to labels.
        label2id (dict or None): A dictionary converting labels to indices.
        num_labels (int or None): The number of labels in the dataset.
        label_synonyms (list of lists of str): Synonyms of the dataset labels.
        evaluate_train (bool): Whether the training set should be evaluated.
        cache_dir (str): Directory where models are cached.
        verbose (bool): Whether to print additional output.
    '''

    id2label = ['acl', 'advcl', 'advmod', 'amod', 'appos', 'aux', 'case',
                'cc', 'ccomp', 'clf', 'compound', 'conj', 'cop', 'csubj',
                'dep', 'det', 'discourse', 'dislocated', 'expl', 'fixed',
                'flat', 'goeswith', 'iobj', 'list', 'mark', 'nmod',
                'nsubj', 'nummod', 'obj', 'obl', 'orphan', 'parataxis',
                'punct', 'reparandum', 'root', 'vocative', 'xcomp']

    def __init__(self,
                 name: str,
                 evaluate_train: bool = False,
                 cache_dir: str = '.benchmark_models',
                 verbose: bool = False):
        self._metric = load_metric('accuracy')
        super().__init__(name=name,
                         task='dependency-parsing',
                         metric_names=dict(las='LAS', uas='UAS'),
                         id2label=self.id2label,
                         label_synonyms=None,
                         cache_dir=cache_dir,
                         evaluate_train=evaluate_train,
                         verbose=verbose)

    def _tokenize_and_align_labels(self,
                                   examples: dict,
                                   tokenizer,
                                   label2id: dict):
        '''Tokenise all texts and align the labels with them.

        Args:
            examples (dict):
                The examples to be tokenised.
            tokenizer (HuggingFace tokenizer):
                A pretrained tokenizer.
            label2id (dict):
                A dictionary that converts dependency relation tags to IDs.

        Returns:
            dict:
                A dictionary containing the tokenized data as well as labels.
        '''
        tokenized_inputs = tokenizer(
            examples['tokens'],
            # We use this argument because the texts in our dataset are lists
            # of words (with a label for each word)
            is_split_into_words=True,
        )
        all_labels = []
        for i, labels in enumerate(examples['orig_labels']):
            word_ids = tokenized_inputs.word_ids(batch_index=i)
            previous_word_idx = None
            label_ids = []
            for word_idx in word_ids:

                # Special tokens have a word id that is None. We set the label
                # to -100 so they are automatically ignored in the loss
                # function
                if word_idx is None:
                    label_ids.append((-100, -100))

                # We set the label for the first token of each word
                elif word_idx != previous_word_idx:
                    head_label, dep_label = labels[word_idx]
                    try:
                        head_label_id = int(head_label)
                        dep_label_id = label2id[dep_label]
                    except KeyError:
                        msg = (f'The label "{dep_label}" was not found '
                               f'in the model\'s config.')
                        raise InvalidBenchmark(msg)
                    label_ids.append((head_label_id, dep_label_id))

                # For the other tokens in a word, we set the label to -100
                else:
                    label_ids.append((-100, -100))

                previous_word_idx = word_idx

            all_labels.append(label_ids)
        tokenized_inputs['labels'] = all_labels
        return tokenized_inputs

    def _preprocess_data(self,
                         dataset: Dataset,
                         framework: str,
                         **kwargs) -> Dataset:
        '''Preprocess a dataset by tokenizing and aligning the labels.

        Args:
            dataset (HuggingFace dataset):
                The dataset to preprocess.
            kwargs:
                Extra keyword arguments containing objects used in
                preprocessing the dataset.

        Returns:
            HuggingFace dataset: The preprocessed dataset.
        '''
        if framework in ['pytorch', 'tensorflow', 'jax']:
            map_fn = partial(self._tokenize_and_align_labels,
                             tokenizer=kwargs['tokenizer'],
                             label2id=kwargs['config'].label2id)
            tokenised_dataset = dataset.map(map_fn, batched=True)
            return tokenised_dataset
        elif framework == 'spacy':
            return dataset

    def _load_data_collator(
            self,
            tokenizer: Optional[PreTrainedTokenizerBase] = None):
        '''Load the data collator used to prepare samples during finetuning.

        Args:
            tokenizer (HuggingFace tokenizer or None, optional):
                A pretrained tokenizer. Can be None if the tokenizer is not
                used in the initialisation of the data collator. Defaults to
                None.

        Returns:
            HuggingFace data collator: The data collator.
        '''
        params = dict(label_pad_token_id=[-100, -100])
        return DataCollatorForTokenClassification(tokenizer, **params)

    def _load_data(self) -> Tuple[Dataset, Dataset]:
        '''Load the datasets.

        Returns:
            A triple of HuggingFace datasets:
                The train and test datasets.
        '''
        X_train, X_test, y_train, y_test = load_dataset(self.short_name)

        train_labels = [list(zip(head, dep))
                        for head, dep in zip(y_train['heads'],
                                             y_train['deps'])]
        test_labels = [list(zip(head, dep))
                       for head, dep in zip(y_test['heads'],
                                            y_test['deps'])]

        train_dict = dict(doc=X_train['doc'],
                          tokens=X_train['tokens'],
                          orig_labels=train_labels)
        test_dict = dict(doc=X_test['doc'],
                         tokens=X_test['tokens'],
                         orig_labels=test_labels)

        train = Dataset.from_dict(train_dict)
        test = Dataset.from_dict(test_dict)
        return train, test

    def _compute_metrics(self,
                         predictions_and_labels: tuple,
                         id2label: Optional[dict] = None) -> Dict[str, float]:
        '''Compute the metrics needed for evaluation.

        Args:
            predictions_and_labels (pair of arrays):
                The first array contains the probability predictions and the
                second array contains the true labels.
            id2label (list or None, optional):
                Conversion of indices to labels. Defaults to None.

        Returns:
            dict:
                A dictionary with the names of the metrics as keys and the
                metric values as values.
        '''
        # Get the predictions from the model
        predictions, labels = predictions_and_labels

        # If `id2label` is given then assume that `predictions` contain ID
        # logits for every token, where an ID can mean both a head and a dep,
        # so it needs to be split up in these two halves.
        if id2label is not None:

            # Here we split up the predictions into the "head part" and the
            # "dep part"
            head_predictions, dep_predictions = predictions

            # With the predictions split up, we can then get the highest logits
            # to get the head and dep label
            head_predictions = np.argmax(head_predictions, axis=-1)
            dep_raw_predictions = np.argmax(dep_predictions, axis=-1)

            # The `labels` are assumed to be of shape
            # (batch_size, sequence_length, label_type), where `label_type` is
            # a binary number indicating either the head label or the dep
            # label. Here we extract the two different labels.
            head_labels = labels[:, 0]
            dep_labels = labels[:, 1]

            # Remove ignored indices from predictions and labels
            dep_predictions = [id2label[pred]
                               for pred in dep_raw_predictions]
            dep_labels = [id2label[lbl] for lbl in dep_labels]

            # Next merge the predictions and labels, so that we have a pair of
            # predicted/gold labels for each token
            heads = range(max(head_predictions.max(), head_labels.max()) + 1)
            deps = list(set(dep_labels).union(set(dep_predictions)))
            head_deps = it.product(heads, deps)
            str_to_int = {str((head, dep)): idx
                          for idx, (head, dep) in enumerate(head_deps)}
            predictions_merged = [str_to_int[str((head, dep))]
                                  for head, dep in zip(head_predictions,
                                                       dep_predictions)]
            labels_merged = [str_to_int[str((head, dep))]
                             for head, dep in zip(head_labels, dep_labels)]

        # If `id2label` is not given then assume that the predictions and
        # labels contain a pair (head, dep) for every token.
        else:

            unique_preds = {tup for tuples in predictions for tup in tuples}
            unique_labels = {tup for tuples in labels for tup in tuples}
            all_merged = list(unique_preds.union(unique_labels))
            str_to_int = {str(tup): idx for idx, tup in enumerate(all_merged)}
            predictions_merged = [str_to_int[str(tup)]
                                  for tuples in predictions
                                  for tup in tuples]
            labels_merged = [str_to_int[str(tup)]
                             for tuples in labels
                             for tup in tuples]

            # Convert the pair of labels to a single one by converting it into
            # strings. This is used in LAS computations.
            predictions_merged = [list(map(str, tuples))
                                  for tuples in predictions]
            labels_merged = [list(map(str, tuples)) for tuples in labels]

            # Extract the heads predictions and labels, used in UAS computation
            head_predictions = [head for tuples in predictions
                                for head, _ in tuples]
            head_labels = [head for tuples in labels for head, _ in tuples]

        # Compute metrics for the heads, which is used in UAS computation
        results_head = self._metric.compute(predictions=head_predictions,
                                            references=head_labels)

        # Compute metrics for the merged heads and deps, which is used in LAS
        # computation
        results_merged = self._metric.compute(predictions=predictions_merged,
                                              references=labels_merged)

        # Extract UAS and LAS and return them
        uas = results_head['accuracy']
        las = results_merged['accuracy']
        return dict(uas=uas, las=las)

    def _get_spacy_predictions_and_labels(self,
                                          model,
                                          dataset: Dataset,
                                          progress_bar: bool) -> tuple:
        '''Get predictions from SpaCy model on dataset.

        Args:
            model (SpaCy model): The model.
            dataset (HuggingFace dataset): The dataset.

        Returns:
            A pair of arrays:
                The first array contains the probability predictions and the
                second array contains the true labels.
        '''
        # Initialise progress bar
        if progress_bar:
            itr = tqdm(dataset['doc'])
        else:
            itr = dataset['doc']

        processed = model.pipe(itr, batch_size=32)
        map_fn = self._extract_spacy_predictions
        predictions = map(map_fn, zip(dataset['tokens'], processed))

        return list(predictions), dataset['orig_labels']

    def _extract_spacy_predictions(self, tokens_processed: tuple) -> list:
        '''Helper function that extracts the predictions from a SpaCy model.

        Aside from extracting the predictions from the model, it also aligns
        the predictions with the gold tokens, in case the SpaCy tokeniser
        tokenises the text different from those.

        Args:
            tokens_processed (tuple):
                A pair of the labels, being a list of strings, and the SpaCy
                processed document, being a Spacy `Doc` instance.

        Returns:
            list:
                A list of predictions for each token, of the same length as the
                gold tokens (first entry of `tokens_processed`).
        '''
        tokens, processed = tokens_processed

        # Get the token labels
        token_labels = self._get_spacy_token_labels(processed)

        # Get the alignment between the SpaCy model's tokens and the gold
        # tokens
        token_idxs = [tok_idx for tok_idx, tok in enumerate(tokens)
                      for _ in str(tok)]
        pred_token_idxs = [tok_idx for tok_idx, tok in enumerate(processed)
                           for _ in str(tok)]
        alignment = list(zip(token_idxs, pred_token_idxs))

        # Get the aligned predictions
        predictions = list()
        for tok_idx, _ in enumerate(tokens):
            aligned_pred_token = [pred_token_idx
                                  for token_idx, pred_token_idx in alignment
                                  if token_idx == tok_idx][0]
            predictions.append(token_labels[aligned_pred_token])

        return predictions

    def _get_spacy_token_labels(self, processed) -> List[List[str]]:
        '''Get predictions from SpaCy model on dataset.

        Args:
            model (SpaCy model): The model.
            dataset (HuggingFace dataset): The dataset.

        Returns:
            A list of list of strings:
                The predicted dependency labels.
        '''
        def get_heads_and_deps(token) -> List[str]:
            dep = token.dep_.lower().split(':')[0]
            if dep == 'root':
                head = '0'
            else:
                head = str(token.head.i + 1)
            return [head, dep]
        return [get_heads_and_deps(token) for token in processed]
