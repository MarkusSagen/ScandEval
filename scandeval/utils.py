'''Utility functions to be used in other scripts'''

from functools import wraps
from typing import Callable
import warnings
import datasets.utils.logging as ds_logging
import logging
import pkg_resources
import re
import transformers.utils.logging as tf_logging
from pydoc import locate
from transformers import (AutoModelForTokenClassification,
                          AutoModelForSequenceClassification,
                          TFAutoModelForTokenClassification,
                          TFAutoModelForSequenceClassification,
                          FlaxAutoModelForTokenClassification,
                          FlaxAutoModelForSequenceClassification,
                          Trainer)

from .dependency_parsing import AutoModelForDependencyParsing


class DepTrainer(Trainer):
    def compute_loss(self, *args, **kwargs):
        loss = super().compute_loss(*args, **kwargs)
        return loss

    def _maybe_log_save_evaluate(self, tr_loss, model, trial, epoch):
        if self.control.should_log:
            logs: Dict[str, float] = {}
            tr_loss_scalar = tr_loss.item()
            # reset tr_loss to zero
            tr_loss -= tr_loss

            logs["loss"] = round(tr_loss_scalar / (self.state.global_step - self._globalstep_last_logged), 4)
            logs["learning_rate"] = self._get_learning_rate()

            self._total_loss_scalar += tr_loss_scalar
            self._globalstep_last_logged = self.state.global_step
            self.store_flos()

            self.log(logs)

        metrics = None
        if self.control.should_evaluate:
            metrics = self.evaluate()
            self._report_to_hp_search(trial, epoch, metrics)
            metrics = self.evaluate(eval_dataset=self.train_dataset,
                                    metric_key_prefix='train')
            self._report_to_hp_search(trial, epoch, metrics)

        if self.control.should_save:
            self._save_checkpoint(model, trial, metrics=metrics)
            self.control = self.callback_handler.on_save(self.args, self.state, self.control)

    def prediction_step(self, *args, **kwargs):
        import torch

        loss, logits, labels = super().prediction_step(*args, **kwargs)

        # Extract the head labels and dep labels
        head_labels = labels[:, :, 0]
        dep_labels = labels[:, :, 1]

        # Extract the head logits and dep logits
        head_logits, dep_logits = logits

        # Get the active labels and logits
        mask = head_labels.ge(0)
        active_head_logits = head_logits[mask]
        active_dep_logits = dep_logits[mask]
        active_head_labels = head_labels[mask]
        active_dep_labels = dep_labels[mask]

        # Get the dependency label logits for the gold arcs and store that in
        # the `logits` variable, to enable proper processing during evaluation
        label_range = torch.arange(len(active_head_labels))
        active_dep_logits = active_dep_logits[label_range, active_head_labels]

        # Collect the head and dep logits into the `logits` variable
        logits = (active_head_logits, active_dep_logits)

        # Collect the head and dep labels into the `labels` variable
        labels = torch.stack((active_head_labels, active_dep_labels), dim=-1)

        return loss, logits, labels

def get_all_datasets() -> list:
    '''Load a list of all datasets.

    Returns:
        list of tuples:
            First entry in each tuple is the short name of the dataset, second
            entry the long name, third entry the benchmark class and fourth
            entry the loading function.
    '''
    return [
        ('dane', 'DaNE',
            locate('scandeval.benchmarks.DaneBenchmark'),
            locate('scandeval.datasets.load_dane')),
        ('ddt-pos', 'the POS part of DDT',
            locate('scandeval.benchmarks.DdtPosBenchmark'),
            locate('scandeval.datasets.load_ddt_pos')),
        ('ddt-dep', 'the DEP part of DDT',
            locate('scandeval.benchmarks.DdtDepBenchmark'),
            locate('scandeval.datasets.load_ddt_dep')),
        ('angry-tweets', 'AngryTweets',
            locate('scandeval.benchmarks.AngryTweetsBenchmark'),
            locate('scandeval.datasets.load_angry_tweets')),
        ('twitter-sent', 'TwitterSent',
            locate('scandeval.benchmarks.TwitterSentBenchmark'),
            locate('scandeval.datasets.load_twitter_sent')),
        ('europarl', 'Europarl',
            locate('scandeval.benchmarks.EuroparlBenchmark'),
            locate('scandeval.datasets.load_europarl')),
        ('dkhate', 'DKHate',
            locate('scandeval.benchmarks.DkHateBenchmark'),
            locate('scandeval.datasets.load_dkhate')),
        ('lcc', 'LCC',
            locate('scandeval.benchmarks.LccBenchmark'),
            locate('scandeval.datasets.load_lcc')),
        ('norec', 'NoReC',
            locate('scandeval.benchmarks.NorecBenchmark'),
            locate('scandeval.datasets.load_norec')),
        ('nordial', 'NorDial',
            locate('scandeval.benchmarks.NorDialBenchmark'),
            locate('scandeval.datasets.load_nordial')),
        ('norne-nb', 'the Bokmål part of NorNE',
            locate('scandeval.benchmarks.NorneNBBenchmark'),
            locate('scandeval.datasets.load_norne_nb')),
        ('norne-nn', 'the Nynorsk part of NorNE',
            locate('scandeval.benchmarks.NorneNNBenchmark'),
            locate('scandeval.datasets.load_norne_nn')),
        ('ndt-nb-pos', 'the Bokmål POS part of NDT',
            locate('scandeval.benchmarks.NdtNBPosBenchmark'),
            locate('scandeval.datasets.load_ndt_nb_pos')),
        ('ndt-nn-pos', 'the Nynorsk POS part of NDT',
            locate('scandeval.benchmarks.NdtNNPosBenchmark'),
            locate('scandeval.datasets.load_ndt_nn_pos')),
        ('ndt-nb-dep', 'the Bokmål DEP part of NDT',
            locate('scandeval.benchmarks.NdtNBDepBenchmark'),
            locate('scandeval.datasets.load_ndt_nb_dep')),
        ('ndt-nn-dep', 'the Nynorsk DEP part of NDT',
            locate('scandeval.benchmarks.NdtNNDepBenchmark'),
            locate('scandeval.datasets.load_ndt_nn_dep')),
        ('dalaj', 'DaLaJ',
            locate('scandeval.benchmarks.DalajBenchmark'),
            locate('scandeval.datasets.load_dalaj')),
        ('absabank-imm', 'ABSAbank-Imm',
            locate('scandeval.benchmarks.AbsabankImmBenchmark'),
            locate('scandeval.datasets.load_absabank_imm')),
        ('sdt-pos', 'the POS part of SDT',
            locate('scandeval.benchmarks.SdtPosBenchmark'),
            locate('scandeval.datasets.load_sdt_pos')),
        ('sdt-dep', 'the DEP part of SDT',
            locate('scandeval.benchmarks.SdtDepBenchmark'),
            locate('scandeval.datasets.load_sdt_dep')),
        ('suc3', 'SUC 3.0',
            locate('scandeval.benchmarks.Suc3Benchmark'),
            locate('scandeval.datasets.load_suc3')),
        ('idt-pos', 'the POS part of IDT',
            locate('scandeval.benchmarks.IdtPosBenchmark'),
            locate('scandeval.datasets.load_idt_pos')),
        ('idt-dep', 'the DEP part of IDT',
            locate('scandeval.benchmarks.IdtDepBenchmark'),
            locate('scandeval.datasets.load_idt_dep')),
        ('wikiann-is', 'the Icelandic part of WikiANN',
            locate('scandeval.benchmarks.WikiannIsBenchmark'),
            locate('scandeval.datasets.load_wikiann_is')),
        ('wikiann-fo', 'the Faroese part of WikiANN',
            locate('scandeval.benchmarks.WikiannFoBenchmark'),
            locate('scandeval.datasets.load_wikiann_fo')),
        ('fdt-pos', 'the POS part of FDT',
            locate('scandeval.benchmarks.FdtPosBenchmark'),
            locate('scandeval.datasets.load_fdt_pos')),
        ('fdt-dep', 'the DEP part of FDT',
            locate('scandeval.benchmarks.FdtDepBenchmark'),
            locate('scandeval.datasets.load_fdt_dep')),
        ('norec-is', 'NoReC-IS',
            locate('scandeval.benchmarks.NorecISBenchmark'),
            locate('scandeval.datasets.load_norec_is')),
        ('norec-fo', 'NoReC-FO',
            locate('scandeval.benchmarks.NorecFOBenchmark'),
            locate('scandeval.datasets.load_norec_fo')),
    ]


PT_CLS = {'token-classification': AutoModelForTokenClassification,
          'text-classification': AutoModelForSequenceClassification,
          'dependency-parsing': AutoModelForDependencyParsing}
TF_CLS = {'token-classification': TFAutoModelForTokenClassification,
          'text-classification': TFAutoModelForSequenceClassification,
          'dependency-parsing': AutoModelForDependencyParsing}
JAX_CLS = {'token-classification': FlaxAutoModelForTokenClassification,
           'text-classification': FlaxAutoModelForSequenceClassification,
           'dependency-parsing': AutoModelForDependencyParsing}
MODEL_CLASSES = dict(pytorch=PT_CLS, tensorflow=TF_CLS, jax=JAX_CLS)


class InvalidBenchmark(Exception):
    def __init__(self, message: str = 'This model cannot be benchmarked '
                                      'on the given dataset.'):
        self.message = message
        super().__init__(self.message)


def is_module_installed(module: str) -> bool:
    '''Check if a module is installed.

    Args:
        module (str): The name of the module.

    Returns:
        bool: Whether the module is installed or not.
    '''
    installed_modules_with_versions = list(pkg_resources.working_set)
    installed_modules = [re.sub('[0-9. ]', '', str(module))
                         for module in installed_modules_with_versions]
    installed_modules_processed = [module.lower().replace('-', '_')
                                   for module in installed_modules]
    return module.lower() in installed_modules_processed


def block_terminal_output():
    '''Blocks libraries from writing output to the terminal'''

    # Ignore miscellaneous warnings
    warnings.filterwarnings('ignore',
                            module='torch.nn.parallel*',
                            message=('Was asked to gather along dimension 0, '
                                     'but all input tensors were scalars; '
                                     'will instead unsqueeze and return '
                                     'a vector.'))
    warnings.filterwarnings('ignore', module='seqeval*')

    logging.getLogger('filelock').setLevel(logging.ERROR)

    # Disable the tokenizer progress bars
    ds_logging.get_verbosity = lambda: ds_logging.NOTSET

    # Disable most of the `transformers` logging
    tf_logging.set_verbosity_error()


class DocInherit(object):
    '''Docstring inheriting method descriptor.

    The class itself is also used as a decorator.
    '''
    def __init__(self, mthd: Callable):
        self.mthd = mthd
        self.name = mthd.__name__

    def __get__(self, obj, cls):
        if obj:
            return self.get_with_inst(obj, cls)
        else:
            return self.get_no_inst(cls)

    def get_with_inst(self, obj, cls):

        overridden = getattr(super(cls, obj), self.name, None)

        @wraps(self.mthd, assigned=('__name__', '__module__'))
        def f(*args, **kwargs):
            return self.mthd(obj, *args, **kwargs)

        return self.use_parent_doc(f, overridden)

    def get_no_inst(self, cls):
        for parent in cls.__mro__[1:]:
            overridden = getattr(parent, self.name, None)
            if overridden:
                break

        @wraps(self.mthd, assigned=('__name__', '__module__'))
        def f(*args, **kwargs):
            return self.mthd(*args, **kwargs)

        return self.use_parent_doc(f, overridden)

    def use_parent_doc(self, func, source):
        if source is None:
            raise NameError(f'Can\'t find "{self.name}" in parents')
        func.__doc__ = source.__doc__
        return func


doc_inherit = DocInherit
