import pytest
from transformers import (
    AutoModelForSequenceClassification,
    AutoModelForTokenClassification,
    AutoTokenizer,
)
from ferret import (
    Benchmark,
    LIMEExplainer,
    SHAPExplainer,
    GradientExplainer,
    IntegratedGradientExplainer,
    TokenClassificationHelper,
)
from ferret.evaluators.faithfulness_measures import (
    AOPC_Comprehensiveness_Evaluation,
    AOPC_Sufficiency_Evaluation,
    TauLOO_Evaluation,
)
from ferret.evaluators.plausibility_measures import (
    AUPRC_PlausibilityEvaluation,
    Tokenf1_PlausibilityEvaluation,
    TokenIOU_PlausibilityEvaluation,
)
from ferret.evaluators.class_measures import AOPC_Comprehensiveness_Evaluation_by_class
from ferret.modeling.text_helpers import SequenceClassificationHelper

DEFAULT_EXPLAINERS_NUM = 6
DEFAULT_EVALUATORS_NUM = 6
DEFAULT_EVALUATORS_BY_CLASS_NUM = 1


### Fixtures that are destroyed at the end of the testing
@pytest.fixture
def model_text_class():
    return AutoModelForSequenceClassification.from_pretrained("lvwerra/distilbert-imdb")


@pytest.fixture
def tokenizer_text_class():
    return AutoTokenizer.from_pretrained("lvwerra/distilbert-imdb")


@pytest.fixture
def model_nli():
    return AutoModelForSequenceClassification.from_pretrained(
        "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli"
    )


@pytest.fixture
def tokenizer_nli():
    return AutoTokenizer.from_pretrained("MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli")


@pytest.fixture
def model_zero_shot():
    return AutoModelForSequenceClassification.from_pretrained(
        "MoritzLaurer/mDeBERTa-v3-base-mnli-xnli"
    )


@pytest.fixture
def tokenizer_zero_shot():
    return AutoTokenizer.from_pretrained("Babelscape/wikineural-multilingual-ner")


@pytest.fixture
def model_ner():
    return AutoModelForTokenClassification.from_pretrained(
        "Babelscape/wikineural-multilingual-ner"
    ).to("cpu")


@pytest.fixture
def tokenizer_ner():
    return AutoTokenizer.from_pretrained("Babelscape/wikineural-multilingual-ner")


###


### Fixture for all fixtures (initialization of the benchmarks)
@pytest.fixture
def all_benchmarks(
    model_text_class,
    tokenizer_text_class,
    model_nli,
    tokenizer_nli,
    model_zero_shot,
    tokenizer_zero_shot,
    model_ner,
    tokenizer_ner,
):
    benchmarks = [
        Benchmark(
            model_text_class, tokenizer_text_class, task_name="text-classification"
        ),
        Benchmark(model_nli, tokenizer_nli, task_name="nli"),
        Benchmark(
            model_zero_shot,
            tokenizer_zero_shot,
            task_name="zero-shot-text-classification",
        ),
        Benchmark(model_ner, tokenizer_ner, task_name="ner"),
    ]
    return benchmarks


###


### Tests


## Setup and Initialization Checks
def test_initialization_benchmarks(all_benchmarks):
    for bench in all_benchmarks:
        assert len(bench.explainers) == DEFAULT_EXPLAINERS_NUM
        assert len(bench.evaluators) == DEFAULT_EVALUATORS_NUM
        assert len(bench.class_based_evaluators) == DEFAULT_EVALUATORS_BY_CLASS_NUM


def test_explainer_types(all_benchmarks):
    for bench in all_benchmarks:
        assert all(
            isinstance(e, SHAPExplainer)
            or isinstance(e, LIMEExplainer)
            or isinstance(e, GradientExplainer)
            or isinstance(e, IntegratedGradientExplainer)
            for e in bench.explainers
        )


def test_evaluator_types(all_benchmarks):
    expected_evaluator_types = [
        AOPC_Comprehensiveness_Evaluation,
        AOPC_Sufficiency_Evaluation,
        TauLOO_Evaluation,
        AUPRC_PlausibilityEvaluation,
        Tokenf1_PlausibilityEvaluation,
        TokenIOU_PlausibilityEvaluation,
    ]
    for bench in all_benchmarks:
        assert all(
            any(isinstance(ev, t) for t in expected_evaluator_types)
            for ev in bench.evaluators
        )
        assert all(
            isinstance(ev_class, AOPC_Comprehensiveness_Evaluation_by_class)
            for ev_class in bench.class_based_evaluators
        )


def test_helper_assignment(all_benchmarks):
    for bench in all_benchmarks:
        for explainer in bench.explainers:
            assert explainer.helper == bench.helper


def test_helper_override_warning(model_ner, tokenizer_ner):
    explainer_with_helper = SHAPExplainer(
        model_ner, tokenizer_ner, helper=SequenceClassificationHelper
    )
    with pytest.warns(UserWarning, match="Overriding helper for explainer"):
        Benchmark(
            model_text_class,
            tokenizer_text_class,
            task_name="ner",
            explainers=[explainer_with_helper],
        )


##


def explainer_utility(explainer, text, target, expected_tokens, expected_target_pos_idx):
    explanation = explainer(text, target=target)
    assert explanation.tokens == expected_tokens
    assert explanation.target_pos_idx == expected_target_pos_idx


@pytest.mark.parametrize(
    "text, target, expected_tokens, expected_target_pos_idx, explainer_cls, explainer_kwargs",
    [
        (
            "You look stunning!",
            1,
            ["[CLS]", "you", "look", "stunning", "!", "[SEP]"],
            1,
            SHAPExplainer,
            {},
        ),
        (
            "You look so stunning!",
            1,
            ["[CLS]", "you", "look", "so", "stunning", "!", "[SEP]"],
            1,
            LIMEExplainer,
            {},
        ),
        (
            "The new movie is awesome!",
            1,
            ["[CLS]", "the", "new", "movie", "is", "awesome", "!", "[SEP]"],
            1,
            GradientExplainer,
            {"multiply_by_inputs": True},
        ),
        (
            "The new movie is awesome!",
            1,
            ["[CLS]", "the", "new", "movie", "is", "awesome", "!", "[SEP]"],
            1,
            IntegratedGradientExplainer,
            {"multiply_by_inputs": True},
        ),
    ],
)
def test_explainers(
    model_text_class,
    tokenizer_text_class,
    text,
    target,
    expected_tokens,
    expected_target_pos_idx,
    explainer_cls,
    explainer_kwargs,
):
    explainer = explainer_cls(model_text_class, tokenizer_text_class, **explainer_kwargs)
    explainer_utility(explainer, text, target, expected_tokens, expected_target_pos_idx)
