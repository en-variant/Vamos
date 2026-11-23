from tokentax_lab.eval.quality import (
    bootstrap_accuracy,
    compare_confidence_intervals,
    deterministic_check,
    group_bootstrap_accuracy,
    shuffle_options,
)


def test_bootstrap_ci():
    r = bootstrap_accuracy([True, False, True, True])
    assert 0 <= r["ci95_low"] <= r["accuracy"] <= r["ci95_high"] <= 1


def test_deterministic_check():
    assert deterministic_check(["A", "B"], ["A", "B"])["deterministic"]


def test_shuffle_is_seeded():
    options = ["A", "B", "C", "D"]
    assert set(shuffle_options(options, 1)) == set(options)


def test_group_bootstrap_reports_groups():
    result = group_bootstrap_accuracy(
        [True, False, True],
        ["passage-1", "passage-1", "passage-2"],
        n_bootstrap=100,
    )
    assert result["groups"] == 2
    assert result["interval_method"] == "passage_cluster_bootstrap"


def test_overlapping_intervals_are_no_measurable_difference():
    a = {"ci95_low": 0.4, "ci95_high": 0.6}
    b = {"ci95_low": 0.5, "ci95_high": 0.7}
    assert compare_confidence_intervals(a, b) == "no measurable difference"
