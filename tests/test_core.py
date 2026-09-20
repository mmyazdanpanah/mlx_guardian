from argparse import Namespace

from mlx_guardian import compute_total_steps, parse_metrics, progress_text


def test_parse_native_mlx_line():
    line = (
        "Iter 4800: Train loss 1.234, Learning Rate 2.000e-04, "
        "It/sec 0.658, Tokens/sec 78.322, Trained Tokens 385, Peak mem 1.518 GB"
    )
    event = parse_metrics(line)
    assert event["step"] == 4800
    assert event["loss"] == 1.234
    assert event["learning_rate"] == 2e-4
    assert event["it_per_sec"] == 0.658
    assert event["tokens_per_sec"] == 78.322
    assert event["trained_tokens"] == 385
    assert event["peak_mem_gb"] == 1.518


def test_total_steps_from_epoch_config():
    args = Namespace(
        total_steps=None,
        train_samples=90000,
        batch_size=2,
        grad_accumulation=4,
        epochs=2,
    )
    assert compute_total_steps(args) == 22500


def test_explicit_total_steps_wins():
    args = Namespace(
        total_steps=1234,
        train_samples=90000,
        batch_size=2,
        grad_accumulation=4,
        epochs=2,
    )
    assert compute_total_steps(args) == 1234


def test_progress_text():
    assert progress_text(4800, 22500) == "4,800 / 22,500  (21.3%)"
