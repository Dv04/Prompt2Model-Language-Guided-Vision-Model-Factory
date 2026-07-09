from prompt2model.cli import _run_pipeline, build_parser


def test_run_parser_accepts_pretrained_flag_and_defaults_false() -> None:
    parser = build_parser()

    args = parser.parse_args(
        [
            "run",
            "--prompt", "Classify beans and keep at least 70% accuracy.",
            "--dataset-root", "output/toy_data/classification",
            "--dataset-format", "imagefolder",
            "--pretrained",
        ]
    )
    assert args.pretrained is True

    default_args = parser.parse_args(
        [
            "run",
            "--prompt", "Classify beans and keep at least 70% accuracy.",
            "--dataset-root", "output/toy_data/classification",
            "--dataset-format", "imagefolder",
        ]
    )
    assert default_args.pretrained is False


def test_run_pipeline_threads_pretrained_into_training_config(monkeypatch) -> None:
    parser = build_parser()
    args = parser.parse_args(
        [
            "run",
            "--prompt", "Classify beans and keep at least 70% accuracy.",
            "--dataset-root", "output/toy_data/classification",
            "--dataset-format", "imagefolder",
            "--pretrained",
        ]
    )

    captured = {}

    class _FakeResult:
        run_dir = "fake_run_dir"
        report_path = "fake_report.md"
        onnx_path = "fake.onnx"
        compressed_onnx_path = None
        deployment = None
        metrics = {}

    def fake_run_from_prompt(*, training_overrides=None, **kwargs):
        captured["pretrained"] = training_overrides.pretrained
        return _FakeResult()

    monkeypatch.setattr("prompt2model.cli.run_from_prompt", fake_run_from_prompt)

    _run_pipeline(args)
    assert captured["pretrained"] is True
