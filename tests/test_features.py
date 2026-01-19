from src.features import build_preprocess_pipeline


def test_preprocess_pipeline_builds():
    pipe = build_preprocess_pipeline()
    assert pipe is not None
