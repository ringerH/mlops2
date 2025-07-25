from mlops2.preprocessing.pipeline import create_preprocessing_pipeline

def test_pipeline_runs(sample_df):
    pipe   = create_preprocessing_pipeline()
    matrix = pipe.fit_transform(sample_df["Sentence"])
    # should produce same number of rows as input
    assert matrix.shape[0] == len(sample_df)
    # sparse matrix with > 0 features
    assert matrix.shape[1] > 0
