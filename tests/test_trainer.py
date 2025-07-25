from mlops2.models.trainer import ModelTrainer

def test_trainer_end_to_end(sample_df):
    trained, reports = ModelTrainer().train(sample_df)
    # Should train all configured models
    assert trained.keys() == reports.keys()
    assert set(trained) >= {"RandomForest","GaussianNB","LogisticRegression","LinearSVC"}
    # Basic sanity: each report contains 'accuracy'
    for rep in reports.values():
        assert "accuracy" in rep
