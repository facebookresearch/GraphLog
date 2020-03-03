from graphlog import GraphLog


def test_get_dataset_ids() -> None:
    expected_ids = list(range(1, 9))
    graphlog = GraphLog(data_dir="")
    assert graphlog.get_dataset_ids() == expected_ids
