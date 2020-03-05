from graphlog import GraphLog
import os


def test_download() -> None:
    gl = GraphLog()
    data_loc = os.path.join(gl.data_dir, gl.data_filename)
    assert os.path.exists(data_loc) & os.path.isdir(data_loc)
    train_data_loc = os.path.join(data_loc, "train")
    assert os.path.exists(train_data_loc) & os.path.isdir(train_data_loc)
    valid_data_loc = os.path.join(data_loc, "valid")
    assert os.path.exists(valid_data_loc) & os.path.isdir(valid_data_loc)
    test_data_loc = os.path.join(data_loc, "test")
    assert os.path.exists(test_data_loc) & os.path.isdir(test_data_loc)


def test_label2id() -> None:
    gl = GraphLog()
    data_loc = os.path.join(gl.data_dir, gl.data_filename)
    label2id_loc = os.path.join(data_loc, "train", "label2id.json")
    assert os.path.exists(label2id_loc) & os.path.isfile(label2id_loc)
    len(gl.label2id.keys()) == 21
    assert gl.label2id["UNK_REL"] == 0


def test_paper_data_ids() -> None:
    gl = GraphLog()
    train_ids = ["rule_{}".format(d) for d in range(0, 51)]
    valid_ids = ["rule_{}".format(d) for d in range(51, 54)]
    test_ids = ["rule_{}".format(d) for d in range(54, 57)]
    data_by_split = gl.get_dataset_names_by_split()
    for world in train_ids:
        assert world in data_by_split["train"]
    for world in valid_ids:
        assert world in data_by_split["valid"]
    for world in test_ids:
        assert world in data_by_split["test"]
