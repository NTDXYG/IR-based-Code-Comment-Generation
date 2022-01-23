def load_dataset(dataset_path) -> list:
    """
    load the dataset from given path
    :param dataset_path: path of dataset
    :return: lines from the dataset
    """
    lines = []
    with open(dataset_path, 'r', encoding='utf-8') as file:
        for line in file.readlines():
            words = line.strip()
            lines.append(words)
    return lines
