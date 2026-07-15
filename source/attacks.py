def flip_labels(labels, total_classes):
    flipped = total_classes - 1 - labels
    return flipped