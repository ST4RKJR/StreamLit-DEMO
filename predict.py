def predict_one(node, x, path=None):
    if path is None:
        path = []

    if node.value is not None:
        return node.value, path

    if x[node.feature] <= node.threshold:
        path.append(f"x[{node.feature}] <= {node.threshold:.2f}")
        return predict_one(node.left, x, path)
    else:
        path.append(f"x[{node.feature}] > {node.threshold:.2f}")
        return predict_one(node.right, x, path)