def parse_dino_layer_indices(args):
    raw = getattr(args, "dino_layer_indices", "")
    if raw:
        parts = [p.strip() for p in raw.split(",") if p.strip()]
        indices = []
        for p in parts:
            layer_num = int(p)
            if layer_num <= 0:
                raise ValueError("dino_layer_indices must be 1-based positive integers.")
            indices.append(layer_num - 1)
        return indices
    idx = getattr(args, "dino_layer_idx", -1)
    if idx is None or idx < 0:
        return [-1]
    return [idx]
