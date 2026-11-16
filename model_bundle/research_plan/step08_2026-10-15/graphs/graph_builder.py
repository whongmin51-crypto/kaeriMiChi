"""Label-free 2D periodic monolayer graphs; NumPy only, no model fitting.

Edges point from center i to image j: D = r_j + S @ cell - r_i.
The third image coefficient is always zero. Neighbor caps preserve a full
distance shell, so a nominal cap of 24 can produce more than 24 neighbors.
"""
from __future__ import annotations
import hashlib
import json
import numpy as np

ELEMENTS = ("H He Li Be B C N O F Ne Na Mg Al Si P S Cl Ar K Ca Sc Ti V Cr Mn Fe Co Ni Cu Zn "
            "Ga Ge As Se Br Kr Rb Sr Y Zr Nb Mo Tc Ru Rh Pd Ag Cd In Sn Sb Te I Xe Cs Ba La Ce "
            "Pr Nd Pm Sm Eu Gd Tb Dy Ho Er Tm Yb Lu Hf Ta W Re Os Ir Pt Au Hg Tl Pb Bi Po At Rn "
            "Fr Ra Ac Th Pa U Np Pu Am Cm Bk Cf Es Fm Md No Lr Rf Db Sg Bh Hs Mt Ds Rg Cn Nh Fl Mc Lv Ts Og").split()
ATOMIC_NUMBERS = {s: i + 1 for i, s in enumerate(ELEMENTS)}
DEFAULT_PARAMETERS = {
    'schema_version': 'anderson_monolayer_graph_v1',
    'cutoff_A': 5.0,
    'max_neighbors': 24,
    'neighbor_cap_policy': 'retain_complete_distance_shell',
    'shell_tolerance_A': 1e-8,
    'cutoff_tolerance_A': 1e-10,
    'zero_distance_tolerance_A': 1e-10,
    'rbf_count': 32,
    'rbf_min_A': 0.0,
    'rbf_width_A': 5.0 / 31.0,
    'pbc': [True, True, False],
    'node_features': 'atomic_number; fixed_one_hot_Z1_to_Z118_in_adapter',
    'optional_node_geometry': 'centered_signed_normal_height_A; not absolute vacuum coordinate',
    'cache_decimal_places': 10,
}


def parameters_with_defaults(parameters=None):
    p = dict(DEFAULT_PARAMETERS)
    if parameters:
        unknown = set(parameters) - set(p)
        if unknown:
            raise ValueError(f'Unknown graph parameters: {sorted(unknown)}')
        p.update(parameters)
    if p['pbc'] != [True, True, False]:
        raise ValueError('This graph protocol requires 2D periodicity [True,True,False]')
    if p['cutoff_A'] <= 0 or int(p['rbf_count']) < 2 or p['rbf_width_A'] <= 0:
        raise ValueError('Invalid cutoff / radial basis parameters')
    if p['max_neighbors'] is not None and int(p['max_neighbors']) < 1:
        raise ValueError('max_neighbors must be positive or None')
    if p['neighbor_cap_policy'] != 'retain_complete_distance_shell':
        raise ValueError('Only the complete-shell neighbor cap is supported')
    return p


def read_structure(structure):
    c = np.asarray(structure['cell_A'], dtype=np.float64)
    x = np.asarray(structure['positions_A'], dtype=np.float64)
    if c.shape != (3, 3) or x.ndim != 2 or x.shape[1] != 3 or not len(x):
        raise ValueError('cell_A must be (3,3), positions_A must be nonempty (N,3)')
    if not np.isfinite(c).all() or not np.isfinite(x).all():
        raise ValueError('Non-finite cell or atomic coordinates')
    z = structure.get('atomic_numbers')
    if z is None:
        try:
            z = [ATOMIC_NUMBERS[s] for s in structure['symbols']]
        except (KeyError, TypeError) as exc:
            raise ValueError('Provide valid atomic_numbers or standard element symbols') from exc
    z0 = np.asarray(z)
    if z0.shape != (len(x),) or not np.isfinite(z0.astype(float)).all():
        raise ValueError('atomic_numbers must contain one finite integer per atom')
    z = z0.astype(np.int64)
    if not np.array_equal(z, z0) or np.any((z < 1) | (z > 118)):
        raise ValueError('Atomic numbers must be integers in [1,118]')
    if structure.get('symbols') is not None:
        if list(z) != [ATOMIC_NUMBERS[s] for s in structure['symbols']]:
            raise ValueError('symbols and atomic_numbers disagree')
    norm = np.cross(c[0], c[1])
    area = np.linalg.norm(norm)
    if area < 1e-8:
        raise ValueError('Degenerate 2D cell')
    normal = norm / area
    return c, x, z, normal


def _all_neighbors(c, x, cutoff, tolerance, zero_tolerance):
    """Enumerate a provably sufficient rectangle in dual-lattice coordinates.

    For every accepted displacement D, |D @ dual[:,k]| <= cutoff*|dual[:,k]|.
    This bound handles arbitrary valid skew cells and coordinates outside a cell.
    """
    dual = np.linalg.pinv(c[:2])
    f = x @ dual
    reach = (cutoff + tolerance) * np.linalg.norm(dual, axis=0)
    neighbors = [[] for _ in range(len(x))]
    for i in range(len(x)):
        for j in range(len(x)):
            dfrac = f[j] - f[i]
            lower = np.ceil(-dfrac - reach - 1e-12).astype(int)
            upper = np.floor(-dfrac + reach + 1e-12).astype(int)
            for s0 in range(lower[0], upper[0] + 1):
                for s1 in range(lower[1], upper[1] + 1):
                    if i == j and s0 == 0 and s1 == 0:
                        continue
                    shift = np.array([s0, s1, 0], dtype=np.int64)
                    d = x[j] + shift @ c - x[i]
                    r = float(np.linalg.norm(d))
                    if r <= cutoff + tolerance:
                        if r <= zero_tolerance:
                            raise ValueError('Coincident distinct periodic atomic sites')
                        neighbors[i].append((r, j, shift, d))
        neighbors[i].sort(key=lambda v: (v[0], v[1], *v[2]))
    return neighbors


def build_graph(structure, parameters=None):
    """Build a sparse directed graph in the supplied coordinate frame.

    Input keys cell_A, positions_A, atomic_numbers or symbols. Output arrays are
    suitable for np.savez; no target/property/split field is inspected.
    """
    p = parameters_with_defaults(parameters)
    c, x, z, normal = read_structure(structure)
    candidate = _all_neighbors(c, x, p['cutoff_A'], p['cutoff_tolerance_A'], p['zero_distance_tolerance_A'])
    src, dst, distances, shifts, vectors = [], [], [], [], []
    candidate_count = np.array([len(v) for v in candidate], dtype=np.int64)
    kept_count = []
    for i, neighbors in enumerate(candidate):
        cap = p['max_neighbors']
        if cap is not None and len(neighbors) > int(cap):
            shell_radius = neighbors[int(cap) - 1][0]
            neighbors = [v for v in neighbors if v[0] <= shell_radius + p['shell_tolerance_A']]
        kept_count.append(len(neighbors))
        for r, j, shift, vector in neighbors:
            src.append(i); dst.append(j); distances.append(r)
            shifts.append(shift); vectors.append(vector)
    d = np.asarray(distances, dtype=np.float64)
    centers = np.linspace(p['rbf_min_A'], p['cutoff_A'], int(p['rbf_count']))
    h = x @ normal
    h = h - h.mean()
    count = np.asarray(kept_count, dtype=np.int64)
    return {
        'atomic_numbers': z,
        'positions_A': x - (x @ normal).mean() * normal,
        'node_height_A': h,
        'edge_index': np.asarray([src, dst], dtype=np.int64).reshape(2, -1),
        'edge_vectors_A': np.asarray(vectors, dtype=np.float64).reshape(-1, 3),
        'edge_distance_A': d,
        'image_shift': np.asarray(shifts, dtype=np.int64).reshape(-1, 3),
        'edge_rbf': np.exp(-((d[:, None] - centers[None, :]) / p['rbf_width_A']) ** 2).astype(np.float32),
        'candidate_neighbor_count': candidate_count,
        'neighbor_count': count,
        'truncated_count': candidate_count - count,
        'cell_A': c,
        'normal_unit_vector': normal,
        'rbf_centers_A': centers,
        'parameters_json': np.asarray(json.dumps(p, sort_keys=True, separators=(',', ':'))),
        'pbc': np.array([True, True, False]),
    }


def canonical_structure(structure, parameters=None):
    """Content-normalize origin/order/rigid-frame/vacuum for safe graph caching.

    In-plane basis is preserved. Distinct strains and opposite polar faces retain
    distinct keys. Quantization (10 decimal places) defines only cache precision,
    not the broader Step04 structural equivalence relation.
    """
    p = parameters_with_defaults(parameters)
    c, x, z, n = read_structure(structure)
    e1 = c[0] / np.linalg.norm(c[0])
    frame = np.array([e1, np.cross(n, e1), n])
    local_c = c @ frame.T
    local_x = x @ frame.T
    frac = local_x[:, :2] @ np.linalg.inv(local_c[:2, :2])
    heights = local_x[:, 2] - local_x[:, 2].mean()
    decimals = int(p['cache_decimal_places'])
    candidates = []
    for anchor in range(len(z)):
        f = np.mod(np.round(np.mod(frac - frac[anchor], 1.), decimals), 1.)
        xx = f @ local_c[:2, :2]
        array = np.column_stack([z, np.round(xx, decimals), np.round(heights, decimals)])
        order = np.lexsort((array[:, 2], array[:, 1], array[:, 3], array[:, 0]))
        key = tuple(array[order].flat)
        candidates.append((key, order, xx))
    _, order, xx = min(candidates, key=lambda t: t[0])
    cc = np.zeros((3, 3)); cc[:2, :2] = np.round(local_c[:2, :2], decimals)
    # c has no graph role; use a reproducible positive nonperiodic reporting axis.
    cc[2, 2] = float(np.round(np.ptp(heights) + 20., decimals))
    pp = np.column_stack([xx, heights])[order]
    pp = np.round(pp, decimals)
    # IEEE -0.0 and +0.0 are the same coordinate but JSON byte strings differ.
    # Remove rotation-induced signed zeros before content hashing.
    cc[cc == 0.] = 0.
    pp[pp == 0.] = 0.
    canonical = {'cell_A': cc.tolist(), 'positions_A': pp.tolist(), 'atomic_numbers': z[order].tolist()}
    content = {'structure': canonical, 'parameters': p}
    graph_id = 'graph:' + hashlib.sha256(json.dumps(content, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()
    inverse = np.empty(len(order), dtype=np.int64); inverse[order] = np.arange(len(order))
    return graph_id, canonical, {'source_to_canonical_atom_index': inverse.tolist(), 'local_frame_rows': frame.tolist(),
                                 'canonical_precision_A': 10. ** -decimals}


def load_graph(path):
    with np.load(path, allow_pickle=False) as source:
        return {key: source[key].copy() for key in source.files}


def to_cgcnn(graph, neighbor_width=None):
    """Dense CGCNN-style arrays, with explicit valid-neighbor mask.

    A future masked CGCNN convolution must exclude false entries from BN and
    message sum. Passing padded arrays to the untouched official convolution is
    not equivalent. Atomic feature width 118 is a frozen one-hot vocabulary.
    """
    z = graph['atomic_numbers']; n = len(z)
    width = max(1, int(graph['neighbor_count'].max(initial=0))) if neighbor_width is None else int(neighbor_width)
    if width < int(graph['neighbor_count'].max(initial=0)):
        raise ValueError('neighbor_width would discard a complete shell')
    idx = np.zeros((n, width), dtype=np.int64)
    mask = np.zeros((n, width), dtype=bool)
    rbf = np.zeros((n, width, graph['edge_rbf'].shape[1]), dtype=np.float32)
    vectors = np.zeros((n, width, 3), dtype=np.float64)
    for i in range(n):
        where = np.flatnonzero(graph['edge_index'][0] == i)
        idx[i, :len(where)] = graph['edge_index'][1, where]
        rbf[i, :len(where)] = graph['edge_rbf'][where]
        vectors[i, :len(where)] = graph['edge_vectors_A'][where]
        mask[i, :len(where)] = True
    return {'atom_features': np.eye(118, dtype=np.float32)[z - 1], 'neighbor_features': rbf,
            'neighbor_indices': idx, 'neighbor_mask': mask, 'neighbor_vectors_A': vectors}


def to_gann(graph):
    """Sparse attention-graph inputs; atomic pooling is permutation invariant."""
    return {'atom_features': np.eye(118, dtype=np.float32)[graph['atomic_numbers'] - 1],
            'edge_index': graph['edge_index'].copy(), 'edge_features': graph['edge_rbf'].copy(),
            'edge_vectors_A': graph['edge_vectors_A'].copy(), 'atomic_numbers': graph['atomic_numbers'].copy()}


def collate_cgcnn(graphs):
    if not graphs:
        raise ValueError('A graph batch must be nonempty')
    width = max(1, max(int(g['neighbor_count'].max(initial=0)) for g in graphs))
    output = {k: [] for k in ('atom_features', 'neighbor_features', 'neighbor_indices', 'neighbor_mask', 'neighbor_vectors_A')}
    members = []; offset = 0
    for graph in graphs:
        dense = to_cgcnn(graph, width)
        for key, value in dense.items():
            if key == 'neighbor_indices':
                value = value + offset
            output[key].append(value)
        members.append(np.arange(len(graph['atomic_numbers']), dtype=np.int64) + offset)
        offset += len(graph['atomic_numbers'])
    result = {k: np.concatenate(v, axis=0) for k, v in output.items()}
    result['crystal_atom_indices'] = members
    return result
