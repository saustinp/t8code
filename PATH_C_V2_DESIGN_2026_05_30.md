# Path C v2 — Periodic-aware t8_forest_balance — Design

**Repo:** saustinp/t8code, branch `fix/periodic-edge-vertex-neighbors`, HEAD `00e962b1`.
**Companion:** Path C v1 already shipped `t8_forest_leaf_periodic_neighbors` API + Kuhn-cube gtest. v2 reuses the same underlying spatial-corner-hash mechanism inside the balance loop.

## Problem

`t8_forest_balance` enforces 2:1 only across face neighbors (`t8_forest_element_half_face_neighbors`). On periodic-edge / periodic-vertex adjacencies, the level difference can grow unbounded, which produces hanging configurations on periodic seams (Path C v1 demonstrated this — sensor-only correction triggers asymmetric balance cascades).

## Strategy

Extend the existing `t8_forest_balance_adapt` callback so each leaf, after the face loop, also queries its periodic edge/vertex neighbors and refines itself if any periodic neighbor's level exceeds `(my_level + 1)`. The query is per-element but the underlying corner spatial hash is built ONCE per adapt round.

Critical: each leaf only refines itself, never marks neighbors — same symmetry contract as face balance. The fixed-point loop in `t8_forest_balance` converges as before.

## Sufficiency argument (why corner hash finds enough)

For balance, we need to find AT LEAST ONE partner with level > `my_level + 1` whenever such a partner exists somewhere on a periodic adjacency. The corner-hash approach finds partners via matching corner positions after applying a periodic translation. Edge cases:

- **I'm coarse (level L), partner refined (level L+k, k>1).** Refinement preserves original parent corners, so at least one of partner's leaf corners coincides with mine after translation. Found via ≥1-corner match.

- **I'm refined (level L+k), partner coarse (level L), I'm a CORNER-child.** My corner-child position includes the parent's original corner, which is preserved through refinement. Partner's level-L corner at the same position → found.

- **I'm refined (level L+k), partner coarse (level L), I'm an INTERIOR-child (no parent-corner).** My corners are at midpoints; they don't match partner's level-L corners. Looks like a miss. BUT my interior-child has a corner-child sibling (also at level L+k), and partner FINDS that sibling via partner→sibling lookup. Partner refines on the sibling's evidence, which is symmetric and equivalent for balance correctness.

So 2:1 balance propagates correctly even though individual queries can miss interior-only partner relationships.

## Data structures

### `t8_forest_periodic_cache_t`

Opaque struct, stored in `t8_forest.cxx`. Holds:
- `n_translations`, `translations[MAX_TRANSLATIONS][3]` — derived once from cmesh face-pair joins via the same logic in `t8_forest_leaf_periodic_neighbors`.
- `corner_hash : unordered_map<CornerKey, vector<CornerLoc>, CornerKeyHash>` — built over all leaf corners in the current forest.
- `bucket`, `tol` — quantization constants.

### `t8_forest_balance_data` (new internal struct)

Replaces the bare `int *` payload behind `forest->t8code_data`:
```cpp
struct t8_forest_balance_data {
  int done;
  t8_forest_periodic_cache_t *periodic_cache;  // owned, nullptr if cmesh has no periodic joins
};
```

## API additions (internal to t8_forest_balance.cxx)

Kept private (`static` linkage) so we don't commit a public API until the design lands:

```cpp
// In t8_forest.cxx (header: t8_forest_general.h, internal only via t8_forest_private.h ideally).
// Or for v2 prototyping: define directly in t8_forest_balance.cxx as a self-contained unit.
static t8_forest_periodic_cache_t *t8_forest_periodic_cache_new(t8_forest_t forest);
static void t8_forest_periodic_cache_destroy(t8_forest_periodic_cache_t *cache);
static int  t8_forest_periodic_cache_max_neighbor_level(
    t8_forest_periodic_cache_t *cache,
    t8_forest_t forest,
    t8_locidx_t ltreeid,
    const t8_element_t *leaf);
```

`_max_neighbor_level` returns:
- `-1` if `leaf` has no periodic neighbors.
- The maximum level among all periodic edge/vertex/face neighbors found.

This is the only query balance needs — no element copies, no classification, no allocation overhead.

## Cache lifecycle

In `t8_forest_balance`:

```cpp
t8_forest_balance_data data = {1, nullptr};
data.periodic_cache = t8_forest_periodic_cache_new(forest->set_from);  // nullptr if no periodic

while (!done_global) {
  data.done = 1;
  // ... existing forest_temp init + adapt setup ...
  
  // REBUILD cache against the NEW forest_from (leaves change each round)
  if (data.periodic_cache) {
    t8_forest_periodic_cache_destroy(data.periodic_cache);
    data.periodic_cache = t8_forest_periodic_cache_new(forest_from);
  }
  forest_temp->t8code_data = &data;
  
  t8_forest_commit(forest_temp);   // runs adapt callback
  
  sc_MPI_Allreduce(&data.done, &done_global, ...);
  forest_from = forest_temp;
}
t8_forest_periodic_cache_destroy(data.periodic_cache);
```

Same for `t8_forest_is_balanced` — build the cache once at entry, pass through `forest->t8code_data`, destroy at exit.

## Callback extension

```cpp
static int
t8_forest_balance_adapt(t8_forest_t forest, t8_forest_t forest_from, t8_locidx_t ltree_id,
                        t8_eclass_t tree_class, t8_locidx_t lelement_id,
                        const t8_scheme *scheme, int is_family, int num_elements,
                        t8_element_t *elements[])
{
  t8_forest_balance_data *data = (t8_forest_balance_data *) forest->t8code_data;
  const t8_element_t *element = elements[0];
  const int my_level = scheme->element_get_level(tree_class, element);

  if (forest_from->maxlevel_existing <= 0 || my_level <= forest_from->maxlevel_existing - 2) {
    // ... existing face loop, modified to refer to data->done instead of pdone ...
    // (returns 1 with data->done = 0 if any face-neighbor too refined)

    // NEW: periodic edge/vertex check
    if (data->periodic_cache != nullptr) {
      const int max_pn_level = t8_forest_periodic_cache_max_neighbor_level(
          data->periodic_cache, forest_from, ltree_id, element);
      if (max_pn_level > my_level + 1) {
        data->done = 0;
        return 1;
      }
    }
  }
  return 0;
}
```

## Multi-rank (deferred to Step 5)

Currently v1's `t8_forest_leaf_periodic_neighbors` asserts `mpisize == 1`. For balance v2 NP > 1:
- Allgatherv on-periodic-boundary leaf corners (a small subset of total leaves) into a per-process global hash.
- Or: reuse t8's ghost layer + add periodic-edge ghost exchange.

Allgatherv is simpler — start there. Defer to after NP=1 is validated.

## Test plan

**t8 gtest** (`test/t8_forest/t8_gtest_balance_periodic_edge.cxx`):
1. Build Kuhn-cube periodic forest at level 1.
2. Hand-refine one tet that has a periodic-edge adjacency (no periodic-face adjacency!) to level 3 (2 levels skip).
3. Call `t8_forest_balance(forest, 0)`.
4. Assert: the periodic edge-shared partner gets refined to level 2 (not just face-only neighbors).
5. Assert: `t8_forest_is_balanced(forest) == 1`.

**amr_dev integration** (test 88):
- Reinstall patched t8 to `~/.local/t8code-cad-fix`.
- Rebuild amr_dev.
- Run test 88 S5 NP=1 cycle 0: expect `final_unmatched_vertices == 0` strict.
- Run S0–S4b: expect no regression (trajectory still = [0]).
- Run full 3D AMR2 regression (tests 38, 52, 55, 56, 57, 58, 59): expect 31/31 PASS.

## File scope

| File | Change |
|---|---|
| `src/t8_forest/t8_forest_balance.cxx` | Add `t8_forest_balance_data` struct, periodic cache helpers, extended callback. Wire into top-level loop + `t8_forest_is_balanced`. |
| `src/t8_forest/t8_forest.cxx` | Extract `compute_periodic_translations` from `t8_forest_leaf_periodic_neighbors` into a reusable helper. Possibly expose as private symbol for balance to consume. |
| `test/t8_forest/t8_gtest_balance_periodic_edge.cxx` | New gtest for v2 contract. |
| `test/CMakeLists.txt` | Register new gtest. |

No public header change (everything internal to balance + t8_forest.cxx) — public API surface for v2 lands only if upstream PR review demands it.

## Phasing

1. **Step 1:** This doc. ✓
2. **Step 2:** Extract `t8_forest_periodic_cache_t` from v1 query (cache struct + new/destroy/query). Single-rank only.
3. **Step 3:** Modify `t8_forest_balance` + `t8_forest_balance_adapt` + `t8_forest_is_balanced` to consume cache.
4. **Step 4:** Write Kuhn-cube periodic-edge gtest. Iterate until green.
5. **Step 5:** Reinstall + amr_dev test 88 NP=1 validation. Iterate.
6. **Step 6 (deferred):** Multi-rank port via Allgatherv.
7. **Step 7 (deferred):** Upstream DLR-AMR PR.

Time horizon for Steps 2–5 (NP=1 closure): ~4–6 working days.
