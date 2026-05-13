/*
  This file is part of t8code.
  t8code is a C library to manage a collection (a forest) of multiple
  connected adaptive space-trees of general element classes in parallel.

  Copyright (C) 2026 the developers

  t8code is free software; you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation; either version 2 of the License, or
  (at your option) any later version.

  t8code is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with t8code; if not, write to the Free Software Foundation, Inc.,
  51 Franklin Street, Fifth Floor, Boston, MA 02110-1301, USA.
*/

/*
 * Tests for the linear_id sidecar cache on t8_element_array_t and its
 * interaction with t8_forest_bin_search_lower.
 *
 * What is being tested
 * --------------------
 *
 * Phase 3a adds three fields to t8_element_array_t (linear_id_cache,
 * linear_id_cache_level, linear_id_cache_count) plus the helpers
 * invalidate_linear_id_cache / get_linear_id_cache / set_linear_id_cache.
 *
 * Phase 3b teaches t8_forest_bin_search_lower to read the cache via
 * get_linear_id_cache; when fresh-for-level it runs std::upper_bound
 * directly over the precomputed t8_linearidx_t values. When the cache
 * is absent / stale / wrong-level the freshness predicate returns NULL
 * and bin_search_lower falls through to its scalar code path.
 *
 * Phase 3c adds t8_forest_element_array_ensure_linear_id_cache (per
 * array) and t8_forest_ensure_linear_id_caches (per forest) that
 * populate the cache, dispatching to t8_d{tri,tet}_linear_id_batch
 * (SIMD) for triangular / tetrahedral schemes and falling back to a
 * scheme->element_get_linear_id loop for other classes.
 *
 * This file verifies the cross-cutting invariants those phases imply:
 *
 *   1. CACHE-VS-SCALAR INVARIANCE. For every leaf array, for a varied
 *      set of queries (in-range, out-of-range, exact-match, off-by-one,
 *      and queries at multiple levels), t8_forest_bin_search_lower
 *      returns bit-identical results before vs after cache population.
 *      This is the primary correctness gate: the cached path must
 *      never produce a different index than the scalar path.
 *
 *   2. POPULATOR ACTUALLY POPULATES. After
 *      ensure_linear_id_cache(array, L), get_linear_id_cache(array, L)
 *      must return non-NULL. (Sanity check: if the populator silently
 *      no-ops, the invariance test would still pass trivially.)
 *
 *   3. IDEMPOTENCY. A second ensure_linear_id_cache call at the same
 *      level leaves the cache pointer unchanged — no extra allocation,
 *      no extra population work.
 *
 *   4. STALE-LEVEL REJECTION. get_linear_id_cache(array, L') returns
 *      NULL whenever L' != populated_level, so a future query at a
 *      different level correctly falls through to the scalar path.
 *
 *   5. EXPLICIT INVALIDATION. After
 *      invalidate_linear_id_cache(array), get_linear_id_cache returns
 *      NULL and subsequent bin_search_lower calls still produce
 *      correct results (via the scalar fallback).
 *
 *   6. FAST PATH IS ACTUALLY EXECUTED. A white-box test: install a
 *      deliberately-corrupted cache via set_linear_id_cache and verify
 *      that bin_search_lower returns the result *implied by the
 *      corrupted cache*. This proves the Phase 3b reader is reading
 *      from the cache rather than silently falling through. After
 *      invalidating the corrupted cache, results return to matching
 *      the scalar path.
 *
 *   7. EMPTY-ARRAY SAFETY. ensure_linear_id_cache on an empty array
 *      is a safe no-op: cache remains NULL and bin_search_lower's
 *      pre-existing behaviour on empty arrays is unchanged.
 *
 * Parametrization
 * ---------------
 *
 * We mirror the parametrization of t8_gtest_bin_search.cxx (over all
 * schemes / eclasses / a small range of uniform levels) and exercise
 * BOTH a uniform forest and an adaptively-refined forest. This gives
 * us coverage across:
 *   - The SIMD-batched code paths (T8_ECLASS_TRIANGLE, T8_ECLASS_TET).
 *   - The scalar fallback (every other class).
 *   - Mixed-level adaptive forests (which is the realistic AMR case).
 */

#include <gtest/gtest.h>
#include <test/t8_gtest_schemes.hxx>
#include <t8_eclass/t8_eclass.h>
#include <t8_cmesh/t8_cmesh.h>
#include <t8_forest/t8_forest_general.h>
#include <t8_forest/t8_forest_private.h>
#include <t8_data/t8_containers.h>
#include <t8_schemes/t8_default/t8_default.hxx>
#include "test/t8_cmesh_generator/t8_cmesh_example_sets.hxx"
#include <test/t8_gtest_adapt_callbacks.hxx>
#include <test/t8_gtest_macros.hxx>

#include <algorithm>
#include <vector>

/* Match the t8_gtest_bin_search.cxx convention. */
#if T8_TEST_LEVEL_INT >= 1
#define T8_BIN_SEARCH_CACHE_MAX_LVL 3
#else
#define T8_BIN_SEARCH_CACHE_MAX_LVL 4
#endif

class t8_bin_search_cache_tester: public testing::TestWithParam<std::tuple<std::tuple<int, t8_eclass_t>, int>> {
 protected:
  void
  SetUp () override
  {
    const int scheme_id = std::get<0> (std::get<0> (GetParam ()));
    scheme = create_from_scheme_id (scheme_id);
    const t8_eclass_t tree_class = std::get<1> (std::get<0> (GetParam ()));
    const int level = std::get<1> (GetParam ());
    t8_cmesh_t cmesh = t8_cmesh_new_from_class (tree_class, sc_MPI_COMM_WORLD);

    forest = t8_forest_new_uniform (cmesh, scheme, level, 0, sc_MPI_COMM_WORLD);
    t8_forest_ref (forest);
    int maxlevel = 7;
    const int recursive_adapt = 1;
    /* t8_test_adapt_first_child is the same helper used by
     * t8_gtest_bin_search.cxx; it produces a non-trivial adaptive
     * forest with a mix of levels. */
    forest_adapt = t8_forest_new_adapt (forest, t8_test_adapt_first_child, recursive_adapt, 0, &maxlevel);
  }

  void
  TearDown () override
  {
    if (forest != NULL) {
      t8_forest_unref (&forest);
    }
    if (forest_adapt != NULL) {
      t8_forest_unref (&forest_adapt);
    }
  }

  t8_forest_t forest { NULL };
  t8_forest_t forest_adapt { NULL };
  const t8_scheme *scheme;
};

/* Build a diverse set of query (id, level) pairs for one tree. The
 * intent is to hit many code paths:
 *   - Exact in-range matches at each leaf's own level (must return
 *     that leaf's index).
 *   - Exact in-range matches at the array-wide max level (used by
 *     bin_search_lower in the face-neighbor hot path).
 *   - Off-by-one queries that pull bin_search_lower into the
 *     before-first-element edge (returns -1) and the after-last-element
 *     edge (returns count-1).
 *   - Queries at a different level than the populated cache, so we
 *     exercise the freshness predicate's level-mismatch branch.
 */
struct CacheQuery {
  t8_linearidx_t id;
  int level;
};

static std::vector<CacheQuery>
build_queries_for_tree (t8_forest_t forest, t8_locidx_t itree, int populate_level)
{
  std::vector<CacheQuery> queries;
  const t8_scheme *scheme = t8_forest_get_scheme (forest);
  const t8_eclass_t tree_class = t8_forest_get_tree_class (forest, itree);
  const t8_locidx_t num_elements_in_tree = t8_forest_get_tree_num_leaf_elements (forest, itree);
  queries.reserve (static_cast<size_t> (num_elements_in_tree) * 4 + 4);

  for (t8_locidx_t ielem = 0; ielem < num_elements_in_tree; ++ielem) {
    const t8_element_t *e = t8_forest_get_leaf_element_in_tree (forest, itree, ielem);
    const int level_E = scheme->element_get_level (tree_class, e);

    /* Query at the element's own level — exact match. */
    queries.push_back ({ scheme->element_get_linear_id (tree_class, e, level_E), level_E });

    /* Query at populate_level — exact match if populate_level >= level_E,
     * which is the common case. This is the query level the cache fast
     * path will actually serve. */
    queries.push_back ({ scheme->element_get_linear_id (tree_class, e, populate_level), populate_level });

    /* Off-by-one before the element at populate_level. Hits the
     * "first element greater" branch in std::upper_bound. */
    const t8_linearidx_t id_at_pop = scheme->element_get_linear_id (tree_class, e, populate_level);
    if (id_at_pop > 0) {
      queries.push_back ({ id_at_pop - 1, populate_level });
    }
    /* Off-by-one after — hits the "no element greater" branch. */
    queries.push_back ({ id_at_pop + 1, populate_level });
  }

  /* Extra global edge cases: id=0, id=very_large. id=0 may be in or
   * out of range depending on the tree; either way the result must
   * match between cached and scalar paths. */
  queries.push_back ({ 0, populate_level });
  queries.push_back ({ (t8_linearidx_t) -1, populate_level }); /* max value */

  return queries;
}

/* Determine the level at which the cache should be populated. For
 * uniform forests every element is at the same level; for adaptive
 * forests we use the deepest level present (mirroring the face-neighbor
 * hot path where bin_search_lower is called with forest->maxlevel). */
static int
max_element_level_in_tree (t8_forest_t forest, t8_locidx_t itree)
{
  const t8_scheme *scheme = t8_forest_get_scheme (forest);
  const t8_eclass_t tree_class = t8_forest_get_tree_class (forest, itree);
  const t8_locidx_t num_elements_in_tree = t8_forest_get_tree_num_leaf_elements (forest, itree);
  int level_max = 0;
  for (t8_locidx_t ielem = 0; ielem < num_elements_in_tree; ++ielem) {
    const t8_element_t *e = t8_forest_get_leaf_element_in_tree (forest, itree, ielem);
    level_max = std::max (level_max, scheme->element_get_level (tree_class, e));
  }
  return level_max;
}

/* Core invariance + idempotency test. For each local tree:
 *   - collect baseline (cache-OFF) bin_search_lower results.
 *   - populate the cache, run idempotency / freshness / fast-path
 *     execution sub-checks.
 *   - re-collect results with cache ON.
 *   - assert pairwise equality.
 *   - invalidate, re-collect, assert equality again. */
static void
t8_test_cache_invariance_for_forest (t8_forest_t forest)
{
  const t8_locidx_t num_local_trees = t8_forest_get_num_local_trees (forest);

  for (t8_locidx_t itree = 0; itree < num_local_trees; ++itree) {
    const t8_locidx_t num_elements_in_tree = t8_forest_get_tree_num_leaf_elements (forest, itree);
    if (num_elements_in_tree == 0) {
      continue;
    }

    /* We need the mutable handle to install/invalidate the cache. */
    t8_element_array_t *leaves = t8_forest_get_tree_leaf_element_array_mutable (forest, itree);

    const int populate_level = max_element_level_in_tree (forest, itree);
    const auto queries = build_queries_for_tree (forest, itree, populate_level);

    /* Ensure we start in a clean (no-cache) state regardless of any
     * residual cache from a previous test run on the same forest. */
    t8_element_array_invalidate_linear_id_cache (leaves);
    EXPECT_EQ (t8_element_array_get_linear_id_cache (leaves, populate_level), nullptr);

    /* Pass 1: cache OFF (scalar path). */
    std::vector<t8_locidx_t> baseline;
    baseline.reserve (queries.size ());
    for (const auto &q : queries) {
      baseline.push_back (t8_forest_bin_search_lower (leaves, q.id, q.level));
    }

    /* Populate. */
    t8_forest_element_array_ensure_linear_id_cache (leaves, populate_level);
    const t8_linearidx_t *cache_after_populate = t8_element_array_get_linear_id_cache (leaves, populate_level);
    ASSERT_NE (cache_after_populate, nullptr)
      << "Populator left the cache empty for itree=" << itree
      << " populate_level=" << populate_level
      << " (eclass=" << static_cast<int> (t8_forest_get_tree_class (forest, itree)) << ")";

    /* Idempotency: second ensure call must not change the cache pointer. */
    t8_forest_element_array_ensure_linear_id_cache (leaves, populate_level);
    const t8_linearidx_t *cache_after_second_populate
      = t8_element_array_get_linear_id_cache (leaves, populate_level);
    EXPECT_EQ (cache_after_populate, cache_after_second_populate)
      << "Idempotent ensure_linear_id_cache changed the cache pointer (it should be a no-op).";

    /* Stale-level rejection: the freshness predicate must require a
     * level match, so querying at any other level returns NULL. */
    if (populate_level > 0) {
      EXPECT_EQ (t8_element_array_get_linear_id_cache (leaves, populate_level - 1), nullptr)
        << "Cache populated at level=" << populate_level
        << " but get_linear_id_cache(level-1) returned non-NULL.";
    }
    EXPECT_EQ (t8_element_array_get_linear_id_cache (leaves, populate_level + 1), nullptr)
      << "Cache populated at level=" << populate_level
      << " but get_linear_id_cache(level+1) returned non-NULL.";

    /* Pass 2: cache ON. Cached fast path will fire whenever
     * q.level == populate_level (which most queries satisfy); for the
     * other queries the freshness predicate falls through to scalar
     * — both must still match the baseline. */
    for (size_t i = 0; i < queries.size (); ++i) {
      const t8_locidx_t cached_result = t8_forest_bin_search_lower (leaves, queries[i].id, queries[i].level);
      EXPECT_EQ (cached_result, baseline[i])
        << "Cache vs scalar mismatch for itree=" << itree
        << " query #" << i << " (id=" << static_cast<long long unsigned> (queries[i].id)
        << ", level=" << queries[i].level << "). baseline=" << baseline[i]
        << " cached=" << cached_result << ".";
    }

    /* Explicit invalidation: cache is gone, results still correct. */
    t8_element_array_invalidate_linear_id_cache (leaves);
    EXPECT_EQ (t8_element_array_get_linear_id_cache (leaves, populate_level), nullptr);
    for (size_t i = 0; i < queries.size (); ++i) {
      const t8_locidx_t after_invalidate_result
        = t8_forest_bin_search_lower (leaves, queries[i].id, queries[i].level);
      EXPECT_EQ (after_invalidate_result, baseline[i])
        << "Post-invalidation result mismatch for itree=" << itree
        << " query #" << i << ".";
    }
  }
}

/* White-box test: install a deliberately-corrupted cache and verify
 * that t8_forest_bin_search_lower returns the result implied by the
 * corrupted cache. If the Phase 3b reader silently bypassed the cache
 * we'd instead see the legitimate (scalar) answer, and this test
 * would fail — so a passing test proves execution path coverage.
 *
 * Construction: install an all-zero cache (length == element count,
 * level == populate_level) so the freshness predicate accepts it.
 * For a query against the first element's actual linear_id `id0`:
 *
 *   Scalar path (real elements, sorted ids):
 *     - cache[0]=id0; query=id0 → not strictly greater → continue.
 *     - upper_bound finds first elem with id > id0 → element 1 (since
 *       id0 is unique and elements are sorted by id at this level).
 *     - Returns iterator index 1 − 1 = 0.
 *
 *   Cached path (all-zero buffer):
 *     - cache[0]=0; query=id0.
 *       - If id0 > 0: 0 > id0 is false → continue. upper_bound finds
 *         first entry > id0; none. Returns count − 1.
 *       - If id0 = 0: 0 > 0 is false → continue. upper_bound finds
 *         first entry > 0; none. Returns count − 1.
 *
 *   Therefore: scalar=0, cached=count−1. They differ whenever count
 *   ≥ 2, which we gate on.
 *
 * After invalidation, the legitimate scalar path is restored and the
 * answer goes back to 0.
 */
static void
t8_test_fast_path_actually_reads_cache (t8_forest_t forest)
{
  const t8_locidx_t num_local_trees = t8_forest_get_num_local_trees (forest);

  bool exercised_any = false;

  for (t8_locidx_t itree = 0; itree < num_local_trees; ++itree) {
    const t8_locidx_t num_elements_in_tree = t8_forest_get_tree_num_leaf_elements (forest, itree);
    if (num_elements_in_tree < 2) {
      /* count < 2 makes the cached vs scalar answers coincide on the
       * id0 query, so this tree cannot discriminate fast-path
       * execution. Skip. */
      continue;
    }

    t8_element_array_t *leaves = t8_forest_get_tree_leaf_element_array_mutable (forest, itree);
    const int populate_level = max_element_level_in_tree (forest, itree);
    const t8_scheme *scheme = t8_forest_get_scheme (forest);
    const t8_eclass_t tree_class = t8_forest_get_tree_class (forest, itree);

    /* Start clean. */
    t8_element_array_invalidate_linear_id_cache (leaves);

    const t8_element_t *e0 = t8_forest_get_leaf_element_in_tree (forest, itree, 0);
    const t8_linearidx_t id0 = scheme->element_get_linear_id (tree_class, e0, populate_level);

    /* Scalar baseline: the self-search at the first element must
     * always return index 0. */
    const t8_locidx_t scalar_result = t8_forest_bin_search_lower (leaves, id0, populate_level);
    ASSERT_EQ (scalar_result, 0) << "Self-search at element 0 should return 0 (scalar path).";

    /* Install corrupted cache. set_linear_id_cache takes ownership of
     * the buffer; subsequent invalidate frees it. */
    const size_t count = static_cast<size_t> (num_elements_in_tree);
    t8_linearidx_t *corrupt_buf = T8_ALLOC (t8_linearidx_t, count);
    for (size_t i = 0; i < count; ++i) {
      corrupt_buf[i] = 0;
    }
    t8_element_array_set_linear_id_cache (leaves, corrupt_buf, populate_level, count);
    ASSERT_NE (t8_element_array_get_linear_id_cache (leaves, populate_level), nullptr);

    /* Discriminating assertion: scalar said 0, cached must say
     * count − 1. If we observe 0, the fast path was bypassed and the
     * scalar path executed even though the cache was fresh. */
    const t8_locidx_t cached_result_wrong = t8_forest_bin_search_lower (leaves, id0, populate_level);
    EXPECT_EQ (cached_result_wrong, static_cast<t8_locidx_t> (count - 1))
      << "Corrupted-cache discriminator failed for itree=" << itree
      << " (count=" << count << ", id0=" << static_cast<long long unsigned> (id0)
      << "). cached_result=" << cached_result_wrong
      << " — expected count-1=" << count - 1
      << ". Observed 0 means bin_search_lower did NOT read from the cache.";
    exercised_any = true;

    /* Invalidate and re-check: scalar path restored. */
    t8_element_array_invalidate_linear_id_cache (leaves);
    EXPECT_EQ (t8_element_array_get_linear_id_cache (leaves, populate_level), nullptr);
    const t8_locidx_t after_invalidate_result = t8_forest_bin_search_lower (leaves, id0, populate_level);
    EXPECT_EQ (after_invalidate_result, 0) << "After invalidation, scalar path should return 0 for id0.";
  }

  if (!exercised_any) {
    GTEST_SKIP ()
      << "No tree had count >= 2 to exercise the white-box fast-path assertion. "
         "For typical forests this should not happen — please check the test fixture.";
  }
}

/* Empty-array safety: ensure_linear_id_cache on an empty array must
 * not populate (no allocation, cache stays NULL). Subsequent operations
 * on the cache helpers must be safe. */
static void
t8_test_empty_array_safety (const t8_scheme *scheme, t8_eclass_t tree_class)
{
  t8_element_array_t *empty = t8_element_array_new (scheme, tree_class);
  ASSERT_NE (empty, nullptr);
  EXPECT_EQ (t8_element_array_get_count (empty), 0);

  /* Cache is unpopulated initially. */
  EXPECT_EQ (t8_element_array_get_linear_id_cache (empty, 0), nullptr);

  /* Populate is a no-op on an empty array. */
  t8_forest_element_array_ensure_linear_id_cache (empty, 0);
  EXPECT_EQ (t8_element_array_get_linear_id_cache (empty, 0), nullptr);
  EXPECT_EQ (t8_element_array_get_count (empty), 0);

  /* Invalidate is also idempotent and safe. */
  t8_element_array_invalidate_linear_id_cache (empty);
  EXPECT_EQ (t8_element_array_get_linear_id_cache (empty, 0), nullptr);

  /* Pair with the T8_ALLOC that t8_element_array_new performed. */
  t8_element_array_reset (empty);
  T8_FREE (empty);
}

TEST_P (t8_bin_search_cache_tester, cache_invariance_uniform)
{
  t8_test_cache_invariance_for_forest (forest);
}

TEST_P (t8_bin_search_cache_tester, cache_invariance_adapt)
{
  t8_test_cache_invariance_for_forest (forest_adapt);
}

TEST_P (t8_bin_search_cache_tester, fast_path_executes_uniform)
{
  t8_test_fast_path_actually_reads_cache (forest);
}

TEST_P (t8_bin_search_cache_tester, fast_path_executes_adapt)
{
  t8_test_fast_path_actually_reads_cache (forest_adapt);
}

TEST_P (t8_bin_search_cache_tester, empty_array_safety)
{
  const t8_eclass_t tree_class = std::get<1> (std::get<0> (GetParam ()));
  t8_test_empty_array_safety (scheme, tree_class);
}

INSTANTIATE_TEST_SUITE_P (t8_gtest_bin_search_cache, t8_bin_search_cache_tester,
                          testing::Combine (AllSchemes, testing::Range (0, T8_BIN_SEARCH_CACHE_MAX_LVL)),
                          pretty_print_eclass_scheme_and_level);
