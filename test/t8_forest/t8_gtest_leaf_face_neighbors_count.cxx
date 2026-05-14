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
 * Tests for t8_forest_leaf_face_neighbors_count (Phase F slim count-only
 * face-neighbor API).
 *
 * The single, decisive correctness contract is:
 *
 *     For every leaf, for every face of that leaf, the integer returned
 *     by t8_forest_leaf_face_neighbors_count MUST equal the value of
 *     `*num_neighbors` produced by the full t8_forest_leaf_face_neighbors
 *     on the same input.
 *
 * If that contract holds across all scheme/eclass/level combinations on
 * both uniform and adaptive forests, the slim API is a drop-in for any
 * caller that previously consulted only `*num_neighbors`. Phase F's
 * caller in amr_dev's remove_hanging_nodes_impl is exactly such a caller,
 * which is why this pairwise-agreement test is the gating verification
 * before we wire amr_dev to call the slim version.
 *
 * Coverage matrix:
 *   - Parametrized over (scheme, eclass, uniform_level) per the project
 *     convention (AllSchemes × Range[0, T8_BIN_SEARCH_CACHE_MAX_LVL))
 *     used by t8_gtest_bin_search.cxx and t8_gtest_bin_search_cache.cxx.
 *   - Both a uniform forest and an adaptively-refined forest (via
 *     t8_test_adapt_first_child with maxlevel=7) are tested per param.
 *   - All leaves, all faces, including boundary faces.
 *
 * No SKIPs are expected — every leaf-face pair is a valid input.
 */

#include <gtest/gtest.h>
#include <test/t8_gtest_schemes.hxx>
#include <t8_eclass/t8_eclass.h>
#include <t8_cmesh/t8_cmesh.h>
#include <t8_forest/t8_forest_general.h>
#include <t8_forest/t8_forest_private.h>
#include <t8_schemes/t8_default/t8_default.hxx>
#include "test/t8_cmesh_generator/t8_cmesh_example_sets.hxx"
#include <test/t8_gtest_adapt_callbacks.hxx>
#include <test/t8_gtest_macros.hxx>

#if T8_TEST_LEVEL_INT >= 1
#define T8_LEAF_FN_COUNT_MAX_LVL 3
#else
#define T8_LEAF_FN_COUNT_MAX_LVL 4
#endif

class t8_leaf_face_neighbors_count_tester
  : public testing::TestWithParam<std::tuple<std::tuple<int, t8_eclass_t>, int>> {
 protected:
  void
  SetUp () override
  {
    const int scheme_id = std::get<0> (std::get<0> (GetParam ()));
    scheme = create_from_scheme_id (scheme_id);
    const t8_eclass_t tree_class = std::get<1> (std::get<0> (GetParam ()));
    const int level = std::get<1> (GetParam ());
    t8_cmesh_t cmesh = t8_cmesh_new_from_class (tree_class, sc_MPI_COMM_WORLD);

    /* do_face_ghost = 1: t8_forest_leaf_face_neighbors and its slim
     * count-only variant both require the ghost structure to be
     * present when MPI size > 1. Without it, the test would abort
     * at the first multi-rank face probe. We pay the modest ghost-
     * exchange cost in the test fixture so both np=1 and np=4 runs
     * exercise the same forests with identical contracts. */
    forest = t8_forest_new_uniform (cmesh, scheme, level, /*do_face_ghost=*/1, sc_MPI_COMM_WORLD);
    t8_forest_ref (forest);
    int maxlevel = 7;
    const int recursive_adapt = 1;
    forest_adapt
      = t8_forest_new_adapt (forest, t8_test_adapt_first_child, recursive_adapt, /*do_face_ghost=*/1, &maxlevel);
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

/* Walk every leaf-face pair in the forest, invoke both the slim and the
 * full face-neighbor APIs, and assert the returned `num_neighbors`
 * agrees. Track an aggregate count of probes so we can verify the test
 * actually exercised some faces (not silently a no-op). */
static void
t8_test_slim_vs_full_for_forest (t8_forest_t forest)
{
  const t8_locidx_t num_local_trees = t8_forest_get_num_local_trees (forest);
  const t8_scheme *scheme = t8_forest_get_scheme (forest);

  size_t total_probes = 0;
  size_t boundary_count = 0;
  size_t regular_count = 0;
  size_t hanging_count = 0;

  for (t8_locidx_t itree = 0; itree < num_local_trees; ++itree) {
    const t8_locidx_t num_elements_in_tree = t8_forest_get_tree_num_leaf_elements (forest, itree);
    const t8_eclass_t tree_class = t8_forest_get_tree_class (forest, itree);

    for (t8_locidx_t ielem = 0; ielem < num_elements_in_tree; ++ielem) {
      const t8_element_t *leaf = t8_forest_get_leaf_element_in_tree (forest, itree, ielem);
      const int num_faces = scheme->element_get_num_faces (tree_class, leaf);

      for (int face = 0; face < num_faces; ++face) {
        /* Full API: we copy its outputs into local variables and then
         * free them, exactly as the production caller in amr_dev's
         * remove_hanging_nodes_impl does. */
        t8_element_t **neighbors = NULL;
        int *dual_faces = NULL;
        int full_num_neighbors = 0;
        t8_locidx_t *neigh_ids = NULL;
        t8_eclass_t neigh_eclass;
        t8_forest_leaf_face_neighbors (forest, itree, leaf, &neighbors, face, &dual_faces, &full_num_neighbors,
                                       &neigh_ids, &neigh_eclass, /* balanced */ 1);

        /* Slim API: independent invocation on the same inputs. The slim
         * version must be reentrant and side-effect-free relative to
         * subsequent full calls (this is implicit in the contract; the
         * fact that the next full call works on a fresh leaf-face pair
         * tests that there is no lingering shared state). */
        const int slim_num_neighbors = t8_forest_leaf_face_neighbors_count (forest, itree, leaf, face);

        EXPECT_EQ (slim_num_neighbors, full_num_neighbors)
          << "Slim/full mismatch at itree=" << itree << " ielem=" << ielem << " face=" << face
          << " (tree_eclass=" << static_cast<int> (tree_class) << "). "
          << "slim=" << slim_num_neighbors << " full=" << full_num_neighbors << ".";

        if (full_num_neighbors == 0) {
          ++boundary_count;
        }
        else if (full_num_neighbors == 1) {
          ++regular_count;
        }
        else {
          ++hanging_count;
        }
        ++total_probes;

        /* Full API cleanup, mirroring the production caller. */
        if (full_num_neighbors > 0) {
          scheme->element_destroy (neigh_eclass, full_num_neighbors, neighbors);
          T8_FREE (neighbors);
          T8_FREE (neigh_ids);
          T8_FREE (dual_faces);
        }
      }
    }
  }

  /* If no leaf-face probes were exercised, the only legitimate cause
   * is an eclass with zero faces (T8_ECLASS_VERTEX, which is a point
   * and has no face structure). For every other eclass, zero probes
   * would indicate a fixture bug; we surface the case with SKIP so
   * the report distinguishes "no work to do" from "no probes hit." */
  if (total_probes == 0u) {
    GTEST_SKIP () << "Forest contains no leaf-face probes (typically T8_ECLASS_VERTEX, "
                     "which has zero faces). Slim API has nothing to verify.";
  }

  /* Suppress unused-variable warnings for the per-category counters.
   * They are kept (rather than removed) because a future diagnostic
   * may want to print them on failure, and the cost of maintaining
   * them is one increment per probe. */
  (void) boundary_count;
  (void) regular_count;
  (void) hanging_count;
}

TEST_P (t8_leaf_face_neighbors_count_tester, slim_vs_full_uniform)
{
  t8_test_slim_vs_full_for_forest (forest);
}

TEST_P (t8_leaf_face_neighbors_count_tester, slim_vs_full_adapt)
{
  t8_test_slim_vs_full_for_forest (forest_adapt);
}

INSTANTIATE_TEST_SUITE_P (t8_gtest_leaf_face_neighbors_count, t8_leaf_face_neighbors_count_tester,
                          testing::Combine (AllSchemes, testing::Range (0, T8_LEAF_FN_COUNT_MAX_LVL)),
                          pretty_print_eclass_scheme_and_level);
