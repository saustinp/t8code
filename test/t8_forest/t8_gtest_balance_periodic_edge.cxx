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
 * Path C v2 gate: t8_forest_balance must satisfy 2:1 across periodic edge
 * and vertex neighbors, not only across face neighbors.
 *
 * Setup:
 *   1. Build the 3D periodic Kuhn cube uniform at level 1.
 *   2. Apply a recursive adapt that refines tree 0 to level 3 and leaves
 *      every other tree at level 1. This creates a 2-level skip across
 *      both face AND periodic-edge/vertex adjacencies.
 *   3. Call t8_forest_balance on the imbalanced forest.
 *
 * Gate:
 *   For every leaf L and every periodic incidence (L -> P) reported by
 *   t8_forest_leaf_periodic_neighbors, assert |level(L) - level(P)| <= 1.
 *
 * Why this distinguishes v2: pre-v2 balance only enforces 2:1 on faces, so
 * leaves sharing only an edge or vertex through periodicity could remain
 * level-skipped. v2 routes those through the new periodic-cache hook so
 * balance honors them too. If the patch regresses, this test fails.
 */

#include <gtest/gtest.h>
#include <t8_eclass/t8_eclass.h>
#include <t8_cmesh/t8_cmesh.h>
#include <t8_cmesh/t8_cmesh_examples.h>
#include <t8_forest/t8_forest_general.h>
#include <t8_forest/t8_forest_geometrical.h>
#include <t8_forest/t8_forest_balance.h>
#include <t8_schemes/t8_default/t8_default.hxx>
#include <t8_schemes/t8_scheme.hxx>

namespace {

/* Free the array allocated by t8_forest_leaf_periodic_neighbors plus the
 * neighbor_leaf copies. */
static void
free_incidences (t8_periodic_incidence_t *inc, int n_inc, const t8_scheme *scheme)
{
  if (n_inc > 0 && inc != nullptr) {
    for (int i = 0; i < n_inc; ++i) {
      scheme->element_destroy (inc[i].neighbor_eclass, 1, &inc[i].neighbor_leaf);
    }
    T8_FREE (inc);
  }
}

/* Adapt callback: refine every element in tree 0 to (at least) level 3,
 * never refine anything else, never coarsen. Driven recursively so a
 * level-1 forest collapses tree 0 down to level 3 in one call. */
static int
adapt_refine_tree0_to_level3 (t8_forest_t /*forest*/, t8_forest_t /*forest_from*/, const t8_locidx_t ltree_id,
                              const t8_eclass_t tree_class, [[maybe_unused]] const t8_locidx_t lelement_id,
                              const t8_scheme *scheme, [[maybe_unused]] const int is_family,
                              [[maybe_unused]] const int num_elements, t8_element_t *elements[])
{
  if (ltree_id != 0) return 0;
  const int level = scheme->element_get_level (tree_class, elements[0]);
  return (level < 3) ? 1 : 0;
}

}  // anonymous namespace

class BalancePeriodicEdge : public ::testing::Test {
 protected:
  void
  SetUp () override
  {
    if (sc_MPI_Comm_size (sc_MPI_COMM_WORLD, &mpi_size) != sc_MPI_SUCCESS) {
      FAIL () << "Failed to get MPI size";
    }
    if (mpi_size > 1) {
      /* t8_forest_new_uniform on MPI_COMM_WORLD SFC-partitions the leaves
       * even when the cmesh is replicated. The periodic cache requires a
       * REPLICATED FOREST (every rank holds the full leaf set) for the
       * per-rank corner hash to be globally complete. To exercise the
       * replicated NP>1 code path correctly, the gtest would need to
       * build the forest on MPI_COMM_SELF — that's a separate test, not
       * a v2-balance gate. Keep the skip here; the replicated MVP is
       * gated by amr_dev integration tests 88/86/87 instead. */
      GTEST_SKIP () << "Replicated NP>1 forests require MPI_COMM_SELF; this "
                       "gtest uses MPI_COMM_WORLD which SFC-partitions. "
                       "Distributed-multi-rank Path C is a future follow-up.";
    }
    scheme = t8_scheme_new_default ();
    /* 3D periodic Kuhn cube: 6 tets at the cmesh level, full x+y+z periodicity. */
    t8_cmesh_t cmesh = t8_cmesh_new_hypercube (T8_ECLASS_TET, sc_MPI_COMM_WORLD,
                                               /*do_bcast=*/0, /*do_partition=*/0,
                                               /*periodic=*/1);
    ASSERT_NE (cmesh, nullptr);
    /* Uniform start: level 1, all trees. */
    t8_forest_t uniform_forest
      = t8_forest_new_uniform (cmesh, scheme, /*level=*/1, /*do_face_ghost=*/1, sc_MPI_COMM_WORLD);
    ASSERT_NE (uniform_forest, nullptr);

    /* Step 1: recursive adapt to drive tree 0 to level 3. */
    t8_forest_t adapted_forest;
    t8_forest_init (&adapted_forest);
    t8_forest_set_adapt (adapted_forest, uniform_forest, adapt_refine_tree0_to_level3, /*recursive=*/1);
    t8_forest_set_ghost (adapted_forest, 1, T8_GHOST_FACES);
    t8_forest_commit (adapted_forest);

    /* Step 2: balance. With v2 the periodic-edge/vertex 2:1 enforcement
     *         fires inside the loop; the resulting forest should satisfy
     *         the gate below. */
    t8_forest_t balanced_forest;
    t8_forest_init (&balanced_forest);
    t8_forest_set_balance (balanced_forest, adapted_forest, /*no_repartition=*/1);
    t8_forest_set_ghost (balanced_forest, 1, T8_GHOST_FACES);
    t8_forest_commit (balanced_forest);

    forest = balanced_forest;
    ASSERT_NE (forest, nullptr);
  }

  void
  TearDown () override
  {
    if (forest != nullptr) {
      t8_forest_unref (&forest);
    }
  }

  int mpi_size = -1;
  const t8_scheme *scheme = nullptr;
  t8_forest_t forest = nullptr;
};

/* Post-balance gate: every periodic-adjacency neighbor pair satisfies 2:1. */
TEST_F (BalancePeriodicEdge, post_balance_periodic_2to1_holds)
{
  const t8_scheme *fscheme = t8_forest_get_scheme (forest);
  const t8_locidx_t n_trees = t8_forest_get_num_local_trees (forest);
  ASSERT_GT (n_trees, 0);

  int violations = 0;
  int n_checked = 0;
  int n_pairs = 0;

  for (t8_locidx_t lt = 0; lt < n_trees; ++lt) {
    const t8_eclass_t lt_class = t8_forest_get_tree_class (forest, lt);
    const t8_locidx_t n_leaves = t8_forest_get_tree_num_leaf_elements (forest, lt);
    for (t8_locidx_t le = 0; le < n_leaves; ++le) {
      const t8_element_t *leaf = t8_forest_get_leaf_element_in_tree (forest, lt, le);
      const int my_level = fscheme->element_get_level (lt_class, leaf);

      t8_periodic_incidence_t *inc = nullptr;
      int n_inc = 0;
      t8_forest_leaf_periodic_neighbors (forest, lt, leaf, &inc, &n_inc, /*forest_is_balanced=*/1);

      for (int i = 0; i < n_inc; ++i) {
        const t8_eclass_t neigh_class = inc[i].neighbor_eclass;
        const int neigh_level = fscheme->element_get_level (neigh_class, inc[i].neighbor_leaf);
        const int diff = std::abs (my_level - neigh_level);
        if (diff > 1) {
          if (violations < 8) {
            ADD_FAILURE () << "Periodic-adjacency 2:1 violated: leaf (tree=" << lt << ", leid=" << le
                           << ", level=" << my_level << ") vs periodic partner at level " << neigh_level
                           << " (incidence type=" << (int) inc[i].incidence_type << ", leaf_entity=" << inc[i].leaf_entity
                           << ", neighbor_entity=" << inc[i].neighbor_entity << ")";
          }
          ++violations;
        }
        ++n_pairs;
      }
      free_incidences (inc, n_inc, fscheme);
      ++n_checked;
    }
  }

  EXPECT_GT (n_pairs, 0) << "Test setup is degenerate: no periodic incidences found at all.";
  EXPECT_EQ (violations, 0) << "Found " << violations << " periodic-adjacency 2:1 violations across "
                            << n_pairs << " pairs (checked " << n_checked << " leaves).";
}

/* Post-balance face check (existing balance contract): every leaf is
 * face-balanced. v2 must not regress this. */
TEST_F (BalancePeriodicEdge, post_balance_face_2to1_holds)
{
  EXPECT_TRUE (t8_forest_is_balanced (forest)) << "Forest reports not-balanced after t8_forest_balance.";
}

/* ------------------------------------------------------------------ */
/* Replicated multi-rank gate (R1, 2026-05-31):                       */
/*                                                                    */
/* Builds the forest on MPI_COMM_SELF so every rank holds the full    */
/* leaf set (replicated forest). At NP=1 this matches BalancePeriodic */
/* Edge exactly. At NP>1 it exercises the R1 code path in             */
/* t8_forest_periodic_cache_new where the (forest->mpisize > 1 &&     */
/* local_num != global_num) short-circuit no longer fires because     */
/* local_num == global_num.                                           */
/* ------------------------------------------------------------------ */
class BalancePeriodicEdgeReplicated : public ::testing::Test {
 protected:
  void
  SetUp () override
  {
    scheme = t8_scheme_new_default ();
    /* COMM_SELF: each rank builds its own private replicated cmesh +
     * forest. No cross-rank dependence in setup. */
    t8_cmesh_t cmesh = t8_cmesh_new_hypercube (T8_ECLASS_TET, sc_MPI_COMM_SELF,
                                               /*do_bcast=*/0, /*do_partition=*/0,
                                               /*periodic=*/1);
    ASSERT_NE (cmesh, nullptr);
    t8_forest_t uniform_forest
      = t8_forest_new_uniform (cmesh, scheme, /*level=*/1, /*do_face_ghost=*/1, sc_MPI_COMM_SELF);
    ASSERT_NE (uniform_forest, nullptr);

    t8_forest_t adapted_forest;
    t8_forest_init (&adapted_forest);
    t8_forest_set_adapt (adapted_forest, uniform_forest, adapt_refine_tree0_to_level3, /*recursive=*/1);
    t8_forest_set_ghost (adapted_forest, 1, T8_GHOST_FACES);
    t8_forest_commit (adapted_forest);

    t8_forest_t balanced_forest;
    t8_forest_init (&balanced_forest);
    t8_forest_set_balance (balanced_forest, adapted_forest, /*no_repartition=*/1);
    t8_forest_set_ghost (balanced_forest, 1, T8_GHOST_FACES);
    t8_forest_commit (balanced_forest);

    forest = balanced_forest;
    ASSERT_NE (forest, nullptr);
  }

  void
  TearDown () override
  {
    if (forest != nullptr) {
      t8_forest_unref (&forest);
    }
  }

  const t8_scheme *scheme = nullptr;
  t8_forest_t forest = nullptr;
};

TEST_F (BalancePeriodicEdgeReplicated, post_balance_periodic_2to1_holds)
{
  const t8_scheme *fscheme = t8_forest_get_scheme (forest);
  const t8_locidx_t n_trees = t8_forest_get_num_local_trees (forest);
  ASSERT_GT (n_trees, 0);

  int violations = 0;
  int n_checked = 0;
  int n_pairs = 0;

  for (t8_locidx_t lt = 0; lt < n_trees; ++lt) {
    const t8_eclass_t lt_class = t8_forest_get_tree_class (forest, lt);
    const t8_locidx_t n_leaves = t8_forest_get_tree_num_leaf_elements (forest, lt);
    for (t8_locidx_t le = 0; le < n_leaves; ++le) {
      const t8_element_t *leaf = t8_forest_get_leaf_element_in_tree (forest, lt, le);
      const int my_level = fscheme->element_get_level (lt_class, leaf);

      t8_periodic_incidence_t *inc = nullptr;
      int n_inc = 0;
      t8_forest_leaf_periodic_neighbors (forest, lt, leaf, &inc, &n_inc, /*forest_is_balanced=*/1);

      for (int i = 0; i < n_inc; ++i) {
        const t8_eclass_t neigh_class = inc[i].neighbor_eclass;
        const int neigh_level = fscheme->element_get_level (neigh_class, inc[i].neighbor_leaf);
        const int diff = std::abs (my_level - neigh_level);
        if (diff > 1) {
          if (violations < 8) {
            ADD_FAILURE () << "Periodic-adjacency 2:1 violated (replicated): leaf (tree=" << lt << ", leid=" << le
                           << ", level=" << my_level << ") vs periodic partner at level " << neigh_level
                           << " (incidence type=" << (int) inc[i].incidence_type << ")";
          }
          ++violations;
        }
        ++n_pairs;
      }
      free_incidences (inc, n_inc, fscheme);
      ++n_checked;
    }
  }

  EXPECT_GT (n_pairs, 0) << "Test setup is degenerate: no periodic incidences found at all.";
  EXPECT_EQ (violations, 0) << "Found " << violations << " periodic-adjacency 2:1 violations across "
                            << n_pairs << " pairs (checked " << n_checked << " leaves).";
}

TEST_F (BalancePeriodicEdgeReplicated, post_balance_face_2to1_holds)
{
  EXPECT_TRUE (t8_forest_is_balanced (forest)) << "Forest reports not-balanced after t8_forest_balance.";
}
