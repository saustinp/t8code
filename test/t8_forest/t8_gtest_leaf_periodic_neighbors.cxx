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
 * Tests for t8_forest_leaf_periodic_neighbors — the periodic-aware companion
 * to t8_forest_leaf_face_neighbors.
 *
 * Contract (single-rank, balanced forest):
 *
 *   1. On a NON-periodic cmesh, every leaf reports zero periodic incidences.
 *   2. On a PERIODIC cmesh, leaves that touch a periodic seam (face/edge/
 *      vertex on the seam) report a non-empty incidence list. Leaves that
 *      do not touch any periodic seam still report zero.
 *   3. Symmetry: if leaf A reports leaf B as a periodic neighbor, then querying
 *      leaf B reports leaf A as a periodic neighbor with the dual entity
 *      indices (face↔face, edge↔edge, vertex↔vertex).
 *   4. Memory contract: callee allocates the incidence array and the neighbor
 *      element copies; caller must T8_FREE the array and destroy each leaf
 *      via the scheme.
 *
 * Fixture: 3D periodic tet hypercube (Kuhn cube), uniform-refined to level 2.
 * This is the workload Path C targets: the cube-edge confluence case where
 * 2-2 and 1-3 vertex splits produce edge-only / vertex-only periodic
 * adjacencies that face_neighbors cannot see.
 */

#include <gtest/gtest.h>
#include <t8_eclass/t8_eclass.h>
#include <t8_cmesh/t8_cmesh.h>
#include <t8_cmesh/t8_cmesh_examples.h>
#include <t8_forest/t8_forest_general.h>
#include <t8_forest/t8_forest_geometrical.h>
#include <t8_schemes/t8_default/t8_default.hxx>

namespace {

/* Helper: destroy one t8_periodic_incidence_t array (call after each query). */
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

}  // anonymous namespace

class PeriodicNeighborsKuhnCube : public ::testing::Test {
 protected:
  void
  SetUp () override
  {
    if (sc_MPI_Comm_size (sc_MPI_COMM_WORLD, &mpi_size) != sc_MPI_SUCCESS) {
      FAIL () << "Failed to get MPI size";
    }
    if (mpi_size > 1) {
      GTEST_SKIP () << "Path C v0 is single-rank only; multi-rank is a follow-up.";
    }
    scheme = t8_scheme_new_default ();
    /* 3D periodic Kuhn cube: 6 tets, full x+y+z periodicity. */
    t8_cmesh_t cmesh = t8_cmesh_new_hypercube (T8_ECLASS_TET, sc_MPI_COMM_WORLD,
                                               /*do_bcast=*/0, /*do_partition=*/0,
                                               /*periodic=*/1);
    ASSERT_NE (cmesh, nullptr);
    forest = t8_forest_new_uniform (cmesh, scheme, kLevel, /*do_face_ghost=*/1, sc_MPI_COMM_WORLD);
    ASSERT_NE (forest, nullptr);
  }

  void
  TearDown () override
  {
    if (forest != nullptr) {
      t8_forest_unref (&forest);
    }
  }

  static constexpr int kLevel = 2;
  int mpi_size = -1;
  const t8_scheme *scheme = nullptr;
  t8_forest_t forest = nullptr;
};

/* Test 1: every leaf gets a non-negative incidence count and pointer pair
 * agrees with num_incidences (i.e. either both null/zero or both non-null/
 * positive). Catches obvious malloc/contract bugs. */
TEST_F (PeriodicNeighborsKuhnCube, basic_contract)
{
  const t8_locidx_t n_trees = t8_forest_get_num_local_trees (forest);
  ASSERT_GT (n_trees, 0);

  int n_leaves_total = 0;
  int n_with_incidences = 0;

  for (t8_locidx_t lt = 0; lt < n_trees; ++lt) {
    const t8_locidx_t n_leaves = t8_forest_get_tree_num_leaf_elements (forest, lt);
    for (t8_locidx_t le = 0; le < n_leaves; ++le) {
      const t8_element_t *leaf = t8_forest_get_leaf_element_in_tree (forest, lt, le);
      t8_periodic_incidence_t *inc = nullptr;
      int n_inc = -1;
      t8_forest_leaf_periodic_neighbors (forest, lt, leaf, &inc, &n_inc, /*forest_is_balanced=*/1);
      ASSERT_GE (n_inc, 0);
      if (n_inc == 0) {
        ASSERT_EQ (inc, nullptr) << "n_inc=0 but array non-null at (lt=" << lt << ", le=" << le << ")";
      }
      else {
        ASSERT_NE (inc, nullptr) << "n_inc>0 but array null at (lt=" << lt << ", le=" << le << ")";
        for (int i = 0; i < n_inc; ++i) {
          ASSERT_NE (inc[i].neighbor_leaf, nullptr);
          ASSERT_GE (inc[i].neighbor_local_idx, 0);
          ASSERT_GE (inc[i].neighbor_ltreeid, 0);
          ASSERT_TRUE (inc[i].incidence_type == T8_PERIODIC_INCIDENCE_FACE
                       || inc[i].incidence_type == T8_PERIODIC_INCIDENCE_EDGE
                       || inc[i].incidence_type == T8_PERIODIC_INCIDENCE_VERTEX);
        }
        ++n_with_incidences;
      }
      free_incidences (inc, n_inc, scheme);
      ++n_leaves_total;
    }
  }

  /* On a fully periodic cube at level 2, only leaves that touch the tree
   * boundary (= cube faces, which are all periodic) participate in periodic
   * adjacency. Interior sub-tets of a refined tree don't. So the gate is
   * the weaker "at least some leaves have incidences" (combined with the
   * per-incidence validity checks above). The symmetry test exercises the
   * cross-leaf correctness contract. */
  EXPECT_GT (n_with_incidences, 0)
      << "Expected at least some leaves on a fully-periodic cube to have a "
         "periodic incidence; got 0 / " << n_leaves_total;
  EXPECT_LE (n_with_incidences, n_leaves_total);
}

/* Test 2: symmetry. For each incidence (A -> B, entity_a -> entity_b),
 * the query on B must return A as a neighbor with the entities swapped. */
TEST_F (PeriodicNeighborsKuhnCube, symmetry)
{
  /* Build a stable (ltreeid, local_in_tree) → linear_idx mapping. */
  const t8_locidx_t n_trees = t8_forest_get_num_local_trees (forest);
  std::vector<std::vector<int>> tree_offsets (n_trees);
  int total = 0;
  for (t8_locidx_t lt = 0; lt < n_trees; ++lt) {
    const t8_locidx_t n_leaves = t8_forest_get_tree_num_leaf_elements (forest, lt);
    tree_offsets[lt].resize (n_leaves);
    for (t8_locidx_t le = 0; le < n_leaves; ++le) {
      tree_offsets[lt][le] = total++;
    }
  }

  /* First pass: collect (src_linear, dst_linear, type, ent_a, ent_b). */
  struct IncRow {
    int src_lin, dst_lin;
    int type;
    int ent_a, ent_b;
  };
  std::vector<IncRow> rows;
  rows.reserve (4096);

  for (t8_locidx_t lt = 0; lt < n_trees; ++lt) {
    const t8_locidx_t n_leaves = t8_forest_get_tree_num_leaf_elements (forest, lt);
    for (t8_locidx_t le = 0; le < n_leaves; ++le) {
      const t8_element_t *leaf = t8_forest_get_leaf_element_in_tree (forest, lt, le);
      const int src_lin = tree_offsets[lt][le];
      t8_periodic_incidence_t *inc = nullptr;
      int n_inc = 0;
      t8_forest_leaf_periodic_neighbors (forest, lt, leaf, &inc, &n_inc, 1);

      for (int i = 0; i < n_inc; ++i) {
        const int neigh_lt = (int) inc[i].neighbor_ltreeid;
        ASSERT_GE (neigh_lt, 0);
        ASSERT_LT (neigh_lt, (int) n_trees);
        /* Convert neighbor_local_idx (forest-wide tree-offset + in-tree
         * idx) back to in-tree idx. */
        const t8_locidx_t tree_offset
            = t8_forest_get_tree_element_offset (forest, (t8_locidx_t) neigh_lt);
        const t8_locidx_t neigh_le = inc[i].neighbor_local_idx - tree_offset;
        ASSERT_GE (neigh_le, 0);
        ASSERT_LT (neigh_le, t8_forest_get_tree_num_leaf_elements (forest, (t8_locidx_t) neigh_lt));
        const int dst_lin = tree_offsets[neigh_lt][neigh_le];
        rows.push_back ({ src_lin, dst_lin, (int) inc[i].incidence_type,
                          inc[i].leaf_entity, inc[i].neighbor_entity });
      }
      free_incidences (inc, n_inc, scheme);
    }
  }

  ASSERT_FALSE (rows.empty ()) << "Expected at least one incidence in a periodic cube.";

  /* Second pass: for every (src, dst, type, a, b) verify (dst, src, type, b, a) exists. */
  std::set<std::tuple<int, int, int, int, int>> row_set;
  for (const auto &r : rows) {
    row_set.emplace (r.src_lin, r.dst_lin, r.type, r.ent_a, r.ent_b);
  }
  for (const auto &r : rows) {
    /* Vertex/edge/face: swapped (a,b) must be a valid row.
     * NOTE: leaf_entity = -1 means "whole element" — those don't need to
     * be symmetric in the strict sense; skip. */
    if (r.ent_a < 0 || r.ent_b < 0) continue;
    const auto reverse_key = std::make_tuple (r.dst_lin, r.src_lin, r.type, r.ent_b, r.ent_a);
    EXPECT_TRUE (row_set.count (reverse_key))
        << "Symmetry violation: (" << r.src_lin << "→" << r.dst_lin
        << ", type=" << r.type << ", " << r.ent_a << "/" << r.ent_b
        << ") has no reverse (" << r.dst_lin << "→" << r.src_lin
        << ", " << r.ent_b << "/" << r.ent_a << ")";
  }
}

/* Test 3: non-periodic baseline — on a NON-periodic hypercube tet cmesh,
 * every leaf must report zero incidences (no periodic seams). */
TEST (PeriodicNeighborsNonPeriodic, returns_zero)
{
  int mpi_size = 1;
  sc_MPI_Comm_size (sc_MPI_COMM_WORLD, &mpi_size);
  if (mpi_size > 1) {
    GTEST_SKIP () << "Path C v0 is single-rank only.";
  }
  const t8_scheme *scheme = t8_scheme_new_default ();
  t8_cmesh_t cmesh = t8_cmesh_new_hypercube (T8_ECLASS_TET, sc_MPI_COMM_WORLD,
                                             /*do_bcast=*/0, /*do_partition=*/0,
                                             /*periodic=*/0);
  ASSERT_NE (cmesh, nullptr);
  t8_forest_t forest = t8_forest_new_uniform (cmesh, scheme, /*level=*/2,
                                              /*do_face_ghost=*/1, sc_MPI_COMM_WORLD);
  ASSERT_NE (forest, nullptr);

  const t8_locidx_t n_trees = t8_forest_get_num_local_trees (forest);
  int total_inc = 0;
  for (t8_locidx_t lt = 0; lt < n_trees; ++lt) {
    const t8_locidx_t n_leaves = t8_forest_get_tree_num_leaf_elements (forest, lt);
    for (t8_locidx_t le = 0; le < n_leaves; ++le) {
      const t8_element_t *leaf = t8_forest_get_leaf_element_in_tree (forest, lt, le);
      t8_periodic_incidence_t *inc = nullptr;
      int n_inc = 0;
      t8_forest_leaf_periodic_neighbors (forest, lt, leaf, &inc, &n_inc, 1);
      total_inc += n_inc;
      free_incidences (inc, n_inc, scheme);
    }
  }
  EXPECT_EQ (total_inc, 0) << "Non-periodic cube must yield zero periodic incidences.";
  t8_forest_unref (&forest);
}
