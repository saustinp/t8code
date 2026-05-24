/*
  This file is part of t8code.
  t8code is a C library to manage a collection (a forest) of multiple
  connected adaptive space-trees of general element types in parallel.

  Copyright (C) 2025 the developers

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

/* Tests for the gmsh v4 $Periodic block import path. We use a fixture
 * mesh (test_msh_file_vers4_periodic.msh) — a 32x32 transfinite-
 * triangulated [-5,5]^2 square with BOTH right↔left and top↔bottom
 * periodicity declared via gmsh's Periodic Curve directives. The
 * expected post-import topology is a 2-torus: NO boundary faces remain.
 */

#include <gtest/gtest.h>
#include <unistd.h>
#include <cmath>
#include <t8.h>
#include <t8_eclass/t8_eclass.h>
#include <t8_cmesh/t8_cmesh.h>
#include <t8_cmesh/t8_cmesh_io/t8_cmesh_readmshfile.h>
#include "t8_test_data_dir.h"
#include <string>

namespace {

/* Open the fixture cmesh. Caller owns; destroy via t8_cmesh_destroy. */
static t8_cmesh_t
open_periodic_fixture ()
{
  const std::string fileprefix = std::string (T8_TEST_DATA_DIR) + "/test_msh_file_vers4_periodic";
  char filename[BUFSIZ];
  snprintf (filename, BUFSIZ, "%s.msh", fileprefix.c_str ());

  EXPECT_FALSE (access (filename, R_OK)) << "Could not open fixture " << filename;
  /* dim=2, partition=0, mpiproc=0, use_cad=0 */
  t8_cmesh_t cmesh = t8_cmesh_from_msh_file (fileprefix.c_str (), 0, sc_MPI_COMM_WORLD, 2, 0, 0);
  EXPECT_TRUE (cmesh != NULL) << "t8_cmesh_from_msh_file returned NULL for " << filename;
  return cmesh;
}

}  // namespace

/* Test 1: basic import. The .msh has 2048 triangles (32x32x2). */
TEST (t8_cmesh_readmshfile_periodic, basic_import)
{
  t8_cmesh_t cmesh = open_periodic_fixture ();
  ASSERT_TRUE (cmesh != NULL);

  const t8_locidx_t num_trees = t8_cmesh_get_num_trees (cmesh);
  EXPECT_EQ (num_trees, 2048) << "Expected 32x32x2 = 2048 trees (transfinite split).";

  t8_cmesh_destroy (&cmesh);
}

/* Test 2: doubly-periodic square has ZERO boundary faces. Every face
 * of every tree should have a neighbor (either interior tree-tree, or
 * periodic wraparound). If any boundary face remains, the periodic
 * import didn't replace it with a join. */
TEST (t8_cmesh_readmshfile_periodic, no_boundary_faces)
{
  t8_cmesh_t cmesh = open_periodic_fixture ();
  ASSERT_TRUE (cmesh != NULL);

  const t8_locidx_t num_trees = t8_cmesh_get_num_trees (cmesh);
  int boundary_count = 0;
  for (t8_locidx_t lt = 0; lt < num_trees; ++lt) {
    /* All trees are TRIs (3 faces). */
    for (int face = 0; face < 3; ++face) {
      if (t8_cmesh_tree_face_is_boundary (cmesh, lt, face)) {
        ++boundary_count;
      }
    }
  }
  EXPECT_EQ (boundary_count, 0) << "Doubly-periodic square should have 0 boundary faces; "
                                << "found " << boundary_count;

  t8_cmesh_destroy (&cmesh);
}

/* Test 3: face-neighbor wraparound geometry sanity for x-periodic side.
 * Walk all trees; for each tree with a vertex at x≈+5 (right boundary)
 * and an opposite-vertex y-coord inside (-5,+5), confirm at least one
 * face's neighbor tree has a vertex at x≈-5 (left boundary) with the
 * SAME y-coord (modulo tolerance). */
TEST (t8_cmesh_readmshfile_periodic, x_periodic_neighbor_geometry)
{
  t8_cmesh_t cmesh = open_periodic_fixture ();
  ASSERT_TRUE (cmesh != NULL);
  const t8_locidx_t num_trees = t8_cmesh_get_num_trees (cmesh);

  constexpr double TOL = 1e-9;
  constexpr double L_X = 10.0;  /* domain length in x: [-5, 5] */

  int wraparound_found = 0;
  for (t8_locidx_t lt = 0; lt < num_trees; ++lt) {
    const double *v = t8_cmesh_get_tree_vertices (cmesh, lt);
    if (v == NULL) continue;

    /* TRI has 3 vertices, each 3 doubles. */
    bool has_right_edge = false;
    for (int i = 0; i < 3; ++i) {
      if (std::abs (v[3 * i + 0] - 5.0) < TOL) {
        has_right_edge = true;
        break;
      }
    }
    if (!has_right_edge) continue;

    /* For each face, check whether its neighbor lives on the left edge. */
    for (int face = 0; face < 3; ++face) {
      int dual_face, orientation;
      const t8_locidx_t nb = t8_cmesh_get_face_neighbor (cmesh, lt, face, &dual_face, &orientation);
      if (nb < 0 || nb == lt) continue;  /* boundary or self-loop */

      const double *vn = t8_cmesh_get_tree_vertices (cmesh, nb);
      if (vn == NULL) continue;
      /* Does the neighbor have ANY vertex on the left edge (x≈-5)? */
      for (int j = 0; j < 3; ++j) {
        if (std::abs (vn[3 * j + 0] + 5.0) < TOL) {
          /* Confirm the wraparound: at least one (right_v, left_v)
           * pair should share y-coordinate (within tol). */
          for (int i = 0; i < 3; ++i) {
            if (std::abs (v[3 * i + 0] - 5.0) > TOL) continue;
            if (std::abs (v[3 * i + 1] - vn[3 * j + 1]) < TOL) {
              ++wraparound_found;
            }
          }
        }
      }
    }
  }
  /* With 32 transfinite segments per side, the right boundary touches
   * at least 32 distinct (tree, face) pairs, each pairing with a left
   * neighbor at matching y. So we expect at least a few dozen
   * wraparound matches. */
  EXPECT_GT (wraparound_found, 30) << "Expected many right↔left wraparound matches; got "
                                   << wraparound_found;
  (void) L_X;  /* documentation only */

  t8_cmesh_destroy (&cmesh);
}

/* Test 4: face-neighbor wraparound geometry sanity for y-periodic side. */
TEST (t8_cmesh_readmshfile_periodic, y_periodic_neighbor_geometry)
{
  t8_cmesh_t cmesh = open_periodic_fixture ();
  ASSERT_TRUE (cmesh != NULL);
  const t8_locidx_t num_trees = t8_cmesh_get_num_trees (cmesh);

  constexpr double TOL = 1e-9;

  int wraparound_found = 0;
  for (t8_locidx_t lt = 0; lt < num_trees; ++lt) {
    const double *v = t8_cmesh_get_tree_vertices (cmesh, lt);
    if (v == NULL) continue;

    bool has_top_edge = false;
    for (int i = 0; i < 3; ++i) {
      if (std::abs (v[3 * i + 1] - 5.0) < TOL) {
        has_top_edge = true;
        break;
      }
    }
    if (!has_top_edge) continue;

    for (int face = 0; face < 3; ++face) {
      int dual_face, orientation;
      const t8_locidx_t nb = t8_cmesh_get_face_neighbor (cmesh, lt, face, &dual_face, &orientation);
      if (nb < 0 || nb == lt) continue;

      const double *vn = t8_cmesh_get_tree_vertices (cmesh, nb);
      if (vn == NULL) continue;
      for (int j = 0; j < 3; ++j) {
        if (std::abs (vn[3 * j + 1] + 5.0) < TOL) {
          for (int i = 0; i < 3; ++i) {
            if (std::abs (v[3 * i + 1] - 5.0) > TOL) continue;
            if (std::abs (v[3 * i + 0] - vn[3 * j + 0]) < TOL) {
              ++wraparound_found;
            }
          }
        }
      }
    }
  }
  EXPECT_GT (wraparound_found, 30) << "Expected many top↔bottom wraparound matches; got "
                                   << wraparound_found;

  t8_cmesh_destroy (&cmesh);
}

/* Test 5: invariant — total in-link + out-link counts for boundary
 * trees. The 4 corner trees on the original mesh should now each have
 * 3 face-neighbors (no boundary faces). Sample a corner tree by
 * looking for one whose vertices include corner (5, 5) and confirm
 * all 3 faces have valid neighbors. */
TEST (t8_cmesh_readmshfile_periodic, corner_tree_full_neighborhood)
{
  t8_cmesh_t cmesh = open_periodic_fixture ();
  ASSERT_TRUE (cmesh != NULL);
  const t8_locidx_t num_trees = t8_cmesh_get_num_trees (cmesh);

  constexpr double TOL = 1e-9;

  t8_locidx_t corner_tree = -1;
  for (t8_locidx_t lt = 0; lt < num_trees; ++lt) {
    const double *v = t8_cmesh_get_tree_vertices (cmesh, lt);
    if (v == NULL) continue;
    for (int i = 0; i < 3; ++i) {
      if (std::abs (v[3 * i + 0] - 5.0) < TOL && std::abs (v[3 * i + 1] - 5.0) < TOL) {
        corner_tree = lt;
        break;
      }
    }
    if (corner_tree >= 0) break;
  }
  ASSERT_GE (corner_tree, 0) << "No tree found with vertex at (5,5); fixture has changed.";

  for (int face = 0; face < 3; ++face) {
    int dual_face, orientation;
    const t8_locidx_t nb = t8_cmesh_get_face_neighbor (cmesh, corner_tree, face, &dual_face, &orientation);
    EXPECT_GE (nb, 0) << "Corner-tree face " << face
                      << " has no neighbor — periodic wraparound at corner failed.";
    EXPECT_FALSE (t8_cmesh_tree_face_is_boundary (cmesh, corner_tree, face))
      << "Corner-tree face " << face << " still marked boundary.";
  }

  t8_cmesh_destroy (&cmesh);
}
