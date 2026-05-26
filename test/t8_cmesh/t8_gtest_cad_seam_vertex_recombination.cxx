/*
  This file is part of t8code.
  t8code is a C library to manage a collection (a forest) of multiple
  connected adaptive space-trees of general element types in parallel.

  Copyright (C) 2015 the developers

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

/* In this file we test the CAD-aware mesh-recombination path in
 * t8_cmesh_from_msh_file (the readmshfile.cxx "Reference parameter on
 * curve not found" failure mode) for a geometry whose mesh has two
 * adjacent vertices both sitting on parametric seams of a closed CAD
 * edge / surface.
 *
 * Without the fix in t8_cmesh_readmshfile.cxx, the edge-on-curve and
 * edge-on-surface linkage loops never set a non-null reference
 * parameter and the cmesh build aborts before t8_cmesh_from_msh_file
 * returns. With the fix the missing reference falls back to the
 * midpoint of the CAD edge / face parametric range -- by construction
 * interior to the range and therefore not at the seam.
 *
 * The fixture cad_seam_vertex_recombination.{brep,msh} is a 3D cut-sphere
 * geometry (sphere of radius 5 cut by the z = 0 plane, meshed by gmsh
 * with one node at each pole of the sphere's (u, v) parameterization).
 * That places two mesh nodes at seam vertices of the closed sphere /
 * circle CAD entities -- the exact configuration this regression test
 * targets.
 *
 * If t8code was not configured with -DT8CODE_ENABLE_OCC=ON then this
 * test does nothing and is always passed.
 */

#include <gtest/gtest.h>
#include <unistd.h> /* Needed to check for file access */
#include <string>
#include <t8.h>
#include <t8_eclass/t8_eclass.h>
#include <t8_cmesh/t8_cmesh.h>
#include <t8_cmesh/t8_cmesh_io/t8_cmesh_readmshfile.h>
#include "t8_test_data_dir.h"

TEST (t8_cmesh_cad_seam_vertex_recombination, cut_sphere_useT8NLElem)
{
#if T8_ENABLE_OCC

  const std::string fileprefix = std::string (T8_TEST_DATA_DIR) + "/cad_seam_vertex_recombination";
  char filename[BUFSIZ];

  snprintf (filename, BUFSIZ, "%s.msh", fileprefix.c_str ());
  ASSERT_FALSE (access (filename, R_OK)) << "Could not open file " << filename;
  snprintf (filename, BUFSIZ, "%s.brep", fileprefix.c_str ());
  ASSERT_FALSE (access (filename, R_OK)) << "Could not open file " << filename;

  t8_debugf ("Checking CAD seam-vertex recombination fallback ...\n");

  /* t8_cmesh_from_msh_file (prefix, partition, comm, dim, master, use_cad_geometry).
   *   partition = 0 -> non-partitioned read
   *   dim       = 3 -> the fixture is a 3D tet mesh
   *   master    = 0 -> rank 0 reads (default)
   *   use_cad_geometry = 1 -> exercise the CAD-aware path we are fixing
   *
   * Without the fix this call aborts the process via SIGSEGV after t8 logs
   * "Error during mesh-cad recombination: Reference parameter on curve not found.";
   * gtest will then report this test as failed-by-abort. With the fix the
   * call returns a valid cmesh. */
  t8_cmesh_t cmesh = t8_cmesh_from_msh_file (fileprefix.c_str (), 0, sc_MPI_COMM_WORLD, 3, 0, 1);

  ASSERT_TRUE (cmesh != NULL) << "t8_cmesh_from_msh_file returned NULL for the CAD seam-vertex fixture.";
  ASSERT_TRUE (t8_cmesh_is_committed (cmesh))
    << "t8_cmesh_from_msh_file produced a non-committed cmesh for the CAD seam-vertex fixture.";

  t8_cmesh_destroy (&cmesh);

#else
  t8_global_productionf ("This version of t8code is not compiled with occ support.\n");
#endif
}
