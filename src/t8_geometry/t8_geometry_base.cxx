/*
  This file is part of t8code.
  t8code is a C library to manage a collection (a forest) of multiple
  connected adaptive space-trees of general element classes in parallel.

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

/* In this file we collect the implementations of the geometry base class
 * and its C interface.
 * */

#include <t8_geometry/t8_geometry_base.hxx>
#include <t8_geometry/t8_geometry_base.h>

#include <atomic>
#include <unordered_map>

/* Static-member definitions: one per-thread cache map per process, one
 * global atomic counter handing out unique instance ids for generation
 * checking. See t8_geometry_base.hxx for the design rationale. */
thread_local std::unordered_map<const t8_geometry *, t8_geometry::TLSEntry>
  t8_geometry::tls_cache_;

std::atomic<uint64_t> t8_geometry::next_instance_id_{ 0 };

const char *
t8_geom_get_name (const t8_geometry_c *geom)
{
  T8_ASSERT (geom != NULL);

  return geom->t8_geom_get_name ().c_str ();
}

t8_geometry_type_t
t8_geom_get_type (const t8_geometry_c *geom)
{
  T8_ASSERT (geom != NULL);

  return geom->t8_geom_get_type ();
}

/* Load the id and class of the newly active tree into this thread's
 * cached TLSEntry for this geometry. (Previously these were instance
 * members; now they live in a per-thread cache to remove the
 * cross-tree race exposed by the t8 thread-safety microbench.) */
void
t8_geometry::t8_geom_load_tree_data (const t8_cmesh_t cmesh, const t8_gloidx_t gtreeid)
{
  /* Set active id and eclass via thread-local setters. */
  const t8_locidx_t ltreeid = t8_cmesh_get_local_id (cmesh, gtreeid);
  set_active_tree (gtreeid);
  const t8_locidx_t num_local_trees = t8_cmesh_get_num_local_trees (cmesh);
  t8_eclass_t this_tree_class;
  if (0 <= ltreeid && ltreeid < num_local_trees) {
    this_tree_class = t8_cmesh_get_tree_class (cmesh, ltreeid);
  }
  else {
    this_tree_class = t8_cmesh_get_ghost_class (cmesh, ltreeid - num_local_trees);
  }
  set_active_tree_class (this_tree_class);

  /* Check whether we support this class */
  T8_ASSERT (this_tree_class == T8_ECLASS_VERTEX || this_tree_class == T8_ECLASS_TRIANGLE
             || this_tree_class == T8_ECLASS_TET || this_tree_class == T8_ECLASS_QUAD
             || this_tree_class == T8_ECLASS_HEX || this_tree_class == T8_ECLASS_LINE
             || this_tree_class == T8_ECLASS_PRISM || this_tree_class == T8_ECLASS_PYRAMID);
}
