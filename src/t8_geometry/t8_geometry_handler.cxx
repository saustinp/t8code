/*
  This file is part of t8code.
  t8code is a C library to manage a collection (a forest) of multiple
  connected adaptive space-trees of general element classes in parallel.

  Copyright (C) 2024 the developers

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

#include <t8_geometry/t8_geometry_handler.hxx>
#include <t8_cmesh/t8_cmesh.h>
#include <t8_cmesh/t8_cmesh_internal/t8_cmesh_types.h>
#include <t8_cmesh/t8_cmesh_geometry.hxx>
#include <t8_geometry/t8_geometry.h>
#include <t8_geometry/t8_geometry_implementations/t8_geometry_zero.hxx>
#include <t8_geometry/t8_geometry_implementations/t8_geometry_linear_axis_aligned.hxx>
#include <t8_geometry/t8_geometry_implementations/t8_geometry_linear.hxx>
#if T8_ENABLE_OCC
#include <t8_geometry/t8_geometry_implementations/t8_geometry_cad.hxx>
#endif
#include <t8_geometry/t8_geometry_implementations/t8_geometry_examples.hxx>
#include <t8_geometry/t8_geometry_implementations/t8_geometry_analytic.hxx>

#include <algorithm>
#include <atomic>
#include <memory>
#include <unordered_map>

/* Static-member definitions: one per-thread cache map per process, one
 * global atomic counter handing out unique instance ids for generation
 * checking. See t8_geometry_handler.hxx for the design rationale. */
thread_local std::unordered_map<t8_geometry_handler *, t8_geometry_handler::ThreadCacheEntry>
  t8_geometry_handler::tl_cache_;

std::atomic<uint64_t> t8_geometry_handler::next_instance_id_{ 0 };

void
t8_geometry_handler::register_geometry (t8_geometry *geom)
{
  std::unique_ptr<t8_geometry> geom_ptr = std::unique_ptr<t8_geometry> (std::move (geom));
  add_geometry<t8_geometry> (std::move (geom_ptr));
}

void
t8_geometry_handler::update_tree (t8_cmesh_t cmesh, t8_gloidx_t gtreeid, ThreadCacheEntry &cache)
{
  T8_ASSERT (0 <= gtreeid && gtreeid < t8_cmesh_get_num_trees (cmesh));
  const int num_geoms = get_num_geometries ();
  SC_CHECK_ABORTF (num_geoms > 0,
                   "The geometry of the tree could not be loaded, because no geometries were registered.");
  T8_ASSERT (cache.active_geometry != nullptr);
  if (cache.active_tree != gtreeid) {
    /* This tree is not the active tree (for this thread). We need to
     * update this thread's cached active tree, its geometry, and its
     * loaded tree data. */
    /* Set the new tree as active (for this thread). */
    cache.active_tree = gtreeid;
    if (num_geoms > 1) {
      /* Find and load the geometry of that tree.
       * Only necessary if we have more than one geometry. */
      const t8_geometry_hash geom_hash = t8_cmesh_get_tree_geom_hash (cmesh, gtreeid);
      cache.active_geometry = get_geometry (geom_hash);
      SC_CHECK_ABORTF (cache.active_geometry != nullptr,
                       "Could not find geometry with hash %lu or tree %" T8_GLOIDX_FORMAT
                       " has no registered geometry.",
                       static_cast<size_t> (geom_hash), gtreeid);
    }
    /* Get the user data for this geometry and this tree.
     *
     * NOTE: t8_geom_load_tree_data writes to the geometry's OWN
     * instance members (active_tree, active_tree_class,
     * active_tree_vertices, derived-class caches like edges/faces/
     * degree/tree_data). Those members are NOT yet thread-localized
     * in Phase T1 — they're addressed in T2 (base geometry), T3
     * (with_vertices), and T4 (concrete derived classes). After
     * T1 alone, the handler-side race is gone but the geometry-side
     * race remains, so concurrent calls on different trees still
     * produce wrong outputs. Phase T4 closes Group A. */
    cache.active_geometry->t8_geom_load_tree_data (cmesh, gtreeid);
  }
}
