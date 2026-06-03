/*
  This file is part of t8code.
  t8code is a C library to manage a collection (a forest) of multiple
  connected adaptive space-trees of general element classes in parallel.

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

/**
 * \file t8_forest_periodic_cache.hxx
 * Internal C++ helper for periodic-adjacency queries. Caches the spatial
 * corner hash + cmesh-derived periodic translations, so balance (and other
 * O(N_leaves) consumers) can perform per-leaf periodic-neighbor lookups
 * without rebuilding the hash on every call.
 *
 * Not part of the public t8 API. Consumed by t8_forest_balance.cxx and
 * t8_forest.cxx (the v1 query t8_forest_leaf_periodic_neighbors may be
 * refactored to use this cache in a follow-up).
 */

#ifndef T8_FOREST_PERIODIC_CACHE_HXX
#define T8_FOREST_PERIODIC_CACHE_HXX

#include <t8.h>
#include <t8_forest/t8_forest_general.h>

/* The implementation lives in t8_forest.cxx, which is wrapped in
 * T8_EXTERN_C_BEGIN/END — so these symbols have C linkage. */
T8_EXTERN_C_BEGIN ();

/** Opaque periodic-adjacency cache. Lifetime is bounded by one adapt round
 * (or one is_balanced check) — call \ref t8_forest_periodic_cache_destroy
 * and re-build before reusing on a different forest. */
struct t8_forest_periodic_cache;
typedef struct t8_forest_periodic_cache t8_forest_periodic_cache_t;

/** Build a periodic-adjacency cache for \a forest. Returns NULL if the
 * forest's cmesh has no periodic face-pair joins (i.e. no periodicity at
 * all in the mesh). Caller owns the returned cache and must destroy with
 * \ref t8_forest_periodic_cache_destroy.
 *
 * Multi-rank handling:
 *   - Replicated forest (mpisize==1, or local_num_leaves == global_num_leaves):
 *     every rank builds the hash from its own (full) leaf set.
 *   - Distributed forest (mpisize>1 and local_num != global_num): each rank
 *     first inserts its local leaves, then exchanges on-periodic-seam corners
 *     (Allgatherv on (xyz, level) records). Remote corners are stored as
 *     PCacheLoc with ltreeid=-1 so the consumer treats them as "trust the
 *     cached level, do not dereference an element handle".
 *
 * The function is COLLECTIVE on \a forest's MPI communicator at NP>1: every
 * rank must call it together. */
t8_forest_periodic_cache_t *
t8_forest_periodic_cache_new (t8_forest_t forest);

/** Free a cache returned by \ref t8_forest_periodic_cache_new. No-op on NULL. */
void
t8_forest_periodic_cache_destroy (t8_forest_periodic_cache_t *cache);

/** Return the maximum element-level among all periodic edge/vertex/face
 * neighbors of \a leaf, or -1 if \a leaf has no periodic neighbors.
 *
 * \param[in] cache    A cache built against \a forest in its current state.
 * \param[in] forest   The forest \a cache was built from. Must match.
 * \param[in] ltreeid  Local tree id containing \a leaf.
 * \param[in] leaf     A leaf element in tree \a ltreeid.
 * \return             Maximum neighbor level, or -1 if no periodic neighbors. */
int
t8_forest_periodic_cache_max_neighbor_level (t8_forest_periodic_cache_t *cache, t8_forest_t forest,
                                             t8_locidx_t ltreeid, const t8_element_t *leaf);

T8_EXTERN_C_END ();

#endif /* !T8_FOREST_PERIODIC_CACHE_HXX */
