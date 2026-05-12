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

/** \file t8_geometry_handler.hxx
 * General geometry definitions
 */

#ifndef T8_GEOMETRY_HANDLER_HXX
#define T8_GEOMETRY_HANDLER_HXX

#include <t8.h>
#include <t8_geometry/t8_geometry.h>
#include <t8_geometry/t8_geometry_base.hxx>
#include <atomic>
#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>

/**
 * Handles the geometries of a \ref t8_cmesh.
 * Each tree can be assigned a geometry in this handler. The geometries
 * get assigned using their hash value.
 * Stores geometries of type \ref t8_geometry.
 */
struct t8_geometry_handler
{
 public:
  /**
   * Constructor.
   *
   * Assigns a unique monotonic instance_id_ from the global atomic counter.
   * This id is the generation key for per-thread caches (see
   * \ref get_thread_cache): each thread's stored cache entry remembers the
   * id of the handler-instance it was populated for, and any mismatch on
   * lookup means the cache is stale (handler destroyed and a new one
   * allocated at the same address, or a structural change like
   * \ref deactivate_tree or \ref add_geometry bumped the id).
   */
  t8_geometry_handler ()
    : instance_id_ (next_instance_id_.fetch_add (1, std::memory_order_relaxed) + 1)
  {
    t8_refcount_init (&rc);
    t8_debugf ("Constructed the geometry_handler.\n");
  };

  /**
   * Destructor.
   */
  ~t8_geometry_handler ()
  {
    if (sc_refcount_is_active (&rc)) {
      T8_ASSERT (t8_refcount_is_last (&rc));
      t8_refcount_unref (&rc);
    }
    t8_debugf ("Deleted the geometry_handler.\n");
  };

  /**
   * Register a geometry with the geometry handler.
   * @tparam      geometry_type The type of the geometry to register.
   * @tparam      _args         The constructor arguments of the geometry.
   * \param [in]  args          The constructor arguments of the geometry.
   * \return                    A pointer to the geometry.
   */
  template <typename geometry_type, typename... _args>
  geometry_type *
  register_geometry (_args &&...args)
  {
    std::unique_ptr<t8_geometry> geom_ptr = std::make_unique<geometry_type> (std::forward<_args> (args)...);
    return add_geometry<geometry_type> (std::move (geom_ptr));
  }

  /**
   * Register a geometry with the geometry handler.
   * The handler will take ownership of the geometry.
   * \param [in]  geom  The geometry to register.
   */
  void
  register_geometry (t8_geometry *geom);

  /**
   * Find a geometry by its name.
   * \param [in]  name  The name of the geometry to find.
   * \return            An iterator to the geometry if found, NULL otherwise.
   */
  inline t8_geometry *
  get_geometry (const std::string &name)
  {
    const t8_geometry_hash hash = t8_geometry_compute_hash (name);
    return t8_geometry_handler::get_geometry (hash);
  }

  /**
   * Find a geometry by its hash.
   * \param [in]  hash  The hash of the geometry to find.
   * \return            An iterator to the geometry if found, NULL otherwise.
   */
  inline t8_geometry *
  get_geometry (const t8_geometry_hash &hash)
  {
    if (t8_geometry_hash_is_null (hash)) {
      /* The hash belongs to a non-existing geometry. */
      return nullptr;
    }
    auto found = registered_geometries.find (hash);
    if (found != registered_geometries.end ()) {
      return found->second.get ();
    }
    t8_errorf ("Geometry with hash value %lu was not found.\n", static_cast<size_t> (hash));
    return nullptr;
  }

  /**
   * Get the number of registered geometries.
   * \return  The number of registered geometries.
   */
  inline size_t
  get_num_geometries () const
  {
    return registered_geometries.size ();
  }

  /** If a geometry handler only has one registered geometry, get a pointer to
   *  this geometry.
   * \return     The only registered geometry of \a geom_handler.
   * \note  Most cmeshes will have only one geometry and this function is an optimization
   *        for that special case. It is used for example in \ref t8_cmesh_get_tree_geometry.
   */
  inline t8_geometry *
  get_unique_geometry ()
  {
    T8_ASSERT (registered_geometries.size () == 1);
    return registered_geometries.begin ()->second.get ();
  }

  /**
   * Deactivate the current active tree. Can be used to reload data,
   * after it has been moved, for example by the partition-algorithm.
   *
   * Implementation: bumps \a instance_id_ to a new unique value. This
   * invalidates ALL threads' cached tree state on their next access
   * (the cache compares its stored generation against \a instance_id_).
   * No need to iterate over per-thread cache state directly — the
   * generation check handles it lazily.
   */
  inline void
  deactivate_tree ()
  {
    instance_id_ = next_instance_id_.fetch_add (1, std::memory_order_relaxed) + 1;
  }

  /**
   * Get the geometry of the provided tree.
   * \param [in] cmesh   The cmesh.
   * \param [in] gtreeid The global tree id of the tree for which the geometry should be returned.
   * \return             The geometry of the tree.
   *
   * Thread-safety: each thread maintains its own cache of the most-recently-
   * accessed tree per handler instance (see \ref get_thread_cache). The
   * cache is keyed by handler-pointer and validated by generation counter,
   * so concurrent calls on different trees from different threads do not
   * race on the cache state.
   */
  inline t8_geometry *
  get_tree_geometry (t8_cmesh_t cmesh, t8_gloidx_t gtreeid)
  {
    ThreadCacheEntry &cache = get_thread_cache ();
    update_tree (cmesh, gtreeid, cache);
    return cache.active_geometry;
  }

  /**
   * Evaluate the geometry of the provided tree at the given reference coordinates.
   * \param [in]  cmesh      The cmesh.
   * \param [in]  gtreeid    The global tree id of the tree for which the geometry should be evaluated.
   * \param [in]  ref_coords The reference coordinates at which to evaluate the geometry.
   * \param [in]  num_coords The number of reference coordinates.
   * \param [out] out_coords The evaluated coordinates.
   */
  inline void
  evaluate_tree_geometry (t8_cmesh_t cmesh, t8_gloidx_t gtreeid, const double *ref_coords, const size_t num_coords,
                          double *out_coords)
  {
    ThreadCacheEntry &cache = get_thread_cache ();
    update_tree (cmesh, gtreeid, cache);
    cache.active_geometry->t8_geom_evaluate (cmesh, gtreeid, ref_coords, num_coords, out_coords);
  }

  /**
   * Evaluate the Jacobian of the geometry of the provided tree at the given reference coordinates.
   * \param [in]  cmesh      The cmesh.
   * \param [in]  gtreeid    The global tree id of the tree for which the geometry should be evaluated.
   * \param [in]  ref_coords The reference coordinates at which to evaluate the geometry.
   * \param [in]  num_coords The number of reference coordinates.
   * \param [out] out_coords The evaluated Jacobian coordinates.
   */
  inline void
  evaluate_tree_geometry_jacobian (t8_cmesh_t cmesh, t8_gloidx_t gtreeid, const double *ref_coords,
                                   const size_t num_coords, double *out_coords)
  {
    ThreadCacheEntry &cache = get_thread_cache ();
    update_tree (cmesh, gtreeid, cache);
    cache.active_geometry->t8_geom_evaluate_jacobian (cmesh, gtreeid, ref_coords, num_coords, out_coords);
  }

  /**
   * Get the geometry type of the provided tree.
   * \param [in] cmesh   The cmesh.
   * \param [in] gtreeid The global tree id of the tree for which the geometry type should be returned.
   * \return             The geometry type of the tree.
   */
  inline t8_geometry_type_t
  get_tree_geometry_type (t8_cmesh_t cmesh, t8_gloidx_t gtreeid)
  {
    ThreadCacheEntry &cache = get_thread_cache ();
    update_tree (cmesh, gtreeid, cache);
    return cache.active_geometry->t8_geom_get_type ();
  }

  /**
   * Check if the volume of a tree is negative.
   * \param [in] cmesh   The cmesh.
   * \param [in] gtreeid The global tree id of the tree to check.
   * \return             True if the volume of the tree is negative, false otherwise.
   */
  inline bool
  tree_negative_volume (const t8_cmesh_t cmesh, const t8_gloidx_t gtreeid)
  {
    ThreadCacheEntry &cache = get_thread_cache ();
    update_tree (cmesh, gtreeid, cache);
    return cache.active_geometry->t8_geom_tree_negative_volume ();
  }

  /**
   * Check for compatibility of the tree with the assigned geometry.
   * \param [in] cmesh   The cmesh.
   * \param [in] gtreeid The global tree id of the tree to check.
   * \return             True if the tree and assigned geometry are compatible.
   */
  inline bool
  tree_compatible_with_geom (const t8_cmesh_t cmesh, const t8_gloidx_t gtreeid)
  {
    ThreadCacheEntry &cache = get_thread_cache ();
    update_tree (cmesh, gtreeid, cache);
    return cache.active_geometry->t8_geom_check_tree_compatibility ();
  }

  /**
   * Get the bounding box of the tree.
   * \param [in]  cmesh    The cmesh.
   * \param [in]  gtreeid  The global tree id of the tree for which the bounding box should be returned.
   * \param [out] bounds   The bounding box of the tree, in the format [xmin, xmax, ymin, ymax, zmin, zmax].
   *
   * \note This function updates the active tree to the provided \a gtreeid.
   */
  inline bool
  get_tree_bounding_box (t8_cmesh_t cmesh, t8_gloidx_t gtreeid, double bounds[6])
  {
    ThreadCacheEntry &cache = get_thread_cache ();
    update_tree (cmesh, gtreeid, cache);
    return cache.active_geometry->get_tree_bounding_box (cmesh, bounds);
  }

  /**
   * Increase the reference count of the geometry handler.
   */
  inline void
  ref ()
  {
    t8_refcount_ref (&rc);
  }

  /**
   * Decrease the reference count of the geometry handler.
   * If the reference count reaches zero, the geometry handler is deleted.
   */
  inline void
  unref ()
  {
    if (t8_refcount_unref (&rc)) {
      t8_debugf ("Deleting the geometry_handler.\n");
      delete this;
    }
  }

 private:
  /**
   * Per-thread cache entry: tracks the most-recently-accessed tree for
   * this handler instance on a particular thread. Generation-checked
   * against \a instance_id_ to handle:
   *   - handler destruction + reuse of the same address (different
   *     instance_id_).
   *   - \ref deactivate_tree (bumps instance_id_).
   *   - \ref add_geometry registering a new geometry (bumps
   *     instance_id_, so a thread's cached active_geometry pointer
   *     gets re-derived from the new registered set).
   *
   * On lookup mismatch, the cache re-initializes lazily.
   */
  struct ThreadCacheEntry
  {
    /** Snapshot of the handler's \a instance_id_ at the time this entry
     * was populated. A mismatch on lookup means the entry is stale and
     * must be re-initialized from the handler's current state. */
    uint64_t generation = 0;
    /** This thread's view of "currently loaded" geometry for this
     * handler. Each thread has its own; concurrent calls on different
     * trees from different threads do not race on this pointer. */
    t8_geometry *active_geometry = nullptr;
    /** This thread's view of "currently loaded" tree id for this
     * handler. The "cache hit" check (\a active_tree == gtreeid)
     * inside \ref update_tree uses this to skip redundant
     * t8_geom_load_tree_data calls on consecutive same-tree
     * evaluations from the same thread. */
    t8_gloidx_t active_tree = -1;
  };

  /** Per-thread cache keyed by handler-instance pointer. Static so all
   * handlers share a single per-thread map; each thread's map is
   * distinct. The map grows by one entry per handler instance ever
   * accessed by the thread (bounded; stale entries from destroyed
   * handlers stay until thread exit but are invalidated on lookup by
   * the generation-counter check). */
  static thread_local std::unordered_map<t8_geometry_handler *, ThreadCacheEntry> tl_cache_;

  /** Global atomic counter that hands out monotonic unique ids to
   * \a instance_id_. The 0 value is reserved as "uninitialized" for
   * \a ThreadCacheEntry::generation, so we start handing out ids at 1
   * (the constructor uses fetch_add+1). */
  static std::atomic<uint64_t> next_instance_id_;

  /**
   * Get (or lazily create) this thread's cache entry for this handler
   * instance. Generation-counter validated: if the stored generation
   * doesn't match \a instance_id_ (i.e. stale from a prior handler
   * that happened to share this address, or from a deactivate_tree /
   * add_geometry that bumped the id), the entry is reset and the
   * single-geometry shortcut from the pre-Phase-T1 add_geometry path
   * is re-applied.
   */
  inline ThreadCacheEntry &
  get_thread_cache ()
  {
    ThreadCacheEntry &entry = tl_cache_[this];
    if (entry.generation != instance_id_) {
      entry.generation = instance_id_;
      entry.active_tree = -1;
      /* Pre-Phase-T1 behavior we must preserve: in the original code,
       * `active_geometry` was set to the first registered geometry
       * during add_geometry (when size==1) and was NEVER reset to
       * nullptr afterward — adding subsequent geometries left
       * active_geometry pointing at the first one. Downstream code
       * (specifically the T8_ASSERT in update_tree) expects
       * active_geometry to be non-null whenever any geometry is
       * registered, even in the multi-geom case where the eventual
       * "correct" geometry per-tree is resolved later by hash lookup.
       *
       * To match: set active_geometry to *some* registered geometry
       * (we pick the first one in iteration order, matching the
       * registered_geometries.begin() pattern). The first call to
       * update_tree in multi-geom mode will replace it with the
       * per-tree-correct one via geom_hash lookup. For single-geom
       * mode it's already correct and stays so. */
      if (registered_geometries.empty ()) {
        entry.active_geometry = nullptr;
      }
      else {
        entry.active_geometry = registered_geometries.begin ()->second.get ();
      }
    }
    return entry;
  }

  /**
   * Add a geometry to the geometry handler.
   * @tparam     geometry_type The type of the geometry to add.
   * \param [in] geom          The geometry to add.
   * \return                   A pointer to the geometry.
   *
   * On a NEW registration we bump \a instance_id_ to invalidate every
   * thread's cached active_geometry pointer. This handles the case
   * where a thread accessed the handler before the (now-registered)
   * second geometry was added — the cache's cached single-geom
   * shortcut would otherwise be stale.
   */
  template <typename geometry_type>
  inline geometry_type *
  add_geometry (std::unique_ptr<t8_geometry> geom)
  {
    t8_debugf ("Registering geometry with name %s\n", geom->t8_geom_get_name ().c_str ());
    const t8_geometry_hash hash = geom->t8_geom_get_hash ();
    if (registered_geometries.find (hash) == registered_geometries.end ()) {
      registered_geometries.emplace (hash, std::move (geom));
      /* Invalidate all per-thread caches: their cached active_geometry
       * pointer was derived from the previous registered_geometries
       * state. */
      instance_id_ = next_instance_id_.fetch_add (1, std::memory_order_relaxed) + 1;
    }
    else {
      t8_productionf ("WARNING: Did not register the geometry %s because it is already registered.\n"
                      "Geometries only need to be registered once per process.\n"
                      "If you are registering a new geometry it probably has the same name as another one.\n",
                      geom->t8_geom_get_name ().c_str ());
    }
    return static_cast<geometry_type *> (registered_geometries.at (hash).get ());
  }

  /**
   * Update the active tree on a specific thread's cache. Called by
   * every public method that resolves geometry for a tree.
   * \param [in]      cmesh    The cmesh.
   * \param [in]      gtreeid  The global tree id of the tree to update.
   * \param [in,out]  cache    The current thread's cache entry for this handler.
   */
  void
  update_tree (t8_cmesh_t cmesh, t8_gloidx_t gtreeid, ThreadCacheEntry &cache);

  /** Stores all geometries that are handled by this geometry_handler. */
  std::unordered_map<t8_geometry_hash, std::unique_ptr<t8_geometry>> registered_geometries = {};

  /** Generation counter that uniquely identifies this handler instance
   * for thread-cache validation. Initialized in the constructor from
   * the global atomic; bumped on \ref deactivate_tree and on
   * \ref add_geometry to force per-thread caches to re-initialize on
   * next access. */
  uint64_t instance_id_;

  /** The reference count of the geometry handler. TODO: Replace by shared_ptr when cmesh becomes a class. */
  t8_refcount_t rc;
};

#endif /* !T8_GEOMETRY_HANDLER_HXX */
