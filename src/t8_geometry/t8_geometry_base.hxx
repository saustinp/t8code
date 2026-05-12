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

/** \file t8_geometry_base.hxx
 * Implements the base pure virtual struct t8_geometry which
 * provides a general template for all geometries.
 */

#ifndef T8_GEOMETRY_BASE_HXX
#define T8_GEOMETRY_BASE_HXX

#include <t8.h>
#include <t8_cmesh/t8_cmesh.h>
#include <t8_eclass/t8_eclass.h>
#include <t8_forest/t8_forest.h>
#include <t8_geometry/t8_geometry.h>
#include <t8_geometry/t8_geometry_hash.hxx>

#include <atomic>
#include <cstdint>
#include <functional>
#include <unordered_map>

T8_EXTERN_C_BEGIN ();

/**
 * The base class for all geometries.
 * This class provides a general template for all geometries.
 * It is a pure virtual class and has to be inherited by a concrete
 * geometry implementation.
 */
struct t8_geometry
{
 public:
  /** Basic constructor that sets the name.
   * \param [in] name The name of the geometry. Used to distinct the geometry from other geometries.
   *
   * Assigns a unique monotonic \a instance_id_ from the global atomic
   * counter. This id is the generation key for the per-thread cache
   * of "currently loaded tree data" (see \ref geom_tls): each thread's
   * cache entry for this geometry remembers the id it was populated
   * for, and any mismatch on lookup means the cache is stale and
   * must be re-initialized.
  */
  t8_geometry (std::string name)
    : name (name), hash (t8_geometry_compute_hash (name)),
      instance_id_ (next_instance_id_.fetch_add (1, std::memory_order_relaxed) + 1)
  {
    if (t8_geometry_hash_is_null (hash)) {
      SC_ABORTF ("Registering geometry with invalid name\"%s\"\n.", name.c_str ());
    }
  }

  /* Base constructor with no arguments. We need this since it
   * is called from derived class constructors.
   * Sets the name to an invalid value. */
  t8_geometry (): t8_geometry ("Invalid")
  {
  }

  /** The destructor. It does nothing but has to be defined since
   * we may want to delete geometry that is actually inherited
   * and providing an implementation
   * for the destructor ensures that the
   * destructor of the child class will be executed. */
  virtual ~t8_geometry ()
  {
  }

  /**
   * Maps points in the reference space \f$ [0,1]^\mathrm{dim} \to \mathbb{R}^3 \f$.
   * \param [in]  cmesh       The cmesh in which the point lies.
   * \param [in]  gtreeid     The global tree (of the cmesh) in which the reference point is.
   * \param [in]  ref_coords  Array of tree dimension x \a num_coords many entries, specifying points in \f$ [0,1]^\mathrm{dim} \f$.
   * \param [in]  num_coords  Amount of points of \f$ \mathrm{dim} \f$ to map.
   * \param [out] out_coords  The mapped coordinates in physical space of \a ref_coords. The length is \a num_coords * 3.
   */
  virtual void
  t8_geom_evaluate (t8_cmesh_t cmesh, t8_gloidx_t gtreeid, const double *ref_coords, const size_t num_coords,
                    double *out_coords) const
    = 0;

  /**
   * Compute the jacobian of the \a t8_geom_evaluate map at a point in the reference space \f$ [0,1]^\mathrm{dim} \f$.
   * \param [in]  cmesh      The cmesh in which the point lies.
   * \param [in]  gtreeid    The global tree (of the cmesh) in which the reference point is.
   * \param [in]  ref_coords  Array of tree dimension x \a num_coords many entries, specifying points in \f$ [0,1]^\mathrm{dim} \f$.
   * \param [in]  num_coords  Amount of points of \f$ \mathrm{dim} \f$ to map.
   * \param [out] jacobian    The jacobian at \a ref_coords. Array of size \a num_coords x dimension x 3. Indices \f$ 3 \cdot i\f$ , \f$ 3 \cdot i+1 \f$ , \f$ 3 \cdot i+2 \f$
   *                          correspond to the \f$ i \f$-th column of the jacobian  (Entry \f$ 3 \cdot i + j \f$ is \f$ \frac{\partial f_j}{\partial x_i} \f$).
   */
  virtual void
  t8_geom_evaluate_jacobian (t8_cmesh_t cmesh, t8_gloidx_t gtreeid, const double *ref_coords, const size_t num_coords,
                             double *jacobian) const
    = 0;

  /** Update a possible internal data buffer for per tree data.
   * This function is called before the first coordinates in a new tree are
   * evaluated.
   * In this base implementation we use it to load the treeid and class
   * to the internal member variables \a active_tree and \a active_tree_class.
   * \param [in]  cmesh      The cmesh.
   * \param [in]  gtreeid    The global tree.
   */
  virtual void
  t8_geom_load_tree_data (const t8_cmesh_t cmesh, const t8_gloidx_t gtreeid);

  /** Query whether a batch of points lies inside an element.
   * \param [in]      forest      The forest.
   * \param [in]      ltreeid     The forest local id of the tree in which the element is.
   * \param [in]      element     The element.
   * \param [in]      points      3-dimensional coordinates of the points to check
   * \param [in]      num_points  The number of points to check
   * \param [in, out] is_inside   An array of length \a num_points, filled with 0/1 on output. True (non-zero) if a \a point
   *                              lies within an \a element, false otherwise. The return value is also true if the point
   *                              lies on the element boundary. Thus, this function may return true for different leaf
   *                              elements, if they are neighbors and the point lies on the common boundary.
   * \param [in]      tolerance   Tolerance that we allow the point to not exactly match the element.
   *                              If this value is larger we detect more points.
   *                              If it is zero we probably do not detect points even if they are inside
   *                              due to rounding errors.
   */
  virtual void
  t8_geom_point_batch_inside_element ([[maybe_unused]] t8_forest_t forest, [[maybe_unused]] t8_locidx_t ltreeid,
                                      [[maybe_unused]] const t8_element_t *element,
                                      [[maybe_unused]] const double *points, [[maybe_unused]] const int num_points,
                                      [[maybe_unused]] int *is_inside, [[maybe_unused]] const double tolerance) const
  {
    SC_ABORTF ("Point batch inside element function not implemented");
  };

  /**
   * Check if the currently active tree has a negative volume.
   * \return                True if the currently loaded tree has a negative volume.
   */
  virtual bool
  t8_geom_tree_negative_volume () const
  {
    SC_ABORTF ("Tree negative volume function not implemented");
    /* To suppress compiler warnings. */
    return 0;
  };

  /**
   * Check for compatibility of the currently loaded tree with the geometry.
   * If the geometry has limitations these can be checked here.
   * This includes for example if only specific tree types or dimensions are supported.
   * If all trees are supported, this function should return true.
   * \return                True if the geometry is compatible with the tree.
   */
  virtual bool
  t8_geom_check_tree_compatibility () const
    = 0;

  /**
   * Get the name of this geometry.
   * \return The name.
   */
  inline const std::string &
  t8_geom_get_name () const
  {
    return name;
  }

  /**
   * Compute the bounding box of the currently active tree.
   *
   * \param [in]  cmesh   The cmesh.
   * \param [out] bounds  The bounding box of the tree in the form (xmin, xmax, ymin, ymax, zmin, zmax).
   * \return              True if the bounding box was computed successfully, false otherwise.
   *
   * \note This function updates the active tree to the provided \a gtreeid.
   */
  virtual bool
  get_tree_bounding_box ([[maybe_unused]] const t8_cmesh_t cmesh, [[maybe_unused]] double bounds[6]) const
  {
    t8_errorf ("Tree bounding box function not implemented");
    return false;
  }

  /**
   * Get the hash value of this geometry.
   * \return The hash.
   */
  inline t8_geometry_hash
  t8_geom_get_hash () const
  {
    return hash;
  }

  /**
   * Get the type of this geometry.
   * \return The type.
   */
  virtual t8_geometry_type_t
  t8_geom_get_type () const
    = 0;

 protected:
  std::string name;      /**< The name of this geometry. */
  t8_geometry_hash hash; /**< The hash of the name of this geometry. See also \ref t8_geometry_compute_hash */

  /**
   * Per-thread cache entry for this geometry. Holds the "currently
   * loaded tree" state that was previously stored as instance members
   * (active_tree, active_tree_class) plus the derived-class caches
   * (active_tree_vertices in t8_geometry_with_vertices, degree in
   * t8_geometry_lagrange, edges/faces in t8_geometry_cad, tree_data
   * in t8_geometry_analytic). Combined into a single struct so the
   * TLS cache map only needs one entry per (thread × geometry),
   * regardless of which derived type.
   *
   * Cache is generation-checked: \a generation is the snapshot of the
   * geometry's \a instance_id_ at the time this entry was populated.
   * A mismatch on lookup means the entry is stale (either from a
   * destroyed geometry whose address was reused, or from before a
   * \ref deactivate_tree-style invalidation). Stale entries are
   * lazily re-initialized to zero state on next access.
   */
  struct TLSEntry
  {
    uint64_t generation = 0;
    /* Base-class state (formerly t8_geometry::active_tree/_class). */
    t8_gloidx_t active_tree = -1;
    t8_eclass_t active_tree_class = T8_ECLASS_INVALID;
    /* t8_geometry_with_vertices state (formerly active_tree_vertices). */
    const double *active_tree_vertices = nullptr;
    /* Derived-class-specific caches. Only one is populated for any
     * given geometry instance — they're combined here so the TLS map
     * doesn't need separate entries per derived-class subtype. */
    const int *active_degree = nullptr;     /**< t8_geometry_lagrange */
    const int *active_edges = nullptr;      /**< t8_geometry_cad */
    const int *active_faces = nullptr;      /**< t8_geometry_cad */
    const void *active_tree_data = nullptr; /**< t8_geometry_analytic */
  };

  /**
   * Get (lazily creating, generation-validating) this thread's cache
   * entry for this geometry instance. Marked \c const because the
   * geometry is logically unchanged — the only mutation is to the
   * thread-local cache, which is per-thread external state.
   */
  inline TLSEntry &
  geom_tls () const noexcept
  {
    TLSEntry &e = tls_cache_[this];
    if (e.generation != instance_id_) {
      e.generation = instance_id_;
      e.active_tree = -1;
      e.active_tree_class = T8_ECLASS_INVALID;
      e.active_tree_vertices = nullptr;
      e.active_degree = nullptr;
      e.active_edges = nullptr;
      e.active_faces = nullptr;
      e.active_tree_data = nullptr;
    }
    return e;
  }

  /* ── Accessor methods replacing the former instance members ──────
   *
   * Naming convention: same as the old field names, suffixed with ()
   * to mark them as accessor calls. This way derived-class
   * conversions are mechanical (`active_tree_class` -> `active_tree_class()`).
   *
   * Const-qualified so they're callable from const member functions
   * like t8_geom_evaluate. Internal mutation of the thread-local
   * cache is fine despite the const qualifier — TLS is per-thread
   * external state, not part of the geometry's observable instance
   * state.
   */
  inline t8_gloidx_t
  active_tree () const noexcept
  {
    return geom_tls ().active_tree;
  }
  inline t8_eclass_t
  active_tree_class () const noexcept
  {
    return geom_tls ().active_tree_class;
  }
  inline const double *
  active_tree_vertices () const noexcept
  {
    return geom_tls ().active_tree_vertices;
  }
  inline const int *
  active_degree () const noexcept
  {
    return geom_tls ().active_degree;
  }
  inline const int *
  active_edges () const noexcept
  {
    return geom_tls ().active_edges;
  }
  inline const int *
  active_faces () const noexcept
  {
    return geom_tls ().active_faces;
  }
  inline const void *
  active_tree_data () const noexcept
  {
    return geom_tls ().active_tree_data;
  }

  /* ── Setters used by t8_geom_load_tree_data implementations ────── */
  inline void
  set_active_tree (t8_gloidx_t v) const noexcept
  {
    geom_tls ().active_tree = v;
  }
  inline void
  set_active_tree_class (t8_eclass_t v) const noexcept
  {
    geom_tls ().active_tree_class = v;
  }
  inline void
  set_active_tree_vertices (const double *v) const noexcept
  {
    geom_tls ().active_tree_vertices = v;
  }
  inline void
  set_active_degree (const int *v) const noexcept
  {
    geom_tls ().active_degree = v;
  }
  inline void
  set_active_edges (const int *v) const noexcept
  {
    geom_tls ().active_edges = v;
  }
  inline void
  set_active_faces (const int *v) const noexcept
  {
    geom_tls ().active_faces = v;
  }
  inline void
  set_active_tree_data (const void *v) const noexcept
  {
    geom_tls ().active_tree_data = v;
  }

 private:
  /**
   * Per-thread cache of TLSEntry, keyed by geometry-instance pointer.
   * Static, so one map per thread (not per (thread × geometry)). The
   * map grows by one entry per (thread, geometry instance ever
   * accessed by that thread). Stale entries from destroyed geometries
   * are detected by the generation-counter mismatch on lookup.
   */
  static thread_local std::unordered_map<const t8_geometry *, TLSEntry> tls_cache_;

  /**
   * Global atomic counter that hands out monotonic unique
   * \a instance_id_ values. Starts at 0; constructor uses
   * fetch_add+1 so the first geometry gets instance_id_=1 (leaves 0
   * as the "uninitialized" sentinel for TLSEntry::generation).
   */
  static std::atomic<uint64_t> next_instance_id_;

  /**
   * Generation token assigned at construction. The per-thread cache
   * uses this to detect stale entries: if the cached generation
   * doesn't match this value, the entry was either populated by a
   * different (now-destroyed) geometry instance at this address, OR
   * the instance was invalidated by a state change. The latter
   * mechanism isn't currently used at the t8_geometry level (no
   * deactivate-style API at this layer), but is reserved for future
   * use if e.g. derived classes need to invalidate caches on
   * structural changes.
   */
  uint64_t instance_id_;
};

T8_EXTERN_C_END ();

#endif /* !T8_GEOMETRY_BASE_HXX */
