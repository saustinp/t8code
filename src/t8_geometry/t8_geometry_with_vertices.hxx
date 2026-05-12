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

/** \file t8_geometry_with_vertices.hxx
 * Implements the inherited struct t8_geometry_with_vertices, which can be
 * used for geometries that use vertex coordinate information of the cmesh.
 */

#ifndef T8_GEOMETRY_WITH_VERTICES_HXX
#define T8_GEOMETRY_WITH_VERTICES_HXX

#include <t8_cmesh/t8_cmesh.h>
#include <t8_forest/t8_forest.h>
#include <t8_geometry/t8_geometry_base.hxx>
#include <t8_geometry/t8_geometry_base.h>
#include <t8_geometry/t8_geometry_with_vertices.h>

T8_EXTERN_C_BEGIN ();

/**
 * Base class for vertex-based geometries. This class implements
 * member variables storing tree vertex information and updates them before
 * each member function call. Derived classes should not completely overwrite
 * the \ref t8_geom_load_tree_data function since this function takes care
 * of the updates. Instead the derived classes \ref t8_geom_load_tree_data
 * function should also call the function of this base class.
 */
struct t8_geometry_with_vertices: public t8_geometry
{
 public:
  /** Basic constructor that sets the name.
   * \param [in] name The name of the geometry. Used to distinct the geometry from other geometries.
   *
   * NOTE (thread-safety refactor): the active_tree / active_tree_vertices
   * "currently loaded tree" state is now per-thread cache (see TLSEntry
   * in t8_geometry_base.hxx). Each thread's cache is lazily initialized
   * to the zero state on first access via the generation-counter
   * mismatch — no constructor action needed.
   */
  t8_geometry_with_vertices (std::string name): t8_geometry (name)
  {
  }

  /* Base constructor with no arguments. We need this since it
   * is called from derived class constructors.
   * Sets the name to an invalid value. */
  t8_geometry_with_vertices (): t8_geometry_with_vertices ("Invalid")
  {
  }

  /** The destructor. It does nothing but has to be defined since
   * we may want to delete geometry that is actually inherited
   * and providing an implementation
   * for the destructor ensures that the
   * destructor of the child class will be executed. */
  virtual ~t8_geometry_with_vertices ()
  {
  }

  /** Update a possible internal data buffer for per tree data.
   * This function is called before the first coordinates in a new tree are
   * evaluated.
   * In this implementation we use it to load the tree's vertex coordinates and class
   * to the internal member variables \a active_tree_class and \a active_tree_vertices.
   * \param [in]  cmesh      The cmesh.
   * \param [in]  gtreeid    The global tree.
   */
  virtual void
  t8_geom_load_tree_data (t8_cmesh_t cmesh, t8_gloidx_t gtreeid);

  /**
   * Check if the currently active tree has a negative volume
   * \return                True (non-zero) if the currently loaded tree has a negative volume. 0 otherwise.
   */
  virtual bool
  t8_geom_tree_negative_volume () const;

  /**
   * Get the type of this geometry.
   * \return The type.
   */
  inline t8_geometry_type_t
  t8_geom_get_type () const
  {
    return T8_GEOMETRY_TYPE_UNDEFINED;
  };

 protected:
  /* Note: the former `active_tree_vertices` instance member has been
   * moved into the t8_geometry base class's per-thread TLSEntry cache.
   * Derived classes that previously read this field directly should
   * now call active_tree_vertices() (the protected accessor inherited
   * from t8_geometry). Writes happen via set_active_tree_vertices(),
   * called by this class's t8_geom_load_tree_data. */
};

T8_EXTERN_C_END ();

#endif /* !T8_GEOMETRY_WITH_VERTICES_HXX */
