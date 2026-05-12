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

/** \file t8_default_common.hxx
 * We provide some functions that are useful across element classes.
 */

#ifndef T8_DEFAULT_COMMON_HXX
#define T8_DEFAULT_COMMON_HXX

#include <t8_element/t8_element.h>
#include <t8_types/t8_operators.hxx>
#include <t8_schemes/t8_scheme_helpers.hxx>
#include <sc_functions.h>
#include <sc_containers.h>
#include <cstdlib>

/** Macro to check whether a pointer (VAR) to a base class, comes from an
 * implementation of a child class (TYPE). */
#define T8_COMMON_IS_TYPE(VAR, TYPE) ((dynamic_cast<TYPE> (VAR)) != NULL)

/** Allocate \a length elements of \a elem_size bytes via plain std::malloc.
 * Each pointer in \a elem is filled with a fresh allocation.
 *
 * Thread-safety: std::malloc is fully thread-safe and uses per-thread
 * arenas in glibc, giving us the desired per-thread isolation without
 * any shared state. This replaces the pre-T5 sc_mempool which had a
 * single free-list head racing across threads.
 *
 * \param [in]     elem_size The size in bytes of each element.
 * \param [in]     length    Non-negative number of elements to allocate.
 * \param [in,out] elem      Array whose members are filled with fresh
 *                            allocations of \a elem_size bytes each.
 */
inline static void
t8_default_element_alloc (size_t elem_size, int length, t8_element_t **elem)
{
  T8_ASSERT (0 <= length);
  T8_ASSERT (elem != NULL);

  for (int i = 0; i < length; ++i) {
    elem[i] = (t8_element_t *) std::malloc (elem_size);
    T8_ASSERT (elem[i] != NULL);
  }
}

/** Free \a length elements allocated by \ref t8_default_element_alloc.
 *
 * Thread-safety: std::free is fully thread-safe. Unlike sc_mempool, an
 * element allocated on one thread may be freed on a different thread
 * (the caller does not need to coordinate which thread frees what).
 *
 * \param [in]     length Non-negative number of elements to destroy.
 * \param [in,out] elem   Array whose members are freed via std::free.
 */
inline static void
t8_default_element_free (int length, t8_element_t **elem)
{
  T8_ASSERT (0 <= length);
  T8_ASSERT (elem != NULL);

  for (int i = 0; i < length; ++i) {
    std::free (elem[i]);
  }
}

/* Given an element's level and dimension, return the number of leaves it
 * produces at a given uniform refinement level */
static inline t8_gloidx_t
count_leaves_from_level (const int element_level, const int refinement_level, const int dimension)
{
  return element_level > refinement_level ? 0 : (1ULL << (dimension * (refinement_level - element_level)));
}

/* ────────────────────────────────────────────────────────────────────
 * Element allocation strategy (t8 thread-safety Phase T5)
 *
 * The pre-T5 design had a single sc_mempool_t* per scheme instance,
 * shared across all threads. Concurrent calls to element_new /
 * element_destroy raced on the sc_mempool free-list head, causing
 * element_is_valid assertion-aborts in t8_forest_leaf_face_neighbors
 * (the canonical t8 caller that hits element_new with high concurrency).
 *
 * Per the user-locked "per-thread" design intent, we replace the shared
 * mempool with PLAIN std::malloc/std::free. glibc malloc already uses
 * per-thread arenas (which gives us exactly the per-thread isolation we
 * want) and is fully thread-safe. The malloc allocator also has no
 * cross-thread cleanup or lifecycle concerns: every element is owned by
 * its allocation, freed independently of any "scheme" or "thread"
 * registry.
 *
 * Why malloc over per-thread sc_mempool:
 *   - Lifecycle: a per-thread sc_mempool must be destroyed at some
 *     point. Either at thread exit (too late: libsc's sc_finalize
 *     memory-balance check fires first because the main thread's TLS
 *     destructor only runs at process exit) or at scheme destruction
 *     (requires a global registry walking other threads' TLS storage,
 *     which is fragile and was the source of the original lifecycle
 *     SEGVs we hit during T5 development).
 *   - Memory accounting: malloc is invisible to libsc's SC_ALLOC/
 *     SC_FREE counters, so it doesn't interact with sc_finalize at all.
 *   - Thread-safety: glibc malloc is fully thread-safe and uses
 *     per-thread arenas internally, giving us the same per-thread
 *     locality benefit a per-thread sc_mempool would.
 *   - Performance: element_t allocations are small and infrequent on
 *     the AMR hot path. Glibc malloc's small-bin fast-path is
 *     comparable to (often within 2× of) sc_mempool_alloc, and the
 *     per-thread isolation eliminates cache-line ping-pong on the
 *     free-list head.
 *
 * Element allocation/free correctness:
 *   - Each element is its own malloc allocation. The same element may
 *     be allocated on one thread and freed on another with no issues
 *     (glibc handles cross-arena frees correctly). This is strictly
 *     more permissive than what the pre-T5 sc_mempool allowed.
 * ──────────────────────────────────────────────────────────────────── */

/** Common interface of the default schemes for each element shape.
 * \tparam TUnderlyingEclassScheme The default scheme class of the element shape.
 */
template <t8_eclass_t TEclass, class TUnderlyingEclassScheme>
struct t8_default_scheme_common: public t8_scheme_helpers<TEclass, TUnderlyingEclassScheme>
{
 private:
  friend TUnderlyingEclassScheme;
  /** Private constructor which can only be used by derived schemes.
   * \param [in] elem_size  The size of the elements this scheme holds.
   */
  t8_default_scheme_common (const size_t elem_size) noexcept: element_size (elem_size) {}

 protected:
  size_t element_size; /**< The size in bytes of an element of class \a eclass */

 public:
  /** Destructor for all default schemes.
   *
   * Pre-T5: destroyed the single shared sc_mempool here.
   * Post-T5: nothing to destroy. Elements are allocated via
   * std::malloc and freed individually via std::free in
   * element_destroy. There is no per-scheme allocator state to
   * clean up. */
  ~t8_default_scheme_common ()
  {
  }

  /** Move constructor */
  t8_default_scheme_common (t8_default_scheme_common &&other) noexcept: element_size (other.element_size)
  {
  }

  /** Move assignment operator */
  t8_default_scheme_common &
  operator= (t8_default_scheme_common &&other) noexcept
  {
    if (this != &other) {
      element_size = other.element_size;
    }
    return *this;
  }

  /** Copy constructor */
  t8_default_scheme_common (const t8_default_scheme_common &other): element_size (other.element_size)
  {
  }

  /** Copy assignment operator */
  t8_default_scheme_common &
  operator= (const t8_default_scheme_common &other)
  {
    if (this != &other) {
      element_size = other.element_size;
    }
    return *this;
  }

  /** Return the size of any element of a given class.
   * \return                      The size of an element of class \b ts.
   * We provide a default implementation of this routine that should suffice
   * for most use cases.
   */
  inline size_t
  get_element_size (void) const
  {
    return element_size;
  }

  /** Compute the number of corners of a given element.
   * \return The number of corners of the element.
   * \note This function is overwritten by the pyramid implementation.
  */
  inline int
  element_get_num_corners ([[maybe_unused]] const t8_element_t *elem) const
  {
    /* use the lookup table of the eclasses.
     * Pyramids should implement their own version of this function. */
    return t8_eclass_num_vertices[TEclass];
  }

  /** Return the max number of children of an eclass.
   * \return            The max number of children of \a element.
   */
  inline int
  get_max_num_children () const
  {
    return t8_eclass_max_num_children[TEclass];
  }

  /** Query whether element A is an ancestor of the element B.
   * An element A is ancestor of an element B if A == B or if B can 
   * be obtained from A via successive refinement.
   * \param [in] element_A An element of class \a eclass in scheme \a scheme.
   * \param [in] element_B An element of class \a eclass in scheme \a scheme.
   * \return     True if and only if \a element_A is an ancestor of \a element_B.
  */
  inline bool
  element_is_ancestor (const t8_element_t *element_A, const t8_element_t *element_B) const
  {
    /* A is ancestor of B if and only if it has smaller or equal level and
      restricted to A's level, B has the same id as A.

        level(A) <= level(B) and ID(A,level(A)) == ID(B,level(B))
    */
    T8_ASSERT (this->underlying ().element_is_valid (element_A));
    T8_ASSERT (this->underlying ().element_is_valid (element_B));

    const int level_A = this->underlying ().element_get_level (element_A);
    const int level_B = this->underlying ().element_get_level (element_B);

    if (level_A > level_B) {
      /* element A is finer than element B and thus cannot be 
      * an ancestor of B. */
      return false;
    }

    const t8_linearidx_t id_A = this->underlying ().element_get_linear_id (element_A, level_A);
    const t8_linearidx_t id_B = this->underlying ().element_get_linear_id (element_B, level_A);

    // If both elements have the same linear ID at level_A then A is an ancestor of B.
    return id_A == id_B;
  }

  /** Allocate space for a bunch of elements via std::malloc.
   * \param [in] length The number of elements to allocate.
   * \param [out] elem  The elements to allocate.
   *
   * Thread-safety: std::malloc is fully thread-safe and uses
   * per-thread arenas in glibc, giving us per-thread allocation
   * isolation. This replaces the pre-T5 sc_mempool which raced on
   * its single free-list head.
   */
  inline void
  element_new (const int length, t8_element_t **elem) const
  {
    t8_default_element_alloc (element_size, length, elem);
  }

  /** Deallocate space for a bunch of elements via std::free.
   *
   * Thread-safety: std::free is fully thread-safe and allows
   * cross-thread frees (an element allocated by one thread may be
   * freed by another). This is strictly more permissive than the
   * pre-T5 sc_mempool which required same-thread alloc/free.
   */
  inline void
  element_destroy (const int length, t8_element_t **elem) const
  {
    t8_default_element_free (length, elem);
  }

  /** Deinitialize an array of allocated elements.
   * \param [in] length   The number of elements to be deinitialized.
   * \param [in,out] elem On input an array of \a length many allocated
   *                       and initialized elements, on output an array of
   *                       \a length many allocated, but not initialized elements.
   * \note Call this function if you called element_init on the element pointers.
   * \see element_init
   */
  inline void
  element_deinit ([[maybe_unused]] int length, [[maybe_unused]] t8_element_t *elem) const
  {
  }

  /** Return the shape of an element
   * \param [in] elem The element.
   * \return The shape of the element.
   * \note This function is overwritten by the pyramid implementation.
  */
  inline t8_element_shape_t
  element_get_shape ([[maybe_unused]] const t8_element_t *elem) const
  {
    /* use the lookup table of the eclasses.
     * Pyramids should implement their own version of this function. */
    return TEclass;
  }

  /** Count how many leaf descendants of a given uniform level an element would produce.
   * \param [in] element   The element to be checked.
   * \param [in] level     A refinement level.
   * \return Suppose \a element is uniformly refined up to level \a level. The return value
   * is the resulting number of elements (of the given level).
   * Each default element (except pyramids) refines into 2^{dim * (level - level(t))}
   * children.
   * \note This function is overwritten by the pyramid implementation.
   */
  inline t8_gloidx_t
  element_count_leaves (const t8_element_t *element, int level) const
  {
    const int element_level = this->underlying ().element_get_level (element);
    const int dim = t8_eclass_to_dimension[TEclass];
    return count_leaves_from_level (element_level, level, dim);
  }

  /**
   * Indicates if an element is refinable. Possible reasons for being not refinable could be
   * that the element has reached its max level.
   * \param [in] elem   The element to check.
   * \return            True if the element is refinable.
   */
  inline bool
  element_is_refinable (const t8_element_t *elem) const
  {
    T8_ASSERT (this->underlying ().element_is_valid (elem));

    return this->underlying ().element_get_level (elem) < this->underlying ().get_maxlevel ();
  }

  /** Compute the number of siblings of an element. That is the number of
   * Children of its parent.
   * \param [in] elem The element.
   * \return          The number of siblings of \a element.
   * \note This function is overwritten by the pyramid implementation.
   * \note that this number is >= 1, since we count the element itself as a sibling.
   */
  inline int
  element_get_num_siblings ([[maybe_unused]] const t8_element_t *elem) const
  {
    const int dim = t8_eclass_to_dimension[TEclass];
    T8_ASSERT (TEclass != T8_ECLASS_PYRAMID);
    return sc_intpow (2, dim);
  }

  /** Count how many leaf descendants of a given uniform level the root element will produce.
   * \param [in] level A refinement level.
   * \return The value of \ref t8_element_count_leaves if the input element
   *      is the root (level 0) element.
   * \note This function is overwritten by the pyramid implementation.
   */
  inline t8_gloidx_t
  count_leaves_from_root (const int level) const
  {
    if (TEclass == T8_ECLASS_PYRAMID) {
      return 2 * sc_intpow64u (8, level) - sc_intpow64u (6, level);
    }
    const int dim = t8_eclass_to_dimension[TEclass];
    return count_leaves_from_level (0, level, dim);
  }

#if T8_ENABLE_DEBUG
  /**
   * Print a given element. For a example for a triangle print the coordinates
   * and the level of the triangle. This function is only available in the
   * debugging configuration.
   * \param [in]        elem  The element to print
   */
  inline void
  element_debug_print (const t8_element_t *elem) const
  {
    char debug_string[BUFSIZ];
    this->underlying ().element_to_string (elem, debug_string, BUFSIZ);
    t8_debugf ("%s\n", debug_string);
  }
#endif
};

#endif /* !T8_DEFAULT_COMMON_HXX */
