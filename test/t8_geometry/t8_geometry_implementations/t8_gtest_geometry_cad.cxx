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

#include <gtest/gtest.h>
#include <t8_cmesh/t8_cmesh.hxx>
#include <t8_forest/t8_forest_general.h>
#include <t8_vtk/t8_vtk.h>
#include <t8_schemes/t8_default/t8_default.hxx>
#include <t8_geometry/t8_geometry_implementations/t8_geometry_cad.hxx>
#include <t8_cmesh/t8_cmesh_examples.h>
#include <test/t8_gtest_macros.hxx>
#include <test/t8_gtest_custom_assertion.hxx>
#include <t8_element/t8_element.h>
#include <t8_vtk/t8_vtk_writer.h>
#include <t8_types/t8_vec.hxx>

#include <t8_schemes/t8_default/t8_default_hex/t8_dhex.h>
#include <array>
#include <memory>
#include <iostream>

#include <GeomAPI_PointsToBSpline.hxx>
#include <GeomAPI_PointsToBSplineSurface.hxx>
#include <BRepBuilderAPI_MakeFace.hxx>
#include <BRepBuilderAPI_MakeEdge.hxx>
#include <TColgp_Array1OfPnt.hxx>
#include <TColgp_Array2OfPnt.hxx>
#include <Geom_BSplineCurve.hxx>
#include <Geom_BSplineSurface.hxx>
#include <Geom_Circle.hxx>
#include <Geom_CylindricalSurface.hxx>
#include <gp_Circ.hxx>
#include <gp_Cylinder.hxx>
#include <gp_Ax2.hxx>
#include <gp_Ax3.hxx>
#include <gp_Dir.hxx>
#include <gp_Pnt.hxx>
#include <TopoDS_Face.hxx>
#include <TopoDS_Edge.hxx>
#include <TopoDS.hxx>
#include <BRep_Tool.hxx>
#include <t8_element/t8_element.h>

#include <test/t8_gtest_custom_assertion.hxx>

/* In this file we collect tests for t8code's OpenCASCADE geometry module.
 * These tests are
 *  - linked_edges: Checks if the geometry mapping works.
 *                  We define an OpenCASCADE curve and link it to an edge of a hexahedron.
 *                  After that we probe whether the correct coordinates inside the hexahedron are returned.
 *                  We repeat this check for all 12 edges.
 *  - linked_faces: Checks if the geometry mapping works.
 *                  We define an OpenCASCADE surface and link it to a face of a hexahedron.
 *                  After that we probe whether the correct coordinates inside the hexahedron are returned.
 *                  We repeat this check for all 6 faces.
 *  - jacobian:     Checks the resulting jacobian of an identity.
 */

/** Euler rotation around intrinsic zxz.
 * \param [in] pos_vec                Position vector of three dimensional points to rotate.
 * \param [in] rot_vec                Three dimensional rotation vector around z, x and z in rad.
 * \param [out] res_vec               Vector for resulting points.
 * \param [in] rot_origin             Origin for Euler rotation.
 * \param [in] n_points               Number of points in pos_vec.
 */
static void
t8_euler_rotation (double *pos_vec, double *rot_vec, double *res_vec, double *rot_origin, int n_points)
{
  for (int point_iterator = 0; point_iterator < n_points; ++point_iterator) {
    res_vec[point_iterator * 3 + 0]
      = (cos (rot_vec[0]) * cos (rot_vec[2]) - sin (rot_vec[0]) * cos (rot_vec[1]) * sin (rot_vec[2]))
        * (pos_vec[point_iterator * 3 + 0] - rot_origin[0]);
    res_vec[point_iterator * 3 + 0]
      += (-cos (rot_vec[0]) * sin (rot_vec[2]) - sin (rot_vec[0]) * cos (rot_vec[1]) * cos (rot_vec[2]))
         * (pos_vec[point_iterator * 3 + 1] - rot_origin[1]);
    res_vec[point_iterator * 3 + 0]
      += (sin (rot_vec[0]) * sin (rot_vec[1])) * (pos_vec[point_iterator * 3 + 2] - rot_origin[2]);
    res_vec[point_iterator * 3 + 0] += rot_origin[0];

    res_vec[point_iterator * 3 + 1]
      = (sin (rot_vec[0]) * cos (rot_vec[2]) + cos (rot_vec[0]) * cos (rot_vec[1]) * sin (rot_vec[2]))
        * (pos_vec[point_iterator * 3 + 0] - rot_origin[0]);
    res_vec[point_iterator * 3 + 1]
      += (-sin (rot_vec[0]) * sin (rot_vec[2]) + cos (rot_vec[0]) * cos (rot_vec[1]) * cos (rot_vec[2]))
         * (pos_vec[point_iterator * 3 + 1] - rot_origin[1]);
    res_vec[point_iterator * 3 + 1]
      += (-cos (rot_vec[0]) * sin (rot_vec[1])) * (pos_vec[point_iterator * 3 + 2] - rot_origin[2]);
    res_vec[point_iterator * 3 + 1] += rot_origin[1];

    res_vec[point_iterator * 3 + 2]
      = (sin (rot_vec[1]) * sin (rot_vec[2])) * (pos_vec[point_iterator * 3 + 0] - rot_origin[0]);
    res_vec[point_iterator * 3 + 2]
      += (sin (rot_vec[1]) * cos (rot_vec[2])) * (pos_vec[point_iterator * 3 + 1] - rot_origin[1]);
    res_vec[point_iterator * 3 + 2] += (cos (rot_vec[1])) * (pos_vec[point_iterator * 3 + 2] - rot_origin[2]);
    res_vec[point_iterator * 3 + 2] += rot_origin[2];
  }
}

/** Constructs a cad surface for testing purposes. Surface is build on the x-z-plane and ranges from 0 to 1 in its dimensions.
 * Saves the surface in the shape.
 * \return                            The shape.
 */
TopoDS_Shape
t8_create_cad_surface_shape_x_z ()
{
  Handle_Geom_Surface surface;
  TopoDS_Shape shape;
  TColgp_Array2OfPnt point_array (1, 3, 1, 3);

  point_array (1, 1) = gp_Pnt (0, 0, 0);
  point_array (2, 1) = gp_Pnt (-0.2, 0.2, 0.5);
  point_array (3, 1) = gp_Pnt (0, 0, 1);

  point_array (1, 2) = gp_Pnt (0.5, 0.2, -0.2);
  point_array (2, 2) = gp_Pnt (0.5, 0, 0.5);
  point_array (3, 2) = gp_Pnt (0.5, -0.2, 1.2);

  point_array (1, 3) = gp_Pnt (1, 0, 0);
  point_array (2, 3) = gp_Pnt (1.2, -0.2, 0.5);
  point_array (3, 3) = gp_Pnt (1, 0, 1);

  surface = GeomAPI_PointsToBSplineSurface (point_array).Surface ();
  shape = BRepBuilderAPI_MakeFace (surface, 1e-6).Face ();
  return shape;
}

/** Constructs a cad surface for testing purposes. Surface is build on the x-y-plane in reference space.
 * Saves the surface in the shape.
 * \return                            The shape.
 */
TopoDS_Shape
t8_create_cad_surface_shape_x_y ()
{
  Handle_Geom_Surface surface;
  TopoDS_Shape shape;
  TColgp_Array2OfPnt point_array (1, 3, 1, 3);

  point_array (1, 1) = gp_Pnt (0, 0, 0);
  point_array (2, 1) = gp_Pnt (-0.2, 0.5, 0.2);
  point_array (3, 1) = gp_Pnt (0, 1, 0);

  point_array (1, 2) = gp_Pnt (0.5, -0.2, 0.2);
  point_array (2, 2) = gp_Pnt (0.5, 0.5, 0);
  point_array (3, 2) = gp_Pnt (0.5, 1.2, -0.2);

  point_array (1, 3) = gp_Pnt (1, 0, 0);
  point_array (2, 3) = gp_Pnt (1.2, 0.5, -0.2);
  point_array (3, 3) = gp_Pnt (1, 1, 0);

  surface = GeomAPI_PointsToBSplineSurface (point_array).Surface ();
  shape = BRepBuilderAPI_MakeFace (surface, 1e-6).Face ();
  return shape;
}

/** Constructs a cad curve for testing purposes. Curve is build on the x-axis in reference space.
 * Saves the curve in the shape.
 * \return                            The cad shape.
 */
TopoDS_Shape
t8_create_cad_curve_shape ()
{
  Handle_Geom_Curve curve;
  TopoDS_Shape shape;
  TColgp_Array1OfPnt point_array (1, 5);

  point_array (1) = gp_Pnt (0.00, 0.00, 0.00);
  point_array (2) = gp_Pnt (0.24, 0.20, 0.20);
  point_array (3) = gp_Pnt (0.50, 0.00, 0.40);
  point_array (4) = gp_Pnt (0.75, -0.20, 0.20);
  point_array (5) = gp_Pnt (1.00, 0.00, 0.00);

  curve = GeomAPI_PointsToBSpline (point_array).Curve ();
  shape = BRepBuilderAPI_MakeEdge (curve).Edge ();
  return shape;
}

/** Constructs a cmesh with an cad geometry linked hypercube.
 * \param [in] rot_vec                The rotation vector to rotate the cube before linking a geometry to it.
 * \param [in] face                   The index of the face to link a surface to. -1 for no face.
 * \param [in] edge                   The index of the edge to link a curve to. -1 for no edge.
 * \param [in] parameters             Parameters of the curve/surface.
 * \return                            A valid cmesh, as if _init and _commit had been called.
 */
t8_cmesh_t
t8_create_cad_hypercube ([[maybe_unused]] double *rot_vec, [[maybe_unused]] int face, [[maybe_unused]] int edge,
                         [[maybe_unused]] double *parameters)
{
  if (edge >= 0 && face >= 0) {
    SC_ABORTF ("Please specify only an edge or a face.");
  }

  t8_cmesh_t cmesh;
  t8_cmesh_init (&cmesh);
  t8_cmesh_set_tree_class (cmesh, 0, T8_ECLASS_HEX);

  double rotated_vertices[24],
    vertices[24] = { 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1 };
  double rotation_origin[3] = { 0.5, 0.5, 0.5 };

  t8_euler_rotation (vertices, rot_vec, rotated_vertices, rotation_origin, 8);
  t8_cmesh_set_tree_vertices (cmesh, 0, rotated_vertices, 24);

  int faces[6] = { 0 };
  int edges[24] = { 0 };
  T8_ASSERT (face < 0 || edge < 0);
  if (face >= 0) {
    faces[face] = 1;
    t8_cmesh_register_geometry<t8_geometry_cad> (cmesh, t8_create_cad_surface_shape_x_z ());
    t8_cmesh_set_attribute (cmesh, 0, t8_get_package_id (), T8_CMESH_CAD_FACE_PARAMETERS_ATTRIBUTE_KEY + face,
                            parameters, 8 * sizeof (double), 0);
  }
  else if (edge >= 0) {
    edges[edge] = 1;
    t8_cmesh_register_geometry<t8_geometry_cad> (cmesh, t8_create_cad_curve_shape ());
    t8_cmesh_set_attribute (cmesh, 0, t8_get_package_id (), T8_CMESH_CAD_EDGE_PARAMETERS_ATTRIBUTE_KEY + edge,
                            parameters, 2 * sizeof (double), 0);
  }
  else {
    /* Even if we do not want to link any geometry to the edges or faces,
     * we have to create a geometry. Hence a cad geometry can only be created
     * with an actual shape, we just create a geometry with a curve and do not
     * link the curve to any edge. */
    t8_cmesh_register_geometry<t8_geometry_cad> (cmesh, t8_create_cad_curve_shape ());
  }
  t8_cmesh_set_attribute (cmesh, 0, t8_get_package_id (), T8_CMESH_CAD_FACE_ATTRIBUTE_KEY, faces, 6 * sizeof (int), 0);
  t8_cmesh_set_attribute (cmesh, 0, t8_get_package_id (), T8_CMESH_CAD_EDGE_ATTRIBUTE_KEY, edges, 24 * sizeof (int), 0);
  t8_cmesh_commit (cmesh, sc_MPI_COMM_WORLD);
  return cmesh;
}

/** Tests the cad geometry functions for hexahedra.
 * \param [in] rot_vec                The rotation vector to rotate the hypercube.
 * \param [in] face                   The face to test. -1 for no face.
 * \param [in] edge                   The edge to test. -1 for no edge.
 * \param [in] parameters             The parameters of the curve/surface.
 * \param [in] test_ref_coords        List of 8 coordinates to test.
 * \param [in] test_return_coords     List of 8 expected output coordinates.
 * \param [in] comm                   The mpi communicator to use.
 * \return                            Returns 1 if passed, 0 if failed.
 */
void
t8_test_geometry_cad_hex (double *rot_vec, int face, int edge, double *parameters, double *test_ref_coords,
                          const t8_vec<24> &test_return_coords)
{
  const int num_coords = 8; /* Number of reference coordinates to test */
  t8_vec<num_coords * 3> out_coords;
  double rotated_test_ref_coords[24];
  double rotation_origin[3] = { 0.5, 0.5, 0.5 };
  double inversed_rot_vec[3];
  double tol = T8_PRECISION_EPS > 1e-10 ? T8_PRECISION_EPS : 1e-10;
  t8_cmesh_t cmesh = t8_create_cad_hypercube (rot_vec, face, edge, parameters);

  for (int i_coord = 0; i_coord < 3; ++i_coord) {
    inversed_rot_vec[2 - i_coord] = -rot_vec[i_coord];
  }
  t8_euler_rotation (test_ref_coords, inversed_rot_vec, rotated_test_ref_coords, rotation_origin, num_coords);
  for (size_t coord = 0; coord < num_coords; ++coord) {
    t8_geometry_evaluate (cmesh, 0, rotated_test_ref_coords, num_coords, out_coords.data ());
    EXPECT_VEC_EQ (out_coords, test_return_coords, tol);
  }
  t8_cmesh_destroy (&cmesh);
}

TEST (t8_gtest_geometry_cad_hex, linked_faces)
{
  /* clang-format off */
  double test_ref_coords[24] = { 0.1, 0.1, 0.1,
                                 0.8, 0.1,  0.1,
                                 0.15, 0.9, 0.1,
                                 0.9,  0.9,  0.3,
                                 0.3, 0.1, 0.7,
                                 0.9, 0.25, 0.95,
                                 0.1,  0.9, 0.9,
                                 0.95, 0.85, 0.8 };
  const t8_vec<24> surface_test_return_coords ({ 0.0396282769,  0.1897542602, 0.0396282769,
                                            0.8553975402, 0.1510451803, -0.0012778561,
                                            0.1434278361, 0.9117760771, 0.0909403721,
                                            0.9149739120, 0.8893780561,  0.2953610950,
                                            0.2190065733, 0.1000000000, 0.7809934267,
                                            0.9318450385,  0.1898146343, 0.9989190836,
                                            0.0932920308, 0.9000000000, 0.9067079692,
                                            0.9673042609, 0.8312979801, 0.8063743210 });

    /* clang-format off */
  double surface_rot_vecs[18] = {
    M_PI / 2, 0, 0,     /* Face 0 */
    M_PI * 3 / 2, 0, 0, /* Face 1 */
    0, 0, 0,            /* Face 2 */
    M_PI, 0, 0,         /* Face 3 */
    0, M_PI * 3 / 2, 0, /* Face 4 */
    0, M_PI / 2, 0      /* Face 5 */
  };
  double surface_parameters[48] = {
    0, 1, 0, 0, 1, 1, 1, 0,  /* Face 0 */
    0, 0, 0, 1, 1, 0, 1, 1,  /* Face 1 */
    0, 0, 0, 1, 1, 0, 1, 1,  /* Face 2 */
    0, 1, 0, 0, 1, 1, 1, 0,  /* Face 3 */
    1, 0, 1, 1, 0, 0, 0, 1,  /* Face 4 */
    0, 0, 0, 1, 1, 0, 1, 1   /* Face 5 */
  };
  for (int i_faces = 0; i_faces < 6; ++i_faces) {
    t8_test_geometry_cad_hex (surface_rot_vecs + i_faces * 3, i_faces, -1, surface_parameters + i_faces * 8,
                          test_ref_coords, surface_test_return_coords);
  }
}

TEST (t8_gtest_geometry_cad_hex, linked_edges)
{
  /* clang-format off */
  double test_ref_coords[24] = { 0.1, 0.1, 0.1,
                                 0.8, 0.1,  0.1,
                                 0.15, 0.9, 0.1,
                                 0.9,  0.9,  0.3,
                                 0.3, 0.1, 0.7,
                                 0.9, 0.25, 0.95,
                                 0.1,  0.9, 0.9,
                                 0.95, 0.85, 0.8 };
  const t8_vec<24> curve_test_return_coords ({ 0.0955204602, 0.2235162028, 0.1217553783,
                                          0.7995278713, -0.0659838746, 0.2083328730,
                                          0.1494299582, 0.9170222805, 0.1069555502,
                                          0.8999105642, 0.8892289094, 0.3015732294,
                                          0.2987855815, 0.1481519479, 0.7726155646,
                                          0.8999520880, 0.2442297729, 0.9508428015,
                                          0.0999446970, 0.9015248914, 0.9002685849,
                                          0.9499697383, 0.8472575225, 0.7998496263 });
  /* clang-format on */

  double curve_rot_vecs[36] = {
    0,         0,         0,          // Edge 0
    0,         -M_PI / 2, 0,          // Edge 1
    0,         M_PI / 2,  0,          // Edge 2
    0,         M_PI,      0,          // Edge 3
    M_PI / 2,  0,         0,          // Edge 4
    -M_PI / 2, 0,         0,          // Edge 5
    0,         M_PI,      -M_PI / 2,  // Edge 6
    0,         M_PI,      M_PI / 2,   // Edge 7
    -M_PI / 2, M_PI / 2,  M_PI / 2,   // Edge 8
    M_PI / 2,  M_PI / 2,  -M_PI / 2,  // Edge 9
    M_PI / 2,  -M_PI / 2, 0,          // Edge 10
    -M_PI / 2, -M_PI / 2, 0,          // Edge 11
  };
  double curve_parameters[24] = {
    0, 1,  // Edge 0
    0, 1,  // Edge 1
    0, 1,  // Edge 2
    0, 1,  // Edge 3
    1, 0,  // Edge 4
    0, 1,  // Edge 5
    0, 1,  // Edge 6
    1, 0,  // Edge 7
    1, 0,  // Edge 8
    0, 1,  // Edge 9
    1, 0,  // Edge 10
    0, 1   // Edge 11
  };
  for (int i_edges = 0; i_edges < 12; ++i_edges) {
    t8_test_geometry_cad_hex (curve_rot_vecs + i_edges * 3, -1, i_edges, curve_parameters + i_edges * 2,
                              test_ref_coords, curve_test_return_coords);
  }
}

/** Constructs a cmesh with an cad geometry linked tetrahedron.
 * \param [in] face                   The index of the face to link a surface to. -1 for no face.
 * \param [in] edge                   The index of the edge to link a curve to. -1 for no edge.
 * \param [in] parameters             Parameters of the curve/surface.
 * \return                            A valid cmesh, as if _init and _commit had been called.
 */
t8_cmesh_t
t8_create_cad_reference_tet ([[maybe_unused]] int face, [[maybe_unused]] int edge, [[maybe_unused]] double *parameters)
{
  if (edge >= 0 && face >= 0) {
    SC_ABORTF ("Please specify only an edge or a face.");
  }

  const int num_vertices = t8_eclass_num_vertices[T8_ECLASS_TET];

  t8_cmesh_t cmesh;
  t8_cmesh_init (&cmesh);
  t8_cmesh_set_tree_class (cmesh, 0, T8_ECLASS_TET);

  double vertices_face[48] = { 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 0, 1,   /* linked face: 0 */
                               0, 0, 0, 1, 1, 1, 1, 0, 0, 1, 0, 1,   /* linked face: 1 */
                               1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 1,   /* linked face: 2 */
                               0, 0, 0, 1, 0, 0, 1, 0, 1, 1, 1, 1 }; /* linked face: 3 */

  double vertices_edge[72] = { 0, 0, 0, 1, 0, 0, 1, 0, 1, 1, 1, 1,   /* linked edge: 0 */
                               1, 0, 0, 1, 0, 1, 0, 0, 0, 1, 1, 1,   /* linekd edge: 1 */
                               1, 0, 0, 1, 1, 1, 1, 0, 1, 0, 0, 0,   /* linekd edge: 2 */
                               1, 0, 1, 0, 0, 0, 1, 0, 0, 1, 1, 1,   /* linekd edge: 3 */
                               1, 1, 1, 0, 0, 0, 1, 0, 1, 1, 0, 0,   /* linekd edge: 4 */
                               1, 1, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0 }; /* linekd edge: 5 */

  t8_cmesh_set_tree_vertices (cmesh, 0, (face >= 0 ? vertices_face + face * 12 : vertices_edge + edge * 12),
                              num_vertices);

  int faces[4] = { 0 };
  int edges[12] = { 0 };
  T8_ASSERT (face < 0 || edge < 0);
  if (face >= 0) {
    faces[face] = 1;
    t8_cmesh_register_geometry<t8_geometry_cad> (cmesh, t8_create_cad_surface_shape_x_z ());
    t8_cmesh_set_attribute (cmesh, 0, t8_get_package_id (), T8_CMESH_CAD_FACE_PARAMETERS_ATTRIBUTE_KEY + face,
                            parameters, 6 * sizeof (double), 0);
  }
  else if (edge >= 0) {
    edges[edge] = 1;
    t8_cmesh_register_geometry<t8_geometry_cad> (cmesh, t8_create_cad_curve_shape ());
    t8_cmesh_set_attribute (cmesh, 0, t8_get_package_id (), T8_CMESH_CAD_EDGE_PARAMETERS_ATTRIBUTE_KEY + edge,
                            parameters, 2 * sizeof (double), 0);
  }
  else {
    /* Even if we do not want to link any geometry to the edges or faces,
     * we have to create a geometry. Hence a cad geometry can only be created
     * with an actual shape, we just create a geometry with a curve and do not
     * link the curve to any edge. */
    t8_cmesh_register_geometry<t8_geometry_cad> (cmesh, t8_create_cad_curve_shape ());
  }
  t8_cmesh_set_attribute (cmesh, 0, t8_get_package_id (), T8_CMESH_CAD_FACE_ATTRIBUTE_KEY, faces, 4 * sizeof (int), 0);
  t8_cmesh_set_attribute (cmesh, 0, t8_get_package_id (), T8_CMESH_CAD_EDGE_ATTRIBUTE_KEY, edges, 12 * sizeof (int), 0);
  t8_cmesh_commit (cmesh, sc_MPI_COMM_WORLD);
  return cmesh;
}

/** Tests the cad geometry functions for tetrahedra.
 * \param [in] face                   The face to test. -1 for no face.
 * \param [in] edge                   The edge to test. -1 for no edge.
 * \param [in] parameters             The parameters of the curve/surface.
 * \param [in] test_ref_coords        List of coordinates to test.
 * \param [in] test_return_coords     List of expected output coordinates.
 * \param [in] comm                   The mpi communicator to use.
 * \return                            Returns 1 if passed, 0 if failed.
 */
template <size_t dimension>
void
t8_test_geometry_cad_tet (const int face, const int edge, double *parameters, double *test_ref_coords,
                          const t8_vec<dimension> &test_return_coords)
{
  /* 4 coords for face --> 3 vertices of face & element centroid
   * 2 coords for edge --> 2 vertices of edge
   * multiplied by 3 it is equal to the dimension template parameter
   */
  const int num_coords = (face >= 0 ? 4 : 2);
  t8_vec<dimension> out_coords;
  double tol = T8_PRECISION_EPS > 1e-10 ? T8_PRECISION_EPS : 1e-10;

  t8_cmesh_t cmesh = t8_create_cad_reference_tet (face, edge, parameters);

  for (int i_coord = 0; i_coord < num_coords; ++i_coord) {
    const int offset_3d = i_coord * 3;
    t8_geometry_evaluate (cmesh, 0, test_ref_coords + offset_3d + (face >= 0 ? face * 12 : edge * 6), 1,
                          &out_coords[offset_3d]);
  }

  EXPECT_VEC_EQ (out_coords, test_return_coords, tol);
  t8_cmesh_destroy (&cmesh);
}

TEST (t8_gtest_geometry_cad_tet, linked_faces)
{
  /* clang-format off */
  double test_ref_coords[48]
    = { 1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0,    // face 0
        0.75, 0.25, 0.5,                                // element centroid
        0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0,    // face 1
        0.75, 0.25, 0.5,                                // element centroid
        1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0,    // face 2
        0.75, 0.25, 0.5,                                // element centroid
        0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0,    // face 3
        0.75, 0.25, 0.5 };                              // element centroid

  const t8_vec<12> surface_test_return_coords
     ({ 0.0, 0.0, 0.0,                      // face vertex 0
        1.0, 0.0, 0.0,                      // face vertex 1
        1.0, 0.0, 1.0,                      // face vertex 2
        0.7953692655, 0.25, 0.4546307344});  // element centroid (shifted)

  double surface_parameters[27]
    = { 0, 1, 0, 0, 1, 1,   // face 0
        0, 0, 0, 1, 1, 1,   // face 1
        0, 1, 0, 0, 1, 1,   // face 2
        0, 0, 0, 1, 1, 1 }; // face 3
  /* clang-format on */

  for (int i_faces = 0; i_faces < 4; i_faces++) {
    t8_test_geometry_cad_tet<12> (i_faces, -1, surface_parameters + i_faces * 6, test_ref_coords,
                                  surface_test_return_coords);
  }
}

TEST (t8_gtest_geometry_cad_tet, linked_edges)
{
  /* clang-format off */
  double test_ref_coords[36]
    = { 0.0, 0.0, 0.0, 1.0, 0.0, 0.0,   // edge 0
        1.0, 0.0, 1.0, 0.0, 0.0, 0.0,   // edge 1
        1.0, 1.0, 1.0, 0.0, 0.0, 0.0,   // edge 2
        1.0, 0.0, 0.0, 1.0, 0.0, 1.0,   // edge 3
        1.0, 0.0, 0.0, 1.0, 1.0, 1.0,   // edge 4
        1.0, 1.0, 1.0, 1.0, 0.0, 1.0 }; // edge 5
  t8_vec<6> curve_test_return_coords
     ({ 0.0, 0.0, 0.0,    // edge vertex 0
        1.0, 0.0, 0.0 });  // edge vertex 1
  double curve_parameters[12] = {
    0, 1,  // edge 0
    1, 0,  // edge 1
    1, 0,  // edge 2
    0, 1,  // edge 3
    0, 1,  // edge 4
    1, 0,  // edge 5
  };
  /* clang-format on */

  for (int i_edges = 0; i_edges < 6; ++i_edges) {
    t8_test_geometry_cad_tet<6> (-1, i_edges, curve_parameters + i_edges * 2, test_ref_coords,
                                 curve_test_return_coords);
  }
}

TEST (t8_gtest_geometry_cad, jacobian)
{
  t8_cmesh_t cmesh;
  double jacobian[9], rot_vec[3] = { 0, 0, 0 }, ref_coords[3] = { 0.5, 0.5, 0.5 };
  /* clang-format off */
  double jacobian_expect[9] = { 1, 0, 0,
                                0, 1, 0,
                                0, 0, 1 };
  /* clang-format on */
  cmesh = t8_create_cad_hypercube (rot_vec, -1, -1, NULL);
  t8_geometry_jacobian (cmesh, 0, ref_coords, 1, jacobian);
  for (int i = 0; i < 9; ++i) {
    EXPECT_FLOAT_EQ (jacobian[i], jacobian_expect[i]);
  }
  t8_cmesh_destroy (&cmesh);
}

/* The test checks if the mapping algorithms for curved 2d elements do not shift values on an edge which is not curved.
 * In that case, the cad geometry should output the same out_coords as the linear geometry function. */
struct class_2d_element_cad_curve: public testing::TestWithParam<std::tuple<t8_eclass, int>>
{
 protected:
  void
  SetUp () override
  {
    eclass = std::get<0> (GetParam ());
    /* curvature prescribes if the linear of curved curve is used */
    curvature = std::get<1> (GetParam ());
    T8_ASSERT (0 <= eclass && eclass < T8_ECLASS_COUNT);
    Handle_Geom_Curve cad_curve_linear, cad_curve_curved;
    TColgp_Array1OfPnt point_array_linear (1, 2);
    TColgp_Array1OfPnt point_array_curved (1, 3);

    /* LINEAR
    *  x--> u-parameter
    *
    *                 curve
    *  ----------------------------------
    *
    *  0                                1
    *
    *  CURVED
    *  x--> u-parameter
    *
    *  ----____       curve       ____----
    *           ----_________----
    *
    *  0               0.5              1
    */

    point_array_linear (1)
      = gp_Pnt (test_ref_coords_out_linear[0], test_ref_coords_out_linear[1], test_ref_coords_out_linear[2]);
    point_array_linear (2)
      = gp_Pnt (test_ref_coords_out_linear[6], test_ref_coords_out_linear[7], test_ref_coords_out_linear[8]);

    point_array_curved (1)
      = gp_Pnt (test_ref_coords_out_curved[0], test_ref_coords_out_curved[1], test_ref_coords_out_curved[2]);
    point_array_curved (2)
      = gp_Pnt (test_ref_coords_out_curved[3], test_ref_coords_out_curved[4], test_ref_coords_out_curved[5]);
    point_array_curved (3)
      = gp_Pnt (test_ref_coords_out_curved[6], test_ref_coords_out_curved[7], test_ref_coords_out_curved[8]);

    cad_curve_linear = GeomAPI_PointsToBSpline (point_array_linear).Curve ();
    cad_curve_curved = GeomAPI_PointsToBSpline (point_array_curved).Curve ();
    shape_linear = BRepBuilderAPI_MakeEdge (cad_curve_linear).Edge ();
    shape_curved = BRepBuilderAPI_MakeEdge (cad_curve_curved).Edge ();

    num_vertices = t8_eclass_num_vertices[eclass];
  }

  void
  TearDown () override
  {
    /* The cmesh is destroyed in the test itself. */
  }
  t8_cmesh_t cmesh;
  t8_eclass_t eclass;
  int curvature;
  TopoDS_Shape shape_linear, shape_curved;
  /* Saving the corner vertices for the given element class. */
  size_t num_vertices;
  const double vertices_tri[27] = { 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0,
                                    0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0 };
  const double vertices_quad[48] = { 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0,
                                     1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0,
                                     0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0 };
  /* Edges are parameterized in one parameter u. The array contains the parameters
   * each vertex of the edge has on the linked curve. */
  double params_tri[6] = { 0, 1, 1, 0, 0, 1 };
  double params_quad[8] = { 0, 1, 1, 0, 1, 0, 0, 1 };
  /* The array prescribes the linkage of the element. No face is linked. */
  std::array<int, 1> faces = { 0 };
  std::array<int, 8> edges = { 0 };

  const int linked_edge_tri[3] = { 2, 1, 0 };
  const int linked_edge_quad[4] = { 2, 0, 3, 1 };

  const double test_ref_coords_tri_in[27] = { 0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.5, 0.5,
                                              0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.5, 0.0, 1.0, 1.0, 0.0 };
  const double test_ref_coords_quad_in[36]
    = { 0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0,
        1.0, 1.0, 0.0, 0.5, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.5, 0.0, 1.0, 1.0, 0.0 };
  const t8_vec<9> test_ref_coords_out_linear = t8_vec<9> ({ 0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 1.0, 0.0, 0.0 });
  const t8_vec<9> test_ref_coords_out_curved = t8_vec<9> ({ 0.0, 0.0, 0.0, 0.5, -0.2, 0.0, 1.0, 0.0, 0.0 });
};

TEST_P (class_2d_element_cad_curve, t8_check_2d_element_cad_curve)
{

  for (size_t i_orientation = 0; i_orientation < num_vertices; ++i_orientation) {
    const int orientation = i_orientation * num_vertices * T8_ECLASS_MAX_DIM;

    edges.fill (0);

    const int linked_edge
      = (eclass == T8_ECLASS_QUAD ? linked_edge_quad[i_orientation] : linked_edge_tri[i_orientation]);
    edges[linked_edge] = 1;

    t8_cmesh_init (&cmesh);
    t8_cmesh_set_tree_class (cmesh, 0, eclass);
    t8_cmesh_register_geometry<t8_geometry_cad> (cmesh, (curvature == 0 ? shape_linear : shape_curved));

    t8_cmesh_set_tree_vertices (
      cmesh, 0, (eclass == T8_ECLASS_QUAD ? vertices_quad + orientation : vertices_tri + orientation), num_vertices);

    /* Passing of the attributes to the element */
    t8_cmesh_set_attribute (cmesh, 0, t8_get_package_id (), T8_CMESH_CAD_FACE_ATTRIBUTE_KEY, faces.data (),
                            sizeof (int), 0);
    t8_cmesh_set_attribute (cmesh, 0, t8_get_package_id (), T8_CMESH_CAD_EDGE_ATTRIBUTE_KEY, edges.data (),
                            2 * num_vertices * sizeof (int), 0);
    t8_cmesh_set_attribute (
      cmesh, 0, t8_get_package_id (), T8_CMESH_CAD_EDGE_PARAMETERS_ATTRIBUTE_KEY + linked_edge,
      (eclass == T8_ECLASS_QUAD ? (params_quad + 2 * i_orientation) : (params_tri + 2 * i_orientation)),
      2 * sizeof (double), 0);

    /* Commit the cmesh */
    t8_cmesh_commit (cmesh, sc_MPI_COMM_WORLD);

    t8_3D_vec out_coords;

    /* out_coords should be equal to the input ref_coords. */
    for (size_t i_coord = 0; i_coord < T8_ECLASS_MAX_DIM; ++i_coord) {
      t8_geometry_evaluate (cmesh, 0,
                            (eclass == T8_ECLASS_QUAD ? test_ref_coords_quad_in : test_ref_coords_tri_in)
                              + i_coord * T8_ECLASS_MAX_DIM + i_orientation * 9,
                            1, out_coords.data ());

      //t8_vec<9> checked_coords = (curvature == 0 ? test_ref_coords_out_linear : test_ref_coords_out_curved);
      t8_3D_vec checked_coords;
      for (int i = 0; i < 3; ++i) {
        checked_coords[i]
          = (curvature == 0 ? test_ref_coords_out_linear : test_ref_coords_out_curved)[i_coord * T8_ECLASS_MAX_DIM + i];
      }
      EXPECT_VEC_EQ (checked_coords, out_coords, T8_PRECISION_EPS);
    }
    t8_cmesh_destroy (&cmesh);
  }
}

INSTANTIATE_TEST_SUITE_P (t8_gtest_check_2d_element_cad_curve, class_2d_element_cad_curve,
                          testing::Combine (AllEclasses2D, testing::Values (0, 1)));

/* The test checks if the mapping algorithms for curved 2d elements do not shift values on a surface which is not curved.
 * In that case, the cad geometry should output the same out_coords as the linear geometry function. */
struct class_2d_element_linear_cad_surface: public testing::TestWithParam<t8_eclass_t>
{
 protected:
  void
  SetUp () override
  {
    eclass = GetParam ();
    T8_ASSERT (0 <= eclass && eclass < T8_ECLASS_COUNT);
    Handle_Geom_Surface cad_surface;
    TColgp_Array2OfPnt point_array (1, 2, 1, 2);

    /*  x--> u-parameter
    *   |
    *   v v-parameter
    *
    *     point_array  1               2
    *
    *         1        -----------------
    *                  |               |
    *                  |               |
    *                  | plane surface |
    *                  |               |
    *                  |               |
    *         2        -----------------
    */

    point_array (1, 1) = gp_Pnt (0.0, 1.0, 0.0);
    point_array (2, 1) = gp_Pnt (1.0, 1.0, 0.0);

    point_array (1, 2) = gp_Pnt (0.0, 0.0, 0.0);
    point_array (2, 2) = gp_Pnt (1.0, 0.0, 0.0);

    cad_surface = GeomAPI_PointsToBSplineSurface (point_array).Surface ();
    shape = BRepBuilderAPI_MakeFace (cad_surface, 1e-6).Face ();

    t8_cmesh_init (&cmesh);
    t8_cmesh_set_tree_class (cmesh, 0, eclass);
  }

  void
  TearDown () override
  {
    t8_cmesh_destroy (&cmesh);
  }
  t8_cmesh_t cmesh;
  t8_eclass_t eclass;
  TopoDS_Shape shape;
  /* The arrays prescribe the linkage of the element. The face of the element is linked and all edges are not */
  int faces[1] = { 1 };
  int edges[8] = { 0 };
  /* First 6 ref_coords for triangle and all 9 ref_coords for quad */
  t8_vec<27> test_ref_coords = t8_vec<27> ({ 0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 1.0, 0.0, 0.0, 0.5, 0.5, 0.0, 0.0, 1.0,
                                             0.0, 0.0, 0.5, 0.0, 0.5, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.5, 0.0 });
  /* TODO: use randomised test_ref_coords, because the out_coords should be the same, no matter the test_ref_coord. */
};

TEST_P (class_2d_element_linear_cad_surface, t8_check_2d_element_linear_cad_surface)
{
  /* Saving the corner vertices for the given element class. */
  const int num_vertices = t8_eclass_num_vertices[eclass];
  const double *vertices = &(t8_element_corner_ref_coords[eclass][0][0]);

  t8_cmesh_set_tree_vertices (cmesh, 0, vertices, num_vertices);

  /* Surfaces are parameterized in two parameters u and v. The arrays contain the parameters
   * each vertex of the element has on the linked surface. The parameters are stored in
   * u0, v0, u1, v1... in order of the element vertices. */
  double params_quad[8] = { 0, 1, 1, 1, 0, 0, 1, 0 };
  double params_tri[6] = { 0, 1, 1, 1, 1, 0 };

  /* Passing of the attributes to the element */
  t8_cmesh_set_attribute (cmesh, 0, t8_get_package_id (), T8_CMESH_CAD_FACE_ATTRIBUTE_KEY, faces, sizeof (int), 0);
  t8_cmesh_set_attribute (cmesh, 0, t8_get_package_id (), T8_CMESH_CAD_EDGE_ATTRIBUTE_KEY, edges,
                          2 * num_vertices * sizeof (int), 0);
  t8_cmesh_set_attribute (cmesh, 0, t8_get_package_id (), T8_CMESH_CAD_FACE_PARAMETERS_ATTRIBUTE_KEY,
                          (eclass == T8_ECLASS_QUAD ? params_quad : params_tri), 2 * num_vertices * sizeof (double), 0);

  /* Register the geometry */
  t8_cmesh_register_geometry<t8_geometry_cad> (cmesh, shape);
  /* Commit the cmesh */
  t8_cmesh_commit (cmesh, sc_MPI_COMM_WORLD);

  t8_3D_vec out_coords;

  /* `out_coords` should be equal to the input `ref_coords`. */
  for (size_t i_coord = 0; i_coord < (eclass == T8_ECLASS_QUAD ? 9 : 6); ++i_coord) {
    t8_geometry_evaluate (cmesh, 0, &test_ref_coords[i_coord * 3], 1, out_coords.data ());
    t8_3D_vec checked_coords;
    for (int i = 0; i < 3; ++i) {
      checked_coords[i] = test_ref_coords[i_coord * T8_ECLASS_MAX_DIM + i];
    }
    EXPECT_VEC_EQ (checked_coords, out_coords, T8_PRECISION_EPS);
  }
}

INSTANTIATE_TEST_SUITE_P (t8_gtest_check_2d_element_linear_cad_surface, class_2d_element_linear_cad_surface,
                          AllEclasses2D, print_eclass);

/* The test checks if the mapping algorithms for curved 2d elements shift values on a curved surface correctly. */
struct class_2d_element_curved_cad_surface: public testing::TestWithParam<t8_eclass_t>
{
 protected:
  void
  SetUp () override
  {
    eclass = GetParam ();
    T8_ASSERT (0 <= eclass && eclass < T8_ECLASS_COUNT);
    Handle_Geom_Surface cad_surface;
    TColgp_Array2OfPnt point_array (1, 3, 1, 3);
    TopoDS_Shape shape;

    /*  x--> u-parameter
    *   |
    *   v v-parameter
    *
    *   x -> shifted point to create curved surface
    *
    *     point_array  1       2       3
    *                          x
    *         1        -----------------
    *                  |               |
    *                  |               |
    *         2      x |       x       | x
    *                  |               |
    *                  |               |
    *         3        -----------------
    *                          x
    */

    point_array (1, 1) = gp_Pnt (test_ref_coords_out[12], test_ref_coords_out[13], test_ref_coords_out[14]);
    point_array (2, 1) = gp_Pnt (test_ref_coords_out[18], test_ref_coords_out[19], test_ref_coords_out[20]);
    point_array (3, 1) = gp_Pnt (test_ref_coords_out[21], test_ref_coords_out[22], test_ref_coords_out[23]);

    point_array (1, 2) = gp_Pnt (test_ref_coords_out[15], test_ref_coords_out[16], test_ref_coords_out[17]);
    point_array (2, 2) = gp_Pnt (test_ref_coords_out[9], test_ref_coords_out[10], test_ref_coords_out[11]);
    point_array (3, 2) = gp_Pnt (test_ref_coords_out[24], test_ref_coords_out[25], test_ref_coords_out[26]);

    point_array (1, 3) = gp_Pnt (test_ref_coords_out[0], test_ref_coords_out[1], test_ref_coords_out[2]);
    point_array (2, 3) = gp_Pnt (test_ref_coords_out[3], test_ref_coords_out[4], test_ref_coords_out[5]);
    point_array (3, 3) = gp_Pnt (test_ref_coords_out[6], test_ref_coords_out[7], test_ref_coords_out[8]);

    cad_surface = GeomAPI_PointsToBSplineSurface (point_array).Surface ();
    shape = BRepBuilderAPI_MakeFace (cad_surface, 1e-6).Face ();

    t8_cmesh_init (&cmesh);
    t8_cmesh_set_tree_class (cmesh, 0, eclass);
    t8_cmesh_register_geometry<t8_geometry_cad> (cmesh, shape);
  }

  void
  TearDown () override
  {
    t8_cmesh_destroy (&cmesh);
  }
  t8_cmesh_t cmesh;
  t8_eclass_t eclass;

  /* The arrays prescribe the linkage of the element. The face of the element is linked and all edges are not */
  int faces[1] = { 1 };
  int edges[8] = { 0 };

  /* First 6 ref_coords for triangle and all 9 ref_coords for quad */
  t8_vec<27> test_ref_coords_out = t8_vec<27> ({ 0.0, 0.0,  0.0, 0.5, -0.2, 0.0, 1.0, 0.0, 0.0, 0.5, 0.5, 0.2, 0.0, 1.0,
                                                 0.0, -0.2, 0.5, 0.0, 0.5,  1.2, 0.0, 1.0, 1.0, 0.0, 1.2, 0.5, 0.0 });
  const t8_vec<27> test_ref_coords_in
    = t8_vec<27> ({ 0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 1.0, 0.0, 0.0, 0.5, 0.5, 0.0, 0.0, 1.0,
                    0.0, 0.0, 0.5, 0.0, 0.5, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.5, 0.0 });
};

TEST_P (class_2d_element_curved_cad_surface, t8_check_2d_element_curved_cad_surface)
{
  /* Saving the corner vertices for the given element class. */
  const int num_vertices = t8_eclass_num_vertices[eclass];
  const double *vertices = &(t8_element_corner_ref_coords[eclass][0][0]);

  t8_cmesh_set_tree_vertices (cmesh, 0, vertices, num_vertices);

  /* Surfaces are parameterized in two parameters u and v. The arrays contain the parameters
   * each vertex of the element has on the linked surface. The parameters are stored in
   * u0, v0, u1, v1... in order of the element vertices. */
  double params_quad[8] = { 0, 1, 1, 1, 0, 0, 1, 0 };
  double params_tri[6] = { 0, 1, 1, 1, 1, 0 };

  /* Passing of the attributes to the element */
  t8_cmesh_set_attribute (cmesh, 0, t8_get_package_id (), T8_CMESH_CAD_FACE_ATTRIBUTE_KEY, faces, sizeof (int), 0);
  t8_cmesh_set_attribute (cmesh, 0, t8_get_package_id (), T8_CMESH_CAD_EDGE_ATTRIBUTE_KEY, edges,
                          2 * num_vertices * sizeof (int), 0);
  t8_cmesh_set_attribute (cmesh, 0, t8_get_package_id (), T8_CMESH_CAD_FACE_PARAMETERS_ATTRIBUTE_KEY,
                          (eclass == T8_ECLASS_QUAD ? params_quad : params_tri), 2 * num_vertices * sizeof (double), 0);

  /* Commit the cmesh */
  t8_cmesh_commit (cmesh, sc_MPI_COMM_WORLD);

  t8_3D_vec out_coords;

  /* out_coords should be equal to the input ref_coords. */
  for (size_t i_coord = 0; i_coord < (eclass == T8_ECLASS_QUAD ? 9 : 6); ++i_coord) {
    t8_geometry_evaluate (cmesh, 0, test_ref_coords_in.data () + i_coord * 3, 1, out_coords.data ());
    t8_3D_vec checked_coords;
    for (int i = 0; i < 3; ++i) {
      checked_coords[i] = test_ref_coords_out[i_coord * T8_ECLASS_MAX_DIM + i];
    }
    EXPECT_VEC_EQ (checked_coords, out_coords, T8_PRECISION_EPS);
  }
}

INSTANTIATE_TEST_SUITE_P (t8_gtest_check_2d_element_curved_cad_surface, class_2d_element_curved_cad_surface,
                          AllEclasses2D, print_eclass);

/** Constructs a cmesh with an cad geometry linked prism.
 * \param [in] face                   The index of the face to link a surface to. -1 for no face.
 * \param [in] edge                   The index of the edge to link a curve to. -1 for no edge.
 * \param [in] parameters             Parameters of the curve/surface.
 * \return                            A valid cmesh, as if _init and _commit had been called.
 */
t8_cmesh_t
t8_create_cad_reference_prism ([[maybe_unused]] int face, [[maybe_unused]] int edge,
                               [[maybe_unused]] double *parameters)
{
  if (edge >= 0 && face >= 0) {
    SC_ABORTF ("Please specify only an edge or a face.");
  }

  const int num_vertices = t8_eclass_num_vertices[T8_ECLASS_PRISM];
  const int face_vertices = (face <= 2 ? 4 : 3); /* The number of vertices of the prism face */

  t8_cmesh_t cmesh;
  t8_cmesh_init (&cmesh);
  t8_cmesh_set_tree_class (cmesh, 0, T8_ECLASS_PRISM);

  double vertices_face[90] = { 1, 1, 1, 1, 0, 1, 0, 0, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0,   /* linked face: 0 */
                               1, 0, 0, 1, 1, 1, 0, 0, 0, 1, 0, 1, 1, 1, 1, 0, 0, 1,   /* linked face: 1 */
                               0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1, 1, 1, 1,   /* linked face: 2 */
                               0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1, 1, 1, 1,   /* linked face: 3 */
                               1, 1, 1, 1, 0, 1, 0, 0, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0 }; /* linked face: 4 */

  double vertices_edge[162] = { 1, 1, 0, 0, 0, 0, 1, 0, 0, 1, 1, 1, 0, 0, 1, 1, 0, 1,   /* linked edge: 0 */
                                1, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 1, 1, 1, 1, 0, 0, 1,   /* linked edge: 1 */
                                0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1, 1, 1, 1,   /* linked edge: 2 */
                                1, 1, 1, 1, 0, 1, 0, 0, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0,   /* linked edge: 3 */
                                0, 0, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0, 1, 1, 0, 1, 0, 0,   /* linked edge: 4 */
                                1, 0, 1, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0,   /* linked edge: 5 */
                                1, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1,   /* linked edge: 6 */
                                0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 1, 0, 1, 1, 0, 0,   /* linked edge: 7 */
                                1, 0, 0, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0 }; /* linked edge: 8 */

  t8_cmesh_set_tree_vertices (cmesh, 0, (face >= 0 ? vertices_face + face * 18 : vertices_edge + edge * 18),
                              num_vertices);

  int faces[5] = { 0 };
  int edges[18] = { 0 };
  T8_ASSERT (face < 0 || edge < 0);
  if (face >= 0) {
    faces[face] = 1;
    t8_cmesh_register_geometry<t8_geometry_cad> (
      cmesh, (face <= 2 ? t8_create_cad_surface_shape_x_z () : t8_create_cad_surface_shape_x_y ()));
    t8_cmesh_set_attribute (cmesh, 0, t8_get_package_id (), T8_CMESH_CAD_FACE_PARAMETERS_ATTRIBUTE_KEY + face,
                            parameters, 2 * face_vertices * sizeof (double), 0);
  }
  else if (edge >= 0) {
    edges[edge] = 1;
    t8_cmesh_register_geometry<t8_geometry_cad> (cmesh, t8_create_cad_curve_shape ());
    t8_cmesh_set_attribute (cmesh, 0, t8_get_package_id (), T8_CMESH_CAD_EDGE_PARAMETERS_ATTRIBUTE_KEY + edge,
                            parameters, 2 * sizeof (double), 0);
  }
  else {
    /* Even if we do not want to link any geometry to the edges or faces,
     * we have to create a geometry. Hence a cad geometry can only be created
     * with an actual shape, we just create a geometry with a curve and do not
     * link the curve to any edge. */
    t8_cmesh_register_geometry<t8_geometry_cad> (cmesh, t8_create_cad_curve_shape ());
  }
  t8_cmesh_set_attribute (cmesh, 0, t8_get_package_id (), T8_CMESH_CAD_FACE_ATTRIBUTE_KEY, faces, 5 * sizeof (int), 0);
  t8_cmesh_set_attribute (cmesh, 0, t8_get_package_id (), T8_CMESH_CAD_EDGE_ATTRIBUTE_KEY, edges, 18 * sizeof (int), 0);
  t8_cmesh_commit (cmesh, sc_MPI_COMM_WORLD);
  return cmesh;
}

/** Tests the cad geometry functions for prisms.
 * \param [in] face                   The face to test. -1 for no face.
 * \param [in] edge                   The edge to test. -1 for no edge.
 * \param [in] parameters             The parameters of the curve/surface.
 * \param [in] test_ref_coords        List of coordinates to test.
 * \param [in] test_return_coords     List of expected output coordinates.
 * \param [in] comm                   The mpi communicator to use.
 * \return                            Returns 1 if passed, 0 if failed.
 */
template <size_t dimension>
void
t8_test_geometry_cad_prism (const int face, const int edge, double *parameters, double *test_ref_coords,
                            const t8_vec<dimension> &test_return_coords)
{
  t8_3D_vec out_coords;
  double tol = T8_PRECISION_EPS > 1e-10 ? T8_PRECISION_EPS : 1e-10;
  const int face_vertices = (face <= 2 ? 4 : 3);

  t8_cmesh_t cmesh = t8_create_cad_reference_prism (face, edge, parameters);

  for (int i_coord = 0; i_coord < (face >= 0 ? face_vertices : 3); ++i_coord) {
    t8_geometry_evaluate (cmesh, 0, test_ref_coords + i_coord * 3 + (face >= 0 ? face * 12 : edge * 9), 1,
                          out_coords.data ());
    t8_3D_vec checked_coords;
    for (int i = 0; i < 3; ++i) {
      checked_coords[i] = test_return_coords[i_coord * T8_ECLASS_MAX_DIM + i];
    }
    EXPECT_VEC_EQ (out_coords, checked_coords, tol);
  }
  t8_cmesh_destroy (&cmesh);
}

TEST (t8_gtest_geometry_cad_prism, linked_faces)
{
  /* clang-format off */
  double test_ref_coords[60]
    = { 1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0,   // face 0
        1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 1.0,   // face 1
        0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0,   // face 2
        0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 2.0, 2.0, 2.0,   // face 3
        1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 2.0, 2.0, 2.0 }; // face 4
    /* The 2.0's at face 3 and 4 are placeholders, because both faces only have 3 vertices. */

  const t8_vec<12> surface_test_return_coords_quad
    ({ 0.0, 0.0, 0.0,    // face vertex 0
        1.0, 0.0, 0.0,    // face vertex 1
        0.0, 0.0, 1.0,    // face vertex 2
        1.0, 0.0, 1.0 });  // face vertex 3
  const t8_vec<9> surface_test_return_coords_tri
    ({ 0.0, 0.0, 0.0,    // face vertex 0
        1.0, 0.0, 0.0,    // face vertex 1
        1.0, 1.0, 0.0 });  // face vertex 2

  double surface_parameters[40]
    = { 1, 1, 1, 0, 0, 1, 0, 0,   // face 0
        0, 1, 0, 0, 1, 1, 1, 0,   // face 1
        0, 0, 0, 1, 1, 0, 1, 1,   // face 2
        0, 0, 0, 1, 1, 1, 2, 2,   // face 3
        1, 1, 0, 1, 0, 0, 2, 2 }; // face 4
    /* The 2's of face 3 and 4 are placeholders, because both faces only have 3 vertices. */
  /* clang-format on */

  for (int i_faces = 0; i_faces < 5; i_faces++) {
    if (i_faces <= 2) {
      const t8_vec<12> surface_test_return_coords = surface_test_return_coords_quad;
      t8_test_geometry_cad_prism (i_faces, -1, surface_parameters + i_faces * 8, test_ref_coords,
                                  surface_test_return_coords);
    }
    else {
      const t8_vec<9> surface_test_return_coords = surface_test_return_coords_tri;
      t8_test_geometry_cad_prism (i_faces, -1, surface_parameters + i_faces * 8, test_ref_coords,
                                  surface_test_return_coords);
    }
  }
}

TEST (t8_gtest_geometry_cad_prism, linked_edges)
{
  /* clang-format off */
  double test_ref_coords[81]
    = { 1.0, 0.0, 0.0, 1.0, 0.5, 0.0, 1.0, 1.0, 0.0,   // edge 0
        1.0, 1.0, 0.0, 0.5, 0.5, 0.0, 0.0, 0.0, 0.0,   // edge 1
        0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 1.0, 0.0, 0.0,   // edge 2
        1.0, 1.0, 1.0, 1.0, 0.5, 1.0, 1.0, 0.0, 1.0,   // edge 3
        0.0, 0.0, 1.0, 0.5, 0.5, 1.0, 1.0, 1.0, 1.0,   // edge 4
        1.0, 0.0, 1.0, 0.5, 0.0, 1.0, 0.0, 0.0, 1.0,   // edge 5
        1.0, 0.0, 1.0, 1.0, 0.0, 0.5, 1.0, 0.0, 0.0,   // edge 6
        1.0, 1.0, 0.0, 1.0, 1.0, 0.5, 1.0, 1.0, 1.0,   // edge 7
        0.0, 0.0, 1.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0 }; // edge 8
  t8_vec<9> curve_test_return_coords
    ({ 0.0, 0.0, 0.0,                            // edge vertex 0
        0.4999500215, 0.0000523914, 0.4000007585, // center of edge
        1.0, 0.0, 0.0 });                          // edge vertex 1
  double curve_parameters[18] = {
    0, 1,  // edge 0
    1, 0,  // edge 1
    0, 1,  // edge 2
    1, 0,  // edge 3
    0, 1,  // edge 4
    1, 0,  // edge 5
    1, 0,  // edge 6
    0, 1,  // edge 7
    1, 0 };// edge 8
  /* clang-format on */

  for (int i_edges = 0; i_edges < 9; ++i_edges) {
    t8_test_geometry_cad_prism (-1, i_edges, curve_parameters + i_edges * 2, test_ref_coords, curve_test_return_coords);
  }
}

/* ====================================================================== *
 * Periodic-seam projection tests.
 *
 * These tests guard against a bug in which the cad geometry linearly
 * interpolated the two stored CAD parameters of a mesh edge/face that
 * straddles the closure ("seam") of a periodic CAD curve or surface
 * without unwrapping across the period. The interpolation then traversed
 * the long arc around the parametric loop, so the high-order nodes were
 * placed on the far side of the geometry (e.g. the antipodal point of a
 * circle) instead of along the short arc where the element actually lives.
 * ====================================================================== */

/** Builds a full circle of radius 1 centred at the origin in the x-y plane.
 * A full circle is a periodic CAD curve (period 2*pi), so it exercises the
 * periodic-seam unwrap. The curve is wrapped into a TopoDS_Edge so it can be
 * linked to a cmesh edge like the other test curves. */
static TopoDS_Shape
t8_create_cad_periodic_circle_shape ()
{
  const gp_Circ circle (gp_Ax2 (gp_Pnt (0, 0, 0), gp_Dir (0, 0, 1)), 1.0);
  Handle_Geom_Circle geom_circle = new Geom_Circle (circle);
  return BRepBuilderAPI_MakeEdge (geom_circle).Edge ();
}

/* Links a periodic circle to one edge of a single quad and evaluates the
 * high-order node at the midpoint of that edge. The edge straddles the
 * circle's seam (its two vertices sit at parameters 7*pi/4 and pi/4). The
 * short arc between them passes through parameter 0 == 2*pi, i.e. the point
 * (1, 0, 0). Before the unwrap fix the evaluator interpolated to parameter
 * pi and returned the antipodal point (-1, 0, 0). */
TEST (t8_gtest_geometry_cad_periodic, quad_curve_seam)
{
  const double radius = 1.0;
  const double p0 = 7.0 * M_PI / 4.0; /* 315 degrees, just before the seam */
  const double p1 = 1.0 * M_PI / 4.0; /*  45 degrees, just after the seam  */

  /* Physical positions of the two seam-straddling edge endpoints. */
  const double v0x = radius * cos (p0), v0y = radius * sin (p0); /* ( 0.707, -0.707) */
  const double v1x = radius * cos (p1), v1y = radius * sin (p1); /* ( 0.707,  0.707) */

  /* Quad vertices in zorder (v0, v1, v2, v3). The curve is linked to edge 2,
   * which connects corners 0 and 1. v2 and v3 are placed outward (+x) so the
   * quad is non-degenerate; they do not influence the on-edge evaluation. */
  /* clang-format off */
  double vertices[12] = {
    v0x,       v0y, 0.0,   /* v0 (corner 0) */
    v1x,       v1y, 0.0,   /* v1 (corner 1) */
    v0x + 1.0, v0y, 0.0,   /* v2 (corner 2) */
    v1x + 1.0, v1y, 0.0    /* v3 (corner 3) */
  };
  /* clang-format on */

  const int linked_edge = 2;
  int faces[1] = { 0 };
  int edges[8] = { 0 };
  edges[linked_edge] = 1; /* link curve index 1 (the circle) to edge 2 */
  double params[2] = { p0, p1 };

  t8_cmesh_t cmesh;
  t8_cmesh_init (&cmesh);
  t8_cmesh_set_tree_class (cmesh, 0, T8_ECLASS_QUAD);
  t8_cmesh_register_geometry<t8_geometry_cad> (cmesh, t8_create_cad_periodic_circle_shape ());
  t8_cmesh_set_tree_vertices (cmesh, 0, vertices, 4);
  t8_cmesh_set_attribute (cmesh, 0, t8_get_package_id (), T8_CMESH_CAD_FACE_ATTRIBUTE_KEY, faces, sizeof (int), 0);
  t8_cmesh_set_attribute (cmesh, 0, t8_get_package_id (), T8_CMESH_CAD_EDGE_ATTRIBUTE_KEY, edges, 8 * sizeof (int), 0);
  t8_cmesh_set_attribute (cmesh, 0, t8_get_package_id (), T8_CMESH_CAD_EDGE_PARAMETERS_ATTRIBUTE_KEY + linked_edge,
                          params, 2 * sizeof (double), 0);
  t8_cmesh_commit (cmesh, sc_MPI_COMM_WORLD);

  /* Evaluate the high-order node at the midpoint of the curved (bottom) edge.
   * The short-arc midpoint is parameter 2*pi == 0, i.e. (radius, 0, 0). */
  double ref_coords[2] = { 0.5, 0.0 };
  t8_3D_vec out_coords;
  t8_geometry_evaluate (cmesh, 0, ref_coords, 1, out_coords.data ());

  const t8_3D_vec expected ({ radius, 0.0, 0.0 });
  EXPECT_VEC_EQ (out_coords, expected, 1e-9);

  t8_cmesh_destroy (&cmesh);
}

/* Same scenario as quad_curve_seam, but for a hexahedron, exercising the 3D
 * curve-projection path (t8_geom_evaluate_cad_hex -> process_curve). The
 * periodic circle is linked to edge 0, which connects corners 0 and 1 and
 * straddles the seam. Evaluating at the midpoint of that edge must land on the
 * short-arc point (radius, 0, 0), not the antipodal point. */
TEST (t8_gtest_geometry_cad_periodic, hex_curve_seam)
{
  const double radius = 1.0;
  const double p0 = 7.0 * M_PI / 4.0;
  const double p1 = 1.0 * M_PI / 4.0;
  const double v0x = radius * cos (p0), v0y = radius * sin (p0);
  const double v1x = radius * cos (p1), v1y = radius * sin (p1);

  /* Hex vertices in zorder. Edge 0 connects corners 0 and 1. Corners 0 and 1
   * sit on the circle straddling the seam; the remaining corners are offset in
   * +x and +z so the hex is non-degenerate but do not affect the on-edge
   * evaluation at ref (0.5, 0, 0). */
  /* clang-format off */
  double vertices[24] = {
    v0x,       v0y, 0.0,   /* corner 0 */
    v1x,       v1y, 0.0,   /* corner 1 */
    v0x + 1.0, v0y, 0.0,   /* corner 2 */
    v1x + 1.0, v1y, 0.0,   /* corner 3 */
    v0x,       v0y, 1.0,   /* corner 4 */
    v1x,       v1y, 1.0,   /* corner 5 */
    v0x + 1.0, v0y, 1.0,   /* corner 6 */
    v1x + 1.0, v1y, 1.0    /* corner 7 */
  };
  /* clang-format on */

  const int linked_edge = 0;
  int faces[6] = { 0 };
  int edges[24] = { 0 };
  edges[linked_edge] = 1; /* link curve index 1 (the circle) to edge 0 */
  double params[2] = { p0, p1 };

  t8_cmesh_t cmesh;
  t8_cmesh_init (&cmesh);
  t8_cmesh_set_tree_class (cmesh, 0, T8_ECLASS_HEX);
  t8_cmesh_register_geometry<t8_geometry_cad> (cmesh, t8_create_cad_periodic_circle_shape ());
  t8_cmesh_set_tree_vertices (cmesh, 0, vertices, 8);
  t8_cmesh_set_attribute (cmesh, 0, t8_get_package_id (), T8_CMESH_CAD_FACE_ATTRIBUTE_KEY, faces, 6 * sizeof (int), 0);
  t8_cmesh_set_attribute (cmesh, 0, t8_get_package_id (), T8_CMESH_CAD_EDGE_ATTRIBUTE_KEY, edges, 24 * sizeof (int), 0);
  t8_cmesh_set_attribute (cmesh, 0, t8_get_package_id (), T8_CMESH_CAD_EDGE_PARAMETERS_ATTRIBUTE_KEY + linked_edge,
                          params, 2 * sizeof (double), 0);
  t8_cmesh_commit (cmesh, sc_MPI_COMM_WORLD);

  double ref_coords[3] = { 0.5, 0.0, 0.0 };
  t8_3D_vec out_coords;
  t8_geometry_evaluate (cmesh, 0, ref_coords, 1, out_coords.data ());

  const t8_3D_vec expected ({ radius, 0.0, 0.0 });
  EXPECT_VEC_EQ (out_coords, expected, 1e-9);

  t8_cmesh_destroy (&cmesh);
}

/** Builds a *clamped* (non-periodic) B-spline curve that closes on itself by
 * interpolating eight points around the unit circle plus a ninth point equal
 * to the first. GeomAPI_PointsToBSpline returns a clamped curve, so
 * IsPeriodic() is false even though the geometry closes (FirstParameter and
 * LastParameter map to the same point). This is the representation that the
 * gmsh-imported "organic_shape" mesh actually uses, and the one the
 * IsPeriodic()-only check would miss. */
static TopoDS_Shape
t8_create_cad_clamped_closed_curve_shape ()
{
  TColgp_Array1OfPnt point_array (1, 9);
  for (int i = 0; i < 8; ++i) {
    const double angle = 2.0 * M_PI * i / 8.0;
    point_array (i + 1) = gp_Pnt (cos (angle), sin (angle), 0.0);
  }
  point_array (9) = point_array (1); /* close the loop: last point == first */
  Handle_Geom_Curve curve = GeomAPI_PointsToBSpline (point_array).Curve ();
  return BRepBuilderAPI_MakeEdge (curve).Edge ();
}

/* Links a CLAMPED (non-periodic) but geometrically closed B-spline to a quad
 * edge straddling its seam. This is the representation OpenCASCADE hands back
 * for the gmsh "organic_shape" hole, so it exercises the coincident-endpoint
 * closed-curve detection and the out-of-range parameter fold-back that the
 * plain IsPeriodic() check cannot. The evaluated edge midpoint must lie on the
 * short arc near the seam, not on the long arc across the loop.
 *
 * Ground truth is taken from OCC's own curve evaluation (process_curve maps a
 * single parameter to a point), so this checks the evaluator's parameter
 * handling rather than re-deriving the B-spline geometry. */
TEST (t8_gtest_geometry_cad_periodic, quad_clamped_closed_curve_seam)
{
  const TopoDS_Shape shape = t8_create_cad_clamped_closed_curve_shape ();
  const TopoDS_Edge edge = TopoDS::Edge (shape);
  Standard_Real first = 0.0, last = 0.0;
  const Handle_Geom_Curve curve = BRep_Tool::Curve (edge, first, last);
  ASSERT_FALSE (curve.IsNull ());
  /* The whole point: clamped representation reports non-periodic ... */
  ASSERT_FALSE (curve->IsPeriodic ());
  /* ... but the endpoints coincide, so the curve is closed. */
  gp_Pnt point_first, point_last;
  curve->D0 (first, point_first);
  curve->D0 (last, point_last);
  ASSERT_LE (point_first.Distance (point_last), 1e-9);

  const double period = last - first;
  /* Edge straddles the seam: one vertex at the seam (param first), the other
   * near the far end (param first + 0.9*period, physically near the seam). */
  const double p0 = first;
  const double p1 = first + 0.9 * period;

  gp_Pnt curve_v0, curve_v1;
  curve->D0 (p0, curve_v0);
  curve->D0 (p1, curve_v1);

  /* clang-format off */
  double vertices[12] = {
    curve_v0.X (),       curve_v0.Y (), 0.0,   /* v0 (corner 0) */
    curve_v1.X (),       curve_v1.Y (), 0.0,   /* v1 (corner 1) */
    curve_v0.X () + 1.0, curve_v0.Y (), 0.0,   /* v2 (corner 2) */
    curve_v1.X () + 1.0, curve_v1.Y (), 0.0    /* v3 (corner 3) */
  };
  /* clang-format on */

  const int linked_edge = 2;
  int faces[1] = { 0 };
  int edges[8] = { 0 };
  edges[linked_edge] = 1;
  double params[2] = { p0, p1 };

  t8_cmesh_t cmesh;
  t8_cmesh_init (&cmesh);
  t8_cmesh_set_tree_class (cmesh, 0, T8_ECLASS_QUAD);
  t8_cmesh_register_geometry<t8_geometry_cad> (cmesh, shape);
  t8_cmesh_set_tree_vertices (cmesh, 0, vertices, 4);
  t8_cmesh_set_attribute (cmesh, 0, t8_get_package_id (), T8_CMESH_CAD_FACE_ATTRIBUTE_KEY, faces, sizeof (int), 0);
  t8_cmesh_set_attribute (cmesh, 0, t8_get_package_id (), T8_CMESH_CAD_EDGE_ATTRIBUTE_KEY, edges, 8 * sizeof (int), 0);
  t8_cmesh_set_attribute (cmesh, 0, t8_get_package_id (), T8_CMESH_CAD_EDGE_PARAMETERS_ATTRIBUTE_KEY + linked_edge,
                          params, 2 * sizeof (double), 0);
  t8_cmesh_commit (cmesh, sc_MPI_COMM_WORLD);

  double ref_coords[2] = { 0.5, 0.0 };
  t8_3D_vec out_coords;
  t8_geometry_evaluate (cmesh, 0, ref_coords, 1, out_coords.data ());

  /* Short-arc midpoint parameter: unwrap (p1 -= period) then interpolate at
   * 0.5 then fold back into [first, last]. Long-arc (buggy) midpoint is the
   * naive interpolation of the stored parameters. */
  gp_Pnt expected_pnt, buggy_pnt;
  curve->D0 (last - 0.05 * period, expected_pnt);
  curve->D0 (first + 0.45 * period, buggy_pnt);
  const t8_3D_vec expected ({ expected_pnt.X (), expected_pnt.Y (), expected_pnt.Z () });
  EXPECT_VEC_EQ (out_coords, expected, 1e-9);

  /* And sanity: clearly not the long-arc point on the far side of the loop. */
  const double dist_to_buggy = std::sqrt ((out_coords[0] - buggy_pnt.X ()) * (out_coords[0] - buggy_pnt.X ())
                                          + (out_coords[1] - buggy_pnt.Y ()) * (out_coords[1] - buggy_pnt.Y ()));
  EXPECT_GT (dist_to_buggy, 1.0);

  t8_cmesh_destroy (&cmesh);
}

/** Builds a cylinder of radius 1 about the z-axis. A cylinder is periodic in
 * its u-direction (the angular parameter, period 2*pi) and non-periodic in v
 * (the axial parameter), so it exercises the surface-seam unwrap in u only.
 * The surface is wrapped into a face so it can be linked to a cmesh edge. */
static TopoDS_Shape
t8_create_cad_periodic_cylinder_shape ()
{
  const gp_Cylinder cylinder (gp_Ax3 (gp_Pnt (0, 0, 0), gp_Dir (0, 0, 1)), 1.0);
  Handle_Geom_CylindricalSurface geom_cylinder = new Geom_CylindricalSurface (cylinder);
  return BRepBuilderAPI_MakeFace (geom_cylinder, 1e-6).Face ();
}

/* Links a periodic cylinder surface to one edge of a quad and evaluates the
 * high-order node at the midpoint of that edge. The edge straddles the
 * cylinder's u-seam: its endpoints sit at (u, v) = (7*pi/4, 0) and (pi/4, 0).
 * The short arc in u passes through u = 0 == 2*pi, i.e. the point
 * (radius, 0, 0). Before the unwrap fix the evaluator interpolated to u = pi
 * and returned the antipodal point (-radius, 0, 0). */
TEST (t8_gtest_geometry_cad_periodic, quad_surface_seam)
{
  const double radius = 1.0;
  const double u0 = 7.0 * M_PI / 4.0;
  const double u1 = 1.0 * M_PI / 4.0;
  const double v0x = radius * cos (u0), v0y = radius * sin (u0);
  const double v1x = radius * cos (u1), v1y = radius * sin (u1);

  /* clang-format off */
  double vertices[12] = {
    v0x,       v0y, 0.0,   /* v0 (corner 0) */
    v1x,       v1y, 0.0,   /* v1 (corner 1) */
    v0x + 1.0, v0y, 0.0,   /* v2 (corner 2) */
    v1x + 1.0, v1y, 0.0    /* v3 (corner 3) */
  };
  /* clang-format on */

  const int num_edges = 4;
  const int linked_edge = 2;
  int faces[1] = { 0 };
  int edges[8] = { 0 };
  /* Link surface index 1 (the cylinder) to edge 2. Surfaces are stored in the
   * second half of the per-edge linkage array. */
  edges[linked_edge + num_edges] = 1;
  /* Surface parameters at the edge's two vertices, laid out u0, v0, u1, v1. */
  double params[4] = { u0, 0.0, u1, 0.0 };

  t8_cmesh_t cmesh;
  t8_cmesh_init (&cmesh);
  t8_cmesh_set_tree_class (cmesh, 0, T8_ECLASS_QUAD);
  t8_cmesh_register_geometry<t8_geometry_cad> (cmesh, t8_create_cad_periodic_cylinder_shape ());
  t8_cmesh_set_tree_vertices (cmesh, 0, vertices, 4);
  t8_cmesh_set_attribute (cmesh, 0, t8_get_package_id (), T8_CMESH_CAD_FACE_ATTRIBUTE_KEY, faces, sizeof (int), 0);
  t8_cmesh_set_attribute (cmesh, 0, t8_get_package_id (), T8_CMESH_CAD_EDGE_ATTRIBUTE_KEY, edges, 8 * sizeof (int), 0);
  t8_cmesh_set_attribute (cmesh, 0, t8_get_package_id (), T8_CMESH_CAD_EDGE_PARAMETERS_ATTRIBUTE_KEY + linked_edge,
                          params, 4 * sizeof (double), 0);
  t8_cmesh_commit (cmesh, sc_MPI_COMM_WORLD);

  double ref_coords[2] = { 0.5, 0.0 };
  t8_3D_vec out_coords;
  t8_geometry_evaluate (cmesh, 0, ref_coords, 1, out_coords.data ());

  const t8_3D_vec expected ({ radius, 0.0, 0.0 });
  EXPECT_VEC_EQ (out_coords, expected, 1e-9);

  t8_cmesh_destroy (&cmesh);
}
