// This code conforms with the UFC specification version 2018.2.0.dev0
// and was automatically generated by FFCx version 0.1.1.dev0.
//
// This code was generated with the following parameters:
//
//  {'assume_aligned': -1,
//   'epsilon': 1e-14,
//   'output_directory': '.',
//   'padlen': 1,
//   'profile': False,
//   'scalar_type': 'double',
//   'table_atol': 1e-09,
//   'table_rtol': 1e-06,
//   'tabulate_tensor_void': False,
//   'ufl_file': ['poisson.ufl'],
//   'verbosity': 30,
//   'visualise': False}


#pragma once

typedef double ufc_scalar_t;
#include <ufc.h>

#ifdef __cplusplus
extern "C" {
#endif

extern ufc_finite_element element_00d4fefb25fc8ed8563136725dd7f20b01dabe87;

extern ufc_finite_element element_30501addce65c40f51d2333fb2adc3f5a9795426;

extern ufc_dofmap dofmap_00d4fefb25fc8ed8563136725dd7f20b01dabe87;

extern ufc_dofmap dofmap_30501addce65c40f51d2333fb2adc3f5a9795426;

extern ufc_integral integral_1d33afccde6662b8b39cd6f6cf291d04ebb6b306;

extern ufc_integral integral_eb8004c26e818f76c60141234b6d9437aea7585c;

extern ufc_form form_4a5635c2e319573a99e1a891162527b8ece1d4d8;

// Helper used to create form using name which was given to the
// form in the UFL file.
// This helper is called in user c++ code.
//
extern ufc_form* form_poisson_a;

// Helper used to create function space using function name
// i.e. name of the Python variable.
//
ufc_function_space* functionspace_form_poisson_a(const char* function_name);

extern ufc_form form_0d1e67a283fa63ae74908f991f7d0bdd56b6cb52;

// Helper used to create form using name which was given to the
// form in the UFL file.
// This helper is called in user c++ code.
//
extern ufc_form* form_poisson_L;

// Helper used to create function space using function name
// i.e. name of the Python variable.
//
ufc_function_space* functionspace_form_poisson_L(const char* function_name);

#ifdef __cplusplus
}
#endif
