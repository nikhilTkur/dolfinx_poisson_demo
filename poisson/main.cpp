#include "poisson.h"
#include <cmath>
#include <dolfinx.h>
#include <dolfinx/fem/Constant.h>
#include <dolfinx/fem/petsc.h>
#include <iostream>
#include <xtensor/xarray.hpp>
#include <xtensor/xview.hpp>

using namespace dolfinx;

int main(int argc, char* argv[])
{
  common::subsystem::init_logging(argc, argv);
  common::subsystem::init_petsc(argc, argv);

  {
    // Create mesh and function space
    auto mesh = std::make_shared<mesh::Mesh>(generation::RectangleMesh::create(
        MPI_COMM_WORLD, {{{0.0, 0.0, 0.0}, {1.0, 1.0, 0.0}}}, {8192, 8192},
        mesh::CellType::triangle, mesh::GhostMode::none));
    auto V = fem::create_functionspace(functionspace_form_poisson_a, "u", mesh);

    auto f = std::make_shared<fem::Function<PetscScalar>>(V);

    // Define variational forms
    std::map<std::string, std::shared_ptr<const fem::Constant<double>>>
        constants;
    auto a = std::make_shared<fem::Form<PetscScalar>>(
        fem::create_form<PetscScalar>(*form_poisson_a, {V, V}, {}, {constants},
                                      {}));

    auto L = std::make_shared<fem::Form<PetscScalar>>(
        fem::create_form<PetscScalar>(*form_poisson_L, {V}, {{"f", f}}, {},
                                      {}));

    auto u0 = std::make_shared<fem::Function<PetscScalar>>(V);
    u0->interpolate(
        [](const xt::xtensor<double, 2>& x) -> xt::xarray<PetscScalar> {
          return 1 + xt::square(xt::row(x, 0)) + 2 * xt::square(xt::row(x, 1));
        });

    const auto bdofs = fem::locate_dofs_geometrical(
        {*V},
        [](const xt::xtensor<double, 2>& x) -> xt::xtensor<bool, 1>
        {
          auto x0 = xt::row(x, 0);
          auto x1 = xt::row(x, 1);
          return xt::isclose(x0, 0.0) or xt::isclose(x0, 1.0)
                 or xt::isclose(x1, 0.0) or xt::isclose(x1, 1.0);
        });

    std::vector bc{std::make_shared<const fem::DirichletBC<PetscScalar>>(
        u0, std::move(bdofs))};

    f->interpolate(
        [](const xt::xtensor<double, 2>& x) -> xt::xarray<PetscScalar> {
          return -6.0 * xt::ones<PetscScalar>({x.shape(1)});
        });

    fem::Function<PetscScalar> u(V);
    la::PETScMatrix A = la::PETScMatrix(fem::create_matrix(*a), false);
    la::PETScVector b(*L->function_spaces()[0]->dofmap()->index_map,
                      L->function_spaces()[0]->dofmap()->index_map_bs());

    MatZeroEntries(A.mat());
    fem::assemble_matrix(la::PETScMatrix::set_block_fn(A.mat(), ADD_VALUES), *a,
                         bc);
    MatAssemblyBegin(A.mat(), MAT_FLUSH_ASSEMBLY);
    MatAssemblyEnd(A.mat(), MAT_FLUSH_ASSEMBLY);
    fem::set_diagonal(la::PETScMatrix::set_fn(A.mat(), INSERT_VALUES), *V, bc);
    MatAssemblyBegin(A.mat(), MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(A.mat(), MAT_FINAL_ASSEMBLY);

    VecSet(b.vec(), 0.0);
    VecGhostUpdateBegin(b.vec(), INSERT_VALUES, SCATTER_FORWARD);
    VecGhostUpdateEnd(b.vec(), INSERT_VALUES, SCATTER_FORWARD);
    fem::assemble_vector_petsc(b.vec(), *L);
    fem::apply_lifting_petsc(b.vec(), {a}, {{bc}}, {}, 1.0);
    VecGhostUpdateBegin(b.vec(), ADD_VALUES, SCATTER_REVERSE);
    VecGhostUpdateEnd(b.vec(), ADD_VALUES, SCATTER_REVERSE);
    fem::set_bc_petsc(b.vec(), bc, nullptr);

    la::PETScKrylovSolver lu(MPI_COMM_WORLD);
    la::PETScOptions::set("ksp_type", "cg");
    la::PETScOptions::set("ksp_rtol", 1.0e-8);
    la::PETScOptions::set("pc_type", "hypre");
    la::PETScOptions::set("pc_hypre_type", "boomeramg");
    la::PETScOptions::set("pc_hypre_boomeramg_strong_threshold", 0.7);
    la::PETScOptions::set("pc_hypre_boomeramg_agg_nl", 4);
    la::PETScOptions::set("pc_hypre_boomeramg_agg_num_paths", 2);
    lu.set_from_options();

    lu.set_operator(A.mat());
    std::cout << "solution started";
    dolfinx::common::Timer timer_cgs("~CG Solver");
    lu.solve(u.vector(), b.vec());
    timer_cgs.stop();
  }
  dolfinx::list_timings(MPI_COMM_WORLD, {dolfinx::TimingType::wall});
  common::subsystem::finalize_petsc();
  return 0;
}
