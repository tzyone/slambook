#include <iostream>
#include <ctime>

using namespace std;

#include <Eigen/Core>
#include <Eigen/Dense>

#define MATRIX_SIZE 50

int main(int argc, char** argv) {

    Eigen::Matrix<float, 2, 3> matrix_23;

    Eigen::Vector3d v_3d;

    Eigen::Matrix3d matrix_33 = Eigen::Matrix3d::Zero();

    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> matrix_dynamic;

    Eigen::MatrixXd matrix_x;

    matrix_23 << 1, 2, 3, 4, 5, 6;

    std::cout << matrix_23 << std::endl;

    for (int i = 0; i < matrix_23.rows(); i++) {
        for (int j = 0; j < matrix_23.cols(); j++) {
            std::cout << matrix_23(i, j) <<std::endl;
        }
    }

    v_3d << 3, 2, 1;
    //Eigen::Matrix<double, 2, 1> result = matrix_23 * v_3d;
    //Eigen::Matrix<double, 2, 3> result = matrix_23.cast<double>() * v_3d;
    Eigen::Matrix<double, 2, 1> result = matrix_23.cast<double>() * v_3d;
    std::cout << result << std::endl << std::endl;

    matrix_33 = Eigen::Matrix3d::Random();
    std::cout << matrix_33 << std::endl << std::endl;

    std::cout << matrix_33.transpose() << std::endl << std::endl;
    std::cout << matrix_33.sum() << std::endl << std::endl;
    std::cout << matrix_33.trace() << std::endl << std::endl;
    std::cout << 10*matrix_33 << std::endl << std::endl;
    std::cout << matrix_33.inverse() << std::endl << std::endl;
    std::cout << matrix_33.determinant() << std::endl << std::endl;


    // solve eigenvalue
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> eigen_solver(matrix_33.transpose() * matrix_33);
    std::cout << "Eigen values = " << std::endl << eigen_solver.eigenvalues() << std::endl;
    std::cout << "Eigen vectors = " << std::endl << eigen_solver.eigenvectors() << std::endl;

    // solve equation
    Eigen::Matrix<double, MATRIX_SIZE, MATRIX_SIZE> matrix_NN;
    matrix_NN = Eigen::MatrixXd::Random(MATRIX_SIZE, MATRIX_SIZE);
    Eigen::Matrix<double, MATRIX_SIZE, 1> v_Nd;
    v_Nd = Eigen::MatrixXd::Random(MATRIX_SIZE, 1);

    clock_t time_start = clock();

    Eigen::Matrix<double, MATRIX_SIZE, 1> x = matrix_NN.inverse() * v_Nd;
    std::cout << "time use in normal inverse is :" << 1000 * (clock() - time_start)/(double)CLOCKS_PER_SEC << "ms" << std::endl;

    time_start = clock();
    x = matrix_NN.colPivHouseholderQr().solve(v_Nd);
    std::cout << "time use in QR composition is :" << 1000 * (clock() - time_start)/(double)CLOCKS_PER_SEC << "ms" << std::endl;

    return 0;
}
