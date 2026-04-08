import numpy as np


def main():
    # 1. 定义矩阵 A
    A = np.array([[1, 2, 3],
                  [3, 4, 5],
                  [5, 6, 9]], dtype=float)

    print("矩阵 A =")
    print(A)

    # 2. 手动按定义计算：先算 B = A^T A
    B = A.T @ A
    print("\nA^T A =")
    print(B)

    # 3. 求 B 的特征值（对称矩阵，使用 eigh 更稳定）
    eigvals, _ = np.linalg.eigh(B)
    print("\nA^T A 的特征值（从小到大） =")
    print(eigvals)

    # 4. 奇异值是特征值的平方根（都应为正）
    singular_values = np.sqrt(eigvals)
    print("\n奇异值（sigma_i = sqrt(lambda_i)） =")
    print(singular_values)

    # 5. 2-范数条件数 = 最大奇异值 / 最小奇异值
    sigma_max = singular_values.max()
    sigma_min = singular_values.min()
    cond_2 = sigma_max / sigma_min

    print("\n最大奇异值 sigma_max =", sigma_max)
    print("最小奇异值 sigma_min =", sigma_min)
    print("\n手动计算的 2-范数条件数 cond_2(A) = sigma_max / sigma_min =")
    print(cond_2)


if __name__ == "__main__":
    main()

