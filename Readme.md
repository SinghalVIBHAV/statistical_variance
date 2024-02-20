The script compute_variance.py calculates the variance of a given dataset based on the Schubert and Gertz's method.

The Schubert and Gertz's algorithm computes the variance for a given dataset using the follwing equation:

$$ XX_{A\cup\{b\}} = XX_A + \frac{1}{\Omega_A(\Omega_A + 1)}(X_A - \Omega_Ax_b)^2 $$

where,
$$ 
X_A = \sum_{i \in A}\omega_ix_i \\
\Omega_A = \sum_{i \in A}\omega_i \\
XX_A = \sum_{i \in A}\omega_i\left(x_i - \frac{1}{\Omega_A}X_A\right)^2
$$

This is further divided by $$\Omega = \sum_{i\in P} w_i $$ to compute the variance of the given dataset.