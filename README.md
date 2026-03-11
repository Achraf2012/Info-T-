This is an empirical implementation of the Mutual Information Neural Estimator (MINE), as proposed by Belghazi et al. (2018). While traditional mutual information ($I(X;Y)$) estimation relies on methods that struggle in higher dimensions, MINE leverages the Donsker-Varadhan representation of the KL-divergence to turn estimation into a dual-optimization problem. The Dosker-Varadhan representation is given by :

$D_{KL}(\mathbb{P}\,\|\,\mathbb{Q}) = \sup_{T:\Omega \to \mathbb{R}}\left\{\mathbb{E}_{\mathbb{P}}[T] -\log\big(\mathbb{E}_{\mathbb{Q}}[e^{T}]\big)\right\}$

1- The objective:
I applied MINE to a Single-Input Single-Output (SISO) Communication Channel to observe how neural networks learn fundamental information theoretic limits. Specifically, I estimated the capacity of an AWGN Channel under three different signaling regimes:
$\newline$
*** Gaussian Signaling: The theoratical Shannon Limit is given by $C = \frac{1}{2}\log_2(1 + \text{SNR})$.
$\newline$
*** BPSK Modulation: A constellation with a 1 bit entropy ceiling .
$\newline$
*** QPSK Modulation: A constellation with a 2-bit entropy ceiling .

2- The result (see results.png):
While the theoretical Shannon formula grows infinitely with $SNR$, MINE correctly identifies that BPSK and QPSK are limited by their source entropy ($H=1$ and $H=2$ respectively). Also, we note that the MINE "statistics network" $T_\theta$ was not given any prior information about the modulation type. It "discovered" the 1 bit and 2 bit limits of bpsk and qpsk purely by observing the joint and marginal distributions of $x$ and $y$.

3- Implementation
Architecture: Multi-layer Perceptron (MLP) with ReLU activations.
Frameworks: PyTorch & NumPy.
